"""Explicit 3D geometry features computed from frozen estimator outputs.

Everything here is plain NumPy/OpenCV so it can be unit tested without a GPU. The deep
estimators in ``estimators.py`` only produce the inputs:

* ``points``      (H, W, 3) metric point map in OpenCV camera coordinates (x right, y down, z forward)
* ``depth``       (H, W) metric depth from the same model (MoGe)
* ``K``           (3, 3) intrinsics in pixels
* ``mask``        (H, W) bool, valid geometry (no sky, finite depth)
* ``normal_pred`` (H, W, 3) normals predicted directly from pixels by an independent model (DSINE)
* ``disp_da``     (H, W) relative inverse depth from Depth Anything V2 (second depth estimate)
* ``shading``     (H, W) grey shading layer from an intrinsic decomposition (or Retinex fallback)
* ``albedo``      (H, W, 3) optional albedo layer

Four cue groups are computed (see docs/methodology.md, section 5):

G1 depth-normal disagreement, G2 shading-geometry consistency,
G3 perspective consistency, G4 occlusion-edge alignment,
plus applicability indicators and missing-value flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------- inputs


@dataclass
class GeometryInputs:
    rgb: np.ndarray  # (H, W, 3) uint8, at the geometry working resolution
    points: np.ndarray  # (H, W, 3)
    depth: np.ndarray  # (H, W)
    K: np.ndarray  # (3, 3) pixels
    mask: np.ndarray  # (H, W) bool
    normal_pred: np.ndarray  # (H, W, 3)
    disp_da: np.ndarray | None = None  # (H, W)
    shading: np.ndarray | None = None  # (H, W)
    albedo: np.ndarray | None = None  # (H, W, 3)
    extras: dict = field(default_factory=dict)


FEATURE_NAMES: list[str] = [
    # G1 depth-normal disagreement
    "g1_theta_mean", "g1_theta_sd", "g1_theta_p90",
    "g1_theta_da_mean", "g1_theta_da_p90",
    "g1_depth_rank_corr", "g1_depth_affine_rel_rmse",
    # G2 shading-geometry consistency
    "g2_res_median", "g2_res_p90", "g2_sh_r2", "g2_light_dispersion", "g2_n_regions",
    # G3 perspective consistency
    "g3_log_n_lines", "g3_n_vp", "g3_inlier_ratio", "g3_delta_f",
    # G4 occlusion-edge alignment
    "g4_aocc_lo", "g4_aocc_hi", "g4_aocc_mean",
    # applicability indicators
    "app_valid_frac", "app_depth_range",
    # missing-value flags
    "miss_g2", "miss_g2_regions", "miss_g3", "miss_g3_f", "miss_g4",
]

GROUPS: dict[str, list[str]] = {
    "G1": [f for f in FEATURE_NAMES if f.startswith("g1_")],
    "G2": [f for f in FEATURE_NAMES if f.startswith("g2_")] + ["miss_g2", "miss_g2_regions"],
    "G3": [f for f in FEATURE_NAMES if f.startswith("g3_")] + ["miss_g3", "miss_g3_f"],
    "G4": [f for f in FEATURE_NAMES if f.startswith("g4_")] + ["miss_g4"],
    "APP": ["app_valid_frac", "app_depth_range"],
}

# ----------------------------------------------------------------------------- geometry helpers


def intrinsics_from_normalized(K_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """MoGe returns intrinsics normalised by image size; convert to pixels."""
    K = np.array(K_norm, dtype=np.float64).copy()
    K[0, :] *= width
    K[1, :] *= height
    K[2, :] = [0.0, 0.0, 1.0]
    return K


def intrinsics_from_fov(width: int, height: int, fov_deg: float = 60.0) -> np.ndarray:
    """Pinhole intrinsics from a horizontal field of view (only used by synthetic tests)."""
    f = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """P(u,v) = D(u,v) K^-1 [u, v, 1]^T  -> (H, W, 3)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return np.stack([x * depth, y * depth, depth], axis=-1)


def _orient_to_camera(n: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Flip normals so they face the camera (n . P < 0); sign of a normal is otherwise arbitrary."""
    dot = np.einsum("hwc,hwc->hw", n, points)
    return n * np.where(dot > 0, -1.0, 1.0)[..., None]


def normals_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals from the cross product of central differences of a point map.

    Returns (normals (H, W, 3), valid (H, W)); border pixels are invalid.
    """
    p = points.astype(np.float64)
    du = np.zeros_like(p)
    dv = np.zeros_like(p)
    du[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / 2.0
    dv[1:-1, :] = (p[2:, :] - p[:-2, :]) / 2.0
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1)
    valid = norm > 1e-12
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    n = np.where(valid[..., None], n / np.maximum(norm, 1e-12)[..., None], 0.0)
    return _orient_to_camera(n, p), valid


def normals_from_depth(depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return normals_from_points(backproject(depth, K))


def orient_predicted_normals(normal_pred: np.ndarray, points: np.ndarray, convention: np.ndarray | None = None) -> np.ndarray:
    """Bring predicted normals into OpenCV camera axes and make them face the camera.

    ``convention`` is a length-3 sign vector estimated once on the fit split by
    :func:`estimate_convention`; it fixes axis sign differences between estimators.
    """
    n = normal_pred.astype(np.float64)
    if convention is not None:
        n = n * np.asarray(convention, dtype=np.float64)[None, None, :]
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-12)
    return _orient_to_camera(n, points)


def estimate_convention(samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Pick the axis sign vector (8 options) that best aligns predicted with depth normals.

    Args:
        samples: list of (normal_pred, normal_from_depth, points, mask) from FIT images only.
    """
    best, best_err = np.ones(3), np.inf
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                conv = np.array([sx, sy, sz], dtype=np.float64)
                errs = []
                for npred, nd, pts, m in samples:
                    a = orient_predicted_normals(npred, pts, conv)
                    # mean, not median: planes facing the camera are unaffected by some flips and
                    # would make the median tie; the mean also sees the curved surfaces
                    errs.append(np.mean(angular_error_deg(a, nd)[m]))
                e = float(np.mean(errs))
                if e < best_err:
                    best, best_err = conv, e
    return best


def angular_error_deg(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    dot = np.clip(np.einsum("hwc,hwc->hw", n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def discontinuity_mask(depth: np.ndarray, rel_thresh: float = 0.05, margin: int = 3) -> np.ndarray:
    """True near depth discontinuities: |grad log D| above ``rel_thresh``, dilated by ``margin`` px."""
    d = np.log(np.maximum(depth.astype(np.float64), 1e-6))
    gy, gx = np.gradient(d)
    edge = np.hypot(gx, gy) > rel_thresh
    if margin > 0:
        k = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), k) > 0
    return edge


def valid_geometry_mask(depth: np.ndarray, mask: np.ndarray, margin: int = 3, rel_thresh: float = 0.05) -> np.ndarray:
    ok = mask.astype(bool) & np.isfinite(depth) & (depth > 0)
    ok &= ~discontinuity_mask(np.where(ok, depth, np.nanmedian(depth[ok]) if ok.any() else 1.0), rel_thresh, margin)
    ok[:margin, :] = False
    ok[-margin:, :] = False
    ok[:, :margin] = False
    ok[:, -margin:] = False
    return ok


# ----------------------------------------------------------------------------- G1


def depth_from_disparity_aligned(disp: np.ndarray, depth_ref: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Align relative disparity to a reference depth: 1/z_ref ~ a*disp + b (least squares).

    Returns (aligned depth, relative RMSE of the inverse-depth fit).
    """
    inv = 1.0 / depth_ref[mask]
    d = disp[mask]
    A = np.stack([d, np.ones_like(d)], axis=1)
    coef, *_ = np.linalg.lstsq(A, inv, rcond=None)
    fit = A @ coef
    rel_rmse = float(np.sqrt(np.mean((fit - inv) ** 2)) / (np.mean(np.abs(inv)) + 1e-12))
    inv_all = coef[0] * disp + coef[1]
    z = 1.0 / np.clip(inv_all, np.percentile(inv, 0.5) * 0.5, None)
    return z, rel_rmse


def g1_features(gi: GeometryInputs, valid: np.ndarray, convention: np.ndarray | None) -> tuple[dict, dict]:
    n_d, ok_d = normals_from_points(gi.points)
    n_p = orient_predicted_normals(gi.normal_pred, gi.points, convention)
    m = valid & ok_d
    theta = angular_error_deg(n_d, n_p)
    out = {k: np.nan for k in GROUPS["G1"]}
    maps = {"normal_depth": n_d, "normal_pred": n_p, "theta": np.where(m, theta, np.nan)}
    if m.sum() < 100:
        return out, maps
    t = theta[m]
    out.update(g1_theta_mean=float(t.mean()), g1_theta_sd=float(t.std()), g1_theta_p90=float(np.percentile(t, 90)))
    if gi.disp_da is not None:
        z_da, rel = depth_from_disparity_aligned(gi.disp_da.astype(np.float64), gi.depth.astype(np.float64), m)
        n_da, ok_da = normals_from_depth(z_da, gi.K)
        m2 = m & ok_da
        if m2.sum() >= 100:
            t2 = angular_error_deg(n_da, n_p)[m2]
            out.update(g1_theta_da_mean=float(t2.mean()), g1_theta_da_p90=float(np.percentile(t2, 90)))
        sub = np.flatnonzero(m.ravel())
        if len(sub) > 50000:
            sub = np.random.default_rng(0).choice(sub, 50000, replace=False)
        rho = spearmanr(1.0 / gi.depth.ravel()[sub], gi.disp_da.ravel()[sub]).statistic
        out.update(g1_depth_rank_corr=float(rho), g1_depth_affine_rel_rmse=rel)
    return out, maps


# ----------------------------------------------------------------------------- G2

SH_C = (0.282095, 0.488603, 1.092548, 0.315392, 0.546274)


def sh_basis(n: np.ndarray, order: int = 2) -> np.ndarray:
    """Real spherical-harmonic basis evaluated at unit normals n (..., 3) -> (..., 4 or 9)."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    c0, c1, c2, c3, c4 = SH_C
    cols = [np.full_like(x, c0), c1 * y, c1 * z, c1 * x]
    if order >= 2:
        cols += [c2 * x * y, c2 * y * z, c3 * (3 * z**2 - 1), c2 * x * z, c4 * (x**2 - y**2)]
    return np.stack(cols, axis=-1)


def fit_sh_huber(s: np.ndarray, Y: np.ndarray, ridge: float = 1e-4, iters: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Robust least squares  min_l sum huber(s - Y l)  by iteratively reweighted least squares.

    The Huber threshold is 1.345 times a robust (MAD) scale estimate, the usual choice that
    keeps 95% efficiency for Gaussian noise while down-weighting outliers (specular spots,
    shadows edges, texture leaking into the shading layer).
    """
    k = Y.shape[1]
    w = np.ones_like(s)
    lvec = np.zeros(k)
    for _ in range(iters):
        Yw = Y * w[:, None]
        lvec = np.linalg.solve(Y.T @ Yw + ridge * np.eye(k), Yw.T @ s)
        r = s - Y @ lvec
        # robust scale with a floor: images with large flat regions give MAD close to zero,
        # which would otherwise down-weight every pixel that is not perfectly fitted
        scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 0.05 * np.std(s), 1e-6)
        delta = 1.345 * scale
        a = np.abs(r)
        w = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
    return lvec, s - Y @ lvec


def retinex_shading(rgb: np.ndarray, sigma_frac: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """Fallback intrinsic decomposition: shading = smoothed linear luminance, albedo = I / S."""
    lin = (rgb.astype(np.float64) / 255.0) ** 2.2
    lum = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    sigma = max(1.0, sigma_frac * np.hypot(*lum.shape))
    s = cv2.GaussianBlur(lum, (0, 0), sigma) + 1e-6
    alb = lin / s[..., None]
    return s, alb


def low_albedo_variance_mask(albedo: np.ndarray, base: np.ndarray, win: int = 7, quantile: float = 0.5) -> np.ndarray:
    """Pixels whose local log-albedo variation is below the ``quantile`` within ``base``."""
    a = np.log(np.maximum(albedo.mean(axis=-1) if albedo.ndim == 3 else albedo, 1e-4))
    mu = cv2.blur(a, (win, win))
    var = np.maximum(cv2.blur(a * a, (win, win)) - mu * mu, 0)
    if base.sum() == 0:
        return base
    thr = np.quantile(var[base], quantile)
    return base & (var <= thr)


def g2_features(gi: GeometryInputs, valid: np.ndarray, n_cam: np.ndarray, grid: int = 3,
                min_region_px: int = 500, max_px: int = 20000) -> tuple[dict, dict]:
    out = {k: np.nan for k in GROUPS["G2"]}
    out.update(miss_g2=1.0, miss_g2_regions=1.0)
    if gi.shading is not None:
        s_full = gi.shading.astype(np.float64)
        alb = gi.albedo if gi.albedo is not None else (gi.rgb.astype(np.float64) / 255.0) ** 2.2 / (s_full[..., None] + 1e-6)
    else:
        s_full, alb = retinex_shading(gi.rgb)
    lo, hi = np.percentile(s_full[valid], [2, 98]) if valid.any() else (0, 1)
    base = valid & (s_full >= lo) & (s_full <= hi) & (s_full > 1e-4)
    m = low_albedo_variance_mask(alb, base)
    maps = {"shading": s_full, "g2_mask": m}
    if m.sum() < 200:
        return out, maps
    s_norm = s_full / (np.median(s_full[m]) + 1e-12)
    idx = np.flatnonzero(m.ravel())
    if len(idx) > max_px:
        idx = np.random.default_rng(0).choice(idx, max_px, replace=False)
    Y = sh_basis(n_cam.reshape(-1, 3)[idx], order=2)
    s = s_norm.ravel()[idx]
    lvec, r = fit_sh_huber(s, Y)
    ss_tot = np.sum((s - s.mean()) ** 2) + 1e-12
    out.update(g2_res_median=float(np.median(np.abs(r))), g2_res_p90=float(np.percentile(np.abs(r), 90)),
               g2_sh_r2=float(1 - np.sum(r**2) / ss_tot), miss_g2=0.0)
    res_map = np.full(s_full.shape, np.nan)
    Yall = sh_basis(n_cam, order=2)
    res_map[m] = (s_norm - Yall @ lvec)[m]
    maps["residual"] = res_map
    # regional first-order fits on a grid; dispersion of the light vectors
    h, w = m.shape
    lights = []
    for gy in range(grid):
        for gx in range(grid):
            sl = (slice(gy * h // grid, (gy + 1) * h // grid), slice(gx * w // grid, (gx + 1) * w // grid))
            mm = m[sl]
            if mm.sum() < min_region_px:
                continue
            Yr = sh_basis(n_cam[sl][mm], order=1)
            lr, _ = fit_sh_huber(s_norm[sl][mm], Yr)
            lights.append(lr)
    out["g2_n_regions"] = float(len(lights))
    if len(lights) >= 4:
        L = np.array(lights)
        lbar = L.mean(axis=0)
        out["g2_light_dispersion"] = float(np.mean(np.sum((L - lbar) ** 2, axis=1)) / (np.sum(lbar**2) + 1e-12))
        out["miss_g2_regions"] = 0.0
    return out, maps


# ----------------------------------------------------------------------------- G3


def detect_lines(gray: np.ndarray, min_len_frac: float = 0.025) -> np.ndarray:
    """LSD line segments (x1, y1, x2, y2) longer than ``min_len_frac`` of the image diagonal."""
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    lines = lines.reshape(-1, 4).astype(np.float64)
    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[length >= min_len_frac * np.hypot(*gray.shape)]


def _homog_lines(seg: np.ndarray) -> np.ndarray:
    p1 = np.column_stack([seg[:, 0], seg[:, 1], np.ones(len(seg))])
    p2 = np.column_stack([seg[:, 2], seg[:, 3], np.ones(len(seg))])
    l = np.cross(p1, p2)
    return l / (np.linalg.norm(l[:, :2], axis=1, keepdims=True) + 1e-12)


def _consistency_deg(seg: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Angle between each segment's direction and the direction from its midpoint to the VP."""
    mid = np.column_stack([(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2])
    d_seg = np.column_stack([seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]])
    if abs(vp[2]) < 1e-9:
        d_vp = np.tile(vp[:2], (len(seg), 1))
    else:
        d_vp = vp[:2] / vp[2] - mid
    cos = np.abs(np.sum(d_seg * d_vp, axis=1)) / (np.linalg.norm(d_seg, axis=1) * np.linalg.norm(d_vp, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def fit_vanishing_points(seg: np.ndarray, max_vp: int = 3, iters: int = 500, thr_deg: float = 2.0,
                         min_frac: float = 0.05, seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    """Sequential RANSAC: returns (list of homogeneous VPs, per-line VP index or -1)."""
    rng = np.random.default_rng(seed)
    n = len(seg)
    assign = -np.ones(n, dtype=int)
    vps: list[np.ndarray] = []
    if n < 2:
        return vps, assign
    lines = _homog_lines(seg)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    min_inl = max(5, int(min_frac * n))
    for k in range(max_vp):
        free = np.flatnonzero(assign < 0)
        if len(free) < min_inl:
            break
        best_score, best_inl, best_vp = 0.0, None, None
        for _ in range(iters):
            i, j = rng.choice(free, 2, replace=False)
            vp = np.cross(lines[i], lines[j])
            if np.linalg.norm(vp) < 1e-12:
                continue
            vp = vp / np.linalg.norm(vp)
            inl = free[_consistency_deg(seg[free], vp) < thr_deg]
            score = length[inl].sum()
            if score > best_score:
                best_score, best_inl, best_vp = score, inl, vp
        if best_inl is None or len(best_inl) < min_inl:
            break
        # refine: VP = smallest right singular vector of stacked inlier lines (weighted by length)
        A = lines[best_inl] * length[best_inl, None]
        vp = np.linalg.svd(A)[2][-1]
        inl = free[_consistency_deg(seg[free], vp) < thr_deg]
        if len(inl) < min_inl:
            inl, vp = best_inl, best_vp
        assign[inl] = k
        vps.append(vp / np.linalg.norm(vp))
    return vps, assign


def focal_from_orthogonal_vps(v1: np.ndarray, v2: np.ndarray, c: np.ndarray, max_dist: float | None = None) -> float | None:
    """f^2 = -(v1 - c).(v2 - c) for two finite VPs of orthogonal directions (square pixels).

    VPs far outside the image make the estimate ill-conditioned (a VP near infinity carries
    almost no focal information), so VPs farther than ``max_dist`` pixels from the principal
    point are rejected.
    """
    if abs(v1[2]) < 1e-9 or abs(v2[2]) < 1e-9:
        return None
    a, b = v1[:2] / v1[2] - c, v2[:2] / v2[2] - c
    if max_dist is not None and (np.linalg.norm(a) > max_dist or np.linalg.norm(b) > max_dist):
        return None
    f2 = -float(np.dot(a, b))
    return float(np.sqrt(f2)) if f2 > 0 else None


def g3_features(gi: GeometryInputs, work_long_side: int = 1024, ortho_tol_deg: float = 10.0,
                strong_vp_frac: float = 0.10) -> tuple[dict, dict]:
    out = {k: np.nan for k in GROUPS["G3"]}
    out.update(miss_g3=1.0, miss_g3_f=1.0)
    gray = cv2.cvtColor(gi.rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    s = work_long_side / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    seg = detect_lines(gray)
    out["g3_log_n_lines"] = float(np.log1p(len(seg)))
    maps = {"lines": seg / s}
    if len(seg) < 10:
        out["g3_n_vp"] = 0.0
        return out, maps
    vps, assign = fit_vanishing_points(seg)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    out.update(g3_n_vp=float(len(vps)), g3_inlier_ratio=float(length[assign >= 0].sum() / length.sum()), miss_g3=0.0)
    maps.update(vps=vps, vp_assign=assign)
    # focal-length agreement from a near-orthogonal pair of finite VPs
    K = gi.K.copy()
    K[:2, :] *= s
    Kinv = np.linalg.inv(K)
    c = K[:2, 2]
    best = None
    # only well-supported VPs are used for the focal check; weak VPs are often spurious
    n_inl = [int((assign == k).sum()) for k in range(len(vps))]
    strong = [k for k in range(len(vps)) if n_inl[k] >= max(8, strong_vp_frac * len(seg))]
    for a_i, i in enumerate(strong):
        for j in strong[a_i + 1:]:
            di, dj = Kinv @ vps[i], Kinv @ vps[j]
            ang = np.degrees(np.arccos(np.clip(abs(di @ dj) / (np.linalg.norm(di) * np.linalg.norm(dj)), 0, 1)))
            if abs(90 - ang) <= ortho_tol_deg:
                f = focal_from_orthogonal_vps(vps[i], vps[j], c, max_dist=3.0 * np.hypot(*gray.shape))
                if f is not None and (best is None or abs(90 - ang) < best[0]):
                    best = (abs(90 - ang), f)
    if best is not None:
        f_ref = 0.5 * (K[0, 0] + K[1, 1])
        out.update(g3_delta_f=float(abs(np.log(best[1]) - np.log(f_ref))), miss_g3_f=0.0)
    return out, maps


# ----------------------------------------------------------------------------- G4


def _canny_percentile(img: np.ndarray, pct: float) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    hi = float(np.percentile(mag, pct))
    hi = max(hi, 1.0)
    return cv2.Canny(img, 0.5 * hi, hi, L2gradient=True) > 0


def g4_features(gi: GeometryInputs, valid_depth: np.ndarray, min_edge_px: int = 50, dilate_px: int = 2) -> tuple[dict, dict]:
    out = {k: np.nan for k in GROUPS["G4"]}
    out["miss_g4"] = 1.0
    gray = cv2.GaussianBlur(cv2.cvtColor(gi.rgb, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    d = np.log(np.maximum(np.where(valid_depth, gi.depth, np.nan), 1e-6))
    fill = np.nanmedian(d) if np.isfinite(d).any() else 0.0
    d = np.nan_to_num(d, nan=fill)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    vals = []
    maps = {}
    for name, pct in (("lo", 85.0), ("hi", 95.0)):
        ed = _canny_percentile(d8, pct) & valid_depth
        ei = cv2.dilate(_canny_percentile(gray, pct).astype(np.uint8), k) > 0
        maps[f"edges_depth_{name}"], maps[f"edges_img_{name}"] = ed, ei
        if ed.sum() >= min_edge_px:
            a = float((ed & ei).sum() / ed.sum())
            out[f"g4_aocc_{name}"] = a
            vals.append(a)
    if vals:
        out.update(g4_aocc_mean=float(np.mean(vals)), miss_g4=0.0)
    return out, maps


# ----------------------------------------------------------------------------- all features


def compute_features(gi: GeometryInputs, convention: np.ndarray | None = None, return_maps: bool = False,
                     edge_margin: int = 3) -> dict | tuple[dict, dict]:
    """Compute the full explicit feature vector (keys = FEATURE_NAMES) for one image."""
    valid = valid_geometry_mask(gi.depth, gi.mask, margin=edge_margin)
    feats: dict[str, float] = {}
    maps: dict = {"valid": valid}
    f1, m1 = g1_features(gi, valid, convention)
    feats.update(f1)
    maps.update(m1)
    # lighting uses the depth-derived (camera-facing) normals so it tests shading against 3D shape
    f2, m2 = g2_features(gi, valid, m1["normal_depth"])
    feats.update(f2)
    maps.update(m2)
    f3, m3 = g3_features(gi)
    feats.update(f3)
    maps.update(m3)
    valid_depth = gi.mask.astype(bool) & np.isfinite(gi.depth) & (gi.depth > 0)
    f4, m4 = g4_features(gi, valid_depth)
    feats.update(f4)
    maps.update(m4)
    dv = gi.depth[valid_depth]
    feats["app_valid_frac"] = float(valid.mean())
    feats["app_depth_range"] = float(np.log(np.percentile(dv, 95) / max(np.percentile(dv, 5), 1e-6))) if dv.size > 10 else np.nan
    ordered = {k: float(feats.get(k, np.nan)) for k in FEATURE_NAMES}
    return (ordered, maps) if return_maps else ordered
