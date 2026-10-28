"""Tiny ray-cast renderer for smoke tests and feature sanity checks.

It renders a ground plane, a back wall and a few spheres with a pinhole camera, so the exact
depth, normals, albedo and shading are known. Two scene types are produced:

* ``consistent``   one directional light for the whole scene, and the normals seen by the
  "pixel normal estimator" equal the true surface normals (plus small noise). This mimics a
  physically consistent photograph.
* ``inconsistent`` every object is lit from its own random direction and shading is computed
  from bumped normals that do not match the 3D shape. This mimics the kind of violation the
  geometry features are designed to measure.

These images are NOT a model of real or generated photographs. They only verify that the
feature code responds in the expected direction and that the full pipeline runs end to end.
"""

from __future__ import annotations

import numpy as np

from .geometry.features import intrinsics_from_fov


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float = 12.0) -> np.ndarray:
    """Low-frequency noise with unit standard deviation (mimics smooth depth-model errors)."""
    import cv2

    n = cv2.GaussianBlur(rng.normal(0, 1, shape), (0, 0), sigma)
    return (n / (n.std() + 1e-12)).astype(np.float32)


def render_scene(rng: np.random.Generator, consistent: bool, height: int = 192, width: int = 256,
                 n_spheres: int = 3) -> dict[str, np.ndarray]:
    K = intrinsics_from_fov(width, height, fov_deg=60.0)
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)  # z = 1

    cam_h = 1.0 + 0.3 * rng.random()
    wall_z = 7.0 + 3.0 * rng.random()
    t = np.full((height, width), np.inf)
    obj = np.full((height, width), -1, dtype=int)
    normal = np.zeros((height, width, 3))

    # ground plane y = cam_h (y points down), normal (0, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[..., 1] > 1e-6, cam_h / d[..., 1], np.inf)
    hit = tg < t
    t[hit], obj[hit] = tg[hit], 0
    normal[hit] = [0.0, -1.0, 0.0]
    # back wall z = wall_z, normal (0, 0, -1)
    tw = np.full_like(t, wall_z)
    hit = tw < t
    t[hit], obj[hit] = tw[hit], 1
    normal[hit] = [0.0, 0.0, -1.0]
    # spheres resting on the ground
    for k in range(n_spheres):
        r = 0.35 + 0.3 * rng.random()
        c = np.array([rng.uniform(-1.6, 1.6), cam_h - r, rng.uniform(3.0, wall_z - 1.0)])
        b = np.einsum("hwc,c->hw", d, c)
        dd = np.einsum("hwc,hwc->hw", d, d)
        disc = b * b - dd * (c @ c - r * r)
        ts = np.where(disc > 0, (b - np.sqrt(np.maximum(disc, 0))) / dd, np.inf)
        hit = (ts > 0) & (ts < t)
        t[hit], obj[hit] = ts[hit], 2 + k
        p = d[hit] * ts[hit][:, None]
        normal[hit] = (p - c) / r

    points = d * t[..., None]
    depth = points[..., 2]

    # albedo: checker ground (gives straight lines for LSD), striped wall, plain spheres
    albedo = np.zeros((height, width, 3))
    g = obj == 0
    check = ((np.floor(points[..., 0] * 1.5) + np.floor(points[..., 2] * 1.5)) % 2).astype(float)
    albedo[g] = (0.35 + 0.3 * check[g])[:, None] * np.array([0.9, 0.85, 0.8])
    w = obj == 1
    stripes = (np.floor(points[..., 0] * 2.0) % 2).astype(float)
    albedo[w] = (0.45 + 0.2 * stripes[w])[:, None] * np.array([0.8, 0.85, 0.95])
    for k in range(n_spheres):
        m = obj == 2 + k
        albedo[m] = rng.uniform(0.3, 0.9, 3)

    # lighting
    # light travel direction in camera axes (y down, z forward): from above and slightly behind the camera
    base_light = _normalize(np.array([rng.uniform(-0.6, 0.6), rng.uniform(0.4, 1.0), rng.uniform(0.2, 0.8)]))
    shade_normal = normal.copy()
    light = np.broadcast_to(base_light, normal.shape).copy()
    if not consistent:
        for k in range(-1, n_spheres + 2):
            m = obj == k
            light[m] = _normalize(np.array([rng.uniform(-1, 1), rng.uniform(-0.2, 1.0), rng.uniform(-0.3, 1.0)]))
        # low-frequency bumps: shading normals disagree with the 3D shape
        bump = np.stack([np.sin(u / 9.0 + rng.random() * 6), np.cos(v / 11.0 + rng.random() * 6), np.zeros_like(u)], -1)
        shade_normal = _normalize(normal + 0.45 * bump)
    lam = np.clip(-np.einsum("hwc,hwc->hw", shade_normal, light), 0, None)
    shading = 0.25 + 0.75 * lam

    noise = rng.normal(0, 0.02, normal.shape)
    normal_pred = _normalize(shade_normal + noise)
    lin = albedo * shading[..., None]
    rgb = np.clip(lin, 0, 1) ** (1 / 2.2)
    rgb = np.clip(rgb * 255 + rng.normal(0, 1.5, rgb.shape), 0, 255).astype(np.uint8)
    return {
        "rgb": rgb,
        "depth": depth.astype(np.float32),
        "K": K,
        "mask": np.isfinite(depth),
        "normal_true": normal.astype(np.float32),
        "normal_pred": normal_pred.astype(np.float32),
        "shading": shading.astype(np.float32),
        "albedo": albedo.astype(np.float32),
        "disp_noise": _smooth_noise(rng, depth.shape),
    }
