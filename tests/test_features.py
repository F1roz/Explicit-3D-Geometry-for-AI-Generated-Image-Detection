"""Geometry feature tests: exact cases with known answers, plus direction-of-effect checks."""

import unittest

import numpy as np

from tests import _path  # noqa: F401
from prism3d.geometry import features as F
from prism3d.synthetic import render_scene


def _gi(s):
    d = s["depth"].astype(float)
    return F.GeometryInputs(rgb=s["rgb"], points=F.backproject(d, s["K"]), depth=d, K=s["K"], mask=s["mask"],
                            normal_pred=s["normal_pred"], disp_da=(1 / d) * (1 + 0.03 * s["disp_noise"]),
                            shading=s["shading"], albedo=s["albedo"])


class TestGeometryHelpers(unittest.TestCase):
    def test_backproject_center_ray(self):
        K = F.intrinsics_from_fov(64, 48, 60)
        P = F.backproject(np.full((48, 48 + 16), 2.0), K)
        # a pixel at the principal point lies on the optical axis
        self.assertAlmostEqual(P[int(K[1, 2]), int(K[0, 2]), 2], 2.0)

    def test_fronto_parallel_plane_normal(self):
        K = F.intrinsics_from_fov(80, 60, 60)
        n, ok = F.normals_from_depth(np.full((60, 80), 3.0), K)
        inner = n[ok]
        # facing the camera means pointing to -z in OpenCV camera axes
        np.testing.assert_allclose(inner.mean(axis=0), [0, 0, -1], atol=1e-6)

    def test_tilted_plane_normal(self):
        K = F.intrinsics_from_fov(80, 60, 60)
        h, w = 60, 80
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        # plane z = 4 + 0.5*y  (ground-like tilt); recover its normal from depth
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]
        z = 4.0 / (1 - 0.5 * y)
        n, ok = F.normals_from_depth(z, K)
        expected = np.array([0, 0.5, -1.0]) / np.linalg.norm([0, 0.5, -1.0])
        np.testing.assert_allclose(np.median(n[ok], axis=0), expected, atol=1e-3)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_angular_error(self):
        a = np.zeros((4, 4, 3))
        a[..., 2] = 1
        b = np.zeros((4, 4, 3))
        b[..., 0] = 1
        np.testing.assert_allclose(F.angular_error_deg(a, a), 0, atol=1e-6)
        np.testing.assert_allclose(F.angular_error_deg(a, b), 90, atol=1e-6)

    def test_convention_recovers_axis_flip(self):
        s = render_scene(np.random.default_rng(1), consistent=True)
        gi = _gi(s)
        nd, ok = F.normals_from_points(gi.points)
        flipped = s["normal_true"] * np.array([1, -1, 1])  # an estimator that uses y up
        conv = F.estimate_convention([(flipped, nd, gi.points, ok & s["mask"])])
        # a global sign change is absorbed by orienting normals towards the camera,
        # so the answer is defined up to sign: it must be proportional to (1, -1, 1)
        np.testing.assert_array_equal(conv * conv[0], [1, -1, 1])


class TestShading(unittest.TestCase):
    def test_sh_fit_recovers_lighting(self):
        rng = np.random.default_rng(0)
        n = rng.normal(size=(5000, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        Y = F.sh_basis(n)
        l_true = rng.normal(size=9)
        s = Y @ l_true
        l_hat, r = F.fit_sh_huber(s, Y)
        np.testing.assert_allclose(l_hat, l_true, atol=1e-3)
        self.assertLess(np.abs(r).max(), 1e-3)

    def test_huber_ignores_outliers(self):
        rng = np.random.default_rng(0)
        n = rng.normal(size=(4000, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        Y = F.sh_basis(n, order=1)
        l_true = np.array([1.0, 0.3, -0.2, 0.5])
        s = Y @ l_true + rng.normal(0, 0.01, len(n))
        s[:200] += 5.0  # 5% gross outliers (specular highlights)
        l_hat, _ = F.fit_sh_huber(s, Y)
        np.testing.assert_allclose(l_hat, l_true, atol=0.05)


class TestPerspective(unittest.TestCase):
    def test_focal_from_orthogonal_vps(self):
        K = F.intrinsics_from_fov(640, 480, 60)
        t1, t2 = np.deg2rad(10), np.deg2rad(35)
        Rx = np.array([[1, 0, 0], [0, np.cos(t1), -np.sin(t1)], [0, np.sin(t1), np.cos(t1)]])
        Ry = np.array([[np.cos(t2), 0, np.sin(t2)], [0, 1, 0], [-np.sin(t2), 0, np.cos(t2)]])
        R = Rx @ Ry
        v1, v2 = K @ R @ [1, 0, 0], K @ R @ [0, 0, 1]
        self.assertAlmostEqual(F.focal_from_orthogonal_vps(v1, v2, K[:2, 2]), K[0, 0], places=6)

    def test_far_vp_rejected(self):
        c = np.array([320.0, 240.0])
        self.assertIsNone(F.focal_from_orthogonal_vps(np.array([1e6, 240, 1.0]), np.array([320, 240, 1.0]), c, max_dist=2000))

    def test_vp_ransac_on_synthetic_lines(self):
        rng = np.random.default_rng(0)
        vp = np.array([400.0, 120.0])
        segs = []
        for _ in range(40):
            p = rng.uniform([0, 200], [300, 480])
            d = (vp - p) / np.linalg.norm(vp - p)
            segs.append([*p, *(p + 60 * d)])
        for _ in range(10):  # clutter
            p = rng.uniform(0, 480, 2)
            a = rng.uniform(0, np.pi)
            segs.append([*p, *(p + 50 * np.array([np.cos(a), np.sin(a)]))])
        vps, assign = F.fit_vanishing_points(np.array(segs), max_vp=1)
        self.assertEqual(len(vps), 1)
        est = vps[0][:2] / vps[0][2]
        np.testing.assert_allclose(est, vp, atol=2.0)
        self.assertGreaterEqual((assign[:40] == 0).mean(), 0.95)


class TestFeatureVector(unittest.TestCase):
    def test_all_names_and_finite_core(self):
        f = F.compute_features(_gi(render_scene(np.random.default_rng(2), True)))
        self.assertEqual(list(f), F.FEATURE_NAMES)
        for k in ("g1_theta_mean", "g2_res_median", "g4_aocc_mean", "app_valid_frac"):
            self.assertTrue(np.isfinite(f[k]), k)

    def test_direction_of_effect(self):
        rng = np.random.default_rng(3)
        cons = [F.compute_features(_gi(render_scene(rng, True))) for _ in range(4)]
        inc = [F.compute_features(_gi(render_scene(rng, False))) for _ in range(4)]
        for k in ("g1_theta_mean", "g2_res_median"):
            self.assertLess(np.mean([c[k] for c in cons]), np.mean([c[k] for c in inc]), k)

    def test_flat_image_sets_missing_flags(self):
        s = render_scene(np.random.default_rng(4), True)
        s["rgb"][:] = 128  # no lines, no edges
        f = F.compute_features(_gi(s))
        self.assertEqual(f["miss_g3"], 1.0)
        self.assertTrue(np.isnan(f["g3_inlier_ratio"]))


if __name__ == "__main__":
    unittest.main()
