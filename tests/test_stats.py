import unittest

import numpy as np

from tests import _path  # noqa: F401
from prism3d import stats as S


def _data(seed=0, n_gen=6, n_per=60, gain=1.0):
    rng = np.random.default_rng(seed)
    y, cl, a, b = [], [], [], []
    for g in range(n_gen):
        y += [1] * n_per
        cl += [f"g{g}"] * n_per
        base = rng.normal(0.8, 1, n_per)
        b += list(base)
        a += list(base + gain * rng.uniform(0.5, 1.0) + rng.normal(0, 0.3, n_per))
    for r in range(2):
        y += [0] * 150
        cl += [f"r{r}"] * 150
        base = rng.normal(0, 1, 150)
        b += list(base)
        a += list(base + rng.normal(0, 0.3, 150))
    return np.array(y), np.array(a), np.array(b), np.array(cl)


class TestStats(unittest.TestCase):
    def test_delong_identical_scores(self):
        y = np.r_[np.zeros(50), np.ones(50)]
        s = np.random.default_rng(0).normal(size=100) + y
        r = S.delong_test(y, s, s)
        self.assertAlmostEqual(r["delta"], 0.0)
        self.assertAlmostEqual(r["p"], 1.0)

    def test_delong_detects_better_model(self):
        y, a, b, _ = _data()
        r = S.delong_test(y, a, b)
        self.assertGreater(r["delta"], 0)
        self.assertLess(r["p"], 0.01)

    def test_delong_auc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score

        y, a, b, _ = _data(1)
        r = S.delong_test(y, a, b)
        self.assertAlmostEqual(r["auc_a"], roc_auc_score(y, a), places=10)

    def test_cluster_bootstrap_positive_gain(self):
        y, a, b, cl = _data()
        r = S.cluster_bootstrap_delta(y, a, b, cl, n_boot=300, seed=0)
        self.assertGreater(r["ci_low"], 0)
        self.assertLessEqual(r["ci_low"], r["delta"])
        self.assertLessEqual(r["delta"], r["ci_high"])
        self.assertEqual(r["n_fake_clusters"], 6)

    def test_cluster_bootstrap_null_covers_zero(self):
        y, a, b, cl = _data(gain=0.0)
        r = S.cluster_bootstrap_delta(y, b, b, cl, n_boot=200, seed=0)
        self.assertEqual(r["delta"], 0.0)
        self.assertLessEqual(r["ci_low"], 0.0)
        self.assertGreaterEqual(r["ci_high"], 0.0)

    def test_cluster_bootstrap_reproducible(self):
        y, a, b, cl = _data()
        r1 = S.cluster_bootstrap_delta(y, a, b, cl, n_boot=100, seed=7)
        r2 = S.cluster_bootstrap_delta(y, a, b, cl, n_boot=100, seed=7)
        self.assertEqual(r1, r2)

    def test_holm(self):
        res = S.holm({"a": 0.001, "b": 0.02, "c": 0.04}, alpha=0.05)
        self.assertTrue(res["a"]["reject"])
        self.assertTrue(res["b"]["reject"])  # 0.02 <= 0.05/2
        self.assertTrue(res["c"]["reject"])  # 0.04 <= 0.05/1
        res = S.holm({"a": 0.03, "b": 0.04}, alpha=0.05)
        self.assertFalse(res["a"]["reject"])  # 0.03 > 0.025, so nothing is rejected
        self.assertFalse(res["b"]["reject"])
        self.assertGreaterEqual(res["b"]["p_holm"], res["a"]["p_holm"])

    def test_mcnemar(self):
        y = np.ones(40, int)
        a = np.ones(40, int)
        b = np.r_[np.ones(10), np.zeros(30)].astype(int)
        r = S.mcnemar_test(y, a, b)
        self.assertEqual(r["a_only_correct"], 30)
        self.assertLess(r["p"], 1e-6)


if __name__ == "__main__":
    unittest.main()
