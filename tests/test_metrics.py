import unittest

import numpy as np

from tests import _path  # noqa: F401
from prism3d import metrics as M


class TestMetrics(unittest.TestCase):
    def test_auroc_manual(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.4, 0.35, 0.8])
        # pairs (fake > real): (0.35>0.1), (0.8>0.1), (0.8>0.4) -> 3 of 4
        self.assertAlmostEqual(M.auroc(y, s), 0.75)

    def test_auroc_single_class_is_nan(self):
        self.assertTrue(np.isnan(M.auroc([1, 1], [0.2, 0.3])))

    def test_tpr_at_fpr_perfect(self):
        y = np.r_[np.zeros(100), np.ones(100)]
        s = np.r_[np.zeros(100), np.ones(100)]
        self.assertAlmostEqual(M.tpr_at_fpr(y, s, 0.01), 1.0)

    def test_rates_at_threshold(self):
        r = M.rates_at_threshold([0, 0, 1, 1], [-1, 1, 1, -1], 0.0)
        self.assertEqual(r["fpr"], 0.5)
        self.assertEqual(r["tpr"], 0.5)

    def test_ece_perfect_calibration(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=200000)
        y = (rng.uniform(size=p.size) < p).astype(int)
        self.assertLess(M.ece(y, p), 0.01)

    def test_macro_weights_generators_equally(self):
        y = np.r_[np.zeros(10), np.ones(100), np.ones(10)]
        s = np.r_[np.zeros(10), np.ones(100), -np.ones(10)]
        g = np.array(["r"] * 10 + ["big"] * 100 + ["small"] * 10)
        self.assertAlmostEqual(M.macro_over_generators(y, s, g), 0.5)
        self.assertGreater(M.auroc(y, s), 0.85)


if __name__ == "__main__":
    unittest.main()
