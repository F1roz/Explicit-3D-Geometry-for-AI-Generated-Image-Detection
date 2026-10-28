import unittest

import numpy as np
import pandas as pd

from tests import _path  # noqa: F401
from prism3d.stacking import Stacker, make_folds


def _frame(seed, n_gen=6, n=40, split="fit"):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_gen):
        for i in range(n):
            rows.append({"label": 1, "cluster": f"gen{g}", "prompt_id": f"p{seed}_{g}_{i}", "split": split,
                         "base_logit": rng.normal(0.5, 1), "geo": rng.normal(1.0, 1), "noise": rng.normal()})
    for i in range(n_gen * n):
        rows.append({"label": 0, "cluster": "real", "prompt_id": f"r{seed}_{i}", "split": split,
                     "base_logit": rng.normal(0, 1), "geo": rng.normal(0, 1), "noise": rng.normal()})
    return pd.DataFrame(rows)


class TestStacker(unittest.TestCase):
    def test_folds_hold_out_whole_generators(self):
        df = _frame(0)
        for tr, te in make_folds(df, 3, 0):
            gtr = set(df.iloc[tr].query("label == 1").cluster)
            gte = set(df.iloc[te].query("label == 1").cluster)
            self.assertFalse(gtr & gte)

    def test_threshold_gives_target_fpr_on_val(self):
        fit, val = _frame(0), _frame(1, split="val")
        st = Stacker(columns=["base_logit", "geo"], C_grid=(0.1, 1.0), n_splits=3, target_fpr=0.05).fit(fit, val)
        z = st.predict_logit(val)
        fpr = (z[val.label.to_numpy() == 0] >= st.threshold).mean()
        self.assertAlmostEqual(fpr, 0.05, delta=0.01)

    def test_informative_feature_gets_weight(self):
        fit, val = _frame(0), _frame(1, split="val")
        st = Stacker(columns=["base_logit", "geo", "noise"], C_grid=(1.0,), n_splits=3).fit(fit, val)
        c = st.coefficients()
        self.assertGreater(abs(c["geo"]), abs(c["noise"]))

    def test_missing_values_are_handled(self):
        fit, val = _frame(0), _frame(1, split="val")
        fit.loc[::3, "geo"] = np.nan
        val.loc[::4, "geo"] = np.nan
        st = Stacker(columns=["base_logit", "geo"], C_grid=(1.0,), n_splits=3).fit(fit, val)
        self.assertTrue(np.all(np.isfinite(st.predict_logit(val))))

    def test_requires_both_classes(self):
        fit, val = _frame(0), _frame(1, split="val")
        with self.assertRaises(ValueError):
            Stacker(columns=["geo"], C_grid=(1.0,)).fit(fit[fit.label == 1], val)

    def test_gbdt_kind(self):
        fit, val = _frame(0), _frame(1, split="val")
        st = Stacker(columns=["base_logit", "geo"], kind="gbdt").fit(fit, val)
        self.assertEqual(len(st.predict_logit(val)), len(val))


if __name__ == "__main__":
    unittest.main()
