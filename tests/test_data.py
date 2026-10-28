import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tests import _path  # noqa: F401
from prism3d.config import load_config
from prism3d.data import dedup, processing
from prism3d.data.manifest import make_frame
from prism3d.data.splits import LeakageError, assign_splits, check_leakage, explode_eval_sets


def _manifest():
    rows = []
    for g in ["a", "b", "c", "d"]:
        for i in range(10):
            rows.append({"image_id": f"x/{g}/{i}", "path": "", "label": 1, "dataset": "x", "generator": g,
                         "prompt_id": f"cap{i}" if g in ("a", "b") else f"{g}{i}"})
    for s in ["r1", "r2"]:
        for i in range(10):
            rows.append({"image_id": f"x/{s}/{i}", "path": "", "label": 0, "dataset": "x", "real_source": s,
                         "prompt_id": f"{s}{i}"})
    return make_frame(rows)


SPLITS = {
    "fit_val": {"val_fraction": 0.3, "selectors": [{"generators": ["a", "b"]}, {"real_sources": ["r1"]}]},
    "tests": {"T1": {"selectors": [{"generators": ["c"]}, {"real_sources": ["r2"]}]},
              "T2": {"selectors": [{"generators": ["d"]}, {"real_sources": ["r2"]}]}},
    "seen_family_patterns": ["^d$"],
}


class TestSplits(unittest.TestCase):
    def test_assignment_and_sharing(self):
        out = assign_splits(_manifest(), SPLITS)
        self.assertEqual(set(out.split), {"fit", "val", "test"})
        r2 = out[out.real_source == "r2"]
        self.assertTrue((r2.eval_sets == "T1|T2").all())  # shared real photos
        ex = explode_eval_sets(out)
        self.assertEqual(len(ex[ex.eval_set == "T2"]), 20)
        self.assertTrue(out[out.generator == "d"].seen_family.all())

    def test_prompts_never_straddle_fit_and_val(self):
        out = assign_splits(_manifest(), SPLITS, seed=3)
        fv = out[out.split.isin(["fit", "val"])]
        self.assertTrue((fv.groupby("prompt_id").split.nunique() == 1).all())

    def test_generator_leak_raises(self):
        bad = {**SPLITS, "tests": {"T1": {"selectors": [{"generators": ["a"]}]}}}
        with self.assertRaises(LeakageError):
            assign_splits(_manifest(), bad)

    def test_check_leakage_on_manual_overlap(self):
        out = assign_splits(_manifest(), SPLITS)
        out.loc[out.generator == "c", "generator"] = "a"
        with self.assertRaises(LeakageError):
            check_leakage(out)


class TestDedupAndProcessing(unittest.TestCase):
    def test_phash_identical_and_different(self):
        rng = np.random.default_rng(0)
        a = (rng.uniform(size=(64, 64, 3)) * 255).astype(np.uint8)
        b = processing.jpeg_roundtrip(a, 90)
        c = (rng.uniform(size=(64, 64, 3)) * 255).astype(np.uint8)
        h = np.array([dedup.phash(a), dedup.phash(b), dedup.phash(c)], dtype=np.uint64)
        d = dedup.hamming_matrix(h, h)
        self.assertEqual(d[0, 0], 0)
        self.assertLessEqual(d[0, 1], 8)
        self.assertGreater(d[0, 2], 8)

    def test_variants_written_with_expected_sizes(self):
        rng = np.random.default_rng(0)
        img = (rng.uniform(size=(100, 120, 3)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as td:
            outs = {v: Path(td) / f"{v}{processing.VARIANT_EXT[v]}" for v in processing.VARIANT_CHAINS}
            processing.write_processed(img, outs)
            sizes = {}
            for v, p in outs.items():
                with Image.open(p) as im:
                    sizes[v] = im.size
        self.assertEqual(sizes["clean"], (120, 100))
        self.assertEqual(sizes["resize05"], (60, 50))
        self.assertEqual(sizes["screenshot"], (90, 75))

    def test_config_inherit_and_replace(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        cfg = load_config(root / "pilot.yaml", ["experiment.seed=5"])
        self.assertEqual(cfg["experiment"]["seed"], 5)
        self.assertEqual(set(cfg["splits"]["tests"]), {"A_open_models", "B_content_matched"})
        self.assertEqual(cfg["stacking"]["C_grid"], [0.001, 0.01, 0.1, 1.0, 10.0])  # inherited


if __name__ == "__main__":
    unittest.main()
