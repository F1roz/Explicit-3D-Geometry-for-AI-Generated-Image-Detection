"""Checks for scripts/fill_results.py: every paper placeholder has a known key, and filling works."""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fill_results as fr  # noqa: E402

D = r"(bfree|cospy|univfd)"
V = r"(jpeg75|resize05|screenshot)"
PATTERNS = [
    rf"main\.{D}\.M[1-5]\.[ABCD]", rf"h[123]\.{D}\.(delta|ci|p)", r"h[123]\.verdict", r"h[234]\.abstract_verb",
    r"h1\.tpr_summary", rf"h4\.{D}\.{V}\.(delta|ci)", rf"h4\.{V}\.(range|extra_fpr_drift)", r"h4\.jpeg75\.fpr_drift_m[13]",
    r"data\.(fitval|[ABCD])\.n_(real|fake)", r"split\.(fit|val|[ABCD])\.(sources|n)",
    r"pilot\.(delta|ci90|gain_m4|extra_fpr_drift|m2_auroc|decision)", r"m2\.macro_auroc",
    r"pergen\.(largest|smallest|seen_family_gain)", r"shift\.top_(feature|smd)",
    rf"abl\.(drop_G[1-4]|only_G[1-4]|no_flags|gbdt|processing_aware_fit|estimator_features_(marigold|retinex))\.{D}",
    r"scene\.(largest|smallest|kappa)", r"compute\.(gpu|geom_sec|feat_sec|det_min|det_max)",
    r"ext\.(n_unseen_generators|earliest|latest)", r"leak\.(generator_overlap|real_source_overlap|prompt_overlap|fitval_prompt_overlap|dedup_removed|unreadable)",
    r"conv\.(sign|median_angle)", r"flags\.g[234]\.(real|fake)", rf"stacker\.{D}\.(C|cv_auroc|temperature|threshold)",
    rf"seen\.{D}\.M[13]", rf"calib\.{D}\.(ece|acc_real|acc_fake)\.M[13]", rf"delong\.{D}\.(z|p)", r"qwen\.(auroc|n)",
]


class TestFillResults(unittest.TestCase):
    def test_every_placeholder_has_a_known_key(self):
        for name in ("main.tex", "supplementary.tex"):
            text = (ROOT / "paper" / name).read_text()
            body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%") and "newcommand" not in l)
            self.assertEqual(len(re.findall(r"\\xyz(?![\[A-Za-z])", body)), 0, f"unkeyed \\xyz in {name}")
            for key, ln, _ in fr.keys_in(ROOT / "paper" / name):
                known = key in fr.MANUAL_HELP or any(re.fullmatch(p, key) for p in PATTERNS)
                self.assertTrue(known, f"{name}:{ln} unknown key {key}")

    def test_fill_text_replaces_known_and_keeps_unknown(self):
        tex = "% comment \\xyz[a]\nGain \\xyz[a]{} and \\xyz[b] here."
        out, n = fr.fill_text(tex, {"a": ("+0.012", "src")})
        self.assertEqual(n, 1)
        self.assertIn("Gain +0.012{} and \\xyz[b] here.", out)
        self.assertIn("% comment \\xyz[a]", out)

    def test_formatting(self):
        self.assertEqual(fr.signed(-0.0123), "$-$0.012")
        self.assertEqual(fr.signed(0.5), "+0.500")
        self.assertEqual(fr.ci(-0.01, 0.02), "[$-$0.010, +0.020]")
        self.assertEqual(fr.pval_boot(0.0, 2000), "$<$0.0005")
        self.assertEqual(fr.rng([0.1, 0.1]), "+0.100")
        self.assertEqual(fr.tt("g1_theta"), "\\texttt{g1\\_theta}")


if __name__ == "__main__":
    unittest.main()
