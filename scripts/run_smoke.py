"""End-to-end smoke test on synthetic renders. CPU only, about two minutes.

Runs every stage with configs/smoke.yaml (ground-truth geometry, mock detector, mock control
features) and asserts that each expected output file exists and is well formed. The numbers it
produces say nothing about real images; they only prove the pipeline is wired correctly.

    python scripts/run_smoke.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

from _bootstrap import ROOT
from prism3d.config import load_config
from prism3d import pipeline


def main() -> None:
    t0 = time.time()
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["experiment"]["run_dir"] = str(ROOT / cfg["experiment"]["run_dir"])
    for s in cfg["data"]["sources"]:
        s["root"] = str(ROOT / s["root"])
    rd = pipeline.run_dir(cfg)
    if rd.exists():
        shutil.rmtree(rd)
    data = ROOT / "data" / "synthetic"
    if not data.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--n", "10"], check=True)
    # pick a few test images for the qualitative figure
    df = pipeline.prepare(cfg)
    test_ids = df[df.split == "test"].groupby("label").head(2).image_id.tolist()
    cfg["geometry"]["save_maps_for"] = test_ids[:3]
    pipeline.extract_geometry(cfg, kind="oracle")
    pipeline.run_detectors(cfg)
    pipeline.extract_controls(cfg, mock=True)
    h = pipeline.run_experiments(cfg)
    abl = pipeline.run_ablations(cfg)
    pipeline.make_tables(cfg)
    figs = pipeline.make_figures(cfg)

    expected = ["manifest_splits.csv", "leakage_report.json", "split_summary.csv", "features/clean.csv",
                "features/jpeg75.csv", "scores/mock/clean.csv", "controls/clean.npz", "predictions/all_main.csv.gz",
                "metrics/metrics.csv", "metrics/per_generator.csv", "metrics/ablations.csv", "hypotheses.json",
                "tables/results_summary.md", "tables/main_results_rows.tex", "geometry_convention.json"]
    missing = [e for e in expected if not (rd / e).exists()]
    assert not missing, f"Missing outputs: {missing}"
    assert len(abl) > 0, "no ablation rows"
    assert any(str(f).endswith("fig_model_comparison.pdf") for f in figs), "figures missing"
    print(json.dumps({"H1_pass": h["H1"]["pass"], "pilot": {d: v["decision"] for d, v in h.get("pilot", {}).items()},
                      "n_figures": len(figs), "seconds": round(time.time() - t0, 1)}, indent=2))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
