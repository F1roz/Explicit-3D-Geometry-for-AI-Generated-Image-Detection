# Results folder

Each run writes to `results/<run name>/` (set by `experiment.run_dir` in the config):

| Path | Content |
| --- | --- |
| `manifest_all.csv`, `manifest_splits.csv` | every ingested image; split and test-set membership |
| `leakage_report.json`, `split_summary.csv` | leakage checks and counts per split, label and generator |
| `processed/<variant>/` | harmonised images and processing variants |
| `features/<variant>.csv` | 26 explicit geometry features per image |
| `scores/<detector>/<variant>.csv` | base detector logits |
| `controls/<variant>.npz`, `scenes.csv` | DINOv2 features, CLIP scene labels |
| `stackers/<detector>/<model>.pkl/.json` | fitted stackers with C, temperature, threshold, coefficients |
| `predictions/` | calibrated test predictions for every model, variant and detector |
| `metrics/metrics.csv`, `per_generator.csv`, `ablations.csv`, `feature_shift.csv` | evaluation tables |
| `hypotheses.json` | H1 to H5 tests, Holm corrections, DeLong, pilot decision |
| `tables/` | LaTeX rows and `results_summary.md` for filling the paper |
| `figures/` | all result figures (PDF and PNG) |
| `logs/` | config, environment and timing for each stage |

`results/smoke/` is produced by `scripts/run_smoke.py` on synthetic renders and is only a wiring test.
