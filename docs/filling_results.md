# Filling the [xyz] values

Every yellow `[xyz]` in the paper has a hidden name (a key) in the LaTeX source, for example

```latex
adding geometry changed macro AUROC by \xyz[h1.bfree.delta]{} for B-Free (95\% CI \xyz[h1.bfree.ci], ...
```

The key is not printed; the PDF still shows a yellow `[xyz]`. The key says exactly which number goes there. You do not need to look the numbers up by hand.

## One command

After the full study (and the pilot) have finished, run on Colab from the repository folder:

```bash
python scripts/fill_results.py --config configs/default.yaml --pilot-config configs/pilot.yaml
```

It reads the result files, computes every value it can, and writes

* `results/full/tables/paper_values.md`: one row per key with its value, the result file it came from, and the line of the paper where it appears;
* `paper/main_filled.tex` and `paper/supplementary_filled.tex`: the paper with those values in place. Anything it could not fill stays yellow.

Read `paper_values.md`, compile `main_filled.tex`, and check a few numbers against the source files. When you are happy, run it again with `--inplace` to write the values into `main.tex` and `supplementary.tex` (a `.bak` copy of each is kept), then `python scripts/build_docx.py` and `python scripts/check_paper.py`.

If you have already edited `main.tex` in Overleaf, upload your edited `main.tex` into the Colab `paper/` folder first, so the script fills your version. Do not delete the `\xyz[...]` keys while editing.

## Status column in paper_values.md

| Status | Meaning | What to do |
| --- | --- | --- |
| filled | value computed from a result file | check it, nothing else |
| missing | the result file is not there yet | run the stage that writes it (see table below), then run the script again |
| write yourself | text only you can write (6 keys) | replace the `\xyz[manual...]` by hand; the source column says what to write |

## Key families and where each comes from

| Key | Appears in | Comes from |
| --- | --- | --- |
| `data.*`, `split.*` | dataset tables | `split_summary.csv` (prepare stage) |
| `main.<detector>.<model>.<A-D>` | main results table | `metrics/metrics.csv`, clean, unseen: macro AUROC / TPR at 1% FPR |
| `h1.*`, `h2.*`, `h3.*` | abstract, results text, hypothesis table | `hypotheses.json` (run_experiments) |
| `h4.*` | robustness text and table | `hypotheses.json`, key H4 |
| `pilot.*` | pilot paragraph | `results/pilot/hypotheses.json` |
| `m2.*`, `seen.*`, `calib.*` | results text, supplementary | `predictions/all_main.csv.gz` |
| `pergen.*` | per generator paragraph | `metrics/per_generator.csv` |
| `shift.*` | feature shift sentence | `metrics/feature_shift.csv` |
| `abl.*` | ablation table | `metrics/ablations.csv` (run_ablations; the estimator rows need `geometry.alternative_feature_dirs=[features_marigold, features_retinex]`) |
| `scene.largest`, `scene.smallest` | scene paragraph | `hypotheses.json`, key H5 |
| `scene.kappa` | scene figure caption | `scene_audit_kappa.json`, written by `export_scene_audit.py --score` |
| `compute.*` | compute cost text | `logs/timing_*.json`, `logs/environment_*.json` (GPU name) |
| `ext.*` | limitations | `manifest_splits.csv` (release dates are known only for OpenFake generators; fill missing dates by hand) |
| `leak.*`, `conv.*`, `flags.*`, `stacker.*`, `delong.*`, `qwen.*` | supplementary | leakage report, geometry convention, features, stacker JSON files, hypotheses, zero shot scores |
| `manual.*` | qualitative analysis, OpenFake retrieval, conclusion | you |

## Things the script does not do

* It does not bold the best value per column in the main table; do that by hand.
* It does not choose the discussion paragraph; keep the one that matches the outcome.
* Verdict words (`supported`, `exceeded`, `held`, `survived`) are filled from the pass flags; reword them if the sentence reads badly, but do not change their meaning.
