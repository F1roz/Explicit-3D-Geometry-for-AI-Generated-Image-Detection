# Reproducibility: exact commands

All commands run from the repository root. Paths are relative; nothing depends on a personal path. Every stage writes `logs/config_<stage>.yaml` and `logs/environment_<stage>.json` (Python, library versions, GPU) next to its outputs, and can resume after an interruption.

## 0. Environment

Tested here (authoring environment, CPU only): Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, scikit-learn 1.9, OpenCV 5.0, Matplotlib 3.11. The deep models need a GPU environment (Google Colab, Python 3.12, PyTorch 2.x); they could not be executed in the authoring environment because it had no GPU and no network access to Hugging Face or Zenodo.

```bash
pip install -r requirements.txt
pip install git+https://github.com/microsoft/MoGe.git
pip install https://github.com/compphoto/Intrinsic/archive/main.zip
pip install -e .            # optional; scripts add src/ to the path themselves
```

## 1. Tests (CPU, under a minute) and smoke test (CPU, about two minutes)

```bash
python -m unittest discover -s tests -t .
PRISM3D_SLOW=1 python -m unittest tests.test_smoke_pipeline      # same as: python scripts/run_smoke.py
```

Expected: `OK (skipped=1)` for the first command; `SMOKE TEST PASSED` for the second. The smoke test renders synthetic scenes (`scripts/make_synthetic_data.py`), uses ground truth geometry instead of the deep models (`--kind oracle`), a mock detector and mock control features. Its numbers say nothing about real images.

## 2. Data and third party code

```bash
python scripts/download_data.py all --dry-run
python scripts/download_data.py detectors synthbuster synthbuster_ext cospy flickr30k
python scripts/download_data.py openfake --openfake-max-per-model 600
python scripts/download_data.py chameleon      # prints the manual access steps
```

## 3. Model check

```bash
python scripts/check_estimators.py --image <path to any real photo>
python scripts/run_detectors.py --config configs/pilot.yaml --check     # after step 4's prepare
```

## 4. Pilot

```bash
C=configs/pilot.yaml
python scripts/prepare_data.py   --config $C
python scripts/extract_geometry.py --config $C
python scripts/run_detectors.py  --config $C
python scripts/extract_controls.py --config $C
python scripts/run_experiments.py --config $C     # prints the GO / REVISE / STOP decision
python scripts/run_ablations.py  --config $C
python scripts/make_tables.py    --config $C
```

## 5. Full study

```bash
C=configs/default.yaml
python scripts/prepare_data.py    --config $C
python scripts/extract_geometry.py --config $C
python scripts/run_detectors.py   --config $C
python scripts/extract_controls.py --config $C
python scripts/run_experiments.py --config $C
# estimator sensitivity (clean copies only)
python scripts/extract_geometry.py --config $C --variants clean --feature-dir features_marigold --set geometry.normal_estimator=marigold
python scripts/extract_geometry.py --config $C --variants clean --feature-dir features_retinex  --set geometry.shading_estimator=retinex
python scripts/run_ablations.py   --config $C --set "geometry.alternative_feature_dirs=[features_marigold, features_retinex]"
# error analysis and qualitative maps
python scripts/export_errors.py   --config $C --detector bfree --model M3 --n 20
python scripts/extract_geometry.py --config $C --variants clean --feature-dir features_err --ids results/full/errors_bfree_M3.txt
python scripts/make_tables.py     --config $C
python scripts/generate_figures.py --config $C
python scripts/export_scene_audit.py --config $C --n 300     # fill human_scene, then --score
# optional outside reference (24 GB GPU)
python scripts/zero_shot_mllm.py  --config $C --limit 2000
```

Note on the estimator ablation: `alternative_feature_dirs` ablations use clean copies only, so the robustness rows in the main paper always use the default estimators.

## 6. Paper

```bash
python scripts/make_static_figures.py --no-placeholders     # only if you changed static figures
cp results/full/figures/*.{pdf,png} paper/figures/ && rm -f paper/figures/*.placeholder
python scripts/export_scene_audit.py --config configs/default.yaml --score results/full/scene_audit.csv   # after labelling
python scripts/fill_results.py --config configs/default.yaml --pilot-config configs/pilot.yaml           # writes paper_values.md and *_filled.tex
python scripts/fill_results.py --config configs/default.yaml --pilot-config configs/pilot.yaml --inplace # when checked
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd ..
python scripts/build_docx.py
python scripts/check_paper.py
```

## Determinism

* Split assignment, subsampling, cross validation folds and bootstraps use `experiment.seed` (default 0).
* The logistic stacker is deterministic given its inputs.
* Deep model outputs can differ in the last decimals across GPU types and library versions; the feature tables are saved so the statistical analysis can be repeated exactly without the GPU.

## Traceability

Every number in the paper comes from one of: `metrics/metrics.csv`, `metrics/per_generator.csv`, `metrics/ablations.csv`, `metrics/feature_shift.csv`, `hypotheses.json`, `split_summary.csv`, `leakage_report.json`, `logs/timing_*.json`, `stackers/*/*.json`. `make_tables.py` writes the LaTeX rows directly from these files.
