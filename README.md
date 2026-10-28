# PRISM-3D: does explicit monocular 3D geometry help AI generated image detectors?

This repository contains the code, configurations, paper sources and documentation for a controlled study of one question:

> Do explicit 3D geometry features, computed from frozen monocular estimators, improve existing AI generated image detectors on generators and real image sources unseen during fitting, does the gain survive common image processing, and can it be attributed to geometry rather than to backbone features or content?

**Status (11 Oct 2026):** code complete and tested on CPU (unit tests and an end to end smoke test on synthetic renders). Real experiments have **not** been run yet; every result in the paper is a yellow `[xyz]` placeholder. Run the pilot first (see below).

## What is where

| Path | Content |
| --- | --- |
| `src/prism3d/` | library: data (manifest, splits, dedup, processing), geometry (estimators, 26 explicit features), detectors (B-Free, CO-SPY, UnivFD adapters), stacking, metrics, statistics, experiment logic, figures |
| `scripts/` | one CLI per stage, plus downloads, checks, smoke test, paper build |
| `configs/` | `default.yaml` (full study), `pilot.yaml` (go / no-go pilot), `smoke.yaml` (synthetic wiring test) |
| `tests/` | unit tests (`python -m unittest discover -s tests -t .`) |
| `notebooks/colab_pipeline.ipynb` | the whole study on Google Colab, cell by cell |
| `paper/` | `main.tex`, `supplementary.tex`, `references.bib`, `research_paper.docx`, compiled PDFs, `figures/` |
| `docs/` | methodology guide, SOP, reproducibility, experiment log, dataset guide, limitations, submission checklist |

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .      # 40 tests, under a minute on CPU
python scripts/run_smoke.py                     # end to end on synthetic data, about 2 minutes
```

Then follow `notebooks/colab_pipeline.ipynb` on a GPU; `docs/how_to_run.md` walks through it step by step (`docs/reproducibility.md` has the plain commands):

1. `scripts/download_data.py` datasets and official detector code and weights
2. `scripts/check_estimators.py` and `scripts/run_detectors.py --check` (sanity checks)
3. Pilot with `configs/pilot.yaml`: prepare, geometry, detectors, controls, experiments. Read the GO / REVISE / STOP decision.
4. Full study with `configs/default.yaml`, ablations, tables, figures
5. Copy figures, fill the `[xyz]` values with `scripts/fill_results.py` (see `docs/filling_results.md`), rebuild with `scripts/build_docx.py`, check with `scripts/check_paper.py`

## Method in one paragraph

Each image is scored by a frozen base detector (B-Free, CO-SPY or UnivFD). Frozen geometry models (MoGe-2, Depth Anything V2, DSINE, an ordinal shading decomposition) produce depth, normals, intrinsics and shading, from which we compute 26 explicit features in four groups: agreement of depth and normals (G1), consistency of shading with shape via a robust spherical harmonic fit (G2), perspective consistency via vanishing points and focal length (G3), and alignment of depth edges with image edges (G4). A logistic stacker, fitted on a separate pool of generators with generator grouped cross validation and calibrated on a validation split, combines the inputs of five models: M1 base, M2 geometry, M3 base + geometry, M4 base + DINOv2 control, M5 all. Four test sets share no generator, real source or caption with the fit pool. Inference uses a generator level cluster bootstrap with Holm correction.

## Licence

Code: MIT (see `LICENSE`). Datasets, weights and third party code keep their own licences (`docs/dataset_guide.md`).
