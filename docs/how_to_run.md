# How to run the study (one laptop, one Colab account)

No code changes are needed. You run `notebooks/colab_pipeline.ipynb` in Google Colab, top to bottom. Everything is saved to Google Drive, so after a disconnect you start the session again and rerun the same cell; finished images are skipped.

## Step 1: one time setup
- About 150 GB free on Google Drive (Google One plan if needed).
- Colab Pro (the free tier's GPU time runs out on the full study).
- Upload `prism3d.zip` to the top level of My Drive.
- Unzip it on your laptop once to get `notebooks/colab_pipeline.ipynb`.
- Email the AIDE authors for Chameleon access (github.com/shilinyan99/AIDE).

## Step 2: start a Colab session (every time)
1. colab.research.google.com, Upload, pick `colab_pipeline.ipynb`.
2. Runtime, Change runtime type, GPU (A100 or L4 if offered; T4 is enough for the pilot).
3. Run section 1 in order: `nvidia-smi`, Drive mount, unzip, install.
4. Continue with the step you were on.
5. Before closing Colab, run the last cell (End of a session).

## Step 3: tests, downloads, model checks
- Section 2: unit tests and smoke test. Pass: `OK` and `SMOKE TEST PASSED`.
- Section 3: dry run, then downloads (hours).
- Check the real photo folder name in `data/raw/synthbuster_ext/` (see settings below).
- Section 4: model check; `panel.png` maps should follow the scene.

## Step 4: pilot (section 5)
1. Prepare and detector check (generated images must score higher).
2. Geometry, detectors, controls, experiments, ablations, tables.
3. Print the decision. GO: Step 5. REVISE: send `results/pilot/hypotheses.json` for review before changing anything. STOP: still run the full study; the paper becomes a negative result.
4. Record the decision in `docs/experiment_log.md`.

## Step 5: full study (section 6)
1. Prepare and detector check. Without Chameleon add `--set data.skip_missing_sources=true` to every full study command.
2. Check `results/full/leakage_report.json`: nothing shared.
3. Geometry (longest job, several sessions).
4. Detectors, controls, experiments. Pass: `results/full/hypotheses.json` and `results/full/metrics/metrics.csv` exist.

## Step 6: ablations, errors, figures, paper values
1. Estimator ablation cell (Marigold, Retinex).
2. New cell:
   `python scripts/export_errors.py --config configs/default.yaml --detector bfree --model M3 --n 20`
   `python scripts/extract_geometry.py --config configs/default.yaml --variants clean --feature-dir features_err --ids results/full/errors_bfree_M3.txt`
3. Scene audit: `python scripts/export_scene_audit.py --config configs/default.yaml --n 300`, label `human_scene`, upload, then `--score results/full/scene_audit.csv`.
4. Optional: zero shot Qwen cell (A100).
5. Ablations cell (ablations, tables, figures).
6. Section 7: figures into `paper/figures/`, `fill_results.py`; check `paper_values.md`, run the `--inplace` line, then copy `paper` to Drive again.
7. Upload `main.tex`, `supplementary.tex`, `figures/` to Overleaf.

## Settings you may need (no code edits)
| Situation | Change |
| --- | --- |
| Real photo folder in `synthbuster_ext` not named `raise`, `real` or `raise_1k` | add it to `real_dirs` in `configs/default.yaml` |
| Chameleon not available | `--set data.skip_missing_sources=true`; remove Test D from the paper |
| GPU out of memory in geometry | `--set geometry.work_long_side=512` on every geometry command; mention in methods |
| DSINE fails to load | `--set geometry.normal_estimator=marigold`; say so in methods; drop the Marigold ablation row |

Files edited inside Colab are lost at the next unzip. Edit `configs/` in the zip on your laptop and replace `prism3d.zip` on Drive.

## If a run fails
Rerun the cell once. If it fails again, keep the last 30 lines of output and get help before moving on; later steps need this step's output.
