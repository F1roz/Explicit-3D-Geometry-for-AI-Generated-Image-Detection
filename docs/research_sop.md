# Research SOP: from the research question to submission

This is the chronological procedure for the PRISM-3D study. Every step lists its objective, tasks, tools, output, how to validate it, common mistakes, an estimated time, what it depends on, and the condition for moving on. Times are estimates for one person with Colab Pro; GPU times are rough until the pilot measures throughput (step 7).

The checklist at the end records what is **done (with evidence)** and what is **pending**. Do not tick an item without the evidence named next to it.

---

## Step 1. Research question selection

* **Objective:** one narrow, testable question.
* **Tasks:** read the v1 proposal, reviewer feedback and v2 doc; fix the question and hypotheses H1 to H5.
* **Tools:** v2 proposal doc, `docs/methodology.md` sections 4 and 18.
* **Output:** the research question and the hypothesis table (paper Section 3, Table 2).
* **Validation:** each hypothesis has a test and a pass criterion written before any data is seen.
* **Common mistakes:** a question that bundles several ideas; hypotheses phrased so any outcome "passes".
* **Time:** 1 day. **Depends on:** nothing. **Proceed when:** hypotheses are frozen in the paper and in `configs/default.yaml` (`evaluation`, `pilot`).

## Step 2. Literature review

* **Objective:** know what is established and where the gap is.
* **Tasks:** read the 100 papers in the spreadsheet; for the 20 closest (geometry, hybrid detectors, bias free training, benchmarks) read the full paper, not the abstract.
* **Tools:** `Synthetic_Image_Detection_Literature_Review_100_Papers.xlsx`.
* **Output:** related work section and Table 1 of the paper.
* **Validation:** every claim about prior work cites a paper you have opened.
* **Common mistakes:** citing from abstracts only; missing recent arXiv work (search arXiv for "AI generated image detection geometry" again before submission).
* **Time:** 1 to 2 weeks. **Depends on:** 1. **Proceed when:** you can state in two sentences why Sarkar et al. (2024) does not already answer the question.

## Step 3. Novelty assessment

* **Objective:** make sure the contribution is a real gap.
* **Tasks:** search Google Scholar, arXiv and CVF open access for "monocular depth" + "AI generated image detection", "surface normals" + "synthetic image", "physical consistency" + "deepfake". Record the queries and date in `docs/experiment_log.md`.
* **Output:** a short novelty note; adjust the paper's novelty wording if a close paper appears.
* **Validation:** the claim in the paper is limited to what you checked ("to our knowledge").
* **Common mistakes:** claiming "first" without a search log.
* **Time:** 1 day. **Depends on:** 2. **Proceed when:** no paper already measures the same thing with the same controls.

## Step 4. Research design

* **Objective:** fix datasets, splits, models, metrics and statistics before running anything.
* **Tasks:** review `configs/default.yaml` (splits, generator lists, seen family patterns), `configs/pilot.yaml`, the metrics and the decision rule.
* **Output:** the frozen configs (commit them with a date).
* **Validation:** `python -m unittest tests.test_data` passes (it checks the config inheritance and the split logic).
* **Common mistakes:** changing generator lists after seeing test results.
* **Time:** 1 day. **Depends on:** 1 to 3. **Proceed when:** configs committed.

## Step 5. Dataset acquisition

* **Objective:** all datasets downloaded legally, in the expected layout.
* **Tasks:** request Chameleon access early (it can take days); run `python scripts/download_data.py all --dry-run`, then the real downloads; check folder names against `docs/dataset_guide.md`.
* **Tools:** Colab, Google Drive (about 100 to 150 GB free space recommended for the full study; less for the pilot).
* **Output:** `data/raw/*` folders.
* **Validation:** `python scripts/prepare_data.py --config configs/pilot.yaml --limit 5` runs and prints counts per split.
* **Common mistakes:** downloading the OpenFake *train* split (we use test only); not checking the real photo folder name in the extended Synthbuster release; ignoring licences.
* **Time:** 1 to 3 days (mostly waiting). **Depends on:** 4. **Proceed when:** every source in the config exists, or Chameleon is replaced by AIGIBench and the config updated.

## Step 6. Environment setup

* **Objective:** a working GPU environment with all models.
* **Tasks:** follow `notebooks/colab_pipeline.ipynb` section 1; run `python scripts/download_data.py detectors`; run `python scripts/check_estimators.py --image <a real photo>`.
* **Output:** installed packages; `results/check_estimators/panel.png`.
* **Validation:** the panel shows depth and normals that look plausible; predicted normals resemble depth normals on the real photo; no import errors.
* **Common mistakes:** mismatched `transformers` version for MoGe or Qwen; forgetting the B-Free weights; running out of GPU memory (lower `geometry.work_long_side`).
* **Time:** half a day. **Depends on:** 5. **Proceed when:** `check_estimators.py` completes and `run_detectors.py --check` prints logits where AI images score higher on average.

## Step 7. Pipeline validation

* **Objective:** prove the code works end to end before using real GPU time.
* **Tasks:** `python -m unittest discover -s tests -t .`; `python scripts/run_smoke.py`; then the pilot (`configs/pilot.yaml`) stages 1 to 7.
* **Output:** `results/smoke/` (wiring test), `results/pilot/` with `hypotheses.json` containing the pilot decision; measured seconds per image in `results/pilot/logs/timing_*.json`.
* **Validation:** all tests pass; smoke test prints `SMOKE TEST PASSED`; pilot decision recorded in `docs/experiment_log.md`.
* **Common mistakes:** treating smoke numbers as results (they are synthetic); changing the decision rule after seeing pilot numbers.
* **Time:** 1 to 2 days (pilot GPU time measured here). **Depends on:** 6. **Proceed when:** decision is GO (or REVISE and the revised pilot is GO). If STOP, go to step 14 and write a negative result.

## Step 8. Baseline implementation

* **Objective:** the three base detectors produce correct scores on every image and copy.
* **Tasks:** `python scripts/run_detectors.py --config configs/default.yaml --check`, then the full run. Compare a few scores with the official demo outputs (B-Free ships `demo_images/results.csv`).
* **Output:** `results/full/scores/<detector>/<variant>.csv`.
* **Validation:** on Test B, M1 AUROC for each detector should be in a plausible range compared with numbers reported by the authors on related data (not identical, our data differ); real and fake mean logits differ in the right direction.
* **Common mistakes:** a detector's preprocessing differs from the official one (we call their code, so do not rewrite it); sign errors.
* **Time:** 1 day plus GPU time. **Depends on:** 7. **Proceed when:** all score files exist for all variants.

## Step 9. Main experiments

* **Objective:** fit M1 to M5 and run the preregistered tests.
* **Tasks:** `extract_geometry.py`, `extract_controls.py`, `run_experiments.py` with `configs/default.yaml`.
* **Output:** `predictions/`, `metrics/metrics.csv`, `metrics/per_generator.csv`, `hypotheses.json`.
* **Validation:** `leakage_report.json` shows empty overlaps; `stackers/*/M1.json` has a sensible temperature (between about 0.3 and 3); number of unseen generators per test set matches the dataset table.
* **Common mistakes:** rerunning with a changed config and keeping the run that looks better (never do this; log every run).
* **Time:** GPU time for geometry dominates (measure in the pilot); fitting and tests take minutes. **Depends on:** 8. **Proceed when:** `hypotheses.json` exists and the run is logged.

## Step 10. Ablation studies

* **Objective:** measure each component's contribution.
* **Tasks:** extract the alternative feature sets (Marigold normals, Retinex shading) for the clean copies, then `run_ablations.py` with `geometry.alternative_feature_dirs` set.
* **Output:** `metrics/ablations.csv`.
* **Validation:** every ablation listed in the paper's Table 8 has a row; intervals present.
* **Common mistakes:** changing two things in one ablation; tuning an ablation on test data.
* **Time:** about one more geometry pass over clean images (alternatives) plus minutes. **Depends on:** 9.

## Step 11. Statistical testing

* **Objective:** correct inference.
* **Tasks:** read `hypotheses.json`; check the number of unseen fake clusters for each test (if below 5, report per generator results first); apply the practical significance rule.
* **Output:** filled hypothesis table (paper Table 6) via `make_tables.py`.
* **Validation:** each claim uses its matching test (cluster bootstrap for generalization, DeLong only for AUROC, McNemar only for accuracy); Holm applied across detectors.
* **Common mistakes:** quoting the image level DeLong p value as the main evidence; calling a 0.003 AUROC gain an improvement.
* **Time:** half a day. **Depends on:** 9, 10.

## Step 12. Error analysis

* **Objective:** understand failures.
* **Tasks:** list M3's false positives and false negatives on Test B and C (`predictions/all_main.csv.gz`, threshold column); write 20 of each id to `errors.txt` and run `python scripts/extract_geometry.py --config configs/default.yaml --variants clean --feature-dir features_err --ids errors.txt` (fast for 40 images; maps land in `results/full/maps/clean/`); look at them; group them by scene type and cause.
* **Output:** error analysis paragraph and the qualitative figure.
* **Validation:** claims about failure patterns are backed by counts, not one example.
* **Common mistakes:** choosing only flattering examples for the qualitative figure.
* **Time:** 1 to 2 days. **Depends on:** 9.

## Step 13. Visualization

* **Objective:** every figure from result files.
* **Tasks:** `python scripts/generate_figures.py --config configs/default.yaml`; copy figures into `paper/figures/` (the notebook's last cell does this and removes the `.placeholder` markers); `python scripts/make_static_figures.py --no-placeholders` if you changed static figures.
* **Output:** all paper figures.
* **Validation:** `python scripts/check_paper.py` reports zero figure placeholders; look at every figure at print size.
* **Common mistakes:** editing numbers in a figure by hand; mismatched colours between figures.
* **Time:** 1 day. **Depends on:** 9 to 12.

## Step 14. Paper writing

* **Objective:** a complete manuscript.
* **Tasks:** run `python scripts/fill_results.py` (see `docs/filling_results.md`): check `results/full/tables/paper_values.md` and `paper/main_filled.tex`, then rerun with `--inplace`; write the six `manual.*` keys by hand; bold the best value per column of the main table; write the discussion paragraph that matches the outcome (delete the others); write the error analysis; rebuild the Word file with `python scripts/build_docx.py`.
* **Output:** `paper/main.pdf`, `paper/research_paper.docx`.
* **Validation:** `python scripts/check_paper.py` shows 0 placeholders and no hyphenated words; abstract, tables, figures and conclusion state the same numbers.
* **Common mistakes:** a number in the abstract that differs from the table; overclaiming in the conclusion.
* **Time:** 1 to 2 weeks. **Depends on:** 13.

## Step 15. Citation verification

* **Objective:** every reference is real and correct.
* **Tasks:** open every link in `paper/references.bib`; complete author lists marked "and others" if the venue requires full lists; replace arXiv entries with the published version where one exists (check CVPR 2026 and NeurIPS 2025 proceedings).
* **Output:** final `references.bib`.
* **Validation:** `check_paper.py` reports no missing keys; each entry checked by hand.
* **Time:** 1 day. **Depends on:** 14.

## Step 16. Internal review and revision

* **Objective:** catch problems before reviewers do.
* **Tasks:** give the paper to a colleague who has not seen it; ask them the 10 reviewer questions in `docs/methodology.md` section 21; read the paper aloud once for awkward sentences.
* **Output:** revised manuscript.
* **Time:** 1 week. **Depends on:** 14, 15.

## Step 17. Reproducibility audit

* **Objective:** someone else can reproduce the results.
* **Tasks:** on a fresh Colab runtime, follow `docs/reproducibility.md` for the pilot only and compare the pilot numbers with yours (they should match exactly for the stacker given the same scores and features; geometry outputs may differ in the last decimals across GPU types).
* **Output:** a note in `docs/experiment_log.md`.
* **Time:** 1 day. **Depends on:** 9.

## Step 18. Venue selection

* **Objective:** choose a venue that fits the actual result. See `docs/submission_checklist.md` for the verified options (quartiles from SCImago 2025, CORE rank of ICIP).
* **Tasks:** decide after results: a positive result with all controls suits a Q1 journal such as *Forensic Science International: Digital Investigation*; a negative or mixed result also fits there or in *Journal of Information Security and Applications*; a short version suits ICIP (CORE B).
* **Time:** half a day. **Depends on:** 14.

## Step 19. Submission preparation

* **Objective:** a submission package that meets the venue's rules.
* **Tasks:** move the body of `main.tex` into the venue template (Elsevier: `elsarticle`); check word and page limits, reference style, highlights, graphical abstract if required, data availability statement, generative AI declaration (Elsevier requires one), CRediT author statement, cover letter; make the code repository public or provide an anonymous link for double blind review.
* **Output:** submission files.
* **Validation:** `docs/submission_checklist.md` all ticked.
* **Time:** 2 to 3 days. **Depends on:** 16 to 18.

## Step 20. Final submission and revision planning

* **Objective:** submit and prepare for reviews.
* **Tasks:** submit; keep the frozen code version tagged; prepare a reply template; list experiments reviewers may ask for (another base detector such as AIDE, adversarial evaluation, partial edits) and estimate their cost.
* **Time:** 1 day, then the review period. **Depends on:** 19.

---

## Checklist

| # | Task | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Research question and hypotheses fixed | Done | v2 proposal doc; paper Table 2; `configs/default.yaml` evaluation and pilot sections |
| 2 | Literature review (100 papers) | Done (abstract level for recent preprints) | spreadsheet; full reading of the 20 closest papers is **pending** |
| 3 | Novelty search log | Pending | add queries and date to `docs/experiment_log.md` |
| 4 | Research design and configs | Done | `configs/*.yaml`; `tests/test_data.py` passes |
| 5 | Datasets downloaded | Pending | requires Colab (network blocked in the authoring environment) |
| 5a | Chameleon access requested | Pending | email to the AIDE authors |
| 6 | GPU environment and model check | Pending | `results/check_estimators/panel.png` |
| 7a | Unit tests | Done | `python -m unittest discover -s tests -t .`: 40 tests passed, 1 slow test skipped (run on 11 Oct 2026) |
| 7b | Smoke test (synthetic, CPU) | Done | `python scripts/run_smoke.py` printed SMOKE TEST PASSED (11 Oct 2026) |
| 7c | Pilot and decision | Pending | `results/pilot/hypotheses.json` key `pilot` |
| 8 | Base detector scores | Pending | `results/full/scores/` |
| 9 | Main experiments | Pending | `results/full/hypotheses.json` |
| 10 | Ablations | Pending | `results/full/metrics/ablations.csv` |
| 11 | Statistical tests reported | Pending | filled Table 6 |
| 12 | Error analysis | Pending | qualitative figure and paragraph |
| 13 | Result figures | Pending | `check_paper.py` shows 0 figure placeholders |
| 13a | Static figures (architecture, design, workflow, synthetic checks) | Done | `paper/figures/fig_architecture.pdf` and others |
| 14 | Paper text with results | Draft done, results pending | `paper/main.tex` (222 placeholders), `research_paper.docx` |
| 15 | Citation verification | Partly done | references checked against arXiv or venue pages on 11 Oct 2026; final manual pass pending |
| 16 | Internal review | Pending | |
| 17 | Reproducibility audit | Pending | |
| 18 | Venue chosen | Pending (options verified) | `docs/submission_checklist.md` |
| 19 | Submission package | Pending | |
| 20 | Submitted | Pending | |
