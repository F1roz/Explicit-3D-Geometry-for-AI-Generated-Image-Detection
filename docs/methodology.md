# Methodology guide: what we are doing, why, and how

This guide explains the whole study so that you can run it, understand every number it produces, and defend it in front of reviewers. It assumes a computer science background but no prior work in image forensics. Each section ends with the files in the repository where the idea is implemented.

Status labels used throughout:

* **Established**: a published finding (with a citation).
* **Design decision**: a choice we made, with the reason.
* **Hypothesis**: something the experiment will test. Nothing in this guide is a result; results come only from running the pipeline.

---

## 1. The problem in plain language

Image generators (Stable Diffusion, FLUX, Midjourney, DALL·E, GPT Image and many others) now produce pictures that look like photographs. A *detector* is a program that takes an image and outputs a score: how likely it is that the image was AI generated.

Detectors work well on images from the generators they were trained on. They work much worse on images from **new** generators, or on images that were **compressed or resized** (which happens every time an image is posted online).

* **Established.** Humans misclassify about 38.7% of AI images in a large study (Lu et al., NeurIPS 2023). Detectors that score highly in the lab fall close to chance on the Chameleon set (Yan et al., ICLR 2025). AIGIBench found big drops for 11 detectors on social media images (Li et al., NeurIPS 2025).

So the useful question is not "can we detect images from a generator we trained on" but "can we detect images from generators we have never seen".

## 2. Why the problem matters

Fabricated photographs are used in misinformation, fraud and harassment. Journalists, platforms and courts need tools whose behaviour on new generators is known, including how often they wrongly flag a real photograph. A detector that flags 10% of real photos is unusable even if it catches most fakes.

## 3. What previous research established

There are three main families of detectors (see the 100 paper spreadsheet, `Synthetic_Image_Detection_Literature_Review_100_Papers.xlsx`, and Section 2 of the paper).

| Family | Idea | Known weakness |
| --- | --- | --- |
| Low level artifacts | Generators leave traces from upsampling and in the frequency spectrum (CNNSpot, NPR, SPAI) | Traces vanish under JPEG and resizing |
| Foundation model features | Large pretrained encoders such as CLIP already separate real from fake (UnivFD, RINE, Effort) | Miss subtle high quality images |
| Reconstruction | Images from a latent diffusion model are reproduced almost perfectly by its autoencoder (DIRE, AEROBLADE) | Tied to one generator family |

Two further findings shape our design:

* **Dataset shortcuts.** In GenImage, real images are JPEG and fakes are PNG, so detectors learn the file format (Grommelt et al., 2024). Any study must remove this.
* **Physics.** Generated scenes often break perspective and lighting rules (Farid, 2022, by hand; Sarkar et al., CVPR 2024, with classifiers on lines and shadows).

## 4. The gap we investigate

Nobody has measured whether **automatic, dense 3D geometry cues** add signal **on top of current state of the art detectors**, on **unseen generators**, with controls that rule out the obvious alternative explanations. That is our gap. We do not claim to build a better detector; we claim to measure something carefully.

## 5. The core idea

A real photograph is a picture of a real 3D scene. That imposes rules:

1. **Shape is consistent.** Depth and surface orientation (normals) must agree. Two independent estimators of them should agree.
2. **Shading follows shape.** How bright a surface is depends on the angle between its normal and the light.
3. **Perspective is consistent.** Parallel lines in the world meet at vanishing points; these determine the camera's focal length.
4. **Occlusions are edges.** Where one object is in front of another, depth jumps and the image has an edge.

A generator learns these rules only from data, so it may break them. We measure each rule with modern geometry models (all frozen, no training) and turn the measurements into 26 numbers per image. Then we ask: if we give these numbers to a stacker alongside an existing detector's score, does detection on unseen generators get better?

**Hypothesis**, not a fact: generated images violate these rules more than photographs, in a way current detectors do not already capture.

## 6. The mathematics you need

### 6.1 Cameras: from depth to 3D points

A pinhole camera maps a 3D point to a pixel. The intrinsic matrix is

```
K = [[f_x, 0,   c_x],
     [0,   f_y, c_y],
     [0,   0,   1  ]]
```

where `f_x, f_y` are focal lengths in pixels and `(c_x, c_y)` is the principal point (usually the image centre). Given the depth `D(u,v)` of pixel `(u,v)`, its 3D position is

```
P(u,v) = D(u,v) · K⁻¹ · [u, v, 1]ᵀ
```

**Example.** With `f = 500`, `c = (320, 240)`, pixel `(420, 240)` at depth 2 m: `K⁻¹[420,240,1] = [0.2, 0, 1]`, so `P = [0.4, 0, 2]` m.

Coordinates follow OpenCV: x right, y down, z forward into the scene.

* Code: `prism3d.geometry.features.backproject`, `intrinsics_from_normalized` (MoGe outputs intrinsics divided by image size).

### 6.2 Surface normals from a point map

Neighbouring pixels give two tangent vectors along the surface; their cross product is perpendicular to the surface:

```
N_D = (∂P/∂u × ∂P/∂v) / ‖∂P/∂u × ∂P/∂v‖
```

The derivatives are central differences (`(P[u+1] − P[u−1]) / 2`). A normal and its negative describe the same surface, so we flip every normal to face the camera (`N · P < 0`).

* Code: `normals_from_points`, `_orient_to_camera`. Tested in `tests/test_features.py` (a flat wall gives exactly (0, 0, −1); a tilted plane gives the analytic normal).

### 6.3 G1: the angle between two normal estimates

```
θ(p) = arccos( ⟨N_D(p), N_pred(p)⟩ )   (degrees)
```

`N_D` comes from MoGe's point map; `N_pred` comes from DSINE, which predicts normals directly from pixels. They are different models trained differently, so if a scene is geometrically coherent they should agree (small θ). We summarise θ by its mean, standard deviation and 90th percentile. A second version uses normals from Depth Anything V2 depth, aligned to MoGe by least squares `1/D ≈ a·d + c` (Depth Anything gives *relative inverse depth* d, known only up to scale and shift).

**Axis conventions.** Models disagree on whether y points up or down, etc. We try all 8 sign flips `(±1, ±1, ±1)` on 50 fit images and keep the one with the smallest mean angle (`estimate_convention`). Because we orient normals towards the camera, a global sign flip is absorbed, so the answer is defined up to sign.

* Code: `g1_features`, `depth_from_disparity_aligned`, `estimate_convention`.

### 6.4 G2: shading, spherical harmonics and robust fitting

For a matte (Lambertian) surface under distant light, the brightness depends only on the surface normal. Ramamoorthi and Hanrahan (2001) showed this function is captured almost exactly by 9 **spherical harmonics**, simple polynomials of the normal `n = (x, y, z)`:

```
Y = [0.282, 0.489·y, 0.489·z, 0.489·x, 1.093·xy, 1.093·yz, 0.315·(3z²−1), 1.093·xz, 0.546·(x²−y²)]
S(p) ≈ Σ_k ℓ_k Y_k(N(p))
```

`ℓ` (9 numbers) describes the lighting of the whole scene. We take the shading layer `S` from an intrinsic decomposition (`I = A ⊙ S`, albedo times shading; Careaga and Aksoy, 2023) and fit `ℓ` by least squares. The **residual** `r = S − Yℓ` is large where shading cannot be explained by the scene's shape under one lighting.

**Robust fitting (Huber loss).** Highlights, shadow edges and texture leaking into `S` create outliers that would dominate ordinary least squares. The Huber loss is quadratic for small residuals and linear for large ones:

```
ρ_δ(r) = r²/2            if |r| ≤ δ
         δ(|r| − δ/2)    otherwise
```

We minimise it by **iteratively reweighted least squares**: fit, compute weights `w = 1` for small residuals and `δ/|r|` for large ones, refit, repeat 15 times. `δ = 1.345 × robust scale` (the classic value that keeps 95% efficiency for Gaussian noise). We floor the scale at 5% of the shading spread, because images with big flat areas otherwise give a scale near zero and the fit collapses (we found this bug on synthetic scenes and fixed it; see the experiment log).

Only pixels with low local albedo variation are used, because texture is the main source of shading errors.

**Light dispersion.** We fit a simple first order lighting vector in each cell of a 3×3 grid and measure how much they differ:

```
δ_L = (1/J) Σ_j ‖ℓ_j − ℓ̄‖² / ‖ℓ̄‖²
```

A real scene usually has one dominant light, so regional fits agree. But mixed lighting and interreflections in real rooms also raise δ_L, which is why G2 is a learned input, never a rule.

* Code: `sh_basis`, `fit_sh_huber`, `low_albedo_variance_mask`, `g2_features`. Tests: `test_sh_fit_recovers_lighting`, `test_huber_ignores_outliers`.

### 6.5 G3: lines, vanishing points and focal length

* **LSD** (Line Segment Detector; von Gioi et al., 2012) finds straight segments.
* A **vanishing point** is where images of parallel 3D lines meet. In homogeneous coordinates, the line through two points is their cross product, and the intersection of two lines is also a cross product.
* **RANSAC**: repeatedly pick two segments, intersect them to get a candidate vanishing point, count how many segments point towards it (direction within 2°), keep the best, refine it with a least squares fit (smallest singular vector), remove its segments, repeat up to 3 times.
* **Focal length from two vanishing points** (Hartley and Zisserman, 2004). If `v1` and `v2` are vanishing points of two perpendicular world directions, then

```
f² = −(v1 − c) · (v2 − c)
```

We compare this `f_VP` with MoGe's focal length: `Δ_f = |log f_VP − log f_MoGe|`. We only compute it when two strong, finite vanishing points within 3 image diagonals of the centre exist and look perpendicular (within 10°). Far vanishing points carry almost no focal information (we found spurious values on synthetic scenes before adding this rule).

**Example (tested).** A camera rotated 10° and 35° with `f = 554.26` px: the formula returns exactly 554.26 (`test_focal_from_orthogonal_vps`).

* Code: `detect_lines`, `fit_vanishing_points`, `focal_from_orthogonal_vps`, `g3_features`.

### 6.6 G4: occlusion edges

Canny edges of the log depth map (`E_D`) should coincide with edges of the image (`E_I`, dilated by 2 px):

```
A_occ = |E_D ∩ E_I| / |E_D|
```

computed at two thresholds (85th and 95th percentile of gradient magnitude).

* Code: `g4_features`.

### 6.7 Missing values and applicability

A close up portrait may have no straight lines; a flat illustration has no depth range. Then a cue cannot be computed. We set the value to NaN, impute it later with the fit split median, and add a flag `miss_* = 1`. Two indicators (`app_valid_frac`, `app_depth_range`) tell the stacker how much geometry the image has.

**Risk.** If generated images were systematically missing a cue more often than real ones, the flag itself would separate them for a reason that has nothing to do with physics. We report flag rates per class and run an ablation without flags.

### 6.8 The logistic stacker

Logistic regression turns a weighted sum into a probability:

```
p(fake | z) = σ(w₀ + wᵀz),   σ(t) = 1 / (1 + e^(−t))
```

`z` is the input vector of a model (Section 7.2). The weights minimise the cross entropy plus an L2 penalty `λ‖w‖²` (`C = 1/λ` in scikit-learn). The output before σ is the **logit** (log odds).

**Temperature scaling** (Guo et al., 2017): divide the logit by a scalar `T` chosen to minimise the negative log likelihood on validation data. It fixes over or under confidence without changing the ranking (AUROC is unchanged).

**Threshold.** We choose the logit threshold so that exactly 5% of validation real images are flagged. We then keep this threshold fixed on every test set and processed copy, so we can see whether processing makes the detector flag more real photographs.

* Code: `prism3d.stacking.Stacker`. Tests: `tests/test_stacking.py`.

## 7. The algorithm step by step

1. **Prepare data** (`scripts/prepare_data.py` → `pipeline.prepare`): read each dataset folder into one manifest table; assign every image to fit, validation or one or more test sets; refuse to continue if anything leaks; remove fit images that nearly duplicate a test image; reencode everything as JPEG 95; make the processed copies.
2. **Extract geometry** (`extract_geometry.py`): for each image and copy, resize to a 640 px long side, run MoGe, Depth Anything V2, DSINE and the intrinsic decomposition, compute the 26 features, save one row per image.
3. **Score with base detectors** (`run_detectors.py`): B-Free, CO-SPY and UnivFD through their official code; save one logit per image.
4. **Controls** (`extract_controls.py`): DINOv2 class token per image and copy; CLIP scene labels.
5. **Fit and evaluate** (`run_experiments.py`): for each base detector, fit M1 to M5 on the fit split, calibrate on validation, predict the test sets, compute metrics and hypothesis tests.
6. **Ablations** (`run_ablations.py`), **tables** (`make_tables.py`), **figures** (`generate_figures.py`).

### 7.1 The four test sets and the fit pool (dataset choice)

| Set | Why it is there |
| --- | --- |
| Fit and validation: Co-Spy-Bench (CC3M captions, 10 older open models) + Flickr30k | A diverse pool to fit the stacker, with real photos from a source used by no base detector and no test set |
| Test A: FLUX, SD3, Playground v2.5, PixArt + RAISE photos | Modern open generators never seen in fitting |
| Test B: Synthbuster (and FLUX/SD 3.5 extension) + RAISE photos | **Content matched control**: fakes were generated from captions of the same RAISE photos, so content and style are similar on both sides |
| Test C: OpenFake test split | The newest commercial systems (GPT Image 1, Imagen 3/4, Midjourney v6/v7, Ideogram 3, ...) |
| Test D: Chameleon | Hard, in the wild images that fooled people |

Tests A and B share the RAISE photographs. The code allows test sets to share images (`eval_sets` column) but never lets a fit image into any test set.

### 7.2 The five models (how our method differs from the baselines)

| Model | Inputs | Role |
| --- | --- | --- |
| M1 | base logit | The baseline: the published detector, calibrated the same way as the others |
| M2 | 26 geometry features | Does geometry carry any signal alone? |
| M3 | base logit + geometry | **Our method**: does geometry add to the detector? |
| M4 | base logit + 26 DINOv2 PCA components | **Control**: is it just a strong backbone? |
| M5 | all of the above | Does geometry add beyond DINOv2? |

All five use the identical stacker procedure, so differences come from the inputs only.

### 7.3 How each baseline works

* **UnivFD** (Ojha et al., CVPR 2023): frozen CLIP ViT-L/14 image features + one linear layer trained on ProGAN fakes vs LSUN photos. It works because CLIP features separate "real photo" from "anything else" surprisingly well. Center crop 224, no resize.
* **B-Free** (Guillaro et al., CVPR 2025): a ViT pretrained with DINOv2, trained on real COCO photos vs SD 2.1 *inpainted versions of the same photos*, so real and fake share content and the model must learn generator traces. Uses 504×504 crops without resizing.
* **CO-SPY** (Cheng et al., CVPR 2025): combines CLIP semantic features with features from VAE reconstruction artifacts, fused adaptively; the released checkpoint we use was trained on DRCT-2M (COCO photos vs SD 1.4 images).

These three come from different families (semantic, bias free artifact, hybrid), so a gain on all three is more convincing than a gain on one.

## 8. Data preprocessing and quality checks

| Step | What and why | Code |
| --- | --- | --- |
| Manifest | One row per image with label, generator, real source, caption id; one source of truth | `data/manifest.py`, `data/ingest.py` |
| Selectors | Config rules choose images per split by dataset, generator, caption source and real source | `data/splits.py::select` |
| Leakage check | Stops if a generator, real source or caption appears in fit and test | `check_leakage` |
| Caption grouping | Fit and validation are split by caption, so the same caption is never on both sides | `assign_splits` |
| Near duplicates | 64 bit perceptual hash (DCT of a 32×32 grey thumbnail, top 8×8 block vs median); fit images within 8 bits of any test image are dropped | `data/dedup.py` |
| Harmonisation | Every image reencoded as JPEG 95, so file format cannot leak the label | `data/processing.py::write_processed` |
| Robustness copies | JPEG 75; half resize; screenshot chain (0.75 resize then JPEG 80); stored so the final file equals the final step | `VARIANT_CHAINS` |
| Unreadable files | Logged in `failed_images.csv` and dropped | `pipeline.prepare` |

## 9. Evaluation metrics (what they measure, why they matter)

* **AUROC**: the probability that a random fake scores higher than a random real image. 0.5 is chance, 1.0 is perfect. Threshold free.
* **Macro AUROC**: the AUROC of each generator's fakes against all real images, averaged over generators. Gives every generator equal weight, which matters because generators contribute different numbers of images. **This is our primary metric.**
* **TPR at 1% FPR**: the share of fakes caught when only 1% of real photos may be flagged. Forensic users care about this low false alarm regime.
* **FPR at the fixed threshold**: share of real photos flagged at the validation threshold. Under processing, a rise means the detector starts accusing real photos.
* **ECE** (expected calibration error): split predictions into 15 probability bins, compare the average predicted probability with the actual share of fakes in each bin, average the gaps weighted by bin size. 0 means perfectly calibrated.
* **Balanced accuracy**: mean of accuracy on reals and on fakes at the threshold.

Code: `prism3d.metrics`. Tests: `tests/test_metrics.py` (includes a check that macro AUROC weights a 10 image generator equally with a 100 image generator).

## 10. Statistical significance and confidence intervals

### 10.1 Why a cluster bootstrap

Our claim is about **new generators**. Images from the same generator are not independent: they share its quirks. If we resample individual images, we pretend we have thousands of independent observations, and the interval comes out far too narrow. The honest unit is the generator.

**Generator level cluster bootstrap** (Field and Welsh, 2007):

1. Resample the unseen fake generators with replacement (and the real sources).
2. Inside each chosen generator, resample its images with replacement.
3. Compute macro AUROC for M3 and for M1 on this same resample; store the difference.
4. Repeat 2,000 times. The 2.5th and 97.5th percentiles are the 95% interval.

If the interval excludes 0, the gain is unlikely to be an accident of which generators happened to be in the test set. With fewer than 5 unseen generators the interval is unreliable; we then lead with per generator results.

Code: `stats.cluster_bootstrap_delta`. Tests check that a real gain gives an interval above zero, that identical models give exactly 0 inside the interval, and that the result is reproducible with a seed.

### 10.2 Multiple testing (Holm)

We test H1 for three detectors. Testing three times raises the chance of one false positive. Holm's procedure: sort p values ascending, compare the smallest with α/3, the next with α/2, the last with α/1; stop at the first failure. It controls the family wise error at α = 0.05 and is less conservative than Bonferroni.

### 10.3 Supplementary tests and matching tests to claims

* **DeLong test**: compares two correlated AUROCs on the same images, image level. Used only for AUROC claims.
* **McNemar test**: compares two classifiers' correctness on the same images at a fixed threshold. Used only for accuracy claims.
* A reviewer correctly pointed out in an earlier version that a DeLong test cannot support an accuracy claim, and that "non overlapping seed ranges" is not a statistical test. Both are fixed.

### 10.4 Practical significance

A gain can be statistically significant but tiny. Gains below 0.01 AUROC are reported as "detectable but small".

## 11. The pilot and its decision rule

Before spending GPU time on the full study, the pilot (`configs/pilot.yaml`) uses UnivFD, Tests A and B (subsampled), clean and JPEG 75 copies. The rule was written before seeing any number (`experiment.pilot_decision`):

* **GO** if (1) the M3 gain is at least 0.01 macro AUROC with the lower 90% bound above 0, (2) the M3 gain is larger than the M4 gain, and (3) the extra false positive drift under JPEG 75 is at most 2 points.
* **REVISE** if geometry alone (M2) reaches 0.65 but rule 1 fails: the signal exists but overlaps with the detector.
* **STOP** otherwise: write up a negative result.

UnivFD is the weakest base, so a gain is easiest to find there. A failure at this stage is strong evidence against the hypothesis; a pass only permits the real test.

## 12. What each ablation demonstrates

Every ablation changes one thing in M3 and keeps splits, base scores, stacker procedure and test images fixed (`experiment.ablation_specs`).

| Ablation | Hypothesis it tests | If performance drops... |
| --- | --- | --- |
| drop_G1 ... drop_G4 | Each cue adds beyond the others | ...that cue carries unique information |
| only_G1 ... only_G4 | Each cue carries signal alone | ...(compare to M1) |
| no_flags | Missing value flags do not leak the class | ...flags were carrying signal; check whether they are physical or a shortcut |
| gbdt | Conclusions do not depend on the linear form | ...nonlinear interactions matter |
| processing_aware_fit | Robustness can be learned from fit data only | ...(gain under processing should rise) |
| estimator_features_marigold / _retinex | Results do not depend on one geometry model | ...the effect is tied to a specific estimator |

## 13. How to read each table and figure

| Item | What to look at |
| --- | --- |
| Table "main results" | Per detector, compare M3 with M1 in each column. Then compare M3 with M4. Bold marks the best model. |
| Model comparison figure | Grey bar (M1) vs blue bar (M3) vs orange bar (M4) within each test set |
| ROC curve | Look at the left part (log axis): that is the low false positive regime |
| Per generator forest plot | Each dot is one generator's gain; dots right of zero mean geometry helped that generator. Consistency across detectors matters more than one large dot |
| Hypothesis table | Estimate, interval, Holm p. A hypothesis passes only if its stated criterion is met |
| Robustness figure | Left: do the solid (M3) lines stay above the dashed (M1)? Right: do the solid lines rise more than the dashed? (extra false positives) |
| Feature shift figure | Long blue bars mean a feature moves on real photos when compressed: a false positive risk |
| Ablation figure | Markers left of zero with intervals not crossing zero mean that component matters |
| Scene heatmap | Red cells: geometry helps; blue: hurts. Expect outdoor and architectural scenes to benefit most (hypothesis H5) |
| Synthetic figures | Only prove the code works; never cite them as evidence about real images |
| Example image figure | Rows are chosen, not random: 2 correct real photos, 2 correct generated images (Test B), then the most confident false positive and false negative of M3. Look at whether the angle and residual maps are brighter on generated images, and what fooled the model in the last two rows |

## 14. Leakage, confounding and misleading improvements

Ask these questions of every improvement:

1. **Could the stacker see the test generator?** No: generator overlap stops the run (`LeakageError`).
2. **Could the base detector have seen the test images?** Base detectors were trained on COCO, LSUN, ImageNet photos; no test set uses those sources.
3. **Could file format or size separate classes?** All images are JPEG 95 reencoded; sizes differ by dataset, but real and fake within Test B come from the same RAISE captions.
4. **Is it the backbone?** Compare M3 with M4 (DINOv2 control).
5. **Is it content?** Look at Test B.
6. **Is it the flags?** Look at the no_flags ablation.
7. **Was anything tuned on the test set?** No: C by grouped CV on fit, temperature and threshold on validation. Do not change the config after looking at test results; if you must, record it in `docs/experiment_log.md` and say so in the paper.
8. **Is it one generator?** Look at the forest plot.

## 15. How to read failed experiments

* **M2 near 0.5**: geometry features carry no signal on that test set. Check `fig_feature_distributions` to see whether real and fake overlap completely.
* **M3 = M1**: the stacker gave geometry near zero weight. Check `stackers/<det>/M3.json` coefficients.
* **M3 > M1 but M3 ≈ M4**: the gain is backbone semantics, not geometry (H2 fails).
* **Gain on Test A but not Test B**: geometry may be tracking content or style (H3 fails).
* **Gain vanishes under JPEG**: geometry estimates are fragile; check `fig_feature_shift`.

None of these is a failure of the project. They are findings, and the paper's discussion section has a paragraph for each.

## 16. Reproducing the results

See `docs/reproducibility.md` for exact commands. Short version on Colab:

```bash
pip install -r requirements.txt
pip install git+https://github.com/microsoft/MoGe.git
pip install https://github.com/compphoto/Intrinsic/archive/main.zip
python -m unittest discover -s tests -t .
python scripts/run_smoke.py
python scripts/download_data.py all
python scripts/prepare_data.py --config configs/pilot.yaml
# ... (see notebooks/colab_pipeline.ipynb)
```

### 16.1 From result files to the paper

Every result in the paper is a yellow `[xyz]` with a hidden name, for example `\xyz[h1.bfree.delta]` (the H1 gain for B-Free). The name says which result file and which number belong there, so nothing is copied by hand:

1. `scripts/generate_figures.py` draws the 12 result figures into `results/full/figures/`. Copying them into `paper/figures/` replaces the yellow placeholder figures, because they have the same file names.
2. `scripts/fill_results.py` computes every named value from the result files and writes `results/full/tables/paper_values.md`: one row per name with its value, the file it came from, and the line of the paper where it appears. It writes `paper/main_filled.tex` for checking, and with `--inplace` it writes the values into `main.tex` and `supplementary.tex` (a `.bak` copy is kept).
3. Six names start with `manual.` (what the example images show, the most common errors, two OpenFake counts, the conclusion). Only you can write those; they stay yellow until you do.
4. `scripts/check_paper.py` counts what is left. The paper is complete when it reports 0 placeholders and 0 placeholder figures.

Full details and the list of name families: `docs/filling_results.md`. Step by step run instructions for one Colab account: `docs/how_to_run.md`.

## 17. Debugging common problems

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `LeakageError` in prepare | A selector picks the same generator for fit and test | Fix the generator lists in the config |
| `FileNotFoundError: Dataset root not found` | Dataset not downloaded or different folder name | Run the download step; or set `data.skip_missing_sources: true` for partial runs |
| Generator names do not match config patterns | Names are normalised (lower case, `@` → `/`) | Print `manifest_all.csv` generators and adjust the patterns |
| `run_detectors --check` shows real images with higher logits | Sign convention | Report to us; for CO-SPY confirm `predict` returns P(fake) |
| CUDA out of memory in geometry | Large images | Lower `geometry.work_long_side` to 512; restart runtime |
| Convention median angle > 40° | Wrong normal convention or estimator failure | Run `check_estimators.py` on a real photo and inspect the panel |
| All `g3_*` features NaN | Few lines (portraits, nature) | Expected; flags handle it |
| Colab disconnects | Session limit | Rerun setup cells; every stage resumes |
| `ValueError: Need at least 5 fake generators in the fit split` | Too few fit generators after filtering | Check the fit selectors and downloads |

## 18. What would support or reject the hypothesis

| Outcome | Interpretation |
| --- | --- |
| H1 to H4 pass | Geometry adds robust, non semantic signal |
| H1 passes, H2 fails | Gain explained by DINOv2 features |
| H1 passes, H3 fails | Gain explained by content or style |
| H1 passes, H4 fails | Gain real but fragile under processing |
| H1 fails | No detectable added value from explicit geometry |

## 19. What this research cannot establish

* That geometry detects images from generators released after the study.
* That a high geometry score proves an image is fake (real scenes violate the simple models too).
* Anything about partially edited images, video, or adversarial attacks.
* Causality inside the geometry models (they may fail differently on generated images for reasons unrelated to physics).

## 20. Contribution to the field

Either a new, physically motivated signal shown to add to state of the art detectors under strict controls, or a careful negative result showing that explicit geometry does not help (or helps only through confounds). Both answer a question the literature has left open since Sarkar et al. (2024).

## 21. Questions reviewers will likely ask (and where the answer is)

1. *Why not train an end to end network on geometry maps?* Explicit features are interpretable and cannot memorise generator traces as easily. A CNN on the saved maps is a natural extension but is not part of this study (it is listed as future work).
2. *Isn't Depth Anything just DINOv2?* Yes; that is why the main claim uses explicit outputs and M4 is the control.
3. *Why logistic regression?* Fair comparison across M1 to M5, interpretable coefficients; the tree model ablation checks the linear assumption.
4. *How do you know there is no leakage?* Leakage report, base detector audit (Table 4 of the paper), deduplication.
5. *Why not GenImage?* Its real images are ImageNet (UnivFD/AIDE family training data) and it has the JPEG bias.
6. *Are the intervals too wide?* Possibly, with few generators; we show per generator results.
7. *Does it work on portraits?* Scene analysis (H5).
8. *What about adversarial attacks?* Out of scope; future work.
9. *Compute cost?* Figure on compute; geometry adds several models per image.
10. *Why should we trust the synthetic validation?* We do not use it as evidence, only as a unit test of the feature code.

## 22. Concepts to study before running the experiments

| Concept | Why | Resource | Where in code |
| --- | --- | --- | --- |
| Pinhole camera, intrinsics, homogeneous coordinates | G1, G3 | Hartley & Zisserman, *Multiple View Geometry*, ch. 6 and 8; Szeliski, *Computer Vision: Algorithms and Applications*, ch. 2 (free online) | `backproject`, `focal_from_orthogonal_vps` |
| Vanishing points and RANSAC | G3 | Szeliski ch. 7; Fischler & Bolles (1981) RANSAC | `fit_vanishing_points` |
| Lambertian shading, spherical harmonics lighting | G2 | Ramamoorthi & Hanrahan (2001) | `sh_basis`, `fit_sh_huber` |
| Robust regression (Huber, IRLS) | G2 | Huber (1964); any robust statistics text | `fit_sh_huber` |
| Logistic regression, regularisation, calibration | Stacker | Bishop, *Pattern Recognition and ML*, ch. 4; Guo et al. (2017) | `stacking.py` |
| ROC, AUROC, TPR at fixed FPR | Metrics | Fawcett (2006) "An introduction to ROC analysis" | `metrics.py` |
| Bootstrap and cluster bootstrap | Statistics | Efron & Tibshirani, *An Introduction to the Bootstrap*; Field & Welsh (2007) | `stats.py` |
| Multiple testing (Holm) | Statistics | Holm (1979) | `stats.holm` |
| Data leakage in ML | Validity | Kapoor & Narayanan (2023) "Leakage and the reproducibility crisis in ML based science" | `splits.py` |
| How the three detectors work | Baselines | Their papers (spreadsheet rows) | `detectors/base.py` |
| Monocular geometry models | Inputs | MoGe, Depth Anything V2, DSINE papers | `geometry/estimators.py` |
