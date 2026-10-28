# Limitations and research risks

## Methodological limitations

| Limitation | Effect | What we do about it |
| --- | --- | --- |
| Geometry models were trained partly on rendered images and may fail differently on generated images for reasons unrelated to physics | A gain could reflect estimator quirks, not scene physics | Report as a confound; estimator sensitivity ablation (Marigold normals, Retinex shading) |
| Real scenes violate the simple physical models (mixed lighting, interreflections, specular materials, wide angle distortion) | Geometry features have false alarms on real photos | Features are learned inputs, never rules; false positive rates reported per scene |
| Content matching is approximate (images from captions, not the same scene) | H3 cannot fully exclude content effects | Stated in the paper; content matched set used as a control, not proof |
| Few unseen generators per test set | Wide cluster bootstrap intervals | Per generator results; intervals labelled indicative below 5 generators |
| Base detectors used with released weights only | Results apply to these checkpoints | Three families covered; leakage audit of their training data |
| Scope: fully generated still images | No claim about partial edits, video, or adversarial attacks | Stated as out of scope; listed as future work |
| Scene labels from zero shot CLIP | Scene analysis inherits labelling errors | 300 image human audit with Cohen's kappa |
| Missing value flags could leak the class | Spurious gain | Flag rates per class reported; ablation without flags |

## Threats to validity

* **Internal:** leakage between fit and test (controlled by code that refuses to run), tuning on test data (forbidden by protocol; log every run), shortcut features (JPEG harmonisation, flags ablation).
* **Construct:** "geometric inconsistency" is measured through imperfect estimators; the features are proxies.
* **External:** generators change quickly; results describe generators available in 2025 and 2026.
* **Statistical:** multiple comparisons (Holm), unit of analysis (generator level bootstrap), practical significance threshold.

## Project risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Chameleon access not granted | Medium | AIGIBench social subsets as Test D |
| Extended Synthbuster folder names differ from the config | Medium | Inspect after download; edit `real_dirs` |
| DSINE torch hub entry point breaks (its README marks it deprecated) | Medium | `geometry.normal_estimator=marigold` or `moge` (less independent; state it) |
| Colab session limits interrupt long runs | High | Every stage resumes; results on Drive |
| Dead OpenFake links | Low (images embedded in parquet on HF) | Report counts actually retrieved |
| GPU memory with large images | Medium | Lower `geometry.work_long_side`; `data.max_side` |
| A third party API changed since 11 Oct 2026 | Medium | `check_estimators.py` and `run_detectors.py --check` catch it before long runs |
| Negative result | Real possibility | Preregistered, publishable as a careful negative result with controls |
| Close prior work appears before submission | Medium | Repeat the novelty search (SOP step 3) |

## What was not verified in the authoring environment

The authoring environment had no GPU and its network blocked Hugging Face, Zenodo and PyPI. Therefore:

* The deep model wrappers (`geometry/estimators.py`, `detectors/base.py`, `controls.py`, `scripts/zero_shot_mllm.py`) follow the official READMEs and source but were **not executed**. Run `scripts/check_estimators.py` and `scripts/run_detectors.py --check` first.
* `scripts/download_data.py` was **not executed**.
* The Word file's equations are native Word equations (83 found in the file) but could not be rendered by the sandbox's LibreOffice; check them once in Word.

Everything else (feature computation, splits, deduplication, processing, stacker, metrics, statistics, tables, figures, the end to end pipeline on synthetic data) was executed and tested.
