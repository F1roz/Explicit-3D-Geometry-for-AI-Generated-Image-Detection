# Experiment log

Record every run here: date, config file, git commit or zip date, what changed, outcome, and the result folder. Never delete an entry, including failed or discarded runs.

| Date | Run | Config | Purpose | Outcome | Evidence |
| --- | --- | --- | --- | --- | --- |
| 2026-10-11 | unit tests | n/a | Verify feature math, metrics, statistics, stacker, splits, processing | 40 passed, 1 slow test skipped | `python -m unittest discover -s tests -t .` |
| 2026-10-11 | smoke | `configs/smoke.yaml` | End to end wiring on 130 synthetic renders (oracle geometry, mock detector, mock controls) | SMOKE TEST PASSED in about 2 minutes on 2 CPU cores; all expected files produced | `results/smoke/` (synthetic; not evidence about real images) |
| | pilot | `configs/pilot.yaml` | Preregistered go / no-go | **pending (Colab)** | `results/pilot/hypotheses.json` |
| | full | `configs/default.yaml` | Main study | **pending** | `results/full/` |

## Implementation issues found and fixed during development (11 Oct 2026)

These were found by checking the features on synthetic scenes with known geometry, before any real data were used.

1. **Synthetic light direction.** The first renderer lit scenes from below, so consistent scenes had flat shading and G2 was undefined. Fixed in `synthetic.py`.
2. **Huber scale collapse.** On images with large flat regions the robust scale of the residuals was almost zero, so the Huber weights ignored every pixel that was not fitted exactly and the shading fit failed (R² far below zero). Fixed by flooring the scale at 5% of the shading spread (`fit_sh_huber`). After the fix, consistent scenes give R² about 0.99 and inconsistent scenes about 0.3.
3. **Spurious focal estimates.** Vanishing points near infinity produced focal lengths an order of magnitude off. Fixed by using only well supported vanishing points (at least 10% of segments) within three image diagonals of the principal point, and a 10° orthogonality tolerance. The focal formula itself was verified exactly on a synthetic rotated camera (`test_focal_from_orthogonal_vps`).
4. **Normal convention ties.** Using the median angle to choose the axis convention tied when planes facing the camera dominated. Switched to the mean angle (`estimate_convention`).
5. **Noisy second depth in tests.** White noise on the synthetic Depth Anything input made derived normals meaningless; replaced by smooth noise, which matches how depth model errors look.

## Novelty search log

| Date | Query | Source | Closest result | Notes |
| --- | --- | --- | --- | --- |
| 2026-10-11 | "Depth Anything" features AI-generated image detection | web search | none on single image detection | Depth Anything 3 (2025) is a geometry model, not a detector |
| 2026-10-11 | depth / surface normal / intrinsic decomposition cue for detecting diffusion images | web search | Sarkar et al. CVPR 2024 (projective geometry); Grab-3D 2025 (video) | |
| | (repeat before submission) | arXiv, Google Scholar, CVF | | |

## Decisions changed after the v1 proposal

* AIDE replaced by UnivFD and CO-SPY as base detectors in the code: AIDE has no documented per image scoring interface (its evaluation script reports only aggregate metrics), while B-Free, CO-SPY and UnivFD can be scored image by image with official code. AIDE can be added later as a fourth base.
* Test A real images: Co-Spy-Bench ships only partial real examples, so Test A uses the RAISE photographs shared with Test B (allowed: tests may share images; fit images never enter tests).
