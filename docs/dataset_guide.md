# Dataset guide

All information below was checked on the official pages on 11 Oct 2026. Sizes are those stated by the providers; anything we could not verify is marked "check on download". No dataset is redistributed with this repository; `scripts/download_data.py` fetches each one from its official source.

## Summary

| Dataset | Role | Official source | Licence / terms | Size (stated) |
| --- | --- | --- | --- | --- |
| Synthbuster | Test B (fakes) | https://zenodo.org/records/10066460 | CC BY-NC-SA 4.0 | 9 generators × 1,000 images, one zip of 12.4 GB (MD5 0695bd328e16ea21c5c9cc2ae1d994ff) |
| Extended Synthbuster (GRIP) | Tests A and B (RAISE reals), Test B (FLUX, SD 3.5 fakes) | https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/ | check on download (no licence stated on the B-Free page) | 1,000 RAISE photos resized to about 1024×1024 area; 1,000 FLUX; 1,000 SD 3.5 |
| Co-Spy-Bench | Fit pool (fakes), Test A (fakes) | https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench | MIT (dataset card) | 22 generators × 5 caption sources; about 1,024×1,024 PNG |
| Flickr30k | Fit pool (reals) | https://huggingface.co/datasets/nlphuji/flickr30k | Flickr terms of use; research use | about 31,000 photos (check on download) |
| OpenFake (test split only) | Test C | https://huggingface.co/datasets/ComplexDataLab/OpenFake (paper: arXiv 2509.09495) | CC BY-NC 4.0 on the HF card (the arXiv page lists CC BY-NC-SA 4.0) | core test split 91.4k rows; we sample up to 600 per model |
| Chameleon | Test D | via https://github.com/shilinyan99/AIDE (email request) | academic research only; commercial use prohibited | check on download |
| AIGIBench (fallback for D) | Test D | GitHub repository linked from https://arxiv.org/abs/2505.12335 | check on download | 23 fake subsets |

Model weights: B-Free (https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip, MD5 f3f53fa647848b16cf81c913f148a198; code "for informational and nonprofit purposes"), CO-SPY (`pretrained/download.sh` in https://github.com/Megum1/Co-Spy, MIT), UnivFD (`pretrained_weights/fc_weights.pth` in https://github.com/WisconsinAIVision/UniversalFakeDetect, MIT), MoGe (MIT), Depth Anything V2 Large (check the model card licence), DSINE (check repository licence), Intrinsic (academic use only, patent pending), DINOv2 and CLIP (model cards).

## Per dataset details

### Synthbuster (Bammey, IEEE OJSP 2023)

* **Content:** for each of 1,000 RAISE-1k photographs, a caption was written and given to 9 generators: DALL·E 2, DALL·E 3, Adobe Firefly, Midjourney v5, Stable Diffusion 1.3, 1.4, 2 and XL, and GLIDE.
* **Real photos are not included.** Synthbuster recommends using the RAISE originals; we use the RAISE photos distributed with the extended Synthbuster release (already resized), so both sides come from one place.
* **Layout after download:** `data/raw/synthbuster/<generator>/<raise id>.png`. The file stem is the RAISE id; `pair_by_stem: true` gives real and generated images the same `prompt_id`, which keeps them on the same side of every split.
* **Seen family:** SD 1.3, 1.4, 2 and XL are tagged seen family (base detectors or the fit pool used these families) and excluded from the headline claim.

### Extended Synthbuster (from the B-Free authors)

* **Content:** 1,000 RAISE photos resized to about a 1024×1024 area, and 1,000 FLUX and 1,000 SD 3.5 images generated from the same captions.
* **Layout:** one folder per source. The real folder name is not stated on the page; the config maps `raise`, `real` and `raise_1k` to the real source `raise`. If the folder has another name, add it to `real_dirs` in `configs/default.yaml`.
* **Leakage note:** made by the B-Free authors as a test set; B-Free was not trained on it.

### Co-Spy-Bench (Cheng et al., CVPR 2025)

* **Content:** images from 22 open generators, each conditioned on captions from five sources (MS COCO 2017, CC3M, Flickr, TextCaps, SBU).
* **Files:** `<caption source>/<org@model>/images_with_metadata.tar.gz`. We download only what the config uses: CC3M captions for the 10 fit generators, TextCaps and SBU captions for the 6 Test A generators.
* **Real images:** the official script downloads only partial real examples (`ruojiruoli/Co-Spy-Misc`), so we do not use Co-Spy-Bench reals.
* **Leakage note:** CO-SPY's released `sd-v1_4` checkpoint was trained on DRCT-2M, not on Co-Spy-Bench; but its authors built Co-Spy-Bench, so CO-SPY results on Test A should be read with that in mind.

### Flickr30k

* **Content:** about 31,000 Flickr photographs with captions (we use images only).
* **Why:** real photos from a source that no base detector and no test set uses.
* **Note:** not related to the Co-Spy-Bench "flickr" caption pool in a way that matters, because we do not use Flickr captioned fakes anywhere.

### OpenFake (Livernoche et al., 2025)

* **Content:** about 3 million real images (mostly LAION, filtered for people and news events) and about 1 million images from modern generators: SD 1.5, 2.1, XL, 3.5; FLUX 1.0 dev, 1.1 Pro, schnell; Midjourney v6 and v7; DALL·E 3; Imagen 3 and 4; GPT Image 1; Ideogram 3.0; Grok 2; HiDream I1; Recraft v3; Chroma; many community fine tunes.
* **We use the test split only**, never train or validation. The HF viewer lists fields `image, prompt, label, model, type, release_date, row_id`.
* **Seen family:** SD 1.5, 2.1, XL and SD fine tunes (Dreamshaper, Epic Realism, RealVisXL, Juggernaut, ...) are tagged seen family.
* **Social media set:** the paper describes 1,220 in the wild social media images; if it is published separately, it can be added as another test source.

### Chameleon (Yan et al., ICLR 2025)

* **Content:** AI generated images collected because human annotators judged them real, plus real images.
* **Access:** by email request to the AIDE authors; academic use only.
* **Layout:** we expect `0_real` and `1_fake` folders at any depth (`binary_folders` ingester).
* **Leakage note:** AIDE's authors built Chameleon; AIDE is not one of our default bases.

## Preprocessing (identical for every dataset)

1. Decode to RGB; images with a side above 4,096 pixels are downscaled with bicubic interpolation (logged).
2. Reencode as JPEG quality 95 (`clean`).
3. Robustness copies from `clean`: JPEG 75; half size bicubic (stored as PNG); screenshot chain (0.75 size, then JPEG 80).
4. Near duplicate removal against test sets (pHash, Hamming distance at most 8).

## Directory structure expected by the config

```
data/raw/
  synthbuster/<generator>/*.png
  synthbuster_ext/<raise|flux|sd3.5|...>/*
  cospy/synthetic/<cc3m|textcaps|sbu>/<org@model>/**/*.png
  flickr30k/**/*.jpg
  openfake/images/<row_id>.png + openfake/metadata.csv
  chameleon/**/0_real/*, chameleon/**/1_fake/*
```

## Storage estimate

Raw downloads used by the default config: Synthbuster 12.4 GB; extended Synthbuster a few GB; the Co-Spy-Bench subsets (16 tarballs of about 1,000 PNGs at 1 megapixel each) roughly 20 to 30 GB; Flickr30k about 4 to 5 GB; OpenFake sample (600 per model) roughly 20 to 40 GB as PNG. Processed copies add roughly 1 to 1.5 MB per image for all four variants. These are planning estimates; check free Drive space before downloading.

## Known quality and leakage issues

* Synthbuster and the extended set share captions with the RAISE reals; that is intended (content matching) and they are all test images.
* OpenFake real images were selected for faces and news topics, so its real side differs in content from its fake side in ways other than generation.
* Generators of the same family may appear under different names (for example "stable-diffusion-xl" and "sdxl"); the seen family patterns in the config use regular expressions to catch both. Check `manifest_all.csv` after the first prepare run.
