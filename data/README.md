# Data folder

Nothing in this folder is committed. Run `python scripts/download_data.py all --dry-run` to see
what will be downloaded, then without `--dry-run`. Expected layout after download:

```
data/raw/
  synthbuster/<generator>/*.png              Zenodo 10066460 (CC BY-NC-SA 4.0)
  synthbuster_ext/<raise|flux|sd3.5...>/      GRIP extended Synthbuster (RAISE photos + FLUX + SD 3.5)
  cospy/synthetic/<caption source>/<org@model>/...   Hugging Face ruojiruoli/Co-Spy-Bench (MIT)
  flickr30k/...                              Hugging Face nlphuji/flickr30k (Flickr terms, research use)
  openfake/images/*.png + metadata.csv       Hugging Face ComplexDataLab/OpenFake, test split only (CC BY-NC)
  chameleon/**/0_real, **/1_fake             on request from the AIDE authors (academic use only)
data/synthetic/                              made by scripts/make_synthetic_data.py (smoke tests)
```

Full details, licences and leakage notes: `docs/dataset_guide.md`.
