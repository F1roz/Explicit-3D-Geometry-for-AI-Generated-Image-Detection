"""The manifest is one CSV row per image and is the single source of truth for every later step.

Columns
-------
image_id     unique string id (dataset prefix + relative path stem)
path         absolute or root-relative path of the ORIGINAL image
label        0 = real photograph, 1 = AI generated
dataset      short dataset key, e.g. ``synthbuster``, ``cospy``, ``openfake``
generator    generator name for fakes (normalised, lower case); empty for reals
real_source  real-image source for reals (e.g. ``raise``, ``flickr30k``, ``laion``); empty for fakes
caption_source  for caption-conditioned fakes, which caption pool was used (Co-Spy-Bench)
cluster      unit of resampling for statistics: generator for fakes, real_source for reals
prompt_id    images sharing a prompt or caption share this id (kept on one side of every split)
pair_id      content-matched pairing key (e.g. RAISE image id for Synthbuster), may be empty
release_date optional ISO date of the generator release (OpenFake), may be empty
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

COLUMNS = [
    "image_id",
    "path",
    "label",
    "dataset",
    "generator",
    "real_source",
    "caption_source",
    "cluster",
    "prompt_id",
    "pair_id",
    "release_date",
]


def normalise_name(name: str) -> str:
    """Lower-case and normalise a generator or source name so lists in configs match reliably."""
    name = name.strip().lower().replace("@", "/").replace("\\", "/")
    name = re.sub(r"\s+", "-", name)
    return name


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def make_frame(rows: Iterable[dict]) -> pd.DataFrame:
    """Build a manifest frame with all columns present and sane dtypes."""
    df = pd.DataFrame(list(rows))
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLUMNS].copy()
    df["label"] = df["label"].astype(int)
    for c in COLUMNS:
        if c != "label":
            df[c] = df[c].fillna("").astype(str)
    df["cluster"] = [
        (g if lab == 1 else s) or ("unknown_fake" if lab == 1 else "unknown_real")
        for g, s, lab in zip(df["generator"], df["real_source"], df["label"])
    ]
    if df["image_id"].duplicated().any():
        dups = df.loc[df["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ValueError(f"Duplicate image_id values, e.g. {dups}")
    return df


def validate_manifest(df: pd.DataFrame) -> None:
    """Raise if the manifest violates basic invariants."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns {missing}")
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("label must be 0 or 1")
    fakes_without_gen = df[(df.label == 1) & (df.generator == "")]
    if len(fakes_without_gen):
        raise ValueError(f"{len(fakes_without_gen)} fake images have no generator")
    reals_without_src = df[(df.label == 0) & (df.real_source == "")]
    if len(reals_without_src):
        raise ValueError(f"{len(reals_without_src)} real images have no real_source")
