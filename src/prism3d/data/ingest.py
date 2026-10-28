"""Turn downloaded dataset folders into manifest rows.

Each public dataset ships with its own folder layout. The functions below map the layouts
documented in ``docs/dataset_guide.md`` onto the common manifest schema. If a dataset
release changes its layout, only the matching function needs editing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .manifest import list_images, make_frame, normalise_name


def ingest_generator_folders(
    root: str | Path,
    dataset: str,
    real_dirs: dict[str, str] | None = None,
    pair_by_stem: bool = False,
) -> pd.DataFrame:
    """Layout ``root/<generator>/**/<image>``; folders named in ``real_dirs`` hold real images.

    Used for Synthbuster, the extended Synthbuster release (RAISE, FLUX, SD 3.5) and any
    benchmark that stores one folder per generator.

    Args:
        root: dataset root.
        dataset: dataset key written to the manifest.
        real_dirs: mapping ``folder name -> real_source name`` for folders with real photos.
        pair_by_stem: if True, the file stem becomes ``pair_id`` and ``prompt_id`` so a real
            photo and the fakes generated from its caption share an id (content matching).
    """
    root = Path(root)
    real_dirs = {k.lower(): v for k, v in (real_dirs or {}).items()}
    rows = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        key = sub.name.lower()
        is_real = key in real_dirs
        for img in list_images(sub):
            stem = img.stem
            rel = img.relative_to(root).with_suffix("")
            rows.append(
                {
                    "image_id": f"{dataset}/{rel.as_posix()}",
                    "path": str(img),
                    "label": 0 if is_real else 1,
                    "dataset": dataset,
                    "generator": "" if is_real else normalise_name(sub.name),
                    "real_source": real_dirs[key] if is_real else "",
                    "prompt_id": f"{dataset}/{stem}" if pair_by_stem else f"{dataset}/{rel.as_posix()}",
                    "pair_id": stem if pair_by_stem else "",
                }
            )
    return make_frame(rows)


def ingest_binary_folders(
    root: str | Path,
    dataset: str,
    real_source: str,
    generator: str,
    real_name: str = "0_real",
    fake_name: str = "1_fake",
) -> pd.DataFrame:
    """Layout ``root/[<subset>/]0_real`` and ``.../1_fake`` (CNNSpot convention).

    Chameleon and several benchmarks use this convention. When a ``<subset>`` level exists,
    its name is used as the generator (fakes) or appended to the real source (reals).
    """
    root = Path(root)
    rows = []
    for real_dir in sorted(root.rglob(real_name)):
        subset = real_dir.parent.relative_to(root).as_posix()
        for img in list_images(real_dir):
            rel = img.relative_to(root).with_suffix("").as_posix()
            src = real_source if subset in ("", ".") else f"{real_source}/{subset}"
            rows.append({"image_id": f"{dataset}/{rel}", "path": str(img), "label": 0, "dataset": dataset,
                         "real_source": normalise_name(src), "prompt_id": f"{dataset}/{rel}"})
    for fake_dir in sorted(root.rglob(fake_name)):
        subset = fake_dir.parent.relative_to(root).as_posix()
        gen = generator if subset in ("", ".") else subset
        for img in list_images(fake_dir):
            rel = img.relative_to(root).with_suffix("").as_posix()
            rows.append({"image_id": f"{dataset}/{rel}", "path": str(img), "label": 1, "dataset": dataset,
                         "generator": normalise_name(gen), "prompt_id": f"{dataset}/{rel}"})
    return make_frame(rows)


def ingest_cospy(root: str | Path, dataset: str = "cospy") -> pd.DataFrame:
    """Co-Spy-Bench synthetic images: ``root/synthetic/<caption_source>/<org@model>/**/<image>``.

    The caption pool (mscoco, flickr, cc3m, textcaps, sbu) is stored in ``caption_source``;
    an image's caption id is its file stem, so all generators that used the same caption share
    a ``prompt_id`` and are kept on one side of every split.
    """
    syn = Path(root) / "synthetic"
    rows = []
    if not syn.exists():
        raise FileNotFoundError(f"Expected Co-Spy-Bench synthetic folder at {syn}")
    for cap_dir in sorted(p for p in syn.iterdir() if p.is_dir()):
        for gen_dir in sorted(p for p in cap_dir.iterdir() if p.is_dir()):
            gen = normalise_name(gen_dir.name)
            for img in list_images(gen_dir):
                rel = img.relative_to(syn).with_suffix("").as_posix()
                rows.append(
                    {
                        "image_id": f"{dataset}/{rel}",
                        "path": str(img),
                        "label": 1,
                        "dataset": dataset,
                        "generator": gen,
                        "caption_source": cap_dir.name.lower(),
                        "prompt_id": f"{dataset}/{cap_dir.name.lower()}/{img.stem}",
                    }
                )
    return make_frame(rows)


def ingest_real_folder(root: str | Path, dataset: str, real_source: str) -> pd.DataFrame:
    """Any folder tree that contains only real photographs (e.g. Flickr30k images)."""
    root = Path(root)
    rows = []
    for img in list_images(root):
        rel = img.relative_to(root).with_suffix("").as_posix()
        rows.append({"image_id": f"{dataset}/{rel}", "path": str(img), "label": 0, "dataset": dataset,
                     "real_source": normalise_name(real_source), "prompt_id": f"{dataset}/{rel}"})
    return make_frame(rows)


def ingest_openfake(root: str | Path, dataset: str = "openfake") -> pd.DataFrame:
    """OpenFake test split exported by ``scripts/download_data.py openfake``.

    The exporter writes ``root/images/<row_id>.<ext>`` and ``root/metadata.csv`` with the
    dataset's own columns (row_id, label, model, type, release_date, prompt).
    """
    root = Path(root)
    meta = pd.read_csv(root / "metadata.csv")
    rows = []
    for r in meta.itertuples(index=False):
        is_real = str(r.label).lower() in ("real", "0")
        rows.append(
            {
                "image_id": f"{dataset}/{r.row_id}",
                "path": str(root / "images" / r.filename),
                "label": 0 if is_real else 1,
                "dataset": dataset,
                "generator": "" if is_real else normalise_name(str(r.model)),
                "real_source": normalise_name(str(r.model)) if is_real else "",
                "prompt_id": f"{dataset}/{r.row_id}",
                "release_date": "" if pd.isna(getattr(r, "release_date", None)) else str(r.release_date),
            }
        )
    return make_frame(rows)


INGESTERS = {
    "generator_folders": ingest_generator_folders,
    "binary_folders": ingest_binary_folders,
    "cospy": ingest_cospy,
    "real_folder": ingest_real_folder,
    "openfake": ingest_openfake,
}


def ingest_from_config(sources: list[dict]) -> pd.DataFrame:
    """Ingest every dataset listed under ``data.sources`` in the config and concatenate."""
    frames = []
    for spec in sources:
        spec = dict(spec)
        kind = spec.pop("kind")
        if not Path(spec["root"]).exists():
            raise FileNotFoundError(f"Dataset root not found for {spec.get('dataset')}: {spec['root']}")
        frames.append(INGESTERS[kind](**spec))
    df = pd.concat(frames, ignore_index=True)
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image ids across datasets; give each source a unique dataset key")
    return df
