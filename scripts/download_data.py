"""Download datasets, official detector code and weights into the layout expected by configs/.

Usage (Colab):
    python scripts/download_data.py all --dry-run          # print every action first
    python scripts/download_data.py synthbuster synthbuster_ext cospy flickr30k openfake
    python scripts/download_data.py detectors

Sources and licences are listed in docs/dataset_guide.md. Chameleon and AIGIBench need a manual
step (access request / manual download); the script prints the instructions.

Nothing here could be executed in the authoring environment (its network allowlist blocks
Zenodo and Hugging Face). URLs, repository ids and file patterns follow the official pages
checked on 11 Oct 2026; if a host changes its layout, the printed error names the file.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TP = ROOT / "third_party"
DRY = False

SYNTHBUSTER_URL = "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"
SYNTHBUSTER_MD5 = "0695bd328e16ea21c5c9cc2ae1d994ff"
EXT_SYNTHBUSTER_INDEX = "https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/"
BFREE_WEIGHTS = "https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip"
BFREE_WEIGHTS_MD5 = "f3f53fa647848b16cf81c913f148a198"

# Co-Spy-Bench: <caption source>/<org@model>/images_with_metadata.tar.gz on HF ruojiruoli/Co-Spy-Bench
COSPY_MODELS = [
    "CompVis@ldm-text2im-large-256", "CompVis@stable-diffusion-v1-4", "runwayml@stable-diffusion-v1-5",
    "segmind@SSD-1B", "segmind@tiny-sd", "segmind@SegMoE-SD-4x2-v0", "segmind@small-sd",
    "stabilityai@stable-diffusion-2-1", "stabilityai@stable-diffusion-3-medium-diffusers", "stabilityai@sdxl-turbo",
    "stabilityai@stable-diffusion-2", "stabilityai@stable-diffusion-xl-base-1.0",
    "playgroundai@playground-v2.5-1024px-aesthetic", "playgroundai@playground-v2-1024px-aesthetic",
    "playgroundai@playground-v2-512px-base", "playgroundai@playground-v2-256px-base",
    "PixArt-alpha@PixArt-XL-2-1024-MS", "PixArt-alpha@PixArt-XL-2-512x512",
    "latent-consistency@lcm-lora-sdxl", "latent-consistency@lcm-lora-sdv1-5",
    "black-forest-labs@FLUX.1-schnell", "black-forest-labs@FLUX.1-dev",
]
# only what the default config uses: cc3m for the fit split, textcaps + sbu for Test A
COSPY_NEEDED = {
    "cc3m": ["segmind@SSD-1B", "segmind@tiny-sd", "segmind@small-sd", "segmind@SegMoE-SD-4x2-v0",
             "latent-consistency@lcm-lora-sdxl", "latent-consistency@lcm-lora-sdv1-5", "stabilityai@sdxl-turbo",
             "CompVis@ldm-text2im-large-256", "runwayml@stable-diffusion-v1-5", "stabilityai@stable-diffusion-2-1"],
    "textcaps": ["black-forest-labs@FLUX.1-dev", "black-forest-labs@FLUX.1-schnell",
                 "stabilityai@stable-diffusion-3-medium-diffusers", "playgroundai@playground-v2.5-1024px-aesthetic",
                 "PixArt-alpha@PixArt-XL-2-1024-MS", "PixArt-alpha@PixArt-XL-2-512x512"],
}
COSPY_NEEDED["sbu"] = COSPY_NEEDED["textcaps"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd), f"(cwd={cwd})" if cwd else "")
    if not DRY:
        subprocess.run(cmd, cwd=cwd, check=True)


def fetch(url: str, dest: Path, md5: str | None = None) -> Path:
    print(f"GET {url} -> {dest}")
    if DRY:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    if md5:
        h = hashlib.md5()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != md5:
            raise IOError(f"MD5 mismatch for {dest}: {h.hexdigest()} != {md5}")
        print("  md5 ok")
    return dest


def extract(archive: Path, out: Path) -> None:
    print(f"extract {archive} -> {out}")
    if DRY:
        return
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")


# ----------------------------------------------------------------------------- datasets


def synthbuster() -> None:
    z = fetch(SYNTHBUSTER_URL, RAW / "_archives" / "synthbuster.zip", SYNTHBUSTER_MD5)
    extract(z, RAW / "_tmp_synthbuster")
    if not DRY:
        # the archive holds one folder per generator; flatten a single top-level folder if present
        tops = [p for p in (RAW / "_tmp_synthbuster").iterdir() if p.is_dir()]
        src = tops[0] if len(tops) == 1 and not any(tops[0].glob("*.png")) else RAW / "_tmp_synthbuster"
        shutil.move(str(src), str(RAW / "synthbuster"))
    print("Synthbuster ready: data/raw/synthbuster/<generator>/*.png (real RAISE photos come from synthbuster_ext)")


def synthbuster_ext() -> None:
    print(f"Reading index {EXT_SYNTHBUSTER_INDEX}")
    if DRY:
        return
    html = urllib.request.urlopen(EXT_SYNTHBUSTER_INDEX).read().decode("utf8", "ignore")
    links = sorted(set(re.findall(r'href="([^"?]+\.(?:zip|tar|tar\.gz|tgz))"', html)))
    if not links:
        raise RuntimeError("No archives found on the extended Synthbuster page; download manually (docs/dataset_guide.md)")
    for ln in links:
        a = fetch(EXT_SYNTHBUSTER_INDEX + ln, RAW / "_archives" / "synthbuster_ext" / Path(ln).name)
        extract(a, RAW / "synthbuster_ext")
    print("Check that data/raw/synthbuster_ext holds one folder per source (e.g. raise/, flux/, sd3.5/).")
    print("If the real-photo folder has another name, add it to data.sources[*].real_dirs in the config.")


def cospy(all_sources: bool = False) -> None:
    if not DRY:
        from huggingface_hub import hf_hub_download
    plan = {s: COSPY_MODELS for s in ("mscoco", "flickr", "cc3m", "textcaps", "sbu")} if all_sources else COSPY_NEEDED
    for src, models in plan.items():
        for m in models:
            fn = f"{src}/{m}/images_with_metadata.tar.gz"
            print(f"HF ruojiruoli/Co-Spy-Bench :: {fn}")
            if DRY:
                continue
            p = hf_hub_download("ruojiruoli/Co-Spy-Bench", fn, repo_type="dataset", local_dir=RAW / "_archives" / "cospy")
            extract(Path(p), RAW / "cospy" / "synthetic" / src)
            Path(p).unlink()  # save disk space, as the official script does
    print("Co-Spy-Bench synthetic images ready under data/raw/cospy/synthetic/<caption source>/<org@model>/")


def flickr30k() -> None:
    print("HF nlphuji/flickr30k :: *.zip")
    if DRY:
        return
    from huggingface_hub import snapshot_download

    d = snapshot_download("nlphuji/flickr30k", repo_type="dataset", allow_patterns=["*.zip"], local_dir=RAW / "_archives" / "flickr30k")
    for z in Path(d).rglob("*.zip"):
        extract(z, RAW / "flickr30k")
    print("Flickr30k real photos ready under data/raw/flickr30k/")


def openfake(max_per_model: int = 600, split: str = "test") -> None:
    """Stream the OpenFake test split and write images + metadata.csv (never the train split)."""
    print(f"HF ComplexDataLab/OpenFake (config 'core', split '{split}'), up to {max_per_model} images per model")
    if DRY:
        return
    import pandas as pd
    from datasets import load_dataset

    ds = load_dataset("ComplexDataLab/OpenFake", "core", split=split, streaming=True)
    label_names = None
    try:
        label_names = ds.features["label"].names
    except Exception:
        pass
    out = RAW / "openfake"
    (out / "images").mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    rows = []
    for ex in ds:
        model = str(ex.get("model"))
        if counts.get(model, 0) >= max_per_model:
            continue
        lab = ex["label"]
        lab = label_names[lab] if label_names and isinstance(lab, int) else str(lab)
        img = ex["image"].convert("RGB")
        fn = f"{ex['row_id']}.png"
        img.save(out / "images" / fn)
        rows.append({"row_id": ex["row_id"], "filename": fn, "label": lab, "model": model, "type": ex.get("type"),
                     "release_date": ex.get("release_date")})
        counts[model] = counts.get(model, 0) + 1
        if len(rows) % 1000 == 0:
            print(f"  {len(rows)} images")
            pd.DataFrame(rows).to_csv(out / "metadata.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metadata.csv", index=False)
    print(f"OpenFake {split}: {len(rows)} images from {len(counts)} models -> data/raw/openfake/")


def chameleon() -> None:
    print("Chameleon (AIDE, ICLR 2025) is shared on request for academic use.")
    print("  1. Email the AIDE authors as described at https://github.com/shilinyan99/AIDE (Chameleon section).")
    print("  2. Unpack to data/raw/chameleon/ so that it contains 0_real/ and 1_fake/ folders (any depth).")
    print("  If access is not granted, use AIGIBench: see docs/dataset_guide.md and add it as a binary_folders source.")


# ----------------------------------------------------------------------------- detectors


def detectors() -> None:
    TP.mkdir(exist_ok=True)
    if not (TP / "UniversalFakeDetect").exists():
        run(["git", "clone", "--depth", "1", "https://github.com/WisconsinAIVision/UniversalFakeDetect", str(TP / "UniversalFakeDetect")])
    if not (TP / "B-Free").exists():
        run(["git", "clone", "--depth", "1", "https://github.com/grip-unina/B-Free", str(TP / "B-Free")])
    z = fetch(BFREE_WEIGHTS, RAW / "_archives" / "BFREE_dino2reg4.zip", BFREE_WEIGHTS_MD5)
    extract(z, TP / "B-Free" / "code" / "weights")
    if not (TP / "Co-Spy").exists():
        run(["git", "clone", "--depth", "1", "https://github.com/Megum1/Co-Spy", str(TP / "Co-Spy")])
    run(["sh", "download.sh"], cwd=TP / "Co-Spy" / "pretrained")
    print("Install each repo's extra requirements if run_detectors.py --check reports a missing package:")
    print("  pip install -r third_party/B-Free/code/requirements.txt")


STEPS = {"synthbuster": synthbuster, "synthbuster_ext": synthbuster_ext, "cospy": cospy, "flickr30k": flickr30k,
         "openfake": openfake, "chameleon": chameleon, "detectors": detectors}


def main() -> None:
    global DRY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--openfake-max-per-model", type=int, default=600)
    a = ap.parse_args()
    DRY = a.dry_run
    steps = list(STEPS) if "all" in a.steps else a.steps
    for s in steps:
        print(f"\n=== {s} ===")
        if s == "openfake":
            openfake(a.openfake_max_per_model)
        else:
            STEPS[s]()


if __name__ == "__main__":
    sys.exit(main())
