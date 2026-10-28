"""Image IO, format harmonisation and the robustness processing chains.

Format harmonisation (Grommelt et al., 2024): every image, real or fake, is decoded and
re-encoded as JPEG quality 95. Without this step a detector can separate PNG fakes from JPEG
reals by compression traces alone.

Robustness variants are deterministic chains applied to the harmonised image:

* ``clean``      JPEG 95 (the harmonised image itself)
* ``jpeg75``     clean -> JPEG 75
* ``resize05``   clean -> bicubic downscale x0.5 (stored losslessly as PNG)
* ``screenshot`` clean -> bicubic downscale x0.75 -> JPEG 80
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 300_000_000  # some OpenFake images are very large

VARIANT_CHAINS: dict[str, list[tuple[str, float]]] = {
    "clean": [],
    "jpeg75": [("jpeg", 75)],
    "resize05": [("resize", 0.5)],
    "screenshot": [("resize", 0.75), ("jpeg", 80)],
}
VARIANT_EXT = {"clean": ".jpg", "jpeg75": ".jpg", "resize05": ".png", "screenshot": ".jpg"}


def load_rgb(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """Read an image as uint8 RGB. Optionally cap the longer side (logged by callers)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8).copy()


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """Encode to JPEG in memory at ``quality`` and decode back (4:2:0 chroma, PIL default)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return np.asarray(Image.fromarray(img).resize((nw, nh), Image.BICUBIC), dtype=np.uint8)


def harmonise(img: np.ndarray, quality: int = 95) -> np.ndarray:
    """The common starting point for every image in every split."""
    return jpeg_roundtrip(img, quality)


def apply_chain(img: np.ndarray, chain: list[tuple[str, float]]) -> np.ndarray:
    for op, arg in chain:
        if op == "jpeg":
            img = jpeg_roundtrip(img, int(arg))
        elif op == "resize":
            img = resize(img, float(arg))
        else:
            raise ValueError(f"Unknown processing op {op}")
    return img


def make_variant(clean: np.ndarray, variant: str) -> np.ndarray:
    if variant not in VARIANT_CHAINS:
        raise KeyError(f"Unknown variant {variant}; choose from {list(VARIANT_CHAINS)}")
    return apply_chain(clean, VARIANT_CHAINS[variant])


def write_processed(original: np.ndarray, out_paths: dict[str, Path], clean_quality: int = 95) -> None:
    """Create every requested variant from the ORIGINAL pixels and write it to disk.

    JPEG outputs are produced by a single final encode at the variant's last quality, so the
    stored file is byte-for-byte the last step of the chain (no double compression on disk).
    """
    buf = io.BytesIO()
    Image.fromarray(original).save(buf, format="JPEG", quality=clean_quality)
    clean_bytes = buf.getvalue()
    with Image.open(io.BytesIO(clean_bytes)) as im:
        clean = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    for variant, path in out_paths.items():
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if variant == "clean":
            path.write_bytes(clean_bytes)
            continue
        chain = VARIANT_CHAINS[variant]
        img = apply_chain(clean, chain[:-1]) if chain and chain[-1][0] == "jpeg" else apply_chain(clean, chain)
        if chain and chain[-1][0] == "jpeg":
            Image.fromarray(img).save(path, format="JPEG", quality=int(chain[-1][1]))
        else:
            Image.fromarray(img).save(path, format="PNG")
