"""Near-duplicate detection with a 64-bit perceptual hash (pHash).

pHash: resize the grey image to 32x32, take the 2D DCT, keep the top-left 8x8 low-frequency
block (excluding nothing), and set each bit to 1 if the coefficient is above the block
median. Two images whose hashes differ in at most ``t`` bits (Hamming distance) are treated
as near-duplicates. We compare every evaluation image against every fit/val image in chunks,
so the cost is O(N_test x N_fit) bit operations, which is fast with NumPy.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# popcount lookup for one byte
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash(img: np.ndarray) -> np.uint64:
    """64-bit DCT perceptual hash of an RGB or grey uint8 image."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(img.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small)[:8, :8]
    bits = (dct > np.median(dct)).flatten()
    value = np.uint64(0)
    for b in bits:
        value = (value << np.uint64(1)) | np.uint64(int(b))
    return value


def phash_file(path: str | Path) -> np.uint64:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read image {path}")
    return phash(img)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between two uint64 hash arrays -> (len(a), len(b)) uint8."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    bytes_view = x.view(np.uint8).reshape(x.shape + (8,))
    return _POP8[bytes_view].sum(axis=-1, dtype=np.uint16).astype(np.uint8)


def find_cross_duplicates(query: np.ndarray, reference: np.ndarray, threshold: int = 8, chunk: int = 2048) -> np.ndarray:
    """Boolean array: is query[i] within ``threshold`` bits of any reference hash?"""
    hit = np.zeros(len(query), dtype=bool)
    if len(query) == 0 or len(reference) == 0:
        return hit
    for s in range(0, len(query), chunk):
        d = hamming_matrix(query[s : s + chunk], reference)
        hit[s : s + chunk] = (d <= threshold).any(axis=1)
    return hit


def drop_fit_duplicates_of_test(df: pd.DataFrame, hashes: pd.Series, threshold: int = 8) -> tuple[pd.DataFrame, int]:
    """Remove fit/val images that nearly duplicate any test image (tests are never altered).

    Args:
        df: manifest with ``split`` column.
        hashes: pHash per ``image_id`` (uint64), indexed by image_id.
    Returns:
        (filtered manifest, number of fit/val images removed)
    """
    fv = df[df["split"].isin(["fit", "val"])]
    te = df[df["split"] == "test"]
    q = hashes.loc[fv["image_id"]].to_numpy(dtype=np.uint64)
    r = hashes.loc[te["image_id"]].to_numpy(dtype=np.uint64)
    dup = find_cross_duplicates(q, r, threshold)
    drop_ids = set(fv.loc[dup, "image_id"])
    out = df.copy()
    out.loc[out["image_id"].isin(drop_ids), "split"] = ""
    return out, len(drop_ids)
