"""Render the synthetic smoke-test dataset (ground-truth geometry saved as .npz sidecars).

Layout written to data/synthetic/: real_a, real_b, real_c (consistent scenes) and
gen_a ... gen_i, gen_e_test (inconsistent scenes), N images each.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from _bootstrap import ROOT  # noqa: F401
from prism3d.synthetic import render_scene

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--n", type=int, default=12, help="images per folder")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    folders = {f"real_{k}": True for k in "abc"} | {f"gen_{k}": False for k in "abcdefghi"} | {"gen_e_test": False}
    from pathlib import Path

    for name, consistent in folders.items():
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.n):
            s = render_scene(rng, consistent)
            Image.fromarray(s["rgb"]).save(d / f"{name}_{i:03d}.png")
            np.savez_compressed(d / f"{name}_{i:03d}.npz", **{k: v for k, v in s.items() if k != "rgb"})
    print(f"Wrote {len(folders) * a.n} images to {a.out}")


if __name__ == "__main__":
    main()
