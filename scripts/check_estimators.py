"""Colab sanity check: load every frozen geometry model, run it on one image, print shapes/ranges,
compute the feature vector and save a qualitative panel. Run this before extract_geometry.py.

    python scripts/check_estimators.py --image path/to/photo.jpg
"""

from __future__ import annotations

import time

import numpy as np

from _bootstrap import config, parser
from prism3d.data.processing import load_rgb
from prism3d.geometry.estimators import GeometryPipeline
from prism3d.geometry.features import FEATURE_NAMES, compute_features
from prism3d import viz


def describe(name: str, a) -> None:
    if a is None:
        print(f"  {name:12s} None")
        return
    a = np.asarray(a)
    print(f"  {name:12s} shape={a.shape} dtype={a.dtype} min={np.nanmin(a):.4g} max={np.nanmax(a):.4g}")


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="results/check_estimators")
    a = p.parse_args()
    cfg = config(a)
    t0 = time.perf_counter()
    pipe = GeometryPipeline(cfg["geometry"])
    print(f"Loaded models in {time.perf_counter() - t0:.1f}s")
    rgb = load_rgb(a.image)
    t0 = time.perf_counter()
    gi = pipe(rgb)
    print(f"Geometry inference {time.perf_counter() - t0:.2f}s")
    for k in ("rgb", "points", "depth", "K", "mask", "normal_pred", "disp_da", "shading", "albedo"):
        describe(k, getattr(gi, k))
    print("  mask fraction", float(np.mean(gi.mask)))
    feats, maps = compute_features(gi, None, return_maps=True)
    for k in FEATURE_NAMES:
        print(f"  {k:28s} {feats[k]:.4f}")
    viz.qualitative_panel([{"rgb": gi.rgb, "depth": gi.depth, "maps": maps}], f"{a.out}/panel", ["check"])
    print(f"Saved {a.out}/panel.png. Check: predicted normals and depth normals should look alike on a real photo.")
