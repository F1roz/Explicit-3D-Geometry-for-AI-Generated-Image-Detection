"""Export a stratified sample of scene labels for the human audit (Cohen's kappa).

    python scripts/export_scene_audit.py --config configs/default.yaml --n 300
Fill the empty ``human_scene`` column, then:
    python scripts/export_scene_audit.py --config configs/default.yaml --score results/full/scene_audit.csv
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from _bootstrap import config, parser
from prism3d.pipeline import processed_path, run_dir

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--score", default=None, help="filled audit CSV to score")
    a = p.parse_args()
    cfg = config(a)
    if a.score:
        t = pd.read_csv(a.score).dropna(subset=["human_scene"])
        k = float(cohen_kappa_score(t.scene, t.human_scene))
        print(f"n={len(t)}  Cohen's kappa = {k:.3f}")
        (run_dir(cfg) / "scene_audit_kappa.json").write_text(json.dumps({"n": len(t), "kappa": k}, indent=2))
    else:
        sc = pd.read_csv(run_dir(cfg) / "scenes.csv")
        per = max(1, a.n // sc.scene.nunique())
        s = sc.groupby("scene", group_keys=False).apply(lambda g: g.sample(min(len(g), per), random_state=0))
        s["image_path"] = [str(processed_path(cfg, i, "clean")) for i in s.image_id]
        s["human_scene"] = ""
        out = run_dir(cfg) / "scene_audit.csv"
        s.to_csv(out, index=False)
        print(f"Wrote {len(s)} rows to {out}")
