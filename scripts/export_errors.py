"""List the most confident errors of a model for error analysis and the qualitative figure.

    python scripts/export_errors.py --config configs/default.yaml --detector bfree --model M3 --n 20
Writes <run_dir>/errors_<detector>_<model>.txt (image ids: n false positives, then n false
negatives, then 2 correct reals and 2 correct fakes from Test B for the qualitative figure) and a
CSV with scores, generators and scene labels. Feed the .txt to extract_geometry.py --ids.
"""

import pandas as pd

from _bootstrap import config, parser
from prism3d.pipeline import run_dir

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--detector", required=True)
    p.add_argument("--model", default="M3")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--variant", default="clean")
    a = p.parse_args()
    cfg = config(a)
    rd = run_dir(cfg)
    pr = pd.read_csv(rd / "predictions" / "all_main.csv.gz", keep_default_na=False)
    q = pr[(pr.detector == a.detector) & (pr.model == a.model) & (pr.variant == a.variant)].copy()
    q["margin"] = q.logit - q.threshold
    fp = q[(q.label == 0) & (q.margin >= 0)].sort_values("margin", ascending=False).head(a.n).assign(kind="false_positive")
    fn = q[(q.label == 1) & (q.margin < 0)].sort_values("margin").head(a.n).assign(kind="false_negative")
    b = q[q.eval_sets.str.contains(cfg["evaluation"]["content_matched_split"])]
    ok_r = b[(b.label == 0) & (b.margin < 0)].sort_values("margin").head(2).assign(kind="correct_real")
    ok_f = b[(b.label == 1) & (b.margin >= 0)].sort_values("margin", ascending=False).head(2).assign(kind="correct_fake")
    out = pd.concat([fp, fn, ok_r, ok_f])
    out[["image_id", "kind", "label", "generator", "scene", "eval_sets", "logit", "threshold"]].to_csv(rd / f"errors_{a.detector}_{a.model}.csv", index=False)
    (rd / f"errors_{a.detector}_{a.model}.txt").write_text("\n".join(out.image_id) + "\n")
    print(out.groupby("kind").size())
    print("Scene counts among errors:\n", out[out.kind.str.startswith("false")].groupby(["kind", "scene"]).size())
