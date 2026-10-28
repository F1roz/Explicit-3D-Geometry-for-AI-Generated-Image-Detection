"""Stage 2: run the frozen geometry models and compute the explicit features.

Examples:
    python scripts/extract_geometry.py --config configs/pilot.yaml
    # maps for error analysis (ids from a text file, one per line; maps go to <run_dir>/maps/):
    python scripts/extract_geometry.py --config configs/default.yaml --variants clean --feature-dir features_err --ids errors.txt
    # estimator ablation (writes features_marigold/):
    python scripts/extract_geometry.py --config configs/default.yaml --feature-dir features_marigold \
        --set geometry.normal_estimator=marigold
"""

from _bootstrap import config, parser
from prism3d.pipeline import extract_geometry

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--kind", choices=["models", "oracle"], default="models", help="'oracle' only for synthetic smoke data")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--feature-dir", default="features")
    p.add_argument("--ids", default=None, help="text file with image ids to process (maps are saved for them)")
    a = p.parse_args()
    ids = [line.strip() for line in open(a.ids) if line.strip()] if a.ids else None
    extract_geometry(config(a), a.kind, a.variants, a.feature_dir, ids)
