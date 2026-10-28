"""Stage 1: ingest datasets, assign leakage-safe splits, remove near duplicates, write variants.

Example:
    python scripts/prepare_data.py --config configs/pilot.yaml
"""

from _bootstrap import config, parser
from prism3d.pipeline import prepare

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--limit", type=int, default=None, help="max images per split and cluster (quick runs)")
    a = p.parse_args()
    df = prepare(config(a), a.limit)
    print(df[df.split != ""].groupby(["split", "label"]).size())
