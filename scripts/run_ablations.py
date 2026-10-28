"""Stage 6: one-change-at-a-time ablations of M3 (needs run_experiments.py first).

    python scripts/run_ablations.py --config configs/default.yaml
"""

from _bootstrap import config, parser
from prism3d.pipeline import run_ablations

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    print(run_ablations(config(a)).to_string(index=False))
