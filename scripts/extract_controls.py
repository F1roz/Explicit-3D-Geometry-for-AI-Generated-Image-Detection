"""Stage 4: DINOv2 control features (per variant) and CLIP scene labels (clean images).

    python scripts/extract_controls.py --config configs/pilot.yaml
"""

from _bootstrap import config, parser
from prism3d.pipeline import extract_controls

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--mock", action="store_true", help="synthetic smoke test only")
    a = p.parse_args()
    extract_controls(config(a), a.mock)
