"""Stage 3: score every processed image with the frozen base detectors.

Run with --check first: it scores 8 images and prints their logits so the sign convention
(higher = AI generated) can be confirmed before the full run.

    python scripts/run_detectors.py --config configs/pilot.yaml --check
    python scripts/run_detectors.py --config configs/pilot.yaml
"""

from _bootstrap import config, parser
from prism3d.pipeline import run_detectors

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--detectors", nargs="*", default=None)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    run_detectors(config(a), a.detectors, a.variants, a.check)
