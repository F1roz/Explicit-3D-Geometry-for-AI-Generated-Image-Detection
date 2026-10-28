"""Stage 5: fit M1-M5 for every base detector, evaluate, run hypothesis tests (and pilot rule).

    python scripts/run_experiments.py --config configs/pilot.yaml
"""

import json

from _bootstrap import config, parser
from prism3d.pipeline import run_experiments

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    h = run_experiments(config(a))
    print(json.dumps({k: h[k].get("pass") for k in ("H1", "H2", "H3") if k in h}, indent=2))
    if "pilot" in h:
        print("Pilot decision:", {d: v["decision"] for d, v in h["pilot"].items()})
