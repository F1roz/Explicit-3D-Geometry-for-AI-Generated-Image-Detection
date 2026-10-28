"""Stage 8: every results figure from saved result files (PDF for LaTeX, PNG for Word).

    python scripts/generate_figures.py --config configs/default.yaml
Copy <run_dir>/figures/*.pdf and *.png into paper/figures/ to replace the [xyz] placeholders.
"""

from _bootstrap import config, parser
from prism3d.pipeline import make_figures

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    for p in make_figures(config(a)):
        print(p)
