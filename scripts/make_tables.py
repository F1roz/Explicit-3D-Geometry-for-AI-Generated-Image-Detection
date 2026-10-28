"""Stage 7: LaTeX rows and a Markdown summary from the result files.

The files in <run_dir>/tables/ map one-to-one onto the [xyz] cells of paper/main.tex.
"""

from _bootstrap import config, parser
from prism3d.pipeline import make_tables

if __name__ == "__main__":
    a = parser(__doc__).parse_args()
    out = make_tables(config(a))
    print((out / "results_summary.md").read_text())
