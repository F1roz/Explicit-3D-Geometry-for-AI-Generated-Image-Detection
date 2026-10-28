"""Fill the yellow [xyz] placeholders of the paper from the result files.

Every placeholder in paper/main.tex and paper/supplementary.tex carries a key, for example
``\\xyz[h1.bfree.delta]``. This script computes the value of every key it can from the
result files of the full run (and the pilot run), then

1. writes ``<run_dir>/tables/paper_values.md`` and ``paper_values.csv``: one row per key with
   its value, the file it came from, and where it appears in the paper;
2. writes ``paper/main_filled.tex`` and ``paper/supplementary_filled.tex`` with the values in place
   (keys it cannot fill stay yellow), or, with ``--inplace``, overwrites main.tex and
   supplementary.tex after saving a ``.bak`` copy.

Keys starting with ``manual.`` are text only you can write (qualitative description,
conclusion); they always stay yellow. Usage:

    python scripts/fill_results.py --config configs/default.yaml --pilot-config configs/pilot.yaml
    python scripts/fill_results.py --config configs/default.yaml --inplace
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ROOT, config, parser
from prism3d.config import load_config
from prism3d.metrics import all_metrics, auroc, macro_over_generators

PAPER = ROOT / "paper"
KEY_RE = re.compile(r"\\xyz\[([^\]]+)\](\{\})?")
DET_NAMES = {"bfree": "B-Free", "cospy": "CO-SPY", "univfd": "UnivFD"}
VARIANTS = ["jpeg75", "resize05", "screenshot"]

MANUAL_HELP = {
    "manual.qualitative": "Describe what the geometry maps show in fig_qualitative for correct, false positive and false negative images (use results/full/errors_*.txt and the saved maps).",
    "manual.common_fp": "Most common kind of real photo wrongly flagged by M3 (look at errors_<detector>_M3.txt, false positives).",
    "manual.common_fn": "Most common kind of generated image missed by M3 (same file, false negatives).",
    "manual.conclusion": "One phrase on what was found, then two sentences on which hypotheses passed.",
    "manual.openfake_retrieved": "Number of OpenFake real images actually obtained (download log). Delete the sentence if all were retrieved.",
    "manual.openfake_requested": "Number of OpenFake real images requested.",
}


# ----------------------------------------------------------------------------- formatting

def ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and not np.isfinite(x))


def num(x, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def signed(x, nd: int = 3) -> str:
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def ci(lo, hi, nd: int = 3) -> str:
    return f"[{signed(lo, nd)}, {signed(hi, nd)}]"


def pval_boot(p, n_boot) -> str:
    if not ok(p):
        return None
    return f"$<${1 / n_boot:.4f}" if p == 0 else f"{p:.4f}"


def pval(p) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def tt(name: str) -> str:
    return "\\texttt{" + str(name).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%") + "}"


def rng(vals, nd: int = 3, sign: bool = True) -> str:
    f = signed if sign else num
    vals = [v for v in vals if ok(v)]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    return f(lo, nd) if np.isclose(lo, hi) else f"between {f(lo, nd)} and {f(hi, nd)}"


# ----------------------------------------------------------------------------- loading

class Results:
    def __init__(self, run_dir: Path, cfg: dict):
        self.rd, self.cfg = run_dir, cfg
        self.dets = [d for d in cfg["detectors"]["use"]]

    def path(self, rel: str) -> Path:
        return self.rd / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def csv(self, rel: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.path(rel), **kw)

    def json(self, rel: str) -> dict:
        return json.loads(self.path(rel).read_text())


def split_letter(name: str) -> str:
    return name.split("_")[0] if "_" in name else name


# ----------------------------------------------------------------------------- value computation

def compute(R: Results, P: Results | None) -> dict[str, tuple[str, str]]:
    """Return {key: (value, source)}. Missing files simply produce no keys."""
    V: dict[str, tuple[str, str]] = {}

    def put(key, value, source):
        if value is not None:
            V[key] = (value, source)

    letters = {split_letter(s): s for s in R.cfg["splits"]["tests"]}

    # --- dataset counts
    if R.exists("split_summary.csv"):
        ss = R.csv("split_summary.csv")
        def counts(sets):
            g = ss[ss.set.isin(sets)]
            return int(g[g.label == 0].n.sum()), int(g[g.label == 1].n.sum()), g
        src = "split_summary.csv"
        nr, nf, _ = counts(["fit", "val"])
        put("data.fitval.n_real", f"{nr:,}", src); put("data.fitval.n_fake", f"{nf:,}", src)
        for name, sets in [("fit", ["fit"]), ("val", ["val"])] + [(L, [f"test:{s}"]) for L, s in letters.items()]:
            nr, nf, g = counts(sets)
            if len(g) == 0:
                continue
            if name in letters:
                put(f"data.{name}.n_real", f"{nr:,}", src); put(f"data.{name}.n_fake", f"{nf:,}", src)
            k_gen, k_real = g[g.label == 1].cluster.nunique(), g[g.label == 0].cluster.nunique()
            put(f"split.{name}.sources", f"{k_gen} generators, {k_real} real source{'s' if k_real != 1 else ''}", src)
            put(f"split.{name}.n", f"{nr:,} / {nf:,}", src + " (real / generated)")

    # --- leakage
    if R.exists("leakage_report.json"):
        lk, src = R.json("leakage_report.json"), "leakage_report.json"
        put("leak.generator_overlap", "none" if not lk["generator_overlap"] else ", ".join(map(tt, lk["generator_overlap"])), src)
        put("leak.real_source_overlap", "none" if not lk["real_source_overlap"] else ", ".join(map(tt, lk["real_source_overlap"])), src)
        put("leak.prompt_overlap", str(lk["prompt_overlap"]), src)
        put("leak.fitval_prompt_overlap", str(lk["fit_val_prompt_overlap"]), src)
        put("leak.dedup_removed", str(lk["n_fit_val_removed_as_duplicates"]), src)
    if R.exists("manifest_splits.csv"):
        n_fail = len(R.csv("failed_images.csv")) if R.exists("failed_images.csv") else 0
        put("leak.unreadable", str(n_fail), "failed_images.csv (0 if the file does not exist)")

    # --- normal convention and missing flags
    if R.exists("geometry_convention.json"):
        gc, src = R.json("geometry_convention.json"), "geometry_convention.json"
        put("conv.sign", "(" + ", ".join("+1" if c > 0 else "$-$1" for c in gc["convention"]) + ")", src)
        put("conv.median_angle", num(gc["median_angle_deg"], 1), src)
    man = None
    if R.exists("manifest_splits.csv"):
        man = R.csv("manifest_splits.csv", keep_default_na=False)
        man = man[man.split != ""].copy()
        man["label"] = man.label.astype(int)
        man["seen_family"] = man.seen_family.astype(str).str.lower().isin(["true", "1"])
    if man is not None and R.exists("features/clean.csv"):
        f = R.csv("features/clean.csv").merge(man[["image_id", "label"]], on="image_id")
        for g in ("g2", "g3", "g4"):
            col = f"miss_{g}"
            if col in f:
                for lab, nm in ((0, "real"), (1, "fake")):
                    put(f"flags.{g}.{nm}", f"{100 * f.loc[f.label == lab, col].mean():.1f}\\%", "features/clean.csv, mean of " + col)

    # --- external validity
    if man is not None:
        te = man[(man.split == "test") & (man.label == 1) & (~man.seen_family)]
        gens = te.groupby("cluster").release_date.first()
        put("ext.n_unseen_generators", str(len(gens)), "manifest_splits.csv (test fakes, not seen family)")
        dates = sorted(d for d in gens if str(d).strip())
        if dates:
            note = f"manifest release_date, known for {len(dates)} of {len(gens)} generators"
            put("ext.earliest", dates[0][:7], note); put("ext.latest", dates[-1][:7], note)

    # --- stackers
    for d in R.dets:
        js = sorted(R.path(f"stackers/{d}").glob("M*.json")) if R.path(f"stackers/{d}").exists() else []
        if not js:
            continue
        rows = [json.loads(p.read_text()) for p in js]
        cv = []
        for r in rows:
            c = r.get("cv_auroc", {})
            cv.append(c.get(str(r["C"]), max(c.values()) if c else np.nan))
        src = f"stackers/{d}/M1..M5.json (range over models)"
        put(f"stacker.{d}.C", rng([r["C"] for r in rows], 3, sign=False), src)
        put(f"stacker.{d}.cv_auroc", rng(cv, 3, sign=False), src)
        put(f"stacker.{d}.temperature", rng([r["temperature"] for r in rows], 2, sign=False), src)
        put(f"stacker.{d}.threshold", rng([r["threshold_logit"] for r in rows], 2, sign=True), src)

    # --- main table
    if R.exists("metrics/metrics.csv"):
        met = R.csv("metrics/metrics.csv")
        m = met[(met.variant == "clean") & (met.subset == "unseen")]
        for r in m.itertuples():
            L = split_letter(r.split)
            put(f"main.{r.detector}.{r.model}.{L}", f"{num(r.macro_auroc)} / {num(r.tpr_at_1fpr)}",
                "metrics/metrics.csv (clean, unseen; macro AUROC / TPR at 1% FPR)")

    # --- pooled quantities from predictions
    if R.exists("predictions/all_main.csv.gz"):
        pred = R.csv("predictions/all_main.csv.gz", keep_default_na=False)
        pred["label"] = pred.label.astype(int)
        pred["seen_family"] = pred.seen_family.astype(str).str.lower().isin(["true", "1"])
        clean = pred[pred.variant == "clean"]
        m2 = []
        for d in R.dets:
            for mdl in ("M1", "M2", "M3"):
                q = clean[(clean.detector == d) & (clean.model == mdl)].drop_duplicates("image_id")
                if q.empty:
                    continue
                un = q[~((q.label == 1) & q.seen_family)]
                if mdl == "M2" and un.label.nunique() == 2:
                    m2.append(macro_over_generators(un.label.to_numpy(), un.logit.to_numpy(), un.cluster.to_numpy()))
                if mdl in ("M1", "M3"):
                    seen = q[(q.label == 0) | q.seen_family]
                    if seen.label.nunique() == 2:
                        put(f"seen.{d}.{mdl}", num(macro_over_generators(seen.label.to_numpy(), seen.logit.to_numpy(), seen.cluster.to_numpy())),
                            "predictions/all_main.csv.gz (clean, seen family fakes vs all reals, macro AUROC)")
                    if un.label.nunique() == 2:
                        a = all_metrics(un.label.to_numpy(), un.logit.to_numpy(), float(un.threshold.iloc[0]))
                        src = "predictions/all_main.csv.gz (clean, pooled unseen, fixed threshold)"
                        put(f"calib.{d}.ece.{mdl}", num(a["ece"]), src)
                        put(f"calib.{d}.acc_real.{mdl}", num(a["acc_real_at_t"]), src)
                        put(f"calib.{d}.acc_fake.{mdl}", num(a["acc_fake_at_t"]), src)
        if m2:
            put("m2.macro_auroc", num(float(np.mean(m2))), "predictions/all_main.csv.gz (M2, clean, pooled unseen; mean over detectors)")

    # --- hypotheses
    if R.exists("hypotheses.json"):
        h, src = R.json("hypotheses.json"), "hypotheses.json"
        B = h["settings"]["n_boot"]
        h1 = h.get("H1", {})
        tprs = []
        for d, v in h1.get("by_detector", {}).items():
            a = v["auroc"]
            if ok(a["delta"]):
                put(f"h1.{d}.delta", signed(a["delta"]), src + " H1.auroc.delta")
                put(f"h1.{d}.ci", ci(a["ci_low"], a["ci_high"]), src + " H1.auroc CI")
            put(f"h1.{d}.p", pval_boot(h1["holm"].get(d, {}).get("p_holm"), B), src + " H1.holm.p_holm")
            t = v.get("tpr_at_1fpr", {})
            if ok(t.get("delta")):
                tprs.append(f"{signed(t['delta'])} for {DET_NAMES.get(d, d)}")
        if tprs:
            txt = tprs[0] if len(tprs) == 1 else ", ".join(tprs[:-1]) + " and " + tprs[-1]
            put("h1.tpr_summary", txt + " in TPR at 1\\% FPR", src + " H1.tpr_at_1fpr.delta")
        if "pass" in h1:
            put("h1.verdict", "supported" if h1["pass"] else "not supported", src + " H1.pass")
        if "H2" in h:
            for d, a in h["H2"]["by_detector"].items():
                if ok(a["delta"]):
                    put(f"h2.{d}.delta", signed(a["delta"]), src + " H2.delta")
                    put(f"h2.{d}.ci", ci(a["ci_low"], a["ci_high"]), src + " H2 CI")
                put(f"h2.{d}.p", pval_boot(h["H2"]["holm"].get(d, {}).get("p_holm"), B), src + " H2.holm.p_holm")
            put("h2.verdict", "supported" if h["H2"]["pass"] else "not supported", src + " H2.pass")
            put("h2.abstract_verb", "exceeded" if h["H2"]["pass"] else "did not exceed", src + " H2.pass (edit the wording if needed)")
        if "H3" in h:
            for d, a in h["H3"]["by_detector"].items():
                if ok(a["delta"]):
                    put(f"h3.{d}.delta", signed(a["delta"]), src + " H3.delta")
                    put(f"h3.{d}.ci", ci(a["ci_low"], a["ci_high"]), src + " H3 CI")
                put(f"h3.{d}.p", pval_boot(a.get("p_boot"), B), src + " H3.p_boot (one test per detector, no Holm step)")
            put("h3.verdict", "supported" if h["H3"]["pass"] else "not supported", src + " H3.pass")
            put("h3.abstract_verb", "held" if h["H3"]["pass"] else "did not hold", src + " H3.pass (edit the wording if needed)")
        h4 = h.get("H4", {}).get("by_detector", {})
        if h4:
            flags = []
            for v in VARIANTS:
                deltas, extra, d1, d3 = [], [], [], []
                for d, vv in h4.items():
                    if v not in vv:
                        continue
                    g = vv[v]["gain"]
                    if ok(g["delta"]):
                        put(f"h4.{d}.{v}.delta", signed(g["delta"]), src + f" H4.{d}.{v}.gain")
                        put(f"h4.{d}.{v}.ci", ci(g["ci_low"], g["ci_high"]), src + f" H4.{d}.{v}.gain CI")
                        deltas.append(g["delta"])
                    extra.append(vv[v]["extra_fpr_drift"]); d1.append(vv[v]["fpr_drift_M1"]); d3.append(vv[v]["fpr_drift_M3"])
                    if v in ("jpeg75", "resize05"):
                        flags.append(bool(vv[v]["pass"]))
                put(f"h4.{v}.range", rng(deltas), src + f" H4 gain under {v}, range over detectors")
                ex = [e for e in extra if ok(e)]
                if ex:
                    put(f"h4.{v}.extra_fpr_drift", signed(100 * float(np.mean(ex)), 1), src + f" H4 extra_fpr_drift under {v}, mean over detectors, points")
                if v == "jpeg75":
                    if [x for x in d1 if ok(x)]:
                        put("h4.jpeg75.fpr_drift_m1", signed(100 * float(np.nanmean(d1)), 1), src + " H4 fpr_drift_M1 (jpeg75), mean over detectors, points")
                    if [x for x in d3 if ok(x)]:
                        put("h4.jpeg75.fpr_drift_m3", signed(100 * float(np.nanmean(d3)), 1), src + " H4 fpr_drift_M3 (jpeg75), mean over detectors, points")
            if flags:
                verb = "survived" if all(flags) else ("partly survived" if any(flags) else "did not survive")
                put("h4.abstract_verb", verb, src + " H4 pass flags for jpeg75 and resize05 (edit the wording if needed)")
        if h.get("H5"):
            sc = pd.DataFrame(h["H5"]).groupby("scene").delta.mean().sort_values()
            put("scene.largest", sc.index[-1].replace("_", " "), src + " H5, mean delta over detectors")
            put("scene.smallest", sc.index[0].replace("_", " "), src + " H5, mean delta over detectors")
        for d, a in h.get("delong_M3_vs_M1", {}).items():
            put(f"delong.{d}.z", num(a["z"], 2), src + " delong_M3_vs_M1")
            put(f"delong.{d}.p", pval(a["p"]), src + " delong_M3_vs_M1")

    # --- per generator
    if R.exists("metrics/per_generator.csv"):
        pg = R.csv("metrics/per_generator.csv")
        pg = pg[pg.variant == "clean"]
        w = pg.pivot_table(index=["detector", "split", "generator", "seen_family"], columns="model", values="auroc").reset_index()
        if {"M1", "M3"} <= set(w.columns):
            w["delta"] = w.M3 - w.M1
            w["seen_family"] = w.seen_family.astype(str).str.lower().isin(["true", "1"])
            un = w[~w.seen_family].groupby("generator").delta.mean().sort_values()
            src = "metrics/per_generator.csv (clean, M3 minus M1 AUROC, mean over detectors)"
            if len(un):
                put("pergen.largest", tt(un.index[-1]), src); put("pergen.smallest", tt(un.index[0]), src)
            sf = w[w.seen_family]
            if len(sf):
                put("pergen.seen_family_gain", signed(sf.delta.mean()), src + ", seen family generators")

    # --- feature shift
    if R.exists("metrics/feature_shift.csv"):
        fs = R.csv("metrics/feature_shift.csv")
        fs = fs[(fs.variant == "jpeg75") & (fs.label == 0)].dropna(subset=["smd"])
        if len(fs):
            r = fs.loc[fs.smd.abs().idxmax()]
            put("shift.top_feature", tt(r.feature), "metrics/feature_shift.csv (jpeg75, real photos, largest |SMD|)")
            put("shift.top_smd", num(abs(r.smd), 2), "metrics/feature_shift.csv (absolute SMD)")

    # --- ablations
    if R.exists("metrics/ablations.csv"):
        ab = R.csv("metrics/ablations.csv")
        for r in ab.itertuples():
            if ok(r.delta):
                put(f"abl.{r.ablation}.{r.detector}", f"{signed(r.delta)} {ci(r.ci_low, r.ci_high)}", "metrics/ablations.csv")

    # --- scene audit
    if R.exists("scene_audit_kappa.json"):
        put("scene.kappa", num(R.json("scene_audit_kappa.json")["kappa"], 2), "scene_audit_kappa.json (export_scene_audit.py --score)")

    # --- compute cost
    if R.exists("logs/timing_features.json"):
        t = R.json("logs/timing_features.json")
        put("compute.geom_sec", num(t["sec_per_image_geometry_models"] + t["sec_per_image_features"], 2), "logs/timing_features.json (models + features)")
        put("compute.feat_sec", num(t["sec_per_image_features"], 2), "logs/timing_features.json")
    if R.exists("logs/timing_detectors.json"):
        t = R.json("logs/timing_detectors.json")
        vals = [v for k, v in t.items() if k.endswith("_clean_sec_per_image")]
        if vals:
            put("compute.det_min", num(min(vals), 3), "logs/timing_detectors.json (clean)")
            put("compute.det_max", num(max(vals), 3), "logs/timing_detectors.json (clean)")
    for env in ("logs/environment_geometry_features.json", "logs/environment_detectors.json"):
        if R.exists(env) and R.json(env).get("gpu"):
            put("compute.gpu", "one " + R.json(env)["gpu"].replace("_", " "), env)
            break

    # --- zero shot reference
    ref = "scores/qwen25vl_zeroshot/clean.csv"
    if R.exists(ref) and man is not None:
        q = R.csv(ref).merge(man, on="image_id")
        q = q[~((q.label == 1) & q.seen_family)]
        if q.label.nunique() == 2:
            put("qwen.auroc", num(auroc(q.label, q.logit)), ref)
            put("qwen.n", f"{len(q):,}", ref)

    # --- pilot
    if P is not None and P.exists("hypotheses.json"):
        ph = P.json("hypotheses.json").get("pilot", {})
        if ph:
            d, v = next(iter(ph.items()))
            src = f"{P.rd}/hypotheses.json pilot.{d}"
            r1 = v["rule1"]
            put("pilot.delta", signed(r1["delta"]), src + ".rule1")
            put("pilot.ci90", ci(r1["ci_low"], r1["ci_high"]), src + ".rule1")
            if ok(v["rule2"].get("gain_M4")):
                put("pilot.gain_m4", signed(v["rule2"]["gain_M4"]), src + ".rule2")
            if ok(v["rule3"].get("extra_fpr_drift")):
                put("pilot.extra_fpr_drift", signed(100 * v["rule3"]["extra_fpr_drift"], 1), src + ".rule3 (points)")
            if ok(v.get("m2_macro_auroc")):
                put("pilot.m2_auroc", num(v["m2_macro_auroc"]), src)
            put("pilot.decision", v["decision"], src)
    return V


# ----------------------------------------------------------------------------- paper handling

def keys_in(path: Path) -> list[tuple[str, int, str]]:
    """(key, line number, short context) for every placeholder."""
    out = []
    for i, line in enumerate(path.read_text().split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in KEY_RE.finditer(line):
            ctx = re.sub(r"\\xyz\[[^\]]+\](\{\})?", "[xyz]", line[max(0, m.start() - 60): m.start()]).strip()
            out.append((m.group(1), i, ctx))
    return out


def fill_text(text: str, values: dict[str, tuple[str, str]]) -> tuple[str, int]:
    n = 0
    out_lines = []
    for line in text.split("\n"):
        if line.lstrip().startswith("%") or "\\newcommand{\\xyz}" in line:
            out_lines.append(line)
            continue

        def sub(m):
            nonlocal n
            k = m.group(1)
            if k in values:
                n += 1
                return values[k][0] + (m.group(2) or "")
            return m.group(0)
        out_lines.append(KEY_RE.sub(sub, line))
    return "\n".join(out_lines), n


def main() -> int:
    p = parser(__doc__)
    p.add_argument("--pilot-config", default=str(ROOT / "configs" / "pilot.yaml"))
    p.add_argument("--inplace", action="store_true", help="overwrite main.tex and supplementary.tex (a .bak copy is kept)")
    a = p.parse_args()
    cfg = config(a)
    R = Results(Path(cfg["experiment"]["run_dir"]), cfg)
    P = None
    if a.pilot_config and Path(a.pilot_config).exists():
        pc = load_config(a.pilot_config, [])
        P = Results(Path(pc["experiment"]["run_dir"]), pc)
    values = compute(R, P)

    files = [PAPER / "main.tex", PAPER / "supplementary.tex"]
    rows, seen = [], {}
    for f in files:
        for k, ln, ctx in keys_in(f):
            seen.setdefault(k, []).append(f"{f.name}:{ln}")
            rows.append((k, f.name, ln, ctx))

    out_dir = R.rd / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for k in sorted(seen):
        if k in values:
            status, val, src = "filled", values[k][0], values[k][1]
        elif k.startswith("manual."):
            status, val, src = "write yourself", "", MANUAL_HELP.get(k, "")
        else:
            status, val, src = "missing", "", "result file not found yet (run the stage that writes it)"
        table.append({"key": k, "status": status, "value": val, "source": src, "where": ", ".join(seen[k])})
    df = pd.DataFrame(table)
    df.to_csv(out_dir / "paper_values.csv", index=False)
    md = ["# Paper values", "", f"Run: `{R.rd}`. Filled {int((df.status == 'filled').sum())} of {len(df)} keys; "
          f"{int((df.status == 'missing').sum())} missing; {int((df.status == 'write yourself').sum())} to write yourself.", "",
          "| key | status | value | source | where in the paper |", "| --- | --- | --- | --- | --- |"]
    md += [f"| `{r.key}` | {r.status} | {r.value.replace('|', '/')} | {r.source} | {r.where} |" for r in df.itertuples()]
    (out_dir / "paper_values.md").write_text("\n".join(md) + "\n")

    total = 0
    for f in files:
        text, n = fill_text(f.read_text(), values)
        total += n
        if a.inplace:
            shutil.copy(f, f.with_suffix(".tex.bak"))
            f.write_text(text)
            target = f
        else:
            target = f.with_name(f.stem + "_filled.tex")
            target.write_text(text)
        left = len(KEY_RE.findall(text))
        print(f"{target.relative_to(ROOT)}: filled {n} placeholders, {left} still yellow")
    print(f"Value list: {out_dir / 'paper_values.md'}")
    unknown = sorted(set(values) - set(seen))
    if unknown:
        print(f"Note: {len(unknown)} computed values are not used in the paper (e.g. {unknown[:3]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
