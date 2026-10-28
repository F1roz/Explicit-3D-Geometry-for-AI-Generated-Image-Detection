"""Stage functions for the whole study. Each ``scripts/*.py`` file is a thin CLI around one of these.

Stages (run in this order, see docs/research_sop.md):

1. :func:`prepare`            ingest -> splits -> dedup -> processed variants -> manifest_splits.csv
2. :func:`extract_geometry`   geometry features per variant -> features/<variant>.csv
3. :func:`run_detectors`      base detector logits -> scores/<detector>/<variant>.csv
4. :func:`extract_controls`   DINOv2 features + scene labels -> controls/, scenes.csv
5. :func:`run_experiments`    M1-M5, metrics, hypothesis tests, pilot decision
6. :func:`run_ablations`      one-change ablations of M3
7. :func:`make_tables`        LaTeX/Markdown tables from result files
8. :func:`make_figures`       figures from result files

Every stage is resumable and writes its configuration and environment next to its outputs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import stats
from .config import save_config
from .data.dedup import drop_fit_duplicates_of_test, phash_file
from .data.ingest import ingest_from_config
from .data.manifest import validate_manifest
from .data.processing import VARIANT_EXT, load_rgb, write_processed
from .data.splits import assign_splits, check_leakage, explode_eval_sets, split_summary
from .geometry.features import FEATURE_NAMES, compute_features, estimate_convention, normals_from_points, orient_predicted_normals, valid_geometry_mask
from .utils import environment_info, ensure_dir, get_logger, load_json, save_json, set_seed

LOG = get_logger()


# ----------------------------------------------------------------------------- paths


def run_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["experiment"]["run_dir"])


def safe_id(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__")


def processed_path(cfg: dict[str, Any], image_id: str, variant: str) -> Path:
    return run_dir(cfg) / "processed" / variant / f"{safe_id(image_id)}{VARIANT_EXT[variant]}"


def load_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(run_dir(cfg) / "manifest_splits.csv", keep_default_na=False)
    df["label"] = df["label"].astype(int)
    return df[df["split"] != ""].reset_index(drop=True)


def _stamp(cfg: dict[str, Any], stage: str) -> None:
    d = ensure_dir(run_dir(cfg) / "logs")
    save_config(cfg, d / f"config_{stage}.yaml")
    save_json(environment_info(), d / f"environment_{stage}.json")


# ----------------------------------------------------------------------------- 1. prepare


def prepare(cfg: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Ingest datasets, assign leakage-safe splits, remove near duplicates, write variants."""
    set_seed(cfg["experiment"]["seed"])
    rd = ensure_dir(run_dir(cfg))
    _stamp(cfg, "prepare")
    sources = cfg["data"]["sources"]
    if cfg["data"].get("skip_missing_sources", False):
        keep = [s for s in sources if Path(s["root"]).exists()]
        for s in sources:
            if s not in keep:
                LOG.warning("Skipping missing source %s (%s)", s.get("dataset", s["kind"]), s["root"])
        sources = keep
    df = ingest_from_config(sources)
    validate_manifest(df)
    df.to_csv(rd / "manifest_all.csv", index=False)
    LOG.info("Ingested %d images from %d sources", len(df), len(sources))
    df = assign_splits(df, cfg["splits"], seed=cfg["experiment"]["seed"])
    if limit:  # pilot / smoke: keep a small balanced subset per split and cluster
        rng = np.random.default_rng(cfg["experiment"]["seed"])
        used = df[df.split != ""]
        keep_idx = []
        for _, g in used.groupby(["split", "cluster"]):
            keep_idx.extend(rng.choice(g.index.to_numpy(), size=min(limit, len(g)), replace=False))
        df.loc[~df.index.isin(keep_idx), ["split", "eval_sets"]] = ["", ""]
    used = df[df.split != ""].copy()
    LOG.info("Hashing %d images for near-duplicate removal", len(used))
    hashes = pd.Series({r.image_id: phash_file(r.path) for r in used.itertuples()}, dtype=np.uint64)
    df, n_drop = drop_fit_duplicates_of_test(df, hashes, cfg["data"]["dedup_threshold"])
    LOG.info("Removed %d fit/val images that nearly duplicate a test image", n_drop)
    report = check_leakage(df)
    report["n_fit_val_removed_as_duplicates"] = n_drop
    save_json(report, rd / "leakage_report.json")
    # processed variants
    variants = cfg["data"]["variants"]
    failed = []
    used = df[df.split != ""]
    for i, r in enumerate(used.itertuples()):
        outs = {v: processed_path(cfg, r.image_id, v) for v in variants}
        if all(p.exists() for p in outs.values()):
            continue
        try:
            img = load_rgb(r.path, cfg["data"].get("max_side"))
            write_processed(img, outs, cfg["data"]["clean_quality"])
        except Exception as e:  # unreadable or corrupt file: drop it and record why
            failed.append({"image_id": r.image_id, "error": repr(e)})
        if (i + 1) % 500 == 0:
            LOG.info("Processed %d / %d images", i + 1, len(used))
    if failed:
        pd.DataFrame(failed).to_csv(rd / "failed_images.csv", index=False)
        df.loc[df.image_id.isin({f["image_id"] for f in failed}), ["split", "eval_sets"]] = ["", ""]
        LOG.warning("%d images could not be read and were dropped (failed_images.csv)", len(failed))
    df.to_csv(rd / "manifest_splits.csv", index=False)
    split_summary(df).to_csv(rd / "split_summary.csv", index=False)
    return df


# ----------------------------------------------------------------------------- 2. geometry


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _done_ids(path: Path) -> set[str]:
    return set(pd.read_csv(path, usecols=["image_id"])["image_id"]) if path.exists() else set()


def make_geometry_fn(cfg: dict[str, Any], kind: str) -> Callable[[np.ndarray, pd.Series], Any]:
    """Return f(rgb, manifest_row) -> GeometryInputs for the configured estimator kind."""
    if kind == "oracle":
        from .geometry.estimators import OracleEstimator

        est = OracleEstimator()
        return lambda rgb, row: est(rgb, Path(row.path).with_suffix(".npz"))
    from .geometry.estimators import GeometryPipeline

    pipe = GeometryPipeline(cfg["geometry"])
    return lambda rgb, row: pipe(rgb)


def estimate_normal_convention(cfg: dict[str, Any], geom_fn, man: pd.DataFrame, n: int) -> np.ndarray:
    fit = man[man.split == "fit"].head(n)
    samples = []
    for r in fit.itertuples():
        gi = geom_fn(load_rgb(processed_path(cfg, r.image_id, "clean")), r)
        nd, ok = normals_from_points(gi.points)
        samples.append((gi.normal_pred, nd, gi.points, valid_geometry_mask(gi.depth, gi.mask) & ok))
    conv = estimate_convention(samples)
    # sanity: median angle after alignment
    med = float(np.median([np.median(np.degrees(np.arccos(np.clip(np.einsum("hwc,hwc->hw", orient_predicted_normals(a, p, conv), b), -1, 1)))[m]) for a, b, p, m in samples]))
    save_json({"convention": conv.tolist(), "n_images": len(samples), "median_angle_deg": med}, run_dir(cfg) / "geometry_convention.json")
    LOG.info("Normal convention %s (median angle %.1f deg on %d fit images)", conv.tolist(), med, len(samples))
    return conv


def extract_geometry(cfg: dict[str, Any], kind: str = "models", variants: list[str] | None = None,
                     feature_dir: str = "features", only_ids: list[str] | None = None) -> None:
    """Compute features for every manifest image (or only ``only_ids``, whose maps are then saved)."""
    set_seed(cfg["experiment"]["seed"])
    _stamp(cfg, f"geometry_{feature_dir}")
    man = load_manifest(cfg)
    if only_ids:
        man = man[man.image_id.isin(set(only_ids))]
        cfg["geometry"]["save_maps_for"] = list(only_ids)
    geom_fn = make_geometry_fn(cfg, kind)
    conv_cfg = cfg["geometry"].get("normal_convention", "auto")
    conv_file = run_dir(cfg) / "geometry_convention.json"
    if conv_cfg == "auto":
        conv = np.array(load_json(conv_file)["convention"]) if conv_file.exists() else \
            estimate_normal_convention(cfg, geom_fn, man, cfg["geometry"].get("convention_images", 50))
    else:
        conv = np.array(conv_cfg, dtype=float)
    out_dir = ensure_dir(run_dir(cfg) / feature_dir)
    save_maps = set(cfg["geometry"].get("save_maps_for", []))
    timing: dict[str, float] = {}
    for v in variants or cfg["data"]["variants"]:
        out = out_dir / f"{v}.csv"
        done = _done_ids(out)
        todo = man[~man.image_id.isin(done)]
        rows: list[dict] = []
        t0 = time.perf_counter()
        for i, r in enumerate(todo.itertuples()):
            rgb = load_rgb(processed_path(cfg, r.image_id, v))
            t_g = time.perf_counter()
            gi = geom_fn(rgb, r)
            timing["geometry_models"] = timing.get("geometry_models", 0.0) + time.perf_counter() - t_g
            t_f = time.perf_counter()
            want_maps = r.image_id in save_maps
            res = compute_features(gi, conv, return_maps=want_maps)
            timing["feature_computation"] = timing.get("feature_computation", 0.0) + time.perf_counter() - t_f
            feats = res[0] if want_maps else res
            if want_maps:
                _save_maps(cfg, r.image_id, v, gi, res[1])
            rows.append({"image_id": r.image_id, **feats})
            if len(rows) >= 200:
                _append_rows(out, rows)
                rows = []
                LOG.info("[%s] %d / %d images (%.2f s/img)", v, i + 1, len(todo), (time.perf_counter() - t0) / (i + 1))
        _append_rows(out, rows)
        timing[f"n_images_{v}"] = len(todo)
    # Accumulate with earlier runs of this stage (resumed runs, or variants run on another machine),
    # so a resume that finds nothing left to do never erases the measured cost.
    tfile = run_dir(cfg) / "logs" / f"timing_{feature_dir}.json"
    total = load_json(tfile) if tfile.exists() else {}
    for k in ("geometry_models", "feature_computation"):
        total[k] = total.get(k, 0.0) + timing.get(k, 0.0)
    for v in variants or cfg["data"]["variants"]:
        total[f"n_images_{v}"] = total.get(f"n_images_{v}", 0) + timing.get(f"n_images_{v}", 0)
    n = sum(val for k, val in total.items() if k.startswith("n_images_"))
    if n:
        total["sec_per_image_geometry_models"] = total["geometry_models"] / n
        total["sec_per_image_features"] = total["feature_computation"] / n
    save_json(total, tfile)


def _save_maps(cfg, image_id, variant, gi, maps) -> None:
    d = ensure_dir(run_dir(cfg) / "maps" / variant)
    keep = {k: maps[k] for k in ("normal_depth", "normal_pred", "theta", "residual") if k in maps}
    np.savez_compressed(d / f"{safe_id(image_id)}.npz", rgb=gi.rgb, depth=gi.depth, **{k: np.asarray(v, np.float32) for k, v in keep.items()})


# ----------------------------------------------------------------------------- 3. detectors


def run_detectors(cfg: dict[str, Any], names: list[str] | None = None, variants: list[str] | None = None,
                  check: bool = False) -> None:
    from .detectors import get_detector

    _stamp(cfg, "detectors")
    man = load_manifest(cfg)
    timing = {}
    for name in names or cfg["detectors"]["use"]:
        det = get_detector(name, cfg["detectors"].get(name, {}) or {})
        out_dir = ensure_dir(run_dir(cfg) / "scores" / name)
        for v in variants or cfg["data"]["variants"]:
            out = out_dir / f"{v}.csv"
            todo = man[~man.image_id.isin(_done_ids(out))]
            if check:
                todo = todo.head(8)
            chunk = int(cfg["detectors"].get("chunk", 500))
            t0 = time.perf_counter()
            for s in range(0, len(todo), chunk):
                part = todo.iloc[s : s + chunk]
                logits = det.score([str(processed_path(cfg, i, v)) for i in part.image_id])
                res = pd.DataFrame({"image_id": part.image_id.to_numpy(), "logit": logits, "label": part.label.to_numpy()})
                if check:
                    print(res.to_string(index=False))
                    print("Mean logit real:", res.loc[res.label == 0, "logit"].mean(), "fake:", res.loc[res.label == 1, "logit"].mean())
                    break
                _append_rows(out, res[["image_id", "logit"]].to_dict("records"))
                LOG.info("[%s/%s] %d / %d", name, v, min(s + chunk, len(todo)), len(todo))
            if len(todo) and not check:
                timing[f"{name}_{v}_sec_per_image"] = (time.perf_counter() - t0) / len(todo)
            if check:
                return
    tfile = run_dir(cfg) / "logs" / "timing_detectors.json"
    total = load_json(tfile) if tfile.exists() else {}
    total.update(timing)  # keep measurements of detectors or variants run earlier or elsewhere
    save_json(total, tfile)


# ----------------------------------------------------------------------------- 4. controls


def extract_controls(cfg: dict[str, Any], mock: bool = False, variants: list[str] | None = None) -> None:
    from .controls import mock_control_features

    _stamp(cfg, "controls")
    man = load_manifest(cfg)
    out_dir = ensure_dir(run_dir(cfg) / "controls")
    variants = variants or cfg["data"]["variants"]
    if mock:
        for v in variants:
            imgs = [load_rgb(processed_path(cfg, i, v)) for i in man.image_id]
            np.savez(out_dir / f"{v}.npz", ids=man.image_id.to_numpy(), feats=mock_control_features(imgs))
        pd.DataFrame({"image_id": man.image_id, "scene": "synthetic", "scene_conf": 1.0}).to_csv(run_dir(cfg) / "scenes.csv", index=False)
        return
    from .controls import CLIPSceneLabeller, DINOv2Extractor

    dino = DINOv2Extractor(cfg["controls"]["dino_model"])
    for v in variants:
        if (out_dir / f"{v}.npz").exists():
            continue
        feats = dino([str(processed_path(cfg, i, v)) for i in man.image_id])
        np.savez(out_dir / f"{v}.npz", ids=man.image_id.to_numpy(), feats=feats)
        LOG.info("DINOv2 features done for %s", v)
    if not (run_dir(cfg) / "scenes.csv").exists():
        lab = CLIPSceneLabeller(cfg["controls"]["clip_model"], min_conf=cfg["controls"]["scene_min_conf"])
        sc = lab([str(processed_path(cfg, i, "clean")) for i in man.image_id])
        sc.insert(0, "image_id", man.image_id.to_numpy())
        sc.to_csv(run_dir(cfg) / "scenes.csv", index=False)


# ----------------------------------------------------------------------------- 5. experiments


def feature_shift_table(cfg: dict[str, Any], man: pd.DataFrame) -> pd.DataFrame:
    """Standardised mean shift of each feature between clean and processed copies (test images)."""
    test_ids = set(man.loc[man.split == "test", "image_id"])
    clean = pd.read_csv(run_dir(cfg) / "features" / "clean.csv").set_index("image_id")
    rows = []
    for v in cfg["data"]["variants"]:
        if v == "clean":
            continue
        proc = pd.read_csv(run_dir(cfg) / "features" / f"{v}.csv").set_index("image_id")
        ids = sorted(test_ids & set(clean.index) & set(proc.index))
        lab = man.set_index("image_id").loc[ids, "label"]
        for f in FEATURE_NAMES:
            if f.startswith("miss_"):
                continue
            sd = clean.loc[ids, f].std()
            for y in (0, 1):
                idy = lab.index[lab == y]
                diff = (proc.loc[idy, f] - clean.loc[idy, f]).mean()
                rows.append({"variant": v, "feature": f, "label": y, "smd": float(diff / sd) if sd and np.isfinite(sd) else np.nan})
    return pd.DataFrame(rows)


def run_experiments(cfg: dict[str, Any]) -> dict[str, Any]:
    from .experiment import RunData, hypothesis_tests, metrics_table, per_generator_table, pilot_decision, run_models

    set_seed(cfg["experiment"]["seed"])
    _stamp(cfg, "experiments")
    rd = RunData.load(run_dir(cfg), cfg["data"]["variants"])
    preds = [run_models(rd, d, cfg, models=cfg["stacking"]["models"]) for d in cfg["detectors"]["use"]]
    pred = pd.concat(preds, ignore_index=True)
    out = ensure_dir(run_dir(cfg) / "metrics")
    pred.to_csv(run_dir(cfg) / "predictions" / "all_main.csv.gz", index=False)
    metrics_table(pred).to_csv(out / "metrics.csv", index=False)
    per_generator_table(pred).to_csv(out / "per_generator.csv", index=False)
    h = hypothesis_tests(pred, cfg)
    if cfg.get("pilot", {}).get("enabled", False):
        h["pilot"] = {d: pilot_decision(h, pred, d, cfg) for d in cfg["detectors"]["use"]}
    save_json(h, run_dir(cfg) / "hypotheses.json")
    try:
        feature_shift_table(cfg, rd.manifest).to_csv(out / "feature_shift.csv", index=False)
    except FileNotFoundError:
        pass
    return h


def run_ablations(cfg: dict[str, Any]) -> pd.DataFrame:
    from .experiment import RunData, _paired, ablation_specs, run_models

    _stamp(cfg, "ablations")
    rd = RunData.load(run_dir(cfg), cfg["data"]["variants"])
    full = pd.read_csv(run_dir(cfg) / "predictions" / "all_main.csv.gz", keep_default_na=False)
    full["seen_family"] = full["seen_family"].astype(str).str.lower().isin(["true", "1"])
    rows = []
    for d in cfg["detectors"]["use"]:
        base = full[(full.detector == d) & (full.model == "M3")]
        for spec in ablation_specs(cfg):
            if spec["name"] not in cfg["ablations"]["run"] and "all" not in cfg["ablations"]["run"]:
                continue
            try:
                p = run_models(rd, d, cfg, models=["M3"], tag=f"_{spec['name']}", geo_cols=spec.get("geo_cols"),
                               feature_dir=spec.get("feature_dir", "features"), fit_variants=spec.get("fit_variants"),
                               kind=spec.get("kind"))
            except FileNotFoundError as e:
                LOG.warning("Ablation %s skipped: %s", spec["name"], e)
                continue
            both = pd.concat([base, p], ignore_index=True)
            j = _paired(both, d, f"M3_{spec['name']}", "M3", None, ["clean"])
            r = stats.cluster_bootstrap_delta(j.label, j.logit, j.logit_b, j.cl, stats.macro_auroc,
                                              cfg["evaluation"]["n_boot"], cfg["evaluation"]["alpha"], cfg["experiment"]["seed"])
            rows.append({"detector": d, "ablation": spec["name"], **r})
            LOG.info("Ablation %s / %s: delta %.4f [%.4f, %.4f]", d, spec["name"], r["delta"], r["ci_low"], r["ci_high"])
    abl = pd.DataFrame(rows)
    abl.to_csv(ensure_dir(run_dir(cfg) / "metrics") / "ablations.csv", index=False)
    return abl


# ----------------------------------------------------------------------------- 7. tables


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal Markdown table writer (avoids an extra dependency)."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(_fmt(v) if isinstance(v, float) else str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _fmt_p(p: float, n_boot: int) -> str:
    """Bootstrap p-values cannot be smaller than 1/B; report them as an upper bound."""
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"<{1 / n_boot:.4f}" if p == 0 else f"{p:.4f}"


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def make_tables(cfg: dict[str, Any]) -> Path:
    """Write LaTeX rows and a Markdown summary that map one-to-one onto the paper's [xyz] cells."""
    rd = run_dir(cfg)
    out = ensure_dir(rd / "tables")
    met = pd.read_csv(rd / "metrics" / "metrics.csv")
    h = load_json(rd / "hypotheses.json")
    lines_md = [f"# Results summary ({time.strftime('%Y-%m-%d %H:%M')})", "", "Source files: metrics/metrics.csv, hypotheses.json, metrics/ablations.csv", ""]
    # Table: main comparison (clean, unseen, macro AUROC and TPR@1%FPR) per detector x model x set
    m = met[(met.variant == "clean") & (met.subset == "unseen")]
    tex = []
    for (d, mdl), g in m.groupby(["detector", "model"]):
        cells = [f"{_fmt(r.macro_auroc)} / {_fmt(r.tpr_at_1fpr)}" for r in g.sort_values("split").itertuples()]
        tex.append(f"{d} & {mdl} & " + " & ".join(cells) + r" \\")
    (out / "main_results_rows.tex").write_text("\n".join(tex) + "\n")
    lines_md += ["## Main results (macro AUROC / TPR at 1% FPR, unseen generators, clean)", "",
                 _md_table(m.pivot_table(index=["detector", "model"], columns="split", values="macro_auroc")), ""]
    # Hypotheses
    hrows = []
    for d, v in h["H1"]["by_detector"].items():
        a = v["auroc"]
        hrows.append(f"H1 & {d} & {_fmt(a['delta'])} & [{_fmt(a['ci_low'])}, {_fmt(a['ci_high'])}] & {_fmt_p(h['H1']['holm'].get(d, {}).get('p_holm', float('nan')), h['settings']['n_boot'])}" + r" \\")
    for d, a in h.get("H2", {}).get("by_detector", {}).items():
        hrows.append(f"H2 & {d} & {_fmt(a['delta'])} & [{_fmt(a['ci_low'])}, {_fmt(a['ci_high'])}] & {_fmt_p(h['H2']['holm'].get(d, {}).get('p_holm', float('nan')), h['settings']['n_boot'])}" + r" \\")
    for d, a in h.get("H3", {}).get("by_detector", {}).items():
        hrows.append(f"H3 & {d} & {_fmt(a['delta'])} & [{_fmt(a['ci_low'])}, {_fmt(a['ci_high'])}] & {_fmt_p(a.get('p_boot', float('nan')), h['settings']['n_boot'])}" + r" \\")
    for d, vv in h.get("H4", {}).get("by_detector", {}).items():
        for var, a in vv.items():
            g = a["gain"]
            hrows.append(f"H4 ({var}) & {d} & {_fmt(g['delta'])} & [{_fmt(g['ci_low'])}, {_fmt(g['ci_high'])}] & extra FPR drift {_fmt(100 * a['extra_fpr_drift'], 1)} pts" + r" \\")
    (out / "hypotheses_rows.tex").write_text("\n".join(hrows) + "\n")
    lines_md += ["## Hypothesis tests", "", "```", *hrows, "```", "",
                 f"H1 pass: {h['H1']['pass']} ({h['H1']['n_detectors_pass']} detectors)",
                 f"H2 pass: {h.get('H2', {}).get('pass')}", f"H3 pass: {h.get('H3', {}).get('pass')}", ""]
    if "pilot" in h:
        lines_md += ["## Pilot decision", "", "```", json.dumps({d: v["decision"] for d, v in h["pilot"].items()}, indent=2), "```", ""]
    abl_p = rd / "metrics" / "ablations.csv"
    if abl_p.exists():
        abl = pd.read_csv(abl_p)
        arows = [f"{r.ablation.replace('_', ' ')} & {r.detector} & {_fmt(r.delta)} & [{_fmt(r.ci_low)}, {_fmt(r.ci_high)}]" + r" \\" for r in abl.itertuples()]
        (out / "ablation_rows.tex").write_text("\n".join(arows) + "\n")
        lines_md += ["## Ablations (change in macro AUROC vs full M3)", "", _md_table(abl, index=False), ""]
    ref = rd / "scores" / "qwen25vl_zeroshot" / "clean.csv"
    if ref.exists():  # optional outside reference (scripts/zero_shot_mllm.py)
        from .metrics import auroc, macro_over_generators

        man = load_manifest(cfg)
        q = pd.read_csv(ref).merge(man, on="image_id")
        q["seen_family"] = q["seen_family"].astype(str).str.lower().isin(["true", "1"])
        q = q[~((q.label == 1) & q.seen_family)]
        lines_md += ["## Zero shot Qwen2.5-VL reference (not a hypothesis test)", "",
                     f"n = {len(q)}, pooled AUROC = {_fmt(auroc(q.label, q.logit))}, "
                     f"macro AUROC = {_fmt(macro_over_generators(q.label.to_numpy(), q.logit.to_numpy(), q.cluster.to_numpy()))}", ""]
    (out / "results_summary.md").write_text("\n".join(lines_md))
    return out


# ----------------------------------------------------------------------------- 8. figures


def make_figures(cfg: dict[str, Any]) -> list[Path]:
    from . import viz

    rd = run_dir(cfg)
    fig_dir = ensure_dir(rd / "figures")
    made: list[Path] = []
    met = pd.read_csv(rd / "metrics" / "metrics.csv")
    gen = pd.read_csv(rd / "metrics" / "per_generator.csv")
    pred = pd.read_csv(rd / "predictions" / "all_main.csv.gz", keep_default_na=False)
    pred["seen_family"] = pred["seen_family"].astype(str).str.lower().isin(["true", "1"])
    made += viz.model_comparison(met, fig_dir / "fig_model_comparison")
    made += viz.forest_per_generator(gen, fig_dir / "fig_per_generator_gain")
    made += viz.robustness_chart(met, fig_dir / "fig_robustness", cfg["data"]["variants"])
    for d in cfg["detectors"]["use"]:
        made += viz.roc_curves(pred, d, fig_dir / f"fig_roc_{d}")
    man = load_manifest(cfg)
    feat = pd.read_csv(rd / "features" / "clean.csv")
    made += viz.feature_distributions(feat, man, fig_dir / "fig_feature_distributions",
                                      ["g1_theta_mean", "g1_theta_da_mean", "g2_res_median", "g2_light_dispersion", "g3_inlier_ratio", "g4_aocc_mean"])
    if (rd / "metrics" / "feature_shift.csv").exists():
        made += viz.feature_shift(pd.read_csv(rd / "metrics" / "feature_shift.csv"), fig_dir / "fig_feature_shift")
    h = load_json(rd / "hypotheses.json")
    if h.get("H5"):
        made += viz.scene_heatmap(pd.DataFrame(h["H5"]), fig_dir / "fig_scene_gain")
    if (rd / "metrics" / "ablations.csv").exists():
        abl = pd.read_csv(rd / "metrics" / "ablations.csv")
        if len(abl):
            made += viz.ablation_chart(abl, fig_dir / "fig_ablation")
    timing = _timing_table(rd)
    if len(timing):
        made += viz.compute_cost(timing, fig_dir / "fig_compute_cost")
    maps_dir = rd / "maps" / "clean"
    if maps_dir.exists():
        made += _qualitative_figure(cfg, rd, man, maps_dir, fig_dir)
    return made


def _qualitative_figure(cfg: dict[str, Any], rd: Path, man: pd.DataFrame, maps_dir: Path, fig_dir: Path) -> list[Path]:
    """Qualitative panel. With errors_<detector>_M3.csv (scripts/export_errors.py) the rows are chosen
    deliberately: 2 correct real, 2 correct generated (Test B), then the most confident false positive
    and false negative whose maps were saved. Without it, the first saved maps are used."""
    from . import viz

    have = {f.stem: f for f in maps_dir.glob("*.npz")}
    lab = man.set_index("image_id")
    chosen: list[tuple[str, str]] = []  # (image_id, row title)
    err_files = sorted(rd.glob("errors_*_M3.csv"))
    if err_files:
        err = pd.read_csv(err_files[0], keep_default_na=False)
        names = {"correct_real": "Real, correct", "correct_fake": "AI, correct",
                 "false_positive": "Real, flagged as AI", "false_negative": "AI, missed"}
        for kind, k in (("correct_real", 2), ("correct_fake", 2), ("false_positive", 1), ("false_negative", 1)):
            rows = err[(err.kind == kind) & err.image_id.map(lambda i: safe_id(i) in have)].head(k)
            for r in rows.itertuples():
                title = names[kind] + (f"\n{r.generator}" if r.label == 1 else "")
                chosen.append((r.image_id, title))
    if not chosen:
        for stem in sorted(have)[: cfg["figures"].get("n_qualitative", 4)]:
            iid = next((i for i in man.image_id if safe_id(i) == stem), stem)
            row = lab.loc[iid] if iid in lab.index else None
            chosen.append((iid, ("AI: " + str(row.generator)) if row is not None and row.label == 1 else "Real photo"))
    if not chosen:
        return []
    items, titles = [], []
    for iid, title in chosen:
        z = np.load(have[safe_id(iid)])
        items.append({"rgb": z["rgb"], "depth": z["depth"], "maps": {k: z[k] for k in ("normal_depth", "normal_pred", "theta", "residual") if k in z}})
        titles.append(title)
    return viz.qualitative_panel(items, fig_dir / "fig_qualitative", titles)


def _timing_table(rd: Path) -> pd.DataFrame:
    rows = []
    p = rd / "logs" / "timing_features.json"
    if p.exists():
        t = load_json(p)
        for k, lab in (("sec_per_image_geometry_models", "Geometry models (all four)"), ("sec_per_image_features", "Feature computation")):
            if k in t:
                rows.append({"component": lab, "sec_per_image": t[k]})
    p = rd / "logs" / "timing_detectors.json"
    if p.exists():
        t = load_json(p)
        for k, v in t.items():
            if k.endswith("_clean_sec_per_image"):
                rows.append({"component": f"Base detector {k.split('_')[0]}", "sec_per_image": v})
    return pd.DataFrame(rows)


def explode_for_report(df: pd.DataFrame) -> pd.DataFrame:  # small helper used by notebooks
    return explode_eval_sets(df)
