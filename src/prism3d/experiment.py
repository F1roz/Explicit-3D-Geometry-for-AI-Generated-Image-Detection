"""Model comparison M1 to M5, ablations, and the pre-registered hypothesis tests.

Inputs live under ``<run_dir>/`` (all produced by the scripts, see docs/reproducibility.md):

* ``manifest_splits.csv``              manifest with ``split`` and ``seen_family`` columns
* ``features/<variant>.csv``           explicit geometry features (prism3d.geometry.features)
* ``scores/<detector>/<variant>.csv``  base detector logits
* ``controls/<variant>.npz``           DINOv2 CLS features (``ids``, ``feats``)
* ``scenes.csv``                       optional scene labels

Outputs: ``predictions/<detector>/<model>.csv``, ``metrics/*.csv``, ``hypotheses.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import stats
from .controls import fit_pca, pca_columns
from .geometry.features import FEATURE_NAMES, GROUPS
from .metrics import all_metrics, auroc, macro_over_generators, rates_at_threshold, tpr_at_fpr
from .stacking import Stacker
from .utils import ensure_dir, save_json

MODELS = ("M1", "M2", "M3", "M4", "M5")
PRED_COLS = ["image_id", "split", "eval_sets", "label", "cluster", "generator", "seen_family", "scene", "dataset", "pair_id"]


def explode_pred(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per (prediction, test set). Test sets may share real images."""
    p = pred.copy()
    p["eval_set"] = p["eval_sets"].astype(str).str.split("|")
    return p.explode("eval_set").reset_index(drop=True)


@dataclass
class RunData:
    run_dir: Path
    manifest: pd.DataFrame
    variants: list[str]

    @classmethod
    def load(cls, run_dir: str | Path, variants: list[str]) -> "RunData":
        run_dir = Path(run_dir)
        man = pd.read_csv(run_dir / "manifest_splits.csv", keep_default_na=False)
        man = man[man["split"] != ""].copy()
        man["label"] = man["label"].astype(int)
        man["seen_family"] = man["seen_family"].astype(str).str.lower().isin(["true", "1"])
        scenes = run_dir / "scenes.csv"
        if scenes.exists():
            man = man.merge(pd.read_csv(scenes), on="image_id", how="left")
        if "scene" not in man.columns:
            man["scene"] = "unlabelled"
        man["scene"] = man["scene"].fillna("unlabelled")
        return cls(run_dir, man, variants)

    def features(self, variant: str, feature_dir: str = "features") -> pd.DataFrame:
        return pd.read_csv(self.run_dir / feature_dir / f"{variant}.csv")

    def scores(self, detector: str, variant: str) -> pd.DataFrame:
        s = pd.read_csv(self.run_dir / "scores" / detector / f"{variant}.csv")
        return s.rename(columns={"logit": "base_logit"})[["image_id", "base_logit"]]

    def controls(self, variant: str) -> tuple[np.ndarray, np.ndarray] | None:
        p = self.run_dir / "controls" / f"{variant}.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        return z["ids"], z["feats"]


def build_table(rd: RunData, detector: str, variant: str, pca=None, feature_dir: str = "features") -> pd.DataFrame:
    """Manifest + base logit + geometry features (+ DINOv2 PCs if a fitted PCA is given)."""
    t = rd.manifest.merge(rd.scores(detector, variant), on="image_id", how="inner")
    t = t.merge(rd.features(variant, feature_dir), on="image_id", how="inner")
    if pca is not None:
        ctrl = rd.controls(variant)
        if ctrl is None:
            raise FileNotFoundError(f"controls/{variant}.npz missing but the DINOv2 control was requested")
        ids, feats = ctrl
        pcs = pca_columns(pca, feats)
        pcs.insert(0, "image_id", ids)
        t = t.merge(pcs, on="image_id", how="inner")
    t["variant"] = variant
    return t


def model_columns(model: str, geo_cols: list[str], dino_cols: list[str]) -> list[str]:
    return {
        "M1": ["base_logit"],
        "M2": list(geo_cols),
        "M3": ["base_logit", *geo_cols],
        "M4": ["base_logit", *dino_cols],
        "M5": ["base_logit", *geo_cols, *dino_cols],
    }[model]


def fit_dino_pca(rd: RunData, n_components: int, seed: int):
    ctrl = rd.controls("clean")
    if ctrl is None:
        return None, []
    ids, feats = ctrl
    fit_ids = set(rd.manifest.loc[rd.manifest.split == "fit", "image_id"])
    m = np.array([i in fit_ids for i in ids])
    pca = fit_pca(feats[m], n_components, seed)
    return pca, [f"dino_pc{i}" for i in range(pca.n_components_)]


def run_models(rd: RunData, detector: str, cfg: dict[str, Any], models=MODELS, geo_cols: list[str] | None = None,
               tag: str = "", feature_dir: str = "features", fit_variants: list[str] | None = None,
               kind: str | None = None) -> pd.DataFrame:
    """Fit each model on fit/val (clean unless ``fit_variants``) and predict every test variant."""
    sc = cfg["stacking"]
    geo_cols = geo_cols or list(FEATURE_NAMES)
    pca, dino_cols = fit_dino_pca(rd, len(geo_cols), cfg["experiment"]["seed"]) if any(m in ("M4", "M5") for m in models) else (None, [])
    tables = {v: build_table(rd, detector, v, pca, feature_dir) for v in rd.variants}
    fit_variants = fit_variants or ["clean"]
    train = pd.concat([tables[v] for v in fit_variants], ignore_index=True)
    fit_df, val_df = train[train.split == "fit"], train[train.split == "val"]
    preds = []
    out_dir = ensure_dir(rd.run_dir / "predictions" / detector)
    for model in models:
        if model in ("M4", "M5") and not dino_cols:
            continue
        cols = model_columns(model, geo_cols, dino_cols)
        st = Stacker(columns=cols, kind=kind or sc.get("kind", "logreg"), C_grid=tuple(sc["C_grid"]),
                     n_splits=sc["cv_folds"], target_fpr=sc["target_fpr"], seed=cfg["experiment"]["seed"]).fit(fit_df, val_df)
        name = f"{model}{tag}"
        st.save(rd.run_dir / "stackers" / detector / f"{name}.pkl")
        save_json(st.summary() | {"coefficients": st.coefficients()}, rd.run_dir / "stackers" / detector / f"{name}.json")
        for v, t in tables.items():
            te = t[t.split == "test"]
            p = te[PRED_COLS].copy()
            p["variant"], p["model"], p["detector"] = v, name, detector
            p["logit"] = st.predict_logit(te)
            p["threshold"] = st.threshold
            preds.append(p)
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(out_dir / f"predictions{tag}.csv.gz", index=False)
    return pred


# ----------------------------------------------------------------------------- evaluation


def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    """All metrics per detector, model, test split and variant (unseen and all generators)."""
    rows = []
    for (det, model, split, var), g in explode_pred(pred).groupby(["detector", "model", "eval_set", "variant"]):
        for subset, gg in (("all", g), ("unseen", g[~((g.label == 1) & g.seen_family)])):
            if gg.label.nunique() < 2:
                continue
            m = all_metrics(gg.label.to_numpy(), gg.logit.to_numpy(), float(gg.threshold.iloc[0]))
            m["macro_auroc"] = macro_over_generators(gg.label.to_numpy(), gg.logit.to_numpy(), gg.cluster.to_numpy())
            rows.append({"detector": det, "model": model, "split": split, "variant": var, "subset": subset, **m})
    return pd.DataFrame(rows)


def per_generator_table(pred: pd.DataFrame) -> pd.DataFrame:
    """AUROC of each generator's fakes against all reals of the same test split."""
    rows = []
    for (det, model, split, var), g in explode_pred(pred).groupby(["detector", "model", "eval_set", "variant"]):
        real = g[g.label == 0]
        for gen, f in g[g.label == 1].groupby("cluster"):
            s = pd.concat([real, f])
            rows.append({"detector": det, "model": model, "split": split, "variant": var, "generator": gen,
                         "seen_family": bool(f.seen_family.iloc[0]), "n_fake": len(f), "n_real": len(real),
                         "auroc": auroc(s.label, s.logit), "tpr_at_1fpr": tpr_at_fpr(s.label, s.logit, 0.01)})
    return pd.DataFrame(rows)


def _paired(pred: pd.DataFrame, det: str, a: str, b: str, sets: list[str] | None, variants: list[str],
            unseen_only: bool = True) -> pd.DataFrame:
    """Rows where both models scored the same image.

    ``sets=None`` pools every test image once (shared real photos are not double counted);
    otherwise only images belonging to the listed test sets are used. Clusters (the bootstrap
    unit) are the generator for fakes and the real source for reals, per variant.
    """
    q = pred[(pred.detector == det) & pred.variant.isin(variants)]
    if sets:
        q = explode_pred(q)
        q = q[q.eval_set.isin(sets)].drop_duplicates(["image_id", "variant", "model"])
    if unseen_only:
        q = q[~((q.label == 1) & q.seen_family)]
    pa = q[q.model == a].set_index(["image_id", "variant"])
    pb = q[q.model == b].set_index(["image_id", "variant"])
    j = pa.join(pb[["logit"]], rsuffix="_b", how="inner").reset_index()
    j["cl"] = j["cluster"] + "/" + j["variant"]
    return j


def hypothesis_tests(pred: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered tests H1 to H4 (+ descriptive H5). See docs/methodology.md, section 9."""
    ev = cfg["evaluation"]
    B, alpha, seed = ev["n_boot"], ev["alpha"], cfg["experiment"]["seed"]
    test_splits = sorted(explode_pred(pred).eval_set.unique())
    dets = sorted(pred.detector.unique())
    res: dict[str, Any] = {"settings": {"n_boot": B, "alpha": alpha, "test_sets": test_splits,
                                        "practical_min_delta_auroc": ev["practical_min_delta_auroc"]}}

    def boot(j: pd.DataFrame, metric=stats.macro_auroc) -> dict:
        if j.empty or j.label.nunique() < 2:
            return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_boot": float("nan")}
        return stats.cluster_bootstrap_delta(j.label.to_numpy(), j.logit.to_numpy(), j.logit_b.to_numpy(),
                                             j.cl.to_numpy(), metric, B, alpha, seed)

    def tpr_metric(y, s, cl):  # noqa: ARG001
        return tpr_at_fpr(y, s, 0.01)

    # H1: geometry adds to each base (all test splits pooled, unseen generators, clean)
    h1 = {}
    for d in dets:
        j = _paired(pred, d, "M3", "M1", None, ["clean"])
        h1[d] = {"auroc": boot(j), "tpr_at_1fpr": boot(j, tpr_metric),
                 "per_set": {s: boot(_paired(pred, d, "M3", "M1", [s], ["clean"])) for s in test_splits}}
    p1 = stats.holm({d: h1[d]["auroc"]["p_boot"] for d in dets if np.isfinite(h1[d]["auroc"]["p_boot"])}, alpha)
    n_pass = sum(1 for d in dets if d in p1 and p1[d]["reject"] and h1[d]["auroc"]["delta"] > 0)
    res["H1"] = {"by_detector": h1, "holm": p1, "n_detectors_pass": n_pass, "pass": n_pass >= min(2, len(dets))}

    # H2: geometry beats the DINOv2 control (M3 vs M4)
    if "M4" in set(pred.model):
        h2 = {d: boot(_paired(pred, d, "M3", "M4", None, ["clean"])) for d in dets}
        p2 = stats.holm({d: v["p_boot"] for d, v in h2.items() if np.isfinite(v["p_boot"])}, alpha)
        res["H2"] = {"by_detector": h2, "holm": p2,
                     "pass": all(d in p2 and p2[d]["reject"] and h2[d]["delta"] > 0 for d in dets)}
    # H3: content-matched control split
    cm = ev.get("content_matched_split")
    if cm and cm in test_splits:
        h3 = {d: boot(_paired(pred, d, "M3", "M1", [cm], ["clean"])) for d in dets}
        res["H3"] = {"split": cm, "by_detector": h3,
                     "pass": all(v["ci_low"] > 0 for v in h3.values() if np.isfinite(v["ci_low"]))}
    # H4: robustness of the gain and FPR drift under processing
    h4 = {}
    for d in dets:
        h4[d] = {}
        for v in ev["robustness_variants"]:
            if v not in set(pred.variant):
                continue
            gain = boot(_paired(pred, d, "M3", "M1", None, [v]))
            drift = {}
            for mdl in ("M1", "M3"):
                q = pred[(pred.detector == d) & (pred.model == mdl)]
                clean = q[(q.variant == "clean") & (q.label == 0)]
                proc = q[(q.variant == v) & (q.label == 0)]
                t = float(q.threshold.iloc[0]) if len(q) else np.nan
                drift[mdl] = rates_at_threshold(proc.label, proc.logit, t)["fpr"] - rates_at_threshold(clean.label, clean.logit, t)["fpr"]
            extra = drift.get("M3", np.nan) - drift.get("M1", np.nan)
            h4[d][v] = {"gain": gain, "fpr_drift_M1": drift.get("M1"), "fpr_drift_M3": drift.get("M3"),
                        "extra_fpr_drift": extra,
                        "pass": bool(gain["ci_low"] > 0 and extra <= ev["max_extra_fpr_drift"])}
    res["H4"] = {"by_detector": h4}
    # H5: per-scene gains (descriptive only)
    h5 = []
    for d in dets:
        j = _paired(pred, d, "M3", "M1", None, ["clean"])
        for sc, g in j.groupby("scene"):
            if g.label.nunique() < 2 or min((g.label == 0).sum(), (g.label == 1).sum()) < ev["min_scene_n"]:
                continue
            h5.append({"detector": d, "scene": sc, "n": len(g), "auroc_M1": auroc(g.label, g.logit_b),
                       "auroc_M3": auroc(g.label, g.logit), "delta": auroc(g.label, g.logit) - auroc(g.label, g.logit_b)})
    res["H5"] = h5
    # supplementary image-level DeLong (AUROC claims only)
    res["delong_M3_vs_M1"] = {}
    for d in dets:
        j = _paired(pred, d, "M3", "M1", None, ["clean"])
        if j.label.nunique() == 2:
            res["delong_M3_vs_M1"][d] = stats.delong_test(j.label, j.logit, j.logit_b)
    return res


def pilot_decision(h: dict[str, Any], pred: pd.DataFrame, detector: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered pilot go/no-go rule (methodology section 11)."""
    pc = cfg["pilot"]
    j31 = _paired(pred, detector, "M3", "M1", None, ["clean"])
    r1 = stats.cluster_bootstrap_delta(j31.label, j31.logit, j31.logit_b, j31.cl, stats.macro_auroc,
                                       cfg["evaluation"]["n_boot"], pc["alpha"], cfg["experiment"]["seed"])
    rule1 = r1["delta"] >= pc["min_delta_auroc"] and r1["ci_low"] > 0
    gain_m4 = None
    if "M4" in set(pred.model):
        j41 = _paired(pred, detector, "M4", "M1", None, ["clean"])
        gain_m4 = stats.macro_auroc(j41.label.to_numpy(), j41.logit.to_numpy(), j41.cl.to_numpy()) - \
            stats.macro_auroc(j41.label.to_numpy(), j41.logit_b.to_numpy(), j41.cl.to_numpy())
    rule2 = gain_m4 is not None and r1["delta"] > gain_m4
    v = pc["fpr_variant"]
    h4 = h.get("H4", {}).get("by_detector", {}).get(detector, {}).get(v, {})
    rule3 = h4.get("extra_fpr_drift", np.inf) <= pc["max_extra_fpr_drift"]
    q = pred[(pred.detector == detector) & (pred.model == "M2") & (pred.variant == "clean")]
    q = q[~((q.label == 1) & q.seen_family)]
    m2 = macro_over_generators(q.label.to_numpy(), q.logit.to_numpy(), q.cluster.to_numpy()) if len(q) else np.nan
    if rule1 and rule2 and rule3:
        decision = "GO"
    elif m2 >= pc["revise_m2_auroc"]:
        decision = "REVISE"
    else:
        decision = "STOP"
    return {"rule1": {"pass": bool(rule1), **r1}, "rule2": {"pass": bool(rule2), "gain_M3": r1["delta"], "gain_M4": gain_m4},
            "rule3": {"pass": bool(rule3), "extra_fpr_drift": h4.get("extra_fpr_drift")}, "m2_macro_auroc": m2,
            "decision": decision}


def ablation_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One-change-at-a-time ablations of M3 (methodology section 10)."""
    specs = [{"name": f"drop_{g}", "geo_cols": [c for c in FEATURE_NAMES if c not in GROUPS[g]]} for g in ("G1", "G2", "G3", "G4")]
    specs += [{"name": f"only_{g}", "geo_cols": GROUPS[g] + GROUPS["APP"]} for g in ("G1", "G2", "G3", "G4")]
    specs += [{"name": "no_flags", "geo_cols": [c for c in FEATURE_NAMES if not c.startswith("miss_")]}]
    specs += [{"name": "gbdt", "kind": "gbdt"}]
    specs += [{"name": "processing_aware_fit", "fit_variants": [v for v in cfg["data"]["variants"]]}]
    for alt in cfg["geometry"].get("alternative_feature_dirs", []):
        specs.append({"name": f"estimator_{Path(alt).name}", "feature_dir": alt})
    return specs
