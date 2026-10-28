"""Publication figures. Every figure is drawn from a saved result file, never from typed numbers.

Colour roles (validated categorical palette, fixed order, light mode for print):
detectors use slots 1-3 (blue, orange, aqua); real vs fake use blue vs orange; the base-only
model M1 is drawn in neutral grey as the reference, geometry models in blue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
MODEL_COLORS = {"M1": "#8a8984", "M2": PALETTE[2], "M3": PALETTE[0], "M4": PALETTE[1], "M5": PALETTE[6]}
MODEL_LABELS = {"M1": "M1 base", "M2": "M2 geometry", "M3": "M3 base+geometry", "M4": "M4 base+DINOv2", "M5": "M5 all"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
    })


def save(fig, out: str | Path) -> list[Path]:
    """Save as PDF (vector, for LaTeX) and PNG (for Word)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in (".pdf", ".png"):
        p = out.with_suffix(ext)
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def _det_colors(dets):
    return {d: PALETTE[i % 3] for i, d in enumerate(sorted(dets))}


# ----------------------------------------------------------------------------- results figures


def forest_per_generator(gen: pd.DataFrame, out: str | Path, variant: str = "clean") -> list[Path]:
    """Per-generator AUROC gain of M3 over M1, one row per generator, one dot per detector."""
    style()
    g = gen[gen.variant == variant]
    piv = g.pivot_table(index=["split", "generator", "seen_family", "detector"], columns="model", values="auroc").reset_index()
    if not {"M1", "M3"} <= set(piv.columns):
        raise ValueError("per-generator table needs M1 and M3")
    piv["delta"] = piv["M3"] - piv["M1"]
    rows = piv[["split", "generator", "seen_family"]].drop_duplicates().sort_values(["split", "generator"]).reset_index(drop=True)
    dets = sorted(piv.detector.unique())
    colors = _det_colors(dets)
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(rows) + 1.2))
    for k, d in enumerate(dets):
        sub = piv[piv.detector == d].merge(rows.reset_index(), on=["split", "generator", "seen_family"])
        ax.scatter(sub["delta"], sub["index"] + (k - (len(dets) - 1) / 2) * 0.18, s=22, color=colors[d], label=d, zorder=3)
    ax.axvline(0, color=INK2, lw=1)
    labels = [f"{r.generator}{' (seen family)' if r.seen_family else ''}  [{r.split.replace('test_', '')}]" for r in rows.itertuples()]
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC gain from adding geometry (M3 minus M1)")
    ax.legend(title="Base detector", loc="lower right")
    return save(fig, out)


def model_comparison(metrics: pd.DataFrame, out: str | Path, variant: str = "clean", subset: str = "unseen") -> list[Path]:
    """Macro AUROC of M1 to M5 per test split, small multiples by detector."""
    style()
    m = metrics[(metrics.variant == variant) & (metrics.subset == subset)]
    dets = sorted(m.detector.unique())
    splits = sorted(m.split.unique())
    models = [x for x in ["M1", "M2", "M3", "M4", "M5"] if x in set(m.model)]
    fig, axes = plt.subplots(1, len(dets), figsize=(3.0 * len(dets), 2.8), sharey=True, squeeze=False)
    w = 0.8 / len(models)
    for ax, d in zip(axes[0], dets):
        for i, mdl in enumerate(models):
            vals = [m[(m.detector == d) & (m.split == s) & (m.model == mdl)]["macro_auroc"].mean() for s in splits]
            ax.bar(np.arange(len(splits)) + (i - (len(models) - 1) / 2) * w, vals, w * 0.92, color=MODEL_COLORS[mdl],
                   label=MODEL_LABELS[mdl])
        ax.set_xticks(range(len(splits)), [s.replace("test_", "") for s in splits], rotation=30, ha="right")
        ax.set_title(d)
        ax.set_ylim(0.5, 1.0)
    axes[0][0].set_ylabel("Macro AUROC over unseen generators")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return save(fig, out)


def ablation_chart(abl: pd.DataFrame, out: str | Path) -> list[Path]:
    """Change in macro AUROC versus full M3 for each one-change ablation, with bootstrap CI."""
    style()
    dets = sorted(abl.detector.unique())
    colors = _det_colors(dets)
    names = list(dict.fromkeys(abl.ablation))
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(names) + 1.2))
    for k, d in enumerate(dets):
        s = abl[abl.detector == d].set_index("ablation").reindex(names)
        y = np.arange(len(names)) + (k - (len(dets) - 1) / 2) * 0.2
        ax.errorbar(s["delta"], y, xerr=[s["delta"] - s["ci_low"], s["ci_high"] - s["delta"]], fmt="o", ms=4,
                    color=colors[d], ecolor=colors[d], elinewidth=1.2, capsize=0, label=d)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Macro AUROC change versus full M3 (95% cluster bootstrap CI)")
    ax.legend(title="Base detector", loc="lower right")
    return save(fig, out)


def robustness_chart(metrics: pd.DataFrame, out: str | Path, variants: list[str]) -> list[Path]:
    """Left: macro AUROC of M1 and M3 across processing. Right: real-photo FPR at the fixed threshold."""
    style()
    m = metrics[(metrics.subset == "unseen") & metrics.model.isin(["M1", "M3"]) & metrics.variant.isin(variants)]
    agg = m.groupby(["detector", "model", "variant"])[["macro_auroc", "fpr_at_t"]].mean().reset_index()
    dets = sorted(agg.detector.unique())
    colors = _det_colors(dets)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    x = np.arange(len(variants))
    for d in dets:
        for mdl, ls in (("M1", "--"), ("M3", "-")):
            s = agg[(agg.detector == d) & (agg.model == mdl)].set_index("variant").reindex(variants)
            axes[0].plot(x, s["macro_auroc"], ls, color=colors[d], marker="o", ms=4, label=f"{d} {mdl}")
            axes[1].plot(x, 100 * s["fpr_at_t"], ls, color=colors[d], marker="o", ms=4)
    for ax in axes:
        ax.set_xticks(x, variants)
    axes[0].set_ylabel("Macro AUROC (unseen)")
    axes[1].set_ylabel("Real photos flagged (%) at fixed threshold")
    axes[0].legend(fontsize=7, ncol=1, loc="lower left")
    fig.tight_layout()
    fig.text(0.5, -0.04, "Dashed = base detector alone (M1); solid = base plus geometry (M3)", ha="center", color=INK2)
    return save(fig, out)


def feature_distributions(feat: pd.DataFrame, manifest: pd.DataFrame, out: str | Path, features: list[str]) -> list[Path]:
    """Real versus AI generated distribution of selected geometry features (test splits, clean)."""
    style()
    t = feat.merge(manifest[["image_id", "label", "split"]], on="image_id")
    t = t[t.split.str.startswith("test_")]
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 2.6))
    for ax, f in zip(np.atleast_1d(axes), features):
        data = [t.loc[t.label == lab, f].dropna().to_numpy() for lab in (0, 1)]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for body, c in zip(parts["bodies"], (PALETTE[0], PALETTE[1])):
            body.set_facecolor(c)
            body.set_alpha(0.55)
            body.set_edgecolor(c)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(INK2)
            parts[key].set_linewidth(1)
        ax.set_xticks([1, 2], ["real", "AI"])
        ax.set_title(f.replace("_", " "), fontsize=8)
    return save(fig, out)


def feature_shift(shift: pd.DataFrame, out: str | Path) -> list[Path]:
    """Standardised mean shift of each feature under processing, real and fake separately."""
    style()
    feats = list(dict.fromkeys(shift.feature))
    variants = sorted(shift.variant.unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(2.6 * len(variants), 0.22 * len(feats) + 1.2), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        for k, (lab, c) in enumerate(((0, PALETTE[0]), (1, PALETTE[1]))):
            s = shift[(shift.variant == v) & (shift.label == lab)].set_index("feature").reindex(feats)
            ax.barh(np.arange(len(feats)) + (k - 0.5) * 0.38, s["smd"], 0.36, color=c, label="real" if lab == 0 else "AI")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(v)
        ax.set_xlabel("Standardised mean shift")
    axes[0][0].set_yticks(range(len(feats)), [f.replace("_", " ") for f in feats], fontsize=7)
    axes[0][0].invert_yaxis()
    axes[0][-1].legend(loc="lower right")
    return save(fig, out)


def scene_heatmap(h5: pd.DataFrame, out: str | Path) -> list[Path]:
    """AUROC gain (M3 minus M1) per scene type and detector; diverging scale centred on zero."""
    style()
    piv = h5.pivot_table(index="scene", columns="detector", values="delta")
    lim = max(0.02, float(np.nanmax(np.abs(piv.to_numpy()))))
    fig, ax = plt.subplots(figsize=(1.2 * piv.shape[1] + 2.2, 0.45 * piv.shape[0] + 1))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [s.replace("_", " ") for s in piv.index])
    ax.grid(False)
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(im, ax=ax, label="AUROC gain")
    return save(fig, out)


def compute_cost(timing: pd.DataFrame, out: str | Path) -> list[Path]:
    """Seconds per image for each geometry model and each base detector (measured)."""
    style()
    t = timing.sort_values("sec_per_image")
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(t) + 1))
    ax.barh(t["component"], t["sec_per_image"], color=PALETTE[0], height=0.6)
    for i, v in enumerate(t["sec_per_image"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Seconds per image (measured)")
    return save(fig, out)


def roc_curves(pred: pd.DataFrame, detector: str, out: str | Path, variant: str = "clean") -> list[Path]:
    from sklearn.metrics import roc_curve

    style()
    q = pred[(pred.detector == detector) & (pred.variant == variant)]
    q = q[~((q.label == 1) & q.seen_family)]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for mdl in ["M1", "M2", "M3", "M4"]:
        s = q[q.model == mdl]
        if s.label.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(s.label, s.logit)
        ax.plot(fpr, tpr, color=MODEL_COLORS[mdl], lw=1.6, label=MODEL_LABELS[mdl])
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel("False positive rate (log scale)")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(detector)
    return save(fig, out)


# ----------------------------------------------------------------------------- qualitative


def _normal_rgb(n: np.ndarray) -> np.ndarray:
    return np.clip((n * np.array([1, -1, -1]) + 1) / 2, 0, 1)


def qualitative_panel(items: list[dict], out: str | Path, titles: list[str]) -> list[Path]:
    """Rows of [image | depth | depth normals | predicted normals | angle map | shading residual]."""
    style()
    cols = ["Image", "Depth", "Normals from depth", "Predicted normals", "Angle between normals", "Shading residual"]
    fig, axes = plt.subplots(len(items), len(cols), figsize=(1.55 * len(cols), 1.25 * len(items) + 0.3), squeeze=False)
    for r, (it, title) in enumerate(zip(items, titles)):
        maps = it["maps"]
        panels = [it["rgb"], it["depth"], _normal_rgb(maps["normal_depth"]), _normal_rgb(maps["normal_pred"]),
                  maps["theta"], maps.get("residual", np.full(it["depth"].shape, np.nan))]
        for c, img in enumerate(panels):
            ax = axes[r][c]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 1:
                ax.imshow(np.log(np.maximum(img, 1e-6)), cmap="Greys")
            elif c == 4:
                ax.imshow(img, cmap="Blues", vmin=0, vmax=45)
            elif c == 5:
                lim = np.nanpercentile(np.abs(img), 98) if np.isfinite(img).any() else 1
                ax.imshow(img, cmap="RdBu", vmin=-lim, vmax=lim)
            else:
                ax.imshow(img)
            if r == 0:
                ax.set_title(cols[c], fontsize=7)
        axes[r][0].set_ylabel(title, fontsize=7)
    return save(fig, out)


def placeholder(out: str | Path, title: str, what: str, size=(6.2, 2.6)) -> list[Path]:
    """Yellow placeholder used in the manuscript until the real figure is generated."""
    style()
    fig, ax = plt.subplots(figsize=size)
    ax.set_facecolor("#fff59d")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(INK2)
    ax.text(0.5, 0.62, "[xyz]  " + title, ha="center", va="center", fontsize=11, weight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.35, what, ha="center", va="center", fontsize=8, color=INK2, transform=ax.transAxes, wrap=True)
    return save(fig, out)
