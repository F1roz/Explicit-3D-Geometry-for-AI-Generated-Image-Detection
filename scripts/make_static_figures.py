"""Figures that do not depend on experimental results, plus yellow placeholders for those that do.

* fig_architecture        method overview (drawn)
* fig_study_design        generator- and source-disjoint split design (drawn)
* fig_workflow            research workflow with the pilot gate (drawn, used in docs)
* fig_cues_synthetic      the four cue groups on two synthetic renders (computed)
* fig_synthetic_check     feature distributions on synthetic renders (computed; implementation check only)
* placeholders            every results figure, marked [xyz] until scripts/generate_figures.py is run

    python scripts/make_static_figures.py --out paper/figures
"""

from __future__ import annotations

import argparse

import matplotlib.patches as mpatches
import numpy as np

from _bootstrap import ROOT
from prism3d import viz
from prism3d.geometry import features as F
from prism3d.synthetic import render_scene

plt = viz.plt
BLUE, ORANGE, AQUA, GREY = viz.PALETTE[0], viz.PALETTE[1], viz.PALETTE[2], "#8a8984"


def box(ax, x, y, w, h, title, lines=(), color=GREY, fill=False, fs=8.5):
    p = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=1.3,
                                edgecolor=color, facecolor=(color if fill else "white"), alpha=1.0)
    if fill:
        p.set_alpha(0.13)
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=1.6,
                                             edgecolor=color, facecolor="none"))
    ax.add_patch(p)
    n = len(lines)
    step = min(0.045, 0.8 * h / (n + 1)) if n else 0
    ty = y + h / 2 + step * n / 2
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs, weight="bold", color=viz.INK)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, ty - step * (i + 1), ln, ha="center", va="center", fontsize=fs - 1.5, color=viz.INK2)


def arrow(ax, x0, y0, x1, y1, color="#52514e", text=None, tdx=0.0, tdy=0.012):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=10))
    if text:
        ax.text((x0 + x1) / 2 + tdx, (y0 + y1) / 2 + tdy, text, ha="center", va="bottom", fontsize=7, color=viz.INK2)


def _canvas(w, h):
    viz.style()
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def architecture(out):
    fig, ax = _canvas(10.0, 4.4)
    box(ax, 0.01, 0.42, 0.09, 0.16, "Input image", ["harmonised", "JPEG 95"], fs=8)
    box(ax, 0.15, 0.75, 0.27, 0.2, "Frozen base detector", ["B-Free, CO-SPY or UnivFD", "official code and weights"], ORANGE, True, fs=8)
    box(ax, 0.15, 0.36, 0.27, 0.3, "Frozen geometry models", ["MoGe-2: points, depth, intrinsics", "Depth Anything V2: second depth",
                                                             "DSINE: normals from pixels", "Intrinsic decomposition: shading"], BLUE, True, fs=8)
    box(ax, 0.15, 0.05, 0.27, 0.2, "Frozen DINOv2-L (control)", ["CLS feature, PCA to", "the geometry dimension"], GREY, True, fs=8)
    box(ax, 0.49, 0.36, 0.22, 0.3, "Explicit cues", ["G1 depth and normal agreement", "G2 shading and geometry", "G3 perspective",
                                                      "G4 occlusion and edges"], BLUE, fs=8)
    box(ax, 0.775, 0.36, 0.205, 0.3, "Logistic stacker", ["fit on the fit split", "C by generator grouped CV", "temperature, threshold",
                                                          "on the validation split"], viz.INK2, fs=8)
    arrow(ax, 0.1, 0.5, 0.15, 0.85)
    arrow(ax, 0.1, 0.5, 0.15, 0.51)
    arrow(ax, 0.1, 0.5, 0.15, 0.15)
    arrow(ax, 0.42, 0.51, 0.49, 0.51)
    arrow(ax, 0.42, 0.85, 0.85, 0.66)
    ax.text(0.66, 0.79, "base logit", ha="center", fontsize=7, color=viz.INK2)
    arrow(ax, 0.71, 0.51, 0.78, 0.51)
    ax.text(0.745, 0.53, "26 features", ha="center", fontsize=6.5, color=viz.INK2)
    arrow(ax, 0.42, 0.15, 0.85, 0.36)
    ax.text(0.6, 0.18, "DINOv2 PCs (M4, M5 only)", ha="center", fontsize=7, color=viz.INK2)
    ax.text(0.887, 0.29, "calibrated score", ha="center", fontsize=7.5, color=viz.INK, weight="bold")
    ax.text(0.887, 0.245, "M1 base | M2 geometry", ha="center", fontsize=6.5, color=viz.INK2)
    ax.text(0.887, 0.205, "M3 base+geometry", ha="center", fontsize=6.5, color=viz.INK2)
    ax.text(0.887, 0.165, "M4 base+DINOv2 | M5 all", ha="center", fontsize=6.5, color=viz.INK2)
    return viz.save(fig, out)


def study_design(out):
    fig, ax = _canvas(10.0, 4.2)
    box(ax, 0.01, 0.1, 0.26, 0.8, "Fit and validation pool", ["AI: Co-Spy-Bench images from cc3m", "captions, 10 open models (SDXL family,",
                                                              "SD 1.5 and 2.1, LDM, tiny-sd, ...)", "Real: Flickr30k photos", "",
                                                              "Stacker, C, temperature and", "threshold are fitted here only"], viz.INK2, fs=8)
    tests = [
        ("Test A: unseen open models", ["FLUX.1 dev and schnell, SD3 medium, Playground", "v2.5, PixArt (Co-Spy-Bench); real: RAISE"], BLUE),
        ("Test B: content matched", ["Synthbuster (9 models), FLUX and SD 3.5 made", "from RAISE captions; real: RAISE"], AQUA),
        ("Test C: newest commercial", ["OpenFake test split: GPT Image 1, Imagen,", "Midjourney v6 and v7, Ideogram, ...; real: LAION"], ORANGE),
        ("Test D: in the wild", ["Chameleon (images that fooled annotators);", "fallback: AIGIBench social media subsets"], viz.PALETTE[6]),
    ]
    for i, (t, ln, c) in enumerate(tests):
        box(ax, 0.36, 0.73 - i * 0.215, 0.38, 0.18, t, ln, c, True, fs=8)
    ax.annotate("", xy=(0.36, 0.5), xytext=(0.27, 0.5), arrowprops=dict(arrowstyle="-|>", color=viz.INK2, lw=1.2, linestyle="--"))
    ax.text(0.315, 0.53, "frozen", ha="center", fontsize=7, color=viz.INK2)
    ax.text(0.315, 0.455, "no refit", ha="center", fontsize=7, color=viz.INK2)
    rules = ["no shared generator", "no shared real source", "no shared caption", "near duplicates removed",
             "no real image that a base", "detector was trained on"]
    box(ax, 0.79, 0.2, 0.19, 0.6, "Leakage rules", rules, viz.INK2, fs=8)
    return viz.save(fig, out)


def workflow(out):
    fig, ax = _canvas(7.2, 2.4)
    steps = ["Download", "Prepare\nsplits", "Geometry\nfeatures", "Base\nscores", "Controls", "Pilot\nrule", "Full study", "Tables and\nfigures"]
    w = 0.092
    for i, s in enumerate(steps):
        x = 0.015 + i * 0.123
        c = ORANGE if s.startswith("Pilot") else (BLUE if i in (2, 6) else GREY)
        box(ax, x, 0.45, w, 0.3, s, (), c, s.startswith("Pilot") or i in (2, 6), fs=7)
        if i:
            arrow(ax, x - 0.026, 0.6, x - 0.005, 0.6)
    ax.text(0.015 + 5 * 0.123 + w / 2, 0.33, "GO / REVISE / STOP", ha="center", fontsize=7, color=viz.INK2)
    ax.text(0.5, 0.12, "Each stage writes its config, environment and outputs to the run folder and can resume.", ha="center",
            fontsize=7.5, color=viz.INK2)
    return viz.save(fig, out)


def _gi(s):
    d = s["depth"].astype(float)
    return F.GeometryInputs(rgb=s["rgb"], points=F.backproject(d, s["K"]), depth=d, K=s["K"], mask=s["mask"],
                            normal_pred=s["normal_pred"], disp_da=(1 / d) * (1 + 0.03 * s["disp_noise"]),
                            shading=s["shading"], albedo=s["albedo"])


def cues_synthetic(out):
    viz.style()
    rng = np.random.default_rng(11)
    scenes = [("Consistent render", render_scene(rng, True, 240, 320)), ("Inconsistent render", render_scene(rng, False, 240, 320))]
    cols = ["Image", "G1: angle between normals", "G2: shading residual", "G3: lines and vanishing points", "G4: depth edges vs image edges"]
    fig, axes = plt.subplots(2, 5, figsize=(9.0, 3.6))
    for r, (name, s) in enumerate(scenes):
        f, m = F.compute_features(_gi(s), None, return_maps=True)
        a = axes[r]
        a[0].imshow(s["rgb"])
        a[1].imshow(m["theta"], cmap="Blues", vmin=0, vmax=45)
        lim = np.nanpercentile(np.abs(m["residual"]), 98)
        a[2].imshow(m["residual"], cmap="RdBu", vmin=-lim, vmax=lim)
        a[3].imshow(s["rgb"], alpha=0.45)
        seg, assign = m["lines"], m.get("vp_assign", np.full(len(m["lines"]), -1))
        for (x1, y1, x2, y2), k in zip(seg, assign):
            a[3].plot([x1, x2], [y1, y2], color=viz.PALETTE[k] if k >= 0 else GREY, lw=1.0)
        ov = np.zeros(s["rgb"].shape[:2] + (3,))
        ov[m["edges_img_lo"]] = [0.85, 0.85, 0.85]
        ov[m["edges_depth_lo"] & m["edges_img_lo"]] = [0.16, 0.47, 0.84]
        ov[m["edges_depth_lo"] & ~m["edges_img_lo"]] = [0.92, 0.41, 0.2]
        a[4].imshow(ov)
        txt = [f"mean angle {f['g1_theta_mean']:.1f} deg", f"median |r| {f['g2_res_median']:.3f}",
               f"inlier ratio {f['g3_inlier_ratio']:.2f}", f"A_occ {f['g4_aocc_lo']:.2f}"]
        for c in range(5):
            a[c].set_xticks([])
            a[c].set_yticks([])
            a[c].grid(False)
            if r == 0:
                a[c].set_title(cols[c], fontsize=7.5)
            if c:
                a[c].set_xlabel(txt[c - 1], fontsize=7)
        a[0].set_ylabel(name, fontsize=8)
    fig.tight_layout()
    return viz.save(fig, out)


def synthetic_check(out, n=30):
    viz.style()
    rng = np.random.default_rng(5)
    feats = {True: [], False: []}
    for i in range(2 * n):
        c = i % 2 == 0
        feats[c].append(F.compute_features(_gi(render_scene(rng, c))))
    show = [("g1_theta_mean", "G1 mean angle (deg)"), ("g2_res_median", "G2 median |residual|"),
            ("g2_light_dispersion", "G2 light dispersion"), ("g4_aocc_lo", "G4 A_occ (low threshold)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, (k, lab) in zip(axes, show):
        data = [np.array([f[k] for f in feats[c]], float) for c in (True, False)]
        data = [d[np.isfinite(d)] for d in data]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for b, col in zip(parts["bodies"], (BLUE, ORANGE)):
            b.set_facecolor(col)
            b.set_edgecolor(col)
            b.set_alpha(0.55)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(viz.INK2)
            parts[key].set_linewidth(1)
        ax.set_xticks([1, 2], ["consistent", "inconsistent"], fontsize=7)
        ax.set_title(lab, fontsize=8)
    fig.tight_layout()
    return viz.save(fig, out)


PLACEHOLDERS = {
    "fig_model_comparison": ("Model comparison", "Macro AUROC of M1 to M5 per test set and base detector\n(scripts/generate_figures.py -> fig_model_comparison)"),
    "fig_per_generator_gain": ("Per generator gain", "AUROC gain of M3 over M1 for every test generator and base detector\n(fig_per_generator_gain)"),
    "fig_robustness": ("Robustness to processing", "Macro AUROC and real photo FPR under JPEG 75, half resize and screenshot chain\n(fig_robustness)"),
    "fig_ablation": ("Ablation study", "Change in macro AUROC versus full M3 for each one change ablation, 95% CI\n(fig_ablation)"),
    "fig_feature_distributions": ("Feature distributions", "Real versus AI generated distributions of six geometry features on the test sets\n(fig_feature_distributions)"),
    "fig_feature_shift": ("Feature shift under processing", "Standardised mean shift of each feature under JPEG and resizing, real and fake\n(fig_feature_shift)"),
    "fig_scene_gain": ("Gain by scene type", "AUROC gain of M3 over M1 per scene label and base detector\n(fig_scene_gain)"),
    "fig_qualitative": ("Qualitative examples", "Real photos and AI images from Test B with depth, normals, angle and residual maps\n(fig_qualitative)"),
    "fig_compute_cost": ("Compute cost", "Measured seconds per image for each geometry model and base detector\n(fig_compute_cost)"),
    "fig_roc_bfree": ("ROC curves", "ROC of M1 to M4 for B-Free on unseen generators, log FPR axis\n(fig_roc_bfree)"),
    "fig_roc_cospy": ("ROC curves", "ROC of M1 to M4 for CO-SPY\n(fig_roc_cospy)"),
    "fig_roc_univfd": ("ROC curves", "ROC of M1 to M4 for UnivFD\n(fig_roc_univfd)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--no-placeholders", action="store_true")
    a = ap.parse_args()
    from pathlib import Path

    out = Path(a.out)
    made = []
    made += architecture(out / "fig_architecture")
    made += study_design(out / "fig_study_design")
    made += workflow(out / "fig_workflow")
    made += cues_synthetic(out / "fig_cues_synthetic")
    made += synthetic_check(out / "fig_synthetic_check")
    if not a.no_placeholders:
        for name, (t, w) in PLACEHOLDERS.items():
            if not (out / f"{name}.pdf").exists() or (out / f"{name}.placeholder").exists():
                made += viz.placeholder(out / name, t, w)
                (out / f"{name}.placeholder").write_text("placeholder: replace by running scripts/generate_figures.py\n")
    for m in made:
        print(m)


if __name__ == "__main__":
    main()
