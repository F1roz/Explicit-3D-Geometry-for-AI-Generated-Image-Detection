"""Statistical tests used in the paper.

* :func:`cluster_bootstrap_delta`  primary test. Resamples generators (fake clusters) and real
  sources (real clusters) with replacement, then images within each, and recomputes a paired
  metric difference on the same resample. The generator is the unit of generalisation, so this
  interval reflects uncertainty about *new generators*, not just new images.
* :func:`delong_test`  image-level paired AUROC test (DeLong et al., 1988; fast algorithm of
  Sun and Xu, 2014). Supplementary, AUROC claims only.
* :func:`mcnemar_test` exact paired test for accuracy at a fixed threshold. Accuracy claims only.
* :func:`holm`  Holm-Bonferroni step-down correction for families of hypotheses.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as st

from .metrics import auroc, macro_over_generators

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def macro_auroc(y: np.ndarray, s: np.ndarray, clusters: np.ndarray) -> float:
    return macro_over_generators(y, s, clusters, auroc)


def pooled_auroc(y: np.ndarray, s: np.ndarray, clusters: np.ndarray) -> float:  # noqa: ARG001
    return auroc(y, s)


def _cluster_resample_index(y: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage resample. Returns (row indices, relabelled cluster ids so repeats stay distinct)."""
    idx_parts, cl_parts = [], []
    for label in (0, 1):
        lab_mask = y == label
        uniq = np.unique(clusters[lab_mask])
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        for k, c in enumerate(chosen):
            members = np.flatnonzero(lab_mask & (clusters == c))
            pick = rng.choice(members, size=len(members), replace=True)
            idx_parts.append(pick)
            cl_parts.append(np.full(len(pick), f"{c}#{label}#{k}", dtype=object))
    return np.concatenate(idx_parts), np.concatenate(cl_parts)


def cluster_bootstrap_delta(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    clusters: np.ndarray,
    metric: MetricFn = macro_auroc,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired difference metric(A) - metric(B) with a generator-level cluster bootstrap CI.

    Args:
        y: labels (1 = fake).
        score_a, score_b: scores of the two models on the SAME images.
        clusters: generator name for fakes, real source for reals.
        metric: f(y, s, clusters) -> float. Default: macro AUROC over generators.
    Returns:
        dict with point estimate, CI bounds, a two-sided bootstrap p-value and the number of
        fake and real clusters (a CI from fewer than 5 fake clusters is only indicative).
    """
    y = np.asarray(y).astype(int)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    clusters = np.asarray(clusters).astype(object)
    rng = np.random.default_rng(seed)
    point = metric(y, a, clusters) - metric(y, b, clusters)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx, cl = _cluster_resample_index(y, clusters, rng)
        deltas[i] = metric(y[idx], a[idx], cl) - metric(y[idx], b[idx], cl)
    deltas = deltas[np.isfinite(deltas)]
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_boot": float(min(1.0, p)),
        "n_boot": int(len(deltas)),
        "n_fake_clusters": int(len(np.unique(clusters[y == 1]))),
        "n_real_clusters": int(len(np.unique(clusters[y == 0]))),
    }


def image_bootstrap_delta(y, score_a, score_b, metric=auroc, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Secondary paired bootstrap that resamples images only (stratified by label)."""
    y = np.asarray(y).astype(int)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = metric(y, a) - metric(y, b)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi)}


# ----------------------------------------------------------------------------- DeLong


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y, score_a, score_b) -> dict[str, float]:
    """Two-sided DeLong test for the difference of two correlated AUROCs."""
    y = np.asarray(y).astype(int)
    preds = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * st.norm.sf(abs(z))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": float(diff), "z": float(z), "p": float(p)}


def mcnemar_test(y, pred_a, pred_b) -> dict[str, float]:
    """Exact McNemar test on paired correctness at a fixed threshold."""
    y = np.asarray(y).astype(int)
    ca = np.asarray(pred_a).astype(int) == y
    cb = np.asarray(pred_b).astype(int) == y
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * st.binom.cdf(min(b, c), n, 0.5)))
    return {"a_only_correct": b, "b_only_correct": c, "p": p}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """Holm-Bonferroni: sort p ascending, compare p_(i) to alpha / (m - i), stop at first failure."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    still = True
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        reject = still and p <= alpha / (m - i)
        if not reject:
            still = False
        out[k] = {"p": float(p), "p_holm": float(adj), "reject": bool(reject)}
    return out
