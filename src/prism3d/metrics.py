"""Detection metrics. Label 1 = AI generated (positive class), score higher = more likely fake."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _check(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have the same shape")
    return y, s


def auroc(y, s) -> float:
    """Probability that a random fake scores higher than a random real (ties count 1/2)."""
    y, s = _check(y, s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def average_precision(y, s) -> float:
    y, s = _check(y, s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def tpr_at_fpr(y, s, fpr_target: float = 0.01) -> float:
    """True positive rate at a fixed false positive rate, linearly interpolated on the ROC."""
    y, s = _check(y, s)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def rates_at_threshold(y, s, threshold: float) -> dict[str, float]:
    """FPR (real photos flagged), TPR (fakes caught), accuracies at a fixed threshold."""
    y, s = _check(y, s)
    pred = s >= threshold
    real, fake = y == 0, y == 1
    fpr = float(pred[real].mean()) if real.any() else float("nan")
    tpr = float(pred[fake].mean()) if fake.any() else float("nan")
    return {
        "fpr": fpr,
        "tpr": tpr,
        "acc_real": 1 - fpr if real.any() else float("nan"),
        "acc_fake": tpr,
        "bacc": float(np.nanmean([1 - fpr, tpr])),
    }


def ece(y, p, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width probability bins.

    ECE = sum_m |B_m| / n * |acc(B_m) - conf(B_m)|, where conf is the mean predicted
    probability of the positive class and acc is the observed fraction of positives.
    """
    y, p = _check(y, p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def all_metrics(y, logit, threshold_logit: float | None = None) -> dict[str, float]:
    """Every metric reported in the paper for one set of predictions."""
    y, logit = _check(y, logit)
    p = 1.0 / (1.0 + np.exp(-logit))
    out = {
        "n": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "auroc": auroc(y, logit),
        "ap": average_precision(y, logit),
        "tpr_at_1fpr": tpr_at_fpr(y, logit, 0.01),
        "ece": ece(y, p),
    }
    if threshold_logit is not None:
        out.update({f"{k}_at_t": v for k, v in rates_at_threshold(y, logit, threshold_logit).items()})
    return out


def macro_over_generators(y, s, fake_groups, metric=auroc) -> float:
    """Average a metric over generators: each generator's fakes vs ALL reals of the test set.

    This gives every generator equal weight regardless of how many images it contributed.
    """
    y, s = _check(y, s)
    fake_groups = np.asarray(fake_groups)
    real = y == 0
    vals = []
    for g in np.unique(fake_groups[y == 1]):
        m = real | ((y == 1) & (fake_groups == g))
        vals.append(metric(y[m], s[m]))
    return float(np.nanmean(vals)) if vals else float("nan")
