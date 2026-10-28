"""The calibrated stacker that combines a base detector score with extra features.

Protocol (identical for every model M1-M5, so comparisons are fair):

1. Design matrix = [base logit] + chosen feature block(s). Missing values are imputed with the
   FIT-split median; features are standardised with FIT-split statistics.
2. The L2 strength C is chosen by cross-validation on the fit split. Fake images are grouped by
   generator so each fold holds out whole generators (selection rewards transfer to new
   generators); real images are assigned to folds at random by prompt group.
3. The model is refitted on the whole fit split with the chosen C.
4. Temperature scaling and the operating threshold are fixed on the VALIDATION split; the
   threshold gives ``target_fpr`` on validation real images.
5. Nothing is ever fitted on a test split.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import auroc


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def make_folds(df: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generator-grouped folds for fakes, prompt-grouped random folds for reals."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(df), dtype=int)
    y = df["label"].to_numpy()
    gens = np.asarray(df.loc[y == 1, "cluster"].unique(), dtype=object)
    if len(gens) < n_splits:
        raise ValueError(f"Need at least {n_splits} fake generators in the fit split for grouped CV, found {len(gens)}")
    rng.shuffle(gens)
    gen_fold = {g: i % n_splits for i, g in enumerate(gens)}
    fold[y == 1] = df.loc[y == 1, "cluster"].map(gen_fold).to_numpy()
    groups = np.asarray(df.loc[y == 0, "prompt_id"].unique(), dtype=object)
    rng.shuffle(groups)
    grp_fold = {g: i % n_splits for i, g in enumerate(groups)}
    fold[y == 0] = df.loc[y == 0, "prompt_id"].map(grp_fold).to_numpy()
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(n_splits)]


def _build_model(kind: str, C: float, seed: int) -> Pipeline:
    if kind == "logreg":
        clf = LogisticRegression(C=C, max_iter=5000)
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)])
    if kind == "gbdt":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                             random_state=seed)
        return Pipeline([("clf", clf)])  # handles NaN natively
    raise ValueError(f"Unknown stacker kind {kind}")


@dataclass
class Stacker:
    columns: list[str]
    kind: str = "logreg"
    C_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    n_splits: int = 5
    target_fpr: float = 0.05
    seed: int = 0
    model: Pipeline | None = None
    C: float | None = None
    temperature: float = 1.0
    threshold: float = 0.0
    cv_scores: dict[float, float] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for stacker: {missing[:5]}")
        return df[self.columns].to_numpy(dtype=np.float64, copy=True)

    def fit(self, fit_df: pd.DataFrame, val_df: pd.DataFrame) -> "Stacker":
        if set(fit_df["label"].unique()) != {0, 1} or set(val_df["label"].unique()) != {0, 1}:
            raise ValueError("fit and val splits must both contain real and fake images")
        X, y = self._X(fit_df), fit_df["label"].to_numpy()
        all_nan = np.all(np.isnan(X), axis=0)
        if all_nan.any():
            X[:, all_nan] = 0.0
        self._all_nan = all_nan
        grid = self.C_grid if self.kind == "logreg" else (1.0,)
        if len(grid) > 1:
            folds = make_folds(fit_df.reset_index(drop=True), self.n_splits, self.seed)
            for C in grid:
                scores = []
                for tr, te in folds:
                    if len(np.unique(y[te])) < 2:
                        continue
                    m = _build_model(self.kind, C, self.seed).fit(X[tr], y[tr])
                    scores.append(auroc(y[te], m.decision_function(X[te])))
                self.cv_scores[C] = float(np.mean(scores))
            best = max(self.cv_scores.values())
            # among C values within 0.001 of the best, prefer the strongest regularisation
            self.C = min(C for C, s in self.cv_scores.items() if s >= best - 1e-3)
        else:
            self.C = grid[0]
        self.model = _build_model(self.kind, self.C, self.seed).fit(X, y)
        # temperature on validation
        z_val = self._raw_logit(val_df)
        yv = val_df["label"].to_numpy()

        def nll(log_t: float) -> float:
            p = 1 / (1 + np.exp(-z_val / np.exp(log_t)))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))

        self.temperature = float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))
        cal_val = z_val / self.temperature
        self.threshold = float(np.quantile(cal_val[yv == 0], 1 - self.target_fpr))
        return self

    def _raw_logit(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Stacker is not fitted")
        X = self._X(df)
        X[:, self._all_nan] = 0.0
        if self.kind == "logreg":
            return self.model.decision_function(X)
        return logit(self.model.predict_proba(X)[:, 1])

    def predict_logit(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated logit (higher = more likely AI generated)."""
        return self._raw_logit(df) / self.temperature

    def coefficients(self) -> dict[str, float]:
        """Standardised logistic coefficients (only for kind='logreg')."""
        if self.kind != "logreg" or self.model is None:
            return {}
        coef = self.model.named_steps["clf"].coef_.ravel()
        return dict(zip(self.columns, map(float, coef)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "Stacker":
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> dict[str, Any]:
        return {"kind": self.kind, "C": self.C, "cv_auroc": self.cv_scores, "temperature": self.temperature,
                "threshold_logit": self.threshold, "n_columns": len(self.columns)}
