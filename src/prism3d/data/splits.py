"""Leakage-safe split assignment.

A split is defined in the config as a list of *selectors*. A selector picks images by
dataset, label, generator, caption source and real source. Every image may belong to at
most one split. After assignment we verify the two rules that make the study valid:

1. No fake generator used in ``fit`` or ``val`` appears in any test split.
2. No real-image source used in ``fit`` or ``val`` appears in any test split.

Generators of the same family as a base detector's training data are tagged
``seen_family`` so the headline claim can be restricted to unseen generators.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any

import numpy as np
import pandas as pd

FIT_SPLITS = ("fit", "val")


class LeakageError(RuntimeError):
    """Raised when a split rule would let training information reach a test set."""


def _match_any(value: str, patterns: list[str] | str | None) -> bool:
    if patterns is None or patterns == "*":
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(fnmatch.fnmatch(value, p.lower()) for p in patterns)


def select(df: pd.DataFrame, sel: dict[str, Any]) -> pd.Series:
    """Boolean mask of rows matching one selector."""
    m = pd.Series(True, index=df.index)
    if "dataset" in sel:
        m &= df["dataset"].apply(lambda v: _match_any(v, sel["dataset"]))
    if "label" in sel:
        m &= df["label"] == int(sel["label"])
    if "generators" in sel:
        m &= df["generator"].apply(lambda v: _match_any(v, sel["generators"]))
    if "exclude_generators" in sel:
        m &= ~df["generator"].apply(lambda v: _match_any(v, sel["exclude_generators"]) if v else False)
    if "caption_sources" in sel:
        m &= df["caption_source"].apply(lambda v: _match_any(v, sel["caption_sources"]))
    if "real_sources" in sel:
        m &= df["real_source"].apply(lambda v: _match_any(v, sel["real_sources"]))
    return m


def _cap_per_group(df: pd.DataFrame, idx: pd.Index, cap: int | None, rng: np.random.Generator) -> pd.Index:
    """Keep at most ``cap`` images per cluster, sampling whole prompt groups when possible."""
    if not cap:
        return idx
    keep = []
    for _, grp in df.loc[idx].groupby("cluster"):
        if len(grp) <= cap:
            keep.extend(grp.index)
        else:
            keep.extend(rng.choice(grp.index.to_numpy(), size=cap, replace=False))
    return pd.Index(sorted(keep))


def assign_splits(df: pd.DataFrame, split_cfg: dict[str, Any], seed: int = 0) -> pd.DataFrame:
    """Return a copy of ``df`` with ``split`` ('fit', 'val', 'test' or '') and ``eval_sets``.

    ``split_cfg`` has ``fit_val`` (selectors, ``val_fraction``, optional ``max_per_cluster``)
    and ``tests`` (name -> selectors, optional ``max_per_cluster``). Test sets may share
    images with each other (e.g. two tests that use the same real photographs); an image's
    test-set memberships are stored ``|``-separated in ``eval_sets``. Fit/val images can never
    be in a test set.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = ""
    out["eval_sets"] = ""

    def _mask(selectors: list[dict]) -> pd.Series:
        m = pd.Series(False, index=out.index)
        for sel in selectors:
            m |= select(out, sel)
        return m

    fv = split_cfg["fit_val"]
    fv_idx = _cap_per_group(out, out.index[_mask(fv["selectors"])], fv.get("max_per_cluster"), rng)
    out.loc[fv_idx, "split"] = "fitval"
    for name, spec in split_cfg["tests"].items():
        m = _mask(spec["selectors"])
        clash = m & out["split"].isin(["fitval"])
        if clash.any():
            raise LeakageError(f"Test set '{name}' selects {int(clash.sum())} images already used for fit/val")
        # reuse an image already sampled for another test set before sampling new ones,
        # so shared real photographs stay identical across test sets
        idx = out.index[m]
        already = idx[out.loc[idx, "eval_sets"] != ""]
        fresh = idx[out.loc[idx, "eval_sets"] == ""]
        fresh = _cap_per_group(out, fresh, spec.get("max_per_cluster"), rng)
        chosen = already.union(fresh)
        out.loc[chosen, "eval_sets"] = [f"{e}|{name}" if e else name for e in out.loc[chosen, "eval_sets"]]
        out.loc[chosen, "split"] = "test"

    # fit / val by prompt group so captions never straddle the two
    fv_idx = out.index[out["split"] == "fitval"]
    groups = np.asarray(out.loc[fv_idx, "prompt_id"].unique(), dtype=object)
    rng.shuffle(groups)
    n_val = int(round(len(groups) * float(fv.get("val_fraction", 0.2))))
    val_groups = set(groups[:n_val])
    out.loc[fv_idx, "split"] = np.where(out.loc[fv_idx, "prompt_id"].isin(val_groups), "val", "fit")

    out["seen_family"] = out["generator"].apply(lambda g: is_seen_family(g, split_cfg.get("seen_family_patterns", [])))
    check_leakage(out)
    return out


def is_seen_family(generator: str, patterns: list[str]) -> bool:
    """True if a fake generator matches a family some base detector was trained on."""
    if not generator:
        return False
    return any(re.search(p, generator) for p in patterns)


def check_leakage(df: pd.DataFrame) -> dict[str, Any]:
    """Raise :class:`LeakageError` if generators, real sources or prompts cross fit/test."""
    fit = df[df["split"].isin(FIT_SPLITS)]
    test = df[df["split"] == "test"]
    report: dict[str, Any] = {}
    gen_overlap = set(fit.loc[fit.label == 1, "generator"]) & set(test.loc[test.label == 1, "generator"])
    src_overlap = set(fit.loc[fit.label == 0, "real_source"]) & set(test.loc[test.label == 0, "real_source"])
    prompt_overlap = set(fit["prompt_id"]) & set(test["prompt_id"])
    fv_prompt_overlap = set(df.loc[df.split == "fit", "prompt_id"]) & set(df.loc[df.split == "val", "prompt_id"])
    report.update(generator_overlap=sorted(gen_overlap), real_source_overlap=sorted(src_overlap),
                  prompt_overlap=len(prompt_overlap), fit_val_prompt_overlap=len(fv_prompt_overlap))
    problems = []
    if gen_overlap:
        problems.append(f"generators in both fit/val and test: {sorted(gen_overlap)}")
    if src_overlap:
        problems.append(f"real sources in both fit/val and test: {sorted(src_overlap)}")
    if prompt_overlap:
        problems.append(f"{len(prompt_overlap)} prompt ids in both fit/val and test")
    if fv_prompt_overlap:
        problems.append(f"{len(fv_prompt_overlap)} prompt ids in both fit and val")
    if problems:
        raise LeakageError("; ".join(problems))
    return report


def explode_eval_sets(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (image, test set); the test-set name goes into ``eval_set``."""
    t = df[df["split"] == "test"].copy()
    t["eval_set"] = t["eval_sets"].str.split("|")
    return t.explode("eval_set").reset_index(drop=True)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per split or test set, label and cluster, for the dataset table in the paper."""
    fv = df[df["split"].isin(FIT_SPLITS)].assign(set=lambda x: x["split"])
    te = explode_eval_sets(df).assign(set=lambda x: "test:" + x["eval_set"])
    used = pd.concat([fv, te], ignore_index=True)
    return used.groupby(["set", "label", "cluster"]).size().rename("n").reset_index().sort_values(["set", "label", "cluster"])
