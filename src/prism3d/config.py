"""YAML configuration loading with command-line overrides.

Configs live in ``configs/``. Any key can be overridden on the command line with
``--set key.subkey=value`` (value parsed as YAML), which keeps every run reproducible
from its saved ``config_used.yaml``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL = ("experiment", "paths", "data", "splits", "geometry", "detectors", "stacking", "evaluation")


def _deep_update(base: dict, upd: dict) -> dict:
    """Recursive merge; a child dict containing ``__replace__: true`` replaces instead of merging."""
    for k, v in upd.items():
        if isinstance(v, dict) and v.get("__replace__"):
            base[k] = {kk: vv for kk, vv in v.items() if kk != "__replace__"}
        elif isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, apply ``inherit`` and ``key=value`` overrides, validate."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = _deep_update(copy.deepcopy(base), cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise KeyError(f"Config {path} is missing sections: {missing}")
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
