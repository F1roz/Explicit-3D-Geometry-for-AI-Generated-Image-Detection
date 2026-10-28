"""Small shared helpers: seeding, logging, environment capture, JSON IO."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional for the numerical core
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "prism3d", log_file: str | Path | None = None) -> logging.Logger:
    """Return a logger that writes to stderr and optionally to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable")


def environment_info() -> dict[str, Any]:
    """Record software and hardware details next to every result file."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for mod in ("sklearn", "scipy", "pandas", "cv2", "torch", "transformers", "diffusers"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[mod] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


@contextmanager
def timer(store: dict[str, float], key: str) -> Iterator[None]:
    """Accumulate wall-clock seconds under ``store[key]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0
