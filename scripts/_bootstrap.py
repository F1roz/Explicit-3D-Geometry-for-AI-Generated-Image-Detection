"""Shared CLI setup: put ``src/`` on the path and parse --config / --set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from prism3d.config import load_config  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="YAML config file")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values, e.g. experiment.seed=1")
    return p


def config(args) -> dict:
    return load_config(args.config, args.set)
