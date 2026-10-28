"""Frozen base detectors. Every detector writes one CSV per image variant with columns
``image_id, logit`` (higher = more likely AI generated)."""

from .base import DETECTORS, get_detector  # noqa: F401
