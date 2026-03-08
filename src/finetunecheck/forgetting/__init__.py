"""Forgetting analysis — metrics, detection, and full analysis pipeline."""

from __future__ import annotations

from finetunecheck.forgetting.analyzer import ForgettingAnalyzer
from finetunecheck.forgetting.detector import ForgettingDetector
from finetunecheck.forgetting.metrics import (
    backward_transfer,
    capability_retention_rate,
    compute_roi_score,
    safety_alignment_retention,
    selective_forgetting_index,
)

__all__ = [
    "ForgettingAnalyzer",
    "ForgettingDetector",
    "backward_transfer",
    "capability_retention_rate",
    "compute_roi_score",
    "safety_alignment_retention",
    "selective_forgetting_index",
]
