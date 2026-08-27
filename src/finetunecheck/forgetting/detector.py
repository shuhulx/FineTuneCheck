"""Forgetting pattern detection and capability impact analysis."""

from __future__ import annotations

from statistics import mean

from finetunecheck.forgetting.metrics import REGRESSION_THRESHOLDS
from finetunecheck.models import ForgettingPattern


class ForgettingDetector:
    """Classify forgetting patterns and identify affected capabilities."""

    @staticmethod
    def classify_pattern(
        crr: dict[str, float | None],
        bwt: float | None,
        sfi: float | None,
    ) -> ForgettingPattern:
        """Classify the forgetting pattern based on retention metrics.

        Decision rules:
        - mean_crr > 0.95 -> MINIMAL (nearly everything retained)
        - mean_crr < 0.70 AND sfi < 0.1 -> CATASTROPHIC (uniform large drop)
        - sfi > 0.15 -> SELECTIVE (high variance: some capabilities hit, others fine)
        - else -> GRADUAL (moderate, roughly uniform degradation)

        Args:
            crr: Capability retention rates per category.
            bwt: Backward transfer value.
            sfi: Selective forgetting index.

        Returns:
            The classified ``ForgettingPattern``.
        """
        measured = [value for value in crr.values() if value is not None]
        if bwt is None or not measured:
            return ForgettingPattern.UNAVAILABLE
        worst = min(measured)
        mean_crr = mean(min(1.0, value) for value in measured)
        collapsed = worst < REGRESSION_THRESHOLDS["individual_collapse"]
        if bwt <= REGRESSION_THRESHOLDS["bwt_catastrophic"] or (
            collapsed and mean_crr < REGRESSION_THRESHOLDS["retention_critical"]
        ):
            return ForgettingPattern.CATASTROPHIC
        if collapsed or (sfi is not None and sfi >= REGRESSION_THRESHOLDS["sfi_selective"]):
            return ForgettingPattern.SELECTIVE
        if bwt <= REGRESSION_THRESHOLDS["bwt_gradual"]:
            return ForgettingPattern.GRADUAL
        if mean_crr >= REGRESSION_THRESHOLDS["retention_warning"]:
            return ForgettingPattern.MINIMAL
        return ForgettingPattern.GRADUAL

    @staticmethod
    def identify_affected_capabilities(
        crr: dict[str, float | None],
        threshold: float = 0.95,
    ) -> tuple[list[str], list[str]]:
        """Identify most affected and resilient capabilities.

        Args:
            crr: Capability retention rates per category.
            threshold: CRR below this value is considered a meaningful regression.

        Returns:
            Tuple of (most_affected, resilient) category name lists.
            ``most_affected`` is sorted by CRR ascending (worst first).
            ``resilient`` is sorted by CRR descending (best first).
        """
        affected: list[tuple[str, float]] = []
        resilient: list[tuple[str, float]] = []

        for cat, rate in crr.items():
            if rate is None:
                continue
            if rate < threshold:
                affected.append((cat, rate))
            else:
                resilient.append((cat, rate))

        affected.sort(key=lambda x: x[1])
        resilient.sort(key=lambda x: x[1], reverse=True)

        return (
            [cat for cat, _ in affected],
            [cat for cat, _ in resilient],
        )
