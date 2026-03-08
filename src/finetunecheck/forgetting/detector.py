"""Forgetting pattern detection and capability impact analysis."""

from __future__ import annotations

from finetunecheck.models import ForgettingPattern


class ForgettingDetector:
    """Classify forgetting patterns and identify affected capabilities."""

    @staticmethod
    def classify_pattern(
        crr: dict[str, float],
        bwt: float,
        sfi: float,
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
        if not crr:
            return ForgettingPattern.MINIMAL

        mean_crr = sum(crr.values()) / len(crr)

        if mean_crr > 0.95:
            return ForgettingPattern.MINIMAL

        if mean_crr < 0.70 and sfi < 0.1:
            return ForgettingPattern.CATASTROPHIC

        if sfi > 0.15:
            return ForgettingPattern.SELECTIVE

        return ForgettingPattern.GRADUAL

    @staticmethod
    def identify_affected_capabilities(
        crr: dict[str, float],
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
