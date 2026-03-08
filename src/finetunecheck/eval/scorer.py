"""Score aggregation and ROI computation."""

from __future__ import annotations

from statistics import mean, stdev

from finetunecheck.forgetting.metrics import compute_roi_score
from finetunecheck.models import (
    CategoryScore,
    ForgettingReport,
    JudgeVerdict,
)


class Scorer:
    @staticmethod
    def compute_category_scores(verdicts: list[JudgeVerdict], category: str) -> CategoryScore:
        """Aggregate verdicts into a CategoryScore."""
        if not verdicts:
            return CategoryScore(category=category, mean_score=0.0)
        scores = [v.score for v in verdicts]
        return CategoryScore(
            category=category,
            mean_score=mean(scores),
            std_score=stdev(scores) if len(scores) > 1 else 0.0,
            num_samples=len(scores),
            sample_scores=scores,
            sample_verdicts=verdicts,
        )

    @staticmethod
    def compute_target_improvement(base_score: CategoryScore, ft_score: CategoryScore) -> float:
        """Relative improvement on target task.

        Returns:
            Fractional improvement (e.g. 0.15 = 15% improvement).
            If base is 0, returns ft score directly.
        """
        if base_score.mean_score == 0:
            return ft_score.mean_score
        improvement = (ft_score.mean_score - base_score.mean_score) / base_score.mean_score
        return max(-1.0, min(10.0, improvement))

    @staticmethod
    def compute_roi(
        target_improvement: float,
        forgetting_report: ForgettingReport | None,
        profile_weights: dict[str, float] | None = None,
    ) -> float:
        """Composite ROI score 0-100.

        Delegates to ``forgetting.metrics.compute_roi_score`` for a unified
        5-component formula that accounts for target improvement, retention,
        safety, selectivity penalty, and backward transfer penalty.

        Profile weights allow domain-specific rebalancing.
        """
        if forgetting_report is not None:
            retention_rates = list(forgetting_report.capability_retention_rates.values())
            mean_crr = mean(retention_rates) if retention_rates else 1.0
            sar = forgetting_report.safety_alignment_retention
            bwt = forgetting_report.backward_transfer
            sfi = forgetting_report.selective_forgetting_index
        else:
            mean_crr = 1.0
            sar = 1.0
            bwt = 0.0
            sfi = 0.0

        return compute_roi_score(
            target_improvement=target_improvement,
            bwt=bwt,
            sar=sar,
            sfi=sfi,
            mean_crr=mean_crr,
            weights=profile_weights,
        )
