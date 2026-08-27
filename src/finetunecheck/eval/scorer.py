"""Score aggregation and ROI computation."""

from __future__ import annotations

from statistics import mean, stdev

from finetunecheck.forgetting.metrics import compute_roi_score
from finetunecheck.models import (
    CategoryScore,
    ForgettingReport,
    JudgeVerdict,
    MeasurementStatus,
)


class Scorer:
    @staticmethod
    def compute_category_scores(verdicts: list[JudgeVerdict], category: str) -> CategoryScore:
        """Aggregate verdicts into a CategoryScore."""
        if not verdicts:
            return CategoryScore(
                category=category,
                status=MeasurementStatus.NOT_RUN,
                error="No sample verdicts were produced",
            )
        measured = [
            verdict
            for verdict in verdicts
            if verdict.status == MeasurementStatus.MEASURED and verdict.score is not None
        ]
        scores = [verdict.score for verdict in measured if verdict.score is not None]
        selected_ids = [verdict.sample_id for verdict in verdicts]
        if len(measured) != len(verdicts):
            statuses = {verdict.status for verdict in verdicts}
            if MeasurementStatus.ERROR in statuses:
                status = MeasurementStatus.ERROR
            elif MeasurementStatus.INCOMPATIBLE in statuses:
                status = MeasurementStatus.INCOMPATIBLE
            elif measured:
                status = MeasurementStatus.INSUFFICIENT_SAMPLE
            else:
                status = MeasurementStatus.NOT_RUN
            errors = [
                verdict.error or verdict.explanation
                for verdict in verdicts
                if verdict.status != MeasurementStatus.MEASURED
            ]
            return CategoryScore(
                category=category,
                status=status,
                error="; ".join(errors),
                num_samples=len(scores),
                expected_samples=len(verdicts),
                sample_scores=scores,
                sample_verdicts=verdicts,
                selected_sample_ids=selected_ids,
            )
        return CategoryScore(
            category=category,
            mean_score=mean(scores),
            std_score=stdev(scores) if len(scores) > 1 else 0.0,
            num_samples=len(scores),
            expected_samples=len(verdicts),
            status=MeasurementStatus.MEASURED,
            sample_scores=scores,
            sample_verdicts=verdicts,
            selected_sample_ids=selected_ids,
        )

    @staticmethod
    def compute_target_improvement(
        base_score: CategoryScore, ft_score: CategoryScore
    ) -> float | None:
        """Absolute score delta for bounded [0, 1] category measurements."""
        if (
            base_score.status != MeasurementStatus.MEASURED
            or ft_score.status != MeasurementStatus.MEASURED
            or base_score.mean_score is None
            or ft_score.mean_score is None
        ):
            return None
        return ft_score.mean_score - base_score.mean_score

    @staticmethod
    def compute_roi(
        target_improvement: float | None,
        forgetting_report: ForgettingReport | None,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Composite ROI score 0-100.

        Delegates to ``forgetting.metrics.compute_roi_score`` for a unified
        5-component formula that accounts for target improvement, retention,
        safety, selectivity penalty, and backward transfer penalty.

        Args:
            target_improvement: Score improvement on the target task.
            forgetting_report: Forgetting metrics, or None if unavailable.
            weights: Optional weight overrides passed through to
                ``compute_roi_score``.
        """
        if forgetting_report is not None:
            retention_rates = [
                value
                for value in forgetting_report.capability_retention_rates.values()
                if value is not None
            ]
            mean_crr = mean(retention_rates) if retention_rates else None
            sar = forgetting_report.safety_alignment_retention
            bwt = forgetting_report.backward_transfer
            sfi = forgetting_report.selective_forgetting_index
        else:
            mean_crr = None
            sar = None
            bwt = None
            sfi = None

        return compute_roi_score(
            target_improvement=target_improvement,
            bwt=bwt,
            sar=sar,
            sfi=sfi,
            mean_crr=mean_crr,
            weights=weights or None,
        )
