"""Full forgetting analysis pipeline."""

from __future__ import annotations

from finetunecheck.forgetting.detector import ForgettingDetector
from finetunecheck.forgetting.metrics import (
    backward_transfer,
    capability_retention_rate,
    safety_alignment_retention,
    selective_forgetting_index,
)
from finetunecheck.models import (
    CategoryScore,
    ForgettingReport,
    JudgeVerdict,
    ProbeSet,
    SampleRegression,
)


class ForgettingAnalyzer:
    """End-to-end forgetting analysis combining metrics, detection, and regression finding."""

    @staticmethod
    def find_regressions(
        base_results: dict[str, list[JudgeVerdict]],
        ft_results: dict[str, list[JudgeVerdict]],
        probes: dict[str, ProbeSet],
        threshold: float = 0.1,
    ) -> list[SampleRegression]:
        """Find specific samples that regressed significantly.

        A regression is a sample where the fine-tuned model scored substantially
        lower than the base model (score drop >= threshold).

        Args:
            base_results: Per-category list of judge verdicts from the base model.
            ft_results: Per-category list of judge verdicts from the fine-tuned model.
            probes: Per-category probe sets (used for prompt text).
            threshold: Minimum score drop to count as a regression.

        Returns:
            List of ``SampleRegression`` objects, sorted by score_change ascending
            (worst regressions first).
        """
        regressions: list[SampleRegression] = []

        for category in base_results:
            if category not in ft_results:
                continue

            base_by_id = {v.sample_id: v for v in base_results[category]}
            ft_by_id = {v.sample_id: v for v in ft_results[category]}

            probe = probes.get(category)
            sample_map: dict[str, str] = {}
            if probe:
                sample_map = {s.id: s.input for s in probe.samples}

            for sample_id, base_verdict in base_by_id.items():
                ft_verdict = ft_by_id.get(sample_id)
                if ft_verdict is None:
                    continue

                score_change = ft_verdict.score - base_verdict.score
                if score_change <= -threshold:
                    regressions.append(
                        SampleRegression(
                            category=category,
                            sample_id=sample_id,
                            prompt=sample_map.get(sample_id, ""),
                            base_answer=base_verdict.explanation,
                            ft_answer=ft_verdict.explanation,
                            base_score=base_verdict.score,
                            ft_score=ft_verdict.score,
                            score_change=score_change,
                        )
                    )

        regressions.sort(key=lambda r: r.score_change)
        return regressions

    @staticmethod
    def analyze(
        base_scores: dict[str, CategoryScore],
        ft_scores: dict[str, CategoryScore],
        base_results: dict[str, list[JudgeVerdict]] | None = None,
        ft_results: dict[str, list[JudgeVerdict]] | None = None,
        probes: dict[str, ProbeSet] | None = None,
        target_categories: list[str] | None = None,
        target_task: str | None = None,
    ) -> ForgettingReport:
        """Run the full forgetting analysis pipeline.

        Computes backward transfer, CRR, SFI, SAR, classifies the forgetting
        pattern, identifies affected capabilities, and optionally finds
        sample-level regressions.

        Args:
            base_scores: Category scores from the base model.
            ft_scores: Category scores from the fine-tuned model.
            base_results: Optional per-category judge verdicts from base model
                (needed for regression analysis).
            ft_results: Optional per-category judge verdicts from fine-tuned model
                (needed for regression analysis).
            probes: Optional per-category probe sets (needed for regression context).
            target_categories: Categories that were the target of fine-tuning.
            target_task: Single target task to exclude from CRR (consistent with
                runner._compute_forgetting).

        Returns:
            A complete ``ForgettingReport``.
        """
        bwt = backward_transfer(base_scores, ft_scores, target_categories)

        crr = capability_retention_rate(base_scores, ft_scores, target_task)

        sfi = selective_forgetting_index(crr)

        base_safety = base_scores.get("safety")
        ft_safety = ft_scores.get("safety")
        sar = safety_alignment_retention(base_safety, ft_safety)

        pattern = ForgettingDetector.classify_pattern(crr, bwt, sfi)

        most_affected, resilient = ForgettingDetector.identify_affected_capabilities(crr)

        regressions: list[SampleRegression] = []
        if base_results is not None and ft_results is not None:
            regressions = ForgettingAnalyzer.find_regressions(
                base_results,
                ft_results,
                probes or {},
            )

        return ForgettingReport(
            backward_transfer=round(bwt, 4),
            capability_retention_rates={k: round(v, 4) for k, v in crr.items()},
            selective_forgetting_index=round(sfi, 4),
            safety_alignment_retention=round(sar, 4) if sar is not None else None,
            pattern=pattern,
            most_affected=most_affected,
            resilient=resilient,
            regressions=regressions,
        )
