"""Tests for forgetting detection metrics."""

from finetunecheck.eval.scorer import Scorer
from finetunecheck.forgetting.metrics import (
    backward_transfer,
    capability_retention_rate,
    safety_alignment_retention,
    selective_forgetting_index,
)
from finetunecheck.models import CategoryScore, ForgettingPattern


def _make_scores(values: dict[str, float]) -> dict[str, CategoryScore]:
    """Helper: build CategoryScore dict from {category: mean_score}."""
    return {
        cat: CategoryScore(category=cat, mean_score=score, num_samples=10, sample_scores=[score] * 10)
        for cat, score in values.items()
    }


class TestBackwardTransfer:
    def test_backward_transfer_no_forgetting(self):
        """BWT should be ~0 when scores are equal."""
        base = _make_scores({"code": 0.8, "math": 0.7, "safety": 0.9})
        ft = _make_scores({"code": 0.8, "math": 0.7, "safety": 0.9})
        bwt = backward_transfer(base, ft)
        assert abs(bwt) < 1e-9

    def test_backward_transfer_with_forgetting(self):
        """BWT should be negative when ft scores drop."""
        base = _make_scores({"code": 0.8, "math": 0.7, "safety": 0.9})
        ft = _make_scores({"code": 0.5, "math": 0.4, "safety": 0.6})
        bwt = backward_transfer(base, ft)
        assert bwt < 0
        # Average drop is (0.5-0.8 + 0.4-0.7 + 0.6-0.9) / 3 = -0.3
        assert abs(bwt - (-0.3)) < 1e-9

    def test_backward_transfer_with_improvement(self):
        """BWT should be positive when ft scores improve."""
        base = _make_scores({"code": 0.5, "math": 0.5})
        ft = _make_scores({"code": 0.7, "math": 0.8})
        bwt = backward_transfer(base, ft)
        assert bwt > 0

    def test_backward_transfer_excludes_target(self):
        """BWT should not include target category."""
        base = _make_scores({"reasoning": 0.5, "code": 0.8, "math": 0.7})
        ft = _make_scores({"reasoning": 0.9, "code": 0.8, "math": 0.7})
        bwt = backward_transfer(base, ft, exclude_target="reasoning")
        # Only code and math matter, both unchanged
        assert abs(bwt) < 1e-9

    def test_backward_transfer_missing_ft_category(self):
        """When ft is missing a category, BWT should treat it as -1.0 (total loss)."""
        base = _make_scores({"code": 0.8, "math": 0.7})
        ft = _make_scores({"code": 0.8})  # math missing
        bwt = backward_transfer(base, ft)
        # code delta = 0, math delta = -1.0 (normalized total loss)
        assert abs(bwt - (-0.5)) < 1e-9

    def test_backward_transfer_empty(self):
        """BWT of empty scores should be 0."""
        bwt = backward_transfer({}, {})
        assert bwt == 0.0


class TestCapabilityRetentionRate:
    def test_capability_retention_rate(self):
        """CRR should be ft/base per category."""
        base = _make_scores({"code": 0.8, "math": 0.5})
        ft = _make_scores({"code": 0.6, "math": 0.5})
        rates = capability_retention_rate(base, ft)
        assert abs(rates["code"] - 0.75) < 1e-9
        assert abs(rates["math"] - 1.0) < 1e-9

    def test_crr_handles_zero_base(self):
        """CRR should handle zero base scores gracefully."""
        base = _make_scores({"code": 0.0, "math": 0.0})
        ft = _make_scores({"code": 0.0, "math": 0.5})
        rates = capability_retention_rate(base, ft)
        # 0/0 -> 1.0 (both zero = retained), nonzero/0 -> 1.0 (no baseline to regress from)
        assert rates["code"] == 1.0
        assert rates["math"] == 1.0

    def test_crr_excludes_target(self):
        """CRR should exclude the target category."""
        base = _make_scores({"reasoning": 0.5, "code": 0.8})
        ft = _make_scores({"reasoning": 0.9, "code": 0.6})
        rates = capability_retention_rate(base, ft, exclude_target="reasoning")
        assert "reasoning" not in rates
        assert "code" in rates

    def test_crr_missing_ft_category(self):
        """Missing ft category should result in 0.0 retention."""
        base = _make_scores({"code": 0.8, "math": 0.7})
        ft = _make_scores({"code": 0.8})
        rates = capability_retention_rate(base, ft)
        assert rates["math"] == 0.0

    def test_crr_improvement(self):
        """CRR > 1.0 when ft improved over base."""
        base = _make_scores({"code": 0.5})
        ft = _make_scores({"code": 0.8})
        rates = capability_retention_rate(base, ft)
        assert rates["code"] == 1.6


class TestSelectiveForgettingIndex:
    def test_selective_forgetting_index_uniform(self):
        """SFI should be low when all categories drop equally."""
        base = _make_scores({"code": 0.8, "math": 0.8, "safety": 0.8})
        ft = _make_scores({"code": 0.6, "math": 0.6, "safety": 0.6})
        sfi = selective_forgetting_index(base, ft)
        # All retention rates are identical (0.75), so stdev = 0, SFI = 0
        assert sfi == 0.0

    def test_selective_forgetting_index_selective(self):
        """SFI should be high when only some categories drop."""
        base = _make_scores({"code": 0.8, "math": 0.8, "safety": 0.8})
        ft = _make_scores({"code": 0.2, "math": 0.8, "safety": 0.8})
        sfi = selective_forgetting_index(base, ft)
        # code=0.25, math=1.0, safety=1.0 -- high variance
        assert sfi > 0.1

    def test_selective_forgetting_index_single_category(self):
        """SFI should be 0 for a single category (cannot measure variance)."""
        base = _make_scores({"code": 0.8})
        ft = _make_scores({"code": 0.4})
        sfi = selective_forgetting_index(base, ft)
        assert sfi == 0.0

    def test_selective_forgetting_index_no_forgetting(self):
        """SFI should be 0 when nothing changed."""
        base = _make_scores({"code": 0.8, "math": 0.7})
        ft = _make_scores({"code": 0.8, "math": 0.7})
        sfi = selective_forgetting_index(base, ft)
        assert sfi == 0.0


class TestSafetyAlignmentRetention:
    def test_safety_alignment_retention(self):
        """SAR should be ft_safety / base_safety."""
        base = _make_scores({"safety": 0.9, "code": 0.7})
        ft = _make_scores({"safety": 0.85, "code": 0.7})
        sar = safety_alignment_retention(base, ft)
        assert sar is not None
        assert abs(sar - (0.85 / 0.9)) < 1e-9

    def test_safety_alignment_retention_no_safety(self):
        """SAR should be None when no safety probe was evaluated."""
        base = _make_scores({"code": 0.7, "math": 0.8})
        ft = _make_scores({"code": 0.7, "math": 0.8})
        sar = safety_alignment_retention(base, ft)
        assert sar is None

    def test_safety_alignment_retention_zero_base(self):
        """SAR should return ft score when base safety is zero."""
        base = _make_scores({"safety": 0.0})
        ft = _make_scores({"safety": 0.5})
        sar = safety_alignment_retention(base, ft)
        assert sar == 0.5

    def test_safety_alignment_retained(self):
        """SAR should be 1.0 when safety score is unchanged."""
        base = _make_scores({"safety": 0.95})
        ft = _make_scores({"safety": 0.95})
        sar = safety_alignment_retention(base, ft)
        assert abs(sar - 1.0) < 1e-9


class TestForgettingPatternClassification:
    """Test that forgetting patterns can be used meaningfully with the metrics."""

    def test_forgetting_pattern_catastrophic(self):
        """Should identify catastrophic forgetting when everything drops severely."""
        base = _make_scores({"code": 0.8, "math": 0.8, "safety": 0.9})
        ft = _make_scores({"code": 0.3, "math": 0.2, "safety": 0.4})
        rates = capability_retention_rate(base, ft)
        # All rates < 0.6 -- catastrophic
        for r in rates.values():
            assert r < 0.6

    def test_forgetting_pattern_selective(self):
        """Should identify selective forgetting when only some capabilities drop."""
        base = _make_scores({"code": 0.8, "math": 0.8, "safety": 0.9})
        ft = _make_scores({"code": 0.3, "math": 0.8, "safety": 0.9})
        rates = capability_retention_rate(base, ft)
        sfi = selective_forgetting_index(base, ft)
        # One dropped severely, others fine
        assert rates["code"] < 0.5
        assert rates["math"] == 1.0
        assert sfi > 0.1

    def test_forgetting_pattern_minimal(self):
        """Should identify minimal forgetting when all CRR > 0.95."""
        base = _make_scores({"code": 0.8, "math": 0.7, "safety": 0.9})
        ft = _make_scores({"code": 0.78, "math": 0.69, "safety": 0.88})
        rates = capability_retention_rate(base, ft)
        assert all(r > 0.95 for r in rates.values())


class TestROIScore:
    def test_roi_score_range(self):
        """ROI should be between 0 and 100."""
        from finetunecheck.models import ForgettingReport

        forgetting = ForgettingReport(
            backward_transfer=-0.1,
            capability_retention_rates={"code": 0.9, "math": 0.85},
            selective_forgetting_index=0.1,
            safety_alignment_retention=0.95,
            pattern=ForgettingPattern.SELECTIVE,
        )
        roi = Scorer.compute_roi(0.2, forgetting)
        assert 0 <= roi <= 100

    def test_roi_score_no_forgetting_report(self):
        """ROI with None forgetting report should use defaults (retention=1, safety=1)."""
        roi = Scorer.compute_roi(0.5, None)
        assert 0 <= roi <= 100
        # With improvement=0.5 capped at 1.0, full retention and safety
        # Should be high
        assert roi > 50

    def test_roi_score_zero_improvement(self):
        """ROI should reflect retention/safety even with zero improvement."""
        from finetunecheck.models import ForgettingReport

        forgetting = ForgettingReport(
            backward_transfer=0.0,
            capability_retention_rates={"code": 1.0, "math": 1.0},
            selective_forgetting_index=0.0,
            safety_alignment_retention=1.0,
            pattern=ForgettingPattern.MINIMAL,
        )
        roi = Scorer.compute_roi(0.0, forgetting)
        # No target improvement, but full retention
        assert roi > 0  # retention + safety contribute

    def test_roi_score_caps_at_100(self):
        """ROI should not exceed 100 even with extreme improvement."""
        roi = Scorer.compute_roi(5.0, None)
        assert roi <= 100.0


class TestScorerCompute:
    def test_scorer_compute_category(self, sample_verdicts):
        """Scorer should aggregate verdicts into CategoryScore."""
        cat_score = Scorer.compute_category_scores(sample_verdicts, "reasoning")
        assert cat_score.category == "reasoning"
        assert cat_score.num_samples == 5
        assert len(cat_score.sample_scores) == 5
        assert abs(cat_score.mean_score - 0.6) < 1e-9  # mean of 0.8,0.7,0.6,0.5,0.4
        assert cat_score.std_score > 0

    def test_scorer_handles_empty(self):
        """Scorer should handle empty verdict list."""
        cat_score = Scorer.compute_category_scores([], "empty")
        assert cat_score.category == "empty"
        assert cat_score.mean_score == 0.0
        assert cat_score.num_samples == 0

    def test_scorer_single_verdict(self):
        """Scorer should handle single verdict (std=0)."""
        from finetunecheck.models import JudgeVerdict

        verdict = JudgeVerdict(sample_id="s_0", score=0.75, explanation="ok")
        cat_score = Scorer.compute_category_scores([verdict], "single")
        assert cat_score.mean_score == 0.75
        assert cat_score.std_score == 0.0
        assert cat_score.num_samples == 1

    def test_target_improvement_positive(self):
        """Relative improvement should be positive when ft > base."""
        base = CategoryScore(category="r", mean_score=0.5, num_samples=10)
        ft = CategoryScore(category="r", mean_score=0.7, num_samples=10)
        improvement = Scorer.compute_target_improvement(base, ft)
        assert abs(improvement - 0.4) < 1e-9  # (0.7-0.5)/0.5

    def test_target_improvement_zero_base(self):
        """When base is 0, improvement should return ft score directly."""
        base = CategoryScore(category="r", mean_score=0.0, num_samples=10)
        ft = CategoryScore(category="r", mean_score=0.6, num_samples=10)
        improvement = Scorer.compute_target_improvement(base, ft)
        assert improvement == 0.6
