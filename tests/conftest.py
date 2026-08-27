import numpy as np
import pytest

from finetunecheck.config import EvalConfig
from finetunecheck.models import (
    ActivationDriftReport,
    CalibrationReport,
    CategoryScore,
    CKAReport,
    DeepAnalysisReport,
    EvalResults,
    ForgettingPattern,
    ForgettingReport,
    JudgeType,
    JudgeVerdict,
    PerplexityDistShift,
    ProbeSample,
    ProbeSet,
    SpectralReport,
    Verdict,
)


@pytest.fixture
def sample_probe():
    """A minimal probe set for testing."""
    return ProbeSet(
        name="test_reasoning",
        version="1.0",
        category="reasoning",
        judge_type=JudgeType.LLM,
        judge_criteria="Evaluate logical reasoning",
        samples=[
            ProbeSample(id=f"test_{i}", input=f"Question {i}?", reference=f"Answer {i}")
            for i in range(5)
        ],
    )


@pytest.fixture
def sample_verdicts():
    """Pre-built judge verdicts for testing."""
    return [
        JudgeVerdict(sample_id=f"test_{i}", score=0.8 - i * 0.1, explanation=f"Good reasoning {i}")
        for i in range(5)
    ]


@pytest.fixture
def base_scores():
    """Simulated base model scores across categories."""
    rng = np.random.default_rng(42)
    categories = ["reasoning", "code", "math", "safety", "instruction_following"]
    return {
        cat: CategoryScore(
            category=cat,
            mean_score=0.75 + rng.uniform(-0.05, 0.05),
            std_score=0.1,
            num_samples=20,
            sample_scores=[0.75 + rng.uniform(-0.2, 0.2) for _ in range(20)],
        )
        for cat in categories
    }


@pytest.fixture
def ft_scores_good():
    """Fine-tuned model scores -- good outcome (minimal forgetting)."""
    return {
        "reasoning": CategoryScore(
            category="reasoning",
            mean_score=0.90,
            std_score=0.08,
            num_samples=20,
            sample_scores=[0.9] * 20,
        ),
        "code": CategoryScore(
            category="code",
            mean_score=0.72,
            std_score=0.12,
            num_samples=20,
            sample_scores=[0.72] * 20,
        ),
        "math": CategoryScore(
            category="math",
            mean_score=0.73,
            std_score=0.1,
            num_samples=20,
            sample_scores=[0.73] * 20,
        ),
        "safety": CategoryScore(
            category="safety",
            mean_score=0.98,
            std_score=0.02,
            num_samples=20,
            sample_scores=[0.98] * 20,
        ),
        "instruction_following": CategoryScore(
            category="instruction_following",
            mean_score=0.76,
            std_score=0.09,
            num_samples=20,
            sample_scores=[0.76] * 20,
        ),
    }


@pytest.fixture
def ft_scores_poor():
    """Fine-tuned model scores -- poor outcome (severe forgetting)."""
    return {
        "reasoning": CategoryScore(
            category="reasoning",
            mean_score=0.85,
            std_score=0.1,
            num_samples=20,
            sample_scores=[0.85] * 20,
        ),
        "code": CategoryScore(
            category="code",
            mean_score=0.40,
            std_score=0.15,
            num_samples=20,
            sample_scores=[0.40] * 20,
        ),
        "math": CategoryScore(
            category="math",
            mean_score=0.35,
            std_score=0.12,
            num_samples=20,
            sample_scores=[0.35] * 20,
        ),
        "safety": CategoryScore(
            category="safety",
            mean_score=0.60,
            std_score=0.2,
            num_samples=20,
            sample_scores=[0.60] * 20,
        ),
        "instruction_following": CategoryScore(
            category="instruction_following",
            mean_score=0.50,
            std_score=0.15,
            num_samples=20,
            sample_scores=[0.50] * 20,
        ),
    }


@pytest.fixture
def sample_eval_results(base_scores, ft_scores_good):
    """Complete EvalResults for report testing."""
    return EvalResults(
        base_model="meta-llama/Llama-3.1-8B",
        finetuned_model="./my-finetuned-model",
        target_task="reasoning",
        base_scores=base_scores,
        ft_scores=ft_scores_good,
        target_improvement=0.20,
        forgetting=ForgettingReport(
            backward_transfer=-0.03,
            capability_retention_rates={
                "reasoning": 1.20,
                "code": 0.96,
                "math": 0.97,
                "safety": 1.04,
                "instruction_following": 1.01,
            },
            selective_forgetting_index=0.08,
            safety_alignment_retention=1.04,
            pattern=ForgettingPattern.MINIMAL,
            most_affected=["code"],
            resilient=["safety", "reasoning"],
        ),
        verdict=Verdict.GOOD,
        roi_score=78.0,
        summary="Fine-tuning improved reasoning by 20% with minimal capability loss.",
        concerns=["Minor code capability decline (4%)"],
        recommendations=["Consider adding coding examples to training data"],
    )


@pytest.fixture
def sample_deep_analysis():
    """Sample deep analysis report."""
    return DeepAnalysisReport(
        cka=CKAReport(
            per_layer_cka={f"layer_{i}": 0.95 - i * 0.02 for i in range(10)},
            most_diverged_layers=["layer_9", "layer_8"],
            mean_cka=0.86,
        ),
        spectral=SpectralReport(
            per_layer_effective_rank={f"layer_{i}.weight": 5.0 + i for i in range(10)},
            per_layer_frobenius_norm={f"layer_{i}.weight": 0.1 + i * 0.01 for i in range(10)},
            mean_effective_rank=10.0,
            is_lora=True,
            lora_rank=16,
        ),
        perplexity=PerplexityDistShift(
            kl_divergence=0.15,
            wasserstein_distance=2.3,
            tail_fraction=0.05,
            mean_ppl_base=12.5,
            mean_ppl_ft=13.1,
        ),
        calibration=CalibrationReport(
            base_ece=0.05,
            ft_ece=0.08,
            ece_delta=0.03,
            per_bin_accuracy_base=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            per_bin_accuracy_ft=[0.08, 0.18, 0.28, 0.42, 0.48, 0.62, 0.68, 0.78, 0.85, 0.92],
            per_bin_confidence=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
        ),
        activation=ActivationDriftReport(
            per_layer_cosine_sim={f"layer_{i}": 0.98 - i * 0.01 for i in range(10)},
            disrupted_heads=[{"layer": 8, "head": 3, "cosine_sim": 0.65}],
            mean_drift=0.93,
        ),
    )


@pytest.fixture
def eval_config():
    """Default eval config for testing."""
    return EvalConfig(
        base_model="test-base",
        finetuned_model="test-ft",
        num_samples=5,
    )
