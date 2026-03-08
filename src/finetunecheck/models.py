"""Pydantic v2 data models — the data contracts for the entire FineTuneCheck system."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    GOOD_WITH_CONCERNS = "GOOD_WITH_CONCERNS"
    POOR = "POOR"
    HARMFUL = "HARMFUL"


class ForgettingPattern(str, Enum):
    CATASTROPHIC = "catastrophic"
    GRADUAL = "gradual"
    SELECTIVE = "selective"
    MINIMAL = "minimal"


class ModelType(str, Enum):
    HF = "hf"
    LORA = "lora"
    GGUF = "gguf"


class JudgeType(str, Enum):
    LLM = "llm"
    EXACT_MATCH = "exact_match"
    F1 = "f1"
    EXECUTION = "execution"
    RULE_BASED = "rule_based"
    ROUGE = "rouge"


class ModelSpec(BaseModel):
    path: str
    model_type: ModelType
    base_model: str | None = None


class ProbeSample(BaseModel):
    id: str
    input: str
    reference: str | None = None
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ProbeSet(BaseModel):
    name: str
    version: str = "1.0"
    category: str = ""
    judge_type: JudgeType
    judge_criteria: str = ""
    samples: list[ProbeSample]


class InferenceResult(BaseModel):
    model_path: str
    probe_name: str
    sample_id: str
    output: str
    logprobs: list[float] | None = None
    latency_ms: float = 0.0


class JudgeVerdict(BaseModel):
    sample_id: str
    score: float
    explanation: str = ""
    judge_type: str = ""


class CategoryScore(BaseModel):
    category: str
    mean_score: float
    std_score: float = 0.0
    num_samples: int = 0
    sample_scores: list[float] = Field(default_factory=list)
    sample_verdicts: list[JudgeVerdict] = Field(default_factory=list)


class SampleRegression(BaseModel):
    category: str
    sample_id: str
    prompt: str
    base_answer: str
    ft_answer: str
    base_score: float
    ft_score: float
    score_change: float


class ForgettingReport(BaseModel):
    backward_transfer: float
    capability_retention_rates: dict[str, float]
    selective_forgetting_index: float
    safety_alignment_retention: float | None = None
    pattern: ForgettingPattern
    most_affected: list[str] = Field(default_factory=list)
    resilient: list[str] = Field(default_factory=list)
    regressions: list[SampleRegression] = Field(default_factory=list)


class PerplexityDistShift(BaseModel):
    kl_divergence: float
    wasserstein_distance: float
    tail_fraction: float
    mean_ppl_base: float
    mean_ppl_ft: float
    base_ppls: list[float] = Field(default_factory=list)
    ft_ppls: list[float] = Field(default_factory=list)


class CKAReport(BaseModel):
    per_layer_cka: dict[str, float]
    most_diverged_layers: list[str] = Field(default_factory=list)
    mean_cka: float = 0.0


class SpectralReport(BaseModel):
    per_layer_effective_rank: dict[str, float]
    per_layer_frobenius_norm: dict[str, float]
    per_layer_top_singular_values: dict[str, list[float]] = Field(default_factory=dict)
    mean_effective_rank: float = 0.0
    is_lora: bool = False
    lora_rank: int | None = None


class CalibrationReport(BaseModel):
    base_ece: float
    ft_ece: float
    ece_delta: float
    per_bin_accuracy_base: list[float] = Field(default_factory=list)
    per_bin_accuracy_ft: list[float] = Field(default_factory=list)
    per_bin_confidence: list[float] = Field(default_factory=list)
    per_bin_confidence_ft: list[float] = Field(default_factory=list)


class ActivationDriftReport(BaseModel):
    per_layer_cosine_sim: dict[str, float]
    disrupted_heads: list[dict] = Field(default_factory=list)
    mean_drift: float = 0.0


class DeepAnalysisReport(BaseModel):
    perplexity: PerplexityDistShift | None = None
    cka: CKAReport | None = None
    spectral: SpectralReport | None = None
    calibration: CalibrationReport | None = None
    activation: ActivationDriftReport | None = None


class EvalResults(BaseModel):
    base_model: str
    finetuned_model: str
    target_task: str | None = None
    base_scores: dict[str, CategoryScore]
    ft_scores: dict[str, CategoryScore]
    target_improvement: float = 0.0
    forgetting: ForgettingReport | None = None
    deep_analysis: DeepAnalysisReport | None = None
    verdict: Verdict = Verdict.GOOD
    roi_score: float = 0.0
    summary: str = ""
    concerns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
