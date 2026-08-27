"""Versioned Pydantic contracts for FineTuneCheck results and evidence."""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finetunecheck._version import __version__

RESULT_SCHEMA_VERSION = "2.0.0"
METRIC_SCHEMA_VERSION = "2.0.0"


class StrictModel(BaseModel):
    """Base for public contracts: reject misspelled and unsupported fields."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def reject_nonfinite_public_values(self) -> StrictModel:
        def check(value: Any, path: str) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Non-finite numeric value at {path}")
            if isinstance(value, dict):
                for key, item in value.items():
                    check(item, f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    check(item, f"{path}[{index}]")

        for field_name, value in self.__dict__.items():
            check(value, field_name)
        return self


class Verdict(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    GOOD_WITH_CONCERNS = "GOOD_WITH_CONCERNS"
    POOR = "POOR"
    HARMFUL = "HARMFUL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class MeasurementStatus(str, Enum):
    MEASURED = "MEASURED"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"
    INCOMPATIBLE = "INCOMPATIBLE"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


class ForgettingPattern(str, Enum):
    CATASTROPHIC = "catastrophic"
    GRADUAL = "gradual"
    SELECTIVE = "selective"
    MINIMAL = "minimal"
    UNAVAILABLE = "unavailable"


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


class ModelSpec(StrictModel):
    path: str = Field(min_length=1)
    model_type: ModelType
    base_model: str | None = None
    revision: str | None = None


KNOWN_CONSTRAINT_TYPES = frozenset(
    {
        "acrostic",
        "all_uppercase",
        "contains",
        "contains_all",
        "contains_pattern",
        "contains_refusal",
        "ends_with_text",
        "exact_words",
        "json_key_count",
        "json_keys",
        "line_count",
        "max_words",
        "min_words",
        "not_contains",
        "not_contains_word",
        "numbered_list",
        "one_of",
        "sentence_count",
        "starts_with",
        "starts_with_text",
        "table_columns",
        "table_data_rows",
        "valid_json",
        "words_per_line",
    }
)


class ProbeSample(StrictModel):
    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    reference: str | None = None
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProbeSet(StrictModel):
    name: str = Field(min_length=1)
    version: str = "1.0"
    category: str = ""
    judge_type: JudgeType
    judge_criteria: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    samples: list[ProbeSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_definitions(self) -> ProbeSet:
        seen: set[str] = set()
        for sample in self.samples:
            if sample.id in seen:
                raise ValueError(f"Duplicate sample id: {sample.id}")
            seen.add(sample.id)
            constraints = sample.metadata.get("constraints", [])
            if constraints is None:
                continue
            if not isinstance(constraints, list):
                raise ValueError(f"Constraints for {sample.id} must be a list")
            for constraint in constraints:
                if not isinstance(constraint, dict) or not isinstance(constraint.get("type"), str):
                    raise ValueError(f"Invalid constraint definition for {sample.id}")
                constraint_type = constraint["type"]
                if constraint_type not in KNOWN_CONSTRAINT_TYPES:
                    raise ValueError(
                        f"Unknown constraint type {constraint_type!r} in sample {sample.id}"
                    )
                self._validate_constraint(sample.id, constraint)
        return self

    @staticmethod
    def _validate_constraint(sample_id: str, constraint: dict[str, Any]) -> None:
        allowed_fields = {"type", "value", "description", "per_line"}
        unknown_fields = set(constraint) - allowed_fields
        if unknown_fields:
            raise ValueError(f"Unknown constraint fields for {sample_id}: {sorted(unknown_fields)}")
        kind = constraint["type"]
        no_value = {"all_uppercase", "contains_refusal", "numbered_list", "valid_json"}
        integer_value = {
            "exact_words",
            "json_key_count",
            "line_count",
            "max_words",
            "min_words",
            "sentence_count",
            "table_columns",
            "table_data_rows",
            "words_per_line",
        }
        list_value = {"contains_all", "json_keys", "one_of"}
        string_value = KNOWN_CONSTRAINT_TYPES - no_value - integer_value - list_value
        value = constraint.get("value")
        if kind in no_value and "value" in constraint:
            raise ValueError(f"Constraint {kind!r} for {sample_id} does not accept value")
        if kind in integer_value and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"Constraint {kind!r} for {sample_id} requires a non-negative integer")
        if kind in list_value and (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ValueError(f"Constraint {kind!r} for {sample_id} requires non-empty strings")
        if kind in string_value and (not isinstance(value, str) or not value):
            raise ValueError(f"Constraint {kind!r} for {sample_id} requires a non-empty string")
        if "per_line" in constraint and not isinstance(constraint["per_line"], bool):
            raise ValueError(f"Constraint per_line for {sample_id} must be boolean")
        if kind == "contains_pattern":
            if not isinstance(value, str):
                raise ValueError(f"Constraint {kind!r} for {sample_id} requires a non-empty string")
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"Constraint regex for {sample_id} is invalid: {exc}") from exc


class InferenceResult(StrictModel):
    model_path: str = Field(min_length=1)
    probe_name: str = ""
    sample_id: str = ""
    output: str
    logprobs: list[float] | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    backend: str = "unknown"
    error: str | None = None


class TestCaseOutcome(StrictModel):
    index: int = Field(ge=0)
    expression: str
    expected: Any = None
    actual: Any = None
    passed: bool = False
    error: str | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)


class JudgeVerdict(StrictModel):
    sample_id: str = Field(min_length=1)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    status: MeasurementStatus = MeasurementStatus.NOT_RUN
    explanation: str = ""
    judge_type: str = ""
    model_output: str | None = None
    raw_judge_output: str | None = None
    error: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    provenance: dict[str, Any] = Field(default_factory=dict)
    test_cases: list[TestCaseOutcome] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def infer_status(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("status") is None:
            data = dict(data)
            data["status"] = (
                MeasurementStatus.MEASURED
                if data.get("score") is not None
                else MeasurementStatus.NOT_RUN
            )
        return data

    @model_validator(mode="after")
    def validate_status(self) -> JudgeVerdict:
        if self.status == MeasurementStatus.MEASURED and self.score is None:
            raise ValueError("MEASURED judge verdicts require a finite score")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("Judge scores must be finite")
        if self.status != MeasurementStatus.MEASURED and self.score is not None:
            raise ValueError(f"{self.status.value} judge verdicts cannot carry a score")
        return self


class CategoryScore(StrictModel):
    category: str = Field(min_length=1)
    mean_score: float | None = Field(default=None, ge=0.0, le=1.0)
    std_score: float | None = Field(default=None, ge=0.0)
    num_samples: int = Field(default=0, ge=0)
    expected_samples: int = Field(default=0, ge=0)
    status: MeasurementStatus = MeasurementStatus.NOT_RUN
    error: str | None = None
    sample_scores: list[float] = Field(default_factory=list)
    sample_verdicts: list[JudgeVerdict] = Field(default_factory=list)
    selected_sample_ids: list[str] = Field(default_factory=list)
    probe_digest: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def infer_status(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("status") is None:
            data = dict(data)
            data["status"] = (
                MeasurementStatus.MEASURED
                if data.get("mean_score") is not None
                else MeasurementStatus.NOT_RUN
            )
        return data

    @model_validator(mode="after")
    def validate_measurement(self) -> CategoryScore:
        if self.expected_samples == 0 and self.num_samples > 0:
            self.expected_samples = self.num_samples
        if self.expected_samples < self.num_samples:
            raise ValueError("expected_samples cannot be less than num_samples")
        if self.mean_score is not None and not math.isfinite(self.mean_score):
            raise ValueError("Category mean must be finite")
        if self.std_score is not None and not math.isfinite(self.std_score):
            raise ValueError("Category deviation must be finite")
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in self.sample_scores):
            raise ValueError("Sample scores must be finite and in [0, 1]")
        if self.num_samples != len(self.sample_scores) and self.sample_scores:
            raise ValueError("num_samples must equal the number of sample_scores")
        if self.status == MeasurementStatus.MEASURED and self.mean_score is None:
            raise ValueError("MEASURED category scores require a mean")
        if self.status != MeasurementStatus.MEASURED and self.mean_score is not None:
            raise ValueError(f"{self.status.value} category scores cannot carry a mean")
        return self


class SampleRegression(StrictModel):
    category: str
    sample_id: str
    prompt: str
    base_answer: str
    ft_answer: str
    base_score: float = Field(ge=0.0, le=1.0)
    ft_score: float = Field(ge=0.0, le=1.0)
    score_change: float
    base_judge_explanation: str = ""
    ft_judge_explanation: str = ""


class ForgettingReport(StrictModel):
    backward_transfer: float | None = None
    capability_retention_rates: dict[str, float | None]
    selective_forgetting_index: float | None = None
    safety_alignment_retention: float | None = None
    status: MeasurementStatus = MeasurementStatus.MEASURED
    pattern: ForgettingPattern
    most_affected: list[str] = Field(default_factory=list)
    resilient: list[str] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)
    regressions: list[SampleRegression] = Field(default_factory=list)


class SafetySmokeMeasurement(StrictModel):
    status: MeasurementStatus = MeasurementStatus.NOT_RUN
    harmful_requests: int = Field(default=0, ge=0)
    successful_refusals: int = Field(default=0, ge=0)
    harmful_refusal_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    benign_controls: int = Field(default=0, ge=0)
    benign_overrefusals: int = Field(default=0, ge=0)
    benign_overrefusal_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class SafetySmokeReport(StrictModel):
    method: str = "refusal/over-refusal heuristic smoke check"
    version: str = "safety-smoke-v2"
    supports_deployment_claims: bool = False
    base: SafetySmokeMeasurement
    fine_tuned: SafetySmokeMeasurement


class PerplexityDistShift(StrictModel):
    kl_divergence: float
    wasserstein_distance: float
    tail_fraction: float
    mean_ppl_base: float
    mean_ppl_ft: float
    base_ppls: list[float] = Field(default_factory=list)
    ft_ppls: list[float] = Field(default_factory=list)


class CKAReport(StrictModel):
    per_layer_cka: dict[str, float]
    most_diverged_layers: list[str] = Field(default_factory=list)
    mean_cka: float = 0.0


class SpectralReport(StrictModel):
    per_layer_effective_rank: dict[str, float]
    per_layer_frobenius_norm: dict[str, float]
    per_layer_top_singular_values: dict[str, list[float]] = Field(default_factory=dict)
    mean_effective_rank: float = 0.0
    is_lora: bool = False
    lora_rank: int | None = None
    rank_label: str = "rank@k"
    magnitude_label: str = "relative_frobenius_norm"


class CalibrationReport(StrictModel):
    base_ece: float
    ft_ece: float
    ece_delta: float
    per_bin_accuracy_base: list[float] = Field(default_factory=list)
    per_bin_accuracy_ft: list[float] = Field(default_factory=list)
    per_bin_confidence: list[float] = Field(default_factory=list)
    per_bin_confidence_ft: list[float] = Field(default_factory=list)


class ActivationDriftReport(StrictModel):
    per_layer_cosine_sim: dict[str, float]
    disrupted_heads: list[dict[str, Any]] = Field(default_factory=list)
    mean_drift: float = 0.0
    attention_status: MeasurementStatus = MeasurementStatus.NOT_RUN
    attention_error: str | None = None


class DeepComponentStatus(StrictModel):
    status: MeasurementStatus
    error: str | None = None


class DeepAnalysisReport(StrictModel):
    status: MeasurementStatus = MeasurementStatus.NOT_RUN
    component_status: dict[str, DeepComponentStatus] = Field(default_factory=dict)
    corpus_size: int = Field(default=0, ge=0)
    samples_requested: int = Field(default=0, ge=0)
    samples_used: int = Field(default=0, ge=0)
    perplexity: PerplexityDistShift | None = None
    cka: CKAReport | None = None
    spectral: SpectralReport | None = None
    calibration: CalibrationReport | None = None
    activation: ActivationDriftReport | None = None


class EvalResults(StrictModel):
    result_schema_version: str = RESULT_SCHEMA_VERSION
    metric_schema_version: str = METRIC_SCHEMA_VERSION
    package_version: str = __version__
    base_model: str = Field(min_length=1)
    finetuned_model: str = Field(min_length=1)
    target_tasks: list[str] = Field(default_factory=list)
    target_task: str | None = None
    base_scores: dict[str, CategoryScore]
    ft_scores: dict[str, CategoryScore]
    target_improvement: float | None = None
    target_improvements: dict[str, float | None] = Field(default_factory=dict)
    target_delta_intervals_95: dict[str, tuple[float, float, float] | None] = Field(
        default_factory=dict
    )
    forgetting: ForgettingReport | None = None
    safety_smoke: SafetySmokeReport | None = None
    deep_analysis: DeepAnalysisReport | None = None
    verdict: Verdict = Verdict.INSUFFICIENT_EVIDENCE
    roi_score: float | None = Field(default=None, ge=0.0, le=100.0)
    roi_formula_version: str = "roi-v2"
    roi_component_weights: dict[str, float] = Field(default_factory=dict)
    roi_component_values: dict[str, float | None] = Field(default_factory=dict)
    roi_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""
    concerns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    probe_digest: str | None = None
    judge_provenance: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_target_alias(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        targets = migrated.get("target_tasks")
        legacy = migrated.get("target_task")
        if not targets and legacy:
            migrated["target_tasks"] = [legacy]
        elif targets and not legacy:
            migrated["target_task"] = targets[0]
        return migrated

    @model_validator(mode="after")
    def validate_result_contract(self) -> EvalResults:
        if len(set(self.target_tasks)) != len(self.target_tasks):
            raise ValueError("target_tasks must not contain duplicates")
        if any(not target.strip() for target in self.target_tasks):
            raise ValueError("target_tasks entries must be non-empty")
        if self.target_task is not None and (
            not self.target_tasks or self.target_task != self.target_tasks[0]
        ):
            raise ValueError("target_task must match the first canonical target_tasks entry")
        canonical_roi_keys = {"target", "retention", "safety", "selectivity", "bwt"}
        unknown_weights = set(self.roi_component_weights) - canonical_roi_keys
        unknown_values = set(self.roi_component_values) - canonical_roi_keys
        if unknown_weights or unknown_values:
            raise ValueError(
                f"Unknown ROI component keys: {sorted(unknown_weights | unknown_values)}"
            )
        if any(value < 0 for value in self.roi_component_weights.values()):
            raise ValueError("ROI component weights must be non-negative")
        return self
