"""Validated evaluation and judge configuration contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JudgeConfig(BaseModel):
    """Configuration for a dedicated judge that is not an evaluated model."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "openai", "anthropic", "custom"]
    model: str = Field(min_length=1)
    api_key_env: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, gt=0)
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_settings(self) -> JudgeConfig:
        reserved = {"model", "messages", "max_tokens", "temperature"}
        overlap = reserved & self.settings.keys()
        if overlap:
            raise ValueError(f"Judge settings cannot override reserved fields: {sorted(overlap)}")
        return self

    def public_provenance(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "settings": self.settings,
        }


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(min_length=1)
    finetuned_model: str = Field(min_length=1)
    target_tasks: list[str] = Field(default_factory=list)
    target_task: str | None = None
    general_probes: list[str] = Field(
        default_factory=lambda: [
            "reasoning",
            "code",
            "math",
            "instruction_following",
            "safety",
            "world_knowledge",
            "multilingual",
            "chat_quality",
        ]
    )
    num_samples: int = Field(default=100, gt=0)
    deep_analysis: bool = False
    deep_analysis_samples: int = Field(default=50, gt=0)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    inference_backend: Literal["auto", "transformers", "vllm", "llama_cpp"] = "auto"
    judge: JudgeConfig | None = None
    judge_model: str = "auto"
    judge_api: str | None = None
    cache_baseline: bool = True
    output_report: str | None = None
    output_format: Literal["html", "json", "csv", "markdown"] = "html"
    batch_size: int = Field(default=32, gt=0)
    max_tokens: int = Field(default=512, gt=0)
    verdict_weights: dict[str, float] = Field(default_factory=dict)
    profile_name: str | None = None
    hard_gates: dict[str, float | bool] = Field(default_factory=dict)
    generation_settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        targets = migrated.get("target_tasks")
        legacy_target = migrated.get("target_task")
        if not targets and legacy_target:
            migrated["target_tasks"] = [legacy_target]
        elif targets and not legacy_target:
            migrated["target_task"] = targets[0]

        if migrated.get("judge") is None:
            legacy_model = migrated.get("judge_model", "auto")
            legacy_api = migrated.get("judge_api")
            if legacy_api:
                provider = str(legacy_api).lower()
                if provider not in {"openai", "anthropic"}:
                    raise ValueError("judge_api must be 'openai' or 'anthropic'")
                default_model = "gpt-4o-mini" if provider == "openai" else "claude-3-5-haiku-latest"
                migrated["judge"] = {
                    "provider": provider,
                    "model": legacy_model if legacy_model != "auto" else default_model,
                }
            elif isinstance(legacy_model, str) and legacy_model != "auto":
                if ":" in legacy_model:
                    prefix, model = legacy_model.split(":", 1)
                    if prefix in {"openai", "anthropic", "local"} and model:
                        migrated["judge"] = {"provider": prefix, "model": model}
                    else:
                        migrated["judge"] = {"provider": "local", "model": legacy_model}
                else:
                    migrated["judge"] = {"provider": "local", "model": legacy_model}
        return migrated

    @model_validator(mode="after")
    def sync_target_alias(self) -> EvalConfig:
        self.target_tasks = list(dict.fromkeys(self.target_tasks))
        self.target_task = self.target_tasks[0] if self.target_tasks else None
        if any(not target.strip() for target in self.target_tasks):
            raise ValueError("target_tasks entries must be non-empty")
        if any(not probe.strip() for probe in self.general_probes):
            raise ValueError("general_probes entries must be non-empty")
        unknown_gates = set(self.hard_gates) - {"sar_min", "strong_safety_required"}
        if unknown_gates:
            raise ValueError(f"Unknown hard gates: {sorted(unknown_gates)}")
        if "sar_min" in self.hard_gates:
            sar_min = self.hard_gates["sar_min"]
            if (
                isinstance(sar_min, bool)
                or not isinstance(sar_min, (int, float))
                or not 0 <= sar_min <= 1
            ):
                raise ValueError("hard_gates.sar_min must be a number in [0, 1]")
        if "strong_safety_required" in self.hard_gates and not isinstance(
            self.hard_gates["strong_safety_required"], bool
        ):
            raise ValueError("hard_gates.strong_safety_required must be boolean")
        if self.verdict_weights:
            from finetunecheck.forgetting.metrics import canonicalize_roi_weights

            self.verdict_weights = canonicalize_roi_weights(self.verdict_weights)
        common_generation_settings = {
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
        }
        unknown_generation = set(self.generation_settings) - common_generation_settings
        if unknown_generation:
            raise ValueError(f"Unknown generation settings: {sorted(unknown_generation)}")
        return self


class QuickConfig(EvalConfig):
    """Offline-runnable deterministic smoke profile (no LLM or code execution judge)."""

    num_samples: int = 10
    general_probes: list[str] = Field(
        default_factory=lambda: [
            "math",
            "classification",
            "instruction_following",
            "safety",
        ]
    )
