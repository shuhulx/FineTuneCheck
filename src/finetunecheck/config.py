"""Evaluation configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    base_model: str
    finetuned_model: str
    target_task: str | None = None
    profile: str | None = None
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
    num_samples: int = 100
    mode: Literal["quick", "standard", "thorough"] = "standard"
    deep_analysis: bool = False
    deep_analysis_samples: int = 256
    device: str = "auto"
    judge_model: str = "auto"
    judge_api: str | None = None
    cache_baseline: bool = True
    output_report: str | None = None
    output_format: Literal["html", "json", "csv", "markdown"] = "html"
    batch_size: int = 32
    max_tokens: int = 512


class QuickConfig(EvalConfig):
    """Quick mode: reduced probes and samples for 5-minute evaluation."""

    num_samples: int = 20
    general_probes: list[str] = Field(
        default_factory=lambda: ["reasoning", "code", "math", "safety"]
    )
    mode: Literal["quick", "standard", "thorough"] = "quick"
