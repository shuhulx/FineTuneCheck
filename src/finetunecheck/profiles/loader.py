"""Strict built-in evaluation profile contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from finetunecheck.config import EvalConfig
from finetunecheck.forgetting.metrics import canonicalize_roi_weights

_BUILTIN_DIR = Path(__file__).parent / "builtin"


class EvalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    target_probes: list[str] = Field(default_factory=list)
    general_probes: list[str] = Field(default_factory=list)
    verdict_weights: dict[str, float] = Field(default_factory=dict)
    hard_gates: dict[str, float | bool] = Field(default_factory=dict)
    num_samples_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_contract(self) -> EvalProfile:
        overlap = set(self.target_probes) & set(self.general_probes)
        if overlap:
            raise ValueError(f"Target/general probes overlap: {sorted(overlap)}")
        if self.verdict_weights:
            canonicalize_roi_weights(self.verdict_weights)
        unknown_gates = set(self.hard_gates) - {"sar_min", "strong_safety_required"}
        if unknown_gates:
            raise ValueError(f"Unknown profile hard gates: {sorted(unknown_gates)}")
        return self


class ProfileLoader:
    _profiles: dict[str, EvalProfile] = {}
    _loaded: bool = False

    @classmethod
    def _load_builtin(cls) -> None:
        if cls._loaded:
            return
        if not _BUILTIN_DIR.is_dir():
            cls._loaded = True
            return
        from finetunecheck.probes.registry import ProbeRegistry

        known_probes = set(ProbeRegistry.list())
        for yml_path in sorted(_BUILTIN_DIR.glob("*.yml")):
            try:
                raw = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
                if raw is None:
                    continue
                profile = EvalProfile.model_validate(raw)
            except (yaml.YAMLError, ValueError) as exc:
                raise ValueError(f"Failed to load profile {yml_path.name}: {exc}") from exc
            missing = (set(profile.target_probes) | set(profile.general_probes)) - known_probes
            if missing:
                raise ValueError(
                    f"Profile {profile.name!r} references unknown probes: {sorted(missing)}"
                )
            cls._profiles[profile.name] = profile
        cls._loaded = True

    @classmethod
    def get(cls, name: str) -> EvalProfile:
        cls._load_builtin()
        if name not in cls._profiles:
            raise KeyError(f"Profile {name!r} not found. Available: {', '.join(cls.list())}")
        return cls._profiles[name]

    @classmethod
    def list(cls) -> list[str]:
        cls._load_builtin()
        return sorted(cls._profiles)

    @classmethod
    def apply_to_config(cls, profile_name: str, config: EvalConfig) -> EvalConfig:
        profile = cls.get(profile_name)
        overrides: dict[str, Any] = {
            "profile_name": profile.name,
            "target_tasks": profile.target_probes,
            "target_task": profile.target_probes[0] if profile.target_probes else None,
            "general_probes": profile.general_probes,
            "verdict_weights": canonicalize_roi_weights(profile.verdict_weights),
            "hard_gates": profile.hard_gates,
        }
        if profile.num_samples_override is not None:
            overrides["num_samples"] = profile.num_samples_override
        payload = config.model_dump()
        payload.update(overrides)
        return EvalConfig.model_validate(payload)

    @classmethod
    def reset(cls) -> None:
        cls._profiles.clear()
        cls._loaded = False
