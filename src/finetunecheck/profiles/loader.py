"""Profile loader — load and apply evaluation profiles."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from finetunecheck.config import EvalConfig

_BUILTIN_DIR = Path(__file__).parent / "builtin"


class EvalProfile(BaseModel):
    """An evaluation profile that selects probes and configures weights."""

    name: str
    description: str = ""
    target_probes: list[str] = Field(default_factory=list)
    general_probes: list[str] = Field(default_factory=list)
    extra_checks: list[str] = Field(default_factory=list)
    verdict_weights: dict[str, float] = Field(default_factory=dict)
    num_samples_override: int | None = None


class ProfileLoader:
    """Central registry for evaluation profiles.

    Built-in profiles are lazy-loaded from YAML files on first access.
    """

    _profiles: dict[str, EvalProfile] = {}
    _loaded: bool = False

    @classmethod
    def _load_builtin(cls) -> None:
        """Load all built-in profile YAML files."""
        if cls._loaded:
            return
        if not _BUILTIN_DIR.is_dir():
            cls._loaded = True
            return
        for yml_path in sorted(_BUILTIN_DIR.glob("*.yml")):
            try:
                raw = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
                if raw is None:
                    continue
                profile = EvalProfile(
                    name=raw["name"],
                    description=raw.get("description", ""),
                    target_probes=raw.get("target_probes", []),
                    general_probes=raw.get("general_probes", []),
                    extra_checks=raw.get("extra_checks", []),
                    verdict_weights=raw.get("verdict_weights", {}),
                    num_samples_override=raw.get("num_samples_override"),
                )
                cls._profiles[profile.name] = profile
            except (yaml.YAMLError, KeyError, ValueError) as exc:
                raise ValueError(f"Failed to load profile {yml_path.name}: {exc}") from exc
        cls._loaded = True

    @classmethod
    def get(cls, name: str) -> EvalProfile:
        """Get a profile by name.

        Raises ``KeyError`` if the name is not found.
        """
        cls._load_builtin()
        if name not in cls._profiles:
            raise KeyError(f"Profile {name!r} not found. Available: {', '.join(cls.list())}")
        return cls._profiles[name]

    @classmethod
    def list(cls) -> list[str]:
        """List all available profile names."""
        cls._load_builtin()
        return sorted(cls._profiles.keys())

    @classmethod
    def apply_to_config(cls, profile_name: str, config: EvalConfig) -> EvalConfig:
        """Apply profile settings to an EvalConfig, returning a modified copy.

        The profile overrides the config's ``general_probes`` with the union of
        the profile's ``target_probes`` and ``general_probes``. If the profile
        defines ``num_samples_override``, it overrides the config's ``num_samples``.
        The profile name is set on the config.
        """
        profile = cls.get(profile_name)

        all_probes = list(dict.fromkeys(profile.target_probes + profile.general_probes))

        overrides: dict = {
            "general_probes": all_probes,
        }
        if profile.num_samples_override is not None:
            overrides["num_samples"] = profile.num_samples_override

        if profile.target_probes:
            overrides["target_task"] = profile.target_probes[0]

        return config.model_copy(update=overrides)

    @classmethod
    def reset(cls) -> None:
        """Reset the loader (useful for testing)."""
        cls._profiles.clear()
        cls._loaded = False
