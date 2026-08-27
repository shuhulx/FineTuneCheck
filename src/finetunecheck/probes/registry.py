"""Central registry for all probe sets."""

from __future__ import annotations

import json
from pathlib import Path

from finetunecheck.models import ProbeSet

_BUILTIN_DIR = Path(__file__).parent / "builtin"


class ProbeRegistry:
    """Central registry for all probe sets.

    Built-in probes are lazy-loaded from JSON files on first access.
    Custom probes can be registered at runtime.
    """

    _builtin: dict[str, ProbeSet] = {}
    _custom: dict[str, ProbeSet] = {}
    _loaded: bool = False

    @classmethod
    def _load_builtin(cls) -> None:
        """Lazy-load all built-in probe JSON files."""
        if cls._loaded:
            return
        if not _BUILTIN_DIR.is_dir():
            cls._loaded = True
            return
        for json_path in sorted(_BUILTIN_DIR.glob("*.json")):
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                probe = ProbeSet.model_validate(raw)
                cls._builtin[probe.name] = probe
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(f"Failed to load builtin probe {json_path.name}: {exc}") from exc
        cls._loaded = True

    @classmethod
    def get(cls, name: str) -> ProbeSet:
        """Get a probe set by name.

        Raises ``KeyError`` if the name is not found in built-in or custom probes.
        """
        cls._load_builtin()
        if name in cls._custom:
            return cls._custom[name]
        if name in cls._builtin:
            return cls._builtin[name]
        raise KeyError(f"Probe set {name!r} not found. Available: {', '.join(cls.list())}")

    @classmethod
    def list(cls) -> list[str]:
        """List all available probe set names (built-in first, then custom)."""
        cls._load_builtin()
        seen: dict[str, None] = {}
        for name in cls._builtin:
            seen[name] = None
        for name in cls._custom:
            seen[name] = None
        return list(seen)

    @classmethod
    def register(cls, probe: ProbeSet) -> None:
        """Register a custom probe set (overwrites any existing custom probe with the same name)."""
        cls._custom[probe.name] = probe

    @classmethod
    def get_for_profile(cls, probe_names: list[str]) -> list[ProbeSet]:
        """Get multiple probe sets for a profile.

        Raises ``KeyError`` if any name is not found.
        """
        return [cls.get(name) for name in probe_names]

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing)."""
        cls._builtin.clear()
        cls._custom.clear()
        cls._loaded = False
