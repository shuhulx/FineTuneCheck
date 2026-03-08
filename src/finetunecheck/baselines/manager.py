"""Manages pre-computed and cached baselines for common base models."""

from __future__ import annotations

import json
from pathlib import Path

from finetunecheck.eval.cache import BaselineCache
from finetunecheck.models import CategoryScore


class BaselineManager:
    """Manages pre-computed and cached baselines.

    Pre-computed baselines are shipped with the package for popular models.
    Runtime baselines are cached via BaselineCache for re-use.
    """

    PRECOMPUTED: dict[str, str] = {
        "meta-llama/Llama-3.1-8B": "llama-3.1-8b.json",
        "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b.json",
        "mistralai/Mistral-7B-v0.3": "mistral-7b-v0.3.json",
        "mistralai/Mistral-7B-Instruct-v0.3": "mistral-7b-v0.3.json",
        "Qwen/Qwen2.5-7B": "qwen-2.5-7b.json",
        "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7b.json",
        "microsoft/Phi-3-mini-4k-instruct": "phi-3-mini.json",
    }

    def __init__(self) -> None:
        self._data_dir = Path(__file__).parent / "data"

    def get_baseline(
        self,
        model_path: str,
        probe_name: str,
        num_samples: int,
        cache: BaselineCache,
    ) -> CategoryScore | None:
        """Check precomputed baselines, then cache. Return None if nothing found.

        Args:
            model_path: HuggingFace model ID or local path.
            probe_name: Name of the probe set.
            num_samples: Number of samples evaluated.
            cache: BaselineCache instance for runtime caching.

        Returns:
            CategoryScore if found, None otherwise.
        """
        # Check runtime cache first (most specific)
        cache_key = cache.get_key(model_path, probe_name, num_samples)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # Check precomputed baselines
        precomputed = self._load_precomputed(model_path, probe_name)
        if precomputed is not None:
            return precomputed

        return None

    def save_baseline(
        self,
        model_path: str,
        probe_name: str,
        num_samples: int,
        score: CategoryScore,
        cache: BaselineCache,
    ) -> None:
        """Save a computed baseline to the runtime cache.

        Args:
            model_path: Model identifier.
            probe_name: Probe set name.
            num_samples: Sample count.
            score: Computed category score.
            cache: BaselineCache instance.
        """
        cache_key = cache.get_key(model_path, probe_name, num_samples)
        cache.set(cache_key, score)

    def _load_precomputed(self, model_path: str, probe_name: str) -> CategoryScore | None:
        """Load a precomputed baseline from the data directory.

        Returns None if no precomputed data exists for this model/probe.
        """
        filename = self.PRECOMPUTED.get(model_path)
        if filename is None:
            return None

        data_file = self._data_dir / filename
        if not data_file.is_file():
            return None

        try:
            with open(data_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        probe_data = data.get(probe_name)
        if probe_data is None:
            return None

        try:
            return CategoryScore.model_validate(probe_data)
        except Exception:
            return None
