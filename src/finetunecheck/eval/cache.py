"""Baseline caching using diskcache for fast re-evaluation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import diskcache

from finetunecheck.models import CategoryScore


class BaselineCache:
    def __init__(self, cache_dir: str = "~/.cache/finetunecheck") -> None:
        resolved = Path(cache_dir).expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(resolved))

    def get_key(self, model_path: str, probe_name: str, num_samples: int) -> str:
        """Generate cache key from model content hash + probe + sample count.

        Uses the model's config.json hash if available, otherwise hashes the path.
        """
        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        else:
            content_hash = hashlib.sha256(model_path.encode()).hexdigest()[:16]
        return f"{content_hash}:{probe_name}:{num_samples}"

    def get(self, key: str) -> CategoryScore | None:
        """Retrieve cached score, or None if not found."""
        raw = self._cache.get(key)
        if raw is None:
            return None
        try:
            return CategoryScore.model_validate_json(raw)
        except Exception:
            return None

    def set(self, key: str, score: CategoryScore) -> None:
        """Store a category score in cache."""
        self._cache.set(key, score.model_dump_json())

    def has(self, model_path: str, probe_name: str, num_samples: int) -> bool:
        """Check if a cached score exists."""
        key = self.get_key(model_path, probe_name, num_samples)
        return key in self._cache

    def clear(self) -> None:
        """Clear all cached baselines."""
        self._cache.clear()

    def close(self) -> None:
        """Close the cache."""
        self._cache.close()
