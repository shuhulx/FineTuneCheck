"""Versioned, provenance-complete baseline cache."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import diskcache
from diskcache.core import MODE_PICKLE
from pydantic import BaseModel, ConfigDict, Field

from finetunecheck._version import __version__
from finetunecheck.models import (
    METRIC_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    CategoryScore,
    MeasurementStatus,
)

CACHE_MANIFEST_VERSION = "2"
_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.ckpt",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


class _JSONOnlyDisk(diskcache.Disk):
    """Reject pickle-mode cache rows before diskcache can deserialize them."""

    def fetch(self, mode: int, filename: str | None, value: Any, read: bool) -> Any:
        if mode == MODE_PICKLE:
            raise ValueError("FineTuneCheck cache values must be JSON, never pickle")
        return super().fetch(mode, filename, value, read)


class CacheManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: str = CACHE_MANIFEST_VERSION
    model: dict[str, Any]
    tokenizer: dict[str, Any]
    adapter: dict[str, Any] = Field(default_factory=dict)
    probe: dict[str, Any]
    judge: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    inference_backend: str
    execution_policy: dict[str, Any] = Field(default_factory=dict)
    package_version: str = __version__
    result_schema_version: str = RESULT_SCHEMA_VERSION
    metric_schema_version: str = METRIC_SCHEMA_VERSION
    cacheable: bool = True

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_fingerprint(paths: list[Path], root: Path) -> str | None:
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(_hash_file(path).encode())
    return digest.hexdigest()


def model_identity_manifest(
    model_path: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    """Fingerprint local weights/tokenizer or require an explicit remote revision."""
    path = Path(model_path).expanduser()
    if path.exists():
        resolved = path.resolve()
        root = resolved if resolved.is_dir() else resolved.parent
        if resolved.is_file():
            weight_files = [resolved]
        else:
            weight_files = [
                candidate
                for pattern in _WEIGHT_PATTERNS
                for candidate in resolved.glob(pattern)
                if candidate.is_file()
            ]
        tokenizer_files = [root / name for name in _TOKENIZER_FILES if (root / name).is_file()]
        config_files = [
            root / name
            for name in ("config.json", "generation_config.json")
            if (root / name).is_file()
        ]
        adapter_files = [
            root / name
            for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
            if (root / name).is_file()
        ]
        weights_digest = _files_fingerprint(weight_files, root)
        model = {
            "kind": "local",
            "resolved_path": str(resolved),
            "weights_sha256": weights_digest,
            "config_sha256": _files_fingerprint(config_files, root),
        }
        tokenizer_digest = _files_fingerprint(tokenizer_files, root)
        tokenizer = {
            "sha256": tokenizer_digest,
            "files": [path.name for path in tokenizer_files],
        }
        if resolved.is_file() and resolved.suffix.casefold() == ".gguf":
            tokenizer = {
                "embedded_in_model_sha256": weights_digest,
                "files": [resolved.name],
            }
            tokenizer_digest = weights_digest
        adapter = {
            "sha256": _files_fingerprint(adapter_files, root),
            "files": [path.name for path in adapter_files],
        }
        adapter_config = root / "adapter_config.json"
        if adapter_config.is_file():
            try:
                parsed_adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                parsed_adapter = {}
            if isinstance(parsed_adapter, dict):
                adapter["base_model_name_or_path"] = parsed_adapter.get("base_model_name_or_path")
                adapter["revision"] = parsed_adapter.get("revision")
        # Scores are reusable only with both strong weights and tokenizer identity.
        return model, tokenizer, adapter, bool(weights_digest and tokenizer_digest)

    model_id, separator, revision = model_path.partition("@")
    strong_revision = bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", revision))
    model = {
        "kind": "remote",
        "model_id": model_id,
        "resolved_revision": revision or None,
    }
    tokenizer = {
        "source_model_id": model_id,
        "resolved_revision": revision or None,
    }
    # Mutable aliases/tags/branches are deliberately non-cacheable. Only a
    # content-addressed commit identifier is accepted offline.
    return model, tokenizer, {}, bool(separator and strong_revision)


def build_cache_manifest(
    *,
    model_path: str,
    probe_name: str,
    probe_version: str,
    probe_digest: str,
    selected_sample_ids: list[str],
    judge: dict[str, Any],
    generation: dict[str, Any],
    inference_backend: str,
    execution_policy: dict[str, Any],
    adapter_relationship: dict[str, Any] | None = None,
) -> CacheManifest:
    model, tokenizer, adapter, cacheable = model_identity_manifest(model_path)
    if adapter_relationship:
        adapter.update(adapter_relationship)
    return CacheManifest(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        probe={
            "name": probe_name,
            "version": probe_version,
            "content_sha256": probe_digest,
            "selected_sample_ids": selected_sample_ids,
        },
        judge=judge,
        generation=generation,
        inference_backend=inference_backend,
        execution_policy=execution_policy,
        cacheable=cacheable,
    )


class BaselineCache:
    def __init__(self, cache_dir: str = "~/.cache/finetunecheck") -> None:
        resolved = Path(cache_dir).expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(resolved), disk=_JSONOnlyDisk)

    def get_key(
        self,
        model_or_manifest: str | CacheManifest,
        probe_name: str | None = None,
        num_samples: int | None = None,
    ) -> str:
        """Return a v2 manifest key; legacy arguments get a safe local fingerprint."""
        if isinstance(model_or_manifest, CacheManifest):
            return f"v{CACHE_MANIFEST_VERSION}:{model_or_manifest.digest()}"
        if probe_name is None or num_samples is None:
            raise TypeError("probe_name and num_samples are required with a model path")
        model, tokenizer, adapter, cacheable = model_identity_manifest(model_or_manifest)
        manifest = CacheManifest(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            probe={"name": probe_name, "legacy_sample_count": num_samples},
            inference_backend="legacy-unknown",
            cacheable=cacheable,
        )
        return f"v{CACHE_MANIFEST_VERSION}:{manifest.digest()}"

    def get(self, key_or_manifest: str | CacheManifest) -> CategoryScore | None:
        manifest: CacheManifest | None = None
        if isinstance(key_or_manifest, CacheManifest):
            if not key_or_manifest.cacheable:
                return None
            manifest = key_or_manifest
            key = self.get_key(key_or_manifest)
        else:
            key = key_or_manifest
        try:
            raw = self._cache.get(key)
        except Exception:
            self._cache.delete(key)
            return None
        if raw is None:
            return None
        if not isinstance(raw, (str, bytes, bytearray)):
            self._cache.delete(key)
            return None
        try:
            score = CategoryScore.model_validate_json(raw)
        except Exception:
            self._cache.delete(key)
            return None
        if score.status != MeasurementStatus.MEASURED:
            self._cache.delete(key)
            return None
        if manifest is not None and not self._matches_manifest(score, manifest):
            self._cache.delete(key)
            return None
        return score

    def set(self, key_or_manifest: str | CacheManifest, score: CategoryScore) -> None:
        # Failed and incomplete measurements remain visible in the current result,
        # but must be retried rather than persisted as reusable baseline evidence.
        if score.status != MeasurementStatus.MEASURED:
            return
        if isinstance(key_or_manifest, CacheManifest):
            if not key_or_manifest.cacheable:
                return
            if not self._matches_manifest(score, key_or_manifest):
                raise ValueError("Category score evidence does not match the cache manifest")
            key = self.get_key(key_or_manifest)
        else:
            key = key_or_manifest
        # diskcache transactions commit the value atomically.
        with self._cache.transact():
            self._cache.set(key, score.model_dump_json())

    @staticmethod
    def _matches_manifest(score: CategoryScore, manifest: CacheManifest) -> bool:
        probe_name = manifest.probe.get("name")
        probe_digest = manifest.probe.get("content_sha256")
        selected_ids = manifest.probe.get("selected_sample_ids")
        if (
            not isinstance(probe_name, str)
            or not isinstance(probe_digest, str)
            or not isinstance(selected_ids, list)
            or any(not isinstance(sample_id, str) for sample_id in selected_ids)
        ):
            return False
        verdict_ids = [verdict.sample_id for verdict in score.sample_verdicts]
        measured_scores = [
            verdict.score
            for verdict in score.sample_verdicts
            if verdict.status == MeasurementStatus.MEASURED and verdict.score is not None
        ]
        expected_mean = mean(measured_scores) if measured_scores else None
        expected_std = stdev(measured_scores) if len(measured_scores) > 1 else 0.0
        aggregate_matches = (
            score.mean_score is None
            if expected_mean is None
            else score.mean_score is not None
            and score.std_score is not None
            and math.isclose(score.mean_score, expected_mean, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(score.std_score, expected_std, rel_tol=1e-12, abs_tol=1e-12)
        )
        return (
            score.status == MeasurementStatus.MEASURED
            and score.category == probe_name
            and score.probe_digest == probe_digest
            and score.selected_sample_ids == selected_ids
            and score.expected_samples == len(selected_ids)
            and verdict_ids == selected_ids
            and score.num_samples == len(measured_scores)
            and score.sample_scores == measured_scores
            and aggregate_matches
            and score.num_samples == len(selected_ids) > 0
        )

    def has(self, model_path: str, probe_name: str, num_samples: int) -> bool:
        return self.get_key(model_path, probe_name, num_samples) in self._cache

    def invalidate(self, manifest: CacheManifest) -> bool:
        return bool(self._cache.delete(self.get_key(manifest)))

    def clear(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        self._cache.close()
