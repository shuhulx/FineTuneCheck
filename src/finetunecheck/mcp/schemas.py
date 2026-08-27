"""Strict JSON Schemas for the supported MCP SDK 1.x API."""

from __future__ import annotations

_NON_EMPTY_STRING = {"type": "string", "minLength": 1}
_DEVICE = {"type": "string", "enum": ["auto", "cpu", "cuda", "mps"], "default": "auto"}
_POSITIVE_SAMPLES = {"type": "integer", "minimum": 1}
_JUDGE = {
    "type": "object",
    "properties": {
        "provider": {"type": "string", "enum": ["local", "openai", "anthropic"]},
        "model": _NON_EMPTY_STRING,
        "api_key_env": _NON_EMPTY_STRING,
        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0},
        "max_tokens": {"type": "integer", "minimum": 1, "default": 256},
        "settings": {"type": "object", "additionalProperties": True},
    },
    "required": ["provider", "model"],
    "additionalProperties": False,
}


def _object(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_PAIRED_PROPERTIES = {
    "base_model": {**_NON_EMPTY_STRING, "description": "Base model path or ID"},
    "finetuned_model": {**_NON_EMPTY_STRING, "description": "Fine-tuned model path or ID"},
    "target_tasks": {
        "type": "array",
        "items": _NON_EMPTY_STRING,
        "uniqueItems": True,
        "description": "Canonical target probe categories",
    },
    "target_task": {
        **_NON_EMPTY_STRING,
        "deprecated": True,
        "description": "Deprecated single-target compatibility alias",
    },
    "num_samples": {**_POSITIVE_SAMPLES, "default": 100},
    "device": _DEVICE,
    "judge": _JUDGE,
}

EVALUATE_FINETUNE_SCHEMA = _object(
    {
        **_PAIRED_PROPERTIES,
        "deep_analysis": {"type": "boolean", "default": False},
        "deep_analysis_samples": {**_POSITIVE_SAMPLES, "default": 50},
        "profile": _NON_EMPTY_STRING,
    },
    ["base_model", "finetuned_model"],
)

QUICK_CHECK_SCHEMA = _object(
    {
        "base_model": _NON_EMPTY_STRING,
        "finetuned_model": _NON_EMPTY_STRING,
        "target_tasks": {
            "type": "array",
            "items": _NON_EMPTY_STRING,
            "uniqueItems": True,
        },
        "device": _DEVICE,
    },
    ["base_model", "finetuned_model"],
)

DETECT_FORGETTING_SCHEMA = _object(_PAIRED_PROPERTIES, ["base_model", "finetuned_model"])

COMPARE_RUNS_SCHEMA = _object(
    {
        "base_model": _NON_EMPTY_STRING,
        "finetuned_models": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": _NON_EMPTY_STRING,
        },
        "target_tasks": {
            "type": "array",
            "items": _NON_EMPTY_STRING,
            "uniqueItems": True,
        },
        "target_task": {**_NON_EMPTY_STRING, "deprecated": True},
        "num_samples": {**_POSITIVE_SAMPLES, "default": 100},
        "device": _DEVICE,
        "judge": _JUDGE,
    },
    ["base_model", "finetuned_models"],
)

GET_VERDICT_SCHEMA = _object(
    {
        "base_model": _NON_EMPTY_STRING,
        "finetuned_model": _NON_EMPTY_STRING,
        "target_tasks": {
            "type": "array",
            "items": _NON_EMPTY_STRING,
            "uniqueItems": True,
        },
        "device": _DEVICE,
    },
    ["base_model", "finetuned_model"],
)

SUGGEST_FIXES_SCHEMA = EVALUATE_FINETUNE_SCHEMA

GENERATE_REPORT_SCHEMA = _object(
    {
        **_PAIRED_PROPERTIES,
        "output_path": _NON_EMPTY_STRING,
        "format": {
            "type": "string",
            "enum": ["html", "json", "csv", "markdown"],
            "default": "html",
        },
        "overwrite": {"type": "boolean", "default": False},
    },
    ["base_model", "finetuned_model", "output_path"],
)

LIST_PROFILES_SCHEMA = _object({})

RUN_PROBE_SCHEMA = _object(
    {
        "model": _NON_EMPTY_STRING,
        "probe_name": _NON_EMPTY_STRING,
        "num_samples": _POSITIVE_SAMPLES,
        "device": _DEVICE,
        "judge": _JUDGE,
    },
    ["model", "probe_name"],
)
