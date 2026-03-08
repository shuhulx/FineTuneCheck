"""JSON Schema definitions for MCP tool inputs."""

from __future__ import annotations

EVALUATE_FINETUNE_SCHEMA = {
    "type": "object",
    "properties": {
        "base_model": {
            "type": "string",
            "description": "Base model path or HuggingFace ID",
        },
        "finetuned_model": {
            "type": "string",
            "description": "Fine-tuned model path or HuggingFace ID",
        },
        "target_task": {
            "type": "string",
            "description": "Target task name (optional)",
        },
        "num_samples": {
            "type": "integer",
            "description": "Number of samples per probe (default: 100)",
            "default": 100,
        },
        "deep_analysis": {
            "type": "boolean",
            "description": "Enable deep analysis (CKA, spectral, etc.)",
            "default": False,
        },
        "device": {
            "type": "string",
            "description": "Device: auto, cpu, cuda, mps",
            "default": "auto",
        },
        "profile": {
            "type": "string",
            "description": "Evaluation profile (e.g. code, chat, safety)",
        },
    },
    "required": ["base_model", "finetuned_model"],
}

QUICK_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "base_model": {
            "type": "string",
            "description": "Base model path or HuggingFace ID",
        },
        "finetuned_model": {
            "type": "string",
            "description": "Fine-tuned model path or HuggingFace ID",
        },
        "device": {
            "type": "string",
            "description": "Device: auto, cpu, cuda, mps",
            "default": "auto",
        },
    },
    "required": ["base_model", "finetuned_model"],
}

DETECT_FORGETTING_SCHEMA = {
    "type": "object",
    "properties": {
        "base_model": {
            "type": "string",
            "description": "Base model path or HuggingFace ID",
        },
        "finetuned_model": {
            "type": "string",
            "description": "Fine-tuned model path or HuggingFace ID",
        },
        "num_samples": {
            "type": "integer",
            "description": "Number of samples per probe",
            "default": 100,
        },
        "device": {
            "type": "string",
            "default": "auto",
        },
    },
    "required": ["base_model", "finetuned_model"],
}

COMPARE_RUNS_SCHEMA = {
    "type": "object",
    "properties": {
        "base_model": {
            "type": "string",
            "description": "Base model path or HuggingFace ID",
        },
        "finetuned_models": {
            "type": "object",
            "description": "Mapping of run name to model path/ID",
            "additionalProperties": {"type": "string"},
        },
        "target_task": {
            "type": "string",
            "description": "Target task name (optional)",
        },
        "num_samples": {
            "type": "integer",
            "default": 100,
        },
        "device": {
            "type": "string",
            "default": "auto",
        },
    },
    "required": ["base_model", "finetuned_models"],
}

GET_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "base_model": {
            "type": "string",
            "description": "Base model path or HuggingFace ID",
        },
        "finetuned_model": {
            "type": "string",
            "description": "Fine-tuned model path or HuggingFace ID",
        },
        "device": {
            "type": "string",
            "default": "auto",
        },
    },
    "required": ["base_model", "finetuned_model"],
}

SUGGEST_FIXES_SCHEMA = {
    "type": "object",
    "properties": {
        "base_model": {
            "type": "string",
            "description": "Base model path or HuggingFace ID",
        },
        "finetuned_model": {
            "type": "string",
            "description": "Fine-tuned model path or HuggingFace ID",
        },
        "device": {
            "type": "string",
            "default": "auto",
        },
    },
    "required": ["base_model", "finetuned_model"],
}

GENERATE_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "base_model": {
            "type": "string",
            "description": "Base model path or HuggingFace ID",
        },
        "finetuned_model": {
            "type": "string",
            "description": "Fine-tuned model path or HuggingFace ID",
        },
        "output_path": {
            "type": "string",
            "description": "Output file path for the report",
        },
        "format": {
            "type": "string",
            "description": "Report format: html, json, csv, markdown",
            "default": "html",
            "enum": ["html", "json", "csv", "markdown"],
        },
        "device": {
            "type": "string",
            "default": "auto",
        },
    },
    "required": ["base_model", "finetuned_model", "output_path"],
}

LIST_PROFILES_SCHEMA = {
    "type": "object",
    "properties": {},
}

RUN_PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {
            "type": "string",
            "description": "Model path or HuggingFace ID to probe",
        },
        "probe_name": {
            "type": "string",
            "description": "Name of the probe set to run",
        },
        "num_samples": {
            "type": "integer",
            "description": "Max samples to run (default: all)",
        },
        "device": {
            "type": "string",
            "default": "auto",
        },
    },
    "required": ["model", "probe_name"],
}
