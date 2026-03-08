"""FineTuneCheck — automated evaluation of LLM fine-tuning outcomes."""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import TYPE_CHECKING

try:
    __version__ = metadata.version("finetunecheck")
except metadata.PackageNotFoundError:
    __version__ = "0.1.5"

if TYPE_CHECKING:
    from finetunecheck.eval.runner import EvalRunner
    from finetunecheck.forgetting.metrics import (
        backward_transfer,
        capability_retention_rate,
        safety_alignment_retention,
        selective_forgetting_index,
    )


def __getattr__(name: str):
    _lazy_imports = {
        "EvalRunner": "finetunecheck.eval.runner",
        "backward_transfer": "finetunecheck.forgetting.metrics",
        "capability_retention_rate": "finetunecheck.forgetting.metrics",
        "selective_forgetting_index": "finetunecheck.forgetting.metrics",
        "safety_alignment_retention": "finetunecheck.forgetting.metrics",
    }
    if name in _lazy_imports:
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvalRunner",
    "__version__",
    "backward_transfer",
    "capability_retention_rate",
    "safety_alignment_retention",
    "selective_forgetting_index",
]
