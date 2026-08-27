"""Auto device detection for torch backends."""

from __future__ import annotations

from typing import Any, cast


def detect_device(preference: str = "auto") -> str:
    """Detect the best available torch device.

    Args:
        preference: One of "auto", "cuda", "mps", "cpu". If "auto", picks the
                    best available. If a specific device is requested but
                    unavailable, falls back to cpu with a warning.

    Returns:
        Device string suitable for torch.device().
    """
    import torch

    if preference != "auto":
        if preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        if (
            preference == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"
        if preference == "cpu":
            return "cpu"
        if preference not in ("cuda", "mps", "cpu"):
            return preference
        import warnings

        warnings.warn(
            f"Requested device '{preference}' is not available, falling back to cpu.",
            stacklevel=2,
        )
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_device(model: object) -> str:
    """Extract device string from a loaded model, defaulting to 'cpu'."""
    model_with_attributes = cast(Any, model)
    if hasattr(model, "device"):
        return str(model_with_attributes.device)
    try:
        return str(next(model_with_attributes.model.parameters()).device)
    except (StopIteration, AttributeError):
        return "cpu"


def get_device_info() -> dict:
    """Return a dictionary of device information.

    Returns:
        Dict with keys: device, device_name, memory_total_gb, memory_available_gb,
        cuda_version (if applicable).
    """
    import torch

    device = detect_device()
    info: dict = {"device": device}

    if device == "cuda":
        idx = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_mem
        info["memory_total_gb"] = round(total / (1024**3), 2)
        free, _ = torch.cuda.mem_get_info(idx)
        info["memory_available_gb"] = round(free / (1024**3), 2)
        info["cuda_version"] = torch.version.cuda or "unknown"
    elif device == "mps":
        info["device_name"] = "Apple Silicon (MPS)"
        try:
            allocated = torch.mps.current_allocated_memory()
            info["memory_allocated_gb"] = round(allocated / (1024**3), 2)
        except Exception:
            info["memory_allocated_gb"] = None
    else:
        info["device_name"] = "CPU"
        try:
            import os

            import psutil

            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024**3), 2)
            info["memory_available_gb"] = round(mem.available / (1024**3), 2)
        except ImportError:
            cpu_count = os.cpu_count() or 1
            info["cpu_count"] = cpu_count

    return info
