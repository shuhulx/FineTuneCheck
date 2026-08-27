"""Model loading utilities for inference and deep analysis."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

from finetunecheck.models import ModelSpec, ModelType
from finetunecheck.utils.device import detect_device

if TYPE_CHECKING:
    import torch
    from torch import Tensor


class AnalysisModel:
    """Wraps a transformers model for deep analysis (hidden states, attention, logits)."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def get_hidden_states(
        self,
        input_ids: Tensor,
        layers: list[int] | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[int, Tensor]:
        """Capture hidden states from specified layers using forward hooks.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len).
            layers: Layer indices to capture. None means all layers.

        Returns:
            Dict mapping layer index to hidden state tensor.
        """
        hidden_states: dict[int, Tensor] = {}
        import torch

        hooks = []

        target_layers = self._get_transformer_layers()
        if layers is not None:
            target_layers = {i: layer for i, layer in target_layers.items() if i in layers}

        for idx, layer in target_layers.items():

            def _make_hook(layer_idx: int):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states[layer_idx] = output[0].detach()
                    else:
                        hidden_states[layer_idx] = output.detach()

                return hook_fn

            hooks.append(layer.register_forward_hook(_make_hook(idx)))

        try:
            with torch.no_grad():
                self.model(input_ids, attention_mask=attention_mask)
        finally:
            for h in hooks:
                h.remove()

        return hidden_states

    def get_logits(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Run forward pass and return logits.

        Args:
            input_ids: Token IDs tensor.

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        import torch

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

    def get_attention_weights(
        self,
        input_ids: Tensor,
        layers: list[int] | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[int, Tensor]:
        """Capture attention weights from specified layers using forward hooks.

        Args:
            input_ids: Token IDs tensor.
            layers: Layer indices. None means all.

        Returns:
            Dict mapping layer index to attention weight tensor.
        """
        attention_weights: dict[int, Tensor] = {}
        import torch

        hooks = []

        attn_modules = self._get_attention_modules()
        if layers is not None:
            attn_modules = {i: m for i, m in attn_modules.items() if i in layers}

        for idx, attn_mod in attn_modules.items():

            def _make_hook(layer_idx: int):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        attention_weights[layer_idx] = output[1].detach()

                return hook_fn

            hooks.append(attn_mod.register_forward_hook(_make_hook(idx)))

        try:
            with torch.no_grad():
                self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
        finally:
            for h in hooks:
                h.remove()

        return attention_weights

    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Iterate over model parameters."""
        return self.model.named_parameters()

    def _get_transformer_layers(self) -> dict[int, Any]:
        """Find the sequential transformer layers in the model."""
        model = self.model
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "layers"):
            layers_list = model.layers
        elif hasattr(model, "h"):
            layers_list = model.h
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layers_list = model.encoder.layer
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers_list = model.transformer.h
        else:
            layers_list = []
            for name, module in model.named_modules():
                if "layer" in name.lower() and name.count(".") == 1:
                    layers_list.append(module)

        return {i: layer for i, layer in enumerate(layers_list)}

    def _get_attention_modules(self) -> dict[int, Any]:
        """Find attention sub-modules inside each transformer layer."""
        layers = self._get_transformer_layers()
        attn_modules = {}
        for idx, layer in layers.items():
            for name, child in layer.named_children():
                if "attn" in name.lower() or "attention" in name.lower():
                    attn_modules[idx] = child
                    break
        return attn_modules


class ModelLoader:
    """Static methods to detect model type and load for inference or analysis."""

    @staticmethod
    def detect_type(path: str) -> ModelSpec:
        """Detect model type from path and return a ModelSpec.

        Rules:
            - If path ends with .gguf -> GGUF
            - If path contains adapter_config.json -> LoRA (base_model read from config)
            - Else -> HF
        """
        if path.startswith("peft://"):
            adapter_reference = path.removeprefix("peft://")
            adapter_id, separator, revision = adapter_reference.rpartition("@")
            if not separator:
                adapter_id = adapter_reference
                revision = ""
            if not adapter_id:
                raise ValueError("peft:// model identifiers must not be empty")
            from peft import PeftConfig

            adapter_config = PeftConfig.from_pretrained(
                adapter_id,
                revision=revision or None,
            )
            return ModelSpec(
                path=adapter_id,
                model_type=ModelType.LORA,
                base_model=adapter_config.base_model_name_or_path,
                revision=revision or None,
            )

        if path.endswith(".gguf"):
            return ModelSpec(path=path, model_type=ModelType.GGUF)

        adapter_config_path = os.path.join(path, "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            import json

            try:
                with open(adapter_config_path) as f:
                    adapter_cfg = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {adapter_config_path}: {exc}") from exc
            if not isinstance(adapter_cfg, dict):
                raise ValueError(
                    f"Expected a JSON object in {adapter_config_path}, "
                    f"got {type(adapter_cfg).__name__}"
                )
            base_model = adapter_cfg.get("base_model_name_or_path", None)
            return ModelSpec(path=path, model_type=ModelType.LORA, base_model=base_model)

        if not os.path.exists(path):
            model_id, separator, revision = path.rpartition("@")
            if separator and model_id and revision:
                return ModelSpec(
                    path=model_id,
                    model_type=ModelType.HF,
                    revision=revision,
                )

        return ModelSpec(path=path, model_type=ModelType.HF)

    @staticmethod
    def load_for_inference(spec: ModelSpec, device: str = "auto"):
        """Load model and return an appropriate InferenceBackend.

        Priority: vLLM > transformers > llama.cpp.
        """
        from finetunecheck.eval.inference import create_backend

        return create_backend(spec, device)

    @staticmethod
    def load_for_analysis(spec: ModelSpec, device: str = "auto") -> AnalysisModel:
        """Load model via transformers for deep analysis (hidden states, etc.).

        For LoRA models, loads base + adapter via PEFT without merging.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = detect_device(device)
        dtype = torch.float32 if resolved_device == "cpu" else torch.float16
        device_map = "auto" if device == "auto" and resolved_device != "cpu" else None

        if spec.model_type == ModelType.GGUF:
            raise ValueError(
                "GGUF models cannot be loaded for analysis. "
                "Use the original HF/LoRA format for deep analysis."
            )

        if spec.model_type == ModelType.LORA:
            if not spec.base_model:
                raise ValueError(
                    "LoRA adapter detected but no base_model found in adapter_config.json."
                )
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                spec.base_model,
                torch_dtype=dtype,
                device_map=device_map,
            )
            adapter_kwargs = {"revision": spec.revision} if spec.revision else {}
            model: Any = cast(Any, PeftModel).from_pretrained(base, spec.path, **adapter_kwargs)
            if device_map is None:
                model = model.to(resolved_device)
            tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                spec.path,
                torch_dtype=dtype,
                device_map=device_map,
                **({"revision": spec.revision} if spec.revision else {}),
            )
            if device_map is None:
                model = model.to(resolved_device)
            tokenizer = AutoTokenizer.from_pretrained(
                spec.path,
                **({"revision": spec.revision} if spec.revision else {}),
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return AnalysisModel(model, tokenizer)
