"""Activation drift and attention head disruption detection.

Measures per-layer cosine similarity of hidden states between base and fine-tuned
models on the same inputs. Low similarity = high drift = potential forgetting.

Also identifies "disrupted" attention heads where attention patterns diverged
significantly, which correlates with capability loss.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from finetunecheck.models import ActivationDriftReport, MeasurementStatus
from finetunecheck.utils.device import resolve_model_device

if TYPE_CHECKING:
    from finetunecheck.utils.model_loader import AnalysisModel

logger = logging.getLogger(__name__)


class ActivationDriftAnalyzer:
    """Measure per-layer activation drift and per-head attention disruption.

    Computes cosine similarity of hidden states between base and fine-tuned
    models on identical inputs. Also identifies attention heads where the
    attention distribution diverged beyond a configurable threshold.
    """

    def __init__(
        self,
        num_samples: int = 256,
        batch_size: int = 32,
        head_threshold: float = 0.7,
        max_length: int = 512,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.head_threshold = head_threshold
        self.max_length = max_length
        self._attention_available = False

    def _collect_hidden_states_pooled(
        self,
        model: AnalysisModel,
        texts: list[str],
        device: str,
    ) -> dict[int, torch.Tensor]:
        """Collect mean-pooled hidden states per layer.

        Args:
            model: AnalysisModel.
            texts: Input texts.
            device: Device string.

        Returns:
            Dict layer_idx -> tensor of shape [num_samples, hidden_dim].
        """
        all_states: dict[int, list[torch.Tensor]] = {}

        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]
            encodings = model.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            hidden_states = model.get_hidden_states(input_ids, attention_mask=attention_mask)
            mask_expanded = attention_mask.unsqueeze(-1).float()

            for layer_idx, states in hidden_states.items():
                masked = states * mask_expanded
                counts = mask_expanded.sum(dim=1).clamp(min=1)
                pooled = masked.sum(dim=1) / counts  # (batch, hidden_dim)

                if layer_idx not in all_states:
                    all_states[layer_idx] = []
                all_states[layer_idx].append(pooled.cpu().float())

        return {idx: torch.cat(parts, dim=0) for idx, parts in all_states.items()}

    def compute_layer_drift(
        self,
        base_states: dict[int, torch.Tensor],
        ft_states: dict[int, torch.Tensor],
    ) -> dict[str, float]:
        """Compute mean cosine similarity per layer between base and ft activations.

        Args:
            base_states: Layer -> [n, hidden_dim] from base model.
            ft_states: Layer -> [n, hidden_dim] from fine-tuned model.

        Returns:
            Dict "layer_N" -> mean cosine similarity (1.0 = identical, 0.0 = orthogonal).
        """
        drift: dict[str, float] = {}
        common_layers = sorted(set(base_states.keys()) & set(ft_states.keys()))

        for layer_idx in common_layers:
            base = base_states[layer_idx]
            ft = ft_states[layer_idx]
            n = min(base.shape[0], ft.shape[0])
            cos_sim = F.cosine_similarity(base[:n], ft[:n], dim=-1)
            drift[f"layer_{layer_idx}"] = float(cos_sim.mean().item())

        return drift

    def find_disrupted_heads(
        self,
        base_model: AnalysisModel,
        ft_model: AnalysisModel,
        texts: list[str],
        base_device: str,
        ft_device: str,
    ) -> list[dict]:
        """Find attention heads where patterns diverged significantly.

        For each layer and head, computes mean cosine similarity of attention
        weight vectors across samples. Heads below self.head_threshold are
        flagged as disrupted.

        Args:
            base_model: AnalysisModel for the base model.
            ft_model: AnalysisModel for the fine-tuned model.
            texts: Input texts (a small subset is used).
            base_device: Device for base model.
            ft_device: Device for fine-tuned model.

        Returns:
            List of dicts with keys: layer, head, cosine_sim.
        """
        # Use a small subset for attention analysis (memory-intensive)
        subset = texts[: min(32, len(texts))]
        if not subset:
            return []

        # Collect per-head attention similarities across batches
        head_sims: dict[tuple[int, int], list[float]] = {}

        for batch_start in range(0, len(subset), self.batch_size):
            batch_texts = subset[batch_start : batch_start + self.batch_size]

            # Tokenize with base model tokenizer (both should share tokenizer)
            encodings = base_model.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids_base = encodings["input_ids"].to(base_device)
            input_ids_ft = encodings["input_ids"].to(ft_device)

            try:
                base_mask = encodings["attention_mask"].to(base_device)
                ft_mask = encodings["attention_mask"].to(ft_device)
                base_attn = base_model.get_attention_weights(
                    input_ids_base, attention_mask=base_mask
                )
                ft_attn = ft_model.get_attention_weights(input_ids_ft, attention_mask=ft_mask)
            except Exception as exc:
                logger.warning("Attention weight collection failed: %s", exc)
                return []

            common_attn_layers = sorted(set(base_attn.keys()) & set(ft_attn.keys()))

            for layer_idx in common_attn_layers:
                base_w = base_attn[layer_idx].cpu().float()
                ft_w = ft_attn[layer_idx].cpu().float()

                # Shape: (batch, num_heads, seq_len, seq_len)
                if base_w.dim() != 4 or ft_w.dim() != 4:
                    continue

                n = min(base_w.shape[0], ft_w.shape[0])
                num_heads = min(base_w.shape[1], ft_w.shape[1])
                seq_len = min(base_w.shape[2], ft_w.shape[2])

                base_w = base_w[:n, :num_heads, :seq_len, :seq_len]
                ft_w = ft_w[:n, :num_heads, :seq_len, :seq_len]

                for head_idx in range(num_heads):
                    # Flatten attention pattern per sample: (n, seq*seq)
                    base_flat = base_w[:, head_idx].reshape(n, -1)
                    ft_flat = ft_w[:, head_idx].reshape(n, -1)

                    cos = F.cosine_similarity(base_flat, ft_flat, dim=-1)
                    key = (layer_idx, head_idx)
                    if key not in head_sims:
                        head_sims[key] = []
                    head_sims[key].append(float(cos.mean().item()))

        self._attention_available = bool(head_sims)

        # Aggregate and filter disrupted heads
        disrupted: list[dict] = []
        for (layer_idx, head_idx), sim_values in head_sims.items():
            mean_sim = float(np.mean(sim_values))
            if mean_sim < self.head_threshold:
                disrupted.append({"layer": layer_idx, "head": head_idx, "cosine_sim": mean_sim})

        # Sort by most disrupted first
        disrupted.sort(key=lambda x: x["cosine_sim"])
        return disrupted

    def analyze(
        self,
        base_model: AnalysisModel,
        ft_model: AnalysisModel,
        sample_texts: list[str],
    ) -> ActivationDriftReport:
        """Full activation drift analysis.

        Args:
            base_model: AnalysisModel for the base model.
            ft_model: AnalysisModel for the fine-tuned model.
            sample_texts: Texts for computing activations.

        Returns:
            ActivationDriftReport with per-layer drift and disrupted heads.
        """
        texts = sample_texts[: self.num_samples]
        if not texts:
            raise ValueError("sample_texts must be non-empty.")

        base_device = resolve_model_device(base_model)
        ft_device = resolve_model_device(ft_model)

        logger.info("Computing activation drift over %d samples...", len(texts))
        base_states = self._collect_hidden_states_pooled(base_model, texts, base_device)
        ft_states = self._collect_hidden_states_pooled(ft_model, texts, ft_device)

        per_layer_sim = self.compute_layer_drift(base_states, ft_states)
        if not per_layer_sim:
            raise ValueError(
                "No compatible hidden-state layers were available for activation drift"
            )

        logger.info("Scanning for disrupted attention heads...")
        disrupted_heads = self.find_disrupted_heads(
            base_model, ft_model, texts, base_device, ft_device
        )

        # Mean drift = 1 - mean_cosine_sim (so higher = more drift)
        sim_values = list(per_layer_sim.values())
        mean_drift = 1.0 - float(np.mean(sim_values)) if sim_values else 0.0

        logger.info(
            "Activation drift complete: mean_drift=%.4f, %d disrupted heads",
            mean_drift,
            len(disrupted_heads),
        )

        attention_status = (
            MeasurementStatus.MEASURED if self._attention_available else MeasurementStatus.NOT_RUN
        )
        return ActivationDriftReport(
            per_layer_cosine_sim=per_layer_sim,
            disrupted_heads=disrupted_heads,
            mean_drift=mean_drift,
            attention_status=attention_status,
            attention_error=(
                None
                if self._attention_available
                else "No comparable attention weights were exposed by both models"
            ),
        )
