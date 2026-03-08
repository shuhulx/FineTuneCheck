"""CKA (Centered Kernel Alignment) layer-by-layer representation similarity analysis.

Measures how much each layer's representational geometry changed during fine-tuning.
Produces a "forgetting heatmap" showing exactly which layers diverged most.

Reference: Kornblith et al., "Similarity of Neural Network Representations Revisited"
(ICML 2019).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from finetunecheck.models import CKAReport
from finetunecheck.utils.device import resolve_model_device

if TYPE_CHECKING:
    from finetunecheck.utils.model_loader import AnalysisModel

logger = logging.getLogger(__name__)


class CKAAnalyzer:
    """Layer-by-layer CKA between base and fine-tuned model representations.

    CKA measures representational similarity independent of rotation/scaling.
    CKA=1.0 means identical representations, CKA<0.8 indicates significant
    divergence in that layer's learned features.
    """

    def __init__(self, num_samples: int = 256, batch_size: int = 32) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size

    @staticmethod
    def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """Compute linear CKA between two representation matrices.

        CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
        where K = X @ X^T and L = Y @ Y^T (linear kernels).

        For computational efficiency, we use the equivalent formulation:
        HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2  (after centering).

        Args:
            X: Activation matrix of shape [n_samples, n_features_x].
            Y: Activation matrix of shape [n_samples, n_features_y].

        Returns:
            CKA similarity score in [0, 1].
        """
        assert X.shape[0] == Y.shape[0], f"Sample counts must match: {X.shape[0]} vs {Y.shape[0]}"

        # Center both matrices (subtract column means)
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # HSIC via cross-covariance formulation
        # HSIC(X, Y) proportional to ||X^T Y||_F^2
        hsic_xy = torch.norm(X.T @ Y, p="fro").pow(2)
        hsic_xx = torch.norm(X.T @ X, p="fro").pow(2)
        hsic_yy = torch.norm(Y.T @ Y, p="fro").pow(2)

        denom = torch.sqrt(hsic_xx * hsic_yy)
        if denom.item() < 1e-10:
            return 0.0
        return float((hsic_xy / denom).item())

    def collect_hidden_states(
        self,
        model: AnalysisModel,
        texts: list[str],
        device: str,
    ) -> dict[int, torch.Tensor]:
        """Collect hidden states from all layers, mean-pooled over sequence length.

        Args:
            model: AnalysisModel wrapping the torch model.
            texts: Input texts to process.
            device: Device string.

        Returns:
            Dict mapping layer_idx -> tensor of shape [num_samples, hidden_dim].
        """
        all_states: dict[int, list[torch.Tensor]] = {}

        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]
            encodings = model.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # Use AnalysisModel's hook-based hidden state collection
            hidden_states = model.get_hidden_states(input_ids)

            # Mean-pool over sequence length (respecting attention mask)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)

            for layer_idx, states in hidden_states.items():
                # states shape: (batch, seq_len, hidden_dim)
                masked_states = states * mask_expanded
                token_counts = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)
                pooled = masked_states.sum(dim=1) / token_counts  # (batch, hidden_dim)

                if layer_idx not in all_states:
                    all_states[layer_idx] = []
                all_states[layer_idx].append(pooled.cpu().float())

        result: dict[int, torch.Tensor] = {}
        for layer_idx, tensors in all_states.items():
            result[layer_idx] = torch.cat(tensors, dim=0)

        return result

    def analyze(
        self,
        base_model: AnalysisModel,
        ft_model: AnalysisModel,
        sample_texts: list[str],
    ) -> CKAReport:
        """Full CKA analysis across all layers.

        Collects hidden states from both models on the same inputs,
        then computes CKA per layer.

        Args:
            base_model: AnalysisModel for the base model.
            ft_model: AnalysisModel for the fine-tuned model.
            sample_texts: Texts to use for collecting activations.

        Returns:
            CKAReport with per-layer CKA scores and divergence analysis.
        """
        texts = sample_texts[: self.num_samples]
        if not texts:
            raise ValueError("sample_texts must be non-empty.")

        logger.info("Collecting hidden states from base model (%d samples)...", len(texts))
        base_device = resolve_model_device(base_model)
        base_states = self.collect_hidden_states(base_model, texts, base_device)

        logger.info("Collecting hidden states from fine-tuned model...")
        ft_device = resolve_model_device(ft_model)
        ft_states = self.collect_hidden_states(ft_model, texts, ft_device)

        per_layer_cka: dict[str, float] = {}
        common_layers = sorted(set(base_states.keys()) & set(ft_states.keys()))

        for layer_idx in common_layers:
            base_act = base_states[layer_idx]
            ft_act = ft_states[layer_idx]

            # Ensure same sample count (should be, but guard)
            n = min(base_act.shape[0], ft_act.shape[0])
            cka_score = self.linear_cka(base_act[:n], ft_act[:n])
            per_layer_cka[f"layer_{layer_idx}"] = cka_score

        if not per_layer_cka:
            logger.warning("No common layers found between models for CKA analysis.")
            return CKAReport(per_layer_cka={}, most_diverged_layers=[], mean_cka=0.0)

        sorted_layers = sorted(per_layer_cka.items(), key=lambda x: x[1])
        most_diverged = [name for name, score in sorted_layers[:5] if score < 0.9]

        mean_cka = float(np.mean(list(per_layer_cka.values())))

        logger.info(
            "CKA analysis complete: mean=%.4f, %d diverged layers",
            mean_cka,
            len(most_diverged),
        )

        return CKAReport(
            per_layer_cka=per_layer_cka,
            most_diverged_layers=most_diverged,
            mean_cka=mean_cka,
        )
