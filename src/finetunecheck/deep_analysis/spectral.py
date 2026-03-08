"""SVD analysis of weight-space perturbations from fine-tuning.

Decomposes the weight delta (dW = W_ft - W_base) to characterize the "shape"
of fine-tuning: how many directions changed (effective rank), how much they
changed (Frobenius norm), and where the change is concentrated (top singular
values).

High effective rank = broad, distributed changes (riskier for forgetting).
Low effective rank = concentrated changes (typical of LoRA, generally safer).

Effective rank definition: Roy & Vetterli, "The effective rank of a matrix"
(EURASIP Journal on Advances in Signal Processing, 2007).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from finetunecheck.models import SpectralReport

if TYPE_CHECKING:
    from finetunecheck.utils.model_loader import AnalysisModel

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """SVD analysis of weight deltas between base and fine-tuned models.

    For each weight matrix, computes dW = W_ft - W_base and analyzes:
    - Effective rank (entropy of normalized singular value distribution)
    - Frobenius norm (total magnitude of change)
    - Top singular values (concentration of change)

    For LoRA models, detects adapter structure and reports the LoRA rank.
    """

    def __init__(self, max_rank: int = 64) -> None:
        self.max_rank = max_rank

    @staticmethod
    def effective_rank(singular_values: torch.Tensor) -> float:
        """Effective rank = exp(entropy of normalized singular values).

        Roy & Vetterli, 2007. Measures the "spread" of the spectrum.
        A rank-1 matrix has effective rank 1.0; a matrix with uniformly
        distributed singular values has effective rank equal to its
        actual rank.

        Args:
            singular_values: 1-D tensor of singular values (non-negative).

        Returns:
            Effective rank (float >= 0).
        """
        sv = singular_values[singular_values > 1e-10]
        if len(sv) == 0:
            return 0.0

        # Normalize to a probability distribution
        p = sv / sv.sum()
        entropy = -(p * torch.log(p)).sum()
        return float(torch.exp(entropy).item())

    def analyze_weight_delta(self, name: str, w_base: torch.Tensor, w_ft: torch.Tensor) -> dict:
        """Analyze a single weight matrix delta via SVD.

        Args:
            name: Parameter name (for logging).
            w_base: Base model weight tensor (2-D).
            w_ft: Fine-tuned model weight tensor (2-D, same shape as w_base).

        Returns:
            Dict with frobenius_norm, effective_rank, top_singular_values.
        """
        delta = (w_ft - w_base).float()
        frob_norm = float(torch.norm(delta, p="fro").item())

        # Use randomized SVD for large matrices (faster, memory-efficient)
        min_dim = min(delta.shape)
        q = min(self.max_rank, min_dim)

        try:
            if min_dim > 512:
                # torch.svd_lowrank is efficient for low-rank approximation
                _U, S, _V = torch.svd_lowrank(delta, q=q)
            else:
                S = torch.linalg.svdvals(delta)
        except RuntimeError as exc:
            logger.warning("SVD failed for %s: %s", name, exc)
            return {
                "frobenius_norm": frob_norm,
                "effective_rank": 0.0,
                "top_singular_values": [],
            }

        eff_rank = self.effective_rank(S)
        top_k = min(10, len(S))
        top_svs = S[:top_k].tolist()

        return {
            "frobenius_norm": frob_norm,
            "effective_rank": eff_rank,
            "top_singular_values": top_svs,
        }

    def analyze(self, base_model: AnalysisModel, ft_model: AnalysisModel) -> SpectralReport:
        """Full spectral analysis across all weight matrices.

        Compares matching weight parameters (>= 2-D) between the two models.
        Skips biases and layer norms for efficiency (they are 1-D).

        Args:
            base_model: AnalysisModel for the base model.
            ft_model: AnalysisModel for the fine-tuned model.

        Returns:
            SpectralReport with per-layer metrics and LoRA detection.
        """
        per_layer_rank: dict[str, float] = {}
        per_layer_norm: dict[str, float] = {}
        per_layer_svs: dict[str, list[float]] = {}

        base_params = dict(base_model.model.named_parameters())
        ft_params = dict(ft_model.model.named_parameters())

        # Detect LoRA before analysis
        is_lora = any("lora" in name.lower() for name in ft_params)
        lora_rank: int | None = None
        if is_lora:
            for name in ft_params:
                if "lora_a" in name.lower() or "lora_A" in name:
                    lora_rank = ft_params[name].shape[0]
                    break

        # Analyze matching 2-D+ weight matrices
        analyzed = 0
        for name, w_base in base_params.items():
            if name not in ft_params:
                continue
            if w_base.dim() < 2:
                continue

            w_ft = ft_params[name]
            if w_base.shape != w_ft.shape:
                logger.debug("Shape mismatch for %s, skipping.", name)
                continue

            result = self.analyze_weight_delta(name, w_base.data, w_ft.data)
            per_layer_rank[name] = result["effective_rank"]
            per_layer_norm[name] = result["frobenius_norm"]
            if result["top_singular_values"]:
                per_layer_svs[name] = result["top_singular_values"]

            analyzed += 1

        ranks = list(per_layer_rank.values())
        mean_rank = float(np.mean(ranks)) if ranks else 0.0

        logger.info(
            "Spectral analysis complete: %d matrices, mean_rank=%.2f, lora=%s (rank=%s)",
            analyzed,
            mean_rank,
            is_lora,
            lora_rank,
        )

        return SpectralReport(
            per_layer_effective_rank=per_layer_rank,
            per_layer_frobenius_norm=per_layer_norm,
            per_layer_top_singular_values=per_layer_svs,
            mean_effective_rank=mean_rank,
            is_lora=is_lora,
            lora_rank=lora_rank,
        )
