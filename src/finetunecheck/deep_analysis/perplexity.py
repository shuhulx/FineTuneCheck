"""Perplexity distribution shift analysis between base and fine-tuned models.

Rather than comparing mean perplexity (which hides catastrophic regressions
on subsets), this analyzes the full distribution of per-sample perplexity
values and detects hidden forgetting through distribution divergence metrics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy import stats

from finetunecheck.models import PerplexityDistShift
from finetunecheck.utils.device import resolve_model_device

if TYPE_CHECKING:
    from finetunecheck.utils.model_loader import AnalysisModel

logger = logging.getLogger(__name__)


class PerplexityAnalyzer:
    """Analyze per-sample perplexity distribution shift between base and fine-tuned models.

    Unlike standard mean-perplexity comparison, this analyzes the FULL DISTRIBUTION:
    - KL divergence between perplexity distributions
    - Wasserstein (Earth Mover's) distance for robust comparison
    - Tail analysis: what fraction of samples have dramatically higher perplexity
    - Per-sample ratios to identify outlier regressions

    This detects "hidden forgetting" where aggregate perplexity looks fine but
    specific sample clusters have exploded perplexity.
    """

    def __init__(self, batch_size: int = 16, max_length: int = 512) -> None:
        self.batch_size = batch_size
        self.max_length = max_length

    def compute_perplexities(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
        device: str,
    ) -> np.ndarray:
        """Compute per-sample perplexity for a list of texts.

        Perplexity = exp(mean(negative_log_likelihood)) per sample.
        Uses a sliding window for texts longer than max_length.

        Args:
            model: A torch.nn.Module (the raw model, already in eval mode).
            tokenizer: The tokenizer with pad_token set.
            texts: List of text strings to evaluate.
            device: Device string (e.g. "cuda", "cpu").

        Returns:
            Array of shape (len(texts),) with per-sample perplexity values.
        """
        perplexities = np.zeros(len(texts), dtype=np.float64)
        model.eval()

        stride = max(1, self.max_length // 2)
        for sample_index, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt", truncation=False)
            full_ids = encoded["input_ids"]
            full_mask = encoded.get("attention_mask", torch.ones_like(full_ids))
            sequence_length = full_ids.shape[1]
            if sequence_length < 2:
                raise ValueError(f"Perplexity sample {sample_index} contains fewer than two tokens")

            nll_sum = 0.0
            scored_tokens = 0
            previous_end = 0
            for begin in range(0, sequence_length, stride):
                end = min(begin + self.max_length, sequence_length)
                new_token_count = end - previous_end
                input_ids = full_ids[:, begin:end].to(device)
                attention_mask = full_mask[:, begin:end].to(device)
                labels = input_ids.clone()
                labels[:, :-new_token_count] = -100

                with torch.no_grad():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                valid = (shift_labels != -100) & attention_mask[:, 1:].bool()
                if valid.any():
                    losses = torch.nn.functional.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.shape[-1]),
                        shift_labels.reshape(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).reshape_as(shift_labels)
                    nll_sum += float(losses[valid].sum().item())
                    scored_tokens += int(valid.sum().item())

                previous_end = end
                if end == sequence_length:
                    break

            if scored_tokens == 0:
                raise ValueError(f"Perplexity sample {sample_index} produced no scored tokens")
            perplexities[sample_index] = float(
                torch.exp(torch.tensor(min(nll_sum / scored_tokens, 100.0))).item()
            )

        return perplexities

    def _kl_divergence(
        self, p_samples: np.ndarray, q_samples: np.ndarray, num_bins: int = 50
    ) -> float:
        """Estimate KL(P||Q) via histogram binning with Laplace smoothing.

        Args:
            p_samples: Samples from the base model perplexity distribution.
            q_samples: Samples from the fine-tuned model perplexity distribution.
            num_bins: Number of bins for histogram estimation.

        Returns:
            KL divergence estimate (non-negative float).
        """
        all_samples = np.concatenate([p_samples, q_samples])
        bin_min = float(np.min(all_samples))
        bin_max = float(np.max(all_samples))

        if bin_max - bin_min < 1e-10:
            return 0.0

        bin_edges = np.linspace(bin_min, bin_max, num_bins + 1)

        p_hist, _ = np.histogram(p_samples, bins=bin_edges, density=False)
        q_hist, _ = np.histogram(q_samples, bins=bin_edges, density=False)

        # Laplace smoothing to avoid log(0)
        alpha = 1.0
        p_smooth = (p_hist.astype(np.float64) + alpha) / (p_hist.sum() + alpha * num_bins)
        q_smooth = (q_hist.astype(np.float64) + alpha) / (q_hist.sum() + alpha * num_bins)

        # KL(P||Q) = sum(p * log(p/q))
        kl = float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))
        return max(kl, 0.0)

    def analyze(
        self,
        base_model: AnalysisModel,
        ft_model: AnalysisModel,
        reference_texts: list[str],
    ) -> PerplexityDistShift:
        """Full perplexity distribution analysis.

        Args:
            base_model: AnalysisModel for the base model.
            ft_model: AnalysisModel for the fine-tuned model.
            reference_texts: Diverse reference corpus for perplexity computation.

        Returns:
            PerplexityDistShift with KL, Wasserstein, tail analysis.
        """
        if not reference_texts:
            raise ValueError("reference_texts must be a non-empty list of strings.")

        logger.info(
            "Computing perplexity distributions over %d samples...",
            len(reference_texts),
        )

        base_device = resolve_model_device(base_model)
        ft_device = resolve_model_device(ft_model)

        base_ppls = self.compute_perplexities(
            base_model.model, base_model.tokenizer, reference_texts, base_device
        )
        ft_ppls = self.compute_perplexities(
            ft_model.model, ft_model.tokenizer, reference_texts, ft_device
        )

        # KL divergence via histogram binning
        kl = self._kl_divergence(base_ppls, ft_ppls)

        # Wasserstein (Earth Mover's) distance — robust to outliers
        wasserstein = float(stats.wasserstein_distance(base_ppls, ft_ppls))

        # Tail analysis: fraction where ft_ppl / base_ppl > 2.0
        ratios = ft_ppls / np.clip(base_ppls, 1e-6, None)
        tail_fraction = float(np.mean(ratios > 2.0))

        logger.info(
            "Perplexity analysis complete: KL=%.4f, Wasserstein=%.4f, tail=%.3f",
            kl,
            wasserstein,
            tail_fraction,
        )

        return PerplexityDistShift(
            kl_divergence=kl,
            wasserstein_distance=wasserstein,
            tail_fraction=tail_fraction,
            mean_ppl_base=float(np.mean(base_ppls)),
            mean_ppl_ft=float(np.mean(ft_ppls)),
            base_ppls=base_ppls.tolist(),
            ft_ppls=ft_ppls.tolist(),
        )
