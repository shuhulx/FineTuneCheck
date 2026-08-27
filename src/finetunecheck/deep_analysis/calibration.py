"""Token-level calibration shift detection.

A well-calibrated model's confidence matches its accuracy: when it says 80%
confidence, it should be correct ~80% of the time. Fine-tuning can silently
break this property, producing overconfident or underconfident predictions.

This module computes Expected Calibration Error (ECE) for both the base and
fine-tuned models, then quantifies how calibration shifted.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from finetunecheck.models import CalibrationReport
from finetunecheck.utils.device import resolve_model_device

if TYPE_CHECKING:
    from finetunecheck.utils.model_loader import AnalysisModel

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """Detect token-level confidence calibration shift from fine-tuning.

    Uses teacher-forcing to collect (confidence, correctness) pairs for every
    predicted token, then bins them to compute Expected Calibration Error (ECE).
    """

    def __init__(self, num_bins: int = 10, batch_size: int = 16, max_length: int = 256) -> None:
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.max_length = max_length

    def collect_predictions(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
        device: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collect per-token (confidence, correctness) pairs via teacher-forcing.

        For each text:
        - Input: tokens[:-1]
        - Target: tokens[1:]
        - Confidence: max(softmax(logits)) for each position
        - Correct: 1 if argmax(logits) == target, 0 otherwise

        Args:
            model: The raw torch.nn.Module.
            tokenizer: Tokenizer with pad_token set.
            texts: Input texts.
            device: Device string.

        Returns:
            (confidences, correctness): both 1-D arrays of length total_tokens.
        """
        model.eval()
        all_confidences: list[np.ndarray] = []
        all_correctness: list[np.ndarray] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start : batch_start + self.batch_size]
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (batch, seq, vocab)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]

            # Compute max-softmax confidence without materializing a full
            # probability tensor the size of batch x sequence x vocabulary.
            max_logits, predicted = shift_logits.max(dim=-1)
            max_probs = torch.exp(max_logits - torch.logsumexp(shift_logits, dim=-1))

            # Per-token correctness
            correct = (predicted == shift_labels).float()

            # Flatten and apply mask
            mask_flat = shift_mask.bool().cpu().numpy().ravel()
            conf_flat = max_probs.cpu().numpy().ravel()
            corr_flat = correct.cpu().numpy().ravel()

            all_confidences.append(conf_flat[mask_flat])
            all_correctness.append(corr_flat[mask_flat])

        confidences = np.concatenate(all_confidences) if all_confidences else np.array([])
        correctness = np.concatenate(all_correctness) if all_correctness else np.array([])

        return confidences, correctness

    def compute_ece(
        self, confidences: np.ndarray, correctness: np.ndarray
    ) -> tuple[float, list[float], list[float], list[float]]:
        """Compute Expected Calibration Error (ECE).

        ECE = sum_b |accuracy(b) - confidence(b)| * weight(b)

        where b ranges over equal-width confidence bins.

        Args:
            confidences: Array of max-softmax confidence values.
            correctness: Array of 0/1 indicating correct next-token prediction.

        Returns:
            (ece, per_bin_accuracy, per_bin_confidence, bin_edges)
        """
        if len(confidences) == 0:
            raise ValueError("Calibration is unavailable because no scored tokens were produced")

        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        per_bin_acc: list[float] = []
        per_bin_conf: list[float] = []
        total = len(confidences)

        for i in range(self.num_bins):
            lo = bin_boundaries[i]
            hi = bin_boundaries[i + 1]

            # Include right edge in last bin
            if i == self.num_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)

            bin_count = int(mask.sum())
            if bin_count == 0:
                per_bin_acc.append(0.0)
                per_bin_conf.append((lo + hi) / 2)
                continue

            bin_acc = float(correctness[mask].mean())
            bin_conf = float(confidences[mask].mean())
            bin_weight = bin_count / total

            ece += abs(bin_acc - bin_conf) * bin_weight
            per_bin_acc.append(bin_acc)
            per_bin_conf.append(bin_conf)

        return ece, per_bin_acc, per_bin_conf, bin_boundaries.tolist()

    def analyze(
        self,
        base_model: AnalysisModel,
        ft_model: AnalysisModel,
        reference_texts: list[str],
    ) -> CalibrationReport:
        """Full calibration analysis comparing base and fine-tuned models.

        Args:
            base_model: AnalysisModel for the base model.
            ft_model: AnalysisModel for the fine-tuned model.
            reference_texts: Texts for teacher-forced prediction.

        Returns:
            CalibrationReport with ECE values and per-bin breakdowns.
        """
        if not reference_texts:
            raise ValueError("reference_texts must be non-empty.")

        logger.info("Collecting calibration data from base model...")
        base_device = resolve_model_device(base_model)
        base_conf, base_corr = self.collect_predictions(
            base_model.model, base_model.tokenizer, reference_texts, base_device
        )

        logger.info("Collecting calibration data from fine-tuned model...")
        ft_device = resolve_model_device(ft_model)
        ft_conf, ft_corr = self.collect_predictions(
            ft_model.model, ft_model.tokenizer, reference_texts, ft_device
        )

        base_ece, base_bins_acc, base_bins_conf, _ = self.compute_ece(base_conf, base_corr)
        ft_ece, ft_bins_acc, ft_bins_conf, _ = self.compute_ece(ft_conf, ft_corr)

        logger.info(
            "Calibration analysis complete: base_ece=%.4f, ft_ece=%.4f, delta=%.4f",
            base_ece,
            ft_ece,
            ft_ece - base_ece,
        )

        return CalibrationReport(
            base_ece=base_ece,
            ft_ece=ft_ece,
            ece_delta=ft_ece - base_ece,
            per_bin_accuracy_base=base_bins_acc,
            per_bin_accuracy_ft=ft_bins_acc,
            per_bin_confidence=base_bins_conf,
            per_bin_confidence_ft=ft_bins_conf,
        )
