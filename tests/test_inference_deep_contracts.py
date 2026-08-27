"""Tiny local inference and experimental deep-analysis contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from finetunecheck.deep_analysis.calibration import CalibrationAnalyzer
from finetunecheck.deep_analysis.orchestrator import REFERENCE_TEXTS, DeepAnalysisOrchestrator
from finetunecheck.deep_analysis.perplexity import PerplexityAnalyzer
from finetunecheck.deep_analysis.spectral import SpectralAnalyzer
from finetunecheck.eval.inference import TransformersBackend, _dtype_for_device
from finetunecheck.models import MeasurementStatus
from finetunecheck.utils.model_loader import AnalysisModel


class TinyBatch(dict):
    def to(self, device: str) -> TinyBatch:
        return TinyBatch({key: value.to(device) for key, value in self.items()})


class TinyTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    vocab_size = 16
    model_max_length = 64
    chat_template = "tiny-chat-template"
    padding_side = "right"

    def __init__(self) -> None:
        self.formatted_messages: list[list[dict[str, str]]] = []
        self.calls: list[tuple[list[str], str]] = []

    def __len__(self) -> int:
        return self.vocab_size

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.formatted_messages.append(messages)
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(
        self,
        texts: str | list[str],
        *,
        return_tensors: str,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> TinyBatch:
        assert return_tensors == "pt"
        values = [texts] if isinstance(texts, str) else texts
        self.calls.append((list(values), self.padding_side))
        encoded: list[list[int]] = []
        for text in values:
            tokens = [2, *[(sum(map(ord, word)) % 13) + 3 for word in text.split()], 1]
            if truncation and max_length is not None:
                tokens = tokens[:max_length]
            encoded.append(tokens)
        width = max(len(row) for row in encoded)
        padded: list[list[int]] = []
        masks: list[list[int]] = []
        for row in encoded:
            missing = width - len(row)
            if padding and self.padding_side == "left":
                padded.append([self.pad_token_id] * missing + row)
                masks.append([0] * missing + [1] * len(row))
            else:
                padded.append(row + [self.pad_token_id] * missing)
                masks.append([1] * len(row) + [0] * missing)
        return TinyBatch(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )

    @staticmethod
    def decode(token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return " ".join(f"token-{int(token)}" for token in token_ids if int(token) > 2)


class TinyAttention(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = hidden @ hidden.transpose(-1, -2)
        weights = scores.softmax(dim=-1).unsqueeze(1)
        return hidden, weights


class TinyLayer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = TinyAttention()
        self.linear = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attn(hidden)
        return torch.tanh(self.linear(attended))


class TinyBackbone(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyLayer(width), TinyLayer(width)])


class TinyLM(torch.nn.Module):
    def __init__(self, *, offset: float = 0.0) -> None:
        super().__init__()
        torch.manual_seed(11)
        self.config = SimpleNamespace(
            max_position_embeddings=64,
            model_type="tiny",
            hidden_size=8,
            num_hidden_layers=2,
            vocab_size=16,
        )
        self.embed = torch.nn.Embedding(16, 8)
        self.model = TinyBackbone(8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)
        self.forward_lengths: list[int] = []
        if offset:
            with torch.no_grad():
                self.model.layers[0].linear.weight.add_(offset)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, output_attentions
        self.forward_lengths.append(input_ids.shape[1])
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        assert "attention_mask" in kwargs
        token = torch.full((len(input_ids), 1), 3, dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, token], dim=1)


def test_transformers_backend_applies_chat_template_left_padding_and_cardinality() -> None:
    tokenizer = TinyTokenizer()
    model = TinyLM()
    backend = TransformersBackend(model, tokenizer, "cpu", "tiny@revision")

    results = backend.generate_batch(["short", "a longer prompt"], max_tokens=2, probe_name="x")

    assert tokenizer.padding_side == "left"
    assert len(tokenizer.formatted_messages) == 2
    assert tokenizer.calls[0][1] == "left"
    assert len(results) == 2
    assert all(result.output == "token-3" for result in results)
    assert all(result.backend == "transformers" for result in results)


def test_transformers_backend_rejects_context_overflow_and_unknown_settings() -> None:
    tokenizer = TinyTokenizer()
    tokenizer.model_max_length = 5
    backend = TransformersBackend(TinyLM(), tokenizer, "cpu", "tiny")

    with pytest.raises(ValueError, match="context limit"):
        backend.generate_batch(["one two three"], max_tokens=2)
    tokenizer.model_max_length = 64
    with pytest.raises(ValueError, match="Unsupported generation settings"):
        backend.generate_batch(["one"], max_tokens=1, unsupported=True)


def test_transformers_backend_logprobs_are_token_aligned_and_finite() -> None:
    backend = TransformersBackend(TinyLM(), TinyTokenizer(), "cpu", "tiny")
    values = backend.get_logprobs(["one two three"])
    assert len(values) == 1
    assert len(values[0]) >= 2
    assert all(np.isfinite(value) and value <= 0 for value in values[0])


def test_cpu_dtype_is_float32() -> None:
    assert _dtype_for_device("cpu") == torch.float32


def test_perplexity_uses_multiple_bounded_sliding_windows() -> None:
    model = TinyLM()
    tokenizer = TinyTokenizer()
    analyzer = PerplexityAnalyzer(max_length=4)
    values = analyzer.compute_perplexities(
        model,
        tokenizer,
        ["one two three four five six seven eight nine"],
        "cpu",
    )
    assert values.shape == (1,)
    assert np.isfinite(values[0])
    assert len(model.forward_lengths) > 1
    assert max(model.forward_lengths) <= 4


def test_calibration_uses_ft_confidence_bins_and_no_full_softmax(monkeypatch) -> None:
    analyzer = CalibrationAnalyzer(num_bins=4, batch_size=2, max_length=12)
    base = AnalysisModel(TinyLM(), TinyTokenizer())
    ft = AnalysisModel(TinyLM(offset=0.01), TinyTokenizer())

    def forbidden_softmax(*_args, **_kwargs):
        raise AssertionError("full softmax should not be materialized")

    monkeypatch.setattr(torch, "softmax", forbidden_softmax)
    report = analyzer.analyze(base, ft, ["one two three", "four five six"])

    assert len(report.per_bin_confidence) == 4
    assert len(report.per_bin_confidence_ft) == 4
    assert np.isfinite(report.base_ece)
    assert np.isfinite(report.ft_ece)


def test_randomized_spectral_analysis_is_deterministic_and_normalized() -> None:
    analyzer = SpectralAnalyzer(max_rank=3)
    torch.manual_seed(3)
    base = torch.randn(513, 513)
    ft = base + torch.eye(513) * 0.01

    first = analyzer.analyze_weight_delta("large", base, ft)
    second = analyzer.analyze_weight_delta("large", base, ft)

    assert first["top_singular_values"] == pytest.approx(second["top_singular_values"])
    assert first["effective_rank"] == pytest.approx(second["effective_rank"])
    assert 0 < first["frobenius_norm"] < 1


def test_deep_orchestrator_runs_all_components_on_tiny_local_models() -> None:
    base = AnalysisModel(TinyLM(), TinyTokenizer())
    ft = AnalysisModel(TinyLM(offset=0.01), TinyTokenizer())
    report = DeepAnalysisOrchestrator(num_samples=3, batch_size=2).run(
        base,
        ft,
        ["one two three", "four five six", "seven eight nine"],
    )

    assert report.status == MeasurementStatus.MEASURED
    assert report.samples_requested == 3
    assert report.samples_used == 3
    assert set(report.component_status) == {
        "spectral",
        "perplexity",
        "calibration",
        "cka",
        "activation",
    }
    assert all(
        component.status == MeasurementStatus.MEASURED
        for component in report.component_status.values()
    )
    assert report.spectral is not None
    assert report.perplexity is not None
    assert report.calibration is not None
    assert report.cka is not None
    assert report.activation is not None
    assert report.activation.attention_status == MeasurementStatus.MEASURED


def test_deep_orchestrator_reports_incompatible_tokenizers_without_claims() -> None:
    base_tokenizer = TinyTokenizer()
    ft_tokenizer = TinyTokenizer()
    ft_tokenizer.eos_token_id = 9
    report = DeepAnalysisOrchestrator(num_samples=1).run(
        AnalysisModel(TinyLM(), base_tokenizer),
        AnalysisModel(TinyLM(), ft_tokenizer),
        ["one two"],
    )
    assert report.status == MeasurementStatus.INCOMPATIBLE
    assert report.perplexity is None
    assert all(
        component.status == MeasurementStatus.INCOMPATIBLE
        for component in report.component_status.values()
    )


def test_deep_orchestrator_discloses_bundled_corpus_size_and_partial_errors(
    monkeypatch,
) -> None:
    base = AnalysisModel(TinyLM(), TinyTokenizer())
    ft = AnalysisModel(TinyLM(offset=0.01), TinyTokenizer())

    def fail(_self, _base, _ft):
        raise RuntimeError("controlled spectral failure")

    monkeypatch.setattr(SpectralAnalyzer, "analyze", fail)
    report = DeepAnalysisOrchestrator(
        num_samples=2,
        batch_size=1,
        enable_perplexity=False,
        enable_calibration=False,
        enable_cka=False,
        enable_activation=False,
    ).run(base, ft)

    assert len(REFERENCE_TEXTS) == 50
    assert report.corpus_size == 50
    assert report.samples_used == 2
    assert report.status == MeasurementStatus.ERROR
    assert report.component_status["spectral"].status == MeasurementStatus.ERROR
    assert "controlled spectral failure" in (report.component_status["spectral"].error or "")
