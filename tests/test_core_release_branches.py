"""Focused branch coverage for release-critical judges, cache, profiles, and verdicts."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from finetunecheck.config import EvalConfig, JudgeConfig
from finetunecheck.eval.cache import BaselineCache, build_cache_manifest, model_identity_manifest
from finetunecheck.eval.judge import (
    APIJudgeProvider,
    BackendJudgeProvider,
    CallableJudgeProvider,
    ExecutionJudge,
    LLMJudge,
    RougeJudge,
)
from finetunecheck.eval.runner import EvalRunner, _make_placeholder_probes
from finetunecheck.models import (
    CategoryScore,
    DeepAnalysisReport,
    ForgettingPattern,
    ForgettingReport,
    InferenceResult,
    JudgeType,
    JudgeVerdict,
    MeasurementStatus,
    ModelSpec,
    ModelType,
    ProbeSample,
    ProbeSet,
    Verdict,
)
from finetunecheck.models import (
    TestCaseOutcome as CodeTestOutcome,
)
from finetunecheck.probes.registry import ProbeRegistry
from finetunecheck.profiles.loader import EvalProfile, ProfileLoader


def _score(category: str, value: float, count: int = 20) -> CategoryScore:
    return CategoryScore(
        category=category,
        mean_score=value,
        num_samples=count,
        expected_samples=count,
        sample_scores=[value] * count,
        selected_sample_ids=[f"{category}-{index}" for index in range(count)],
    )


def _release_probe(name: str = "target") -> ProbeSet:
    return ProbeSet(
        name=name,
        category=name,
        judge_type=JudgeType.EXACT_MATCH,
        provenance={"source": "test", "supports_release_claims": True},
        samples=[
            ProbeSample(id=f"{name}-{index}", input="q", reference="a") for index in range(20)
        ],
    )


def _forgetting(
    *,
    pattern: ForgettingPattern = ForgettingPattern.MINIMAL,
    sar: float | None = 1.0,
    missing: list[str] | None = None,
) -> ForgettingReport:
    return ForgettingReport(
        backward_transfer=0.0,
        capability_retention_rates={"general": 1.0},
        selective_forgetting_index=0.0,
        safety_alignment_retention=sar,
        pattern=pattern,
        missing_categories=missing or [],
    )


def _roi(score: float = 90.0, coverage: float = 1.0) -> dict[str, Any]:
    return {
        "score": score,
        "coverage": coverage,
        "weights": {},
        "values": {},
        "formula_version": "roi-v2",
    }


def test_cache_manifest_covers_local_adapter_tokenizer_gguf_and_remote_revisions(
    tmp_path: Path,
) -> None:
    model = tmp_path / "adapter"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text('{"vocab": {}}', encoding="utf-8")
    (model / "config.json").write_text('{"model_type": "tiny"}', encoding="utf-8")
    (model / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "base-id", "revision": "abc"}',
        encoding="utf-8",
    )

    local_model, tokenizer, adapter, cacheable = model_identity_manifest(str(model))
    assert cacheable is True
    assert local_model["weights_sha256"]
    assert tokenizer["sha256"]
    assert adapter["base_model_name_or_path"] == "base-id"
    assert adapter["revision"] == "abc"

    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"gguf-weights-and-tokenizer")
    _, gguf_tokenizer, _, gguf_cacheable = model_identity_manifest(str(gguf))
    assert gguf_cacheable is True
    assert gguf_tokenizer["embedded_in_model_sha256"]

    commit = "a" * 40
    remote_model, remote_tokenizer, _, remote_cacheable = model_identity_manifest(
        f"org/model@{commit}"
    )
    assert remote_cacheable is True
    assert remote_model["resolved_revision"] == commit
    assert remote_tokenizer["resolved_revision"] == commit
    assert model_identity_manifest("org/model@main")[3] is False
    assert model_identity_manifest("org/model")[3] is False


def test_cache_lifecycle_is_atomic_fail_closed_and_invalidatable(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    manifest = build_cache_manifest(
        model_path=str(model),
        probe_name="math",
        probe_version="2.0.0",
        probe_digest="digest",
        selected_sample_ids=["math-0"],
        judge={"judge_type": "exact_match"},
        generation={"max_tokens": 1},
        inference_backend="fake",
        execution_policy={"executor": "none"},
        adapter_relationship={"base_model_name_or_path": "base"},
    )
    score = CategoryScore(
        category="math",
        mean_score=0.75,
        std_score=0.0,
        num_samples=1,
        expected_samples=1,
        sample_scores=[0.75],
        sample_verdicts=[JudgeVerdict(sample_id="math-0", score=0.75)],
        selected_sample_ids=["math-0"],
        probe_digest="digest",
    )
    cache = BaselineCache(str(tmp_path / "cache"))
    try:
        assert cache.get(manifest) is None
        cache.set(manifest, score)
        assert cache.get(manifest) == score
        assert cache.get_key(manifest).startswith("v2:")
        assert cache.has(str(model), "math", 1) is False
        assert cache.invalidate(manifest) is True
        assert cache.get(manifest) is None

        key = cache.get_key(str(model), "math", 1)
        cache.set(key, score)
        assert cache.has(str(model), "math", 1) is True
        cache._cache.set(key, "not valid category JSON")
        assert cache.get(key) is None
        assert key not in cache._cache

        cache._cache.set(key, {"pickle": "must never deserialize"})
        monkeypatch.setattr(
            "diskcache.core.pickle.load",
            lambda *_args, **_kwargs: pytest.fail("pickle cache row was deserialized"),
        )
        assert cache.get(key) is None

        mismatched = score.model_copy(update={"probe_digest": "wrong"})
        manifest_key = cache.get_key(manifest)
        cache._cache.set(manifest_key, mismatched.model_dump_json())
        assert cache.get(manifest) is None
        with pytest.raises(ValueError, match="does not match"):
            cache.set(manifest, mismatched)

        partial_manifest = build_cache_manifest(
            model_path=str(model),
            probe_name="math",
            probe_version="2.0.0",
            probe_digest="digest",
            selected_sample_ids=["math-0", "math-1"],
            judge={"judge_type": "exact_match"},
            generation={"max_tokens": 1},
            inference_backend="fake",
            execution_policy={"executor": "none"},
        )
        partial = CategoryScore(
            category="math",
            status=MeasurementStatus.ERROR,
            error="one sample failed",
            num_samples=1,
            expected_samples=2,
            sample_scores=[0.75],
            sample_verdicts=[
                JudgeVerdict(sample_id="math-0", score=0.75),
                JudgeVerdict(
                    sample_id="math-1",
                    status=MeasurementStatus.ERROR,
                    error="judge failed",
                ),
            ],
            selected_sample_ids=["math-0", "math-1"],
            probe_digest="digest",
        )
        cache.set(partial_manifest, partial)
        assert cache.get(partial_manifest) is None

        cache.set(key, score)
        cache.clear()
        assert cache.get(key) is None
        with pytest.raises(TypeError, match="required"):
            cache.get_key(str(model))
    finally:
        cache.close()

    noncacheable = build_cache_manifest(
        model_path="org/model@main",
        probe_name="math",
        probe_version="2",
        probe_digest="x",
        selected_sample_ids=["x"],
        judge={},
        generation={},
        inference_backend="fake",
        execution_policy={},
    )
    other = BaselineCache(str(tmp_path / "other-cache"))
    try:
        other.set(noncacheable, score)
        assert other.get(noncacheable) is None
    finally:
        other.close()


def test_invalid_adapter_json_is_nonfatal_but_not_trusted(tmp_path: Path) -> None:
    model = tmp_path / "adapter"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "adapter_config.json").write_text("not-json", encoding="utf-8")
    _, _, adapter, cacheable = model_identity_manifest(str(model))
    assert cacheable is True
    assert adapter["base_model_name_or_path"] is None


def test_backend_judge_provider_forwards_settings_and_cleans_up() -> None:
    class Backend:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}
            self.cleaned = False

        def generate_batch(self, prompts, **kwargs):
            self.kwargs = kwargs
            return [
                InferenceResult(
                    model_path="judge",
                    output='{"score": 9, "explanation": "ok"}',
                )
                for _ in prompts
            ]

        def cleanup(self) -> None:
            self.cleaned = True

    backend = Backend()
    config = JudgeConfig(
        provider="local",
        model="judge",
        temperature=0.2,
        settings={"top_p": 0.8},
    )
    provider = BackendJudgeProvider(backend, config)
    raw = provider.generate("prompt", max_tokens=17, temperature=0.2)
    assert json.loads(raw)["score"] == 9
    assert backend.kwargs == {
        "max_tokens": 17,
        "probe_name": "judge",
        "top_p": 0.8,
        "temperature": 0.2,
        "do_sample": True,
    }
    assert provider.provenance["backend"] == "Backend"
    provider.close()
    assert backend.cleaned is True


def test_backend_judge_rejects_wrong_cardinality() -> None:
    backend = SimpleNamespace(generate_batch=lambda *_args, **_kwargs: [], cleanup=lambda: None)
    provider = BackendJudgeProvider(
        backend,
        JudgeConfig(provider="local", model="judge"),
    )
    with pytest.raises(ValueError, match="0 outputs"):
        provider.generate("prompt", max_tokens=5, temperature=0)


def test_api_judge_preflight_requires_dependency_and_key(monkeypatch) -> None:
    provider = APIJudgeProvider(JudgeConfig(provider="openai", model="judge"))
    monkeypatch.setattr("importlib.util.find_spec", lambda _package: None)
    with pytest.raises(ValueError, match="dependencies"):
        provider.preflight()

    monkeypatch.setattr("importlib.util.find_spec", lambda _package: object())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        provider.preflight()


def test_api_judge_openai_and_anthropic_paths_use_explicit_configuration(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class OpenAI:
        def __init__(self, **kwargs):
            calls.append(("openai-init", kwargs))
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **payload: (
                        calls.append(("openai-call", payload))
                        or SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content="openai"))]
                        )
                    )
                )
            )

    class Anthropic:
        def __init__(self, **kwargs):
            calls.append(("anthropic-init", kwargs))
            self.messages = SimpleNamespace(
                create=lambda **payload: (
                    calls.append(("anthropic-call", payload))
                    or SimpleNamespace(content=[SimpleNamespace(text="anthropic")])
                )
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Anthropic))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-key")

    openai_provider = APIJudgeProvider(
        JudgeConfig(provider="openai", model="gpt-test", settings={"seed": 3})
    )
    assert openai_provider.generate("prompt", max_tokens=8, temperature=0.0) == "openai"
    anthropic_provider = APIJudgeProvider(
        JudgeConfig(provider="anthropic", model="claude-test", settings={"top_p": 0.9})
    )
    assert anthropic_provider.generate("prompt", max_tokens=8, temperature=0.1) == "anthropic"
    assert calls[1][1]["model"] == "gpt-test"
    assert calls[3][1]["model"] == "claude-test"


def test_llm_judge_missing_provider_and_provider_error_are_explicit() -> None:
    sample = ProbeSample(id="s", input="q", reference="a")
    missing = LLMJudge().evaluate(sample, "candidate")
    assert missing.status == MeasurementStatus.NOT_RUN
    assert missing.score is None

    def fail(_prompt: str) -> str:
        raise RuntimeError("judge unavailable")

    provider = CallableJudgeProvider(fail)
    errored = LLMJudge(provider).evaluate(sample, "candidate")
    assert errored.status == MeasurementStatus.ERROR
    assert "judge unavailable" in (errored.error or "")


def test_legacy_explicit_judge_client_is_supported_only_when_supplied() -> None:
    client = SimpleNamespace(
        _judge_model="legacy-judge",
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"score": 7, "explanation": "legacy"}')
                        )
                    ]
                )
            )
        ),
    )
    verdict = LLMJudge(api_client=client).evaluate(
        ProbeSample(id="s", input="q", reference="a"),
        "candidate",
    )
    assert verdict.score == pytest.approx(0.7)
    assert verdict.provenance["provider"] == "legacy_explicit_client"


class BrokenExecutor:
    available = True
    provenance = {"executor": "broken-test-double"}

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def execute(self, _code, cases, *, timeout_seconds):
        del timeout_seconds
        if self.mode == "raise":
            raise RuntimeError("isolated runtime failed")
        if self.mode == "short":
            return []
        return [CodeTestOutcome(index=0, expression=cases[0]["input"], actual=1.0001)]


def test_execution_judge_error_and_cardinality_paths() -> None:
    sample = ProbeSample(
        id="code",
        input="code",
        reference="pass",
        metadata={"test_cases": [{"input": "answer()", "output": "1.0"}]},
    )
    no_code = ExecutionJudge(executor=BrokenExecutor("ok")).evaluate(sample, "plain text")
    assert no_code.score == 0.0
    errored = ExecutionJudge(executor=BrokenExecutor("raise")).evaluate(
        sample, "```python\npass\n```"
    )
    assert errored.status == MeasurementStatus.ERROR
    mismatch = ExecutionJudge(executor=BrokenExecutor("short")).evaluate(
        sample, "```python\npass\n```"
    )
    assert mismatch.error == "executor_cardinality_mismatch"
    close = ExecutionJudge(executor=BrokenExecutor("ok")).evaluate(sample, "```python\npass\n```")
    assert close.score == 0.0
    sample.metadata["test_tolerance"] = 0.001
    tolerant = ExecutionJudge(executor=BrokenExecutor("ok")).evaluate(
        sample, "```python\npass\n```"
    )
    assert tolerant.score == 1.0
    with pytest.raises(RuntimeError, match="removed"):
        ExecutionJudge()._run_code("print('no')")


def test_execution_judge_missing_cases_is_error() -> None:
    sample = ProbeSample(id="code", input="code", reference="pass")
    verdict = ExecutionJudge(executor=BrokenExecutor("ok")).evaluate(sample, "```python\npass\n```")
    assert verdict.status == MeasurementStatus.ERROR
    assert verdict.error == "missing_test_cases"


def test_rouge_reports_overlap_only_and_missing_reference() -> None:
    judge = RougeJudge()
    measured = judge.evaluate(
        ProbeSample(id="s", input="summarize", reference="one two three"),
        "one two",
    )
    assert measured.status == MeasurementStatus.MEASURED
    assert measured.details["claim_scope"] == "lexical_overlap_only"
    missing = judge.evaluate(ProbeSample(id="m", input="summarize"), "output")
    assert missing.status == MeasurementStatus.ERROR


def test_profile_contract_validation_and_reset(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="overlap"):
        EvalProfile(
            name="bad",
            target_probes=["math"],
            general_probes=["math"],
        )
    with pytest.raises(ValidationError, match="hard gates"):
        EvalProfile(name="bad", hard_gates={"unknown": True})
    with pytest.raises(KeyError, match="not found"):
        ProfileLoader.get("does-not-exist")

    ProfileLoader.reset()
    assert ProfileLoader._loaded is False
    monkeypatch.setattr("finetunecheck.profiles.loader._BUILTIN_DIR", tmp_path / "missing")
    assert ProfileLoader.list() == []
    assert ProfileLoader._loaded is True
    ProfileLoader.reset()


def test_placeholder_probe_and_empty_probe_selection_are_fail_closed() -> None:
    placeholder = _make_placeholder_probes("unknown", 2)
    assert len(placeholder.samples) == 2
    assert placeholder.judge_type == JudgeType.LLM
    assert placeholder.provenance["supports_release_claims"] is False

    runner = EvalRunner(
        EvalConfig(
            base_model="base",
            finetuned_model="ft",
            general_probes=[],
            target_tasks=[],
            cache_baseline=False,
        )
    )
    with pytest.raises(ValueError, match="At least one probe"):
        runner._build_probes()


def test_runner_rejects_self_judge_custom_provider_and_invalid_execution_preflight() -> None:
    provider = CallableJudgeProvider(lambda _prompt: "{}", model="base")
    runner = EvalRunner(
        EvalConfig(
            base_model="base",
            finetuned_model="ft",
            general_probes=["reasoning"],
            cache_baseline=False,
        ),
        judge_provider=provider,
    )
    with pytest.raises(ValueError, match="self-judging"):
        runner._preflight(
            [_release_probe("reasoning").model_copy(update={"judge_type": JudgeType.LLM})]
        )

    custom = EvalRunner(
        EvalConfig(
            base_model="base",
            finetuned_model="ft",
            general_probes=["reasoning"],
            judge=JudgeConfig(provider="custom", model="external"),
            cache_baseline=False,
        )
    )
    with pytest.raises(ValueError, match="explicit judge_provider"):
        custom._preflight(
            [_release_probe("reasoning").model_copy(update={"judge_type": JudgeType.LLM})]
        )

    invalid_execution = _release_probe("code").model_copy(
        update={"judge_type": JudgeType.EXECUTION}
    )
    deterministic = EvalRunner(
        EvalConfig(
            base_model="base",
            finetuned_model="ft",
            general_probes=["code"],
            cache_baseline=False,
        )
    )
    with pytest.raises(ValueError, match="test_cases"):
        deterministic._preflight([invalid_execution])


def test_runner_local_judge_resolution_rejects_alias_and_loads_dedicated_backend() -> None:
    config = EvalConfig(
        base_model="org/base@main",
        finetuned_model="org/ft",
        general_probes=["reasoning"],
        judge=JudgeConfig(provider="local", model="ORG/BASE@deadbeef"),
        cache_baseline=False,
    )
    runner = EvalRunner(config, backend_factory=lambda *_args: SimpleNamespace())
    base = ModelSpec(path="org/base", model_type=ModelType.HF)
    ft = ModelSpec(path="org/ft", model_type=ModelType.HF)
    loader = SimpleNamespace(
        detect_type=lambda value: ModelSpec(path=value, model_type=ModelType.HF)
    )
    with pytest.raises(ValueError, match="must never judge itself"):
        runner._resolve_configured_judge(loader, base, ft)

    backend = SimpleNamespace(
        cleanup=lambda: None,
        generate_batch=lambda *_args, **_kwargs: [],
    )
    config.judge = JudgeConfig(provider="local", model="org/dedicated")
    runner = EvalRunner(config, backend_factory=lambda *_args: backend)
    runner._resolve_configured_judge(loader, base, ft)
    assert isinstance(runner._judge_provider, BackendJudgeProvider)
    assert runner._local_judge_backend is backend


@pytest.mark.parametrize(
    ("bwt", "rates", "sfi", "expected"),
    [
        (None, {}, None, ForgettingPattern.UNAVAILABLE),
        (-0.3, {"a": 0.9}, 0.0, ForgettingPattern.CATASTROPHIC),
        (-0.01, {"a": 0.69, "b": 1.0, "c": 1.0}, 0.2, ForgettingPattern.SELECTIVE),
        (-0.1, {"a": 0.9}, 0.0, ForgettingPattern.GRADUAL),
        (0.0, {"a": 1.0}, 0.0, ForgettingPattern.MINIMAL),
    ],
)
def test_runner_pattern_classification_checks_collapse_before_mean(
    bwt, rates, sfi, expected
) -> None:
    assert EvalRunner._classify_pattern(bwt, rates, sfi) == expected


@pytest.mark.parametrize(
    ("pattern", "sar", "hard_gates", "roi_score", "target_delta", "expected"),
    [
        (ForgettingPattern.MINIMAL, 0.5, {}, 90.0, 0.2, Verdict.HARMFUL),
        (ForgettingPattern.CATASTROPHIC, 1.0, {}, 90.0, 0.2, Verdict.POOR),
        (ForgettingPattern.MINIMAL, 0.98, {"sar_min": 0.99}, 90.0, 0.2, Verdict.POOR),
        (ForgettingPattern.MINIMAL, 1.0, {}, 90.0, 0.2, Verdict.EXCELLENT),
        (ForgettingPattern.MINIMAL, 1.0, {}, 70.0, 0.2, Verdict.GOOD),
        (ForgettingPattern.MINIMAL, 1.0, {}, 40.0, 0.2, Verdict.POOR),
        (ForgettingPattern.MINIMAL, 1.0, {}, 90.0, 0.0, Verdict.POOR),
    ],
)
def test_verdict_branches_require_complete_release_evidence(
    pattern, sar, hard_gates, roi_score, target_delta, expected
) -> None:
    config = EvalConfig(
        base_model="base",
        finetuned_model="ft",
        target_tasks=["target"],
        general_probes=[],
        hard_gates=hard_gates,
        cache_baseline=False,
    )
    runner = EvalRunner(config)
    runner._probes = [_release_probe()]
    base = {"target": _score("target", 0.5)}
    ft = {"target": _score("target", min(1.0, 0.5 + max(target_delta, 0.0)))}
    verdict, summary, concerns, _ = runner._compute_verdict(
        target_delta,
        _forgetting(pattern=pattern, sar=sar),
        base,
        ft,
        _roi(roi_score),
    )
    assert verdict == expected
    assert "deployment approval" in summary
    if pattern == ForgettingPattern.CATASTROPHIC or sar < 1.0:
        assert concerns


def test_safety_critical_missing_sar_is_insufficient() -> None:
    config = EvalConfig(
        base_model="base",
        finetuned_model="ft",
        target_tasks=["target"],
        general_probes=[],
        hard_gates={"sar_min": 0.99, "strong_safety_required": True},
        cache_baseline=False,
    )
    runner = EvalRunner(config)
    runner._probes = [_release_probe()]
    scores = {"target": _score("target", 0.7)}
    verdict, _, concerns, _ = runner._compute_verdict(
        0.2,
        _forgetting(sar=None),
        scores,
        scores,
        _roi(),
    )
    assert verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert any("unmeasured" in concern for concern in concerns)
    assert any("stronger configured safety judge" in concern for concern in concerns)


def test_runner_finds_raw_sample_regressions() -> None:
    runner = EvalRunner(
        EvalConfig(
            base_model="base",
            finetuned_model="ft",
            target_tasks=["target"],
            general_probes=[],
            cache_baseline=False,
        )
    )
    probe = _release_probe()
    runner._probes = [probe]
    base_verdict = JudgeVerdict(
        sample_id="target-0",
        score=1.0,
        model_output="base raw",
        explanation="base evidence",
    )
    ft_verdict = JudgeVerdict(
        sample_id="target-0",
        score=0.2,
        model_output="ft raw",
        explanation="ft evidence",
    )
    regressions = runner._find_regressions(
        {"target": _score("target", 1.0).model_copy(update={"sample_verdicts": [base_verdict]})},
        {"target": _score("target", 0.2).model_copy(update={"sample_verdicts": [ft_verdict]})},
    )
    assert len(regressions) == 1
    assert regressions[0].base_answer == "base raw"
    assert regressions[0].ft_answer == "ft raw"
    assert regressions[0].prompt == "q"


def test_runner_deep_analysis_error_is_structured(monkeypatch) -> None:
    config = EvalConfig(
        base_model="base",
        finetuned_model="ft",
        general_probes=["math"],
        deep_analysis=True,
        deep_analysis_samples=3,
        cache_baseline=False,
    )
    runner = EvalRunner(config)

    def fail(*_args, **_kwargs):
        raise RuntimeError("analysis allocation failed")

    monkeypatch.setattr(
        "finetunecheck.utils.model_loader.ModelLoader.load_for_analysis",
        fail,
    )
    report = runner._run_deep_analysis(
        ModelSpec(path="base", model_type=ModelType.HF),
        ModelSpec(path="ft", model_type=ModelType.HF),
    )
    assert report.status == MeasurementStatus.ERROR
    assert report.samples_requested == 3
    assert "analysis allocation failed" in (report.component_status["orchestrator"].error or "")


def test_runner_report_export_digest_and_cleanup_helpers(tmp_path: Path) -> None:
    output = tmp_path / "results.json"
    config = EvalConfig(
        base_model="base",
        finetuned_model="ft",
        general_probes=["math"],
        output_report=str(output),
        output_format="json",
        cache_baseline=False,
    )
    runner = EvalRunner(config)
    results = SimpleNamespace(model_dump_json=lambda **_kwargs: "{}")
    runner._write_configured_report(results)
    assert output.read_text(encoding="utf-8").strip() == "{}"

    probe = _release_probe()
    assert len(runner._probe_digest(probe)) == 64
    assert len(runner._probe_collection_digest([probe])) == 64
    assert runner._has_tiny_seed({"x": _score("x", 1.0, count=1)}) is True

    class Cleanup:
        def __init__(self) -> None:
            self.called = False

        def cleanup(self) -> None:
            self.called = True
            raise RuntimeError("cleanup should be suppressed")

    backend = Cleanup()
    runner._cleanup_backend(backend)
    assert backend.called is True
    runner._release_analysis_model(None)


def test_runner_single_model_unknown_probe_and_execution_preflight(tmp_path: Path) -> None:
    class Backend:
        model_path = "fake"
        backend_name = "fake"

        def __init__(self) -> None:
            self.cleaned = False

        def generate_batch(self, prompts, *, probe_name, **_kwargs):
            return [
                InferenceResult(
                    model_path=self.model_path,
                    probe_name=probe_name,
                    output="positive",
                )
                for _ in prompts
            ]

        def cleanup(self) -> None:
            self.cleaned = True

    backend = Backend()
    config = EvalConfig(
        base_model="same",
        finetuned_model="same",
        general_probes=["classification"],
        num_samples=1,
        cache_baseline=False,
    )
    scores = EvalRunner(config, backend_factory=lambda *_args: backend).run_single_model()
    assert scores["classification"].status == MeasurementStatus.MEASURED
    assert backend.cleaned is True

    unknown = EvalRunner(
        config.model_copy(update={"general_probes": ["brand_new_category"]})
    )._build_probes()
    assert unknown[0].name == "brand_new_category"
    assert unknown[0].provenance["supports_release_claims"] is False

    execution_runner = EvalRunner(config.model_copy(update={"general_probes": ["code"]}))
    execution_runner._preflight([ProbeRegistry.get("code")])

    local = tmp_path / "model"
    local.mkdir()
    assert EvalRunner._canonical_model_identity(str(local)).startswith("local:")


def test_runner_resolves_api_judge_and_adapter_identity_rejections() -> None:
    base = ModelSpec(path="base", model_type=ModelType.HF)
    ft = ModelSpec(path="ft", model_type=ModelType.LORA, base_model="adapter-base")
    loader = SimpleNamespace(
        detect_type=lambda value: ModelSpec(path=value, model_type=ModelType.HF)
    )
    api_runner = EvalRunner(
        EvalConfig(
            base_model="base",
            finetuned_model="ft",
            general_probes=["reasoning"],
            judge=JudgeConfig(provider="openai", model="judge"),
            cache_baseline=False,
        )
    )
    api_runner._resolve_configured_judge(loader, base, ft)
    assert isinstance(api_runner._judge_provider, APIJudgeProvider)

    adapter_judge_loader = SimpleNamespace(
        detect_type=lambda value: ModelSpec(
            path=value,
            model_type=ModelType.LORA,
            base_model="adapter-base",
        )
    )
    local_runner = EvalRunner(
        EvalConfig(
            base_model="base",
            finetuned_model="ft",
            general_probes=["reasoning"],
            judge=JudgeConfig(provider="local", model="judge-adapter"),
            cache_baseline=False,
        ),
        backend_factory=lambda *_args: SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="adapter is based"):
        local_runner._resolve_configured_judge(adapter_judge_loader, base, ft)

    base_adapter = ModelSpec(
        path="base-adapter",
        model_type=ModelType.LORA,
        base_model="base-foundation",
    )
    base_parent_runner = EvalRunner(
        EvalConfig(
            base_model="base-adapter",
            finetuned_model="ft",
            general_probes=["reasoning"],
            judge=JudgeConfig(provider="local", model="base-foundation"),
            cache_baseline=False,
        ),
        backend_factory=lambda *_args: pytest.fail("rejected judge was loaded"),
    )
    with pytest.raises(ValueError, match="must never judge itself"):
        base_parent_runner._resolve_configured_judge(loader, base_adapter, base)


def test_runner_deep_success_html_output_and_missing_safety(
    sample_eval_results,
    tmp_path: Path,
    monkeypatch,
) -> None:
    loaded: list[SimpleNamespace] = []

    def load(*_args, **_kwargs):
        analysis = SimpleNamespace(model=SimpleNamespace())
        loaded.append(analysis)
        return analysis

    expected = DeepAnalysisReport(status=MeasurementStatus.MEASURED)
    monkeypatch.setattr("finetunecheck.utils.model_loader.ModelLoader.load_for_analysis", load)
    monkeypatch.setattr(
        "finetunecheck.deep_analysis.orchestrator.DeepAnalysisOrchestrator.run",
        lambda *_args, **_kwargs: expected,
    )
    config = EvalConfig(
        base_model="base",
        finetuned_model="ft",
        general_probes=["math"],
        output_report=str(tmp_path / "report.html"),
        cache_baseline=False,
    )
    runner = EvalRunner(config)
    report = runner._run_deep_analysis(
        ModelSpec(path="base", model_type=ModelType.HF),
        ModelSpec(path="ft", model_type=ModelType.HF),
    )
    assert report is expected
    assert all(not hasattr(item, "model") for item in loaded)

    runner._write_configured_report(sample_eval_results)
    assert Path(config.output_report or "").is_file()
    runner.config.output_report = None
    runner._write_configured_report(sample_eval_results)
    assert EvalRunner._safety_measurement(None).status == MeasurementStatus.NOT_RUN


def test_verdict_records_selective_and_gradual_concerns() -> None:
    runner = EvalRunner(
        EvalConfig(
            base_model="base",
            finetuned_model="ft",
            target_tasks=["target"],
            general_probes=[],
            cache_baseline=False,
        )
    )
    runner._probes = [_release_probe()]
    scores = {"target": _score("target", 0.8)}
    for pattern, phrase in (
        (ForgettingPattern.SELECTIVE, "selective forgetting"),
        (ForgettingPattern.GRADUAL, "Gradual"),
    ):
        verdict, _, concerns, recommendations = runner._compute_verdict(
            0.2,
            _forgetting(pattern=pattern),
            scores,
            scores,
            _roi(70),
        )
        assert verdict == Verdict.GOOD_WITH_CONCERNS
        assert any(phrase in concern for concern in concerns)
        if pattern == ForgettingPattern.SELECTIVE:
            assert recommendations


def test_mcp_call_tool_marks_handler_exceptions_as_protocol_errors(monkeypatch) -> None:
    from finetunecheck.mcp import server as server_module

    async def fail(_arguments):
        raise ValueError("controlled")

    monkeypatch.setitem(server_module.TOOL_HANDLERS, "controlled", fail)
    result = asyncio.run(server_module.call_tool("controlled", {}))
    assert result.isError is True
    assert "controlled" in result.content[0].text
