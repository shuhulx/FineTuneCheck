"""Offline release-contract tests for FineTuneCheck 2.0.0."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from finetunecheck.config import EvalConfig
from finetunecheck.eval.judge import (
    CallableJudgeProvider,
    ExecutionJudge,
    Executor,
    RuleBasedJudge,
)
from finetunecheck.eval.runner import EvalRunner
from finetunecheck.models import (
    CategoryScore,
    InferenceResult,
    MeasurementStatus,
    ProbeSample,
    Verdict,
)
from finetunecheck.models import (
    TestCaseOutcome as CodeTestOutcome,
)
from finetunecheck.probes.registry import ProbeRegistry


class FakeBackend:
    """Small deterministic backend that never imports an inference framework."""

    backend_name = "fake-offline"

    def __init__(
        self,
        model_path: str,
        outputs: dict[str, str] | None = None,
        *,
        fail_probe: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.outputs = outputs or {}
        self.fail_probe = fail_probe
        self.cleaned = False
        self.calls: list[tuple[str, list[str]]] = []

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        probe_name: str = "",
        **settings: Any,
    ) -> list[InferenceResult]:
        del max_tokens, settings
        self.calls.append((probe_name, list(prompts)))
        if probe_name == self.fail_probe:
            raise RuntimeError("offline fake inference failure")
        return [
            InferenceResult(
                model_path=self.model_path,
                probe_name=probe_name,
                sample_id=str(index),
                output=self.outputs.get(prompt, "candidate output"),
                latency_ms=1.0,
                backend=self.backend_name,
            )
            for index, prompt in enumerate(prompts)
        ]

    def get_logprobs(self, texts: list[str]) -> list[list[float]]:
        return [[-0.1] for _ in texts]

    def cleanup(self) -> None:
        self.cleaned = True


def _reference_outputs(*probe_names: str) -> dict[str, str]:
    return {
        sample.input: sample.reference or ""
        for probe_name in probe_names
        for sample in ProbeRegistry.get(probe_name).samples
    }


def _paired_runner(
    config: EvalConfig,
    *,
    base_outputs: dict[str, str] | None = None,
    ft_outputs: dict[str, str] | None = None,
    judge_provider: CallableJudgeProvider | None = None,
    fail_ft_probe: str | None = None,
) -> tuple[EvalRunner, list[FakeBackend]]:
    backends: list[FakeBackend] = []

    def factory(spec, _device):
        outputs = base_outputs if spec.path == config.base_model else ft_outputs
        backend = FakeBackend(
            spec.path,
            outputs,
            fail_probe=fail_ft_probe if spec.path == config.finetuned_model else None,
        )
        backends.append(backend)
        return backend

    return (
        EvalRunner(
            config,
            backend_factory=factory,
            judge_provider=judge_provider,
        ),
        backends,
    )


def test_fake_backend_end_to_end_preserves_evidence_and_gates_small_seeds() -> None:
    probes = ("math", "classification", "instruction_following", "safety")
    outputs = _reference_outputs(*probes)
    config = EvalConfig(
        base_model="fake-base",
        finetuned_model="fake-ft",
        target_tasks=["math"],
        general_probes=["classification", "instruction_following", "safety"],
        num_samples=2,
        cache_baseline=False,
    )
    runner, backends = _paired_runner(
        config,
        base_outputs=outputs,
        ft_outputs=outputs,
    )

    results = runner.run()

    assert results.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert results.target_improvement == pytest.approx(0.0)
    assert results.provenance["supports_independent_deployment_approval"] is False
    assert results.safety_smoke is not None
    assert results.safety_smoke.supports_deployment_claims is False
    assert set(results.base_scores) == set(probes)
    assert all(score.status == MeasurementStatus.MEASURED for score in results.ft_scores.values())
    assert all(score.selected_sample_ids for score in results.ft_scores.values())
    assert all(
        verdict.model_output is not None and verdict.details["inference_backend"] == "fake-offline"
        for score in results.ft_scores.values()
        for verdict in score.sample_verdicts
    )
    assert all(backend.cleaned for backend in backends)


def test_fake_llm_provider_is_used_and_provenance_is_retained() -> None:
    config = EvalConfig(
        base_model="fake-base",
        finetuned_model="fake-ft",
        target_tasks=["reasoning"],
        general_probes=[],
        num_samples=2,
        cache_baseline=False,
    )
    provider = CallableJudgeProvider(
        lambda _prompt: '{"score": 8, "explanation": "offline evidence"}',
        name="fixture",
        model="judge-only",
        settings={"seed": 7},
    )
    runner, _ = _paired_runner(config, judge_provider=provider)

    results = runner.run()

    assert results.base_scores["reasoning"].mean_score == pytest.approx(0.8)
    assert results.ft_scores["reasoning"].mean_score == pytest.approx(0.8)
    assert results.judge_provenance == {
        "provider": "fixture",
        "model": "judge-only",
        "settings": {"seed": 7},
    }
    verdict = results.ft_scores["reasoning"].sample_verdicts[0]
    assert verdict.raw_judge_output is not None
    assert verdict.explanation == "offline evidence"


def test_fake_llm_error_becomes_error_evidence_not_a_neutral_score() -> None:
    config = EvalConfig(
        base_model="fake-base",
        finetuned_model="fake-ft",
        target_tasks=["reasoning"],
        general_probes=[],
        num_samples=1,
        cache_baseline=False,
    )
    provider = CallableJudgeProvider(lambda _prompt: "not JSON", model="judge-only")
    runner, _ = _paired_runner(config, judge_provider=provider)

    results = runner.run()

    score = results.ft_scores["reasoning"]
    assert score.status == MeasurementStatus.ERROR
    assert score.mean_score is None
    assert score.sample_scores == []
    assert score.sample_verdicts[0].error
    assert results.verdict == Verdict.INSUFFICIENT_EVIDENCE


def test_fake_inference_error_is_missing_evidence_and_never_good() -> None:
    config = EvalConfig(
        base_model="fake-base",
        finetuned_model="fake-ft",
        target_tasks=["classification"],
        general_probes=["math"],
        num_samples=1,
        cache_baseline=False,
    )
    outputs = _reference_outputs("classification", "math")
    runner, _ = _paired_runner(
        config,
        base_outputs=outputs,
        ft_outputs=outputs,
        fail_ft_probe="classification",
    )

    results = runner.run()

    assert results.ft_scores["classification"].status == MeasurementStatus.ERROR
    assert "fake inference failure" in (results.ft_scores["classification"].error or "")
    assert results.target_improvement is None
    assert results.verdict == Verdict.INSUFFICIENT_EVIDENCE


class StaticExecutor(Executor):
    """Fake external boundary: supplies predetermined values without running code."""

    def __init__(self, actual_values: list[Any], errors: set[int] | None = None) -> None:
        self.actual_values = actual_values
        self.errors = errors or set()
        self.received_code: str | None = None
        self.received_cases: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "executor": "static-test-double",
            "network": "none",
            "host_execution": False,
        }

    def execute(
        self,
        code: str,
        test_cases: list[dict[str, Any]],
        *,
        timeout_seconds: int,
    ) -> list[CodeTestOutcome]:
        assert timeout_seconds > 0
        self.received_code = code
        self.received_cases = list(test_cases)
        return [
            CodeTestOutcome(
                index=index,
                expression=str(case["input"]),
                actual=self.actual_values[index],
                error="isolated failure" if index in self.errors else None,
            )
            for index, case in enumerate(test_cases)
        ]


def _expected_values(sample: ProbeSample) -> list[Any]:
    return [
        ExecutionJudge._parse_expected(case["output"]) for case in sample.metadata["test_cases"]
    ]


def test_all_bundled_code_references_pass_every_fake_isolated_case() -> None:
    for sample in ProbeRegistry.get("code").samples:
        executor = StaticExecutor(_expected_values(sample))
        verdict = ExecutionJudge(executor=executor).evaluate(
            sample,
            f"```python\n{sample.reference}\n```",
        )
        assert verdict.score == 1.0, sample.id
        assert len(verdict.test_cases) == len(sample.metadata["test_cases"])
        assert all(outcome.passed for outcome in verdict.test_cases)
        assert executor.received_code == sample.reference


def test_execution_scoring_uses_all_cases_and_awards_partial_credit() -> None:
    sample = ProbeRegistry.get("code").samples[0]
    values = _expected_values(sample)
    values[-1] = object()
    executor = StaticExecutor(values)

    verdict = ExecutionJudge(executor=executor).evaluate(
        sample,
        f"```python\n{sample.reference}\n```",
    )

    assert verdict.score == pytest.approx((len(values) - 1) / len(values))
    assert len(verdict.test_cases) == len(values)
    assert verdict.test_cases[-1].passed is False


def test_printing_reference_source_scores_zero_without_host_execution() -> None:
    sample = ProbeRegistry.get("code").samples[0]
    executor = StaticExecutor([None] * len(sample.metadata["test_cases"]))
    output = f"```python\nprint({sample.reference!r})\n```"

    verdict = ExecutionJudge(executor=executor).evaluate(sample, output)

    assert verdict.score == 0.0
    assert all(not outcome.passed for outcome in verdict.test_cases)
    assert executor.received_code == f"print({sample.reference!r})"


def test_code_harness_supports_setup_followed_by_a_final_expression() -> None:
    source = ExecutionJudge.build_test_harness(
        "def answer(value):\n    return value + 1",
        "value = 41\nanswer(value)",
    )
    assert "value = 41" in source
    assert "__finetunecheck_result__ = (answer(value))" in source
    with pytest.raises(ValueError, match="end with an expression"):
        ExecutionJudge.build_test_harness("pass", "value = 41")


@pytest.mark.parametrize(
    ("constraint", "passing", "failing"),
    [
        ({"type": "line_count", "value": 2}, "a\nb", "a"),
        ({"type": "starts_with", "value": "-", "per_line": True}, "- a\n- b", "a\n- b"),
        ({"type": "valid_json"}, '{"a": 1}', "not json"),
        ({"type": "json_keys", "value": ["a"]}, '{"a": 1}', '{"b": 1}'),
        ({"type": "json_key_count", "value": 1}, '{"a": 1}', '{"a": 1, "b": 2}'),
        ({"type": "sentence_count", "value": 1}, "One.", "One. Two."),
        ({"type": "max_words", "value": 2}, "one two", "one two three"),
        ({"type": "min_words", "value": 2}, "one two", "one"),
        ({"type": "exact_words", "value": 2}, "one two", "one"),
        ({"type": "words_per_line", "value": 2}, "one two\nred blue", "one\nred blue"),
        ({"type": "all_uppercase"}, "ALL CAPS", "Not CAPS"),
        ({"type": "numbered_list"}, "1. one\n2. two", "- one\n- two"),
        ({"type": "contains_all", "value": ["one", "two"]}, "two one", "one"),
        ({"type": "contains_pattern", "value": r"^[A-Z]+$"}, "ABC", "Abc"),
        ({"type": "starts_with_text", "value": "Start"}, "Start here", "No start"),
        ({"type": "ends_with_text", "value": "end."}, "the end.", "the end"),
        ({"type": "not_contains", "value": "blocked"}, "allowed", "blocked"),
        ({"type": "not_contains_word", "value": "no"}, "nobody", "say no"),
        ({"type": "contains", "value": "|"}, "a | b", "a b"),
        ({"type": "table_columns", "value": 2}, "| a | b |\n|---|---|", "| a | b | c |"),
        (
            {"type": "table_data_rows", "value": 1},
            "| a | b |\n|---|---|\n| 1 | 2 |",
            "| a | b |\n|---|---|",
        ),
        ({"type": "one_of", "value": ["yes", "no"]}, "yes", "maybe"),
        ({"type": "acrostic", "value": "AB"}, "Apple\nBerry", "Apple\nCherry"),
    ],
)
def test_every_instruction_constraint_has_a_real_failure_case(
    constraint: dict[str, Any], passing: str, failing: str
) -> None:
    passed, _ = RuleBasedJudge._check_constraint(constraint, passing)
    rejected, _ = RuleBasedJudge._check_constraint(constraint, failing)
    assert passed is True
    assert rejected is False


def test_safety_smoke_reports_harmful_refusal_and_benign_overrefusal_rates() -> None:
    harmful = ProbeSample(
        id="harmful",
        input="harmful",
        metadata={"should_refuse": True},
    )
    benign = ProbeSample(
        id="benign",
        input="benign",
        metadata={"should_refuse": False},
    )
    judge = RuleBasedJudge(category="safety")
    score = CategoryScore(
        category="safety",
        mean_score=0.5,
        num_samples=2,
        expected_samples=2,
        sample_scores=[1.0, 0.0],
        sample_verdicts=[
            judge.evaluate(harmful, "I cannot help with that."),
            judge.evaluate(benign, "I cannot help with that."),
        ],
    )

    measurement = EvalRunner._safety_measurement(score)

    assert measurement.status == MeasurementStatus.MEASURED
    assert measurement.harmful_refusal_rate == 1.0
    assert measurement.benign_overrefusal_rate == 1.0


def test_mcp_registers_exactly_nine_tools_and_uses_protocol_errors() -> None:
    from finetunecheck.mcp.server import call_tool, list_tools

    tools = asyncio.run(list_tools())
    assert len(tools) == 9
    assert {tool.name for tool in tools} == {
        "evaluate_finetune",
        "quick_check",
        "detect_forgetting",
        "compare_runs",
        "get_verdict",
        "suggest_fixes",
        "generate_report",
        "list_profiles",
        "run_probe",
    }
    unknown = asyncio.run(call_tool("not-a-tool", {}))
    assert unknown.isError is True


def test_mcp_worker_yields_to_the_event_loop() -> None:
    from finetunecheck.mcp.tools import _run_blocking

    async def scenario() -> bool:
        blocking = asyncio.create_task(_run_blocking(lambda: time.sleep(0.05)))
        await asyncio.sleep(0.005)
        yielded_before_completion = not blocking.done()
        await blocking
        return yielded_before_completion

    assert asyncio.run(scenario()) is True


def test_mcp_report_path_is_rejected_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finetunecheck.mcp.tools import handle_generate_report

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside-report.html"
    arguments = {
        "base_model": "base",
        "finetuned_model": "ft",
        "output_path": str(outside),
    }
    with pytest.raises(ValueError, match="working directory"):
        asyncio.run(handle_generate_report(arguments))


@pytest.mark.parametrize("overwrite", [False, True])
def test_mcp_report_commit_never_follows_a_raced_destination_symlink(
    sample_eval_results,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    from finetunecheck.eval import runner as runner_module
    from finetunecheck.mcp.tools import handle_generate_report

    victim = tmp_path / "victim.json"
    victim.write_text("do not replace", encoding="utf-8")
    destination = tmp_path / "report.json"

    class Runner:
        def __init__(self, _config):
            pass

        def run(self):
            destination.symlink_to(victim)
            return sample_eval_results

    monkeypatch.setattr(runner_module, "EvalRunner", Runner)
    monkeypatch.chdir(tmp_path)
    arguments = {
        "base_model": "base",
        "finetuned_model": "ft",
        "output_path": destination.name,
        "format": "json",
        "overwrite": overwrite,
    }

    if overwrite:
        message = asyncio.run(handle_generate_report(arguments))
        assert "Report generated" in message
        assert destination.is_file()
        assert destination.is_symlink() is False
        assert json.loads(destination.read_text(encoding="utf-8"))["package_version"] == "2.0.2"
    else:
        with pytest.raises(ValueError, match="overwrite"):
            asyncio.run(handle_generate_report(arguments))
        assert destination.is_symlink()
    assert victim.read_text(encoding="utf-8") == "do not replace"


def test_mcp_report_commit_rejects_a_raced_parent_directory(
    sample_eval_results,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from finetunecheck.eval import runner as runner_module
    from finetunecheck.mcp.tools import handle_generate_report

    parent = tmp_path / "reports"
    displaced_parent = tmp_path / "reports-original"
    victim_directory = tmp_path / "victim"
    parent.mkdir()
    victim_directory.mkdir()

    class Runner:
        def __init__(self, _config):
            pass

        def run(self):
            parent.rename(displaced_parent)
            parent.symlink_to(victim_directory, target_is_directory=True)
            return sample_eval_results

    monkeypatch.setattr(runner_module, "EvalRunner", Runner)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="parent changed"):
        asyncio.run(
            handle_generate_report(
                {
                    "base_model": "base",
                    "finetuned_model": "ft",
                    "output_path": "reports/report.json",
                    "format": "json",
                }
            )
        )
    assert not (victim_directory / "report.json").exists()
    assert not (displaced_parent / "report.json").exists()


def test_report_browser_has_no_errors_renders_charts_and_blocks_script_injection(
    sample_eval_results,
    tmp_path: Path,
) -> None:
    if os.environ.get("FINETUNECHECK_BROWSER_TEST") != "1":
        pytest.skip("Browser rendering runs in the dedicated browser test job")
    playwright = pytest.importorskip("playwright.sync_api")
    from finetunecheck.report.generator import ReportGenerator

    payload = '</script><script>window.__finetunecheck_injected = true</script><span data-x="'
    sample_eval_results.base_model = payload
    sample_eval_results.finetuned_model = payload
    sample_eval_results.summary = payload
    output = tmp_path / "adversarial-report.html"
    ReportGenerator().generate(sample_eval_results, str(output))

    console_errors: list[str] = []
    page_errors: list[str] = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page()
        page.on(
            "console",
            lambda message: (
                console_errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(output.resolve().as_uri(), wait_until="load")
        page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length >= 3")
        rendered = page.locator(".js-plotly-plot").count()
        injected = page.evaluate("Boolean(window.__finetunecheck_injected)")
        browser.close()

    assert rendered >= 3
    assert console_errors == []
    assert page_errors == []
    assert injected is False


def test_report_and_result_provenance_use_release_version(
    sample_eval_results, tmp_path: Path
) -> None:
    from finetunecheck import __version__
    from finetunecheck.report.generator import ReportGenerator

    assert sample_eval_results.package_version == __version__ == "2.0.2"
    output = tmp_path / "version-report.html"
    ReportGenerator().generate(sample_eval_results, str(output))
    assert "FineTuneCheck v2.0.2" in output.read_text(encoding="utf-8")


def test_mcp_schemas_are_closed_and_positive_counts_are_enforced() -> None:
    from finetunecheck.mcp import schemas

    schema_names = [name for name in dir(schemas) if name.endswith("_SCHEMA")]
    assert len(schema_names) == 9
    for name in schema_names:
        schema = getattr(schemas, name)
        assert schema["additionalProperties"] is False
    for schema in (
        schemas.EVALUATE_FINETUNE_SCHEMA,
        schemas.DETECT_FORGETTING_SCHEMA,
        schemas.COMPARE_RUNS_SCHEMA,
        schemas.GENERATE_REPORT_SCHEMA,
    ):
        assert schema["properties"]["num_samples"]["minimum"] == 1
        assert schema["properties"]["device"]["enum"] == ["auto", "cpu", "cuda", "mps"]


def test_serialized_result_has_schema_metric_and_package_provenance(sample_eval_results) -> None:
    payload = json.loads(sample_eval_results.model_dump_json())
    assert payload["result_schema_version"] == "2.0.0"
    assert payload["metric_schema_version"] == "2.0.0"
    assert payload["package_version"] == "2.0.2"
