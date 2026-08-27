"""Operational CLI, MCP, export, and utility checks without external model access."""

from __future__ import annotations

import asyncio
import io
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from finetunecheck.baselines.manager import BaselineManager
from finetunecheck.compare.multi_run import ComparisonResult
from finetunecheck.forgetting.analyzer import ForgettingAnalyzer
from finetunecheck.forgetting.detector import ForgettingDetector
from finetunecheck.models import (
    CategoryScore,
    ForgettingPattern,
    JudgeType,
    JudgeVerdict,
    MeasurementStatus,
    ProbeSample,
    ProbeSet,
    Verdict,
)
from finetunecheck.report.exporters import CSVExporter, JSONExporter, MarkdownExporter


def _score(category: str, value: float | None, *, status=None) -> CategoryScore:
    if value is None:
        return CategoryScore(
            category=category,
            status=status or MeasurementStatus.NOT_RUN,
        )
    return CategoryScore(
        category=category,
        mean_score=value,
        num_samples=1,
        sample_scores=[value],
    )


def test_real_exporters_preserve_versions_missing_statuses_and_deep_evidence(
    sample_eval_results,
    sample_deep_analysis,
    tmp_path: Path,
) -> None:
    sample_eval_results.deep_analysis = sample_deep_analysis
    sample_eval_results.ft_scores.pop("code")
    sample_eval_results.base_scores["missing"] = _score(
        "missing", None, status=MeasurementStatus.ERROR
    )

    json_path = Path(
        JSONExporter.export(sample_eval_results, str(tmp_path / "nested/results.json"))
    )
    csv_path = Path(CSVExporter.export(sample_eval_results, str(tmp_path / "results.csv")))
    markdown_path = Path(MarkdownExporter.export(sample_eval_results, str(tmp_path / "results.md")))

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["package_version"] == "2.0.2"
    assert payload["deep_analysis"]["calibration"] is not None
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "base_status" in csv_text
    assert "MISSING" in csv_text
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Deep Analysis" in markdown
    assert "Unavailable" in markdown or "MISSING" in markdown
    assert "Review the samples before shipping" in markdown


def test_exporters_handle_no_forgetting_no_target_no_summary(tmp_path: Path) -> None:
    from finetunecheck.models import EvalResults

    results = EvalResults(
        base_model="base",
        finetuned_model="ft",
        base_scores={"math": _score("math", 1.0)},
        ft_scores={"math": _score("math", 0.5)},
    )
    CSVExporter.export(results, str(tmp_path / "plain.csv"))
    MarkdownExporter.export(results, str(tmp_path / "plain.md"))
    assert "Retention" not in (tmp_path / "plain.csv").read_text(encoding="utf-8")
    assert "Target improvement:** unavailable" in (tmp_path / "plain.md").read_text(
        encoding="utf-8"
    )


class MemoryCache:
    def __init__(self) -> None:
        self.values: dict[str, CategoryScore] = {}

    @staticmethod
    def get_key(model: str, probe: str, count: int) -> str:
        return f"{model}:{probe}:{count}"

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value: CategoryScore) -> None:
        self.values[key] = value


def test_baseline_manager_runtime_and_local_precomputed_paths(tmp_path: Path) -> None:
    manager = BaselineManager()
    manager._data_dir = tmp_path
    manager.PRECOMPUTED = {"known": "known.json", "bad": "bad.json"}
    score = _score("math", 0.8)
    (tmp_path / "known.json").write_text(
        json.dumps({"math": score.model_dump(mode="json")}),
        encoding="utf-8",
    )
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    cache = MemoryCache()

    assert manager.get_baseline("known", "math", 1, cache) == score
    assert manager.get_baseline("unknown", "math", 1, cache) is None
    assert manager.get_baseline("bad", "math", 1, cache) is None
    assert manager._load_precomputed("known", "missing") is None
    manager.save_baseline("runtime", "math", 1, score, cache)
    assert manager.get_baseline("runtime", "math", 1, cache) == score

    (tmp_path / "known.json").write_text(
        json.dumps({"math": {"category": "math", "unexpected": True}}),
        encoding="utf-8",
    )
    cache.values.clear()
    assert manager.get_baseline("known", "math", 1, cache) is None


def test_shipped_basic_example_executes(monkeypatch, capsys, sample_eval_results) -> None:
    import finetunecheck.eval.runner as runner_module

    class ExampleRunner:
        def __init__(self, config) -> None:
            assert config.base_model == "local-base"
            assert config.finetuned_model == "local-ft"

        def run(self):
            return sample_eval_results

    monkeypatch.setattr(runner_module, "EvalRunner", ExampleRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        ["basic_evaluation.py", "local-base", "local-ft"],
    )
    example = Path(__file__).parents[1] / "examples" / "basic_evaluation.py"
    runpy.run_path(str(example), run_name="__main__")

    output = capsys.readouterr().out
    assert "Verdict:" in output
    assert "ROI Score:" in output
    assert "domain checks before shipping" in output


def test_forgetting_analyzer_retains_evidence_and_missing_semantics() -> None:
    base = {
        "target": _score("target", 0.5),
        "general": _score("general", 1.0),
        "missing": _score("missing", 0.8),
        "safety": _score("safety", 1.0),
    }
    ft = {
        "target": _score("target", 0.8),
        "general": _score("general", 0.6),
        "safety": _score("safety", 0.9),
    }
    probe = ProbeSet(
        name="general",
        category="general",
        judge_type=JudgeType.EXACT_MATCH,
        samples=[ProbeSample(id="s", input="prompt", reference="answer")],
    )
    base_verdict = JudgeVerdict(
        sample_id="s",
        score=1.0,
        model_output="base raw",
        explanation="base reason",
    )
    ft_verdict = JudgeVerdict(
        sample_id="s",
        score=0.4,
        model_output="ft raw",
        explanation="ft reason",
    )
    report = ForgettingAnalyzer.analyze(
        base,
        ft,
        base_results={"general": [base_verdict], "only_base": [base_verdict]},
        ft_results={"general": [ft_verdict]},
        probes={"general": probe},
        target_categories=["target"],
        target_task="target",
    )
    assert report.status == MeasurementStatus.INSUFFICIENT_SAMPLE
    assert report.missing_categories == ["missing"]
    assert report.pattern in {
        ForgettingPattern.CATASTROPHIC,
        ForgettingPattern.SELECTIVE,
    }
    assert report.regressions[0].base_answer == "base raw"
    assert report.regressions[0].ft_answer == "ft raw"


@pytest.mark.parametrize(
    ("crr", "bwt", "sfi", "expected"),
    [
        ({}, None, None, ForgettingPattern.UNAVAILABLE),
        ({"a": 0.5, "b": 0.6}, -0.3, 0.05, ForgettingPattern.CATASTROPHIC),
        ({"a": 0.69, "b": 1.0, "c": 1.0}, -0.01, 0.2, ForgettingPattern.SELECTIVE),
        ({"a": 0.9}, -0.06, 0.0, ForgettingPattern.GRADUAL),
        ({"a": 0.99}, 0.0, 0.0, ForgettingPattern.MINIMAL),
        ({"a": 0.9}, 0.0, 0.0, ForgettingPattern.GRADUAL),
    ],
)
def test_forgetting_detector_table(crr, bwt, sfi, expected) -> None:
    assert ForgettingDetector.classify_pattern(crr, bwt, sfi) == expected


def test_forgetting_detector_sorts_affected_and_resilient() -> None:
    affected, resilient = ForgettingDetector.identify_affected_capabilities(
        {"worst": 0.5, "less_bad": 0.8, "best": 1.1, "ok": 1.0, "missing": None}
    )
    assert affected == ["worst", "less_bad"]
    assert resilient == ["best", "ok"]


def test_terminal_formatters_cover_measured_missing_and_all_messages(monkeypatch) -> None:
    from rich.console import Console

    from finetunecheck.utils import formatting

    stream = io.StringIO()
    monkeypatch.setattr(
        formatting,
        "console",
        Console(file=stream, force_terminal=False, width=120),
    )
    formatting.print_verdict(Verdict.INSUFFICIENT_EVIDENCE, None, "not enough evidence")
    formatting.print_category_scores(
        {
            "gain": _score("gain", 0.5),
            "drop": _score("drop", 0.9),
            "base_only": _score("base_only", None),
        },
        {
            "gain": _score("gain", 0.8),
            "drop": _score("drop", 0.5),
            "ft_only": _score("ft_only", None, status=MeasurementStatus.ERROR),
        },
    )
    formatting.print_concerns([])
    formatting.print_concerns(["concern"])
    formatting.print_recommendations([])
    formatting.print_recommendations(["recommendation"])
    formatting.print_progress("half", 1, 2)
    formatting.print_progress("done", 0, 0)
    text = stream.getvalue()
    assert "INSUFFICIENT EVIDENCE" in text
    assert "concern" in text
    assert "recommendation" in text


def test_device_helpers_are_observable_and_fall_back_safely(monkeypatch) -> None:
    import torch

    from finetunecheck.utils.device import detect_device, get_device_info, resolve_model_device

    assert detect_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="falling back"):
        assert detect_device("cuda") == "cpu"
    assert detect_device("auto") == "cpu"
    assert detect_device("custom-accelerator") == "custom-accelerator"
    info = get_device_info()
    assert info["device"] == "cpu"
    assert info["device_name"] == "CPU"
    assert resolve_model_device(SimpleNamespace(device="mps")) == "mps"
    assert (
        resolve_model_device(SimpleNamespace(model=SimpleNamespace(parameters=lambda: iter(()))))
        == "cpu"
    )


def test_mcp_handlers_cover_every_operation_with_protocol_summaries(
    sample_eval_results,
    monkeypatch,
    tmp_path: Path,
) -> None:
    from finetunecheck.compare import multi_run
    from finetunecheck.eval import runner as runner_module
    from finetunecheck.mcp import tools

    captured_configs = []

    class Runner:
        def __init__(self, config):
            captured_configs.append(config)

        def run(self):
            return sample_eval_results

        def run_single_model(self):
            return {"classification": _score("classification", 0.75)}

    comparison = ComparisonResult(
        base_model="base",
        runs={"run": sample_eval_results},
        best_run="run",
        best_target_perf="run",
        least_forgetting="run",
        pareto_frontier=["run"],
        recommendation="Use run.",
    )

    class Comparator:
        def compare(self, *_args, **_kwargs):
            return comparison

    monkeypatch.setattr(runner_module, "EvalRunner", Runner)
    monkeypatch.setattr(multi_run, "MultiRunComparator", Comparator)
    monkeypatch.chdir(tmp_path)

    common = {
        "base_model": "base",
        "finetuned_model": "ft",
        "target_task": "math",
        "num_samples": 1,
        "device": "auto",
    }
    assert tools._targets(common) == ["math"]
    assert tools._targets({"target_tasks": ["math", "classification"]}) == [
        "math",
        "classification",
    ]
    assert tools._targets({}) == []
    assert tools._judge({}) is None
    assert tools._judge({"judge": {"provider": "local", "model": "judge"}}).model == "judge"
    assert tools._paired_config(common).device == "auto"
    summary = tools._results_summary(sample_eval_results)
    assert "run your own checks before shipping" in summary

    evaluated = asyncio.run(tools.handle_evaluate_finetune(common | {"profile": "classification"}))
    quick = asyncio.run(tools.handle_quick_check(common))
    forgetting = asyncio.run(tools.handle_detect_forgetting(common))
    compared = asyncio.run(
        tools.handle_compare_runs(
            {
                "base_model": "base",
                "finetuned_models": {"run": "ft"},
                "target_tasks": ["math"],
                "num_samples": 1,
            }
        )
    )
    verdict = asyncio.run(tools.handle_get_verdict(common))
    fixes = asyncio.run(tools.handle_suggest_fixes(common))
    profiles = asyncio.run(tools.handle_list_profiles({}))
    probe = asyncio.run(
        tools.handle_run_probe({"model": "ft", "probe_name": "classification", "num_samples": 1})
    )
    generated = asyncio.run(
        tools.handle_generate_report(
            common
            | {
                "output_path": "result.json",
                "format": "json",
            }
        )
    )

    assert "Verdict:" in evaluated
    assert "Verdict:" in quick
    assert "Capability retention" in forgetting
    assert "Best overall: run" in compared
    assert "Verdict:" in verdict
    assert "Evidence verdict" in fixes
    assert "classification" in profiles
    assert "Mean score: 0.750" in probe
    assert "Report generated" in generated
    assert (tmp_path / "result.json").is_file()
    assert captured_configs


def test_mcp_missing_forgetting_no_recommendations_and_path_guards(
    sample_eval_results,
    monkeypatch,
    tmp_path: Path,
) -> None:
    from finetunecheck.eval import runner as runner_module
    from finetunecheck.mcp import tools

    class Runner:
        def __init__(self, _config):
            pass

        def run(self):
            return sample_eval_results.model_copy(
                update={"forgetting": None, "recommendations": []}
            )

    monkeypatch.setattr(runner_module, "EvalRunner", Runner)
    with pytest.raises(ValueError, match="not produced"):
        asyncio.run(tools.handle_detect_forgetting({"base_model": "base", "finetuned_model": "ft"}))
    fixes = asyncio.run(tools.handle_suggest_fixes({"base_model": "base", "finetuned_model": "ft"}))
    assert "No recommendation" in fixes

    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "existing.json"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        tools._validated_output_path({"output_path": str(existing)})
    assert (
        tools._validated_output_path({"output_path": str(existing), "overwrite": True}) == existing
    )
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="file, not a directory"):
        tools._validated_output_path({"output_path": str(directory), "overwrite": True})


def test_cli_commands_use_release_contracts_and_generate_requested_output(
    sample_eval_results,
    monkeypatch,
    tmp_path: Path,
) -> None:
    from finetunecheck import cli
    from finetunecheck.compare import multi_run
    from finetunecheck.eval import runner as runner_module

    monkeypatch.setattr(runner_module.EvalRunner, "run", lambda _self: sample_eval_results)
    comparison = ComparisonResult(
        base_model="base",
        runs={"run_0": sample_eval_results},
        best_run="run_0",
        best_target_perf="run_0",
        least_forgetting="run_0",
        pareto_frontier=["run_0"],
        recommendation="Use run_0.",
    )
    monkeypatch.setattr(
        multi_run.MultiRunComparator,
        "compare",
        lambda _self, *_args, **_kwargs: comparison,
    )
    runner = CliRunner()

    quick = runner.invoke(cli.app, ["quick", "base", "ft"])
    report = tmp_path / "cli-result.json"
    run = runner.invoke(
        cli.app,
        [
            "run",
            "base",
            "ft",
            "--profile",
            "classification",
            "--num-samples",
            "1",
            "--report",
            str(report),
            "--format",
            "json",
        ],
    )
    compared = runner.invoke(
        cli.app,
        ["compare", "base", "ft-one", "ft-two", "--num-samples", "1"],
    )
    invalid = runner.invoke(cli.app, ["run", "base", "ft", "--num-samples", "0"])
    version = runner.invoke(cli.app, ["version"])

    assert quick.exit_code == 0
    assert run.exit_code == 0
    assert report.is_file()
    assert compared.exit_code == 0
    assert "Comparison Summary" in compared.stdout
    assert invalid.exit_code == 1
    assert version.stdout.strip() == "finetunecheck 2.0.2"
    with pytest.raises(typer.BadParameter):
        cli._resolve_device("invalid")


def test_cli_generate_report_helper_supports_all_formats_and_unknown_format(
    sample_eval_results,
    tmp_path: Path,
) -> None:
    from finetunecheck import cli

    for output_format, suffix in (
        ("json", "json"),
        ("csv", "csv"),
        ("markdown", "md"),
    ):
        path = cli._generate_report(
            sample_eval_results,
            str(tmp_path / f"report.{suffix}"),
            output_format,
        )
        assert Path(path).is_file()
    with pytest.raises(typer.Exit):
        cli._generate_report(sample_eval_results, str(tmp_path / "x"), "unknown")
