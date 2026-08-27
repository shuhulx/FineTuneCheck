"""Async MCP handlers backed by a bounded worker pool."""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from finetunecheck.config import EvalConfig, JudgeConfig, QuickConfig

_T = TypeVar("_T")
_MODEL_WORKERS = asyncio.Semaphore(2)


async def _run_blocking(function: Callable[[], _T]) -> _T:
    async with _MODEL_WORKERS:
        return await asyncio.to_thread(function)


def _judge(arguments: dict[str, Any]) -> JudgeConfig | None:
    payload = arguments.get("judge")
    return JudgeConfig.model_validate(payload) if payload is not None else None


def _targets(arguments: dict[str, Any]) -> list[str]:
    if arguments.get("target_tasks"):
        return list(arguments["target_tasks"])
    if arguments.get("target_task"):
        return [arguments["target_task"]]
    return []


def _paired_config(arguments: dict[str, Any], **updates: Any) -> EvalConfig:
    payload: dict[str, Any] = {
        "base_model": arguments["base_model"],
        "finetuned_model": arguments["finetuned_model"],
        "target_tasks": _targets(arguments),
        "num_samples": arguments.get("num_samples", 100),
        "device": arguments.get("device") or "auto",
        "judge": _judge(arguments),
        **updates,
    }
    return EvalConfig(**payload)


def _results_summary(results) -> str:
    roi = f"{results.roi_score:.0f}/100" if results.roi_score is not None else "unavailable"
    lines = [
        f"Verdict: {results.verdict.value} (ROI: {roi}, coverage: {results.roi_coverage:.0%})",
        f"Base model: {results.base_model}",
        f"Fine-tuned model: {results.finetuned_model}",
        f"Target tasks: {', '.join(results.target_tasks) if results.target_tasks else 'none'}",
        "",
        results.summary,
        "",
        "Category evidence:",
    ]
    for category in sorted(set(results.base_scores) | set(results.ft_scores)):
        base = results.base_scores.get(category)
        ft = results.ft_scores.get(category)
        base_text = (
            f"{base.mean_score:.3f}"
            if base and base.mean_score is not None
            else base.status.value
            if base
            else "MISSING"
        )
        ft_text = (
            f"{ft.mean_score:.3f}"
            if ft and ft.mean_score is not None
            else ft.status.value
            if ft
            else "MISSING"
        )
        lines.append(f"  {category}: {base_text} -> {ft_text}")
    if results.concerns:
        lines.extend(["", "Concerns:", *[f"  - {item}" for item in results.concerns]])
    lines.extend(
        [
            "",
            "FineTuneCheck results support investigation and are not independent deployment approval.",
        ]
    )
    return "\n".join(lines)


async def handle_evaluate_finetune(arguments: dict[str, Any]) -> str:
    from finetunecheck.eval.runner import EvalRunner
    from finetunecheck.profiles.loader import ProfileLoader

    config = _paired_config(
        arguments,
        deep_analysis=arguments.get("deep_analysis", False),
        deep_analysis_samples=arguments.get("deep_analysis_samples", 50),
    )
    if arguments.get("profile"):
        config = ProfileLoader.apply_to_config(arguments["profile"], config)
    results = await _run_blocking(lambda: EvalRunner(config).run())
    return _results_summary(results)


async def handle_quick_check(arguments: dict[str, Any]) -> str:
    from finetunecheck.eval.runner import EvalRunner

    config = QuickConfig(
        base_model=arguments["base_model"],
        finetuned_model=arguments["finetuned_model"],
        target_tasks=arguments.get("target_tasks", []),
        device=arguments.get("device") or "auto",
    )
    return _results_summary(await _run_blocking(lambda: EvalRunner(config).run()))


async def handle_detect_forgetting(arguments: dict[str, Any]) -> str:
    from finetunecheck.eval.runner import EvalRunner

    results = await _run_blocking(lambda: EvalRunner(_paired_config(arguments)).run())
    report = results.forgetting
    if report is None:
        raise ValueError("Forgetting report was not produced")
    lines = [
        f"Status: {report.status.value}",
        f"Pattern: {report.pattern.value}",
        f"Backward transfer: {report.backward_transfer if report.backward_transfer is not None else 'unavailable'}",
        f"Selective forgetting index: {report.selective_forgetting_index if report.selective_forgetting_index is not None else 'unavailable'}",
        "Capability retention:",
    ]
    lines.extend(
        f"  {category}: {rate:.1%}" if rate is not None else f"  {category}: unavailable"
        for category, rate in sorted(report.capability_retention_rates.items())
    )
    return "\n".join(lines)


async def handle_compare_runs(arguments: dict[str, Any]) -> str:
    from finetunecheck.compare.multi_run import MultiRunComparator

    config = EvalConfig(
        base_model=arguments["base_model"],
        finetuned_model="placeholder",
        target_tasks=_targets(arguments),
        num_samples=arguments.get("num_samples", 100),
        device=arguments.get("device") or "auto",
        judge=_judge(arguments),
    )
    result = await _run_blocking(
        lambda: MultiRunComparator().compare(
            arguments["base_model"], arguments["finetuned_models"], config
        )
    )
    lines = [
        f"Compared {len(result.runs)} compatible runs",
        f"Best overall: {result.best_run}",
        f"Best target performance: {result.best_target_perf}",
        f"Least forgetting: {result.least_forgetting}",
        f"Pareto frontier: {', '.join(result.pareto_frontier)}",
        "",
        result.recommendation,
    ]
    return "\n".join(lines)


async def handle_get_verdict(arguments: dict[str, Any]) -> str:
    return await handle_quick_check(arguments)


async def handle_suggest_fixes(arguments: dict[str, Any]) -> str:
    from finetunecheck.eval.runner import EvalRunner

    results = await _run_blocking(lambda: EvalRunner(_paired_config(arguments)).run())
    lines = [f"Evidence verdict: {results.verdict.value}"]
    lines.extend(f"{index}. {item}" for index, item in enumerate(results.recommendations, 1))
    if not results.recommendations:
        lines.append("No recommendation was generated; inspect evidence coverage before acting.")
    return "\n".join(lines)


def _validated_output_path(arguments: dict[str, Any]) -> Path:
    output = Path(os.path.abspath(arguments["output_path"]))
    cwd = Path.cwd().resolve()
    try:
        output.relative_to(cwd)
        output.parent.resolve().relative_to(cwd)
    except ValueError as exc:
        raise ValueError("output_path must be within the working directory") from exc
    exists = output.exists() or output.is_symlink()
    if exists and not arguments.get("overwrite", False):
        raise ValueError(f"Refusing to overwrite existing report: {output}")
    if output.is_dir() and not output.is_symlink():
        raise ValueError("output_path must be a file, not a directory")
    return output


@dataclass
class _SecureOutputTarget:
    path: Path
    parent_fd: int
    filename: str
    overwrite: bool

    def close(self) -> None:
        os.close(self.parent_fd)

    def parent_is_current(self) -> bool:
        try:
            path_stat = os.stat(self.path.parent, follow_symlinks=False)
            fd_stat = os.fstat(self.parent_fd)
        except OSError:
            return False
        return (
            stat.S_ISDIR(path_stat.st_mode)
            and path_stat.st_dev == fd_stat.st_dev
            and path_stat.st_ino == fd_stat.st_ino
        )


def _open_secure_output_target(arguments: dict[str, Any]) -> _SecureOutputTarget:
    """Open every output-directory component without following symlinks."""
    output = _validated_output_path(arguments)
    cwd = Path.cwd().resolve()
    relative = output.relative_to(cwd)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(cwd, directory_flags)
    try:
        for component in relative.parent.parts:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=parent_fd)
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd

        try:
            existing = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not arguments.get("overwrite", False):
                raise ValueError(f"Refusing to overwrite existing report: {output}")
            if stat.S_ISDIR(existing.st_mode):
                raise ValueError("output_path must be a file, not a directory")
        return _SecureOutputTarget(
            path=output,
            parent_fd=parent_fd,
            filename=relative.name,
            overwrite=bool(arguments.get("overwrite", False)),
        )
    except Exception:
        os.close(parent_fd)
        raise


def _commit_secure_output(target: _SecureOutputTarget, payload: bytes) -> None:
    """Atomically commit bytes relative to the already-open trusted directory."""
    if not target.parent_is_current():
        raise ValueError("output_path parent changed during evaluation")
    temporary_name = f".finetunecheck-{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary_name, flags, 0o600, dir_fd=target.parent_fd)
    temporary_exists = True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if not target.parent_is_current():
            raise ValueError("output_path parent changed during evaluation")
        if target.overwrite:
            os.replace(
                temporary_name,
                target.filename,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
            )
            temporary_exists = False
        else:
            os.link(
                temporary_name,
                target.filename,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=target.parent_fd)
            temporary_exists = False
        os.fsync(target.parent_fd)
    except FileExistsError as exc:
        raise ValueError(f"Refusing to overwrite existing report: {target.path}") from exc
    except IsADirectoryError as exc:
        raise ValueError("output_path must be a file, not a directory") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=target.parent_fd)


async def handle_generate_report(arguments: dict[str, Any]) -> str:
    # Validate path before expensive model evaluation.
    target = _open_secure_output_target(arguments)
    try:
        from finetunecheck.eval.runner import EvalRunner

        results = await _run_blocking(lambda: EvalRunner(_paired_config(arguments)).run())
        output_format = arguments.get("format", "html")

        def export() -> str:
            with tempfile.TemporaryDirectory(prefix="finetunecheck-report-") as staging:
                suffix = ".md" if output_format == "markdown" else f".{output_format}"
                staged_output = Path(staging) / f"report{suffix}"
                if output_format == "html":
                    from finetunecheck.report.generator import ReportGenerator

                    generated = ReportGenerator().generate(results, str(staged_output))
                else:
                    from finetunecheck.report.exporters import (
                        CSVExporter,
                        JSONExporter,
                        MarkdownExporter,
                    )

                    exporter = {
                        "json": JSONExporter,
                        "csv": CSVExporter,
                        "markdown": MarkdownExporter,
                    }[output_format]
                    generated = exporter.export(results, str(staged_output))
                _commit_secure_output(target, Path(generated).read_bytes())
            return str(target.path)

        return f"Report generated at: {await _run_blocking(export)}"
    finally:
        target.close()


async def handle_list_profiles(arguments: dict[str, Any]) -> str:
    del arguments
    from finetunecheck.profiles.loader import ProfileLoader

    return "\n".join(
        [
            "Available evaluation profiles:",
            *[f"  {name}: {ProfileLoader.get(name).description}" for name in ProfileLoader.list()],
        ]
    )


async def handle_run_probe(arguments: dict[str, Any]) -> str:
    from finetunecheck.eval.runner import EvalRunner
    from finetunecheck.probes.registry import ProbeRegistry

    probe = ProbeRegistry.get(arguments["probe_name"])
    config = EvalConfig(
        base_model=arguments["model"],
        finetuned_model=arguments["model"],
        general_probes=[probe.name],
        num_samples=arguments.get("num_samples", len(probe.samples)),
        device=arguments.get("device") or "auto",
        judge=_judge(arguments),
        cache_baseline=False,
    )
    scores = await _run_blocking(lambda: EvalRunner(config).run_single_model())
    score = scores[probe.name]
    value = f"{score.mean_score:.3f}" if score.mean_score is not None else "unavailable"
    return (
        f"Probe: {probe.name}\nStatus: {score.status.value}\n"
        f"Measured samples: {score.num_samples}/{score.expected_samples}\nMean score: {value}"
    )


TOOL_HANDLERS = {
    "evaluate_finetune": handle_evaluate_finetune,
    "quick_check": handle_quick_check,
    "detect_forgetting": handle_detect_forgetting,
    "compare_runs": handle_compare_runs,
    "get_verdict": handle_get_verdict,
    "suggest_fixes": handle_suggest_fixes,
    "generate_report": handle_generate_report,
    "list_profiles": handle_list_profiles,
    "run_probe": handle_run_probe,
}
