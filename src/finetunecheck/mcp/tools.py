"""MCP tool handler implementations for FineTuneCheck."""

from __future__ import annotations

import os
from typing import Any

from finetunecheck.utils.device import detect_device


def _results_summary(results) -> str:
    """Build a concise textual summary of EvalResults."""
    lines = [
        f"Verdict: {results.verdict.value} (ROI: {results.roi_score:.0f}/100)",
        f"Base model: {results.base_model}",
        f"Fine-tuned model: {results.finetuned_model}",
        f"Target improvement: {results.target_improvement:+.1%}",
        "",
        results.summary,
    ]

    if results.forgetting:
        f = results.forgetting
        lines.append("")
        lines.append(f"Forgetting pattern: {f.pattern.value}")
        lines.append(f"Backward transfer: {f.backward_transfer:+.3f}")
        lines.append(f"Selective forgetting index: {f.selective_forgetting_index:.3f}")
        if f.safety_alignment_retention is not None:
            lines.append(f"Safety retention: {f.safety_alignment_retention:.1%}")
        if f.most_affected:
            lines.append(f"Most affected: {', '.join(f.most_affected)}")

    lines.append("")
    lines.append("Category scores:")
    for cat in sorted(results.base_scores.keys()):
        bs = results.base_scores[cat].mean_score
        fs = results.ft_scores[cat].mean_score
        delta = fs - bs
        lines.append(f"  {cat}: {bs:.3f} -> {fs:.3f} ({delta:+.3f})")

    if results.concerns:
        lines.append("")
        lines.append("Concerns:")
        for c in results.concerns:
            lines.append(f"  - {c}")

    if results.recommendations:
        lines.append("")
        lines.append("Recommendations:")
        for r in results.recommendations:
            lines.append(f"  - {r}")

    return "\n".join(lines)


async def handle_evaluate_finetune(arguments: dict[str, Any]) -> str:
    """Run comprehensive evaluation."""
    from finetunecheck.config import EvalConfig
    from finetunecheck.eval.runner import EvalRunner

    config = EvalConfig(
        base_model=arguments["base_model"],
        finetuned_model=arguments["finetuned_model"],
        target_task=arguments.get("target_task"),
        profile=arguments.get("profile"),
        num_samples=arguments.get("num_samples", 100),
        deep_analysis=arguments.get("deep_analysis", False),
        device=detect_device(arguments.get("device") or "auto"),
    )

    runner = EvalRunner(config)
    results = runner.run()
    return _results_summary(results)


async def handle_quick_check(arguments: dict[str, Any]) -> str:
    """Fast 5-minute evaluation."""
    from finetunecheck.config import QuickConfig
    from finetunecheck.eval.runner import EvalRunner

    config = QuickConfig(
        base_model=arguments["base_model"],
        finetuned_model=arguments["finetuned_model"],
        device=detect_device(arguments.get("device") or "auto"),
    )

    runner = EvalRunner(config)
    results = runner.run()
    return _results_summary(results)


async def handle_detect_forgetting(arguments: dict[str, Any]) -> str:
    """Analyze catastrophic forgetting specifically."""
    from finetunecheck.config import EvalConfig
    from finetunecheck.eval.runner import EvalRunner

    config = EvalConfig(
        base_model=arguments["base_model"],
        finetuned_model=arguments["finetuned_model"],
        num_samples=arguments.get("num_samples", 100),
        device=detect_device(arguments.get("device") or "auto"),
    )

    runner = EvalRunner(config)
    results = runner.run()

    if not results.forgetting:
        return "No forgetting analysis available. The evaluation may have failed to complete."

    f = results.forgetting
    lines = [
        f"Forgetting Analysis for: {results.finetuned_model}",
        f"Pattern: {f.pattern.value}",
        f"Backward transfer: {f.backward_transfer:+.3f}",
        f"Selective forgetting index: {f.selective_forgetting_index:.3f}",
    ]
    if f.safety_alignment_retention is not None:
        lines.append(f"Safety alignment retention: {f.safety_alignment_retention:.1%}")

    lines.append("")
    lines.append("Capability retention rates:")
    for cat, rate in sorted(f.capability_retention_rates.items(), key=lambda x: x[1]):
        status = "OK" if rate >= 0.95 else "WARN" if rate >= 0.85 else "CRITICAL"
        lines.append(f"  {cat}: {rate:.1%} [{status}]")

    if f.most_affected:
        lines.append(f"\nMost affected categories: {', '.join(f.most_affected)}")
    if f.resilient:
        lines.append(f"Resilient categories: {', '.join(f.resilient)}")

    if f.regressions:
        lines.append(f"\nTop regressions ({len(f.regressions)} total):")
        for reg in f.regressions[:5]:
            lines.append(
                f"  [{reg.category}] {reg.sample_id}: "
                f"{reg.base_score:.3f} -> {reg.ft_score:.3f} ({reg.score_change:+.3f})"
            )

    return "\n".join(lines)


async def handle_compare_runs(arguments: dict[str, Any]) -> str:
    """Compare multiple fine-tuning runs."""
    from finetunecheck.compare.multi_run import MultiRunComparator
    from finetunecheck.config import EvalConfig

    config = EvalConfig(
        base_model=arguments["base_model"],
        finetuned_model="",
        target_task=arguments.get("target_task"),
        num_samples=arguments.get("num_samples", 100),
        device=detect_device(arguments.get("device") or "auto"),
    )

    comparator = MultiRunComparator()
    result = comparator.compare(
        arguments["base_model"],
        arguments["finetuned_models"],
        config,
    )

    lines = [
        f"Comparison of {len(result.runs)} fine-tuning runs",
        f"Base model: {result.base_model}",
        "",
        f"Best overall (ROI): {result.best_run}",
        f"Best target performance: {result.best_target_perf}",
        f"Least forgetting: {result.least_forgetting}",
        f"Pareto frontier: {', '.join(result.pareto_frontier)}",
        "",
    ]

    for name, res in result.runs.items():
        bwt = res.forgetting.backward_transfer if res.forgetting else 0.0
        lines.append(
            f"  {name}: verdict={res.verdict.value}, "
            f"ROI={res.roi_score:.0f}, "
            f"target_imp={res.target_improvement:+.1%}, "
            f"BWT={bwt:+.3f}"
        )

    lines.append("")
    lines.append(result.recommendation)
    return "\n".join(lines)


async def handle_get_verdict(arguments: dict[str, Any]) -> str:
    """Get deployment readiness assessment."""
    from finetunecheck.config import QuickConfig
    from finetunecheck.eval.runner import EvalRunner

    config = QuickConfig(
        base_model=arguments["base_model"],
        finetuned_model=arguments["finetuned_model"],
        device=detect_device(arguments.get("device") or "auto"),
    )

    runner = EvalRunner(config)
    results = runner.run()

    verdict = results.verdict.value.replace("_", " ")
    lines = [
        f"VERDICT: {verdict}",
        f"ROI Score: {results.roi_score:.0f}/100",
        f"Target improvement: {results.target_improvement:+.1%}",
        "",
        results.summary,
    ]
    return "\n".join(lines)


async def handle_suggest_fixes(arguments: dict[str, Any]) -> str:
    """Get recommendations for improving fine-tuning outcome."""
    from finetunecheck.config import EvalConfig
    from finetunecheck.eval.runner import EvalRunner

    config = EvalConfig(
        base_model=arguments["base_model"],
        finetuned_model=arguments["finetuned_model"],
        device=detect_device(arguments.get("device") or "auto"),
    )

    runner = EvalRunner(config)
    results = runner.run()

    if not results.recommendations:
        return "No specific recommendations. The fine-tuning outcome looks good."

    lines = [f"Recommendations for {results.finetuned_model}:", ""]
    for i, rec in enumerate(results.recommendations, 1):
        lines.append(f"{i}. {rec}")

    if results.concerns:
        lines.append("")
        lines.append("Concerns to address:")
        for c in results.concerns:
            lines.append(f"  - {c}")

    return "\n".join(lines)


async def handle_generate_report(arguments: dict[str, Any]) -> str:
    """Create diagnostic report."""
    from finetunecheck.config import EvalConfig
    from finetunecheck.eval.runner import EvalRunner

    config = EvalConfig(
        base_model=arguments["base_model"],
        finetuned_model=arguments["finetuned_model"],
        device=detect_device(arguments.get("device") or "auto"),
    )

    runner = EvalRunner(config)
    results = runner.run()

    output_path = arguments["output_path"]

    resolved = os.path.realpath(output_path)
    cwd = os.path.realpath(os.getcwd())
    if not resolved.startswith(cwd + os.sep) and resolved != cwd:
        return "Error: output_path must be within the working directory."

    fmt = arguments.get("format", "html")

    if fmt == "html":
        from finetunecheck.report.generator import ReportGenerator

        path = ReportGenerator().generate(results, output_path)
    elif fmt == "json":
        from finetunecheck.report.exporters import JSONExporter

        path = JSONExporter.export(results, output_path)
    elif fmt == "csv":
        from finetunecheck.report.exporters import CSVExporter

        path = CSVExporter.export(results, output_path)
    elif fmt == "markdown":
        from finetunecheck.report.exporters import MarkdownExporter

        path = MarkdownExporter.export(results, output_path)
    else:
        return f"Unknown format: {fmt}. Supported: html, json, csv, markdown."

    return f"Report generated at: {path}"


async def handle_list_profiles(arguments: dict[str, Any]) -> str:
    """Show available evaluation profiles."""
    from finetunecheck.profiles.loader import ProfileLoader

    names = ProfileLoader.list()
    if not names:
        return "No evaluation profiles found."

    lines = ["Available evaluation profiles:", ""]
    for name in names:
        profile = ProfileLoader.get(name)
        lines.append(f"  {name}: {profile.description}")
    return "\n".join(lines)


async def handle_run_probe(arguments: dict[str, Any]) -> str:
    """Run a specific probe set on a model."""
    from finetunecheck.probes.registry import ProbeRegistry

    probe_name = arguments["probe_name"]
    try:
        probe = ProbeRegistry.get(probe_name)
    except KeyError:
        available = ProbeRegistry.list()
        return f"Probe '{probe_name}' not found. Available: {', '.join(available)}"

    # Run inference on the probe
    from finetunecheck.config import EvalConfig
    from finetunecheck.eval.runner import EvalRunner

    config = EvalConfig(
        base_model=arguments["model"],
        finetuned_model=arguments["model"],
        general_probes=[probe_name],
        num_samples=arguments.get("num_samples", len(probe.samples)),
        device=detect_device(arguments.get("device") or "auto"),
    )

    runner = EvalRunner(config)
    results = runner.run()

    lines = [
        f"Probe: {probe_name} ({probe.category})",
        f"Judge: {probe.judge_type.value}",
        f"Samples: {len(probe.samples)}",
        "",
    ]

    if probe_name in results.ft_scores:
        score = results.ft_scores[probe_name]
        lines.append(f"Mean score: {score.mean_score:.3f} (+/- {score.std_score:.3f})")
    else:
        lines.append("Score not available.")

    return "\n".join(lines)


# Dispatch table
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
