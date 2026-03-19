"""FineTuneCheck CLI — evaluate fine-tuning outcomes from the command line."""

from __future__ import annotations

import typer
from rich.console import Console


def _version_callback(value: bool) -> None:
    if value:
        from finetunecheck._version import __version__

        typer.echo(f"finetunecheck {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="ftcheck",
    help="FineTuneCheck -- Evaluate fine-tuning outcomes with forgetting detection.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """FineTuneCheck -- Evaluate fine-tuning outcomes with forgetting detection."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> str:
    from finetunecheck.utils.device import detect_device

    return detect_device(device)


def _generate_report(results, output: str, fmt: str) -> str:
    """Write report in the requested format and return the path."""
    if fmt == "html":
        from finetunecheck.report.generator import ReportGenerator

        return ReportGenerator().generate(results, output)
    elif fmt == "json":
        from finetunecheck.report.exporters import JSONExporter

        return JSONExporter.export(results, output)
    elif fmt == "csv":
        from finetunecheck.report.exporters import CSVExporter

        return CSVExporter.export(results, output)
    elif fmt == "markdown":
        from finetunecheck.report.exporters import MarkdownExporter

        return MarkdownExporter.export(results, output)
    else:
        console.print(f"[red]Unknown format: {fmt}[/red]")
        raise typer.Exit(1)


def _print_results(results) -> None:
    """Pretty-print results to the terminal using rich."""
    from finetunecheck.utils.formatting import (
        print_category_scores,
        print_concerns,
        print_recommendations,
        print_verdict,
    )

    console.print()
    print_verdict(results.verdict, results.roi_score, results.summary)
    console.print()
    print_category_scores(results.base_scores, results.ft_scores)

    if results.forgetting:
        f = results.forgetting
        console.print()
        console.print(f"[bold]Forgetting pattern:[/bold] {f.pattern.value}")
        console.print(f"[bold]Backward transfer:[/bold] {f.backward_transfer:+.3f}")
        if f.most_affected:
            console.print(f"[bold]Most affected:[/bold] [red]{', '.join(f.most_affected)}[/red]")
        if f.resilient:
            console.print(f"[bold]Resilient:[/bold] [green]{', '.join(f.resilient)}[/green]")

    print_concerns(results.concerns)
    print_recommendations(results.recommendations)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    base_model: str = typer.Argument(..., help="Base model path or HuggingFace ID"),
    finetuned_model: str = typer.Argument(..., help="Fine-tuned model path or HuggingFace ID"),
    target_task: str | None = typer.Option(None, "--target-task", "-t", help="Target task name"),
    profile: str | None = typer.Option(
        None,
        "--profile",
        "-p",
        help="Evaluation profile (e.g. code, chat, safety)",
    ),
    num_samples: int = typer.Option(100, "--num-samples", "-n", help="Samples per probe"),
    deep: bool = typer.Option(False, "--deep", help="Enable deep analysis (CKA, spectral, etc.)"),
    report: str | None = typer.Option(None, "--report", "-r", help="Output report path"),
    output_format: str = typer.Option(
        "html", "--format", "-f", help="Report format: html, json, csv, markdown"
    ),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, mps"),
    judge: str = typer.Option("auto", "--judge", help="Judge model or 'auto'"),
    exit_code: bool = typer.Option(
        False,
        "--exit-code",
        help="Return non-zero exit code for POOR/HARMFUL verdicts",
    ),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Cache baseline results"),
) -> None:
    """Run comprehensive fine-tuning evaluation."""
    if num_samples <= 0:
        console.print("[red]Error: --num-samples must be greater than 0.[/red]")
        raise typer.Exit(1)

    from finetunecheck.config import EvalConfig
    from finetunecheck.eval.runner import EvalRunner
    from finetunecheck.models import Verdict

    config = EvalConfig(
        base_model=base_model,
        finetuned_model=finetuned_model,
        target_task=target_task,
        profile=profile,
        num_samples=num_samples,
        deep_analysis=deep,
        device=_resolve_device(device),
        judge_model=judge,
        cache_baseline=cache,
        output_format=output_format,
    )

    if profile:
        from finetunecheck.profiles.loader import ProfileLoader

        config = ProfileLoader.apply_to_config(profile, config)

    with console.status("[bold cyan]Running evaluation...[/bold cyan]", spinner="dots"):
        runner = EvalRunner(config)
        results = runner.run()

    _print_results(results)

    if report:
        path = _generate_report(results, report, output_format)
        console.print(f"\n[bold green]Report saved:[/bold green] {path}")

    if exit_code and results.verdict in (Verdict.POOR, Verdict.HARMFUL):
        raise typer.Exit(1)


@app.command()
def quick(
    base_model: str = typer.Argument(..., help="Base model path or HuggingFace ID"),
    finetuned_model: str = typer.Argument(..., help="Fine-tuned model path or HuggingFace ID"),
    report: str | None = typer.Option(None, "--report", "-r", help="Output report path"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, mps"),
) -> None:
    """Quick 5-minute evaluation on core capabilities (20 samples, 4 categories)."""
    from finetunecheck.config import QuickConfig
    from finetunecheck.eval.runner import EvalRunner

    config = QuickConfig(
        base_model=base_model,
        finetuned_model=finetuned_model,
        device=_resolve_device(device),
    )

    with console.status("[bold cyan]Running quick evaluation...[/bold cyan]", spinner="dots"):
        runner = EvalRunner(config)
        results = runner.run()

    _print_results(results)

    if report:
        path = _generate_report(results, report, "html")
        console.print(f"\n[bold green]Report saved:[/bold green] {path}")


@app.command()
def compare(
    base_model: str = typer.Argument(..., help="Base model path or HuggingFace ID"),
    finetuned_models: list[str] = typer.Argument(
        ..., help="Fine-tuned model paths (space-separated)"
    ),
    target_task: str | None = typer.Option(None, "--target-task", "-t", help="Target task name"),
    report: str | None = typer.Option(None, "--report", "-r", help="Output report path"),
    device: str = typer.Option("auto", "--device", help="Device"),
    num_samples: int = typer.Option(100, "--num-samples", "-n", help="Samples per probe"),
) -> None:
    """Compare multiple fine-tuning runs against the same base model."""
    if num_samples <= 0:
        console.print("[red]Error: --num-samples must be greater than 0.[/red]")
        raise typer.Exit(1)

    from finetunecheck.compare.multi_run import MultiRunComparator
    from finetunecheck.config import EvalConfig

    ft_map = {f"run_{i}": path for i, path in enumerate(finetuned_models)}

    config = EvalConfig(
        base_model=base_model,
        finetuned_model="",  # overridden per run
        target_task=target_task,
        num_samples=num_samples,
        device=_resolve_device(device),
    )

    with console.status(f"[bold cyan]Comparing {len(ft_map)} runs...[/bold cyan]", spinner="dots"):
        comparator = MultiRunComparator()
        result = comparator.compare(base_model, ft_map, config)

    # Print results for each run
    for name, res in result.runs.items():
        console.print(f"\n[bold]--- {name}: {res.finetuned_model} ---[/bold]")
        _print_results(res)

    # Print comparison summary
    console.rule("[bold]Comparison Summary[/bold]")
    console.print(f"[bold]Best overall (ROI):[/bold]      {result.best_run}")
    console.print(f"[bold]Best target perf:[/bold]        {result.best_target_perf}")
    console.print(f"[bold]Least forgetting:[/bold]        {result.least_forgetting}")
    console.print(f"[bold]Pareto frontier:[/bold]         {', '.join(result.pareto_frontier)}")
    console.print()
    console.print(f"[italic]{result.recommendation}[/italic]")

    if report:
        best = result.runs[result.best_run]
        path = _generate_report(best, report, "html")
        console.print(f"\n[bold green]Report saved:[/bold green] {path}")


@app.command(name="serve")
def serve(
    stdio: bool = typer.Option(True, "--stdio", help="Use stdio transport (default)"),
) -> None:
    """Start MCP server for AI assistant integration."""
    import asyncio

    from finetunecheck.mcp.server import main as mcp_main

    asyncio.run(mcp_main())


@app.command(name="list-profiles")
def list_profiles() -> None:
    """List available evaluation profiles."""
    from rich.table import Table

    from finetunecheck.profiles.loader import ProfileLoader

    names = ProfileLoader.list()

    if not names:
        console.print("[yellow]No profiles found.[/yellow]")
        return

    table = Table(title="Available Profiles", show_lines=True)
    table.add_column("Profile", style="bold cyan", min_width=15)
    table.add_column("Description", min_width=50)

    for name in names:
        profile = ProfileLoader.get(name)
        table.add_row(name, profile.description)

    console.print(table)


@app.command(name="list-probes")
def list_probes() -> None:
    """List available probe sets."""
    from rich.table import Table

    from finetunecheck.probes.registry import ProbeRegistry

    names = ProbeRegistry.list()

    if not names:
        console.print("[yellow]No probe sets found.[/yellow]")
        console.print(
            "Built-in probes will be available once JSON files are placed in probes/builtin/."
        )
        return

    table = Table(title="Available Probe Sets", show_lines=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Category")
    table.add_column("Judge")
    table.add_column("Samples", justify="right")

    for name in names:
        probe = ProbeRegistry.get(name)
        table.add_row(
            probe.name,
            probe.category or "-",
            probe.judge_type.value,
            str(len(probe.samples)),
        )

    console.print(table)


@app.command()
def version() -> None:
    """Print version."""
    from finetunecheck._version import __version__

    typer.echo(f"finetunecheck {__version__}")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
