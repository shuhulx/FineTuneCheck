"""Rich-based terminal formatting utilities."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from finetunecheck.models import CategoryScore, Verdict

console = Console()

_VERDICT_STYLES = {
    Verdict.EXCELLENT: ("bold green", "EXCELLENT"),
    Verdict.GOOD: ("bold cyan", "GOOD"),
    Verdict.GOOD_WITH_CONCERNS: ("bold yellow", "GOOD WITH CONCERNS"),
    Verdict.POOR: ("bold red", "POOR"),
    Verdict.HARMFUL: ("bold white on red", "HARMFUL"),
    Verdict.INSUFFICIENT_EVIDENCE: ("bold white on grey50", "INSUFFICIENT EVIDENCE"),
}


def print_verdict(verdict: Verdict, roi_score: float | None, summary: str) -> None:
    """Print a colored verdict banner."""
    style, label = _VERDICT_STYLES.get(verdict, ("bold white", str(verdict)))
    roi_text = f"{roi_score:.1f}/100" if roi_score is not None else "unavailable"
    title = Text(f"  Verdict: {label}  |  ROI Score: {roi_text}  ", style=style)
    panel = Panel(
        summary,
        title=title,
        border_style=style.split()[-1],
        padding=(1, 2),
    )
    console.print(panel)


def print_category_scores(
    base_scores: dict[str, CategoryScore],
    ft_scores: dict[str, CategoryScore],
) -> None:
    """Print a side-by-side comparison table of base vs fine-tuned scores."""
    table = Table(title="Category Scores: Base vs Fine-Tuned", show_lines=True)
    table.add_column("Category", style="bold", min_width=20)
    table.add_column("Base", justify="center", min_width=10)
    table.add_column("Fine-Tuned", justify="center", min_width=10)
    table.add_column("Delta", justify="center", min_width=10)
    table.add_column("Samples", justify="center", min_width=8)

    all_categories = sorted(set(base_scores) | set(ft_scores))
    for cat in all_categories:
        base = base_scores.get(cat)
        ft = ft_scores.get(cat)
        base_val = base.mean_score if base else None
        ft_val = ft.mean_score if ft else None
        delta = ft_val - base_val if base_val is not None and ft_val is not None else None
        n_samples = max(base.num_samples if base else 0, ft.num_samples if ft else 0)

        if delta is not None and delta > 0.05:
            delta_style = "green"
        elif delta is not None and delta < -0.05:
            delta_style = "red"
        else:
            delta_style = "white"

        delta_str = (
            f"[{delta_style}]{delta:+.3f}[/{delta_style}]" if delta is not None else "unavailable"
        )
        table.add_row(
            cat,
            f"{base_val:.3f}" if base_val is not None else base.status.value if base else "missing",
            f"{ft_val:.3f}" if ft_val is not None else ft.status.value if ft else "missing",
            delta_str,
            str(n_samples),
        )

    console.print(table)


def print_concerns(concerns: list[str]) -> None:
    """Print a list of concerns as warnings."""
    if not concerns:
        return
    console.print()
    console.print("[bold yellow]Concerns:[/bold yellow]")
    for concern in concerns:
        console.print(f"  [yellow]![/yellow] {concern}")
    console.print()


def print_recommendations(recommendations: list[str]) -> None:
    """Print a list of recommendations."""
    if not recommendations:
        return
    console.print("[bold cyan]Recommendations:[/bold cyan]")
    for rec in recommendations:
        console.print(f"  [cyan]>[/cyan] {rec}")
    console.print()


def print_progress(message: str, current: int, total: int) -> None:
    """Print a simple progress indicator."""
    pct = (current / total * 100) if total > 0 else 0
    bar_width = 30
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)
    console.print(f"  [{bar}] {current}/{total} ({pct:.0f}%) {message}", end="\r")
    if current >= total:
        console.print()
