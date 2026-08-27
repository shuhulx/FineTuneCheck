"""Export EvalResults to JSON, CSV, and Markdown formats."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from finetunecheck._version import __version__
from finetunecheck.models import EvalResults


class JSONExporter:
    """Export results as structured JSON."""

    @staticmethod
    def export(results: EvalResults, output: str) -> str:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(results.model_dump_json(indent=2), encoding="utf-8")
        return str(out)


class CSVExporter:
    """Export category scores as a flat CSV table."""

    @staticmethod
    def export(results: EvalResults, output: str) -> str:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)

        buf = io.StringIO()
        writer = csv.writer(buf)

        forgetting = results.forgetting
        has_forgetting = forgetting is not None
        header = ["category", "base_status", "base_score", "ft_status", "ft_score", "change"]
        if has_forgetting:
            header.append("retention_rate")
        writer.writerow(header)

        for cat in sorted(set(results.base_scores) | set(results.ft_scores)):
            base_cat = results.base_scores.get(cat)
            bs = base_cat.mean_score if base_cat else None
            ft_cat = results.ft_scores.get(cat)
            fs = ft_cat.mean_score if ft_cat else None
            delta = fs - bs if fs is not None and bs is not None else None
            row = [
                cat,
                base_cat.status.value if base_cat else "MISSING",
                f"{bs:.4f}" if bs is not None else "",
                ft_cat.status.value if ft_cat else "MISSING",
                f"{fs:.4f}" if fs is not None else "",
                f"{delta:+.4f}" if delta is not None else "",
            ]
            if forgetting is not None:
                ret = forgetting.capability_retention_rates.get(cat)
                row.append(f"{ret:.4f}" if ret is not None else "")
            writer.writerow(row)

        out.write_text(buf.getvalue(), encoding="utf-8")
        return str(out)


class MarkdownExporter:
    """Export results as a Markdown report."""

    @staticmethod
    def export(results: EvalResults, output: str) -> str:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        verdict_label = results.verdict.value.replace("_", " ")

        lines.append("# FineTuneCheck Report")
        lines.append("")
        roi_text = (
            f"{results.roi_score:.0f}/100" if results.roi_score is not None else "unavailable"
        )
        lines.append(
            f"**Verdict:** {verdict_label} (ROI: {roi_text}; evidence coverage: {results.roi_coverage:.0%})"
        )
        lines.append("")
        lines.append(f"- **Base model:** `{results.base_model}`")
        lines.append(f"- **Fine-tuned model:** `{results.finetuned_model}`")
        if results.target_tasks:
            lines.append(f"- **Target tasks:** `{', '.join(results.target_tasks)}`")
        target_text = (
            f"{results.target_improvement:+.3f} absolute score"
            if results.target_improvement is not None
            else "unavailable"
        )
        lines.append(f"- **Target improvement:** {target_text}")
        lines.append("")

        # Summary
        if results.summary:
            lines.append("## Executive Summary")
            lines.append("")
            lines.append(results.summary)
            lines.append("")

        # Category scores table
        lines.append("## Category Scores")
        lines.append("")

        forgetting = results.forgetting
        has_forgetting = forgetting is not None
        if has_forgetting:
            lines.append("| Category | Base | Fine-tuned | Change | Retention |")
            lines.append("|----------|------|-----------|--------|-----------|")
        else:
            lines.append("| Category | Base | Fine-tuned | Change |")
            lines.append("|----------|------|-----------|--------|")

        for cat in sorted(set(results.base_scores) | set(results.ft_scores)):
            base_cat = results.base_scores.get(cat)
            bs = base_cat.mean_score if base_cat else None
            ft_cat = results.ft_scores.get(cat)
            fs = ft_cat.mean_score if ft_cat else None
            delta = fs - bs if fs is not None and bs is not None else None
            base_text = (
                f"{bs:.3f}" if bs is not None else base_cat.status.value if base_cat else "MISSING"
            )
            ft_text = (
                f"{fs:.3f}" if fs is not None else ft_cat.status.value if ft_cat else "MISSING"
            )
            delta_text = f"{delta:+.3f}" if delta is not None else "Unavailable"
            row = f"| {cat} | {base_text} | {ft_text} | {delta_text} |"
            if forgetting is not None:
                ret = forgetting.capability_retention_rates.get(cat)
                row += f" {ret:.1%} |" if ret is not None else " Unavailable |"
            lines.append(row)
        lines.append("")

        # Forgetting
        if results.forgetting:
            f = results.forgetting
            lines.append("## Forgetting Analysis")
            lines.append("")
            lines.append(f"- **Pattern:** {f.pattern.value}")
            bwt_text = (
                f"{f.backward_transfer:.3f}" if f.backward_transfer is not None else "unavailable"
            )
            sfi_text = (
                f"{f.selective_forgetting_index:.3f}"
                if f.selective_forgetting_index is not None
                else "unavailable"
            )
            lines.append(f"- **Backward transfer:** {bwt_text}")
            lines.append(f"- **Selective forgetting index:** {sfi_text}")
            if f.safety_alignment_retention is not None:
                lines.append(f"- **Safety retention:** {f.safety_alignment_retention:.1%}")
            if f.most_affected:
                lines.append(f"- **Most affected:** {', '.join(f.most_affected)}")
            if f.resilient:
                lines.append(f"- **Resilient:** {', '.join(f.resilient)}")
            lines.append("")

            # Regressions
            if f.regressions:
                lines.append("### Top Regressions")
                lines.append("")
                lines.append("| Category | Sample | Base | FT | Change |")
                lines.append("|----------|--------|------|-----|--------|")
                for reg in f.regressions[:10]:
                    prompt_short = reg.prompt[:50] + "..." if len(reg.prompt) > 50 else reg.prompt
                    lines.append(
                        f"| {reg.category} | {prompt_short} "
                        f"| {reg.base_score:.3f} | {reg.ft_score:.3f} "
                        f"| {reg.score_change:+.3f} |"
                    )
                lines.append("")

        # Deep analysis
        if results.deep_analysis:
            da = results.deep_analysis
            lines.append("## Deep Analysis")
            lines.append("")
            if da.perplexity:
                lines.append(
                    f"- **Perplexity shift:** KL={da.perplexity.kl_divergence:.4f}, "
                    f"base mean={da.perplexity.mean_ppl_base:.1f}, "
                    f"FT mean={da.perplexity.mean_ppl_ft:.1f}"
                )
            if da.cka:
                lines.append(f"- **Mean CKA similarity:** {da.cka.mean_cka:.3f}")
                if da.cka.most_diverged_layers:
                    lines.append(
                        f"  - Most diverged layers: {', '.join(da.cka.most_diverged_layers)}"
                    )
            if da.spectral:
                lines.append(f"- **Mean effective rank:** {da.spectral.mean_effective_rank:.1f}")
            if da.calibration:
                lines.append(
                    f"- **Calibration:** base ECE={da.calibration.base_ece:.4f}, "
                    f"FT ECE={da.calibration.ft_ece:.4f} "
                    f"(delta={da.calibration.ece_delta:+.4f})"
                )
            if da.activation:
                lines.append(
                    f"- **Activation drift:** mean={da.activation.mean_drift:.4f}, "
                    f"{len(da.activation.disrupted_heads)} disrupted heads"
                )
            lines.append("")

        # Concerns
        if results.concerns:
            lines.append("## Concerns")
            lines.append("")
            for c in results.concerns:
                lines.append(f"- {c}")
            lines.append("")

        # Recommendations
        if results.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for r in results.recommendations:
                lines.append(f"- {r}")
            lines.append("")

        lines.append("---")
        lines.append(
            f"*Generated by FineTuneCheck v{__version__}; diagnostic evidence, not deployment approval.*"
        )
        lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")
        return str(out)
