"""Generate self-contained HTML reports with embedded Plotly visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader

from finetunecheck._version import __version__
from finetunecheck.models import (
    ActivationDriftReport,
    CalibrationReport,
    CKAReport,
    EvalResults,
    PerplexityDistShift,
    SpectralReport,
    Verdict,
)

if TYPE_CHECKING:
    from finetunecheck.compare.multi_run import ComparisonResult

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_VERDICT_COLORS = {
    Verdict.EXCELLENT: "#10B981",
    Verdict.GOOD: "#10B981",
    Verdict.GOOD_WITH_CONCERNS: "#F59E0B",
    Verdict.POOR: "#EF4444",
    Verdict.HARMFUL: "#991B1B",
    Verdict.INSUFFICIENT_EVIDENCE: "#64748B",
}

_LAYOUT_DEFAULTS: dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", color="#334155"),
    margin=dict(l=40, r=40, t=40, b=40),
)


def _figure_json(figure: go.Figure) -> str:
    """Return Plotly JSON while defending against incomplete third-party stubs."""
    payload = figure.to_json()
    if not isinstance(payload, str):
        raise RuntimeError("Plotly did not produce a JSON document")
    return payload


class ReportGenerator:
    """Generate self-contained HTML reports with embedded Plotly visualizations."""

    def __init__(self, *, plotly_mode: str = "inline") -> None:
        if plotly_mode not in {"inline", "cdn"}:
            raise ValueError("plotly_mode must be 'inline' or 'cdn'")
        self.plotly_mode = plotly_mode
        self.env = Environment(
            loader=PackageLoader("finetunecheck", "report/templates"),
            autoescape=True,
        )

    def generate(
        self,
        results: EvalResults,
        output: str,
        title: str = "FineTuneCheck Report",
    ) -> str:
        """Generate full HTML report and write to *output* path.

        Returns the resolved output path.
        """
        template = self.env.get_template("base.html.j2")

        figures = {
            "radar": self._build_radar_chart(results),
            "retention_bars": self._build_retention_bars(results),
            "target_bars": self._build_target_bars(results),
            "roi_breakdown": self._build_roi_breakdown(results) if results.forgetting else None,
        }

        if results.deep_analysis:
            figures["deep_analysis"] = self._build_deep_analysis_figures(results)
        else:
            figures["deep_analysis"] = None

        figure_payloads = {
            key: (
                {nested_key: json.loads(nested_value) for nested_key, nested_value in value.items()}
                if isinstance(value, dict)
                else json.loads(value)
                if isinstance(value, str)
                else value
            )
            for key, value in figures.items()
        }
        if self.plotly_mode == "inline":
            from plotly.offline import get_plotlyjs

            plotly_inline = get_plotlyjs()
            plotly_src = None
        else:
            plotly_inline = None
            plotly_src = _PLOTLY_CDN

        html = template.render(
            title=title,
            results=results,
            figures=figure_payloads,
            plotly_inline=plotly_inline,
            plotly_src=plotly_src,
            version=__version__,
            verdict_color=_VERDICT_COLORS.get(results.verdict, "#6B7280"),
            verdict_label=results.verdict.value.replace("_", " "),
        )

        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return str(out)

    def generate_comparison(
        self,
        comparison: ComparisonResult,
        output: str,
        title: str = "FineTuneCheck Comparison Report",
    ) -> str:
        """Generate a genuine all-run comparison report."""
        template = self.env.get_template("comparison.html.j2")
        points = []
        for name, result in comparison.runs.items():
            points.append(
                {
                    "name": name,
                    "model": result.finetuned_model,
                    "target": result.target_improvement,
                    "bwt": (result.forgetting.backward_transfer if result.forgetting else None),
                    "roi": result.roi_score,
                    "coverage": result.roi_coverage,
                    "verdict": result.verdict.value,
                    "pareto": name in comparison.pareto_frontier,
                }
            )
        measured = [
            point for point in points if point["target"] is not None and point["bwt"] is not None
        ]
        figure: dict = {}
        if measured:
            figure_object = go.Figure()
            figure_object.add_trace(
                go.Scatter(
                    x=[point["target"] for point in measured],
                    y=[point["bwt"] for point in measured],
                    mode="markers+text",
                    text=[point["name"] for point in measured],
                    textposition="top center",
                    marker={
                        "size": 14,
                        "color": [
                            "#10B981" if point["pareto"] else "#94A3B8" for point in measured
                        ],
                    },
                    customdata=[
                        [point["verdict"], point["roi"], point["coverage"]] for point in measured
                    ],
                    hovertemplate=(
                        "%{text}<br>Target delta %{x:.3f}<br>BWT %{y:.3f}<br>"
                        "Verdict %{customdata[0]}<br>ROI %{customdata[1]}<br>"
                        "Coverage %{customdata[2]:.0%}<extra></extra>"
                    ),
                )
            )
            figure_object.update_layout(
                **_LAYOUT_DEFAULTS,
                title="Target Improvement vs Backward Transfer",
                xaxis_title="Target absolute delta (higher is better)",
                yaxis_title="Backward transfer (higher is better)",
                height=460,
            )
            figure = json.loads(_figure_json(figure_object))
        plotly_inline = None
        plotly_src = _PLOTLY_CDN
        if self.plotly_mode == "inline":
            from plotly.offline import get_plotlyjs

            plotly_inline = get_plotlyjs()
            plotly_src = None
        html = template.render(
            title=title,
            comparison=comparison,
            points=points,
            figure=figure,
            plotly_inline=plotly_inline,
            plotly_src=plotly_src,
            version=__version__,
        )
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return str(out)

    # ------------------------------------------------------------------
    # Primary charts
    # ------------------------------------------------------------------

    def _build_radar_chart(self, results: EvalResults) -> str:
        categories = self._measured_categories(results)
        if not categories:
            return "{}"

        base_vals = [results.base_scores[c].mean_score for c in categories]
        ft_vals = [results.ft_scores[c].mean_score for c in categories]

        # Close the polygon
        categories_closed = [*categories, categories[0]]
        base_closed = [*base_vals, base_vals[0]]
        ft_closed = [*ft_vals, ft_vals[0]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=base_closed,
                theta=categories_closed,
                name="Base Model",
                fill="toself",
                fillcolor="rgba(99, 102, 241, 0.15)",
                line=dict(color="#6366F1", width=2),
                marker=dict(size=5),
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=ft_closed,
                theta=categories_closed,
                name="Fine-tuned",
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.15)",
                line=dict(color="#10B981", width=2),
                marker=dict(size=5),
            )
        )
        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                    gridcolor="#E2E8F0",
                ),
                angularaxis=dict(gridcolor="#E2E8F0"),
                bgcolor="rgba(0,0,0,0)",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            showlegend=True,
            height=420,
        )
        return _figure_json(fig)

    def _build_retention_bars(self, results: EvalResults) -> str:
        if not results.forgetting:
            return "{}"

        retention = {
            category: value
            for category, value in results.forgetting.capability_retention_rates.items()
            if value is not None
        }
        cats = sorted(retention.keys(), key=lambda c: retention[c])
        if not cats:
            return "{}"
        values = [retention[c] for c in cats]
        colors = ["#10B981" if v >= 0.95 else "#F59E0B" if v >= 0.85 else "#EF4444" for v in values]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=cats,
                x=values,
                orientation="h",
                marker=dict(color=colors, cornerradius=4),
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
                textfont=dict(size=12, color="#334155"),
            )
        )
        fig.add_vline(x=0.95, line_dash="dash", line_color="#10B981", opacity=0.6)
        fig.add_vline(x=0.85, line_dash="dash", line_color="#F59E0B", opacity=0.6)
        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            xaxis=dict(
                range=[0, 1.12],
                title="Retention Rate",
                gridcolor="#F1F5F9",
                tickformat=".0%",
            ),
            yaxis=dict(title="", gridcolor="#F1F5F9"),
            height=max(260, len(cats) * 36 + 80),
        )
        return _figure_json(fig)

    def _build_target_bars(self, results: EvalResults) -> str:
        categories = self._measured_categories(results)
        if not categories:
            return "{}"

        base_vals = [results.base_scores[c].mean_score for c in categories]
        ft_vals = [results.ft_scores[c].mean_score for c in categories]
        base_err = [results.base_scores[c].std_score or 0.0 for c in categories]
        ft_err = [results.ft_scores[c].std_score or 0.0 for c in categories]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Base Model",
                x=categories,
                y=base_vals,
                marker=dict(color="#6366F1", cornerradius=4),
                error_y=dict(
                    type="data",
                    array=base_err,
                    visible=True,
                    color="#6366F1",
                    thickness=1.5,
                    width=4,
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                name="Fine-tuned",
                x=categories,
                y=ft_vals,
                marker=dict(color="#10B981", cornerradius=4),
                error_y=dict(
                    type="data", array=ft_err, visible=True, color="#10B981", thickness=1.5, width=4
                ),
            )
        )
        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            barmode="group",
            xaxis=dict(title="Category", gridcolor="#F1F5F9"),
            yaxis=dict(title="Score", range=[0, 1.05], gridcolor="#F1F5F9", tickformat=".0%"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=380,
        )
        return _figure_json(fig)

    # ------------------------------------------------------------------
    # Deep analysis charts
    # ------------------------------------------------------------------

    def _build_deep_analysis_figures(self, results: EvalResults) -> dict[str, str]:
        figures: dict[str, str] = {}
        da = results.deep_analysis
        if da is None:
            return figures

        if da.cka:
            figures["cka_heatmap"] = self._build_cka_figure(da.cka)
        if da.perplexity:
            figures["ppl_dist"] = self._build_ppl_figure(da.perplexity)
        if da.spectral:
            figures["spectral"] = self._build_spectral_figure(da.spectral)
        if da.calibration:
            figures["calibration"] = self._build_calibration_figure(da.calibration)
        if da.activation:
            figures["activation"] = self._build_activation_figure(da.activation)
        return figures

    def _build_cka_figure(self, cka: CKAReport) -> str:
        layers = list(cka.per_layer_cka.keys())
        values = list(cka.per_layer_cka.values())
        colors = ["#10B981" if v >= 0.9 else "#F59E0B" if v >= 0.7 else "#EF4444" for v in values]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=layers,
                y=values,
                marker=dict(color=colors, cornerradius=4),
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(size=10),
            )
        )
        fig.add_hline(y=cka.mean_cka, line_dash="dash", line_color="#6366F1", opacity=0.7)
        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=dict(text="CKA Similarity per Layer", font=dict(size=14)),
            xaxis=dict(title="Layer", gridcolor="#F1F5F9", tickangle=-45),
            yaxis=dict(title="CKA Score", range=[0, 1.1], gridcolor="#F1F5F9"),
            height=360,
        )
        return _figure_json(fig)

    def _build_ppl_figure(self, ppl: PerplexityDistShift) -> str:
        fig = go.Figure()

        if ppl.base_ppls:
            fig.add_trace(
                go.Histogram(
                    x=ppl.base_ppls,
                    name="Base Model",
                    marker_color="rgba(99, 102, 241, 0.6)",
                    nbinsx=40,
                )
            )
        if ppl.ft_ppls:
            fig.add_trace(
                go.Histogram(
                    x=ppl.ft_ppls,
                    name="Fine-tuned",
                    marker_color="rgba(16, 185, 129, 0.6)",
                    nbinsx=40,
                )
            )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=dict(text="Perplexity Distribution", font=dict(size=14)),
            xaxis=dict(title="Perplexity", gridcolor="#F1F5F9"),
            yaxis=dict(title="Count", gridcolor="#F1F5F9"),
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=360,
        )
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=(
                f"Wasserstein: {ppl.wasserstein_distance:.3f}<br>"
                f"Tail fraction (2x): {ppl.tail_fraction:.1%}"
            ),
            showarrow=False,
            font=dict(size=11, color="#475569"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            xanchor="right",
            yanchor="top",
            align="right",
        )
        return _figure_json(fig)

    def _build_spectral_figure(self, spectral: SpectralReport) -> str:
        layers = list(spectral.per_layer_effective_rank.keys())
        ranks = list(spectral.per_layer_effective_rank.values())

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=ranks,
                mode="lines+markers",
                line=dict(color="#6366F1", width=2),
                marker=dict(size=6, color="#6366F1"),
                name="Effective Rank",
            )
        )
        fig.add_hline(
            y=spectral.mean_effective_rank,
            line_dash="dash",
            line_color="#F59E0B",
            opacity=0.7,
            annotation_text=f"Mean: {spectral.mean_effective_rank:.1f}",
        )
        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=dict(text="Effective Rank per Layer", font=dict(size=14)),
            xaxis=dict(title="Layer", gridcolor="#F1F5F9", tickangle=-45),
            yaxis=dict(title="Effective Rank", gridcolor="#F1F5F9"),
            height=360,
        )
        return _figure_json(fig)

    def _build_calibration_figure(self, cal: CalibrationReport) -> str:
        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#94A3B8", dash="dash", width=1),
                name="Perfect",
                showlegend=False,
            )
        )

        if cal.per_bin_confidence:
            if cal.per_bin_accuracy_base:
                fig.add_trace(
                    go.Scatter(
                        x=cal.per_bin_confidence,
                        y=cal.per_bin_accuracy_base,
                        mode="lines+markers",
                        line=dict(color="#6366F1", width=2),
                        marker=dict(size=6),
                        name=f"Base (ECE={cal.base_ece:.3f})",
                    )
                )
            if cal.per_bin_accuracy_ft:
                fig.add_trace(
                    go.Scatter(
                        x=cal.per_bin_confidence_ft or cal.per_bin_confidence,
                        y=cal.per_bin_accuracy_ft,
                        mode="lines+markers",
                        line=dict(color="#10B981", width=2),
                        marker=dict(size=6),
                        name=f"Fine-tuned (ECE={cal.ft_ece:.3f})",
                    )
                )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=dict(text="Reliability Diagram", font=dict(size=14)),
            xaxis=dict(title="Confidence", range=[0, 1], gridcolor="#F1F5F9"),
            yaxis=dict(title="Accuracy", range=[0, 1], gridcolor="#F1F5F9"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=360,
        )
        return _figure_json(fig)

    def _build_activation_figure(self, act: ActivationDriftReport) -> str:
        layers = list(act.per_layer_cosine_sim.keys())
        values = list(act.per_layer_cosine_sim.values())
        drift = [1.0 - v for v in values]
        colors = ["#10B981" if d < 0.05 else "#F59E0B" if d < 0.15 else "#EF4444" for d in drift]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=layers,
                y=drift,
                marker=dict(color=colors, cornerradius=4),
                text=[f"{d:.3f}" for d in drift],
                textposition="outside",
                textfont=dict(size=10),
            )
        )
        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=dict(text="Activation Drift per Layer", font=dict(size=14)),
            xaxis=dict(title="Layer", gridcolor="#F1F5F9", tickangle=-45),
            yaxis=dict(title="Drift (1 - cosine sim)", gridcolor="#F1F5F9"),
            height=360,
        )
        return _figure_json(fig)

    def _build_roi_breakdown(self, results: EvalResults) -> str:
        """Stacked horizontal bar showing the 5 weighted ROI components."""
        if not results.forgetting:
            return "{}"

        key_labels = {
            "target": "Target",
            "retention": "Retention",
            "safety": "Safety",
            "selectivity": "Selectivity",
            "bwt": "BWT",
        }
        weights = results.roi_component_weights
        values = results.roi_component_values
        if not weights:
            from finetunecheck.forgetting.metrics import compute_roi_details

            rates = [
                value
                for value in results.forgetting.capability_retention_rates.values()
                if value is not None
            ]
            details = compute_roi_details(
                results.target_improvement,
                results.forgetting.backward_transfer,
                results.forgetting.safety_alignment_retention,
                results.forgetting.selective_forgetting_index,
                sum(rates) / len(rates) if rates else None,
            )
            weights = details["weights"]
            values = details["values"]
        components = {
            key_labels[key]: weight * (values.get(key) or 0.0) for key, weight in weights.items()
        }
        max_points = {key_labels[key]: weight for key, weight in weights.items()}
        colors = {
            "Target": "#10B981",
            "Retention": "#6366F1",
            "Safety": "#3B82F6",
            "Selectivity": "#F59E0B",
            "BWT": "#EF4444",
        }

        fig = go.Figure()
        for name, score in components.items():
            fig.add_trace(
                go.Bar(
                    name=f"{name} ({score:.1f}/{max_points[name]})",
                    x=[score],
                    y=["ROI"],
                    orientation="h",
                    marker_color=colors[name],
                    text=f"{name}<br>{score:.1f}",
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=11, color="white"),
                )
            )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=dict(
                text=(
                    f"ROI Score Breakdown — Total: {results.roi_score:.0f}/100 "
                    f"(coverage {results.roi_coverage:.0%})"
                    if results.roi_score is not None
                    else "ROI Score Breakdown — unavailable"
                ),
                font=dict(size=14),
            ),
            barmode="stack",
            xaxis=dict(title="Points", range=[0, 102], gridcolor="#F1F5F9"),
            yaxis=dict(title=""),
            height=180,
            legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="center", x=0.5),
        )
        return _figure_json(fig)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _measured_categories(results: EvalResults) -> list[str]:
        return [
            category
            for category in sorted(set(results.base_scores) & set(results.ft_scores))
            if results.base_scores[category].mean_score is not None
            and results.ft_scores[category].mean_score is not None
        ]

    @staticmethod
    def _get_plotly_js_cdn() -> str:
        return _PLOTLY_CDN
