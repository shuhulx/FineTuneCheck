"""Generate self-contained HTML reports with embedded Plotly visualizations."""

from __future__ import annotations

from pathlib import Path

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

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.32.0.min.js"

_VERDICT_COLORS = {
    Verdict.EXCELLENT: "#10B981",
    Verdict.GOOD: "#10B981",
    Verdict.GOOD_WITH_CONCERNS: "#F59E0B",
    Verdict.POOR: "#EF4444",
    Verdict.HARMFUL: "#991B1B",
}

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", color="#334155"),
    margin=dict(l=40, r=40, t=40, b=40),
)


class ReportGenerator:
    """Generate self-contained HTML reports with embedded Plotly visualizations."""

    def __init__(self) -> None:
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

        html = template.render(
            title=title,
            results=results,
            figures=figures,
            plotly_js=_PLOTLY_CDN,
            version=__version__,
            verdict_color=_VERDICT_COLORS.get(results.verdict, "#6B7280"),
            verdict_label=results.verdict.value.replace("_", " "),
        )

        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return str(out)

    # ------------------------------------------------------------------
    # Primary charts
    # ------------------------------------------------------------------

    def _build_radar_chart(self, results: EvalResults) -> str:
        categories = sorted(results.base_scores.keys())
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
        return fig.to_json()

    def _build_retention_bars(self, results: EvalResults) -> str:
        if not results.forgetting:
            return "{}"

        retention = results.forgetting.capability_retention_rates
        cats = sorted(retention.keys(), key=lambda c: retention[c])
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
        return fig.to_json()

    def _build_target_bars(self, results: EvalResults) -> str:
        categories = sorted(results.base_scores.keys())
        if not categories:
            return "{}"

        base_vals = [results.base_scores[c].mean_score for c in categories]
        ft_vals = [results.ft_scores[c].mean_score for c in categories]
        base_err = [results.base_scores[c].std_score for c in categories]
        ft_err = [results.ft_scores[c].std_score for c in categories]

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
        return fig.to_json()

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
        return fig.to_json()

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
        return fig.to_json()

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
        return fig.to_json()

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
                        x=cal.per_bin_confidence,
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
        return fig.to_json()

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
        return fig.to_json()

    def _build_roi_breakdown(self, results: EvalResults) -> str:
        """Stacked horizontal bar showing the 5 weighted ROI components."""
        if not results.forgetting:
            return "{}"

        f = results.forgetting
        crr_vals = list(f.capability_retention_rates.values())
        mean_crr = sum(crr_vals) / len(crr_vals) if crr_vals else 0.0

        components = {
            "Target": max(0.0, min(1.0, results.target_improvement)) * 30,
            "Retention": max(0.0, min(1.0, mean_crr)) * 25,
            "Safety": (
                max(0.0, min(1.0, f.safety_alignment_retention))
                if f.safety_alignment_retention is not None
                else 1.0
            )
            * 25,
            "Selectivity": max(0.0, 1.0 - f.selective_forgetting_index) * 10,
            "BWT": max(0.0, min(1.0, 1.0 + max(-1.0, f.backward_transfer))) * 10,
        }
        max_points = {"Target": 30, "Retention": 25, "Safety": 25, "Selectivity": 10, "BWT": 10}
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
                text=f"ROI Score Breakdown — Total: {results.roi_score:.0f}/100",
                font=dict(size=14),
            ),
            barmode="stack",
            xaxis=dict(title="Points", range=[0, 102], gridcolor="#F1F5F9"),
            yaxis=dict(title=""),
            height=180,
            legend=dict(orientation="h", yanchor="bottom", y=-0.55, xanchor="center", x=0.5),
        )
        return fig.to_json()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_plotly_js_cdn() -> str:
        return _PLOTLY_CDN
