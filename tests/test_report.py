"""Tests for report generation."""

import json
import re
from pathlib import Path

import pytest

from finetunecheck.models import EvalResults, ForgettingPattern, ForgettingReport, Verdict
from finetunecheck.report.generator import ReportGenerator


class TestHTMLReportGeneration:
    def test_html_report_generation(self, sample_eval_results, tmp_path):
        """Should generate valid HTML file."""
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        result_path = gen.generate(sample_eval_results, str(output))
        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "<html" in content.lower()
        assert "</html>" in content.lower()
        assert len(content) > 1000  # non-trivial content

    def test_html_report_contains_verdict(self, sample_eval_results, tmp_path):
        """Report should contain the verdict."""
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        assert "GOOD" in content

    def test_html_report_contains_model_names(self, sample_eval_results, tmp_path):
        """Report should contain model names."""
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        assert "meta-llama" in content or "Llama" in content

    def test_html_report_contains_scores(self, sample_eval_results, tmp_path):
        """Report should contain plotly chart data."""
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        # Should contain plotly.js reference
        assert "plotly" in content.lower()

    def test_html_report_with_deep_analysis(self, sample_eval_results, sample_deep_analysis, tmp_path):
        """Report should include deep analysis section when provided."""
        sample_eval_results.deep_analysis = sample_deep_analysis
        output = tmp_path / "report_deep.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        assert Path(output).exists()
        # Deep analysis charts should be present
        assert "CKA" in content or "cka" in content.lower()
        assert len(content) > 2000

    def test_html_report_creates_parent_dirs(self, sample_eval_results, tmp_path):
        """Should create parent directories if they don't exist."""
        output = tmp_path / "sub" / "dir" / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        assert output.exists()

    def test_html_report_custom_title(self, sample_eval_results, tmp_path):
        """Report should use custom title."""
        output = tmp_path / "titled.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output), title="My Custom Report")
        content = output.read_text()
        assert "My Custom Report" in content

    def test_html_report_all_verdicts(self, sample_eval_results, tmp_path):
        """Report should render for all verdict types."""
        gen = ReportGenerator()
        for verdict in Verdict:
            sample_eval_results.verdict = verdict
            output = tmp_path / f"report_{verdict.value}.html"
            gen.generate(sample_eval_results, str(output))
            assert output.exists()
            content = output.read_text()
            assert len(content) > 500

    def test_html_report_no_forgetting(self, base_scores, ft_scores_good, tmp_path):
        """Report should work when forgetting report is None."""
        results = EvalResults(
            base_model="base",
            finetuned_model="ft",
            base_scores=base_scores,
            ft_scores=ft_scores_good,
        )
        output = tmp_path / "no_forgetting.html"
        gen = ReportGenerator()
        gen.generate(results, str(output))
        assert output.exists()


class TestReportExport:
    def test_json_export(self, sample_eval_results, tmp_path):
        """Should export valid JSON."""
        output = tmp_path / "results.json"
        data = sample_eval_results.model_dump()
        output.write_text(json.dumps(data, indent=2, default=str))
        assert output.exists()
        loaded = json.loads(output.read_text())
        assert loaded["base_model"] == "meta-llama/Llama-3.1-8B"
        assert loaded["verdict"] == "GOOD"
        assert "forgetting" in loaded
        assert loaded["forgetting"]["pattern"] == "minimal"

    def test_json_roundtrip(self, sample_eval_results, tmp_path):
        """JSON export should be loadable back into EvalResults."""
        data = sample_eval_results.model_dump()
        json_str = json.dumps(data, default=str)
        loaded = json.loads(json_str)
        # Verify key fields survived roundtrip
        assert loaded["roi_score"] == 78.0
        assert len(loaded["concerns"]) == 1
        assert len(loaded["recommendations"]) == 1

    def test_csv_export(self, sample_eval_results, tmp_path):
        """Should export valid CSV of category scores."""
        import csv
        output = tmp_path / "results.csv"
        rows = []
        for cat, score in sample_eval_results.base_scores.items():
            ft_score = sample_eval_results.ft_scores.get(cat)
            rows.append({
                "category": cat,
                "base_mean": score.mean_score,
                "ft_mean": ft_score.mean_score if ft_score else None,
                "base_std": score.std_score,
                "ft_std": ft_score.std_score if ft_score else None,
            })

        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "base_mean", "ft_mean", "base_std", "ft_std"])
            writer.writeheader()
            writer.writerows(rows)

        assert output.exists()
        with output.open() as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 5
        assert csv_rows[0]["category"] in sample_eval_results.base_scores

    def test_markdown_export(self, sample_eval_results, tmp_path):
        """Should export valid Markdown summary."""
        output = tmp_path / "results.md"
        lines = [
            "# FineTuneCheck Report",
            "",
            f"**Verdict:** {sample_eval_results.verdict.value}",
            f"**ROI Score:** {sample_eval_results.roi_score}",
            "",
            "## Summary",
            f"{sample_eval_results.summary}",
            "",
            "## Category Scores",
            "| Category | Base | Fine-tuned |",
            "|----------|------|------------|",
        ]
        for cat in sorted(sample_eval_results.base_scores.keys()):
            base = sample_eval_results.base_scores[cat].mean_score
            ft = sample_eval_results.ft_scores[cat].mean_score
            lines.append(f"| {cat} | {base:.3f} | {ft:.3f} |")

        if sample_eval_results.concerns:
            lines.append("\n## Concerns")
            for concern in sample_eval_results.concerns:
                lines.append(f"- {concern}")

        output.write_text("\n".join(lines))
        content = output.read_text()
        assert "Verdict" in content
        assert "GOOD" in content
        assert "reasoning" in content
        assert "78.0" in content


# ---------------------------------------------------------------------------
# _build_roi_breakdown
# ---------------------------------------------------------------------------

class TestBuildROIBreakdown:
    def test_returns_valid_json(self, sample_eval_results):
        gen = ReportGenerator()
        result = gen._build_roi_breakdown(sample_eval_results)
        parsed = json.loads(result)
        assert "data" in parsed

    def test_returns_empty_dict_when_no_forgetting(self, base_scores, ft_scores_good):
        results = EvalResults(
            base_model="base",
            finetuned_model="ft",
            base_scores=base_scores,
            ft_scores=ft_scores_good,
        )
        gen = ReportGenerator()
        result = gen._build_roi_breakdown(results)
        assert result == "{}"

    def test_has_five_traces(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(sample_eval_results))
        assert len(parsed["data"]) == 5

    def test_trace_names_cover_all_components(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(sample_eval_results))
        names_concat = " ".join(t["name"] for t in parsed["data"])
        for component in ("Target", "Retention", "Safety", "Selectivity", "BWT"):
            assert component in names_concat

    def test_all_scores_non_negative(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(sample_eval_results))
        for trace in parsed["data"]:
            assert trace["x"][0] >= 0.0

    def test_scores_within_max_points(self, sample_eval_results):
        """Each component score must not exceed its weight cap."""
        max_points = {"Target": 30, "Retention": 25, "Safety": 25, "Selectivity": 10, "BWT": 10}
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(sample_eval_results))
        for trace in parsed["data"]:
            name = trace["name"].split(" (")[0]  # strip "(score/max)" suffix
            assert trace["x"][0] <= max_points[name] + 1e-6, f"{name} exceeded cap"

    def test_total_does_not_exceed_100(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(sample_eval_results))
        total = sum(t["x"][0] for t in parsed["data"])
        assert total <= 100.0 + 1e-6

    def test_orientation_is_horizontal(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(sample_eval_results))
        for trace in parsed["data"]:
            assert trace.get("orientation") == "h"

    def test_barmode_is_stack(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(sample_eval_results))
        assert parsed["layout"]["barmode"] == "stack"

    def test_title_contains_roi_score(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(sample_eval_results))
        title_text = parsed["layout"]["title"]["text"]
        assert "78" in title_text  # roi_score=78.0

    def test_safety_defaults_to_full_when_none(self, base_scores, ft_scores_good):
        """When safety_alignment_retention is None, Safety component should use 1.0."""
        results = EvalResults(
            base_model="base",
            finetuned_model="ft",
            base_scores=base_scores,
            ft_scores=ft_scores_good,
            target_improvement=0.5,
            forgetting=ForgettingReport(
                backward_transfer=0.0,
                capability_retention_rates={"reasoning": 1.0},
                selective_forgetting_index=0.0,
                safety_alignment_retention=None,
                pattern=ForgettingPattern.MINIMAL,
                most_affected=[],
                resilient=[],
            ),
        )
        gen = ReportGenerator()
        parsed = json.loads(gen._build_roi_breakdown(results))
        safety_trace = next(t for t in parsed["data"] if t["name"].startswith("Safety"))
        assert safety_trace["x"][0] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# _build_target_bars — error bars
# ---------------------------------------------------------------------------

class TestBuildTargetBarsErrorBars:
    def test_error_bars_present_on_both_traces(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_target_bars(sample_eval_results))
        assert len(parsed["data"]) >= 2
        for trace in parsed["data"]:
            assert "error_y" in trace, f"Trace '{trace.get('name')}' missing error_y"

    def test_error_bars_visible(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_target_bars(sample_eval_results))
        for trace in parsed["data"]:
            assert trace["error_y"]["visible"] is True

    def test_error_bar_array_length_matches_categories(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_target_bars(sample_eval_results))
        num_cats = len(sample_eval_results.base_scores)
        for trace in parsed["data"]:
            assert len(trace["error_y"]["array"]) == num_cats

    def test_error_bar_values_are_non_negative(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_target_bars(sample_eval_results))
        for trace in parsed["data"]:
            for v in trace["error_y"]["array"]:
                assert v >= 0.0

    def test_base_error_bars_match_std_scores(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_target_bars(sample_eval_results))
        base_trace = next(t for t in parsed["data"] if "Base" in t["name"])
        categories = sorted(sample_eval_results.base_scores.keys())
        expected = [sample_eval_results.base_scores[c].std_score for c in categories]
        assert base_trace["error_y"]["array"] == pytest.approx(expected)

    def test_ft_error_bars_match_std_scores(self, sample_eval_results):
        gen = ReportGenerator()
        parsed = json.loads(gen._build_target_bars(sample_eval_results))
        ft_trace = next(t for t in parsed["data"] if "Fine" in t["name"])
        categories = sorted(sample_eval_results.ft_scores.keys())
        expected = [sample_eval_results.ft_scores[c].std_score for c in categories]
        assert ft_trace["error_y"]["array"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _build_ppl_figure — Wasserstein annotation
# ---------------------------------------------------------------------------

class TestBuildPPLFigureAnnotation:
    def test_annotation_present_in_figure(self, sample_deep_analysis):
        gen = ReportGenerator()
        ppl = sample_deep_analysis.perplexity
        parsed = json.loads(gen._build_ppl_figure(ppl))
        annotations = parsed.get("layout", {}).get("annotations", [])
        assert len(annotations) >= 1

    def test_annotation_contains_wasserstein_value(self, sample_deep_analysis):
        gen = ReportGenerator()
        ppl = sample_deep_analysis.perplexity
        parsed = json.loads(gen._build_ppl_figure(ppl))
        annotations = parsed["layout"]["annotations"]
        text_concat = " ".join(a["text"] for a in annotations)
        assert "Wasserstein" in text_concat
        assert "2.300" in text_concat  # wasserstein_distance=2.3

    def test_annotation_contains_tail_fraction(self, sample_deep_analysis):
        gen = ReportGenerator()
        ppl = sample_deep_analysis.perplexity
        parsed = json.loads(gen._build_ppl_figure(ppl))
        annotations = parsed["layout"]["annotations"]
        text_concat = " ".join(a["text"] for a in annotations)
        assert "Tail" in text_concat or "tail" in text_concat.lower()

    def test_annotation_positioned_top_right(self, sample_deep_analysis):
        gen = ReportGenerator()
        ppl = sample_deep_analysis.perplexity
        parsed = json.loads(gen._build_ppl_figure(ppl))
        annotation = parsed["layout"]["annotations"][0]
        assert annotation["xanchor"] == "right"
        assert annotation["yanchor"] == "top"


# ---------------------------------------------------------------------------
# Full HTML report — new feature presence
# ---------------------------------------------------------------------------


class TestHTMLReportNewFeatures:
    def test_roi_breakdown_section_present(self, sample_eval_results, tmp_path):
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        assert "ROI Score Breakdown" in content

    def test_roi_breakdown_chart_div_present(self, sample_eval_results, tmp_path):
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        assert "chart-roi-breakdown" in content

    def test_roi_breakdown_absent_when_no_forgetting(self, base_scores, ft_scores_good, tmp_path):
        results = EvalResults(
            base_model="base",
            finetuned_model="ft",
            base_scores=base_scores,
            ft_scores=ft_scores_good,
        )
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(results, str(output))
        content = output.read_text()
        assert "ROI Score Breakdown" not in content

    def test_error_y_in_category_chart_json(self, sample_eval_results, tmp_path):
        """Category scores chart should embed error_y in the HTML."""
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        assert "error_y" in content

    def test_wasserstein_in_deep_analysis_report(self, sample_eval_results, sample_deep_analysis, tmp_path):
        sample_eval_results.deep_analysis = sample_deep_analysis
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        assert "Wasserstein" in content

    def test_ppl_annotation_value_in_report(self, sample_eval_results, sample_deep_analysis, tmp_path):
        sample_eval_results.deep_analysis = sample_deep_analysis
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        assert "2.300" in content  # wasserstein_distance=2.3

    def test_roi_breakdown_json_parseable(self, sample_eval_results, tmp_path):
        """ROI breakdown JSON embedded in the report must be parseable."""
        output = tmp_path / "report.html"
        gen = ReportGenerator()
        gen.generate(sample_eval_results, str(output))
        content = output.read_text()
        match = re.search(r"renderChart\('chart-roi-breakdown',\s*(.+?)\);", content)
        assert match, "chart-roi-breakdown renderChart call not found"
        parsed = json.loads(match.group(1))
        assert "data" in parsed
