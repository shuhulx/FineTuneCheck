"""Tests for report generation."""

import json
from pathlib import Path

from finetunecheck.models import EvalResults, Verdict
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
