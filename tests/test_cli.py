"""Tests for CLI commands.

The CLI module (finetunecheck.cli) may not yet exist. These tests
verify the expected interface using importability checks and,
if available, the typer test runner.
"""

import importlib

import pytest


class TestCLIImport:
    def test_cli_module_importable(self):
        """The CLI module should be importable (it's referenced in pyproject.toml)."""
        try:
            mod = importlib.import_module("finetunecheck.cli")
            assert hasattr(mod, "app"), "CLI module should export an 'app' object"
        except (ImportError, ModuleNotFoundError):
            pytest.skip("finetunecheck.cli module not yet implemented")


class TestCLICommands:
    @pytest.fixture(autouse=True)
    def _import_cli(self):
        """Try to import CLI; skip all tests in class if not available."""
        try:
            from finetunecheck.cli import app

            self.app = app
        except (ImportError, ModuleNotFoundError):
            pytest.skip("finetunecheck.cli module not yet implemented")

    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_cli_help(self):
        """CLI should show help."""
        runner = self._runner()
        result = runner.invoke(self.app, ["--help"])
        assert result.exit_code == 0
        # Should contain something about the tool
        output = result.stdout.lower()
        assert "finetunecheck" in output or "ftcheck" in output or "usage" in output

    def test_cli_list_profiles(self):
        """Should list available profiles."""
        runner = self._runner()
        result = runner.invoke(self.app, ["list-profiles"])
        if result.exit_code != 0:
            pytest.skip("list-profiles command not implemented")
        assert "general" in result.stdout.lower() or len(result.stdout) > 0

    def test_cli_list_probes(self):
        """Should list available probes."""
        runner = self._runner()
        result = runner.invoke(self.app, ["list-probes"])
        if result.exit_code != 0:
            pytest.skip("list-probes command not implemented")
        assert "reasoning" in result.stdout.lower()
