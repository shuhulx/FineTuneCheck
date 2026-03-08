"""Tests for probe registry and custom probe creation."""

import json
from pathlib import Path

import pytest

from finetunecheck.models import JudgeType, ProbeSample, ProbeSet
from finetunecheck.probes.custom import CustomProbe
from finetunecheck.probes.registry import ProbeRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the probe registry before each test to avoid cross-test contamination."""
    ProbeRegistry.reset()
    yield
    ProbeRegistry.reset()


class TestProbeRegistry:
    def test_probe_registry_list(self):
        """All built-in probes should be listed (at least the 4 that exist on disk)."""
        names = ProbeRegistry.list()
        assert isinstance(names, list)
        # We have at least reasoning, code, math, instruction_following on disk
        assert len(names) >= 4
        assert "reasoning" in names
        assert "code" in names
        assert "math" in names

    def test_probe_registry_get(self):
        """Should load a probe by name and return a valid ProbeSet."""
        probe = ProbeRegistry.get("reasoning")
        assert isinstance(probe, ProbeSet)
        assert probe.name == "reasoning"
        assert probe.category == "reasoning"
        assert probe.judge_type == JudgeType.LLM
        assert len(probe.samples) > 0
        # Each sample should have an id and input
        for sample in probe.samples:
            assert sample.id
            assert sample.input

    def test_probe_registry_get_unknown(self):
        """Should raise KeyError for unknown probe."""
        with pytest.raises(KeyError, match="not found"):
            ProbeRegistry.get("nonexistent_probe_xyz")

    def test_probe_registry_get_for_profile(self):
        """get_for_profile should return a list of ProbeSets."""
        probes = ProbeRegistry.get_for_profile(["reasoning", "code"])
        assert len(probes) == 2
        assert probes[0].name == "reasoning"
        assert probes[1].name == "code"

    def test_probe_registry_get_for_profile_missing(self):
        """get_for_profile should raise KeyError if any probe is missing."""
        with pytest.raises(KeyError):
            ProbeRegistry.get_for_profile(["reasoning", "nonexistent_xyz"])

    def test_probe_registry_register_custom(self):
        """Should be able to register and retrieve a custom probe."""
        custom = ProbeSet(
            name="my_custom",
            category="custom",
            judge_type=JudgeType.EXACT_MATCH,
            samples=[ProbeSample(id="c_0", input="What is 2+2?", reference="4")],
        )
        ProbeRegistry.register(custom)
        retrieved = ProbeRegistry.get("my_custom")
        assert retrieved.name == "my_custom"
        assert retrieved.category == "custom"
        assert "my_custom" in ProbeRegistry.list()

    def test_probe_registry_custom_overrides_builtin(self):
        """Custom probe with same name should be returned over builtin."""
        custom = ProbeSet(
            name="reasoning",
            category="custom_reasoning",
            judge_type=JudgeType.EXACT_MATCH,
            samples=[ProbeSample(id="override_0", input="override?", reference="yes")],
        )
        ProbeRegistry.register(custom)
        retrieved = ProbeRegistry.get("reasoning")
        assert retrieved.category == "custom_reasoning"
        assert len(retrieved.samples) == 1

    def test_probe_json_structure(self):
        """Each built-in probe JSON should have valid structure (name, judge_type, samples with input)."""
        builtin_dir = Path(__file__).parent.parent / "src" / "finetunecheck" / "probes" / "builtin"
        if not builtin_dir.is_dir():
            pytest.skip("builtin probe directory not found")

        for json_path in builtin_dir.glob("*.json"):
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            assert "name" in raw, f"{json_path.name} missing 'name'"
            assert "judge_type" in raw, f"{json_path.name} missing 'judge_type'"
            assert "samples" in raw, f"{json_path.name} missing 'samples'"
            assert isinstance(raw["samples"], list), f"{json_path.name} 'samples' is not a list"
            assert len(raw["samples"]) > 0, f"{json_path.name} has no samples"
            for i, s in enumerate(raw["samples"]):
                assert "input" in s, f"{json_path.name} sample {i} missing 'input'"

    def test_probe_registry_lazy_load(self):
        """Registry should not load until first access."""
        ProbeRegistry.reset()
        assert ProbeRegistry._loaded is False
        _ = ProbeRegistry.list()
        assert ProbeRegistry._loaded is True


class TestCustomProbe:
    def test_custom_probe_from_samples(self):
        """Create custom probe from sample dicts."""
        samples = [
            {"input": "What is Python?", "reference": "A programming language", "id": "q1"},
            {"input": "What is Java?", "reference": "A programming language", "difficulty": "easy"},
            {"input": "What is Rust?"},
        ]
        probe = CustomProbe.from_samples(
            name="languages",
            samples=samples,
            judge_type="exact_match",
            category="trivia",
        )
        assert isinstance(probe, ProbeSet)
        assert probe.name == "languages"
        assert probe.category == "trivia"
        assert probe.judge_type == JudgeType.EXACT_MATCH
        assert len(probe.samples) == 3
        assert probe.samples[0].id == "q1"
        assert probe.samples[1].id == "languages_1"  # auto-generated
        assert probe.samples[2].reference is None
        assert probe.samples[1].difficulty == "easy"

    def test_custom_probe_from_samples_empty(self):
        """Should raise ValueError for empty samples list."""
        with pytest.raises(ValueError, match="must not be empty"):
            CustomProbe.from_samples(name="empty", samples=[])

    def test_custom_probe_from_samples_missing_input(self):
        """Should raise ValueError when a sample is missing the 'input' key."""
        with pytest.raises(ValueError, match="missing required 'input'"):
            CustomProbe.from_samples(
                name="bad",
                samples=[{"reference": "no input here"}],
            )

    def test_custom_probe_from_csv(self, tmp_path):
        """Create custom probe from CSV file."""
        csv_file = tmp_path / "probes.csv"
        csv_file.write_text(
            "input,reference,difficulty\n"
            "What is 1+1?,2,easy\n"
            "What is 2+2?,4,medium\n"
            "What is 3+3?,6,hard\n"
        )
        probe = CustomProbe.from_csv(
            name="math_csv",
            csv_path=str(csv_file),
            judge_type="exact_match",
            difficulty_col="difficulty",
        )
        assert isinstance(probe, ProbeSet)
        assert probe.name == "math_csv"
        assert len(probe.samples) == 3
        assert probe.samples[0].input == "What is 1+1?"
        assert probe.samples[0].reference == "2"
        assert probe.samples[0].difficulty == "easy"
        assert probe.samples[2].difficulty == "hard"

    def test_custom_probe_from_csv_missing_column(self, tmp_path):
        """Should raise ValueError when input column is missing from CSV."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("question,answer\nWhat?,This\n")
        with pytest.raises(ValueError, match="not found in CSV"):
            CustomProbe.from_csv(name="bad", csv_path=str(csv_file))

    def test_custom_probe_from_csv_file_not_found(self):
        """Should raise FileNotFoundError for missing CSV file."""
        with pytest.raises(FileNotFoundError):
            CustomProbe.from_csv(name="nope", csv_path="/nonexistent/path.csv")

    def test_custom_probe_from_csv_empty(self, tmp_path):
        """Should raise ValueError for CSV with headers but no data rows."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("input,reference\n")
        with pytest.raises(ValueError, match="No samples found"):
            CustomProbe.from_csv(name="empty", csv_path=str(csv_file))

    def test_custom_probe_from_csv_with_id_col(self, tmp_path):
        """CSV with custom id column should use those IDs."""
        csv_file = tmp_path / "with_id.csv"
        csv_file.write_text("qid,input,reference\nQ001,Hello?,World\nQ002,Bye?,Later\n")
        probe = CustomProbe.from_csv(
            name="with_ids",
            csv_path=str(csv_file),
            id_col="qid",
        )
        assert probe.samples[0].id == "Q001"
        assert probe.samples[1].id == "Q002"

    def test_custom_probe_from_jsonl(self, tmp_path):
        """Create custom probe from JSONL file."""
        jsonl_file = tmp_path / "probes.jsonl"
        lines = [
            json.dumps({"id": "j_0", "input": "Q1?", "reference": "A1"}),
            json.dumps({"input": "Q2?", "tags": ["tag1"]}),
        ]
        jsonl_file.write_text("\n".join(lines))
        probe = CustomProbe.from_jsonl(name="jsonl_test", jsonl_path=str(jsonl_file))
        assert len(probe.samples) == 2
        assert probe.samples[0].id == "j_0"
        assert probe.samples[1].id == "jsonl_test_1"
        assert probe.samples[1].tags == ["tag1"]

    def test_custom_probe_from_jsonl_invalid_json(self, tmp_path):
        """Should raise ValueError for malformed JSON lines."""
        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text('{"input": "ok"}\n{bad json}\n')
        with pytest.raises(ValueError, match="Invalid JSON"):
            CustomProbe.from_jsonl(name="bad", jsonl_path=str(jsonl_file))

    def test_custom_probe_from_jsonl_missing_input(self, tmp_path):
        """Should raise ValueError if a JSONL line is missing 'input'."""
        jsonl_file = tmp_path / "noinput.jsonl"
        jsonl_file.write_text('{"reference": "no input"}\n')
        with pytest.raises(ValueError, match="missing required 'input'"):
            CustomProbe.from_jsonl(name="bad", jsonl_path=str(jsonl_file))
