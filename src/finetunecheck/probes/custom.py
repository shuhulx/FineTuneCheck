"""Create custom probe sets from user data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from finetunecheck.models import JudgeType, ProbeSample, ProbeSet


class CustomProbe:
    """Factory for creating custom probe sets from various data sources."""

    @staticmethod
    def from_samples(
        name: str,
        samples: list[dict],
        judge_type: str = "llm",
        judge_criteria: str = "",
        category: str = "custom",
        version: str = "1.0",
    ) -> ProbeSet:
        """Create a ProbeSet from a list of sample dicts.

        Each dict should have at least ``"input"`` and optionally ``"reference"``,
        ``"id"``, ``"difficulty"``, ``"tags"``, and ``"metadata"``.
        """
        if not samples:
            raise ValueError("samples list must not be empty")
        parsed: list[ProbeSample] = []
        for i, s in enumerate(samples):
            if "input" not in s:
                raise ValueError(f"Sample {i} missing required 'input' key")
            parsed.append(
                ProbeSample(
                    id=s.get("id", f"{name}_{i}"),
                    input=s["input"],
                    reference=s.get("reference"),
                    difficulty=s.get("difficulty", "medium"),
                    tags=s.get("tags", []),
                    metadata=s.get("metadata", {}),
                )
            )
        return ProbeSet(
            name=name,
            version=version,
            category=category,
            judge_type=JudgeType(judge_type),
            judge_criteria=judge_criteria,
            samples=parsed,
        )

    @staticmethod
    def from_csv(
        name: str,
        csv_path: str,
        input_col: str = "input",
        reference_col: str = "reference",
        id_col: str | None = None,
        difficulty_col: str | None = None,
        judge_type: str = "llm",
        judge_criteria: str = "",
        category: str = "custom",
        version: str = "1.0",
    ) -> ProbeSet:
        """Load a ProbeSet from a CSV file.

        The CSV must have at least the ``input_col`` column.
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        samples: list[ProbeSample] = []
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if input_col not in (reader.fieldnames or []):
                raise ValueError(
                    f"Column {input_col!r} not found in CSV. Available: {reader.fieldnames}"
                )
            for i, row in enumerate(reader):
                sample_id = row[id_col] if id_col and id_col in row else f"{name}_{i}"
                difficulty = (
                    row[difficulty_col] if difficulty_col and difficulty_col in row else "medium"
                )
                samples.append(
                    ProbeSample(
                        id=sample_id,
                        input=row[input_col],
                        reference=row.get(reference_col),
                        difficulty=difficulty,
                    )
                )

        if not samples:
            raise ValueError(f"No samples found in {csv_path}")

        return ProbeSet(
            name=name,
            version=version,
            category=category,
            judge_type=JudgeType(judge_type),
            judge_criteria=judge_criteria,
            samples=samples,
        )

    @staticmethod
    def from_jsonl(
        name: str,
        jsonl_path: str,
        judge_type: str = "llm",
        judge_criteria: str = "",
        category: str = "custom",
        version: str = "1.0",
    ) -> ProbeSet:
        """Load a ProbeSet from a JSONL file.

        Each line must be a JSON object with at least ``"input"`` and optionally
        ``"reference"``, ``"id"``, ``"difficulty"``, ``"tags"``, ``"metadata"``.
        """
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        samples: list[ProbeSample] = []
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {i + 1} of {jsonl_path}: {exc}"
                    ) from exc
                if "input" not in obj:
                    raise ValueError(f"Line {i + 1} missing required 'input' key")
                samples.append(
                    ProbeSample(
                        id=obj.get("id", f"{name}_{i}"),
                        input=obj["input"],
                        reference=obj.get("reference"),
                        difficulty=obj.get("difficulty", "medium"),
                        tags=obj.get("tags", []),
                        metadata=obj.get("metadata", {}),
                    )
                )

        if not samples:
            raise ValueError(f"No samples found in {jsonl_path}")

        return ProbeSet(
            name=name,
            version=version,
            category=category,
            judge_type=JudgeType(judge_type),
            judge_criteria=judge_criteria,
            samples=samples,
        )
