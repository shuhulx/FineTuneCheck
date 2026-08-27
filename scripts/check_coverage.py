"""Enforce per-module release coverage after a coverage.py run."""

from __future__ import annotations

from pathlib import Path

from coverage import Coverage

MINIMUM = 90.0
CRITICAL_MODULES = (
    "src/finetunecheck/eval/runner.py",
    "src/finetunecheck/eval/judge.py",
    "src/finetunecheck/eval/cache.py",
    "src/finetunecheck/forgetting/metrics.py",
    "src/finetunecheck/profiles/loader.py",
)


def main() -> None:
    coverage = Coverage()
    coverage.load()
    failures: list[str] = []
    for relative in CRITICAL_MODULES:
        path = Path(relative).resolve()
        _filename, statements, _excluded, missing, _formatted = coverage.analysis2(str(path))
        percent = 100.0 * (len(statements) - len(missing)) / len(statements)
        print(f"{relative}: {percent:.1f}%")
        if percent < MINIMUM:
            failures.append(f"{relative}={percent:.1f}%")
    if failures:
        raise SystemExit(
            f"Critical coverage must be at least {MINIMUM:.0f}%: " + ", ".join(failures)
        )


if __name__ == "__main__":
    main()
