"""Basic deterministic smoke-evaluation example.

Usage:
    python examples/basic_evaluation.py <base_model> <finetuned_model>
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python basic_evaluation.py <base_model> <finetuned_model>")
        sys.exit(1)

    from finetunecheck.config import QuickConfig
    from finetunecheck.eval.runner import EvalRunner

    config = QuickConfig(
        base_model=sys.argv[1],
        finetuned_model=sys.argv[2],
    )

    runner = EvalRunner(config)
    result = runner.run()

    print(f"Verdict: {result.verdict.value}")
    roi = f"{result.roi_score:.0f}/100" if result.roi_score is not None else "unavailable"
    print(f"ROI Score: {roi} (coverage {result.roi_coverage:.0%})")
    print("\nCategory Scores:")
    for cat in sorted(set(result.base_scores) | set(result.ft_scores)):
        if cat not in result.base_scores or cat not in result.ft_scores:
            print(f"  {cat}: missing paired evidence")
            continue
        bs = result.base_scores[cat].mean_score
        fs = result.ft_scores[cat].mean_score
        if bs is None or fs is None:
            print(
                f"  {cat}: {result.base_scores[cat].status.value} -> "
                f"{result.ft_scores[cat].status.value}"
            )
            continue
        delta = fs - bs
        print(f"  {cat}: {bs:.3f} -> {fs:.3f} ({delta:+.3f})")

    if result.forgetting:
        print(f"\nForgetting pattern: {result.forgetting.pattern.value}")
        bwt = result.forgetting.backward_transfer
        print(
            f"Backward Transfer: {bwt:+.3f}"
            if bwt is not None
            else "Backward Transfer: unavailable"
        )

    print("\nDiagnostic evidence only; not independent deployment approval.")


if __name__ == "__main__":
    main()
