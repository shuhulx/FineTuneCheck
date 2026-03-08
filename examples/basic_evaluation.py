"""Basic fine-tuning evaluation example.

Usage:
    python examples/basic_evaluation.py <base_model> <finetuned_model>
"""
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: python basic_evaluation.py <base_model> <finetuned_model>")
        sys.exit(1)

    from finetunecheck.config import EvalConfig
    from finetunecheck.eval.runner import EvalRunner

    config = EvalConfig(
        base_model=sys.argv[1],
        finetuned_model=sys.argv[2],
        profile="general",
    )

    runner = EvalRunner(config)
    result = runner.run()

    print(f"Verdict: {result.verdict.value}")
    print(f"ROI Score: {result.roi_score:.0f}/100")
    print("\nCategory Scores:")
    for cat in sorted(result.base_scores.keys()):
        bs = result.base_scores[cat].mean_score
        fs = result.ft_scores[cat].mean_score
        delta = fs - bs
        print(f"  {cat}: {bs:.3f} -> {fs:.3f} ({delta:+.3f})")

    if result.forgetting:
        print(f"\nForgetting pattern: {result.forgetting.pattern.value}")
        print(f"Backward Transfer: {result.forgetting.backward_transfer:+.3f}")


if __name__ == "__main__":
    main()
