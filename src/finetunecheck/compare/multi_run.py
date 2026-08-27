"""Compare multiple fine-tuning runs against the same base model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from finetunecheck.compare.pareto import compute_pareto_frontier
from finetunecheck.config import EvalConfig
from finetunecheck.models import EvalResults


class ComparisonResult(BaseModel):
    """Aggregated comparison across multiple fine-tuning runs."""

    base_model: str
    runs: dict[str, EvalResults]
    best_run: str = ""
    best_target_perf: str = ""
    least_forgetting: str = ""
    pareto_frontier: list[str] = Field(default_factory=list)
    recommendation: str = ""


class MultiRunComparator:
    """Compare multiple fine-tuned checkpoints against a shared baseline.

    Usage::

        comparator = MultiRunComparator()
        result = comparator.compare(
            base_model="meta-llama/Llama-3-8B",
            finetuned_models={
                "run_lr3e4": "./checkpoints/lr3e-4",
                "run_lr1e4": "./checkpoints/lr1e-4",
            },
            config=EvalConfig(base_model="meta-llama/Llama-3-8B", finetuned_model=""),
        )
    """

    def compare(
        self,
        base_model: str,
        finetuned_models: dict[str, str],
        config: EvalConfig,
    ) -> ComparisonResult:
        """Run evaluation for each fine-tuned model and compare.

        Parameters
        ----------
        base_model:
            Path or HF ID of the base model.
        finetuned_models:
            Mapping of run name to model path/ID.
        config:
            Evaluation config (``base_model`` and ``finetuned_model`` fields
            will be overridden per run).
        """
        # Lazy import to avoid circular deps and heavy torch import at module level
        from finetunecheck.eval.runner import EvalRunner

        run_results: dict[str, EvalResults] = {}

        shared_base_scores = None
        for run_name, ft_path in finetuned_models.items():
            payload = config.model_dump()
            payload.update({"base_model": base_model, "finetuned_model": ft_path})
            run_config = EvalConfig.model_validate(payload)
            runner = EvalRunner(run_config, shared_base_scores=shared_base_scores)
            result = runner.run()
            run_results[run_name] = result
            if shared_base_scores is None:
                shared_base_scores = result.base_scores

        return self._analyze(base_model, run_results)

    def compare_from_results(
        self,
        base_model: str,
        run_results: dict[str, EvalResults],
    ) -> ComparisonResult:
        """Compare pre-computed results (useful when evaluation is done externally)."""
        return self._analyze(base_model, run_results)

    def _analyze(
        self,
        base_model: str,
        run_results: dict[str, EvalResults],
    ) -> ComparisonResult:
        if not run_results:
            return ComparisonResult(
                base_model=base_model,
                runs={},
                recommendation="No runs provided for comparison.",
            )

        self._validate_compatibility(base_model, run_results)

        # Best by ROI
        best_run = max(run_results, key=lambda n: run_results[n].roi_score or 0.0)

        # Best target performance
        def _target_gain(name: str) -> float:
            value = run_results[name].target_improvement
            return value if value is not None else float("-inf")

        best_target = max(run_results, key=_target_gain)

        # Least forgetting (highest backward_transfer, i.e. closest to 0)
        def _bwt(name: str) -> float:
            f = run_results[name].forgetting
            return (
                f.backward_transfer
                if f is not None and f.backward_transfer is not None
                else float("-inf")
            )

        least_forg = max(run_results, key=_bwt)

        # Pareto frontier: target improvement and BWT are both directly higher-is-better.
        pareto_points: dict[str, tuple[float, float]] = {}
        for name, res in run_results.items():
            if (
                res.target_improvement is None
                or res.forgetting is None
                or res.forgetting.backward_transfer is None
            ):
                raise ValueError(
                    f"Run {name!r} lacks measured target improvement or backward transfer"
                )
            pareto_points[name] = (
                res.target_improvement,
                res.forgetting.backward_transfer,
            )

        frontier = compute_pareto_frontier(pareto_points)

        # Build recommendation
        recommendation = self._build_recommendation(
            run_results, best_run, best_target, least_forg, frontier
        )

        return ComparisonResult(
            base_model=base_model,
            runs=run_results,
            best_run=best_run,
            best_target_perf=best_target,
            least_forgetting=least_forg,
            pareto_frontier=frontier,
            recommendation=recommendation,
        )

    @staticmethod
    def _validate_compatibility(base_model: str, run_results: dict[str, EvalResults]) -> None:
        first_name, first = next(iter(run_results.items()))
        if not first.probe_digest:
            raise ValueError(
                f"Run {first_name!r} lacks a probe digest and cannot be compared safely"
            )
        expected = {
            "base_model": first.base_model,
            "probe_digest": first.probe_digest,
            "judge_provenance": first.judge_provenance,
            "target_tasks": first.target_tasks,
            "result_schema_version": first.result_schema_version,
            "metric_schema_version": first.metric_schema_version,
        }
        if first.base_model != base_model:
            raise ValueError(
                f"Comparison base {base_model!r} does not match run {first_name!r} base "
                f"{first.base_model!r}"
            )
        for name, result in run_results.items():
            actual = {
                "base_model": result.base_model,
                "probe_digest": result.probe_digest,
                "judge_provenance": result.judge_provenance,
                "target_tasks": result.target_tasks,
                "result_schema_version": result.result_schema_version,
                "metric_schema_version": result.metric_schema_version,
            }
            mismatches = [key for key in expected if actual[key] != expected[key]]
            if mismatches:
                raise ValueError(
                    f"Run {name!r} is incompatible with {first_name!r}: " + ", ".join(mismatches)
                )

    @staticmethod
    def _build_recommendation(
        runs: dict[str, EvalResults],
        best_run: str,
        best_target: str,
        least_forg: str,
        frontier: list[str],
    ) -> str:
        parts: list[str] = []

        best = runs[best_run]
        parts.append(
            f"Best overall run: '{best_run}' with ROI {(best.roi_score or 0.0):.0f}/100 "
            f"and verdict {best.verdict.value}."
        )

        if best_target != best_run:
            t = runs[best_target]
            parts.append(
                f"Highest target improvement: '{best_target}' ({(t.target_improvement or 0.0):+.3f})."
            )

        if least_forg != best_run:
            f = runs[least_forg]
            bwt = f.forgetting.backward_transfer if f.forgetting else 0.0
            parts.append(f"Least forgetting: '{least_forg}' (BWT={bwt:+.3f}).")

        if len(frontier) > 1:
            parts.append(
                f"Pareto-optimal runs: {', '.join(frontier)}. "
                "These represent different tradeoffs between target improvement "
                "and capability retention."
            )
        elif len(frontier) == 1:
            parts.append(f"'{frontier[0]}' dominates all other runs on both axes.")

        return " ".join(parts)
