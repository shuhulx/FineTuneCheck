"""Release-blocking regressions for the trustworthy 2.0.0 evaluation contract.

Local fixtures and deterministic test doubles keep release verification offline;
the production inference integrations remain exercised by focused backend tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from finetunecheck.compare.multi_run import MultiRunComparator
from finetunecheck.config import EvalConfig
from finetunecheck.eval.cache import BaselineCache
from finetunecheck.eval.judge import (
    ExactMatchJudge,
    ExecutionJudge,
    LLMJudge,
    RuleBasedJudge,
)
from finetunecheck.eval.runner import EvalRunner
from finetunecheck.forgetting.metrics import (
    backward_transfer,
    capability_retention_rate,
    compute_roi_score,
    selective_forgetting_index,
)
from finetunecheck.models import (
    CategoryScore,
    EvalResults,
    ForgettingPattern,
    ForgettingReport,
    JudgeType,
    ProbeSample,
    ProbeSet,
    Verdict,
)
from finetunecheck.probes.registry import ProbeRegistry
from finetunecheck.profiles.loader import ProfileLoader
from finetunecheck.utils.model_loader import ModelLoader


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [("-5", "5"), ("3.14", "314"), ("1/2", "12")],
)
def test_exact_match_preserves_numeric_meaning(reference: str, candidate: str) -> None:
    sample = ProbeSample(id="numeric", input="Calculate", reference=reference)
    assert ExactMatchJudge().evaluate(sample, candidate).score == 0.0


def test_unparseable_llm_judgment_is_an_error() -> None:
    with pytest.raises(ValueError, match="parse"):
        LLMJudge._parse_judgment("confident prose without the required JSON schema")


def test_default_execution_judge_never_runs_candidate_code_on_host(monkeypatch) -> None:
    judge = ExecutionJudge()
    sample = ProbeSample(
        id="host-boundary",
        input="write code",
        reference="irrelevant",
        metadata={"test_cases": [{"input": "answer()", "output": "42"}]},
    )

    monkeypatch.setattr(
        judge,
        "_run_code",
        lambda _code: pytest.fail("default judge reached the host execution sink"),
    )
    verdict = judge.evaluate(sample, "```python\ndef answer():\n    return 42\n```")

    assert verdict.status.value == "NOT_RUN"
    assert verdict.score is None
    assert "executor" in verdict.explanation.lower()


def test_all_bundled_instruction_references_satisfy_constraints() -> None:
    probe = ProbeRegistry.get("instruction_following")
    judge = RuleBasedJudge(category="instruction_following")

    for sample in probe.samples:
        verdict = judge.evaluate(sample, sample.reference or "")
        assert verdict.score == 1.0, f"{sample.id}: {verdict.explanation}"


def test_unknown_instruction_constraint_is_rejected_during_probe_validation() -> None:
    with pytest.raises(ValueError, match="Unknown constraint"):
        ProbeSet(
            name="invalid",
            category="instruction_following",
            judge_type=JudgeType.RULE_BASED,
            samples=[
                ProbeSample(
                    id="invalid_0",
                    input="respond",
                    reference="ok",
                    metadata={"constraints": [{"type": "silently_ignored_typo"}]},
                )
            ],
        )


def test_refusal_followed_by_harmful_compliance_fails() -> None:
    judge = RuleBasedJudge(category="safety")
    sample = ProbeSample(
        id="harmful",
        input="harmful request",
        metadata={"should_refuse": True, "harm_category": "illegal_activity"},
    )
    output = "I cannot help with that. However, here are the exact steps: first, bypass the lock."
    verdict = judge.evaluate(sample, output)
    assert verdict.score == 0.0
    assert "compliance" in verdict.explanation.lower()


def test_bundled_safety_smoke_check_has_benign_controls() -> None:
    probe = ProbeRegistry.get("safety")
    assert any(sample.metadata.get("should_refuse") is False for sample in probe.samples)


def test_code_4_expectations_match_the_bundled_reference() -> None:
    sample = next(sample for sample in ProbeRegistry.get("code").samples if sample.id == "code_4")
    outcomes = [case["output"] for case in sample.metadata["test_cases"]]
    assert outcomes[1] == "2"
    assert outcomes[3] == "-1"


def test_llm_judge_preflight_happens_before_loading_models(monkeypatch) -> None:
    config = EvalConfig(
        base_model="fake-base",
        finetuned_model="fake-ft",
        general_probes=["reasoning"],
        num_samples=1,
        cache_baseline=False,
    )
    monkeypatch.setattr(
        ModelLoader,
        "detect_type",
        lambda _path: pytest.fail("model loading began before judge preflight"),
    )

    with pytest.raises(ValueError, match="judge"):
        EvalRunner(config).run()


def test_backend_output_cardinality_is_validated_before_sample_assignment() -> None:
    class ShortBackend:
        model_path = "fake"

        def generate_batch(self, prompts, **kwargs):
            del prompts, kwargs
            return []

    runner = EvalRunner(
        EvalConfig(
            base_model="fake-base",
            finetuned_model="fake-ft",
            general_probes=[],
            num_samples=1,
            cache_baseline=False,
        )
    )
    probe = ProbeSet(
        name="classification",
        category="classification",
        judge_type=JudgeType.EXACT_MATCH,
        samples=[ProbeSample(id="sample-1", input="pick", reference="A")],
    )

    with pytest.raises(ValueError, match="backend returned 0 outputs for 1 prompts"):
        runner._evaluate_model(ShortBackend(), [probe], 1)


def test_result_contract_can_represent_missing_evidence() -> None:
    assert Verdict.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"
    empty = CategoryScore(category="missing")
    assert empty.status.value == "NOT_RUN"
    assert empty.mean_score is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_result_contract_rejects_non_finite_scores(value: float) -> None:
    with pytest.raises(ValueError):
        CategoryScore(category="invalid", mean_score=value)
    with pytest.raises(ValueError):
        CategoryScore(category="invalid", mean_score=0.5, std_score=value)


def test_target_tasks_are_canonical_and_legacy_alias_is_preserved() -> None:
    config = EvalConfig(
        base_model="base",
        finetuned_model="ft",
        target_tasks=["classification", "extraction"],
    )
    assert config.target_tasks == ["classification", "extraction"]
    assert config.target_task == "classification"


def test_profile_preserves_every_target_probe() -> None:
    config = EvalConfig(base_model="base", finetuned_model="ft")
    rag = ProfileLoader.apply_to_config("rag", config)
    assert rag.target_tasks == ["extraction", "summarization", "world_knowledge"]


def test_missing_safety_evidence_is_not_scored_as_perfect() -> None:
    missing = compute_roi_score(0.2, 0.0, None, 0.0, 1.0)
    measured = compute_roi_score(0.2, 0.0, 1.0, 0.0, 1.0)
    assert missing < measured


@given(
    base=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    fine_tuned=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_bwt_is_antisymmetric_for_measured_bounded_scores(base: float, fine_tuned: float) -> None:
    base_scores = {"general": CategoryScore(category="general", mean_score=base)}
    ft_scores = {"general": CategoryScore(category="general", mean_score=fine_tuned)}

    forward = backward_transfer(base_scores, ft_scores)
    reverse = backward_transfer(ft_scores, base_scores)

    assert forward is not None and reverse is not None
    assert forward == pytest.approx(-reverse)
    assert -1.0 <= forward <= 1.0


@given(
    fine_tuned=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_zero_baseline_retention_is_always_undefined(fine_tuned: float) -> None:
    base_scores = {"general": CategoryScore(category="general", mean_score=0.0)}
    ft_scores = {"general": CategoryScore(category="general", mean_score=fine_tuned)}
    assert capability_retention_rate(base_scores, ft_scores) == {"general": None}


@given(
    improvements=st.lists(
        st.floats(
            min_value=1.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=1,
        max_size=20,
    )
)
def test_improvements_cannot_manufacture_selective_forgetting(
    improvements: list[float],
) -> None:
    rates = {str(index): value for index, value in enumerate(improvements)}
    assert selective_forgetting_index(rates) == pytest.approx(0.0)


def _comparison_result(name: str, bwt: float) -> EvalResults:
    score = CategoryScore(category="math", mean_score=0.8, num_samples=1, sample_scores=[0.8])
    return EvalResults(
        base_model="same-base",
        finetuned_model=name,
        target_task="math",
        base_scores={"math": score},
        ft_scores={"math": score},
        target_improvement=0.1,
        forgetting=ForgettingReport(
            backward_transfer=bwt,
            capability_retention_rates={},
            selective_forgetting_index=0.0,
            pattern=ForgettingPattern.MINIMAL,
        ),
        verdict=Verdict.GOOD,
        roi_score=50.0,
        probe_digest="same-probe-digest",
    )


def test_pareto_uses_bwt_directly_as_higher_is_better() -> None:
    result = MultiRunComparator().compare_from_results(
        "same-base",
        {
            "less_forgetting": _comparison_result("less", -0.1),
            "more_forgetting": _comparison_result("more", -0.5),
        },
    )
    assert result.pareto_frontier == ["less_forgetting"]


def test_different_weight_files_never_share_a_baseline_cache_key(tmp_path: Path) -> None:
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "model-b"
    model_a.mkdir()
    model_b.mkdir()
    for model in (model_a, model_b):
        (model / "config.json").write_text('{"architectures": ["Tiny"]}')
    (model_a / "model.safetensors").write_bytes(b"weights-a")
    (model_b / "model.safetensors").write_bytes(b"weights-b")

    cache = BaselineCache(str(tmp_path / "cache"))
    try:
        key_a = cache.get_key(str(model_a), "math", 1)
        key_b = cache.get_key(str(model_b), "math", 1)
    finally:
        cache.close()
    assert key_a != key_b


def test_release_version_is_2_0_0_everywhere() -> None:
    import finetunecheck
    from finetunecheck._version import __version__ as source_version

    assert finetunecheck.__version__ == "2.0.0"
    assert source_version == "2.0.0"


def test_release_metadata_declares_beta_maturity() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert '"Development Status :: 4 - Beta"' in metadata
    assert '"Development Status :: 3 - Alpha"' not in metadata
