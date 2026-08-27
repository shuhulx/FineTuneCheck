# FineTuneCheck

Evidence-aware comparison of base and fine-tuned language models.

[![CI](https://github.com/shuhulx/finetunecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/shuhulx/finetunecheck/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/finetunecheck)](https://pypi.org/project/finetunecheck/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

FineTuneCheck measures target-task change, general-capability retention, safety smoke behavior, and sample-level regressions. It preserves the underlying evidence and returns `INSUFFICIENT_EVIDENCE` whenever required measurements are missing, errored, incompatible, or too small for a confident verdict.

Results support investigation. They are not independently sufficient for deployment approval.

## Highlights

- Paired base-versus-fine-tuned evaluation with raw outputs and judge evidence
- Explicit local, OpenAI, Anthropic, or caller-supplied judge providers
- Fail-closed code evaluation through an external `Executor` boundary
- BWT, CRR, SFI, SAR, bounded-score target deltas, paired intervals, and ROI provenance
- Multi-target profiles and compatible multi-run Pareto comparison
- Transformers, vLLM, llama.cpp/GGUF, and local or remote PEFT adapters
- Self-contained HTML plus JSON, CSV, and Markdown reports
- Experimental CKA, rank@k spectral, sliding-window perplexity, calibration, and activation diagnostics
- Nine asynchronous MCP tools

## Install

The core package keeps configuration, metrics, caching, and reporting lightweight:

```bash
pip install finetunecheck
```

Install a model backend for evaluation:

```bash
pip install "finetunecheck[inference]"   # Transformers + PEFT
pip install "finetunecheck[vllm]"        # vLLM
pip install "finetunecheck[gguf]"        # llama.cpp / GGUF
```

Local PEFT adapter directories are detected automatically. Use `peft://ORG/ADAPTER` or
`peft://ORG/ADAPTER@REVISION` for an adapter hosted on Hugging Face.

Other extras:

```bash
pip install "finetunecheck[deep]"        # experimental deep analysis
pip install "finetunecheck[api-judge]"   # OpenAI and Anthropic judge clients
pip install "finetunecheck[mcp]"         # MCP SDK 1.x
```

## Quick check

Quick mode is an offline-runnable evaluation path: it needs local model weights but no API judge. It selects 10 cases from each of math, classification, instruction following, and safety.

```bash
ftcheck quick BASE_MODEL FINETUNED_MODEL --report quick-report.html
```

The bundled cases are small smoke probes, so the verdict will normally be `INSUFFICIENT_EVIDENCE`. That is intentional.

## Full evaluation

LLM-judged probes require a dedicated judge. FineTuneCheck never silently reuses either evaluated model.

```bash
ftcheck run BASE_MODEL FINETUNED_MODEL \
  --profile classification \
  --judge local:JUDGE_MODEL \
  --report report.html
```

API judges are explicit:

```bash
export OPENAI_API_KEY=...
ftcheck run BASE_MODEL FINETUNED_MODEL \
  --profile chat \
  --judge openai:gpt-4o-mini
```

If a required judge is missing, evaluation fails before the base or fine-tuned model is loaded. Unparseable judge output becomes `ERROR`, not a neutral score.

Code probes do not execute generated Python on the host. Without a caller-supplied isolation runtime implementing `Executor`, their status is `NOT_RUN` and the overall verdict is evidence-limited.

## Python API

```python
from finetunecheck.config import EvalConfig, JudgeConfig
from finetunecheck.eval.runner import EvalRunner
from finetunecheck.profiles.loader import ProfileLoader

config = EvalConfig(
    base_model="BASE_MODEL",
    finetuned_model="FINETUNED_MODEL",
    judge=JudgeConfig(provider="local", model="JUDGE_MODEL"),
    device="auto",
)
config = ProfileLoader.apply_to_config("classification", config)

results = EvalRunner(config).run()

print(results.verdict.value)
print(results.target_improvements)
print(results.roi_score, results.roi_coverage)
if results.forgetting:
    print(results.forgetting.backward_transfer)
```

For deterministic smoke evaluation, use `QuickConfig`:

```python
from finetunecheck.config import QuickConfig
from finetunecheck.eval.runner import EvalRunner

results = EvalRunner(
    QuickConfig(base_model="BASE_MODEL", finetuned_model="FINETUNED_MODEL")
).run()
```

`device="auto"` is preserved through Python, CLI, and MCP. The selected inference backend is recorded in result provenance.

## Evidence and verdicts

Every category carries one of these statuses:

- `MEASURED`
- `NOT_RUN`
- `ERROR`
- `INCOMPATIBLE`
- `INSUFFICIENT_SAMPLE`

Overall verdicts are `EXCELLENT`, `GOOD`, `GOOD_WITH_CONCERNS`, `POOR`, `HARMFUL`, or `INSUFFICIENT_EVIDENCE`.

Missing evidence contributes no perfect retention or safety points. Confident verdicts require complete paired measurements, target evidence, full ROI coverage, adequate sample counts, and probe provenance that supports the claim. Even a confident verdict is decision support, not deployment authorization.

## Metrics

| Metric | 2.0.0 meaning |
|---|---|
| Target delta | Fine-tuned minus base bounded score, aggregated as a macro mean across every target |
| BWT | Mean fine-tuned minus base score on non-target categories; higher is better |
| CRR | Fine-tuned/base ratio on non-target categories; undefined near a zero baseline |
| SFI | Dispersion of downside-only retention losses |
| SAR | Safety smoke-score ratio; undefined when safety evidence or its baseline is missing |
| ROI | Versioned weighted composite with component values, weights, and evidence coverage |

The `target_task` field remains as a compatibility alias for the first entry in `target_tasks`.

## Bundled probes

All bundled probes are versioned Apache-2.0 smoke fixtures, not independently validated benchmark datasets.

| Probe | Seed cases | Judge |
|---|---:|---|
| reasoning | 15 | dedicated LLM |
| code | 15 | external isolated executor |
| math | 15 | numeric equivalence |
| safety | 15 | refusal/over-refusal heuristic smoke check |
| chat_quality | 10 | dedicated LLM |
| creative_writing | 8 | dedicated LLM |
| summarization | 10 | ROUGE-L lexical overlap only |
| extraction | 10 | token F1 |
| classification | 12 | exact label |
| instruction_following | 12 | validated constraints |
| multilingual | 10 | dedicated LLM |
| world_knowledge | 15 | exact answer/alias |

Safety reports separate harmful-request refusal from benign over-refusal and detect refusal followed by apparent compliance. The heuristic is not called alignment certification and cannot satisfy the stronger safety requirement in `safety_critical`.

## Profiles

```bash
ftcheck list-profiles
ftcheck list-probes
```

Profiles: `general`, `code`, `chat`, `classification`, `rag`, `medical`, `legal`, and `safety_critical`.

Every target in a profile is evaluated and excluded consistently from retention metrics. `safety_critical` enforces measured SAR >= 0.99 and also requires stronger safety evidence than the bundled heuristic.

## Custom probes

```python
from finetunecheck.probes.custom import CustomProbe
from finetunecheck.probes.registry import ProbeRegistry

probe = CustomProbe.from_csv(
    name="domain_eval",
    csv_path="domain_eval.csv",
    category="domain",
    judge_type="exact_match",
)
ProbeRegistry.register(probe)
```

`CustomProbe.from_jsonl(...)` follows the same pattern. Use sourced, licensed, contamination-reviewed data with enough paired samples when making claims beyond smoke diagnosis.

## Reports and comparison

```bash
ftcheck run BASE_MODEL FINETUNED_MODEL \
  --profile classification \
  --judge local:JUDGE_MODEL \
  --report results.html

ftcheck compare BASE_MODEL RUN_1 RUN_2 RUN_3 \
  --profile classification \
  --judge local:JUDGE_MODEL \
  --report comparison.html
```

HTML reports embed Plotly by default and include statuses, configured ROI weights, selected sample IDs, raw outputs, judge/test evidence, and provenance. Comparison rejects runs with mismatched bases, probe digests, judges, targets, or schema versions and renders all compatible runs.

## MCP

Install `finetunecheck[mcp]`, then configure:

```json
{
  "mcpServers": {
    "finetunecheck": {
      "command": "ftcheck",
      "args": ["serve", "--stdio"]
    }
  }
}
```

The server exposes `evaluate_finetune`, `quick_check`, `detect_forgetting`, `compare_runs`, `get_verdict`, `suggest_fixes`, `generate_report`, `list_profiles`, and `run_probe`. Model work runs through a bounded asynchronous worker gate. Tool failures are protocol errors.

## Development

```bash
pip install -e ".[dev,mcp]"
ruff check .
ruff format --check .
pyright
pytest
python -m build
```

CI tests Python 3.10, 3.11, and 3.12, enforces coverage, builds the package, installs fresh wheels, verifies MCP registration, and loads a report in Chromium.

See [VALIDATION.md](VALIDATION.md), [MIGRATION.md](MIGRATION.md), and [LIMITATIONS.md](LIMITATIONS.md) before interpreting results.

## License

Apache-2.0
