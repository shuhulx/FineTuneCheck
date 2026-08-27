# FineTuneCheck

Compare a fine-tuned model with its base and see what improved, what slipped, and why.

[![CI](https://github.com/shuhulx/finetunecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/shuhulx/finetunecheck/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-304%20passed-brightgreen.svg)](https://github.com/shuhulx/finetunecheck/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/finetunecheck)](https://pypi.org/project/finetunecheck/)
[![Downloads](https://static.pepy.tech/badge/finetunecheck)](https://pepy.tech/project/finetunecheck)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

FineTuneCheck answers the question that comes after training: did the fine-tune get better at
the task you care about without quietly getting worse somewhere else? It runs the same checks
against both models, keeps the outputs behind every score, and points you to the regressions
worth looking at.

When a run is incomplete or too small to judge fairly, FineTuneCheck says
`INSUFFICIENT_EVIDENCE` instead of guessing. Think of it as a practical second pair of eyes;
you should still run the domain and safety tests that matter for your use case.

## Highlights

- Side-by-side evaluation of the base and fine-tuned model, with the raw outputs kept
- Local, OpenAI, Anthropic, or caller-supplied judges—you choose which one is used
- Generated code runs only through an external `Executor` that you provide
- BWT, CRR, SFI, SAR, target deltas, paired intervals, and a fully explained ROI score
- Multi-target profiles and compatible multi-run Pareto comparison
- Transformers, vLLM, llama.cpp/GGUF, and local or remote PEFT adapters
- Self-contained HTML plus JSON, CSV, and Markdown reports
- Experimental CKA, rank@k spectral, sliding-window perplexity, calibration, and activation diagnostics
- Nine asynchronous MCP tools

## Install

The base install includes configuration, metrics, caching, and reports:

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

Quick mode stays local and does not need an API judge. It runs 10 checks each for math,
classification, instruction following, and safety.

```bash
ftcheck quick BASE_MODEL FINETUNED_MODEL --report quick-report.html
```

These bundled checks are deliberately small, so quick mode will usually return
`INSUFFICIENT_EVIDENCE`. It is meant to catch obvious problems, not hand out a release grade.

## Full evaluation

Probes that need subjective grading require a separate judge model. FineTuneCheck never asks
either model being compared to grade itself.

```bash
ftcheck run BASE_MODEL FINETUNED_MODEL \
  --profile classification \
  --judge local:JUDGE_MODEL \
  --report report.html
```

To use an API judge, name it explicitly:

```bash
export OPENAI_API_KEY=...
ftcheck run BASE_MODEL FINETUNED_MODEL \
  --profile chat \
  --judge openai:gpt-4o-mini
```

FineTuneCheck checks the judge setup before loading either evaluated model. If the judge returns
something it cannot parse, that sample is marked `ERROR` rather than quietly receiving an
average score.

Code probes do not run generated Python directly on your machine. Pass an isolated `Executor`
if you want execution-based scoring; otherwise those probes are marked `NOT_RUN`.

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

results = EvalRunner(QuickConfig(base_model="BASE_MODEL", finetuned_model="FINETUNED_MODEL")).run()
```

`device="auto"` works the same way in Python, the CLI, and MCP. Results also record which
inference backend actually ran.

## Evidence and verdicts

Every category tells you whether it actually ran:

- `MEASURED`
- `NOT_RUN`
- `ERROR`
- `INCOMPATIBLE`
- `INSUFFICIENT_SAMPLE`

Overall verdicts are `EXCELLENT`, `GOOD`, `GOOD_WITH_CONCERNS`, `POOR`, `HARMFUL`, or `INSUFFICIENT_EVIDENCE`.

Missing results never turn into free retention or safety points. A strong verdict needs complete
paired measurements, enough samples, a measured target task, and full ROI coverage. Whatever
the verdict says, read the failed samples before shipping a model.

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

FineTuneCheck ships with 147 small checks across 12 categories. They are useful for smoke
testing and examples, but they are not a replacement for a real benchmark built around your
data and users.

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

The safety check separates refusing harmful requests from refusing harmless ones, and it catches
answers that start with a refusal but then provide the harmful instructions anyway. It is still a
small heuristic check, not proof that a model is safe.

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

`CustomProbe.from_jsonl(...)` works the same way. For serious evaluation, use data you have the
right to use, check it for leakage, and include enough paired examples to make the result useful.

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

HTML reports include the status of every check, ROI weights, selected sample IDs, raw outputs,
judge details, and run metadata. Multi-run comparison refuses to mix results that used different
bases, probes, judges, targets, or result formats.

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

The server exposes `evaluate_finetune`, `quick_check`, `detect_forgetting`, `compare_runs`,
`get_verdict`, `suggest_fixes`, `generate_report`, `list_profiles`, and `run_probe`. Model work is
moved off the MCP event loop and limited to two jobs at a time. Failures come back as normal MCP
tool errors.

## Development

```bash
pip install -e ".[dev,mcp]"
ruff check .
ruff format --check .
pyright
pytest
python -m build
```

CI runs on Python 3.10, 3.11, and 3.12. It checks formatting and types, enforces coverage, builds
and installs the wheel from scratch, verifies all nine MCP tools, and opens a report in Chromium.

For the exact metric rules and known rough edges, see [VALIDATION.md](VALIDATION.md),
[MIGRATION.md](MIGRATION.md), and [LIMITATIONS.md](LIMITATIONS.md).

## License

Apache-2.0. See [LICENSE](LICENSE) for the full text.
