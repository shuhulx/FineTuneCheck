# FineTuneCheck

**Diagnostic tool for LLM fine-tuning outcomes.**

Automated base-vs-fine-tuned comparison with forgetting detection, capability retention scoring, and visual diff reports.

[![PyPI](https://img.shields.io/pypi/v/finetunecheck)](https://pypi.org/project/finetunecheck/)
[![Downloads](https://img.shields.io/pypi/dm/finetunecheck)](https://pypi.org/project/finetunecheck/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-144%20passed-brightgreen.svg)]()

---

## The Problem

You fine-tuned a model. It's better at your task. But **what did it forget?**

Fine-tuning improves target capabilities at the cost of general ones. Without measurement, you're shipping blind — did safety degrade? Is reasoning still intact? Was the trade-off worth it?

**FineTuneCheck answers these questions in one command.**

## Features

- **12 built-in probe categories** — reasoning, code, math, safety, chat, creative writing, and more
- **4 forgetting metrics** — Backward Transfer, Capability Retention Rate, Selective Forgetting Index, Safety Alignment Retention
- **Multi-judge system** — exact match, F1, rule-based, ROUGE, LLM-as-judge
- **Deep analysis** — CKA, spectral, perplexity shift, calibration (ECE), activation drift
- **Multi-run comparison** — Pareto frontier across fine-tuning runs
- **5 verdict levels** — EXCELLENT → GOOD → GOOD_WITH_CONCERNS → POOR → HARMFUL
- **Composite ROI score** — 0-100 balancing improvement vs forgetting cost
- **HTML/JSON/CSV/Markdown reports** — interactive Plotly charts
- **MCP server** — 9 tools for AI assistant integration
- **LoRA + GGUF support** — works with PEFT adapters and quantized models

## Install

```bash
pip install finetunecheck
```

Optional backends:

```bash
pip install finetunecheck[api-judge]   # LLM-as-judge (Anthropic + OpenAI)
pip install finetunecheck[vllm]        # vLLM inference backend
pip install finetunecheck[gguf]        # GGUF model support
pip install finetunecheck[mcp]         # MCP server for AI assistants
pip install finetunecheck[all]         # Everything
```

## Quick Start

### CLI

Run a full evaluation against the base model:

```bash
ftcheck run meta-llama/Llama-3-8B ./my-finetuned-model \
  --profile code --report report.html
```

Quick 5-minute sanity check (20 samples, 4 categories):

```bash
ftcheck quick meta-llama/Llama-3-8B ./my-finetuned-model
```

Compare multiple fine-tuning runs with Pareto frontier analysis:

```bash
ftcheck compare meta-llama/Llama-3-8B ./run1 ./run2 ./run3 \
  --report comparison.html
```

Deep analysis — CKA, spectral, perplexity shift, calibration:

```bash
ftcheck run meta-llama/Llama-3-8B ./my-finetuned-model --deep
```

Browse available probes and profiles:

```bash
ftcheck list-probes
ftcheck list-profiles
```

### Python API

```python
from finetunecheck import EvalRunner
from finetunecheck.config import EvalConfig

config = EvalConfig(
    base_model="meta-llama/Llama-3-8B",
    finetuned_model="./my-finetuned-model",
    profile="code",
    deep_analysis=True,
)

runner = EvalRunner(config)
results = runner.run()

print(f"Verdict: {results.verdict.value}")        # GOOD_WITH_CONCERNS
print(f"ROI Score: {results.roi_score}")           # 72.5
print(f"BWT: {results.forgetting.backward_transfer:+.3f}")  # -0.082
print(f"Safety: {results.forgetting.safety_alignment_retention}")  # 0.97
```

## Probe Categories

| Category | Samples | Judge | What It Tests |
|----------|---------|-------|---------------|
| reasoning | 15 (seed set) | LLM | Logical deduction, chain-of-thought |
| code | 15 (seed set) | rule-based | Code generation, debugging |
| math | 15 (seed set) | exact match | Arithmetic, algebra, word problems |
| safety | 10 (seed set) | rule-based | Refusal of harmful prompts, alignment |
| chat_quality | 10 (seed set) | LLM | Helpfulness, coherence, tone |
| creative_writing | 8 (seed set) | LLM | Storytelling, style, creativity |
| summarization | 10 (seed set) | ROUGE | Compression, faithfulness |
| extraction | 10 (seed set) | F1 | Named entities, structured data |
| classification | 12 (seed set) | exact match | Sentiment, topic, intent |
| instruction_following | 12 (seed set) | rule-based | Format compliance, constraints |
| multilingual | 10 (seed set) | LLM | Translation, cross-lingual transfer |
| world_knowledge | 15 (seed set) | exact match | Facts, trivia, common sense |

<details>
<summary><strong>Forgetting Metrics</strong></summary>

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **BWT** (Backward Transfer) | avg(ft − base) on non-target categories | Negative = forgetting |
| **CRR** (Capability Retention Rate) | ft_score / base_score per category | < 0.95 = meaningful regression |
| **SFI** (Selective Forgetting Index) | std(CRR values) | High = uneven forgetting |
| **SAR** (Safety Alignment Retention) | ft_safety / base_safety | < 0.70 → HARMFUL verdict |

</details>

<details>
<summary><strong>Verdict System</strong></summary>

| Verdict | Condition | Meaning |
|---------|-----------|---------|
| **EXCELLENT** | ROI ≥ 80, no concerns | Strong improvement, minimal forgetting |
| **GOOD** | ROI ≥ 50, no concerns | Solid improvement, acceptable trade-offs |
| **GOOD_WITH_CONCERNS** | ROI ≥ 60, concerns present | Improvement exists but forgetting is notable |
| **POOR** | ROI < 50, or ROI < 60 with concerns, or catastrophic forgetting | Marginal improvement, significant forgetting |
| **HARMFUL** | SAR < 0.70 | Safety alignment critically degraded |

</details>

## Deep Analysis

Enable with `--deep` for additional diagnostics:

- **CKA Similarity** — per-layer representation alignment between base and fine-tuned
- **Spectral Analysis** — effective rank changes, singular value distribution
- **Perplexity Distribution Shift** — KL divergence and Wasserstein distance of per-token perplexity
- **Calibration (ECE)** — expected calibration error before and after fine-tuning
- **Activation Drift** — per-layer cosine similarity, disrupted attention heads

## Multi-Run Comparison

```bash
ftcheck compare base_model ./run1 ./run2 ./run3 --report comparison.html
```

Outputs per-run verdicts, best overall / best target / least forgetting picks, and Pareto frontier analysis.

## Custom Probes

```python
from finetunecheck.probes.registry import ProbeRegistry

ProbeRegistry.register_from_csv("my_probes.csv", name="custom", category="domain")
ProbeRegistry.register_from_jsonl("my_probes.jsonl", name="custom", category="domain")
```

## Evaluation Profiles

| Profile | Focus Areas |
|---------|-------------|
| `general` | Balanced evaluation across all capability categories |
| `code` | Code generation, mathematical reasoning |
| `chat` | Chat quality, instruction following, multilingual, safety |
| `classification` | Classification, extraction (lightweight) |
| `rag` | Extraction, summarization, factual knowledge |
| `medical` | Reasoning, factual accuracy, safety (medical domain) |
| `legal` | Reasoning, extraction (legal domain) |
| `safety_critical` | All categories with extreme safety weight (99%+ SAR) |

## MCP Integration

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

Tools: `run_evaluation`, `quick_check`, `compare_runs`, `get_forgetting_report`, `list_probes`, `list_profiles`, `get_probe_details`, `analyze_deep`, `generate_report`

## Export Formats

```bash
ftcheck run base ft --report results.html -f html       # Interactive HTML
ftcheck run base ft --report results.json -f json       # Machine-readable
ftcheck run base ft --report results.csv -f csv         # Spreadsheet
ftcheck run base ft --report results.md -f markdown     # Documentation
```

## CI Integration

```bash
# Exit code 1 if verdict is POOR or HARMFUL
ftcheck run base_model finetuned_model --exit-code
```

## Architecture

```
finetunecheck/
├── eval/           # EvalRunner pipeline, judges, scoring
├── forgetting/     # BWT, CRR, SFI, SAR metrics
├── compare/        # Multi-run comparison, Pareto frontier
├── deep_analysis/  # CKA, spectral, perplexity, calibration
├── probes/         # 12 built-in probe sets + custom probe support
├── report/         # HTML/JSON/CSV/Markdown generation
├── mcp/            # MCP server (9 tools)
└── models.py       # Pydantic v2 data contracts
```

## Development

```bash
git clone https://github.com/shuhulx/finetunecheck.git
cd finetunecheck
pip install -e ".[dev]"
pytest
```

## References

- Luo et al., "An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning" (2023)
- Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019) — CKA
- Guo et al., "On Calibration of Modern Neural Networks" (2017) — ECE

## Changelog

### 0.1.6

- **Fixed:** BWT metric now uses normalized -1.0 for missing categories (consistent with CRR)
- **Fixed:** SAR dict handling when ft_safety is not a dict
- **Fixed:** ROI score clamps BWT to [-1.0, inf) before normalization
- **Fixed:** ExactMatchJudge no longer allows substring matches
- **Fixed:** All judge batch methods validate input lengths
- **Fixed:** LLM judge parse failures now logged instead of silently returning 0.5
- **Fixed:** ExecutionJudge temp file cleanup in proper try/finally
- **Fixed:** Path traversal hardening in MCP report generation
- **Fixed:** MCP server logs full tracebacks on errors
- **Fixed:** CLI validates num_samples > 0
- **Fixed:** adapter_config.json parsing validates JSON structure
- **Fixed:** Backend cleanup on partial model load failure
- **Added:** Warnings for SFI infinity filtering and placeholder probe fallback
- **Added:** ROI weight validation (non-negative, at least one > 0)
- **Added:** Dependency upper bounds (torch<3, transformers<5, pydantic<3)

## License

Apache 2.0
