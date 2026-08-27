# Validation and methods

## Scope

FineTuneCheck compares paired outputs from a base model and a fine-tuned model. It is a diagnostic framework, not an independently validated deployment-approval system.

The bundled probe files contain 147 small seed cases across 12 categories. They are Apache-2.0 project fixtures, versioned with the package, and marked as smoke evidence. They have not been established as representative, contamination-free benchmark datasets.

## Measurement rules

- The same selected sample IDs are assigned to both models.
- Backend output cardinality is checked before IDs are attached.
- Raw model output, judge output, explanations, errors, latency, and test-case outcomes are retained.
- LLM judgments require an explicit dedicated provider and schema-valid JSON.
- Code is `NOT_RUN` unless a caller supplies an external isolation runtime.
- Safety is a refusal/over-refusal heuristic smoke check unless a stronger provider is added.
- Summarization uses ROUGE-L lexical overlap and does not measure factual faithfulness.

## Metrics

Target change is the fine-tuned-minus-base bounded-score delta, macro-averaged across configured targets. BWT is the mean fine-tuned-minus-base bounded-score delta on measured non-target categories and is higher-is-better. CRR is a ratio and is undefined when the base score is effectively zero. SFI uses downside-only loss so improvements do not create artificial forgetting. SAR is a ratio of measured safety smoke scores.

ROI formula `roi-v2` records canonical weights, normalized component values, and coverage. Missing components contribute zero points and reduce coverage.

Regression thresholds are centralized in `finetunecheck.forgetting.metrics.REGRESSION_THRESHOLDS`. Paired 95% intervals use a normal approximation when matching raw samples exist; tiny seed sets are labeled low-confidence smoke evidence.

## Verdict gates

A confident `GOOD` or `EXCELLENT` verdict requires complete measured evidence, at least 20 paired samples per required category, target evidence, full ROI coverage, and probe provenance that supports release claims. Missing, errored, incompatible, undersized, or smoke-only evidence produces `INSUFFICIENT_EVIDENCE` unless a measured harmful/catastrophic condition requires `HARMFUL` or `POOR`.

The `safety_critical` profile requires measured SAR >= 0.99 and stronger evidence than the bundled heuristic. The medical profile makes no clinical hallucination claim.

## Reproducibility and provenance

Baseline cache entries include a strong local weight fingerprint or immutable remote revision; tokenizer/chat template, adapter/base relationship, probe content and sample IDs, judge configuration, generation settings, inference backend, execution policy, package version, and schema versions. Mutable remote aliases and incomplete local identities are not cached.

Deep analysis is experimental. It uses at most the requested count from a bundled corpus of 50 texts, checks architecture and tokenizer compatibility, passes attention masks, reports component errors, uses sliding-window perplexity, avoids a full softmax tensor for ECE, and labels truncated spectral estimates as rank@k.

## Verification

The release suite is offline: no model download, paid API call, or candidate-code execution is permitted. Tiny locally constructed torch models exercise inference and deep-analysis mechanics; deterministic test doubles exercise model/judge boundary failures. A Chromium test loads the self-contained HTML report, checks rendered Plotly traces, records console/page errors, and verifies adversarial text cannot execute script.

CI enforces Ruff, formatting, Pyright, Python 3.10–3.12 tests, at least 80% overall coverage, at least 90% coverage for runner, judges, metrics/verdict logic, profile loading, and cache, package build, fresh-wheel installs, MCP registration, and browser rendering.
