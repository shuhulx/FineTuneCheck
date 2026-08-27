# Validation and methods

## Scope

FineTuneCheck runs the same prompts through a base model and a fine-tuned model, then compares
the answers. It is meant to show you where a fine-tune helped or hurt. It cannot decide whether a
model is ready to ship.

The package includes 147 small checks across 12 categories. They are Apache-2.0 project data and
are versioned with the code. They are good for smoke testing, but they have not been shown to be
representative or free from training-data overlap.

## Measurement rules

- The same selected sample IDs are assigned to both models.
- Backend output cardinality is checked before IDs are attached.
- The result keeps model output, judge output, explanations, errors, latency, and test-case outcomes.
- LLM-judged probes need a separate judge that returns valid JSON.
- Code is `NOT_RUN` unless a caller supplies an external isolation runtime.
- Safety is a refusal/over-refusal heuristic smoke check unless a stronger provider is added.
- Summarization uses ROUGE-L lexical overlap and does not measure factual faithfulness.

## Metrics

Target change is the fine-tuned score minus the base score, averaged equally across configured
targets. BWT applies the same subtraction to measured non-target categories, so higher is better.
CRR is a ratio and is undefined when the base score is effectively zero. SFI only looks at losses,
so improvements do not create fake forgetting. SAR is the ratio between the two measured safety
smoke scores.

ROI formula `roi-v2` records canonical weights, normalized component values, and coverage. Missing components contribute zero points and reduce coverage.

Regression thresholds are centralized in `finetunecheck.forgetting.metrics.REGRESSION_THRESHOLDS`. Paired 95% intervals use a normal approximation when matching raw samples exist; tiny seed sets are labeled low-confidence smoke evidence.

## Verdict gates

FineTuneCheck only returns `GOOD` or `EXCELLENT` when every required category was measured with
at least 20 paired samples, the target task was measured, the ROI is complete, and the probes are
suitable for more than a smoke test. Missing, failed, incompatible, or undersized results produce
`INSUFFICIENT_EVIDENCE`. A measured harmful or catastrophic regression can still produce
`HARMFUL` or `POOR`.

The `safety_critical` profile requires measured SAR >= 0.99 and stronger evidence than the bundled heuristic. The medical profile makes no clinical hallucination claim.

## Reproducibility

A baseline cache key records the model weights or immutable revision, tokenizer and chat template,
adapter relationship, probes and sample IDs, judge, generation settings, backend, execution
policy, package version, and result format. Mutable remote branches and incomplete local models
are not cached.

Deep analysis is experimental. It uses at most the requested count from a bundled corpus of 50 texts, checks architecture and tokenizer compatibility, passes attention masks, reports component errors, uses sliding-window perplexity, avoids a full softmax tensor for ECE, and labels truncated spectral estimates as rank@k.

## Verification

The release tests run offline: they do not download models, call paid APIs, or execute generated
code. Tiny local torch models cover the inference and deep-analysis plumbing, while deterministic
test doubles cover model and judge failures. A Chromium test opens the HTML report, checks its
charts and browser errors, and confirms that hostile text cannot run as JavaScript.

CI enforces Ruff, formatting, Pyright, Python 3.10–3.12 tests, at least 80% overall coverage, at least 90% coverage for runner, judges, metrics/verdict logic, profile loading, and cache, package build, fresh-wheel installs, MCP registration, and browser rendering.
