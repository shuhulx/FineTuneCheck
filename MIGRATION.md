# Migrating to 2.0.0

FineTuneCheck 2.0.0 keeps the main CLI and Python entry points. It makes result semantics stricter and preserves the old single-target name as an alias.

## What to update

- Prefer `target_tasks=[...]`. `target_task` still maps to the first target.
- Configure LLM judging with `JudgeConfig` or `--judge provider:model`. Evaluated models are rejected as judges.
- Treat `NOT_RUN`, `ERROR`, `INCOMPATIBLE`, `INSUFFICIENT_SAMPLE`, and `INSUFFICIENT_EVIDENCE` explicitly.
- Use ROI keys `target`, `retention`, `safety`, `selectivity`, and `bwt`. The old `target_improvement` and `general_retention` aliases are migrated.
- Install `finetunecheck[inference]` for Transformers/PEFT or another backend extra. Core metrics and reports no longer require the inference stack.
- Supply an external isolated `Executor` for code tests. Host Python execution is no longer available.

Result JSON now includes package, result-schema, metric-schema, probe, judge, generation, backend, execution, and cache provenance. BWT is higher-is-better, target change is a fine-tuned-minus-base bounded-score delta, zero-baseline CRR is undefined, and missing evidence receives no perfect points.
