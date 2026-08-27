# Migrating to 2.0.0

Most existing CLI and Python usage still works in 2.0.0. The important difference is that missing
or incomplete results are now reported plainly instead of being folded into a normal score. The
old single-target field remains available as an alias.

## What to update

- Prefer `target_tasks=[...]`. `target_task` still maps to the first target.
- Configure LLM judging with `JudgeConfig` or `--judge provider:model`. Evaluated models are rejected as judges.
- Handle `NOT_RUN`, `ERROR`, `INCOMPATIBLE`, `INSUFFICIENT_SAMPLE`, and `INSUFFICIENT_EVIDENCE` in code that reads results.
- Use ROI keys `target`, `retention`, `safety`, `selectivity`, and `bwt`. The old `target_improvement` and `general_retention` aliases are migrated.
- Install `finetunecheck[inference]` for Transformers/PEFT or another backend extra. Core metrics and reports no longer require the inference stack.
- Supply an external isolated `Executor` for code tests. Host Python execution is no longer available.

Result JSON now records the package and schema versions along with probe, judge, generation,
backend, execution, and cache details. BWT is higher-is-better, target change is the fine-tuned
score minus the base score, CRR is undefined when the base score is zero, and missing results do
not receive perfect points.
