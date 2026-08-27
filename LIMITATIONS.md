# Limitations

- Bundled probes are small smoke fixtures, not representative benchmarks.
- LLM judges can be biased or prompt-sensitive and require independent provider review.
- The safety heuristic measures refusal and over-refusal patterns, not alignment or deployment safety.
- ROUGE-L measures lexical overlap, not summarization faithfulness.
- Paired intervals on small samples are unstable and do not replace power analysis.
- Cache reuse depends on immutable revisions or complete local fingerprints. Treat the cache directory as local trusted storage; malformed or inconsistent entries are discarded, but entries are not cryptographically signed.
- Deep-analysis signals are experimental correlations and do not prove causal forgetting.
- FineTuneCheck results should be combined with domain evaluation, red teaming, human review, and production monitoring.
