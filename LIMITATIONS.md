# Limitations

FineTuneCheck is useful for finding problems, but it has some sharp edges worth knowing about:

- The bundled probes are small smoke tests, not representative benchmarks.
- LLM judges can be biased and sensitive to wording. Review important samples yourself.
- The safety check looks at refusal and over-refusal patterns; it does not prove alignment or deployment safety.
- ROUGE-L measures word overlap, not whether a summary is true.
- Confidence intervals from small samples can move around a lot and do not replace a proper power analysis.
- Cache reuse needs an immutable model revision or a complete local fingerprint. The cache directory is trusted local storage: malformed entries are discarded, but valid-looking entries are not cryptographically signed.
- Deep-analysis signals are experimental correlations. They cannot tell you what caused a regression.
- Before shipping, add domain evaluation, red teaming, human review, and production monitoring that fit your use case.
