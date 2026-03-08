"""Deep analysis modules for fine-tuning evaluation.

Provides five analysis techniques:
- PerplexityAnalyzer: Per-sample perplexity distribution shift detection
- CKAAnalyzer: Layer-by-layer representation similarity (CKA)
- SpectralAnalyzer: Weight-space SVD of fine-tuning perturbations
- CalibrationAnalyzer: Token-level confidence calibration shift
- ActivationDriftAnalyzer: Hidden state drift and attention head disruption

Use DeepAnalysisOrchestrator to run all analyses in a coordinated fashion.
"""

from finetunecheck.deep_analysis.activation import ActivationDriftAnalyzer
from finetunecheck.deep_analysis.calibration import CalibrationAnalyzer
from finetunecheck.deep_analysis.orchestrator import DeepAnalysisOrchestrator
from finetunecheck.deep_analysis.perplexity import PerplexityAnalyzer
from finetunecheck.deep_analysis.representation import CKAAnalyzer
from finetunecheck.deep_analysis.spectral import SpectralAnalyzer

__all__ = [
    "ActivationDriftAnalyzer",
    "CKAAnalyzer",
    "CalibrationAnalyzer",
    "DeepAnalysisOrchestrator",
    "PerplexityAnalyzer",
    "SpectralAnalyzer",
]
