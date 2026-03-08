"""Deep analysis orchestrator — coordinates all analysis modules.

Runs perplexity, CKA, spectral, calibration, and activation drift analyses
with sensible defaults and a built-in reference corpus for out-of-the-box use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from finetunecheck.deep_analysis.activation import ActivationDriftAnalyzer
from finetunecheck.deep_analysis.calibration import CalibrationAnalyzer
from finetunecheck.deep_analysis.perplexity import PerplexityAnalyzer
from finetunecheck.deep_analysis.representation import CKAAnalyzer
from finetunecheck.deep_analysis.spectral import SpectralAnalyzer
from finetunecheck.models import DeepAnalysisReport

if TYPE_CHECKING:
    from finetunecheck.utils.model_loader import AnalysisModel

logger = logging.getLogger(__name__)

# Built-in diverse reference corpus for perplexity and calibration analysis.
# Covers science, history, technology, medicine, law, daily life, code docs, fiction.
# Each entry is 2-3 sentences to provide enough context for meaningful perplexity.
REFERENCE_TEXTS: list[str] = [
    # Science
    "The speed of light in a vacuum is approximately 299,792 kilometers per second. This constant is fundamental to both special and general relativity.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight. This process occurs primarily in the chloroplasts of plant cells.",
    "DNA replication is semiconservative, meaning each new double helix contains one original and one new strand. Errors during replication are corrected by proofreading enzymes.",
    "Quantum entanglement allows particles to be correlated regardless of the distance separating them. Measurement of one particle instantaneously affects the state of the other.",
    "The Krebs cycle generates electron carriers that drive oxidative phosphorylation in mitochondria. This process is the primary source of ATP in aerobic organisms.",
    # History
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and established the modern concept of state sovereignty. It marked the beginning of the nation-state system in Europe.",
    "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing through mechanization. Steam power and textile mills were among the earliest innovations.",
    "The fall of Constantinople in 1453 marked the end of the Byzantine Empire. Ottoman forces under Sultan Mehmed II captured the city after a prolonged siege.",
    "The French Revolution of 1789 overthrew the monarchy and established a republic based on Enlightenment principles. The Declaration of the Rights of Man proclaimed liberty and equality.",
    "The Silk Road facilitated trade and cultural exchange between East Asia and the Mediterranean for over a millennium. Goods, ideas, and technologies flowed along its network of routes.",
    # Technology
    "Transformer architectures use self-attention mechanisms to process sequential data in parallel. This approach has become the foundation of modern large language models.",
    "Version control systems like Git track changes to source code over time. Branching and merging allow multiple developers to work on the same codebase simultaneously.",
    "Containerization with Docker packages applications with their dependencies into isolated units. This ensures consistent behavior across development, testing, and production environments.",
    "TCP/IP is the foundational protocol suite of the internet, handling packet routing and reliable delivery. HTTP runs on top of TCP to enable web communication.",
    "Solid-state drives use flash memory to store data without moving parts. They offer significantly faster read and write speeds compared to traditional hard disk drives.",
    # Medicine
    "Antibiotics target bacterial infections but are ineffective against viruses. Overuse of antibiotics has led to the emergence of drug-resistant bacterial strains.",
    "The human immune system consists of innate and adaptive components. T cells and B cells are key players in the adaptive immune response.",
    "Magnetic resonance imaging uses strong magnetic fields and radio waves to produce detailed images of internal body structures. It is particularly useful for soft tissue visualization.",
    "Insulin regulates blood glucose levels by facilitating cellular uptake of sugar. Type 1 diabetes results from autoimmune destruction of insulin-producing beta cells.",
    "Vaccination works by exposing the immune system to a weakened or inactivated pathogen. This primes the body to mount a rapid response upon future exposure.",
    # Law
    "The principle of habeas corpus protects individuals from unlawful detention by requiring a court hearing. It is considered a fundamental safeguard of personal liberty.",
    "Contract law requires offer, acceptance, and consideration for a binding agreement. Breach of contract may result in damages or specific performance remedies.",
    "The presumption of innocence places the burden of proof on the prosecution in criminal cases. The standard of proof is typically beyond a reasonable doubt.",
    "Intellectual property law encompasses patents, trademarks, copyrights, and trade secrets. Each provides different protections for creative works and inventions.",
    "Administrative law governs the activities of government agencies and their regulatory powers. Judicial review ensures that agency actions comply with statutory authority.",
    # Daily life
    "Regular physical exercise reduces the risk of cardiovascular disease and improves mental health. The recommended minimum is 150 minutes of moderate activity per week.",
    "Balanced nutrition requires adequate intake of macronutrients and micronutrients. Whole grains, fruits, vegetables, and lean proteins form the basis of a healthy diet.",
    "Sleep quality affects cognitive function, mood regulation, and physical recovery. Most adults require between seven and nine hours of sleep per night for optimal health.",
    "Public transportation systems reduce traffic congestion and carbon emissions in urban areas. Investment in rail and bus networks benefits both commuters and the environment.",
    "Financial literacy includes understanding budgeting, saving, investing, and debt management. Compound interest can significantly grow savings over long time horizons.",
    # Code documentation
    "The map function applies a given function to each element of an iterable and returns a new iterable. It is commonly used for data transformation in functional programming.",
    "Unit tests verify the correctness of individual functions or methods in isolation. A good test suite covers edge cases and expected failure modes.",
    "REST APIs use HTTP methods to perform CRUD operations on resources identified by URLs. GET retrieves data while POST creates new resources on the server.",
    "Garbage collection automatically reclaims memory occupied by objects that are no longer referenced. Different algorithms trade off between latency and throughput.",
    "Hash tables provide average-case constant-time lookup by mapping keys to array indices via a hash function. Collisions are handled through chaining or open addressing.",
    # Fiction-style prose
    "The old lighthouse keeper climbed the spiral staircase each evening to light the beacon. Ships in the fog depended on its steady glow to navigate the rocky coast.",
    "Rain drummed against the window as she opened the letter that had traveled across two continents. The handwriting was unmistakable, though the words inside were unexpected.",
    "The forest floor was carpeted with fallen leaves in shades of amber and crimson. A narrow path wound between ancient oaks whose branches formed a cathedral ceiling overhead.",
    "The clock tower struck midnight as the last train pulled out of the station. A single figure remained on the platform, clutching a battered suitcase and a faded photograph.",
    "Waves crashed against the sea wall, sending plumes of spray into the winter air. The fishing boats rocked in the harbor, their masts tracing arcs against the grey sky.",
    # Mathematics
    "The fundamental theorem of calculus links differentiation and integration. It states that integration can be reversed by differentiation under mild continuity conditions.",
    "Prime numbers are natural numbers greater than one that have no positive divisors other than one and themselves. Their distribution among the integers follows irregular patterns.",
    "Matrix multiplication is not commutative in general, meaning AB does not necessarily equal BA. However, it is associative, so A(BC) always equals (AB)C.",
    "Bayesian inference updates prior beliefs using observed evidence through Bayes' theorem. The posterior probability is proportional to the product of the likelihood and the prior.",
    "The central limit theorem states that the sum of many independent random variables tends toward a normal distribution. This holds regardless of the underlying distribution shape.",
    # Engineering
    "Reinforced concrete combines the compressive strength of concrete with the tensile strength of steel. This composite material is used extensively in modern construction.",
    "Feedback control systems use sensor measurements to adjust actuator outputs and maintain desired setpoints. PID controllers are the most common implementation in industrial automation.",
    "Heat exchangers transfer thermal energy between two fluid streams without mixing them. Counter-flow configurations achieve higher efficiency than parallel-flow designs.",
    "Semiconductor devices exploit the properties of materials with conductivity between metals and insulators. Doping with impurities creates n-type and p-type regions essential for transistors.",
    "Aerodynamic lift is generated by pressure differences between the upper and lower surfaces of a wing. The angle of attack and airfoil shape both influence lift magnitude.",
]


class DeepAnalysisOrchestrator:
    """Coordinates all deep analysis modules.

    Runs enabled analyses and assembles a unified DeepAnalysisReport.
    Uses the built-in reference corpus by default, or accepts a custom one.
    """

    def __init__(
        self,
        num_samples: int = 256,
        batch_size: int = 16,
        enable_perplexity: bool = True,
        enable_cka: bool = True,
        enable_spectral: bool = True,
        enable_calibration: bool = True,
        enable_activation: bool = True,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.enable_perplexity = enable_perplexity
        self.enable_cka = enable_cka
        self.enable_spectral = enable_spectral
        self.enable_calibration = enable_calibration
        self.enable_activation = enable_activation

    def run(
        self,
        base_model: AnalysisModel,
        ft_model: AnalysisModel,
        reference_texts: list[str] | None = None,
    ) -> DeepAnalysisReport:
        """Run all enabled deep analysis modules.

        Args:
            base_model: AnalysisModel for the base model.
            ft_model: AnalysisModel for the fine-tuned model.
            reference_texts: Optional custom reference corpus. Uses built-in if None.

        Returns:
            DeepAnalysisReport aggregating all enabled analyses.
        """
        if reference_texts is None:
            texts = REFERENCE_TEXTS[: self.num_samples]
        else:
            texts = reference_texts[: self.num_samples]

        if not texts:
            raise ValueError(
                "No reference texts available. Provide reference_texts or use the built-in corpus."
            )

        report = DeepAnalysisReport()

        # --- Spectral analysis (weight-space only, no forward pass) ---
        if self.enable_spectral:
            logger.info("Running spectral analysis...")
            try:
                analyzer = SpectralAnalyzer()
                report.spectral = analyzer.analyze(base_model, ft_model)
            except Exception:
                logger.exception("Spectral analysis failed.")

        # --- Perplexity distribution shift ---
        if self.enable_perplexity:
            logger.info("Running perplexity distribution analysis...")
            try:
                analyzer = PerplexityAnalyzer(batch_size=self.batch_size)
                report.perplexity = analyzer.analyze(base_model, ft_model, texts)
            except Exception:
                logger.exception("Perplexity analysis failed.")

        # --- Calibration shift ---
        if self.enable_calibration:
            logger.info("Running calibration analysis...")
            try:
                analyzer = CalibrationAnalyzer(batch_size=self.batch_size)
                report.calibration = analyzer.analyze(base_model, ft_model, texts)
            except Exception:
                logger.exception("Calibration analysis failed.")

        # --- CKA representation similarity ---
        if self.enable_cka:
            logger.info("Running CKA representation analysis...")
            try:
                analyzer = CKAAnalyzer(
                    num_samples=min(self.num_samples, 256),
                    batch_size=self.batch_size,
                )
                report.cka = analyzer.analyze(base_model, ft_model, texts)
            except Exception:
                logger.exception("CKA analysis failed.")

        # --- Activation drift + head disruption ---
        if self.enable_activation:
            logger.info("Running activation drift analysis...")
            try:
                analyzer = ActivationDriftAnalyzer(
                    num_samples=min(self.num_samples, 256),
                    batch_size=self.batch_size,
                )
                report.activation = analyzer.analyze(base_model, ft_model, texts)
            except Exception:
                logger.exception("Activation drift analysis failed.")

        logger.info("Deep analysis complete.")
        return report
