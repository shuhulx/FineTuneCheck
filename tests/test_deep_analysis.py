"""Tests for deep analysis modules: CKA, spectral, calibration, perplexity."""

import numpy as np
import pytest
import torch

from finetunecheck.deep_analysis.calibration import CalibrationAnalyzer
from finetunecheck.deep_analysis.perplexity import PerplexityAnalyzer
from finetunecheck.deep_analysis.representation import CKAAnalyzer
from finetunecheck.deep_analysis.spectral import SpectralAnalyzer


class TestLinearCKA:
    def test_linear_cka_identical(self):
        """CKA of identical matrices should be 1.0."""
        X = torch.randn(100, 64)
        score = CKAAnalyzer.linear_cka(X, X)
        assert abs(score - 1.0) < 1e-5

    def test_linear_cka_scaled(self):
        """CKA should be invariant to isotropic scaling."""
        X = torch.randn(100, 64)
        Y = X * 3.7
        score = CKAAnalyzer.linear_cka(X, Y)
        assert abs(score - 1.0) < 1e-4

    def test_linear_cka_similar(self):
        """CKA of similar matrices (with small noise) should be high."""
        torch.manual_seed(42)
        X = torch.randn(100, 64)
        Y = X + 0.1 * torch.randn(100, 64)
        score = CKAAnalyzer.linear_cka(X, Y)
        assert score > 0.8

    def test_linear_cka_different(self):
        """CKA of unrelated random matrices should be low."""
        torch.manual_seed(0)
        X = torch.randn(200, 64)
        torch.manual_seed(999)
        Y = torch.randn(200, 64)
        score = CKAAnalyzer.linear_cka(X, Y)
        assert score < 0.5

    def test_linear_cka_mismatched_features(self):
        """CKA should work with different feature dimensions."""
        X = torch.randn(100, 32)
        Y = torch.randn(100, 64)
        score = CKAAnalyzer.linear_cka(X, Y)
        assert 0.0 <= score <= 1.0

    def test_linear_cka_mismatched_samples(self):
        """CKA should raise when sample counts differ."""
        X = torch.randn(50, 64)
        Y = torch.randn(100, 64)
        with pytest.raises(AssertionError, match="Sample counts"):
            CKAAnalyzer.linear_cka(X, Y)

    def test_linear_cka_zero_matrix(self):
        """CKA with zero matrix should return 0.0."""
        X = torch.zeros(100, 64)
        Y = torch.randn(100, 64)
        score = CKAAnalyzer.linear_cka(X, Y)
        assert score == 0.0

    def test_linear_cka_symmetry(self):
        """CKA(X, Y) should equal CKA(Y, X)."""
        torch.manual_seed(7)
        X = torch.randn(80, 32)
        Y = torch.randn(80, 32)
        assert abs(CKAAnalyzer.linear_cka(X, Y) - CKAAnalyzer.linear_cka(Y, X)) < 1e-5

    def test_linear_cka_range(self):
        """CKA should always be in [0, 1]."""
        for seed in range(5):
            torch.manual_seed(seed)
            X = torch.randn(50, 16)
            Y = torch.randn(50, 16)
            score = CKAAnalyzer.linear_cka(X, Y)
            assert 0.0 <= score <= 1.0 + 1e-6


class TestEffectiveRank:
    def test_effective_rank_identity(self):
        """Effective rank of uniform SVs should be full rank."""
        svs = torch.ones(10)
        rank = SpectralAnalyzer.effective_rank(svs)
        assert abs(rank - 10.0) < 0.1

    def test_effective_rank_single(self):
        """Effective rank of single dominant SV should be ~1."""
        svs = torch.tensor([10.0, 0.01, 0.01, 0.01])
        rank = SpectralAnalyzer.effective_rank(svs)
        assert rank < 2.0

    def test_effective_rank_zeros(self):
        """Effective rank of zero SVs should be 0."""
        svs = torch.zeros(10)
        rank = SpectralAnalyzer.effective_rank(svs)
        assert rank == 0.0

    def test_effective_rank_two_equal(self):
        """Two equal SVs should give effective rank ~2."""
        svs = torch.tensor([1.0, 1.0, 1e-12, 1e-12])
        rank = SpectralAnalyzer.effective_rank(svs)
        assert abs(rank - 2.0) < 0.1

    def test_effective_rank_decreasing(self):
        """More spread SVs should have higher effective rank."""
        svs_concentrated = torch.tensor([10.0, 0.1, 0.01])
        svs_spread = torch.tensor([3.0, 3.0, 3.0])
        rank_c = SpectralAnalyzer.effective_rank(svs_concentrated)
        rank_s = SpectralAnalyzer.effective_rank(svs_spread)
        assert rank_s > rank_c

    def test_effective_rank_nonnegative(self):
        """Effective rank should always be >= 0."""
        for _ in range(10):
            svs = torch.rand(20)
            rank = SpectralAnalyzer.effective_rank(svs)
            assert rank >= 0.0


class TestSpectralAnalyzer:
    def test_analyze_weight_delta_unchanged(self):
        """Zero delta should have zero Frobenius norm and zero effective rank."""
        analyzer = SpectralAnalyzer()
        w = torch.randn(64, 32)
        result = analyzer.analyze_weight_delta("test", w, w)
        assert result["frobenius_norm"] < 1e-6
        assert result["effective_rank"] == 0.0

    def test_analyze_weight_delta_rank1(self):
        """Rank-1 perturbation should have effective rank ~1."""
        analyzer = SpectralAnalyzer()
        w_base = torch.zeros(64, 32)
        u = torch.randn(64, 1)
        v = torch.randn(1, 32)
        w_ft = w_base + u @ v
        result = analyzer.analyze_weight_delta("test", w_base, w_ft)
        assert result["effective_rank"] < 1.5
        assert result["frobenius_norm"] > 0


class TestCalibrationAnalyzer:
    def test_ece_perfect_calibration(self):
        """ECE should be ~0 for perfectly calibrated predictions."""
        cal = CalibrationAnalyzer()
        # Create samples where accuracy matches confidence per bin
        rng = np.random.default_rng(42)
        confidences = []
        correctness = []
        for bin_center in np.arange(0.05, 1.0, 0.1):
            n = 200
            confs = rng.uniform(bin_center - 0.05, bin_center + 0.05, size=n)
            # Make accuracy match the bin center
            correct = (rng.random(n) < bin_center).astype(float)
            confidences.extend(confs)
            correctness.extend(correct)

        confidences = np.array(confidences)
        correctness = np.array(correctness)
        ece, _, _, _ = cal.compute_ece(confidences, correctness)
        assert ece < 0.05

    def test_ece_overconfident(self):
        """ECE should be high when model is always confident but often wrong."""
        cal = CalibrationAnalyzer()
        confidences = np.array([0.95] * 100)
        correctness = np.array([1] * 50 + [0] * 50)  # 50% accuracy at 95% confidence
        ece, _, _, _ = cal.compute_ece(confidences, correctness)
        assert ece > 0.3

    def test_ece_underconfident(self):
        """ECE should be high when model is uncertain but always correct."""
        cal = CalibrationAnalyzer()
        confidences = np.array([0.1] * 100)
        correctness = np.ones(100)  # 100% accuracy at 10% confidence
        ece, _, _, _ = cal.compute_ece(confidences, correctness)
        assert ece > 0.5

    def test_ece_empty(self):
        """ECE of empty arrays should be 0."""
        cal = CalibrationAnalyzer()
        ece, acc, conf, _edges = cal.compute_ece(np.array([]), np.array([]))
        assert ece == 0.0
        assert len(acc) == cal.num_bins
        assert len(conf) == cal.num_bins

    def test_ece_bins_count(self):
        """ECE should return correct number of bins."""
        cal = CalibrationAnalyzer(num_bins=5)
        confidences = np.random.rand(100)
        correctness = np.random.randint(0, 2, 100).astype(float)
        _ece, acc, conf, edges = cal.compute_ece(confidences, correctness)
        assert len(acc) == 5
        assert len(conf) == 5
        assert len(edges) == 6  # num_bins + 1

    def test_ece_nonnegative(self):
        """ECE should always be >= 0."""
        cal = CalibrationAnalyzer()
        for seed in range(5):
            rng = np.random.default_rng(seed)
            confidences = rng.random(200)
            correctness = rng.integers(0, 2, 200).astype(float)
            ece, _, _, _ = cal.compute_ece(confidences, correctness)
            assert ece >= 0.0


class TestPerplexityAnalyzer:
    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions should be ~0."""
        ppl = PerplexityAnalyzer()
        rng = np.random.default_rng(42)
        samples = rng.lognormal(2.5, 0.5, size=1000)
        kl = ppl._kl_divergence(samples, samples)
        assert kl < 0.01

    def test_kl_divergence_different(self):
        """KL divergence of different distributions should be positive."""
        ppl = PerplexityAnalyzer()
        rng = np.random.default_rng(42)
        p = rng.lognormal(2.5, 0.5, size=1000)
        q = rng.lognormal(3.5, 0.5, size=1000)
        kl = ppl._kl_divergence(p, q)
        assert kl > 0.1

    def test_kl_divergence_nonnegative(self):
        """KL divergence should always be >= 0."""
        ppl = PerplexityAnalyzer()
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.exponential(5, size=500)
            q = rng.exponential(5, size=500)
            kl = ppl._kl_divergence(p, q)
            assert kl >= 0.0

    def test_kl_divergence_constant(self):
        """KL divergence of constant (degenerate) distributions should be 0."""
        ppl = PerplexityAnalyzer()
        samples = np.ones(100) * 5.0
        kl = ppl._kl_divergence(samples, samples)
        assert kl == 0.0

    def test_kl_divergence_asymmetric(self):
        """KL(P||Q) != KL(Q||P) in general."""
        ppl = PerplexityAnalyzer()
        rng = np.random.default_rng(42)
        p = rng.lognormal(2.0, 0.3, size=2000)
        q = rng.lognormal(3.0, 0.8, size=2000)
        kl_pq = ppl._kl_divergence(p, q)
        kl_qp = ppl._kl_divergence(q, p)
        # They should generally differ (KL is not symmetric)
        assert abs(kl_pq - kl_qp) > 0.001 or (kl_pq == 0 and kl_qp == 0)
