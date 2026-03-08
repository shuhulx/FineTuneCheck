"""Tests for multi-run comparison and Pareto analysis."""

import numpy as np

from finetunecheck.compare.pareto import compute_pareto_frontier


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Thin wrapper: adapts list-of-tuples interface to dict-based compute_pareto_frontier."""
    named = {str(i): pt for i, pt in enumerate(points)}
    frontier_names = compute_pareto_frontier(named)
    return sorted(int(n) for n in frontier_names)


class TestParetoFrontier:
    def test_pareto_frontier_single(self):
        """Single point is always on the frontier."""
        points = [(0.8, 0.9)]
        frontier = pareto_frontier(points)
        assert frontier == [0]

    def test_pareto_frontier_dominated(self):
        """Dominated point should not be on frontier."""
        points = [
            (0.5, 0.5),  # dominated by point 1
            (0.8, 0.9),  # dominates point 0
        ]
        frontier = pareto_frontier(points)
        assert 0 not in frontier
        assert 1 in frontier

    def test_pareto_frontier_all_optimal(self):
        """Non-dominated points should all be on frontier."""
        # Points along the tradeoff curve -- none dominates another
        points = [
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.5),
        ]
        frontier = pareto_frontier(points)
        # (0.5, 0.5) is dominated by neither (1.0, 0.0) nor (0.0, 1.0)
        assert len(frontier) == 3

    def test_pareto_frontier_clear_domination(self):
        """Only one point should survive clear domination."""
        points = [
            (0.1, 0.1),
            (0.2, 0.2),
            (0.9, 0.9),  # dominates all others
        ]
        frontier = pareto_frontier(points)
        assert frontier == [2]

    def test_pareto_frontier_identical_points(self):
        """Identical points are not dominated by each other (they are equal)."""
        points = [
            (0.5, 0.5),
            (0.5, 0.5),
        ]
        frontier = pareto_frontier(points)
        assert len(frontier) == 2

    def test_pareto_frontier_tradeoff_curve(self):
        """Points on a tradeoff curve should all be on the frontier."""
        # As one objective goes up, the other goes down
        points = [
            (1.0, 0.0),
            (0.75, 0.25),
            (0.5, 0.5),
            (0.25, 0.75),
            (0.0, 1.0),
        ]
        frontier = pareto_frontier(points)
        assert len(frontier) == 5

    def test_pareto_frontier_empty(self):
        """Empty list should return empty frontier."""
        assert pareto_frontier([]) == []

    def test_pareto_frontier_mixed(self):
        """Mix of dominated and non-dominated points."""
        points = [
            (0.3, 0.9),  # non-dominated (high y)
            (0.9, 0.3),  # non-dominated (high x)
            (0.5, 0.5),  # dominated by neither of the above
            (0.2, 0.2),  # dominated by all above
        ]
        frontier = pareto_frontier(points)
        assert 3 not in frontier  # (0.2, 0.2) is dominated
        assert 0 in frontier
        assert 1 in frontier

    def test_pareto_frontier_large_set(self):
        """Should work correctly on a larger set."""
        rng = np.random.default_rng(42)
        n = 50
        points = [(float(rng.random()), float(rng.random())) for _ in range(n)]
        frontier = pareto_frontier(points)
        # Verify no frontier point is dominated by another
        for i in frontier:
            for j in frontier:
                if i == j:
                    continue
                dominated = (points[j][0] >= points[i][0] and
                             points[j][1] >= points[i][1] and
                             (points[j][0] > points[i][0] or points[j][1] > points[i][1]))
                assert not dominated, f"Point {i} is dominated by {j} on frontier"
