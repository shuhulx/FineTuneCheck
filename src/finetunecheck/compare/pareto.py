"""Pareto frontier computation for multi-run comparison."""

from __future__ import annotations


def compute_pareto_frontier(
    points: dict[str, tuple[float, float]],
) -> list[str]:
    """Find Pareto-optimal runs on target-improvement vs BWT.

    Each point is ``(target_improvement, backward_transfer)`` where higher is better
    for *both* dimensions.  A run is Pareto-optimal if no other run dominates it
    (i.e., is >= on both dimensions and strictly > on at least one).

    Parameters
    ----------
    points:
        Mapping of run name to ``(target_improvement, backward_transfer)``.
        Both axes are "higher is better".

    Returns
    -------
    list[str]
        Names of Pareto-optimal runs, sorted by target_improvement descending.
    """
    if not points:
        return []

    names = list(points.keys())
    frontier: list[str] = []

    for name in names:
        x, y = points[name]
        dominated = False
        for other_name in names:
            if other_name == name:
                continue
            ox, oy = points[other_name]
            if ox >= x and oy >= y and (ox > x or oy > y):
                dominated = True
                break
        if not dominated:
            frontier.append(name)

    frontier.sort(key=lambda n: points[n][0], reverse=True)
    return frontier
