"""
A* Search with Admissible Heuristic

Uses f(n) = g(n) + h(n) where g(n) is cost from start and h(n) is
an admissible heuristic. Delegates to astar_theory with Manhattan heuristic.
"""

from typing import Optional

from environment.grid_world import GridWorld, SearchResult
from .base import SearchConstraints
from .heuristics import make_heuristic
from .astar_theory import astar_theory


def astar(
    env: GridWorld,
    constraints: Optional[SearchConstraints] = None,
) -> SearchResult:
    """
    Run A* on the grid world using Manhattan distance heuristic.

    Constraints (when set):
    - max_depth: do not expand nodes beyond this depth (g-value)
    - max_expansions: stop after this many node expansions
    - time_budget_seconds: stop if wall-clock time exceeds this
    """
    h = make_heuristic(env, "manhattan")
    return astar_theory(env, h, constraints, trace=False)
