"""
Heuristic functions for A* search (Hart, Nilsson, Raphael 1968).

Provides h_zero, h_manhattan, h_scaled, h_inconsistent for testing
admissibility and consistency properties.
"""

from typing import Callable

from environment.grid_world import GridWorld, State


def h_zero(env: GridWorld, state: State) -> float:
    """
    Always returns 0. Equivalent to Uniform Cost Search (UCS).
    Admissible and consistent.
    """
    return 0.0


def h_manhattan(env: GridWorld, state: State) -> float:
    """
    Manhattan distance to goal. Admissible and consistent for 4-directional grid.
    Never overestimates (each step moves at most 1 unit toward goal).
    """
    return abs(state.row - env.goal.row) + abs(state.col - env.goal.col)


def h_scaled(env: GridWorld, weight: float = 1.0) -> Callable[[State], float]:
    """
    Returns a heuristic h(n) = weight * Manhattan(n).
    Admissible iff weight <= 1; consistent iff weight <= 1.
    """
    def _h(s: State) -> float:
        return weight * (abs(s.row - env.goal.row) + abs(s.col - env.goal.col))
    return _h


def h_inconsistent(env: GridWorld, state: State) -> float:
    """
    Intentionally violates consistency (triangle inequality).
    Admissible but not consistent: anchor at grid center returns 0,
    causing h(n) > c(n,n') + h(n') for neighbors n of the anchor.

    Consistency requires: h(n) <= c(n,n') + h(n').
    """
    # Anchor at center (likely traversable)
    ar, ac = env.rows // 2, env.cols // 2
    if state.row == ar and state.col == ac:
        return 0.0
    return abs(state.row - env.goal.row) + abs(state.col - env.goal.col)


def make_heuristic(
    env: GridWorld,
    name: str = "manhattan",
    weight: float = 1.0,
) -> Callable[[State], float]:
    """
    Factory: return heuristic function by name.

    Args:
        env: GridWorld (for goal reference)
        name: zero | manhattan | scaled | inconsistent
        weight: for scaled heuristic only
    """
    if name == "zero":
        return lambda s: h_zero(env, s)
    if name == "manhattan":
        return lambda s: h_manhattan(env, s)
    if name == "scaled":
        return h_scaled(env, weight)
    if name == "inconsistent":
        return lambda s: h_inconsistent(env, s)
    raise ValueError(f"Unknown heuristic: {name}")
