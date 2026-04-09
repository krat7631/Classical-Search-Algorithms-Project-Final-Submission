"""
Iterative Deepening Depth-First Search (IDDFS)

Repeatedly runs depth-limited DFS with increasing depth limits.
Combines the space efficiency of DFS with the completeness and
optimality (for uniform costs) of BFS.
"""

import time
from typing import Optional

from environment.grid_world import GridWorld, State, SearchResult, build_path
from .base import SearchConstraints


def _dls(
    env: GridWorld,
    state: State,
    goal: State,
    depth_limit: int,
    parent: dict,
    nodes_expanded: list[int],
    total_before: int,
    constraints: SearchConstraints,
    start_time: float,
    failure_reason: list,
) -> tuple[Optional[State], bool]:
    """
    Depth-Limited Search: DFS with a depth cutoff.
    Returns (goal_state_if_found, cutoff_reached).
    failure_reason list is mutated when we hit time/expansion limits.
    """
    if constraints.time_budget_seconds is not None:
        if time.perf_counter() - start_time > constraints.time_budget_seconds:
            failure_reason[0] = "time_budget"
            return None, False

    total_so_far = total_before + nodes_expanded[0]
    if constraints.max_expansions is not None and total_so_far >= constraints.max_expansions:
        failure_reason[0] = "expansion_limit"
        return None, False

    nodes_expanded[0] += 1

    if env.is_goal(state):
        return state, False

    if depth_limit == 0:
        return None, True  # cutoff

    cutoff_occurred = False
    for successor, _ in env.get_successors(state):
        if successor not in parent:
            parent[successor] = state
            result, cutoff = _dls(
                env, successor, goal, depth_limit - 1,
                parent, nodes_expanded, total_before, constraints, start_time,
                failure_reason,
            )
            if result is not None:
                return result, False
            if cutoff:
                cutoff_occurred = True
            del parent[successor]  # backtrack

    return None, cutoff_occurred


def iddfs(
    env: GridWorld,
    constraints: Optional[SearchConstraints] = None,
) -> SearchResult:
    """
    Run IDDFS on the grid world.

    IDDFS iteratively increases the depth limit and runs DFS until
    a solution is found or no new nodes can be explored. Optimal for
    uniform step costs.

    Constraints (when set):
    - max_depth: overall depth limit (max iteration depth)
    - max_expansions: stop after this many total node expansions across iterations
    - time_budget_seconds: stop if wall-clock time exceeds this
    """
    constraints = constraints or SearchConstraints()
    start_time = time.perf_counter()
    total_expansions = 0
    failure_reason: list = ["exhausted"]

    depth_limit = 0
    max_depth = constraints.max_depth

    while True:
        if constraints.time_budget_seconds is not None:
            if time.perf_counter() - start_time > constraints.time_budget_seconds:
                failure_reason[0] = "time_budget"
                break

        if constraints.max_expansions is not None and total_expansions >= constraints.max_expansions:
            failure_reason[0] = "expansion_limit"
            break

        if max_depth is not None and depth_limit > max_depth:
            failure_reason[0] = "depth_limit"
            break

        parent: dict[State, Optional[State]] = {env.start: None}
        nodes_expanded = [0]
        result, cutoff = _dls(
            env, env.start, env.goal, depth_limit,
            parent, nodes_expanded, total_expansions, constraints, start_time,
            failure_reason,
        )
        total_expansions += nodes_expanded[0]

        if result is not None:
            path = build_path(parent, result)
            path_cost = len(path) - 1
            return SearchResult(
                solution_found=True,
                path=path,
                path_cost=float(path_cost),
                nodes_expanded=total_expansions,
                runtime_seconds=time.perf_counter() - start_time,
            )

        if not cutoff:
            break

        depth_limit += 1

    runtime = time.perf_counter() - start_time
    return SearchResult(
        solution_found=False,
        path=[],
        path_cost=0.0,
        nodes_expanded=total_expansions,
        runtime_seconds=runtime,
        failure_reason=failure_reason[0],
    )
