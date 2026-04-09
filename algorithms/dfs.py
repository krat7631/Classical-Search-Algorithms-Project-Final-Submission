"""
Depth-First Search (DFS)

Explores the state space by going as deep as possible before backtracking.
Uses a LIFO stack. Does not guarantee an optimal path.
"""

import time
from typing import Optional

from environment.grid_world import GridWorld, State, SearchResult, build_path
from .base import SearchConstraints


def dfs(
    env: GridWorld,
    constraints: Optional[SearchConstraints] = None,
) -> SearchResult:
    """
    Run DFS on the grid world.

    DFS uses a stack (LIFO) to expand the most recently discovered node,
    exploring depth-first. First goal found may not be optimal.

    Constraints (when set):
    - max_depth: do not expand nodes beyond this depth
    - max_expansions: stop after this many node expansions
    - time_budget_seconds: stop if wall-clock time exceeds this
    """
    constraints = constraints or SearchConstraints()
    start_time = time.perf_counter()
    nodes_expanded = 0

    # Stack: (state, depth). LIFO gives depth-first order; first goal found may not be optimal.
    stack: list[tuple[State, int]] = [(env.start, 0)]
    visited = {env.start}
    parent: dict[State, Optional[State]] = {env.start: None}
    failure_reason: Optional[str] = "exhausted"

    while stack:
        # Same constraint checks as BFS so we can compare behaviour under limits.
        if constraints.time_budget_seconds is not None:
            if time.perf_counter() - start_time > constraints.time_budget_seconds:
                failure_reason = "time_budget"
                break

        if constraints.max_expansions is not None and nodes_expanded >= constraints.max_expansions:
            failure_reason = "expansion_limit"
            break

        state, depth = stack.pop()
        nodes_expanded += 1

        if env.is_goal(state):
            path = build_path(parent, state)
            path_cost = len(path) - 1  # N states → N-1 steps (unit cost)
            return SearchResult(
                solution_found=True,
                path=path,
                path_cost=float(path_cost),
                nodes_expanded=nodes_expanded,
                runtime_seconds=time.perf_counter() - start_time,
            )

        if constraints.max_depth is not None and depth >= constraints.max_depth:
            continue

        for successor, _ in env.get_successors(state):
            if successor not in visited:
                visited.add(successor)
                parent[successor] = state
                stack.append((successor, depth + 1))

    runtime = time.perf_counter() - start_time
    if failure_reason == "exhausted" and constraints.max_depth is not None:
        failure_reason = "depth_limit"
    return SearchResult(
        solution_found=False,
        path=[],
        path_cost=0.0,
        nodes_expanded=nodes_expanded,
        runtime_seconds=runtime,
        failure_reason=failure_reason,
    )
