"""
Breadth-First Search (BFS)

Explores the state space level by level, guaranteeing an optimal
(minimum-cost) path when step costs are uniform.
"""

import time
from collections import deque
from typing import Optional

from environment.grid_world import GridWorld, State, SearchResult, build_path
from .base import SearchConstraints


def bfs(
    env: GridWorld,
    constraints: Optional[SearchConstraints] = None,
) -> SearchResult:
    """
    Run BFS on the grid world.

    BFS uses a FIFO queue to expand nodes in order of increasing depth,
    so the first goal found has minimum path cost for uniform costs.

    Constraints (when set):
    - max_depth: do not expand nodes beyond this depth
    - max_expansions: stop after this many node expansions
    - time_budget_seconds: stop if wall-clock time exceeds this
    """
    constraints = constraints or SearchConstraints()
    start_time = time.perf_counter()
    nodes_expanded = 0

    # Frontier: (state, depth). FIFO order gives level-by-level expansion → optimal path for unit cost.
    frontier = deque([(env.start, 0)])
    visited = {env.start}
    parent: dict[State, Optional[State]] = {env.start: None}  # for path reconstruction
    failure_reason: Optional[str] = "exhausted"

    while frontier:
        # Check resource limits before expanding; same order used in DFS/other algorithms.
        if constraints.time_budget_seconds is not None:
            if time.perf_counter() - start_time > constraints.time_budget_seconds:
                failure_reason = "time_budget"
                break

        # Max expansions check
        if constraints.max_expansions is not None and nodes_expanded >= constraints.max_expansions:
            failure_reason = "expansion_limit"
            break

        state, depth = frontier.popleft()
        nodes_expanded += 1

        if env.is_goal(state):
            path = build_path(parent, state)
            # Path has N states → N-1 edges; we use unit cost per step.
            path_cost = len(path) - 1
            return SearchResult(
                solution_found=True,
                path=path,
                path_cost=float(path_cost),
                nodes_expanded=nodes_expanded,
                runtime_seconds=time.perf_counter() - start_time,
            )

        # Max depth: do not add successors beyond max_depth
        if constraints.max_depth is not None and depth >= constraints.max_depth:
            continue

        for successor, cost in env.get_successors(state):
            if successor not in visited:
                visited.add(successor)
                parent[successor] = state
                frontier.append((successor, depth + 1))

    runtime = time.perf_counter() - start_time
    # depth_limit: had max_depth and exhausted search within that depth
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
