"""
A* Search - Theory-driven implementation (Hart, Nilsson, Raphael 1968).

Modular structure with:
- Explicit open list (priority queue)
- Explicit closed set (expanded nodes)
- Reopening mechanism (for inconsistent heuristics)
- Heuristic injection
- Consistency/admissibility runtime checks
"""

import heapq
import itertools
import time
from typing import Callable, Optional

from environment.grid_world import GridWorld, State, SearchResult, build_path
from .base import SearchConstraints

# Diagonal steps cost sqrt(2), so two equally short paths can differ by ~1e-15 in
# floating point. Without a tolerance, A* would count those as "cheaper paths" and
# reopen nodes, and the consistency check would flag violations that do not exist.
EPS = 1e-9


def astar_theory(
    env: GridWorld,
    heuristic: Callable[[State], float],
    constraints: Optional[SearchConstraints] = None,
    trace: bool = False,
    tie_break: str = "fifo",
    reopen: bool = True,
) -> SearchResult:
    """
    A* with modular heuristic, explicit closed set, and reopening support.

    tie_break decides between open nodes with equal f:
    - "fifo": the one generated first (default; reproduces the report's results)
    - "high_g": the one with the larger g, i.e. closer to the goal along its path,
      which avoids expanding whole plateaus of equal-f nodes on open maps

    reopen controls what happens when a cheaper path to an already-expanded state is
    found. True (default) re-expands it, which keeps A* optimal with an inconsistent
    heuristic. False ignores it: with weighted A* (h = w * consistent h) the path is
    still within a factor w of optimal, and it avoids cascades of re-expansions.

    Tracks:
    - nodes_reopened: how often a previously closed state is improved
    - consistency_violations_detected: count of h(n) > c(n,n') + h(n')
    - nodes_generated: distinct states ever stored (the search's memory footprint)
    """
    if tie_break not in ("fifo", "high_g"):
        raise ValueError(f"Unknown tie_break: {tie_break}")
    prefer_high_g = tie_break == "high_g"
    constraints = constraints or SearchConstraints()
    start_time = time.perf_counter()
    nodes_expanded = 0
    nodes_reopened = 0
    consistency_violations = 0
    tie_counter = itertools.count()
    failure_reason: Optional[str] = "exhausted"

    h_start = heuristic(env.start)
    # Heap entries: (f, tie_key, insertion_order, g, state). tie_key is -g when
    # preferring high g, otherwise 0 so insertion order (FIFO) decides.
    open_list: list[tuple[float, float, int, float, State]] = [
        (h_start, 0.0, next(tie_counter), 0.0, env.start)
    ]
    g_score: dict[State, float] = {env.start: 0.0}
    parent: dict[State, Optional[State]] = {env.start: None}
    closed: set[State] = set()

    while open_list:
        if constraints.time_budget_seconds is not None:
            if time.perf_counter() - start_time > constraints.time_budget_seconds:
                failure_reason = "time_budget"
                break

        if constraints.max_expansions is not None and nodes_expanded >= constraints.max_expansions:
            failure_reason = "expansion_limit"
            break

        # The same state can appear multiple times on the heap (each relax pushes a
        # new tuple). Older entries carry superseded (larger) g values and must be ignored.
        f, _, _, g, state = heapq.heappop(open_list)

        # Duplicate / stale open-list entry: already closed this state with a strictly
        # better g than this tuple's g (leftover heap junk from before we closed).
        if state in closed and g_score.get(state, float("inf")) < g:
            continue

        # Stale entry: best known g_score[state] is already lower than this pop's g.
        if g > g_score.get(state, float("inf")):
            continue

        nodes_expanded += 1

        if env.is_goal(state):
            path = build_path(parent, state)
            runtime = time.perf_counter() - start_time
            return SearchResult(
                solution_found=True,
                path=path,
                path_cost=g,
                nodes_expanded=nodes_expanded,
                runtime_seconds=runtime,
                nodes_reopened=nodes_reopened,
                consistency_violations_detected=consistency_violations if consistency_violations > 0 else None,
                nodes_generated=len(g_score),
            )

        # Depth bound is enforced on expansion depth (g in this unit-cost setting).
        if constraints.max_depth is not None and int(g) >= constraints.max_depth:
            continue

        closed.add(state)

        h_n = heuristic(state)
        for successor, cost in env.get_successors(state):
            new_g = g + cost
            old_g = g_score.get(successor, float("inf"))

            # Consistency check: h(n) <= c(n,n') + h(n')
            h_nprime = heuristic(successor)
            if h_n > cost + h_nprime + EPS:
                consistency_violations += 1
                if trace:
                    print(
                        f"  [CONSISTENCY VIOLATION] h({state})= {h_n:.1f} > {cost:.1f} + h({successor})= {h_nprime:.1f}",
                        flush=True,
                    )

            if new_g < old_g - EPS:
                # Reopening: if h is inconsistent, a cheaper path to successor can appear
                # after successor was already expanded and placed in closed; we must allow
                # another expansion, so we drop successor from closed and count a reopening.
                if successor in closed:
                    if not reopen:
                        continue
                    nodes_reopened += 1
                    closed.discard(successor)
                    if trace:
                        print(
                            f"  [REOPEN] {successor} g: {old_g} -> {new_g} (f={new_g + h_nprime:.1f})",
                            flush=True,
                        )

                g_score[successor] = new_g
                parent[successor] = state
                f_new = new_g + h_nprime
                tie_key = -new_g if prefer_high_g else 0.0
                heapq.heappush(open_list, (f_new, tie_key, next(tie_counter), new_g, successor))

    runtime = time.perf_counter() - start_time
    # No goal found: if a depth cap was in effect and the open list drained, we hit the
    # depth frontier before proving reachability; otherwise the frontier is exhausted.
    if failure_reason == "exhausted" and constraints.max_depth is not None:
        failure_reason = "depth_limit"
    return SearchResult(
        solution_found=False,
        path=[],
        path_cost=0.0,
        nodes_expanded=nodes_expanded,
        runtime_seconds=runtime,
        failure_reason=failure_reason,
        nodes_reopened=nodes_reopened,
        consistency_violations_detected=consistency_violations if consistency_violations > 0 else None,
        nodes_generated=len(g_score),
    )
