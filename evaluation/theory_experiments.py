"""
Theoretical experiments validating A* properties (Hart, Nilsson, Raphael 1968).

Compares BFS (optimal baseline), A* with h=0 (UCS), A* with admissible heuristic,
A* with inconsistent heuristic, and weighted A*.
"""

import json
import sys
from typing import Any, Dict, List, Optional

from environment.grid_world import GridWorld, generate_grid_world
from algorithms.base import SearchConstraints
from algorithms.bfs import bfs
from algorithms.heuristics import make_heuristic
from algorithms.astar_theory import astar_theory

# Weights for scaled heuristic in theory sweep
SCALED_WEIGHTS = [1.0, 1.2, 1.5, 2.0, 3.0]


def run_theory_experiment(
    env: GridWorld,
    constraint_configs: Optional[List[tuple]] = None,
    trace: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run theory-validating experiment: BFS + A* variants.

    For each config, runs:
    - BFS (optimal baseline)
    - A* h=0 (UCS)
    - A* Manhattan (admissible, consistent)
    - A* inconsistent (admissible, not consistent)
    - A* scaled with weights [1.0, 1.2, 1.5, 2.0, 3.0]
    """
    if constraint_configs is None:
        constraint_configs = [("baseline", None)]

    results: List[Dict[str, Any]] = []
    for config_label, constraints in constraint_configs:
        # BFS - optimal cost baseline
        bfs_result = bfs(env, constraints)
        optimal_cost = bfs_result.path_cost if bfs_result.solution_found else None

        # Add BFS row first
        results.append({
            "config_label": config_label,
            "algorithm": "bfs",
            "heuristic": None,
            "heuristic_weight": None,
            "solution_found": bfs_result.solution_found,
            "path_cost": bfs_result.path_cost,
            "optimal_cost_baseline": optimal_cost,
            "nodes_expanded": bfs_result.nodes_expanded,
            "nodes_reopened": 0,
            "consistency_violations_detected": 0,
            "runtime_seconds": bfs_result.runtime_seconds,
            "grid_size": env.get_grid_size_str(),
            "obstacle_density": round(env.get_obstacle_density(), 4),
            "optimal": True,
            "suboptimality_gap": None,
        })

        # Non-scaled heuristics
        for heuristic_name in ["zero", "manhattan", "inconsistent"]:
            weight = 1.0
            h = make_heuristic(env, heuristic_name, weight)
            r = astar_theory(env, h, constraints, trace=trace)

            optimal = None
            suboptimality_gap = None
            if r.solution_found and optimal_cost is not None:
                optimal = abs(r.path_cost - optimal_cost) < 1e-6
                if not optimal:
                    suboptimality_gap = r.path_cost - optimal_cost

            row: Dict[str, Any] = {
                "config_label": config_label,
                "algorithm": f"astar_{heuristic_name}",
                "heuristic": heuristic_name,
                "heuristic_weight": None,
                "solution_found": r.solution_found,
                "path_cost": r.path_cost,
                "optimal_cost_baseline": optimal_cost,
                "nodes_expanded": r.nodes_expanded,
                "nodes_reopened": r.nodes_reopened or 0,
                "consistency_violations_detected": r.consistency_violations_detected or 0,
                "runtime_seconds": r.runtime_seconds,
                "grid_size": env.get_grid_size_str(),
                "obstacle_density": round(env.get_obstacle_density(), 4),
                "optimal": optimal,
                "suboptimality_gap": suboptimality_gap,
            }
            results.append(row)

        # Scaled heuristic: iterate over weights
        for weight in SCALED_WEIGHTS:
            h = make_heuristic(env, "scaled", weight)
            r = astar_theory(env, h, constraints, trace=trace)

            optimal = None
            suboptimality_gap = None
            if r.solution_found and optimal_cost is not None:
                optimal = abs(r.path_cost - optimal_cost) < 1e-6
                if not optimal:
                    suboptimality_gap = r.path_cost - optimal_cost

            row = {
                "config_label": config_label,
                "algorithm": f"astar_scaled_w{weight}",
                "heuristic": "scaled",
                "heuristic_weight": weight,
                "solution_found": r.solution_found,
                "path_cost": r.path_cost,
                "optimal_cost_baseline": optimal_cost,
                "nodes_expanded": r.nodes_expanded,
                "nodes_reopened": r.nodes_reopened or 0,
                "consistency_violations_detected": r.consistency_violations_detected or 0,
                "runtime_seconds": r.runtime_seconds,
                "grid_size": env.get_grid_size_str(),
                "obstacle_density": round(env.get_obstacle_density(), 4),
                "optimal": optimal,
                "suboptimality_gap": suboptimality_gap,
            }
            results.append(row)

    return results


def run_theory_sweep(
    grid_sizes: Optional[List[tuple]] = None,
    obstacle_densities: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Sweep over grid sizes, obstacle densities, and seeds.

    Returns raw per-run rows (not pre-aggregated), so downstream scripts can
    compute means/medians tailored to a figure or table.
    """
    grid_sizes = grid_sizes or [(5, 5), (10, 10), (15, 15), (20, 20)]
    obstacle_densities = obstacle_densities or [0.0, 0.1, 0.2]
    seeds = seeds or [42, 43, 44, 45, 46]

    all_results: List[Dict[str, Any]] = []
    for (rows, cols) in grid_sizes:
        for density in obstacle_densities:
            for seed in seeds:
                env = generate_grid_world(rows, cols, obstacle_density=density, seed=seed)
                batch = run_theory_experiment(env, [("baseline", None)])
                for r in batch:
                    r["seed"] = seed
                all_results.extend(batch)

    return all_results


def format_theory_report(results: List[Dict[str, Any]]) -> str:
    """
    Format a readable console report for --theory-report.

    Note: this formatter summarizes one representative row per algorithm in the
    provided results list. The full CSV/JSON should be used for aggregate stats.
    """
    lines = ["=== A* Theory Report (Hart, Nilsson, Raphael 1968) ===\n"]

    by_algo: Dict[str, List[Dict]] = {}
    for r in results:
        algo = r.get("algorithm", "?")
        by_algo.setdefault(algo, []).append(r)

    base_algos = ["bfs", "astar_zero", "astar_manhattan", "astar_inconsistent"]
    scaled_algos = sorted(k for k in by_algo if k.startswith("astar_scaled_"))
    for algo in base_algos + scaled_algos:
        if algo not in by_algo:
            continue
        rows = by_algo[algo]
        r = rows[0]
        admissible = "N/A" if algo == "bfs" else ("Yes" if r.get("optimal", False) or not r.get("solution_found") else "No (suboptimal)")
        consistent = "N/A" if algo == "bfs" else ("No" if (r.get("consistency_violations_detected", 0) or r.get("nodes_reopened", 0)) else "Yes")
        optimal = "Yes" if r.get("optimal", True) else f"No (gap={r.get('suboptimality_gap', 0)})"
        lines.append(f"Algorithm: {algo}")
        lines.append(f"  Heuristic: {r.get('heuristic', 'N/A')}")
        if r.get("heuristic_weight") is not None:
            lines.append(f"  Weight: {r['heuristic_weight']}")
        lines.append(f"  Admissible (empirical): {admissible}")
        lines.append(f"  Consistent (empirical): {consistent}")
        lines.append(f"  Nodes expanded: {r.get('nodes_expanded', 0)}")
        lines.append(f"  Nodes reopened: {r.get('nodes_reopened', 0)}")
        lines.append(f"  Consistency violations: {r.get('consistency_violations_detected', 0)}")
        lines.append(f"  Optimal solution: {optimal}")
        lines.append("")

    return "\n".join(lines)
