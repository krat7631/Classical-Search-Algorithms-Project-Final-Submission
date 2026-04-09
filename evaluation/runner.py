"""
Experiment runner for evaluating search algorithms under constraints.

Runs each algorithm on the same environment with the same constraint
values, collecting metrics for Methods and Results sections.

Supports sweeps over grid sizes, obstacle densities, and constraint values.
Outputs raw per-run rows suitable for JSON/CSV export and later aggregation.
"""

import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from environment.grid_world import GridWorld, SearchResult, generate_grid_world
from algorithms.base import SearchConstraints
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.iddfs import iddfs
from algorithms.astar import astar


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    env: GridWorld
    constraints: Optional[SearchConstraints] = None
    algorithm_name: str = ""


# Map algorithm names to their runnable functions
ALGORITHMS: Dict[str, Callable[[GridWorld, Optional[SearchConstraints]], SearchResult]] = {
    "bfs": bfs,
    "dfs": dfs,
    "iddfs": iddfs,
    "astar": astar,
}

# Tighter time budgets for stress testing (seconds)
TIME_BUDGET_STRESS = [0.001, 0.002, 0.005, 0.01, 0.1]

# Default sweep values for experiments
DEFAULT_GRID_SIZES = [(5, 5), (10, 10), (15, 15)]
DEFAULT_OBSTACLE_DENSITIES = [0.0, 0.1, 0.2]
DEFAULT_SEED = 42


def run_single(
    env: GridWorld,
    algorithm_name: str,
    constraints: Optional[SearchConstraints] = None,
    config_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one algorithm on one environment with optional constraints.

    Returns a dict with metrics: algorithm, solution_found, path_cost,
    nodes_expanded, runtime_seconds, failure_reason, grid_size, obstacle_density.
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    fn = ALGORITHMS[algorithm_name]
    result = fn(env, constraints)

    constraints_dict: Dict[str, Any] = {}
    if constraints:
        if constraints.max_depth is not None:
            constraints_dict["max_depth"] = constraints.max_depth
        if constraints.max_expansions is not None:
            constraints_dict["max_expansions"] = constraints.max_expansions
        if constraints.time_budget_seconds is not None:
            constraints_dict["time_budget_seconds"] = constraints.time_budget_seconds

    out: Dict[str, Any] = {
        "algorithm": algorithm_name,
        "solution_found": result.solution_found,
        "path_length": len(result.path),
        "path_cost": result.path_cost,
        "nodes_expanded": result.nodes_expanded,
        "runtime_seconds": result.runtime_seconds,
        "grid_size": env.get_grid_size_str(),
        "obstacle_density": round(env.get_obstacle_density(), 4),
        "constraints": constraints_dict if constraints_dict else None,
    }
    if config_label is not None:
        out["config_label"] = config_label
    if result.failure_reason is not None:
        out["failure_reason"] = result.failure_reason
    if result.nodes_reopened is not None:
        out["nodes_reopened"] = result.nodes_reopened
    if result.consistency_violations_detected is not None:
        out["consistency_violations_detected"] = result.consistency_violations_detected
    if result.optimal is not None:
        out["optimal"] = result.optimal
    if result.suboptimality_gap is not None:
        out["suboptimality_gap"] = result.suboptimality_gap
    return out


def run_experiment(
    env: GridWorld,
    algorithms: Optional[List[str]] = None,
    constraints: Optional[SearchConstraints] = None,
    config_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run all (or specified) algorithms on the same environment with the same constraints.

    Returns a list of result dicts, one per algorithm.
    """
    if algorithms is None:
        algorithms = list(ALGORITHMS.keys())

    results = []
    for name in algorithms:
        results.append(run_single(env, name, constraints, config_label))
    return results


def run_sweep(
    grid_sizes: Optional[List[tuple]] = None,
    obstacle_densities: Optional[List[float]] = None,
    algorithms: Optional[List[str]] = None,
    constraint_configs: Optional[List[tuple]] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Sweep over grid sizes, obstacle densities, and constraint configurations.

    Args:
        grid_sizes: List of (rows, cols), e.g. [(5,5), (10,10)]
        obstacle_densities: List of densities 0.0-1.0, e.g. [0.0, 0.1, 0.2]
        algorithms: Which algorithms to run (default: all)
        constraint_configs: List of (label, SearchConstraints), e.g.
            [("baseline", None), ("max_depth_5", SearchConstraints(max_depth=5))]
        seed: Random seed for grid generation (deterministic)

    Returns:
        List of raw result dicts (one row per algorithm per environment/config).
        Aggregation (means, medians, rates) is done later in analysis/plotting.
    """
    grid_sizes = grid_sizes or DEFAULT_GRID_SIZES
    obstacle_densities = obstacle_densities or DEFAULT_OBSTACLE_DENSITIES
    algorithms = algorithms or list(ALGORITHMS.keys())
    seed = seed if seed is not None else DEFAULT_SEED

    # Default: baseline + a few constraint configs including tight time budgets
    if constraint_configs is None:
        constraint_configs = [
            ("baseline", None),
            ("max_depth_5", SearchConstraints(max_depth=5)),
            ("max_expansions_50", SearchConstraints(max_expansions=50)),
            ("time_0.001", SearchConstraints(time_budget_seconds=0.001)),
            ("time_0.002", SearchConstraints(time_budget_seconds=0.002)),
            ("time_0.005", SearchConstraints(time_budget_seconds=0.005)),
        ]

    all_results: List[Dict[str, Any]] = []
    for (rows, cols) in grid_sizes:
        for density in obstacle_densities:
            env = generate_grid_world(rows, cols, obstacle_density=density, seed=seed)
            for config_label, constraints in constraint_configs:
                if verbose:
                    print(f"  {rows}x{cols} d={density} {config_label}...", file=sys.stderr, flush=True)
                for name in algorithms:
                    r = run_single(env, name, constraints, config_label=config_label)
                    all_results.append(r)

    return all_results
