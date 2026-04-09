"""
Main entry point for running search algorithm experiments.

Supports classical search, theory experiments (Hart, Nilsson, Raphael 1968),
and heuristic analysis.

Example usage:
  python main.py                          # baseline on default grid
  python main.py --heuristic zero         # A* with h=0 (UCS)
  python main.py --heuristic scaled --weight 1.5
  python main.py --theory-report          # theory validation report
  python main.py --theory-sweep --format json > theory_results.json
"""

import argparse
import json
import sys
from typing import Optional

# Add project root to path
sys.path.insert(0, ".")

from environment.grid_world import GridWorld
from algorithms.base import SearchConstraints
from algorithms.heuristics import make_heuristic
from algorithms.astar_theory import astar_theory
from evaluation.runner import run_experiment, run_single, run_sweep, TIME_BUDGET_STRESS
from evaluation.theory_experiments import run_theory_experiment, run_theory_sweep, format_theory_report


# Default grid: 5x5 with a few obstacles. Start (0,0), Goal (4,4)
DEFAULT_GRID = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
]
DEFAULT_START = (0, 0)
DEFAULT_GOAL = (4, 4)


def main() -> None:
    # CLI is intentionally mode-based: one primary action per invocation
    # (theory report, theory sweep, heuristic probe, sweep, or baseline/all).
    parser = argparse.ArgumentParser(description="Run search algorithm experiments")
    parser.add_argument("--max-depth", type=int, default=None, help="Max search depth")
    parser.add_argument("--max-expansions", type=int, default=None, help="Max node expansions")
    parser.add_argument("--time-budget", type=float, default=None, help="Time budget in seconds")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run baseline + constrained experiments on default grid",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run full sweep (grid sizes, obstacle densities, constraints including time stress)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller sweep (5x5 only, fewer configs) for fast feedback",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Max grid dimension for sweep (e.g. 10 = skip 15x15). Default: no limit.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["bfs", "dfs", "iddfs", "astar"],
        help="Algorithms to run (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--heuristic",
        choices=["zero", "manhattan", "scaled", "inconsistent"],
        default=None,
        help="A* heuristic variant (runs A* only when set)",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Weight for scaled heuristic (default 1.0)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Trace node reopenings and consistency violations",
    )
    parser.add_argument(
        "--theory-report",
        action="store_true",
        help="Run theory experiment and output structured report",
    )
    parser.add_argument(
        "--theory-sweep",
        action="store_true",
        help="Sweep over grid sizes, densities, seeds (theory experiment)",
    )
    args = parser.parse_args()

    # Theory report: one fixed environment, structured summary across heuristics.
    if args.theory_report:
        env = GridWorld(DEFAULT_GRID, DEFAULT_START, DEFAULT_GOAL)
        results = run_theory_experiment(env, [("baseline", None)], trace=args.trace)
        report = format_theory_report(results)
        if args.format == "json":
            print(json.dumps(results, indent=2))
        else:
            print(report)
        return

    # Theory sweep: randomized environments over sizes/densities/seeds.
    if args.theory_sweep:
        print("Running theory sweep...", file=sys.stderr, flush=True)
        grid_sizes = [(5, 5), (10, 10), (15, 15), (20, 20)]
        if args.max_size is not None:
            grid_sizes = [(r, c) for r, c in grid_sizes if r <= args.max_size]
        results = run_theory_sweep(
            grid_sizes=grid_sizes,
            obstacle_densities=[0.0, 0.1, 0.2],
            seeds=[42, 43, 44, 45, 46],
        )
        print("Done.", file=sys.stderr, flush=True)
        print(json.dumps(results, indent=2))
        return

    # Heuristic probe mode: run theory A* only, then print/export one-row style output.
    if args.heuristic is not None:
        env = GridWorld(DEFAULT_GRID, DEFAULT_START, DEFAULT_GOAL)
        h = make_heuristic(env, args.heuristic, args.weight)
        constraints = SearchConstraints(
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
            time_budget_seconds=args.time_budget,
        )
        r = astar_theory(env, h, constraints, trace=args.trace)
        out = {
            "algorithm": f"astar_{args.heuristic}",
            "solution_found": r.solution_found,
            "path_cost": r.path_cost,
            "nodes_expanded": r.nodes_expanded,
            "nodes_reopened": r.nodes_reopened or 0,
            "consistency_violations_detected": r.consistency_violations_detected or 0,
            "runtime_seconds": r.runtime_seconds,
            "grid_size": env.get_grid_size_str(),
            "obstacle_density": round(env.get_obstacle_density(), 4),
        }
        if args.format == "json":
            print(json.dumps([out], indent=2))
        else:
            print(
                f"A* (h={args.heuristic}): found={r.solution_found}, "
                f"path_cost={r.path_cost}, nodes={r.nodes_expanded}, "
                f"reopened={r.nodes_reopened or 0}, violations={r.consistency_violations_detected or 0}"
            )
        return

    # General sweep mode: compare all selected algorithms under matched settings.
    if args.sweep:
        constraint_configs = [
            ("baseline", None),
            ("max_depth_5", SearchConstraints(max_depth=5)),
            ("max_expansions_50", SearchConstraints(max_expansions=50)),
        ] + [
            (f"time_{t}s", SearchConstraints(time_budget_seconds=t))
            for t in TIME_BUDGET_STRESS
        ]
        if args.quick:
            grid_sizes = [(5, 5)]
            obstacle_densities = [0.0, 0.1]
            constraint_configs = constraint_configs[:5]  # baseline, depth, expansions, time 0.001, 0.002
        else:
            grid_sizes = [(5, 5), (10, 10), (15, 15)]
            if args.max_size is not None:
                grid_sizes = [(r, c) for r, c in grid_sizes if r <= args.max_size and c <= args.max_size]
            obstacle_densities = [0.0, 0.1, 0.2]
        print("Running sweep...", file=sys.stderr, flush=True)
        all_results = run_sweep(
            grid_sizes=grid_sizes,
            obstacle_densities=obstacle_densities,
            algorithms=args.algorithms,
            constraint_configs=constraint_configs,
            seed=42,
            verbose=True,
        )
        print("Done.", file=sys.stderr, flush=True)
        if args.format == "json":
            print(json.dumps(all_results, indent=2))
        else:
            for r in all_results:
                fail = f", fail={r.get('failure_reason', '-')}" if not r["solution_found"] else ""
                print(
                    f"[{r['grid_size']} d={r['obstacle_density']}] {r.get('config_label', '?')} "
                    f"{r['algorithm']}: found={r['solution_found']}, nodes={r['nodes_expanded']}{fail}"
                )
        return

    env = GridWorld(DEFAULT_GRID, DEFAULT_START, DEFAULT_GOAL)

    if args.all:
        # Run baseline + several constraint configurations
        configs = [
            ("Baseline (no constraints)", None),
            ("max_depth=3", SearchConstraints(max_depth=3)),
            ("max_depth=5", SearchConstraints(max_depth=5)),
            ("max_expansions=50", SearchConstraints(max_expansions=50)),
            ("max_expansions=100", SearchConstraints(max_expansions=100)),
            ("time_budget=0.01s", SearchConstraints(time_budget_seconds=0.01)),
            ("time_budget=0.1s", SearchConstraints(time_budget_seconds=0.1)),
        ]
    else:
        constraints = SearchConstraints(
            max_depth=args.max_depth,
            max_expansions=args.max_expansions,
            time_budget_seconds=args.time_budget,
        )
        if args.max_depth is None and args.max_expansions is None and args.time_budget is None:
            configs = [("Baseline (no constraints)", None)]
        else:
            configs = [("Constrained", constraints)]

    all_results = []
    for label, constraints in configs:
        results = run_experiment(
            env, algorithms=args.algorithms, constraints=constraints, config_label=label
        )
        all_results.extend(results)

    if args.format == "json":
        print(json.dumps(all_results, indent=2))
    else:
        for r in all_results:
            line = (
                f"[{r['config_label']}] {r['algorithm']}: "
                f"found={r['solution_found']}, "
                f"path_cost={r['path_cost']}, "
                f"nodes={r['nodes_expanded']}, "
                f"time={r['runtime_seconds']:.6f}s"
            )
            print(line)


if __name__ == "__main__":
    main()
