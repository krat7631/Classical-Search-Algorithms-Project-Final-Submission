"""
Moving AI benchmark evaluation: correctness against published optima, and the cost of
search on real game and city maps.

For each map, problems are sampled evenly across the scenario file's difficulty buckets
(bucket = optimal length // 4), then every algorithm solves every problem with
8-directional movement:

- dijkstra                 A* with h = 0 (uniform-cost search): the uninformed baseline
- astar_fifo               A* + octile heuristic, ties broken first-in-first-out
- astar_high_g             A* + octile heuristic, ties broken towards larger g
- wastar_<w>_reopen        weighted A*, h = w * octile, re-expanding improved closed nodes
- wastar_<w>_no_reopen     the same without re-expansions (w-suboptimality still guaranteed)

Each optimal algorithm's path cost is compared with the scenario's published optimal
length; weighted A* reports its suboptimality, which must stay within (w - 1) * 100%.

Usage:
    python scripts/download_benchmarks.py
    python -m evaluation.benchmark                      # all maps, ~15 minutes
    python -m evaluation.benchmark --per-map 5 --quick  # pipeline check
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.astar_theory import astar_theory  # noqa: E402
from algorithms.heuristics import make_heuristic  # noqa: E402
from environment.movingai import load_map, load_scenarios, scenario_world  # noqa: E402

MAPS = {
    "dao": ["arena2", "brc202d", "den520d", "lak303d", "ost003d", "orz100d"],
    "street": ["Berlin_0_256", "Boston_0_256", "London_0_256", "NewYork_0_256"],
}
# name: (heuristic, weight, tie_break, reopen)
ALGORITHMS = {
    "dijkstra": ("zero", 1.0, "fifo", True),
    "astar_fifo": ("octile", 1.0, "fifo", True),
    "astar_high_g": ("octile", 1.0, "high_g", True),
    "wastar_1.5_reopen": ("scaled_octile", 1.5, "high_g", True),
    "wastar_1.5_no_reopen": ("scaled_octile", 1.5, "high_g", False),
    "wastar_2.0_reopen": ("scaled_octile", 2.0, "high_g", True),
    "wastar_2.0_no_reopen": ("scaled_octile", 2.0, "high_g", False),
}
TOLERANCE = 1e-4  # published optima are rounded to 8 decimal places
FIELDS = ["map_set", "map", "rows", "cols", "bucket", "start", "goal", "optimal_length", "algorithm", "weight", "path_cost",
          "matches_optimum", "suboptimality_pct", "nodes_expanded", "nodes_reopened", "nodes_generated",
          "runtime_seconds"]


def sample_by_bucket(scenarios: list, per_map: int) -> list:
    """Evenly spaced problems from the bucket-ordered list, so short and long paths are both represented."""
    ordered = sorted(scenarios, key=lambda s: s.bucket)
    if per_map >= len(ordered):
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (per_map - 1))] for i in range(per_map)]


def run(per_map: int, algorithms: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for map_set, names in MAPS.items():
            for name in names:
                grid = load_map(ROOT / "benchmarks" / map_set / "map" / f"{name}.map")
                scenarios = sample_by_bucket(
                    load_scenarios(ROOT / "benchmarks" / map_set / "scen" / f"{name}.map.scen"), per_map)
                map_started = time.time()
                for scenario in scenarios:
                    env = scenario_world(grid, scenario)
                    for algorithm, (heuristic, weight, tie_break, reopen) in algorithms.items():
                        result = astar_theory(env, make_heuristic(env, heuristic, weight),
                                              tie_break=tie_break, reopen=reopen)
                        gap = (result.path_cost - scenario.optimal_length) / scenario.optimal_length * 100 \
                            if scenario.optimal_length else 0.0
                        writer.writerow({
                            "map_set": map_set, "map": name, "rows": env.rows, "cols": env.cols,
                            "bucket": scenario.bucket, "start": f"{scenario.start[0]},{scenario.start[1]}",
                            "goal": f"{scenario.goal[0]},{scenario.goal[1]}", "optimal_length": scenario.optimal_length,
                            "algorithm": algorithm, "weight": weight, "path_cost": round(result.path_cost, 6),
                            "matches_optimum": abs(result.path_cost - scenario.optimal_length) < TOLERANCE,
                            "suboptimality_pct": round(gap, 4), "nodes_expanded": result.nodes_expanded,
                            "nodes_reopened": result.nodes_reopened, "nodes_generated": result.nodes_generated,
                            "runtime_seconds": round(result.runtime_seconds, 5),
                        })
                fh.flush()
                print(f"{map_set}/{name}: {len(scenarios)} problems in {time.time() - map_started:.0f}s "
                      f"(total {time.time() - started:.0f}s)", flush=True)
    print(f"Wrote {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-map", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Skip Dijkstra, the slowest algorithm")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark_results.csv")
    args = parser.parse_args()
    algorithms = {k: v for k, v in ALGORITHMS.items() if not (args.quick and k == "dijkstra")}
    run(args.per_map, algorithms, args.output)


if __name__ == "__main__":
    main()
