"""
Export results.json to CSV.

Usage: python3 export_to_excel.py results.json [output.csv]

CSV can be opened in Excel. No extra dependencies.
"""

import csv
import json
import sys


def export_csv(results: list, output_path: str) -> None:
    """
    Export JSON result rows to a flat CSV table.

    The function auto-detects whether rows come from:
    - theory runs (heuristic + optimality metadata),
    - sweep runs (grid/config/failure metadata), or
    - simple baseline/all runs.
    """
    has_sweep = results and "grid_size" in results[0]
    has_theory = results and ("nodes_reopened" in results[0] or "optimal" in results[0])
    if has_theory:
        headers = [
            "Config", "Grid Size", "Obstacle Density", "Algorithm", "Heuristic", "Weight",
            "Found", "Path Cost", "Optimal Cost", "Optimal?", "Subopt Gap",
            "Nodes Expanded", "Nodes Reopened", "Consistency Violations",
            "Time (s)", "Seed",
        ]
    elif has_sweep:
        headers = ["Config", "Grid Size", "Obstacle Density", "Algorithm", "Found", "Path Cost", "Nodes Expanded", "Time (s)", "Failure Reason"]
    else:
        headers = ["Config", "Algorithm", "Found", "Path Cost", "Nodes Expanded", "Time (s)"]
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in results:
            found = "Yes" if r["solution_found"] else "No"
            path_cost = r["path_cost"] if r["solution_found"] else "-"
            fail = r.get("failure_reason", "")
            if has_theory:
                w.writerow([
                    r.get("config_label", ""),
                    r.get("grid_size", ""),
                    r.get("obstacle_density", ""),
                    r.get("algorithm", ""),
                    r.get("heuristic", ""),
                    r.get("heuristic_weight", ""),
                    found,
                    path_cost,
                    r.get("optimal_cost_baseline", ""),
                    r.get("optimal", ""),
                    r.get("suboptimality_gap", ""),
                    r["nodes_expanded"],
                    r.get("nodes_reopened", 0),
                    r.get("consistency_violations_detected", 0),
                    round(r["runtime_seconds"], 6),
                    r.get("seed", ""),
                ])
            elif has_sweep:
                w.writerow([
                    r.get("config_label", ""),
                    r.get("grid_size", ""),
                    r.get("obstacle_density", ""),
                    r["algorithm"],
                    found,
                    path_cost,
                    r["nodes_expanded"],
                    round(r["runtime_seconds"], 6),
                    fail,
                ])


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 export_to_excel.py results.json [output.csv]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results.csv"

    with open(input_path) as f:
        results = json.load(f)

    export_csv(results, output_path)
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    main()
