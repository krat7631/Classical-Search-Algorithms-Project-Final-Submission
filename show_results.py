"""
Load JSON results and print a formatted table for Methods/Results section.

Usage: python3 show_results.py results.json
"""

import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 show_results.py results.json")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        results = json.load(f)

    # Detect schema shape from keys in the first row.
    # Sweep/theory outputs include grid metadata; plain --all output may not.
    has_sweep = results and "grid_size" in results[0]

    if has_sweep:
        header = f"{'Grid':<8} {'Density':<8} {'Config':<18} {'Algo':<6} {'Found':<6} {'Cost':<6} {'Nodes':<6} {'Time(s)':<10} {'Fail':<12}"
    else:
        header = f"{'Config':<25} {'Algorithm':<8} {'Found':<6} {'Path Cost':<10} {'Nodes':<8} {'Time (s)':<12}"
    print(header)
    print("-" * len(header))

    for r in results:
        found = "Yes" if r["solution_found"] else "No"
        cost = r["path_cost"] if r["solution_found"] else "-"
        fail = r.get("failure_reason", "-")
        cfg = r.get("config_label", "?")
        if has_sweep:
            gs = r.get("grid_size", "?")
            dens = r.get("obstacle_density", "?")
            print(
                f"{gs:<8} {dens:<8} {cfg:<18} {r['algorithm']:<6} {found:<6} "
                f"{str(cost):<6} {r['nodes_expanded']:<6} {r['runtime_seconds']:<10.6f} {str(fail):<12}"
            )
        else:
            print(
                f"{cfg:<25} {r['algorithm']:<8} {found:<6} "
                f"{str(cost):<10} {r['nodes_expanded']:<8} {r['runtime_seconds']:<12.6f}"
            )


if __name__ == "__main__":
    main()
