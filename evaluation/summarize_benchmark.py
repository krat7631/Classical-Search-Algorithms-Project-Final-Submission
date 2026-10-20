"""
Summarise results/benchmark_results.csv into results/benchmark_summary.json and figures 7-9.

Ratios are computed per problem against plain A* (astar_high_g) on the same problem and
then summarised by the median, so a few very long paths do not dominate.

Usage:
    python -m evaluation.summarize_benchmark
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark_results.csv"
SUMMARY = ROOT / "results" / "benchmark_summary.json"
FIGURES = ROOT / "figures"
REFERENCE = "astar_high_g"
LABELS = {
    "dijkstra": "Dijkstra (h = 0)",
    "astar_fifo": "A*, FIFO ties",
    "astar_high_g": "A*, high-g ties",
    "wastar_1.5_reopen": "WA* w=1.5, reopen",
    "wastar_1.5_no_reopen": "WA* w=1.5, no reopen",
    "wastar_2.0_reopen": "WA* w=2.0, reopen",
    "wastar_2.0_no_reopen": "WA* w=2.0, no reopen",
}


def with_problem_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Every algorithm solves a map's problems in the same order, so the n-th row of an
    algorithm on a map is problem n. (Bucket and optimal length alone are not unique.)"""
    out = df.copy()
    out["problem"] = out.groupby(["map", "algorithm"]).cumcount()
    return out


def with_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = with_problem_ids(df)
    key = ["map", "problem"]
    ref = df[df["algorithm"] == REFERENCE].set_index(key)[["nodes_expanded", "runtime_seconds"]]
    out = df.join(ref, on=key, rsuffix="_ref")
    out["expansion_ratio"] = out["nodes_expanded"] / out["nodes_expanded_ref"]
    out["runtime_ratio"] = out["runtime_seconds"] / out["runtime_seconds_ref"]
    return out


def summarize(df: pd.DataFrame) -> dict:
    df = with_ratios(df)
    per_algorithm = {}
    for algorithm, g in df.groupby("algorithm"):
        per_algorithm[algorithm] = {
            "problems": int(len(g)),
            "matches_published_optimum": round(float(g["matches_optimum"].mean()), 4),
            "mean_suboptimality_pct": round(float(g["suboptimality_pct"].mean()), 3),
            "max_suboptimality_pct": round(float(g["suboptimality_pct"].max()), 3),
            "median_nodes_expanded": int(g["nodes_expanded"].median()),
            "median_expansion_ratio_vs_astar": round(float(g["expansion_ratio"].median()), 3),
            "median_runtime_ratio_vs_astar": round(float(g["runtime_ratio"].median()), 3),
            "mean_nodes_reopened": round(float(g["nodes_reopened"].mean()), 1),
            "median_nodes_generated": int(g["nodes_generated"].median()),
        }
    weighted = df[df["weight"] > 1]
    bound_ok = (weighted["suboptimality_pct"] <= (weighted["weight"] - 1) * 100 + 1e-6).all()
    return {
        "maps": sorted(df["map"].unique().tolist()),
        "problems": int(df.drop_duplicates(["map", "problem"]).shape[0]),
        "optimal_algorithms_all_match_published_optima": bool(
            df[df["algorithm"].isin(["dijkstra", "astar_fifo", "astar_high_g"])]["matches_optimum"].all()),
        "weighted_astar_within_w_bound": bool(bound_ok),
        "algorithms": per_algorithm,
    }


def fig7_expansions_vs_length(df: pd.DataFrame) -> None:
    keep = ["dijkstra", "astar_high_g", "wastar_2.0_no_reopen"]
    d = df[df["algorithm"].isin(keep)].copy()
    d["length_bin"] = pd.cut(d["optimal_length"], bins=np.linspace(0, d["optimal_length"].max(), 12))
    agg = d.groupby(["algorithm", "length_bin"], observed=True).agg(
        length=("optimal_length", "median"), nodes=("nodes_expanded", "median")).reset_index()
    plt.figure(figsize=(9, 5))
    for algorithm in keep:
        a = agg[agg["algorithm"] == algorithm]
        plt.plot(a["length"], a["nodes"], marker="o", label=LABELS[algorithm])
    plt.yscale("log")
    plt.title("Figure 7: Nodes Expanded vs Optimal Path Length (Moving AI maps)")
    plt.xlabel("Published optimal path length")
    plt.ylabel("Median nodes expanded (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig7_benchmark_expansions.png", dpi=200)
    plt.close()


def fig8_weighted_tradeoff(summary: dict) -> None:
    rows = [{"Variant": LABELS[k], "Expansions vs A*": v["median_expansion_ratio_vs_astar"],
             "Mean path cost above optimal (%)": v["mean_suboptimality_pct"]}
            for k, v in summary["algorithms"].items() if k.startswith("wastar")]
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.barplot(data=d, x="Variant", y="Expansions vs A*", hue="Variant", legend=False, ax=axes[0], palette="mako")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Median nodes expanded relative to A*")
    sns.barplot(data=d, x="Variant", y="Mean path cost above optimal (%)", hue="Variant", legend=False,
                ax=axes[1], palette="rocket")
    axes[1].set_title("Mean suboptimality")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("")
    fig.suptitle("Figure 8: Weighted A* with and without re-expansions")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_weighted_astar_reopening.png", dpi=200)
    plt.close(fig)


def fig9_tie_breaking(df: pd.DataFrame) -> None:
    d = with_ratios(df)
    d = d[d["algorithm"] == "astar_fifo"]
    plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=d, x="map", y="expansion_ratio", hue="map_set", dodge=False)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.title("Figure 9: FIFO tie-breaking vs high-g tie-breaking (nodes expanded, ratio)")
    plt.xlabel("")
    plt.ylabel("FIFO / high-g expansions (log scale)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig9_tie_breaking.png", dpi=200)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(RESULTS)
    summary = summarize(df)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    FIGURES.mkdir(exist_ok=True)
    fig7_expansions_vs_length(df)
    fig8_weighted_tradeoff(summary)
    fig9_tie_breaking(df)
    print(json.dumps({k: v for k, v in summary.items() if k != "algorithms"}, indent=2))
    for name, s in summary["algorithms"].items():
        print(f"{name:22} optimal {s['matches_published_optimum']:.0%} | mean gap {s['mean_suboptimality_pct']:.2f}% "
              f"| max gap {s['max_suboptimality_pct']:.2f}% | expansions x{s['median_expansion_ratio_vs_astar']:.2f} "
              f"| time x{s['median_runtime_ratio_vs_astar']:.2f} | reopened {s['mean_nodes_reopened']:.0f}")


if __name__ == "__main__":
    main()
