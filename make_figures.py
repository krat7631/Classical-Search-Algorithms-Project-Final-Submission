"""
Publication-style figures from exported experiment CSVs.

Expects:
  - final_theory.csv — from export_to_excel.py on theory JSON (theory sweep output)
  - sweep_results.csv — from export_to_excel.py on sweep JSON

Column names match export_to_excel.py headers. Run from the project root:

  python3 make_figures.py

Writes PNGs under ./figures/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

THEORY_CSV = "final_theory.csv"
SWEEP_CSV = "sweep_results.csv"
OUT_DIR = "figures"


def _to_bool(series):
    # Handles True/False, Yes/No, 1/0 safely
    return series.astype(str).str.lower().map(
        {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
    )


def _clean_numeric(df, cols):
    """Safely coerce selected columns to numeric, leaving others unchanged."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fig1_astar_vs_bfs_optimality(theory: pd.DataFrame):
    """
    Figure 1:
    A* variants vs BFS baseline on path cost.
    """
    df = theory.copy()
    df = _clean_numeric(df, ["Path Cost", "Optimal Cost", "Weight"])
    keep = df["Algorithm"].astype(str).str.startswith("astar_") | (df["Algorithm"] == "bfs")
    df = df[keep].dropna(subset=["Path Cost"])

    # Use baseline config only so we compare algorithms on the same setup.
    if "Config" in df.columns:
        df = df[df["Config"] == "baseline"]

    order = [
        "bfs",
        "astar_zero",
        "astar_manhattan",
        "astar_scaled_w1.0",
        "astar_scaled_w1.2",
        "astar_scaled_w1.5",
        "astar_scaled_w2.0",
        "astar_scaled_w3.0",
        "astar_inconsistent",
    ]
    plot_df = df.groupby("Algorithm", as_index=False)["Path Cost"].mean()
    plot_df = plot_df[plot_df["Algorithm"].isin(order)].copy()
    plot_df["Algorithm"] = pd.Categorical(plot_df["Algorithm"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("Algorithm")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Algorithm", y="Path Cost", palette="viridis")
    plt.title("Figure 1: Mean Path Cost on Baseline Instance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig1_astar_vs_bfs_path_cost.png"), dpi=200)
    plt.close()


def fig2_weight_tradeoff(theory: pd.DataFrame):
    """
    Figure 2:
    Weighted A*: weight vs nodes expanded (speed proxy) and suboptimality gap.
    """
    df = theory.copy()
    df = _clean_numeric(df, ["Weight", "Nodes Expanded", "Subopt Gap"])
    # scaled heuristic rows only
    df = df[df["Heuristic"].astype(str).str.lower() == "scaled"]
    df = df.dropna(subset=["Weight"])
    if df.empty:
        return

    agg = df.groupby("Weight", as_index=False).agg(
        mean_nodes=("Nodes Expanded", "mean"),
        mean_gap=("Subopt Gap", "mean"),
    ).sort_values("Weight")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=agg, x="Weight", y="mean_nodes", marker="o", ax=ax1, color="tab:blue")
    ax1.set_ylabel("Mean Nodes Expanded", color="tab:blue")
    ax1.set_xlabel("Heuristic Weight (w)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    sns.lineplot(data=agg, x="Weight", y="mean_gap", marker="s", ax=ax2, color="tab:red")
    ax2.set_ylabel("Mean Suboptimality Gap", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Figure 2: Weighted A* Tradeoff (Efficiency vs Optimality)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_weight_tradeoff.png"), dpi=200)
    plt.close()


def fig3_consistency_reopenings(theory: pd.DataFrame):
    """
    Figure 3:
    Heuristic vs consistency violations and node reopenings.
    """
    df = theory.copy()
    df = _clean_numeric(df, ["Nodes Reopened", "Consistency Violations"])
    df["heuristic_label"] = df["Heuristic"].fillna("none").astype(str)

    agg = df.groupby("heuristic_label", as_index=False).agg(
        reopened=("Nodes Reopened", "mean"),
        violations=("Consistency Violations", "mean"),
    )

    melted = agg.melt(id_vars="heuristic_label", value_vars=["reopened", "violations"],
                      var_name="Metric", value_name="Mean Value")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="heuristic_label", y="Mean Value", hue="Metric", palette="Set2")
    plt.title("Figure 3: Consistency Violations and Reopenings by Heuristic")
    plt.xlabel("Heuristic")
    plt.ylabel("Mean Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig3_consistency_reopenings.png"), dpi=200)
    plt.close()


def fig4_admissible_heuristic_comparison(theory: pd.DataFrame):
    """
    Figure 4:
    Admissible heuristics comparison on nodes expanded.
    """
    df = theory.copy()
    df = _clean_numeric(df, ["Weight", "Nodes Expanded"])

    # admissible set: zero, manhattan, scaled with w<=1
    h = df["Heuristic"].astype(str).str.lower()
    admissible = (h.isin(["zero", "manhattan"])) | ((h == "scaled") & (df["Weight"] <= 1.0))
    df = df[admissible].copy()
    if df.empty:
        return

    df["label"] = df.apply(
        lambda r: f"scaled_w={r['Weight']:.1f}" if str(r["Heuristic"]).lower() == "scaled" else str(r["Heuristic"]),
        axis=1
    )
    agg = df.groupby("label", as_index=False)["Nodes Expanded"].mean().sort_values("Nodes Expanded")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=agg, x="label", y="Nodes Expanded", palette="magma")
    plt.title("Figure 4: Admissible Heuristics vs Mean Nodes Expanded")
    plt.xlabel("Heuristic")
    plt.ylabel("Mean Nodes Expanded")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig4_admissible_nodes.png"), dpi=200)
    plt.close()


def fig5_failure_reason_distribution(sweep: pd.DataFrame):
    """
    Figure 5:
    Failure reasons per algorithm (constraint extension).
    """
    df = sweep.copy()
    # normalize found column
    if "Found" in df.columns:
        found = _to_bool(df["Found"])
        df = df[found == False]  # failures only
    else:
        return

    # normalize failure reason label
    if "Failure Reason" not in df.columns:
        return

    df["Failure Reason"] = df["Failure Reason"].fillna("unknown").astype(str)
    counts = df.groupby(["Algorithm", "Failure Reason"]).size().reset_index(name="Count")
    totals = counts.groupby("Algorithm")["Count"].transform("sum")
    counts["Percent"] = 100 * counts["Count"] / totals

    plt.figure(figsize=(11, 6))
    sns.barplot(data=counts, x="Algorithm", y="Percent", hue="Failure Reason")
    plt.title("Figure 5: Failure Reason Distribution by Algorithm")
    plt.ylabel("Failure Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig5_failure_reasons.png"), dpi=200)
    plt.close()


def fig6_scaling_runtime(sweep: pd.DataFrame):
    """
    Figure 6:
    Scaling trend: grid size vs runtime for each algorithm.
    """
    df = sweep.copy()
    df = _clean_numeric(df, ["Time (s)"])
    if "Grid Size" not in df.columns:
        return

    # convert "5x5" -> 5
    df["size"] = df["Grid Size"].astype(str).str.split("x").str[0].astype(float)
    agg = df.groupby(["size", "Algorithm"], as_index=False)["Time (s)"].mean()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=agg, x="size", y="Time (s)", hue="Algorithm", marker="o")
    # Log scale keeps the faster methods visible when IDDFS dominates runtime.
    plt.yscale("log")
    plt.title("Figure 6: Mean Runtime vs Grid Size (Log Scale)")
    plt.xlabel("Grid Size (N for NxN)")
    plt.ylabel("Mean Runtime (s, log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig6_scaling_runtime.png"), dpi=200)
    plt.close()


def main() -> None:
    """Load theory + sweep CSVs and regenerate all report figures."""
    os.makedirs(OUT_DIR, exist_ok=True)

    theory = pd.read_csv(THEORY_CSV)
    sweep = pd.read_csv(SWEEP_CSV)

    fig1_astar_vs_bfs_optimality(theory)
    fig2_weight_tradeoff(theory)
    fig3_consistency_reopenings(theory)
    fig4_admissible_heuristic_comparison(theory)
    fig5_failure_reason_distribution(sweep)
    fig6_scaling_runtime(sweep)

    print(f"Done. Figures saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
