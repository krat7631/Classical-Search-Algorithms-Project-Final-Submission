# Classical Search Algorithms: Implementation and Evaluation

Academic project implementing and evaluating classical AI search algorithms under explicit resource constraints.

## Problem Domain

Grid-based pathfinding: navigate from a start cell to a goal cell on a 2D grid with obstacles. Deterministic, 4-directional movement, uniform step cost of 1.

## Algorithms Implemented

| Algorithm | Description |
|-----------|-------------|
| **BFS** | Breadth-First Search; optimal for uniform costs |
| **DFS** | Depth-First Search; not optimal |
| **IDDFS** | Iterative Deepening DFS; optimal, memory-efficient |
| **A*** | A* with Manhattan distance heuristic; optimal |

All algorithms support the same resource constraints:

- `max_depth`: maximum search depth
- `max_expansions`: maximum number of node expansions
- `time_budget_seconds`: wall-clock time limit

## Project Structure

```
├── environment/
│   └── grid_world.py    # Grid environment, State, SearchResult
├── algorithms/
│   ├── base.py          # SearchConstraints
│   ├── heuristics.py    # h_zero, h_manhattan, h_scaled, h_inconsistent
│   ├── astar_theory.py  # A* with reopening, consistency checks
│   ├── bfs.py, dfs.py, iddfs.py, astar.py
├── evaluation/
│   ├── runner.py        # Experiment runner
│   └── theory_experiments.py  # BFS + A* variants, theory report
├── tests/
│   └── test_algorithms.py
├── main.py              # CLI entry point
├── run_tests.py         # Unit tests (no pytest needed)
├── show_results.py      # Format JSON results as table
├── export_to_excel.py   # Export JSON to CSV/Excel
├── make_figures.py      # Figures from final_theory.csv + sweep_results.csv
└── requirements.txt
```

## Quick Start

Clone and enter the repository:

```bash
git clone https://github.com/krat7631/Classical-Search-Algorithms-Project-Final-Submission.git
cd Classical-Search-Algorithms-Project-Final-Submission
```

Then run files in this order:

| Order | Command | Purpose |
|:-----:|---------|---------|
| 1 | `python3 main.py` | Run all four algorithms on the default grid (no constraints) |
| 2 | `python3 run_tests.py` | Run tests to verify correctness |
| 3 | `python3 main.py --all` | Compare baseline vs constrained runs |
| 3b | `python3 main.py --sweep` | Full sweep (grid sizes, densities, time stress) |
| 4 | `python3 main.py --all --format json > results.json` | Save results to JSON for analysis |
| 5 | `python3 show_results.py results.json` | Print a formatted table for your report |
| 6 | `python3 export_to_excel.py results.json results.csv` | Export to CSV (open in Excel) |

```bash
# 1. Run algorithms (baseline)
python3 main.py

# 2. Run tests
python3 run_tests.py

# 3. Compare with constraints
python3 main.py --all

# 4–6. For formatted table and CSV/Excel
python3 main.py --all --format json > results.json
python3 show_results.py results.json
python3 export_to_excel.py results.json results.csv
```

## Usage

### Run experiments

```bash
# Baseline (no constraints)
python3 main.py

# With constraints
python3 main.py --max-depth 5
python3 main.py --max-expansions 100
python3 main.py --time-budget 0.1

# Run baseline + multiple constraint configs
python3 main.py --all

# JSON output for data collection (save to file)
python3 main.py --all --format json > results.json
```

### Run tests

```bash
python3 run_tests.py
```

Or with pytest (if installed):

```bash
pytest tests/ -v
```

### Format results as a table

After generating JSON results:

```bash
python3 main.py --all --format json > results.json
python3 show_results.py results.json
```

This prints a formatted table for your Methods/Results section.

### Export to results.csv (for Excel)

To get `results.csv` (open in Excel):

```bash
python3 main.py --all --format json > results.json
python3 export_to_excel.py results.json results.csv
```

## Metrics Collected

Each run logs:

- `solution_found`: whether a path was found
- `path_cost`: cost of solution (number of steps for uniform cost)
- `path_length`: number of states in path
- `nodes_expanded`: number of node expansions
- `runtime_seconds`: wall-clock time
- `failure_reason`: when no solution (time_budget, expansion_limit, depth_limit, exhausted)
- `grid_size`: e.g. "5x5", "10x10"
- `obstacle_density`: fraction of cells blocked (0.0–1.0)

## Experimental Evaluation (Sweep)

### Variable environment scaling

Generate grids with variable size and obstacle density:

```python
from environment.grid_world import generate_grid_world

env = generate_grid_world(rows=10, cols=10, obstacle_density=0.2, seed=42)
# start=(0,0), goal=(rows-1, cols-1); deterministic when seed is set
```

### Full sweep (grid sizes, densities, constraints)

Sweeps over grid sizes (5×5, 10×10, 15×15), obstacle densities (0%, 10%, 20%), and constraints including tight time budgets (0.001s, 0.002s, 0.005s). Rows are exported in raw form so you can aggregate them differently for tables vs figures:

```bash
python3 main.py --sweep
python3 main.py --sweep --format json > sweep_results.json
```

**JSON → CSV:** You can export to any CSV filename (for example `my_sweep.csv`). The write-up and `make_figures.py` use the same pipeline with stable names: **`sweep_results.json` → `sweep_results.csv`** via `export_to_excel.py`. That mirrors the **`main.py --all`** flow, where people often save **`results.json` → `results.csv`** for a smaller baseline table—not the full multi-grid sweep.

```bash
python3 export_to_excel.py sweep_results.json sweep_results.csv
```

### Run different grid sizes

| Command | Grid sizes | Use case |
|---------|------------|----------|
| `python3 main.py --sweep --quick` | 5×5 only | Fast feedback (~10 seconds) |
| `python3 main.py --sweep --max-size 10` | 5×5, 10×10 | Skip 15×15 (~1 minute) |
| `python3 main.py --sweep` | 5×5, 10×10, 15×15 | Full sweep (15×15 can take 10+ minutes) |

Examples:

```bash
# Quick run (5×5 only, fewer configs)
python3 main.py --sweep --quick

# Sweep up to 10×10 (skip slow 15×15)
python3 main.py --sweep --max-size 10 --format json > sweep_results.json

# Full sweep including 15×15 (slow; use to show scalability limits)
python3 main.py --sweep --format json > sweep_results.json
```

### Programmatic sweep

```python
from evaluation.runner import run_sweep
from algorithms.base import SearchConstraints

results = run_sweep(
    grid_sizes=[(5,5), (10,10), (15,15)],
    obstacle_densities=[0.0, 0.1, 0.2],
    constraint_configs=[
        ("baseline", None),
        ("time_0.001", SearchConstraints(time_budget_seconds=0.001)),
    ],
    seed=42,
)
```

## Theory Experiments (Hart, Nilsson, Raphael 1968)

Validates A* theoretical properties: admissibility, consistency, reopenings, optimality.

### Heuristic variants

| Heuristic | Admissible | Consistent | Use |
|-----------|------------|------------|-----|
| `zero` | Yes | Yes | UCS baseline |
| `manhattan` | Yes | Yes | Standard A* |
| `scaled` (w≤1) | Yes | Yes | Weighted A* |
| `scaled` (w>1) | No | No | Suboptimality test |
| `inconsistent` | Yes | No | Reopening test |

### Commands

```bash
# A* with specific heuristic
python3 main.py --heuristic zero
python3 main.py --heuristic manhattan
python3 main.py --heuristic scaled --weight 1.5
python3 main.py --heuristic inconsistent

# Trace reopenings and consistency violations
python3 main.py --heuristic inconsistent --trace

# Theory report (BFS + A* variants, structured analysis)
python3 main.py --theory-report

# Theory sweep (grid sizes up to 20×20, 5 seeds)
python3 main.py --theory-sweep --format json > theory_results.json
python3 main.py --theory-sweep --max-size 10 --format json > theory_results.json

# Export theory results to CSV (any name works for your own analysis)
python3 export_to_excel.py theory_results.json theory_results.csv

# Same JSON; report + make_figures.py expect this output name:
python3 export_to_excel.py theory_results.json final_theory.csv
```

### Theory metrics

- `nodes_reopened`: nodes re-added to open after being expanded (inconsistent heuristics)
- `consistency_violations_detected`: h(n) > c(n,n') + h(n') count
- `optimal`: path cost equals BFS (optimal) cost
- `suboptimality_gap`: path_cost - optimal_cost when non-admissible

## Academic Use

The code is structured to support clear **Methods** (algorithm implementations, constraint handling) and **Results** (metrics from `run_experiment` / `run_sweep` / theory experiments, reproducible via `main.py --all`, `--sweep`, or `--theory-sweep --format json`). For tables and figures aligned with a written report, export **`sweep_results.json` → `sweep_results.csv`** and **`theory_results.json` → `final_theory.csv`**, then run `python3 make_figures.py`.


