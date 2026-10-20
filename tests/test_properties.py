"""
Property tests: the theoretical guarantees of each algorithm, checked on many seeded
random grids rather than a few hand-picked ones.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.astar_theory import astar_theory  # noqa: E402
from algorithms.base import SearchConstraints  # noqa: E402
from algorithms.bfs import bfs  # noqa: E402
from algorithms.dfs import dfs  # noqa: E402
from algorithms.heuristics import make_heuristic  # noqa: E402
from environment.grid_world import GridWorld, State, generate_grid_world  # noqa: E402
from environment.movingai import load_map, load_scenarios  # noqa: E402

SEEDS = range(40)


def random_world(seed: int, diagonal: bool = False) -> GridWorld:
    rng = random.Random(seed)
    size = rng.randint(6, 18)
    env = generate_grid_world(size, size, obstacle_density=rng.choice([0.0, 0.15, 0.3]), seed=seed)
    return GridWorld(env.grid, (env.start.row, env.start.col), (env.goal.row, env.goal.col), diagonal=diagonal)


def assert_valid_path(env: GridWorld, path: list, cost: float) -> None:
    assert path[0] == env.start and path[-1] == env.goal
    total = 0.0
    for a, b in zip(path, path[1:]):
        step = dict(env.get_successors(a)).get(b)
        assert step is not None, f"illegal move {a} -> {b}"
        total += step
    assert total == pytest.approx(cost)


@pytest.mark.parametrize("seed", SEEDS)
def test_admissible_heuristics_match_bfs_optimum(seed):
    env = random_world(seed)
    optimum = bfs(env).path_cost
    for name in ("zero", "manhattan", "inconsistent"):
        result = astar_theory(env, make_heuristic(env, name))
        assert result.path_cost == optimum
        assert_valid_path(env, result.path, result.path_cost)


@pytest.mark.parametrize("seed", SEEDS)
def test_manhattan_never_expands_more_than_uniform_cost(seed):
    env = random_world(seed)
    informed = astar_theory(env, make_heuristic(env, "manhattan"))
    uninformed = astar_theory(env, make_heuristic(env, "zero"))
    assert informed.nodes_expanded <= uninformed.nodes_expanded


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("weight", [1.5, 2.0, 3.0])
@pytest.mark.parametrize("reopen", [True, False])
def test_weighted_astar_stays_within_its_bound(seed, weight, reopen):
    env = random_world(seed, diagonal=True)
    optimum = astar_theory(env, make_heuristic(env, "octile")).path_cost
    result = astar_theory(env, make_heuristic(env, "scaled_octile", weight), tie_break="high_g", reopen=reopen)
    assert optimum - 1e-9 <= result.path_cost <= weight * optimum + 1e-9
    assert_valid_path(env, result.path, result.path_cost)


@pytest.mark.parametrize("seed", SEEDS)
def test_octile_astar_is_optimal_and_never_reopens(seed):
    env = random_world(seed, diagonal=True)
    dijkstra = astar_theory(env, make_heuristic(env, "zero"))
    for tie_break in ("fifo", "high_g"):
        result = astar_theory(env, make_heuristic(env, "octile"), tie_break=tie_break)
        assert result.path_cost == pytest.approx(dijkstra.path_cost)
        # A consistent heuristic never finds a cheaper path to a closed node; floating-point
        # ties from sqrt(2) step costs must not be mistaken for one.
        assert result.nodes_reopened == 0
        assert result.consistency_violations_detected is None


@pytest.mark.parametrize("seed", SEEDS)
def test_dfs_paths_are_valid_even_when_not_optimal(seed):
    env = random_world(seed)
    result = dfs(env)
    assert result.path_cost >= bfs(env).path_cost
    assert_valid_path(env, result.path, result.path_cost)


@pytest.mark.parametrize("limit", [1, 5, 20])
def test_expansion_limit_is_respected(limit):
    env = generate_grid_world(15, 15, obstacle_density=0.1, seed=3)
    result = astar_theory(env, make_heuristic(env, "zero"), SearchConstraints(max_expansions=limit))
    assert result.nodes_expanded <= limit
    assert not result.solution_found and result.failure_reason == "expansion_limit"


def test_diagonal_moves_do_not_cut_corners():
    grid = [[0, 1],
            [0, 0]]
    env = GridWorld(grid, (0, 0), (1, 1), diagonal=True)
    # (0,0) -> (1,1) would squeeze past the obstacle at (0,1), so it is not allowed.
    assert State(1, 1) not in dict(env.get_successors(State(0, 0)))
    assert astar_theory(env, make_heuristic(env, "octile")).path_cost == 2.0


def test_movingai_loader_reads_maps_and_scenarios(tmp_path):
    # The wall at (0,1) blocks both diagonal shortcuts (no corner cutting), so the
    # shortest path goes down, across and up: 4 orthogonal steps.
    (tmp_path / "tiny.map").write_text("type octile\nheight 2\nwidth 3\nmap\n.@.\n...\n")
    (tmp_path / "tiny.map.scen").write_text("version 1\n0\ttiny.map\t3\t2\t0\t0\t2\t0\t4.00000000\n")
    grid = load_map(tmp_path / "tiny.map")
    assert grid == [[0, 1, 0], [0, 0, 0]]
    (scenario,) = load_scenarios(tmp_path / "tiny.map.scen")
    assert scenario.start == (0, 0) and scenario.goal == (0, 2)
    env = GridWorld(grid, scenario.start, scenario.goal, diagonal=True)
    assert astar_theory(env, make_heuristic(env, "octile")).path_cost == pytest.approx(scenario.optimal_length)
