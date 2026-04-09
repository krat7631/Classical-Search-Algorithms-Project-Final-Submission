"""
Simple test runner. Run with: python3 run_tests.py

Uses unittest so no pytest install is required.
"""

import sys
import unittest

sys.path.insert(0, ".")

from environment.grid_world import GridWorld, generate_grid_world
from algorithms.base import SearchConstraints
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.iddfs import iddfs
from algorithms.astar import astar
from evaluation.theory_experiments import run_theory_experiment

SIMPLE_GRID = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
GRID_WITH_OBSTACLE = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
DEFAULT_GRID = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
]


def make_env(grid, start=(0, 0), goal=None):
    r, c = len(grid), len(grid[0])
    goal = goal or (r - 1, c - 1)
    return GridWorld(grid, start, goal)


def path_valid(env, path):
    if len(path) < 2:
        return len(path) == 1 and path[0] == env.start == env.goal
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if abs(b.row - a.row) + abs(b.col - a.col) != 1:
            return False
        if env._is_obstacle(b.row, b.col):
            return False
    return path[0] == env.start and path[-1] == env.goal


class TestAlgorithms(unittest.TestCase):
    """Regression tests covering correctness, constraints, and theory hooks."""

    def test_simple_grid_all(self):
        env = make_env(SIMPLE_GRID)
        for algo in [bfs, dfs, iddfs, astar]:
            r = algo(env, None)
            self.assertTrue(r.solution_found, f"{algo.__name__} failed")
            self.assertTrue(path_valid(env, r.path))
            self.assertEqual(r.path_cost, 4)

    def test_grid_with_obstacle_optimal(self):
        env = make_env(GRID_WITH_OBSTACLE)
        for algo in [bfs, iddfs, astar]:
            r = algo(env, None)
            self.assertTrue(r.solution_found)
            self.assertEqual(r.path_cost, 4)

    def test_default_grid_all_solve(self):
        env = make_env(DEFAULT_GRID)
        for algo in [bfs, dfs, iddfs, astar]:
            r = algo(env, None)
            self.assertTrue(r.solution_found, f"{algo.__name__} failed")
            self.assertTrue(path_valid(env, r.path))

    def test_constraint_max_depth(self):
        env = make_env(SIMPLE_GRID)
        r = bfs(env, SearchConstraints(max_depth=2))
        self.assertFalse(r.solution_found)

    def test_constraint_max_expansions(self):
        env = make_env(DEFAULT_GRID)
        for algo in [bfs, dfs, iddfs, astar]:
            r = algo(env, SearchConstraints(max_expansions=50))
            self.assertLessEqual(r.nodes_expanded, 50)

    def test_failure_reason(self):
        env = make_env(SIMPLE_GRID)
        r = bfs(env, SearchConstraints(max_depth=2))
        self.assertFalse(r.solution_found)
        self.assertEqual(r.failure_reason, "depth_limit")

    def test_generate_grid_world(self):
        env = generate_grid_world(5, 5, obstacle_density=0.1, seed=42)
        self.assertEqual(env.rows, 5)
        self.assertEqual(env.cols, 5)
        self.assertEqual((env.start.row, env.start.col), (0, 0))
        self.assertEqual((env.goal.row, env.goal.col), (4, 4))
        r = bfs(env, None)
        self.assertTrue(r.solution_found)

    def test_theory_experiment(self):
        env = make_env(SIMPLE_GRID)
        results = run_theory_experiment(env, [("baseline", None)])
        bfs_row = next(r for r in results if r["algorithm"] == "bfs")
        astar_zero = next(r for r in results if r["algorithm"] == "astar_zero")
        self.assertEqual(bfs_row["path_cost"], astar_zero["path_cost"])
        self.assertTrue(astar_zero["optimal"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
