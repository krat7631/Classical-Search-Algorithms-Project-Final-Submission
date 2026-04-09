"""
Test cases for search algorithms.

Verifies correctness on simple grids: solution found, path validity, path cost.
Uses unittest (no pytest required).
"""

import sys
import unittest

sys.path.insert(0, ".")

from environment.grid_world import GridWorld, State
from algorithms.base import SearchConstraints
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.iddfs import iddfs
from algorithms.astar import astar


# Minimal grid: 3x3, start (0,0), goal (2,2)
SIMPLE_GRID = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

# Grid with obstacle: must go around
GRID_WITH_OBSTACLE = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]

# Default 5x5 from main.py
DEFAULT_GRID = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
]


def _make_env(grid, start=(0, 0), goal=None):
    rows, cols = len(grid), len(grid[0])
    if goal is None:
        goal = (rows - 1, cols - 1)
    return GridWorld(grid, start, goal)


def _path_valid(env: GridWorld, path: list) -> bool:
    """Check that path is contiguous and stays in bounds."""
    if len(path) < 2:
        return len(path) == 1 and path[0] == env.start == env.goal
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        dr, dc = abs(b.row - a.row), abs(b.col - a.col)
        if dr + dc != 1:
            return False
        if env._is_obstacle(b.row, b.col):
            return False
    return path[0] == env.start and path[-1] == env.goal


class TestAlgorithms(unittest.TestCase):
    """Algorithm correctness tests."""

    def test_simple_grid_all_algorithms(self):
        """All algorithms find a solution on an empty 3x3 grid."""
        env = _make_env(SIMPLE_GRID)
        for algo in [bfs, dfs, iddfs, astar]:
            with self.subTest(algo=algo.__name__):
                result = algo(env, None)
                self.assertTrue(result.solution_found)
                self.assertTrue(_path_valid(env, result.path))
                self.assertEqual(result.path_cost, 4)
                self.assertGreater(result.nodes_expanded, 0)

    def test_grid_with_obstacle(self):
        """All algorithms find a solution when path exists around obstacle."""
        env = _make_env(GRID_WITH_OBSTACLE)
        for algo in [bfs, dfs, iddfs, astar]:
            with self.subTest(algo=algo.__name__):
                result = algo(env, None)
                self.assertTrue(result.solution_found)
                self.assertTrue(_path_valid(env, result.path))
                self.assertGreaterEqual(result.path_cost, 4)

    def test_bfs_optimal(self):
        """BFS returns optimal path on grid with obstacle."""
        env = _make_env(GRID_WITH_OBSTACLE)
        result = bfs(env, None)
        self.assertTrue(result.solution_found)
        self.assertEqual(result.path_cost, 4)

    def test_astar_optimal(self):
        """A* returns optimal path on grid with obstacle."""
        env = _make_env(GRID_WITH_OBSTACLE)
        result = astar(env, None)
        self.assertTrue(result.solution_found)
        self.assertEqual(result.path_cost, 4)

    def test_iddfs_optimal(self):
        """IDDFS returns optimal path on grid with obstacle."""
        env = _make_env(GRID_WITH_OBSTACLE)
        result = iddfs(env, None)
        self.assertTrue(result.solution_found)
        self.assertEqual(result.path_cost, 4)

    def test_default_grid_all_solve(self):
        """All algorithms solve the default 5x5 grid."""
        env = _make_env(DEFAULT_GRID)
        for algo in [bfs, dfs, iddfs, astar]:
            result = algo(env, None)
            self.assertTrue(result.solution_found, f"{algo.__name__} failed")
            self.assertTrue(_path_valid(env, result.path))

    def test_constraint_max_depth(self):
        """With max_depth=2, BFS may not find goal on 3x3 grid."""
        env = _make_env(SIMPLE_GRID)
        constraints = SearchConstraints(max_depth=2)
        result = bfs(env, constraints)
        self.assertFalse(result.solution_found)

    def test_constraint_max_expansions(self):
        """With very low max_expansions, solution may not be found."""
        env = _make_env(SIMPLE_GRID)
        constraints = SearchConstraints(max_expansions=3)
        result = bfs(env, constraints)
        self.assertLessEqual(result.nodes_expanded, 3)

    def test_constraint_respected(self):
        """Nodes expanded does not exceed max_expansions when set."""
        env = _make_env(DEFAULT_GRID)
        constraints = SearchConstraints(max_expansions=50)
        for algo in [bfs, dfs, iddfs, astar]:
            result = algo(env, constraints)
            self.assertLessEqual(result.nodes_expanded, 50)

    def test_start_equals_goal(self):
        """When start is goal, path has length 0."""
        env = GridWorld(SIMPLE_GRID, (1, 1), (1, 1))
        for algo in [bfs, dfs, iddfs, astar]:
            result = algo(env, None)
            self.assertTrue(result.solution_found)
            self.assertEqual(result.path_cost, 0)
            self.assertEqual(len(result.path), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
