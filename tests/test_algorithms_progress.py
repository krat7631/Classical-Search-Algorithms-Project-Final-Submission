"""
Progress-report test suite: BFS and DFS only.

Verifies that (1) both algorithms find a path on a small grid when unconstrained,
and (2) they correctly respect resource limits (max_depth, max_expansions) and
report the expected failure_reason when a limit is hit.
"""

import unittest

from environment.grid_world import GridWorld
from algorithms.base import SearchConstraints
from algorithms.bfs import bfs
from algorithms.dfs import dfs


class TestBFSDFSProgress(unittest.TestCase):
    """Tests for BFS and DFS suitable for the first progress report."""

    def setUp(self) -> None:
        # Same 5x5 grid as main_progress (0=free, 1=obstacle); start (0,0), goal (4,4).
        self.grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ]
        self.start = (0, 0)
        self.goal = (4, 4)
        self.env = GridWorld(self.grid, self.start, self.goal)

    def test_bfs_finds_path_without_constraints(self) -> None:
        """BFS should find a path when no constraints are set."""
        result = bfs(self.env)
        self.assertTrue(result.solution_found)
        # On unit-cost grids, BFS is optimal; path_cost = number of steps.
        self.assertGreater(result.path_cost, 0.0)

    def test_dfs_finds_path_without_constraints(self) -> None:
        """DFS should find some path (not necessarily optimal)."""
        result = dfs(self.env)
        self.assertTrue(result.solution_found)
        self.assertGreater(result.path_cost, 0.0)

    def test_bfs_respects_max_depth(self) -> None:
        """When max_depth is too small to reach the goal, BFS should stop and report depth_limit."""
        constraints = SearchConstraints(max_depth=1)
        result = bfs(self.env, constraints)
        self.assertFalse(result.solution_found)
        self.assertEqual(result.failure_reason, "depth_limit")

    def test_dfs_respects_max_expansions(self) -> None:
        """When max_expansions is hit before finding the goal, we expect expansion_limit."""
        constraints = SearchConstraints(max_expansions=1)
        result = dfs(self.env, constraints)
        self.assertFalse(result.solution_found)
        self.assertEqual(result.failure_reason, "expansion_limit")


if __name__ == "__main__":
    unittest.main()

