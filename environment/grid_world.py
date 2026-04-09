"""
Grid World Environment for Classical Search Algorithms

A deterministic grid-based pathfinding domain where an agent moves from
a start cell to a goal cell, avoiding obstacles.
- Actions: 4-directional movement (up, down, left, right)
- Cost: uniform 1 per step (for path cost metrics)

Supports variable grid sizes and obstacle density via generate_grid_world().
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class State:
    """Represents a cell position (row, col) in the grid."""
    row: int
    col: int

    def __iter__(self) -> Iterator[int]:
        yield self.row
        yield self.col


@dataclass
class SearchResult:
    """Result of running a search algorithm."""
    solution_found: bool
    path: List[State] = field(default_factory=list)
    path_cost: float = 0.0
    nodes_expanded: int = 0
    runtime_seconds: float = 0.0
    # When solution_found=False: why the search stopped (for progress report we care about these)
    failure_reason: Optional[str] = None  # time_budget, expansion_limit, depth_limit, exhausted
    # A* theory metrics (optional)
    nodes_reopened: Optional[int] = None
    consistency_violations_detected: Optional[int] = None
    optimal: Optional[bool] = None  # path_cost equals optimal (e.g. BFS) cost
    suboptimality_gap: Optional[float] = None  # path_cost - optimal_cost when non-admissible

    def to_dict(self) -> dict:
        d = {
            "solution_found": self.solution_found,
            "path_length": len(self.path),
            "path_cost": self.path_cost,
            "nodes_expanded": self.nodes_expanded,
            "runtime_seconds": self.runtime_seconds,
        }
        if self.failure_reason is not None:
            d["failure_reason"] = self.failure_reason
        if self.nodes_reopened is not None:
            d["nodes_reopened"] = self.nodes_reopened
        if self.consistency_violations_detected is not None:
            d["consistency_violations_detected"] = self.consistency_violations_detected
        if self.optimal is not None:
            d["optimal"] = self.optimal
        if self.suboptimality_gap is not None:
            d["suboptimality_gap"] = self.suboptimality_gap
        return d


class GridWorld:
    """
    Deterministic grid environment for pathfinding.
    - 0: traversable
    - 1: obstacle
    """

    # Actions: (delta_row, delta_col). Cost is 1 for all actions.
    ACTIONS: List[Tuple[int, int]] = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
    ]

    def __init__(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ):
        """
        Initialize the grid world.

        Args:
            grid: 2D list where 0=traversable, 1=obstacle
            start: (row, col) start position
            goal: (row, col) goal position
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.start = State(start[0], start[1])
        self.goal = State(goal[0], goal[1])
        self._validate()

    def _validate(self) -> None:
        """Validate grid, start, and goal."""
        if self.rows == 0 or self.cols == 0:
            raise ValueError("Grid must be non-empty")
        if not self._in_bounds(self.start.row, self.start.col):
            raise ValueError("Start position out of bounds")
        if not self._in_bounds(self.goal.row, self.goal.col):
            raise ValueError("Goal position out of bounds")
        if self._is_obstacle(self.start.row, self.start.col):
            raise ValueError("Start position is an obstacle")
        if self._is_obstacle(self.goal.row, self.goal.col):
            raise ValueError("Goal position is an obstacle")

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_obstacle(self, row: int, col: int) -> bool:
        return self.grid[row][col] == 1

    def is_goal(self, state: State) -> bool:
        return state == self.goal

    def get_successors(self, state: State) -> List[Tuple[State, float]]:
        """
        Return list of (successor_state, step_cost) for valid 4-neighbour moves.
        Step cost is 1 (uniform cost); used by BFS/DFS and later by A*.
        """
        successors = []
        for dr, dc in self.ACTIONS:
            nr, nc = state.row + dr, state.col + dc
            if self._in_bounds(nr, nc) and not self._is_obstacle(nr, nc):
                successors.append((State(nr, nc), 1.0))
        return successors

    def manhattan_distance(self, state: State) -> float:
        """
        Admissible heuristic for A*: Manhattan distance to goal.
        Never overestimates true cost (each step moves at most 1 unit).
        """
        return abs(state.row - self.goal.row) + abs(state.col - self.goal.col)

    def get_grid_size_str(self) -> str:
        """Return grid size as 'rowsxcols' for logging."""
        return f"{self.rows}x{self.cols}"

    def get_obstacle_density(self) -> float:
        """Return fraction of cells that are obstacles (0.0 to 1.0)."""
        total = self.rows * self.cols
        obstacles = sum(row.count(1) for row in self.grid)
        return obstacles / total if total > 0 else 0.0


def build_path(parent: dict, state: State) -> List[State]:
    """Reconstruct path from goal to start using parent pointers."""
    path = []
    current = state
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path


def _is_reachable(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """BFS to check if goal is reachable from start on grid (0=free, 1=obstacle)."""
    rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return False
    visited = set()
    q = deque([start])
    visited.add(start)
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr][nc] == 0:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return False


def generate_grid_world(
    rows: int,
    cols: int,
    obstacle_density: float = 0.0,
    seed: Optional[int] = None,
) -> GridWorld:
    """
    Generate a grid world with variable size and obstacle density.

    Args:
        rows: grid height
        cols: grid width
        obstacle_density: fraction of cells that are obstacles (0.0 to 1.0, e.g. 0.1 = 10%)
        seed: random seed for deterministic generation; ensures start and goal remain reachable

    Returns:
        GridWorld with start=(0,0), goal=(rows-1, cols-1)
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    start = (0, 0)
    goal = (rows - 1, cols - 1)
    # Cells that must stay free
    protected = {start, goal}

    for attempt in range(100):  # max retries to get connected grid
        if seed is not None:
            rng = random.Random(seed + attempt)
        grid = [[0] * cols for _ in range(rows)]
        candidates = [
            (r, c) for r in range(rows) for c in range(cols) if (r, c) not in protected
        ]
        n_obstacles = int(len(candidates) * obstacle_density)
        chosen = rng.sample(candidates, min(n_obstacles, len(candidates)))
        for r, c in chosen:
            grid[r][c] = 1

        if _is_reachable(grid, start, goal):
            return GridWorld(grid, start, goal)

    # Fallback: empty grid if density too high
    return GridWorld([[0] * cols for _ in range(rows)], start, goal)
