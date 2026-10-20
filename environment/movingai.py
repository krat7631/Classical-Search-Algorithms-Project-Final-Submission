"""
Loader for the Moving AI Lab grid pathfinding benchmarks (Sturtevant, 2012).

Map files (.map):
    type octile
    height H
    width W
    map
    <H lines of W characters>

Passable terrain is '.' and 'G' (and 'S', swamp); '@', 'O', 'T' and 'W' are blocked.

Scenario files (.scen), one problem per line after a "version 1" header:
    bucket  map  width  height  start_x  start_y  goal_x  goal_y  optimal_length
x is the column and y the row. optimal_length is the published shortest-path cost
with 8-directional movement, diagonal cost sqrt(2) and no corner cutting, which is
what GridWorld(diagonal=True) implements, so every problem has an external answer
to check a search result against.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .grid_world import GridWorld

PASSABLE = {".", "G", "S"}


@dataclass(frozen=True)
class Scenario:
    bucket: int
    map_name: str
    start: tuple[int, int]  # (row, col)
    goal: tuple[int, int]
    optimal_length: float


def load_map(path: Path) -> List[List[int]]:
    """Read a .map file into a grid of 0 (free) / 1 (blocked)."""
    lines = Path(path).read_text().splitlines()
    header = {}
    i = 0
    while lines[i].strip() != "map":
        key, value = lines[i].split(maxsplit=1)
        header[key] = value.strip()
        i += 1
    if header.get("type") != "octile":
        raise ValueError(f"Unsupported map type: {header.get('type')}")
    height, width = int(header["height"]), int(header["width"])
    rows = lines[i + 1:i + 1 + height]
    if len(rows) != height or any(len(r) != width for r in rows):
        raise ValueError(f"Map body does not match {height}x{width} header")
    return [[0 if ch in PASSABLE else 1 for ch in row] for row in rows]


def load_scenarios(path: Path) -> List[Scenario]:
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].lower().startswith("version"):
        raise ValueError("Scenario file must start with a version line")
    scenarios = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        bucket, map_name = int(parts[0]), parts[1]
        sx, sy, gx, gy = (int(v) for v in parts[4:8])
        scenarios.append(Scenario(bucket, map_name, (sy, sx), (gy, gx), float(parts[8])))
    return scenarios


def scenario_world(grid: List[List[int]], scenario: Scenario) -> GridWorld:
    return GridWorld(grid, scenario.start, scenario.goal, diagonal=True)
