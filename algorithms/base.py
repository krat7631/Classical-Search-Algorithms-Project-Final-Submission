"""
Base definitions for search algorithms.

Provides a common SearchConstraints dataclass used consistently across
all algorithms (BFS, DFS, IDDFS, A*).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchConstraints:
    """
    Explicit resource constraints for search algorithms.
    None means no constraint (baseline/unconstrained). All algorithms
    check these in the same order (time, expansions, then expand node).
    """

    max_depth: Optional[int] = None
    """Maximum search depth. None = unlimited."""

    max_expansions: Optional[int] = None
    """Maximum number of node expansions. None = unlimited."""

    time_budget_seconds: Optional[float] = None
    """Maximum wall-clock time in seconds. None = unlimited."""
