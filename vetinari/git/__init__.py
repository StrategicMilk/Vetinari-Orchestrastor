"""Git integration — commit trailers linking decisions to code changes.

Provides ``generate_trailers()`` to create Decision-Ref trailers
that connect git commits to decision journal entries and ADRs.
"""

from __future__ import annotations

from vetinari.git.trailers import generate_trailers

__all__ = [
    "generate_trailers",
]
