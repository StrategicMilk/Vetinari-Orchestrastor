"""Data migration scripts for evolving Vetinari's storage schemas.

Provides :func:`run_migrations` to initialise or upgrade SQLite databases
used by the plan-tracking, durable-execution, and memory subsystems.
"""

from __future__ import annotations

from vetinari.migrations.runner import run_migrations

__all__ = ["run_migrations"]
