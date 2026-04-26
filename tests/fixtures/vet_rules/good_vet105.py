"""Module using dataclasses.asdict instead of manual to_dict — clean for VET105."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class TaskRecord:
    """A persisted record of a completed task."""

    task_id: str
    status: str


def task_record_to_dict(record: TaskRecord) -> dict:
    """Serialize a TaskRecord to a plain dictionary.

    Uses dataclasses.asdict so field additions to TaskRecord are picked up
    automatically without updating a manual mapping.

    Args:
        record: The record to serialize.

    Returns:
        Dictionary representation of the record.
    """
    return dataclasses.asdict(record)
