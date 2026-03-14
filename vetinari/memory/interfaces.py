"""Memory Abstraction Layer - Unified interface for dual memory backends.

This module provides the IMemoryStore interface and supporting types
for Vetinari's dual-memory backend architecture.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from vetinari.types import MemoryType as MemoryEntryType  # canonical — MemoryType is the superset

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry in the store."""

    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:8]}")
    agent: str = ""
    entry_type: MemoryEntryType = MemoryEntryType.DISCOVERY
    content: str = ""
    summary: str = ""
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1000))
    provenance: str = ""
    source_backends: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            The result string.
        """
        data = asdict(self)
        if isinstance(self.entry_type, MemoryEntryType):
            data["entry_type"] = self.entry_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create from dictionary.

        Returns:
            The MemoryEntry result.
        """
        if "entry_type" in data and isinstance(data["entry_type"], str):
            data["entry_type"] = MemoryEntryType(data["entry_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MemoryStats:
    """Statistics about the memory store."""

    total_entries: int = 0
    file_size_bytes: int = 0
    oldest_entry: int = 0
    newest_entry: int = 0
    entries_by_agent: dict[str, int] = field(default_factory=dict)
    entries_by_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalDetails:
    """Details about an approval decision.

    JSON Schema:
    {
        "audit_id": "string (required, auto-generated)",
        "task_id": "string (optional)",
        "plan_id": "string (required)",
        "task_type": "string (optional)",
        "approval_status": "string (required: approved/rejected/auto_approved)",
        "approver": "string (required)",
        "reason": "string (optional)",
        "risk_score": "float (optional)",
        "timestamp": "string (required, auto-generated)"
    }
    """

    audit_id: str = ""
    task_id: str | None = None
    plan_id: str = ""
    task_type: str = ""
    approval_status: str = ""  # "approved", "rejected", "auto_approved"
    approver: str = ""
    reason: str = ""
    risk_score: float | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalDetails:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()


class IMemoryStore(ABC):
    """Abstract interface for memory stores."""

    @abstractmethod
    def remember(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns entry ID."""

    @abstractmethod
    def search(
        self, query: str, agent: str | None = None, entry_types: list[str] | None = None, limit: int = 10
    ) -> list[MemoryEntry]:
        """Search memories by keyword or semantic similarity.

        Args:
            query: The query.
            agent: The agent.
            entry_types: The entry types.
            limit: The limit.
        """

    @abstractmethod
    def timeline(
        self, agent: str | None = None, start_time: int | None = None, end_time: int | None = None, limit: int = 100
    ) -> list[MemoryEntry]:
        """Browse memories chronologically.

        Args:
            agent: The agent.
            start_time: The start time.
            end_time: The end time.
            limit: The limit.
        """

    @abstractmethod
    def ask(self, question: str, agent: str | None = None) -> list[MemoryEntry]:
        """Ask a natural language question against the memory store.

        Args:
            question: The question.
            agent: The agent.
        """

    @abstractmethod
    def export(self, path: str) -> bool:
        """Export memories to a JSON file."""

    @abstractmethod
    def forget(self, entry_id: str, reason: str) -> bool:
        """Mark a memory as obsolete (tombstone).

        Args:
            entry_id: The entry id.
            reason: The reason.
        """

    @abstractmethod
    def compact(self, max_age_days: int | None = None) -> int:
        """Remove forgotten entries and optionally prune old data."""

    @abstractmethod
    def stats(self) -> MemoryStats:
        """Get memory store statistics."""

    @abstractmethod
    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry by ID."""


# Environment configuration
MEMORY_BACKEND_MODE = os.environ.get("MEMORY_BACKEND_MODE", "dual")
OC_MEMORY_PATH = os.environ.get("OC_MEMORY_PATH", "./.vetinari/mnemoria")
MNEMOSYNE_PATH = os.environ.get("MNEMOSYNE_PATH", "./.vetinari/mnemosyne")
MEMORY_PRIMARY_READ = os.environ.get("MEMORY_PRIMARY_READ", "oc")
MEMORY_DEDUP_ENABLED = os.environ.get("MEMORY_DEDUP_ENABLED", "true").lower() in ("1", "true", "yes")
MEMORY_MERGE_LIMIT = int(os.environ.get("MEMORY_MERGE_LIMIT", "100"))
