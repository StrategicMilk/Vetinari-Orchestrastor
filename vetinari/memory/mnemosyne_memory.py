"""MnemosyneMemoryStore - Adapter for Mnemosyne-style memory OS.

This adapter provides the IMemoryStore interface using a JSON file-based
storage optimized for semantic search and memory retrieval.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .interfaces import MNEMOSYNE_PATH, IMemoryStore, MemoryEntry, MemoryEntryType, MemoryStats, content_hash

logger = logging.getLogger(__name__)


class MnemosyneMemoryStore(IMemoryStore):
    """Memory store adapter using Mnemosyne-style JSON storage."""

    def __init__(self, path: str = MNEMOSYNE_PATH):
        self.path = Path(path)
        self.data_file = self.path / "memories.json"
        self._data: dict[str, Any] = {"memories": [], "metadata": {}}
        self._load()

    def _load(self):
        """Load data from disk."""
        self.path.mkdir(parents=True, exist_ok=True)

        if self.data_file.exists():
            try:
                with open(self.data_file, encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("MnemosyneMemoryStore loaded %s entries", len(self._data.get("memories", [])))
            except json.JSONDecodeError as e:
                logger.warning("Failed to load memory data: %s, starting fresh", e)
                self._data = {"memories": [], "metadata": {}}
        else:
            self._data = {"memories": [], "metadata": {}}
            self._save()

    def _save(self):
        """Save data to disk."""
        self._data["metadata"]["last_updated"] = datetime.now().isoformat()
        self._data["metadata"]["entry_count"] = len(self._data["memories"])

        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    def remember(self, entry: MemoryEntry) -> str:
        """Store a memory entry.

        Returns:
            The result string.
        """
        if "mnemosyne" not in entry.source_backends:
            entry.source_backends = [*entry.source_backends, "mnemosyne"]

        entry_hash = content_hash(entry.content)

        memory_dict = {
            "id": entry.id,
            "agent": entry.agent,
            "entry_type": entry.entry_type.value if isinstance(entry.entry_type, MemoryEntryType) else entry.entry_type,
            "content": entry.content,
            "summary": entry.summary,
            "timestamp": entry.timestamp,
            "provenance": entry.provenance,
            "source_backends": entry.source_backends,
            "content_hash": entry_hash,
            "forgotten": False,
            "created_at": datetime.now().isoformat(),
        }

        memories = self._data.get("memories", [])

        existing_idx = None
        for i, m in enumerate(memories):
            if m.get("id") == entry.id:
                existing_idx = i
                break

        if existing_idx is not None:
            memories[existing_idx] = memory_dict
        else:
            memories.append(memory_dict)

        self._data["memories"] = memories
        self._save()

        logger.debug("MnemosyneMemoryStore: stored entry %s", entry.id)
        return entry.id

    def search(
        self, query: str, agent: str | None = None, entry_types: list[str] | None = None, limit: int = 10
    ) -> list[MemoryEntry]:
        """Search memories by keyword.

        Args:
            query: The query.
            agent: The agent.
            entry_types: The entry types.
            limit: The limit.

        Returns:
            List of results.
        """
        query_lower = query.lower()
        results = []

        for memory in self._data.get("memories", []):
            if memory.get("forgotten", False):
                continue

            if (
                query_lower not in memory.get("content", "").lower()
                and query_lower not in memory.get("summary", "").lower()
            ):
                continue

            if agent and memory.get("agent") != agent:
                continue

            if entry_types and memory.get("entry_type") not in entry_types:
                continue

            results.append(self._dict_to_entry(memory))

        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def timeline(
        self, agent: str | None = None, start_time: int | None = None, end_time: int | None = None, limit: int = 100
    ) -> list[MemoryEntry]:
        """Browse memories chronologically.

        Args:
            agent: The agent.
            start_time: The start time.
            end_time: The end time.
            limit: The limit.

        Returns:
            List of results.
        """
        results = []

        for memory in self._data.get("memories", []):
            if memory.get("forgotten", False):
                continue

            if agent and memory.get("agent") != agent:
                continue

            ts = memory.get("timestamp", 0)
            if start_time is not None and ts < start_time:
                continue
            if end_time is not None and ts > end_time:
                continue

            results.append(self._dict_to_entry(memory))

        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def ask(self, question: str, agent: str | None = None) -> list[MemoryEntry]:
        """Ask a natural language question (simplified: keyword search).

        Args:
            question: The question.
            agent: The agent.

        Returns:
            List of results.
        """
        keywords = question.lower().split()
        if not keywords:
            return []

        query = " ".join(keywords[:3])
        return self.search(query, agent=agent, limit=5)

    def export(self, path: str) -> bool:
        """Export memories to JSON.

        Returns:
            True if successful, False otherwise.
        """
        export_data = {"memories": self._data.get("memories", []), "exported_at": datetime.now().isoformat()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info("MnemosyneMemoryStore: exported %s entries to %s", len(self._data.get("memories", [])), path)
        return True

    def forget(self, entry_id: str, reason: str) -> bool:
        """Mark a memory as forgotten (tombstone).

        Args:
            entry_id: The entry id.
            reason: The reason.

        Returns:
            True if successful, False otherwise.
        """
        memories = self._data.get("memories", [])

        for memory in memories:
            if memory.get("id") == entry_id:
                memory["forgotten"] = True
                memory["forgotten_reason"] = reason
                self._save()
                logger.debug("MnemosyneMemoryStore: marked entry %s as forgotten", entry_id)
                return True

        return False

    def compact(self, max_age_days: int | None = None) -> int:
        """Remove forgotten entries and optionally prune old data.

        Returns:
            The computed value.
        """
        import time

        memories = self._data.get("memories", [])
        cutoff = int((time.time() - max_age_days * 24 * 3600) * 1000) if max_age_days else 0

        original_count = len(memories)

        memories = [
            m
            for m in memories
            if not m.get("forgotten", False) and (not max_age_days or m.get("timestamp", 0) >= cutoff)
        ]

        deleted = original_count - len(memories)
        self._data["memories"] = memories
        self._save()

        logger.info("MnemosyneMemoryStore: compacted, removed %d entries", deleted)
        return deleted

    def stats(self) -> MemoryStats:
        """Get memory store statistics.

        Returns:
            The MemoryStats result.
        """
        memories = [m for m in self._data.get("memories", []) if not m.get("forgotten", False)]

        file_size = self.data_file.stat().st_size if self.data_file.exists() else 0

        timestamps = [m.get("timestamp", 0) for m in memories]
        oldest = min(timestamps) if timestamps else 0
        newest = max(timestamps) if timestamps else 0

        by_agent: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for m in memories:
            agent = m.get("agent", "unknown")
            by_agent[agent] = by_agent.get(agent, 0) + 1

            entry_type = m.get("entry_type", "unknown")
            by_type[entry_type] = by_type.get(entry_type, 0) + 1

        return MemoryStats(
            total_entries=len(memories),
            file_size_bytes=file_size,
            oldest_entry=oldest,
            newest_entry=newest,
            entries_by_agent=by_agent,
            entries_by_type=by_type,
        )

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry by ID.

        Returns:
            The MemoryEntry | None result.
        """
        for memory in self._data.get("memories", []):
            if memory.get("id") == entry_id:
                return self._dict_to_entry(memory)
        return None

    def _dict_to_entry(self, data: dict[str, Any]) -> MemoryEntry:
        """Convert a dictionary to a MemoryEntry."""
        return MemoryEntry(
            id=data.get("id", ""),
            agent=data.get("agent", ""),
            entry_type=MemoryEntryType(data.get("entry_type", "discovery")),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            timestamp=data.get("timestamp", 0),
            provenance=data.get("provenance", ""),
            source_backends=data.get("source_backends", []),
        )
