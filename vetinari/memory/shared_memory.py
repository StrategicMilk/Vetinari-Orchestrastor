"""Shared Memory module."""

from __future__ import annotations

import json
import logging
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

warnings.warn(
    "vetinari.shared_memory is deprecated. Use vetinari.memory.dual_memory.DualMemoryStore instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


class AgentName(Enum):
    """Agent names for memory namespacing."""

    PLAN = "plan"
    BUILD = "build"
    ASK = "ask"
    REVIEW = "review"
    ORACLE = "oracle"
    CONSOLIDATED_RESEARCHER = "consolidated_researcher"
    QUALITY = "quality"
    OPERATIONS = "operations"
    RESEARCHER = "researcher"
    ORCHESTRATOR = "orchestrator"
    ARCHITECT = "architect"


@dataclass
class MemoryEntry:
    """Memory entry."""
    entry_id: str
    agent_name: str
    memory_type: str
    summary: str
    content: str
    timestamp: str
    tags: list[str] = field(default_factory=list)
    project_id: str = ""
    session_id: str = ""
    options: list[dict] = field(default_factory=list)
    resolved: bool = False
    # Phase 1 additions - plan-aware fields
    plan_id: str = ""
    wave_id: str = ""
    task_id: str = ""
    provenance: str = "agent"
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _to_dual_entry(entry: MemoryEntry):
    """Convert a legacy MemoryEntry to a DualMemoryStore MemoryEntry."""
    try:
        from vetinari.memory.interfaces import MemoryEntry as DualEntry
        from vetinari.memory.interfaces import MemoryEntryType

        entry_type = MemoryEntryType.FACT
        if entry.memory_type in ("decision", "choice"):
            entry_type = MemoryEntryType.DECISION
        elif entry.memory_type in ("insight", "learning"):
            entry_type = MemoryEntryType.INSIGHT
        return DualEntry(
            entry_id=entry.entry_id,
            entry_type=entry_type,
            content=entry.content,
            summary=entry.summary,
            tags=entry.tags,
            source_agent=entry.agent_name,
            plan_id=entry.plan_id,
            task_id=entry.task_id,
            confidence=entry.confidence,
            metadata=entry.metadata,
        )
    except Exception:
        return None


class SharedMemory:
    """Legacy SharedMemory — delegates to DualMemoryStore when available.

    All public methods first try the modern DualMemoryStore backend.
    If DualMemoryStore is not importable (e.g. during isolated tests),
    the class falls back to its original JSON-file implementation.
    """

    _instance = None

    @classmethod
    def get_instance(cls, storage_path: str | None = None) -> SharedMemory:
        """Get or create singleton instance.

        Returns:
            The SharedMemory result.
        """
        if cls._instance is None:
            cls._instance = cls(storage_path)
        return cls._instance

    def __init__(self, storage_path: str | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "memory"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.storage_path / "memory.json"
        self.entries: list[MemoryEntry] = []

        # Try to delegate to DualMemoryStore
        self._dual: Any = None
        try:
            from vetinari.memory.dual_memory import get_dual_memory_store

            self._dual = get_dual_memory_store()
            logger.info("SharedMemory delegating to DualMemoryStore")
        except Exception:
            logger.debug("DualMemoryStore unavailable, using JSON fallback")

        self._load()

    def _load(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for e in data:
                        # Backward compatibility: add default fields if missing
                        if "options" not in e:
                            e["options"] = []
                        if "resolved" not in e:
                            e["resolved"] = False
                        self.entries.append(MemoryEntry(**e))
            except Exception as e:
                logger.warning("Could not load memory: %s", e)
                self.entries = []

    def _save(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self.entries], f, indent=2)
        except Exception as e:
            logger.error("Could not save memory: %s", e)

    def add(
        self,
        agent_name: str,
        memory_type: str,
        summary: str,
        content: str,
        tags: list[str] | None = None,
        project_id: str = "",
        session_id: str = "",
        plan_id: str = "",
        wave_id: str = "",
        task_id: str = "",
        provenance: str = "agent",
    ) -> MemoryEntry:
        """Add.

        Returns:
            The MemoryEntry result.

        Args:
            agent_name: The agent name.
            memory_type: The memory type.
            summary: The summary.
            content: The content.
            tags: The tags.
            project_id: The project id.
            session_id: The session id.
            plan_id: The plan id.
            wave_id: The wave id.
            task_id: The task id.
            provenance: The provenance.
        """
        entry = MemoryEntry(
            entry_id=str(uuid.uuid4())[:8],
            agent_name=agent_name,
            memory_type=memory_type,
            summary=summary,
            content=content,
            timestamp=datetime.now().isoformat(),
            tags=tags or [],
            project_id=project_id,
            session_id=session_id,
            plan_id=plan_id,
            wave_id=wave_id,
            task_id=task_id,
            provenance=provenance,
        )

        # Delegate to DualMemoryStore
        if self._dual is not None:
            try:
                dual_entry = _to_dual_entry(entry)
                if dual_entry is not None:
                    self._dual.remember(dual_entry)
            except Exception as exc:
                logger.debug("DualMemoryStore delegation failed: %s", exc)

        self.entries.append(entry)
        self._save()

        logger.info("Memory entry added: %s (%s) by %s", entry.entry_id, memory_type, agent_name)

        return entry

    def search(
        self, query: str, agent_name: str | None = None, memory_type: str | None = None, limit: int = 10
    ) -> list[MemoryEntry]:
        """Search.

        Returns:
            List of results.

        Args:
            query: The query.
            agent_name: The agent name.
            memory_type: The memory type.
            limit: The limit.
        """
        results = self.entries

        if agent_name:
            results = [e for e in results if e.agent_name == agent_name]

        if memory_type:
            results = [e for e in results if e.memory_type == memory_type]

        if query:
            query_lower = query.lower()
            results = [e for e in results if query_lower in e.summary.lower() or query_lower in e.content.lower()]

        results = sorted(results, key=lambda x: x.timestamp, reverse=True)

        return results[:limit]

    def get_recent(self, limit: int = 20) -> list[MemoryEntry]:
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_by_agent(self, agent_name: str, limit: int = 20) -> list[MemoryEntry]:
        """Get by agent.

        Returns:
            List of results.

        Args:
            agent_name: The agent name.
            limit: The limit.
        """
        results = [e for e in self.entries if e.agent_name == agent_name]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_by_type(self, memory_type: str, limit: int = 20) -> list[MemoryEntry]:
        """Get by type.

        Returns:
            List of results.

        Args:
            memory_type: The memory type.
            limit: The limit.
        """
        results = [e for e in self.entries if e.memory_type == memory_type]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_timeline(self, limit: int = 50) -> list[MemoryEntry]:
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_stats(self) -> dict:
        """Get stats.

        Returns:
            Dictionary of results.
        """
        total = len(self.entries)

        by_type = {}
        for entry in self.entries:
            by_type[entry.memory_type] = by_type.get(entry.memory_type, 0) + 1

        by_agent = {}
        for entry in self.entries:
            by_agent[entry.agent_name] = by_agent.get(entry.agent_name, 0) + 1

        oldest = min([e.timestamp for e in self.entries]) if self.entries else None
        newest = max([e.timestamp for e in self.entries]) if self.entries else None

        return {
            "total_entries": total,
            "by_type": by_type,
            "by_agent": by_agent,
            "oldest_entry": oldest,
            "newest_entry": newest,
        }

    def get_all(self, limit: int = 100) -> list[dict]:
        """Get all memories as dicts (for API compatibility).

        Returns:
            List of results.
        """
        sorted_entries = sorted(self.entries, key=lambda x: x.timestamp, reverse=True)
        return [e.to_dict() for e in sorted_entries[:limit]]

    def get_memories_by_type(self, memory_type: str, limit: int = 20) -> list[dict]:
        """Get memories by type (for API compatibility).

        Args:
            memory_type: The memory type.
            limit: The limit.

        Returns:
            List of results.
        """
        results = self.get_by_type(memory_type, limit)
        return [e.to_dict() for e in results]

    def resolve_decision(self, decision_id: str, choice: str):
        """Mark a decision as resolved with the chosen option.

        Args:
            decision_id: The decision id.
            choice: The choice.

        Returns:
            True if successful, False otherwise.
        """
        for entry in self.entries:
            if entry.entry_id == decision_id:
                entry.content = f"{entry.content}\n\n[RESOLVED]: {choice}"
                entry.tags = [*entry.tags, "resolved"] if entry.tags else ["resolved"]
                self._save()
                logger.info("Decision %s resolved with: %s", decision_id, choice)
                return True
        return False

    def forget(self, entry_id: str) -> bool:
        """Forget.

        Returns:
            True if successful, False otherwise.
        """
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.memory_type = "forgotten"
                self._save()
                return True
        return False

    def compact(self, max_age_days: int | None = None):
        """Compact for the current context."""
        if max_age_days:
            from datetime import timedelta

            cutoff = datetime.now() - timedelta(days=max_age_days)
            self.entries = [e for e in self.entries if datetime.fromisoformat(e.timestamp) > cutoff]
            self._save()

    # Phase 1: Plan-aware methods

    def get_by_plan(self, plan_id: str, limit: int = 50) -> list[MemoryEntry]:
        """Get all memories linked to a specific plan.

        Args:
            plan_id: The plan id.
            limit: The limit.

        Returns:
            List of results.
        """
        results = [e for e in self.entries if e.plan_id == plan_id]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_by_task(self, task_id: str, limit: int = 20) -> list[MemoryEntry]:
        """Get all memories linked to a specific task.

        Args:
            task_id: The task id.
            limit: The limit.

        Returns:
            List of results.
        """
        results = [e for e in self.entries if e.task_id == task_id]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_plan_timeline(self, plan_id: str) -> list[MemoryEntry]:
        """Get timeline of all events for a plan (waves, tasks, decisions).

        Returns:
            List of results.
        """
        results = [e for e in self.entries if e.plan_id == plan_id]
        return sorted(results, key=lambda x: x.timestamp)

    def add_plan_memory(
        self,
        plan_id: str,
        agent_name: str,
        memory_type: str,
        summary: str,
        content: str,
        tags: list[str] | None = None,
        wave_id: str = "",
        task_id: str = "",
        provenance: str = "agent",
    ) -> MemoryEntry:
        """Add a memory with plan linkage."""
        return self.add(
            agent_name=agent_name,
            memory_type=memory_type,
            summary=summary,
            content=content,
            tags=tags or [],
            plan_id=plan_id,
            wave_id=wave_id,
            task_id=task_id,
            provenance=provenance,
        )

    def get_model_selections(self, limit: int = 20) -> list[MemoryEntry]:
        """Get all model selection memories."""
        return self.get_by_type("model_selection", limit)

    def migrate_to_dual_memory(self) -> int:
        """Migrate all legacy entries to DualMemoryStore.

        Returns the number of entries successfully migrated.

        Returns:
            The computed value.
        """
        if self._dual is None:
            logger.warning("DualMemoryStore not available for migration")
            return 0

        migrated = 0
        for entry in self.entries:
            try:
                dual_entry = _to_dual_entry(entry)
                if dual_entry is not None:
                    self._dual.remember(dual_entry)
                    migrated += 1
            except Exception as exc:
                logger.debug("Migration failed for %s: %s", entry.entry_id, exc)
        logger.info("Migrated %d/%d entries to DualMemoryStore", migrated, len(self.entries))
        return migrated


# Global singleton instance
shared_memory = SharedMemory.get_instance()
