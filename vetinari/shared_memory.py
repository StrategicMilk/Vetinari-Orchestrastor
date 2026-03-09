import json
import logging
import time
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid

from vetinari.types import MemoryType  # canonical enum from types.py

warnings.warn(
    "vetinari.shared_memory is deprecated. Use vetinari.memory.dual_memory.DualMemoryStore instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

class AgentName(Enum):
    """Legacy agent names — kept for backward compatibility."""
    PLAN = "plan"
    BUILD = "build"
    ASK = "ask"
    REVIEW = "review"
    ORACLE = "oracle"
    EXPLORER = "explorer"
    LIBRARIAN = "librarian"
    UI_PLANNER = "ui_planner"
    RESEARCHER = "researcher"
    EVALUATOR = "evaluator"
    SYNTHESIZER = "synthesizer"

@dataclass
class MemoryEntry:
    entry_id: str
    agent_name: str
    memory_type: str
    summary: str
    content: str
    timestamp: str
    tags: List[str] = field(default_factory=list)
    project_id: str = ""
    session_id: str = ""
    options: List[Dict] = field(default_factory=list)
    resolved: bool = False
    # Phase 1 additions - plan-aware fields
    plan_id: str = ""
    wave_id: str = ""
    task_id: str = ""
    provenance: str = "agent"
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _to_dual_entry(entry: MemoryEntry):
    """Convert a legacy MemoryEntry to a DualMemoryStore MemoryEntry."""
    try:
        from vetinari.memory.interfaces import MemoryEntry as DualEntry, MemoryEntryType
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
    def get_instance(cls, storage_path: str = None) -> "SharedMemory":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(storage_path)
        return cls._instance

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "memory"

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.storage_path / "memory.json"
        self.entries: List[MemoryEntry] = []

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
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    for e in data:
                        # Backward compatibility: add default fields if missing
                        if 'options' not in e:
                            e['options'] = []
                        if 'resolved' not in e:
                            e['resolved'] = False
                        self.entries.append(MemoryEntry(**e))
            except Exception as e:
                logger.warning("Could not load memory: %s", e)
                self.entries = []

    def _save(self):
        try:
            with open(self.memory_file, 'w') as f:
                json.dump([e.to_dict() for e in self.entries], f, indent=2)
        except Exception as e:
            logger.error("Could not save memory: %s", e)

    def add(
        self,
        agent_name: str,
        memory_type: str,
        summary: str,
        content: str,
        tags: List[str] = None,
        project_id: str = "",
        session_id: str = "",
        plan_id: str = "",
        wave_id: str = "",
        task_id: str = "",
        provenance: str = "agent"
    ) -> MemoryEntry:
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
            provenance=provenance
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
        self,
        query: str,
        agent_name: str = None,
        memory_type: str = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        results = self.entries

        if agent_name:
            results = [e for e in results if e.agent_name == agent_name]

        if memory_type:
            results = [e for e in results if e.memory_type == memory_type]

        if query:
            query_lower = query.lower()
            results = [
                e for e in results
                if query_lower in e.summary.lower() or query_lower in e.content.lower()
            ]

        results = sorted(results, key=lambda x: x.timestamp, reverse=True)

        return results[:limit]

    def get_recent(self, limit: int = 20) -> List[MemoryEntry]:
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_by_agent(self, agent_name: str, limit: int = 20) -> List[MemoryEntry]:
        results = [e for e in self.entries if e.agent_name == agent_name]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_by_type(self, memory_type: str, limit: int = 20) -> List[MemoryEntry]:
        results = [e for e in self.entries if e.memory_type == memory_type]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_timeline(self, limit: int = 50) -> List[MemoryEntry]:
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_stats(self) -> Dict:
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
            "newest_entry": newest
        }

    def get_all(self, limit: int = 100) -> List[Dict]:
        """Get all memories as dicts (for API compatibility)."""
        sorted_entries = sorted(self.entries, key=lambda x: x.timestamp, reverse=True)
        return [e.to_dict() for e in sorted_entries[:limit]]

    def get_memories_by_type(self, memory_type: str, limit: int = 20) -> List[Dict]:
        """Get memories by type (for API compatibility)."""
        results = self.get_by_type(memory_type, limit)
        return [e.to_dict() for e in results]

    def resolve_decision(self, decision_id: str, choice: str):
        """Mark a decision as resolved with the chosen option."""
        for entry in self.entries:
            if entry.entry_id == decision_id:
                entry.content = f"{entry.content}\n\n[RESOLVED]: {choice}"
                entry.tags = entry.tags + ["resolved"] if entry.tags else ["resolved"]
                self._save()
                logger.info("Decision %s resolved with: %s", decision_id, choice)
                return True
        return False

    def forget(self, entry_id: str) -> bool:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.memory_type = "forgotten"
                self._save()
                return True
        return False

    def compact(self, max_age_days: int = None):
        if max_age_days:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(days=max_age_days)
            self.entries = [
                e for e in self.entries
                if datetime.fromisoformat(e.timestamp) > cutoff
            ]
            self._save()

    # Phase 1: Plan-aware methods

    def get_by_plan(self, plan_id: str, limit: int = 50) -> List[MemoryEntry]:
        """Get all memories linked to a specific plan."""
        results = [e for e in self.entries if e.plan_id == plan_id]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_by_task(self, task_id: str, limit: int = 20) -> List[MemoryEntry]:
        """Get all memories linked to a specific task."""
        results = [e for e in self.entries if e.task_id == task_id]
        return sorted(results, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_plan_timeline(self, plan_id: str) -> List[MemoryEntry]:
        """Get timeline of all events for a plan (waves, tasks, decisions)."""
        results = [e for e in self.entries if e.plan_id == plan_id]
        return sorted(results, key=lambda x: x.timestamp)

    def add_plan_memory(
        self,
        plan_id: str,
        agent_name: str,
        memory_type: str,
        summary: str,
        content: str,
        tags: List[str] = None,
        wave_id: str = "",
        task_id: str = "",
        provenance: str = "agent"
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
            provenance=provenance
        )

    def get_model_selections(self, limit: int = 20) -> List[MemoryEntry]:
        """Get all model selection memories."""
        return self.get_by_type("model_selection", limit)

    def migrate_to_dual_memory(self) -> int:
        """Migrate all legacy entries to DualMemoryStore.

        Returns the number of entries successfully migrated.
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
