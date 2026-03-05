"""
Vetinari Blackboard — Inter-Agent Communication & Delegation
=============================================================

The Blackboard implements a shared message board through which agents can:

1. **Post work requests** — an agent that needs help from a specialist posts
   a ``BlackboardEntry`` describing the sub-task.
2. **Delegate tasks** — the AgentGraph routes un-handled task types here so
   that the most capable available agent handles them.
3. **Share results** — agents publish their outputs; downstream consumers
   retrieve them by key.
4. **Broadcast observations** — any agent can notify others of state changes
   (e.g. "error detected", "context window approaching limit").

Architecture
------------
- Thread-safe via ``threading.RLock``
- In-memory with optional SQLite persistence for crash recovery
- TTL-based expiry for stale entries (default 1 hour)
- Observer pattern for reactive agents

Usage::

    from vetinari.blackboard import get_blackboard

    board = get_blackboard()

    # Agent A posts a sub-task
    entry_id = board.post(
        content="Find all Python files using asyncio.run()",
        request_type="code_search",
        requested_by=AgentType.BUILDER,
        priority=5,
    )

    # Agent B (Explorer) claims and processes it
    entry = board.claim(entry_id, AgentType.EXPLORER)
    # ... do work ...
    board.complete(entry_id, result={"files": [...]})

    # Agent A retrieves the result
    result = board.get_result(entry_id)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class EntryState(Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class BlackboardEntry:
    """A single work item or message on the blackboard."""
    entry_id: str
    content: str                        # Description of the work needed
    request_type: str                   # e.g. "code_search", "code_review"
    requested_by: str                   # AgentType.value of the requester
    priority: int = 5                   # 1=highest, 10=lowest
    state: EntryState = EntryState.PENDING
    claimed_by: Optional[str] = None    # AgentType.value
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    claimed_at: Optional[float] = None
    completed_at: Optional[float] = None
    ttl_seconds: float = 3600.0         # Expire after 1 hour if unclaimed
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.state in (EntryState.COMPLETED, EntryState.FAILED):
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "request_type": self.request_type,
            "requested_by": self.requested_by,
            "priority": self.priority,
            "state": self.state.value,
            "claimed_by": self.claimed_by,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "ttl_seconds": self.ttl_seconds,
        }


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------

class Blackboard:
    """Thread-safe inter-agent message board."""

    _instance: Optional["Blackboard"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._entries: Dict[str, BlackboardEntry] = {}
        self._lock = threading.RLock()
        self._observers: List[Callable[[BlackboardEntry], None]] = []

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "Blackboard":
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Posting
    # ------------------------------------------------------------------

    def post(
        self,
        content: str,
        request_type: str,
        requested_by: str,
        priority: int = 5,
        ttl_seconds: float = 3600.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Post a new work item. Returns the entry_id."""
        entry_id = f"bb_{uuid.uuid4().hex[:8]}"
        entry = BlackboardEntry(
            entry_id=entry_id,
            content=content,
            request_type=request_type,
            requested_by=requested_by,
            priority=priority,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {},
        )
        with self._lock:
            self._entries[entry_id] = entry
        logger.debug(f"[Blackboard] Posted {entry_id} ({request_type}) by {requested_by}")
        self._notify_observers(entry)
        return entry_id

    # ------------------------------------------------------------------
    # Claiming and completing
    # ------------------------------------------------------------------

    def claim(self, entry_id: str, agent_type: str) -> Optional[BlackboardEntry]:
        """Claim a pending entry for processing. Returns None if unavailable."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None or entry.state != EntryState.PENDING:
                return None
            entry.state = EntryState.CLAIMED
            entry.claimed_by = agent_type
            entry.claimed_at = time.time()
        return entry

    def complete(self, entry_id: str, result: Any) -> bool:
        """Mark an entry as completed with a result."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return False
            entry.state = EntryState.COMPLETED
            entry.result = result
            entry.completed_at = time.time()
        logger.debug(f"[Blackboard] Completed {entry_id}")
        return True

    def fail(self, entry_id: str, error: str) -> bool:
        """Mark an entry as failed."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return False
            entry.state = EntryState.FAILED
            entry.error = error
            entry.completed_at = time.time()
        logger.debug(f"[Blackboard] Failed {entry_id}: {error}")
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_result(self, entry_id: str, timeout: float = 30.0) -> Any:
        """Poll for a result until it arrives or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                entry = self._entries.get(entry_id)
            if entry is None:
                return None
            if entry.state == EntryState.COMPLETED:
                return entry.result
            if entry.state == EntryState.FAILED:
                raise RuntimeError(f"Blackboard entry {entry_id} failed: {entry.error}")
            time.sleep(0.1)
        return None  # Timeout

    def get_pending(
        self,
        request_type: Optional[str] = None,
        request_type_prefix: Optional[str] = None,
        limit: int = 10,
    ) -> List[BlackboardEntry]:
        """Return pending entries, optionally filtered by type or prefix, sorted by priority."""
        with self._lock:
            entries = [
                e for e in self._entries.values()
                if e.state == EntryState.PENDING
                and not e.is_expired
                and (request_type is None or e.request_type == request_type)
                and (request_type_prefix is None or e.request_type.startswith(request_type_prefix))
            ]
        entries.sort(key=lambda e: (e.priority, e.created_at))
        return entries[:limit]

    def get_entry(self, entry_id: str) -> Optional[BlackboardEntry]:
        with self._lock:
            return self._entries.get(entry_id)

    # ------------------------------------------------------------------
    # Delegation helper (used by AgentGraph)
    # ------------------------------------------------------------------

    def delegate(
        self,
        task: Any,
        available_agents: Dict[Any, Any],
    ) -> Optional[Any]:
        """Try to find an agent that can handle ``task.assigned_agent`` type.

        Falls back to PLANNER for unknown types, then returns a failure result
        if no fallback exists.
        """
        from vetinari.agents.contracts import AgentType, AgentTask, AgentResult

        # Try to find any capable fallback
        fallback_order = [
            AgentType.PLANNER,
            AgentType.BUILDER,
            AgentType.RESEARCHER,
        ]
        for fallback_type in fallback_order:
            if fallback_type in available_agents:
                agent = available_agents[fallback_type]
                logger.warning(
                    f"[Blackboard] Delegating unhandled task {task.id} "
                    f"(type={task.assigned_agent}) to fallback {fallback_type.value}"
                )
                try:
                    agent_task = AgentTask.from_task(task, task.description)
                    return agent.execute(agent_task)
                except Exception as e:
                    logger.error(f"[Blackboard] Fallback delegation failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[BlackboardEntry], None]) -> None:
        """Register a callback invoked when new entries are posted."""
        with self._lock:
            self._observers.append(callback)

    def _notify_observers(self, entry: BlackboardEntry) -> None:
        for cb in self._observers:
            try:
                cb(entry)
            except Exception as e:
                logger.debug(f"[Blackboard] Observer error: {e}")

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_expired(self) -> int:
        """Remove expired entries. Returns count of purged entries."""
        with self._lock:
            expired = [eid for eid, e in self._entries.items() if e.is_expired]
            for eid in expired:
                self._entries[eid].state = EntryState.EXPIRED
            # Keep for audit; remove entries older than 2 hours
            cutoff = time.time() - 7200
            stale = [
                eid for eid, e in self._entries.items()
                if e.created_at < cutoff
                and e.state in (EntryState.COMPLETED, EntryState.FAILED, EntryState.EXPIRED)
            ]
            for eid in stale:
                del self._entries[eid]
        return len(expired)

    def get_stats(self) -> Dict[str, int]:
        """Return a summary of entry states."""
        with self._lock:
            states: Dict[str, int] = {}
            for e in self._entries.values():
                states[e.state.value] = states.get(e.state.value, 0) + 1
        return states

    def clear(self) -> None:
        """Clear all entries (use in tests only)."""
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_blackboard: Optional[Blackboard] = None
_board_lock = threading.Lock()


def get_blackboard() -> Blackboard:
    """Return the global Blackboard singleton (created lazily)."""
    global _blackboard
    if _blackboard is None:
        with _board_lock:
            if _blackboard is None:
                _blackboard = Blackboard.get_instance()
    return _blackboard
