"""Vetinari Blackboard — Inter-Agent Communication & Delegation.

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

    # Agent B (ConsolidatedResearcher) claims and processes it
    entry = board.claim(entry_id, AgentType.CONSOLIDATED_RESEARCHER)
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
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EntryState(Enum):
    """Entry state."""
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class BlackboardEntry:
    """A single work item or message on the blackboard."""

    entry_id: str
    content: str  # Description of the work needed
    request_type: str  # e.g. "code_search", "code_review"
    requested_by: str  # AgentType.value of the requester
    priority: int = 5  # 1=highest, 10=lowest
    state: EntryState = EntryState.PENDING
    claimed_by: str | None = None  # AgentType.value
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    claimed_at: float | None = None
    completed_at: float | None = None
    ttl_seconds: float = 3600.0  # Expire after 1 hour if unclaimed
    metadata: dict[str, Any] = field(default_factory=dict)
    _completion_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
        compare=False,
    )

    @property
    def is_expired(self) -> bool:
        if self.state in (EntryState.COMPLETED, EntryState.FAILED):
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
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

    _instance: Blackboard | None = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._entries: dict[str, BlackboardEntry] = {}
        self._lock = threading.RLock()
        self._observers: list[Callable[[BlackboardEntry], None]] = []

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> Blackboard:
        """Get instance.

        Returns:
            The Blackboard result.
        """
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
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Post a new work item. Returns the entry_id.

        Args:
            content: The content.
            request_type: The request type.
            requested_by: The requested by.
            priority: The priority.
            ttl_seconds: The ttl seconds.
            metadata: The metadata.

        Returns:
            The result string.
        """
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
        logger.debug("[Blackboard] Posted %s (%s) by %s", entry_id, request_type, requested_by)
        self._notify_observers(entry)
        return entry_id

    # ------------------------------------------------------------------
    # Claiming and completing
    # ------------------------------------------------------------------

    def claim(self, entry_id: str, agent_type: str) -> BlackboardEntry | None:
        """Claim a pending entry for processing. Returns None if unavailable.

        Phase 7.9H: Checks MODEL_INFERENCE permission before allowing claim.

        Args:
            entry_id: The entry id.
            agent_type: The agent type.

        Returns:
            The BlackboardEntry | None result.
        """
        # Permission check — agents can only claim work if permitted
        try:
            from vetinari.execution_context import ToolPermission, get_context_manager

            ctx_mgr = get_context_manager()
            if not ctx_mgr.check_permission(ToolPermission.MODEL_INFERENCE):
                logger.warning(
                    "[Blackboard] Claim denied for %s on %s — MODEL_INFERENCE not allowed in current mode",
                    agent_type,
                    entry_id,
                )
                return None
        except Exception:
            # Context manager not configured — allow claim
            logger.debug("Execution context manager not configured, allowing claim for %s", entry_id, exc_info=True)

        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None or entry.state != EntryState.PENDING:
                return None
            entry.state = EntryState.CLAIMED
            entry.claimed_by = agent_type
            entry.claimed_at = time.time()
        return entry

    def complete(self, entry_id: str, result: Any) -> bool:
        """Mark an entry as completed with a result.

        Args:
            entry_id: The entry id.
            result: The result.

        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return False
            entry.state = EntryState.COMPLETED
            entry.result = result
            entry.completed_at = time.time()
            entry._completion_event.set()  # Wake up waiters immediately
        logger.debug("[Blackboard] Completed %s", entry_id)
        return True

    def fail(self, entry_id: str, error: str) -> bool:
        """Mark an entry as failed.

        Args:
            entry_id: The entry id.
            error: The error.

        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return False
            entry.state = EntryState.FAILED
            entry.error = error
            entry.completed_at = time.time()
            entry._completion_event.set()  # Wake up waiters on failure too
        logger.debug("[Blackboard] Failed %s: %s", entry_id, error)
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_result(self, entry_id: str, timeout: float = 30.0) -> Any:
        """Wait for a result using threading.Event (no polling).

        Args:
            entry_id: The entry id.
            timeout: The timeout.

        Returns:
            The Any result.

        Raises:
            RuntimeError: If the operation fails.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
        if entry is None:
            return None
        # Fast path: already done
        if entry.state == EntryState.COMPLETED:
            return entry.result
        if entry.state == EntryState.FAILED:
            raise RuntimeError(f"Blackboard entry {entry_id} failed: {entry.error}")
        # Block until signalled or timeout
        entry._completion_event.wait(timeout=timeout)
        if entry.state == EntryState.COMPLETED:
            return entry.result
        if entry.state == EntryState.FAILED:
            raise RuntimeError(f"Blackboard entry {entry_id} failed: {entry.error}")
        return None  # Timeout

    def get_pending(
        self,
        request_type: str | None = None,
        limit: int = 10,
    ) -> list[BlackboardEntry]:
        """Return pending entries, optionally filtered by type, sorted by priority.

        Args:
            request_type: The request type.
            limit: The limit.

        Returns:
            List of results.
        """
        with self._lock:
            entries = [
                e
                for e in self._entries.values()
                if e.state == EntryState.PENDING
                and not e.is_expired
                and (request_type is None or e.request_type == request_type)
            ]
        entries.sort(key=lambda e: (e.priority, e.created_at))
        return entries[:limit]

    def get_entry(self, entry_id: str) -> BlackboardEntry | None:
        """Get entry.

        Returns:
            The BlackboardEntry | None result.
        """
        with self._lock:
            return self._entries.get(entry_id)

    # ------------------------------------------------------------------
    # Delegation helper (used by AgentGraph)
    # ------------------------------------------------------------------

    def delegate(
        self,
        task: Any,
        available_agents: dict[Any, Any],
    ) -> Any | None:
        """Try to find an agent that can handle ``task.assigned_agent`` type.

        Falls back to PLANNER for unknown types, then returns a failure result
        if no fallback exists.

        Args:
            task: The task.
            available_agents: The available agents.

        Returns:
            The Any | None result.
        """
        from vetinari.agents.contracts import AgentTask
        from vetinari.types import AgentType

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
                    "[Blackboard] Delegating unhandled task %s (type=%s) to fallback %s",
                    task.id,
                    task.assigned_agent,
                    fallback_type.value,
                )
                try:
                    agent_task = AgentTask.from_task(task, task.description)
                    return agent.execute(agent_task)
                except Exception as e:
                    logger.error("[Blackboard] Fallback delegation failed: %s", e)

        return None

    # ------------------------------------------------------------------
    # Phase 7.9B: Inter-agent delegation patterns
    # ------------------------------------------------------------------

    def request_help(
        self,
        requesting_agent: str,
        request_type: str,
        description: str,
        priority: int = 5,
        timeout: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Post a help request and wait for a capable agent to fulfil it.

        This is a synchronous convenience method: it posts, then blocks
        until a result is available or timeout expires.

        Args:
            requesting_agent: AgentType.value of the requester.
            request_type: Category of work (must match REQUEST_TYPE_ROUTING).
            description: Human-readable task description.
            priority: 1=highest, 10=lowest.
            timeout: Seconds to wait for result.
            metadata: Optional extra context for the handler.

        Returns:
            The result from the handling agent, or None on timeout.
        """
        entry_id = self.post(
            content=description,
            request_type=request_type,
            requested_by=requesting_agent,
            priority=priority,
            metadata=metadata or {},
        )
        logger.info("[Blackboard] %s requests help: %s (%s)", requesting_agent, request_type, entry_id)
        try:
            return self.get_result(entry_id, timeout=timeout)
        except RuntimeError:
            return None

    def escalate_error(
        self,
        agent_type: str,
        task_id: str,
        error: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Escalate an error to the blackboard for error recovery.

        Posts a high-priority error_recovery request that the
        ErrorRecoveryAgent (or OPERATIONS) can pick up.

        Returns:
            The entry_id of the escalation.
        """
        return self.post(
            content=f"Error in {agent_type} task {task_id}: {error}",
            request_type="error_recovery",
            requested_by=agent_type,
            priority=1,  # High priority
            metadata={
                "original_task_id": task_id,
                "error": error,
                **(context or {}),
            },
        )

    def request_consensus(
        self,
        requesting_agent: str,
        subject: str,
        options: list,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Post a consensus-check request for multi-agent voting.

        Multiple agents can claim the entry and vote.  The caller
        should collect votes from the result.

        Returns:
            The entry_id of the consensus request.
        """
        return self.post(
            content=f"Consensus needed: {subject}",
            request_type="architecture_decision",
            requested_by=requesting_agent,
            priority=3,
            metadata={
                "consensus_request": True,
                "subject": subject,
                "options": options,
                **(metadata or {}),
            },
        )

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
                logger.debug("[Blackboard] Observer error: %s", e)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_expired(self) -> int:
        """Remove expired entries. Returns count of purged entries.

        Returns:
            The computed value.
        """
        with self._lock:
            expired = [eid for eid, e in self._entries.items() if e.is_expired]
            for eid in expired:
                self._entries[eid].state = EntryState.EXPIRED
            # Keep for audit; remove entries older than 2 hours
            cutoff = time.time() - 7200
            stale = [
                eid
                for eid, e in self._entries.items()
                if e.created_at < cutoff and e.state in (EntryState.COMPLETED, EntryState.FAILED, EntryState.EXPIRED)
            ]
            for eid in stale:
                del self._entries[eid]
        return len(expired)

    def get_stats(self) -> dict[str, int]:
        """Return a summary of entry states.

        Returns:
            The result string.
        """
        with self._lock:
            states: dict[str, int] = {}
            for e in self._entries.values():
                states[e.state.value] = states.get(e.state.value, 0) + 1
        return states

    def clear(self) -> None:
        """Clear all entries (use in tests only)."""
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Request-type routing (Phase 7.9G)
# ---------------------------------------------------------------------------

# Maps request_type strings to the agent types capable of handling them.
# Used by the blackboard to auto-notify the right agents when work is posted.
REQUEST_TYPE_ROUTING: dict[str, list[str]] = {
    "code_search": ["CONSOLIDATED_RESEARCHER"],
    "code_review": ["QUALITY"],
    "security_audit": ["QUALITY"],
    "architecture_decision": ["CONSOLIDATED_ORACLE", "ORACLE"],
    "documentation": ["OPERATIONS"],
    "implementation": ["BUILDER"],
    "test_generation": ["QUALITY"],
    "cost_analysis": ["OPERATIONS"],
    "research": ["CONSOLIDATED_RESEARCHER"],
    "ui_design": ["CONSOLIDATED_RESEARCHER", "ARCHITECT"],
    "devops": ["ARCHITECT"],
    "error_recovery": ["OPERATIONS"],
    "image_generation": ["OPERATIONS"],
    "data_engineering": ["CONSOLIDATED_RESEARCHER", "ARCHITECT"],
    "creative_writing": ["OPERATIONS"],
}


def get_capable_agents(request_type: str) -> list[str]:
    """Return agent type strings capable of handling a given request type."""
    return REQUEST_TYPE_ROUTING.get(request_type, [])


# ---------------------------------------------------------------------------
# Shared Execution Context (Phase 7.9E)
# ---------------------------------------------------------------------------


class SharedExecutionContext:
    """Key-value store accessible to all agents during a single plan execution.

    Lifetime: created at plan start, cleaned up at plan completion.

    Use case: RESEARCHER stores ``codebase_map`` mid-execution and BUILDER
    reads it without requiring an explicit DAG edge between them.

    Thread-safe via RLock.
    """

    def __init__(self, plan_id: str):
        self.plan_id = plan_id
        self._store: dict[str, Any] = {}
        self._provenance: dict[str, str] = {}  # key -> agent_type that wrote it
        self._lock = threading.RLock()

    def set(self, key: str, value: Any, agent_type: str) -> None:
        """Store a value, recording which agent wrote it.

        Args:
            key: The key.
            value: The value.
            agent_type: The agent type.
        """
        with self._lock:
            self._store[key] = value
            self._provenance[key] = agent_type
        logger.debug("[SharedCtx:%s] %s set '%s'", self.plan_id, agent_type, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a value (returns *default* if missing).

        Args:
            key: The key.
            default: The default.

        Returns:
            The Any result.
        """
        with self._lock:
            return self._store.get(key, default)

    def get_all(self) -> dict[str, Any]:
        """Return a shallow copy of all stored key-value pairs.

        Returns:
            The result string.
        """
        with self._lock:
            return dict(self._store)

    def get_all_by_agent(self, agent_type: str) -> dict[str, Any]:
        """Return all entries written by a specific agent type.

        Returns:
            The result string.
        """
        with self._lock:
            return {k: self._store[k] for k, a in self._provenance.items() if a == agent_type}

    def keys(self) -> list[str]:
        """Return list of stored keys.

        Returns:
            The result string.
        """
        with self._lock:
            return list(self._store.keys())

    def clear(self) -> None:
        """Remove all entries (called at plan completion)."""
        with self._lock:
            self._store.clear()
            self._provenance.clear()


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_blackboard: Blackboard | None = None
_board_lock = threading.Lock()


def get_blackboard() -> Blackboard:
    """Return the global Blackboard singleton (created lazily).

    Returns:
        The Blackboard result.
    """
    global _blackboard
    if _blackboard is None:
        with _board_lock:
            if _blackboard is None:
                _blackboard = Blackboard.get_instance()
    return _blackboard
