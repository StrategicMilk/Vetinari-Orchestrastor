"""Multi-turn conversation memory with token-aware context windowing and SQLite persistence.

Conversation history is stored both in-memory (for fast access) and in the unified
SQLite database (for durability across process restarts).

Database table (part of the unified schema in ``vetinari.database``):

.. code-block:: sql

    CREATE TABLE IF NOT EXISTS conversation_messages (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   TEXT NOT NULL,
        role         TEXT NOT NULL,
        content      TEXT NOT NULL,
        timestamp    REAL NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_messages(session_id);
    CREATE INDEX IF NOT EXISTS idx_conv_session_ts ON conversation_messages(session_id, timestamp);

On startup, ``ConversationStore.__init__`` loads the most recent ``_MAX_SESSIONS``
sessions from SQLite into the in-memory LRU cache.  When the in-memory cache evicts
a session (LRU overflow), the messages remain in SQLite and are re-loaded on demand
by ``get_history()`` and ``get_context_window()``.

This is step 3 support in the pipeline: requests carry multi-turn context that must
survive process restarts so users can resume conversations after server reboots.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from vetinari.exceptions import ExecutionError

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN: int = 4  # rough estimate used for token budgeting


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ConversationMessage:
    """A single message in a conversation session.

    Attributes:
        role: Speaker role — typically ``"user"``, ``"assistant"``, or
            ``"system"``.
        content: Text content of the message.
        timestamp: UNIX timestamp (seconds) when the message was added.
        metadata: Arbitrary key-value metadata.
        token_count: Estimated token count for the message content.
    """

    role: str
    content: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = field(default=0)
    is_compressed: bool = field(default=False)

    def __repr__(self) -> str:
        return f"ConversationMessage(role={self.role!r}, content={self.content[:40]!r})"


# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------


class ConversationStore:
    """Thread-safe conversation session store backed by SQLite for durability.

    The store keeps up to ``_MAX_SESSIONS`` sessions in memory as an LRU cache.
    Every message is also written to the ``conversation_messages`` table in the
    unified SQLite database so history survives process restarts.

    On construction, existing sessions are restored from SQLite (up to
    ``_MAX_SESSIONS`` most recent by session_id sort order).  When a session is
    evicted from the in-memory LRU, its messages remain in SQLite and are
    re-loaded transparently when ``get_history()`` or ``get_context_window()``
    is called for that session.

    Obtain the singleton instance via :func:`get_conversation_store`.

    All public methods are safe to call from multiple threads simultaneously.
    """

    _MAX_SESSIONS = 200  # prevent unbounded session accumulation

    def __init__(self) -> None:
        # Module-level mutable state:
        #   _sessions: LRU cache keyed by session_id; written by add_message/
        #              create_session, read by get_history/get_context_window.
        #   _lock: protects all reads/writes to _sessions (not SQLite — SQLite
        #          handles its own locking via WAL mode).
        self._sessions: OrderedDict[str, list[ConversationMessage]] = OrderedDict()
        self._lock = threading.Lock()
        self._restore_from_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _restore_from_db(self) -> None:
        """Load existing sessions from SQLite into the in-memory LRU cache.

        Queries the ``conversation_messages`` table for distinct session IDs,
        loads the most recent ``_MAX_SESSIONS`` sessions (sorted by session_id),
        and populates ``_sessions``.  Called once from ``__init__``.
        """
        try:
            # Lazy import to avoid circular dependency at module load time.
            from vetinari.database import get_connection

            conn = get_connection()
            cursor = conn.execute("SELECT DISTINCT session_id FROM conversation_messages ORDER BY session_id")
            session_ids = [row[0] for row in cursor.fetchall()]

            # Load only the most recent _MAX_SESSIONS to respect the LRU limit.
            sessions_to_load = session_ids[-self._MAX_SESSIONS :]

            loaded = 0
            for sid in sessions_to_load:
                messages = self._load_session_from_db(sid)
                with self._lock:
                    self._sessions[sid] = messages
                loaded += 1

            if loaded:
                logger.info("Restored %d conversation sessions from SQLite", loaded)
        except Exception as exc:
            logger.warning(
                "Could not restore conversation sessions from SQLite — starting with empty in-memory store: %s",
                exc,
            )

    def _load_session_from_db(self, session_id: str) -> list[ConversationMessage]:
        """Fetch all messages for *session_id* from SQLite, ordered by timestamp.

        Args:
            session_id: The session whose messages to load.

        Returns:
            List of :class:`ConversationMessage` ordered oldest-first.
            Returns an empty list if the session has no persisted messages or
            if the database is unavailable.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            cursor = conn.execute(
                "SELECT role, content, timestamp, metadata_json "
                "FROM conversation_messages "
                "WHERE session_id = ? "
                "ORDER BY timestamp ASC, id ASC",
                (session_id,),
            )
            messages: list[ConversationMessage] = []
            for row in cursor.fetchall():
                try:
                    metadata = json.loads(row[3] or "{}")
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                messages.append(
                    ConversationMessage(
                        role=row[0],
                        content=row[1],
                        timestamp=row[2],
                        metadata=metadata,
                    )
                )
            return messages
        except Exception as exc:
            logger.warning(
                "Could not load session %s from SQLite — returning empty history: %s",
                session_id,
                exc,
            )
            return []

    def _persist_message(
        self,
        session_id: str,
        msg: ConversationMessage,
    ) -> None:
        """Write *msg* to the SQLite ``conversation_messages`` table.

        Best-effort: logs a warning on failure but never raises to the caller.

        Args:
            session_id: Session the message belongs to.
            msg: The message to persist.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            conn.execute(
                "INSERT INTO conversation_messages "
                "(session_id, role, content, timestamp, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    msg.role,
                    msg.content,
                    msg.timestamp,
                    json.dumps(msg.metadata),
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.warning(
                "Could not persist message for session %s to SQLite — message stored in-memory only: %s",
                session_id,
                exc,
            )

    def _delete_session_from_db(self, session_id: str) -> None:
        """Delete all persisted messages for *session_id* from SQLite.

        Best-effort: logs a warning on failure but never raises.

        Args:
            session_id: The session whose messages to delete.
        """
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            conn.execute(
                "DELETE FROM conversation_messages WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
        except Exception as exc:
            logger.warning(
                "Could not delete session %s from SQLite — messages may reappear on next restart: %s",
                session_id,
                exc,
            )

    def _ensure_in_memory(self, session_id: str) -> bool:
        """Ensure *session_id* is in the in-memory cache, loading from SQLite if needed.

        Must be called WITHOUT holding ``_lock``.  Acquires the lock internally
        when writing into ``_sessions``.

        Args:
            session_id: The session ID to load.

        Returns:
            ``True`` if the session is now in ``_sessions`` (either it was
            already there or was successfully loaded from SQLite).
            ``False`` if the session does not exist in SQLite either.
        """
        with self._lock:
            if session_id in self._sessions:
                return True

        # Not in memory — try loading from SQLite.
        messages = self._load_session_from_db(session_id)
        # An empty list could mean the session truly doesn't exist, or it has
        # no messages yet.  Check whether the session_id appears in SQLite at
        # all (any row for it) to distinguish the two cases.
        try:
            from vetinari.database import get_connection

            conn = get_connection()
            row = conn.execute(
                "SELECT 1 FROM conversation_messages WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            exists_in_db = row is not None
        except Exception:
            exists_in_db = bool(messages)

        if not exists_in_db and not messages:
            return False

        with self._lock:
            # Another thread may have loaded it while we were querying.
            if session_id not in self._sessions:
                # Evict oldest if at capacity.
                while len(self._sessions) >= self._MAX_SESSIONS:
                    evicted_id, _ = self._sessions.popitem(last=False)
                    logger.debug("Evicted oldest conversation session %s", evicted_id)
                self._sessions[session_id] = messages
        return True

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, session_id: str | None = None) -> str:
        """Create a new conversation session.

        Args:
            session_id: Optional explicit ID.  A UUID4 is generated when
                omitted.

        Returns:
            The session ID string.

        Raises:
            ExecutionError: If *session_id* already exists.
        """
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if sid in self._sessions:
                raise ExecutionError(f"Session '{sid}' already exists")
            # Evict oldest sessions when at capacity.
            while len(self._sessions) >= self._MAX_SESSIONS:
                evicted_id, _ = self._sessions.popitem(last=False)
                logger.debug("Evicted oldest conversation session %s", evicted_id)
            self._sessions[sid] = []
        logger.debug("Created conversation session %s", sid)
        return sid

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a message to an existing session and persist it to SQLite.

        The message is appended to the in-memory list inside ``_lock``, then
        written to SQLite outside the lock (best-effort — a DB failure logs a
        warning but never blocks the caller).

        Args:
            session_id: Target session ID.
            role: Speaker role (e.g. ``"user"`` or ``"assistant"``).
            content: Message text.
            metadata: Optional metadata dict.

        Raises:
            KeyError: If *session_id* does not exist.
        """
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},  # noqa: VET112 - empty fallback preserves optional request metadata contract
        )
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' does not exist")
            self._sessions[session_id].append(msg)

        # Write to SQLite outside the lock — WAL mode handles concurrent writers.
        self._persist_message(session_id, msg)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_history(self, session_id: str, limit: int = 50) -> list[ConversationMessage]:
        """Return up to *limit* most recent messages for a session.

        If the session was evicted from the in-memory LRU, it is transparently
        re-loaded from SQLite before returning.

        Args:
            session_id: Target session ID.
            limit: Maximum number of messages to return.  0 means all.

        Returns:
            List of :class:`ConversationMessage` objects, oldest first.

        Raises:
            KeyError: If *session_id* does not exist in memory or SQLite.
        """
        if not self._ensure_in_memory(session_id):
            raise KeyError(f"Session '{session_id}' does not exist")

        with self._lock:
            messages = list(self._sessions[session_id])

        if limit and len(messages) > limit:
            return messages[-limit:]
        return messages

    def get_context_window(self, session_id: str, max_tokens: int = 4096) -> list[ConversationMessage]:
        """Return the most recent messages that fit within a token budget.

        Token count is estimated at :data:`_CHARS_PER_TOKEN` characters per
        token.  Messages are included from newest to oldest until the budget
        is exhausted, then returned in chronological order.

        If the session was evicted from the in-memory LRU, it is transparently
        re-loaded from SQLite before returning.

        Args:
            session_id: Target session ID.
            max_tokens: Maximum number of tokens to include.

        Returns:
            List of :class:`ConversationMessage` objects fitting within the
            token budget, oldest first.

        Raises:
            KeyError: If *session_id* does not exist in memory or SQLite.
        """
        if not self._ensure_in_memory(session_id):
            raise KeyError(f"Session '{session_id}' does not exist")

        with self._lock:
            messages = list(self._sessions[session_id])

        selected: list[ConversationMessage] = []
        budget = max_tokens * _CHARS_PER_TOKEN  # convert to chars

        for msg in reversed(messages):
            cost = len(msg.content)
            if cost > budget:
                break
            selected.append(msg)
            budget -= cost

        selected.reverse()
        return selected

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear_session(self, session_id: str) -> None:
        """Remove all messages from a session (in memory and in SQLite).

        Args:
            session_id: Target session ID.

        Raises:
            KeyError: If *session_id* does not exist.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' does not exist")
            self._sessions[session_id] = []

        # Delete persisted rows outside the lock — best-effort.
        self._delete_session_from_db(session_id)

    def list_sessions(self) -> list[str]:
        """Return the IDs of all existing in-memory sessions.

        Returns:
            Sorted list of session ID strings currently held in the LRU cache.
        """
        with self._lock:
            return sorted(self._sessions.keys())


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_store: ConversationStore | None = None
_store_lock = threading.Lock()


def get_conversation_store() -> ConversationStore:
    """Return the process-wide singleton :class:`ConversationStore`.

    The store is created on first call and restores persisted sessions from
    SQLite during construction.  Thread-safe via double-checked locking.

    Returns:
        The singleton :class:`ConversationStore` instance.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ConversationStore()
    return _store


def _reset_conversation_store() -> None:
    """Reset the singleton and clear the SQLite backing table (for testing only).

    Drops all rows from ``conversation_messages`` so tests start from a clean
    state regardless of prior test runs.
    """
    global _store
    with _store_lock:
        _store = None

    # Clear the SQLite table so persistence tests are isolated.
    try:
        from vetinari.database import get_connection

        conn = get_connection()
        conn.execute("DELETE FROM conversation_messages")
        conn.commit()
    except Exception as exc:
        logger.warning("Could not clear conversation_messages table during store reset: %s", exc)


# ---------------------------------------------------------------------------
# ContextReconstructor
# ---------------------------------------------------------------------------


class ContextReconstructor:
    """Build a formatted prompt context string from conversation history.

    The reconstructor fits the most recent messages into a token budget,
    prepends a system header, and summarises older messages when present.
    """

    _SYSTEM_HEADER = "You are a helpful AI assistant.\n\n"
    _SUMMARY_HEADER = "[Earlier conversation summarised]\n\n"

    def reconstruct(
        self,
        messages: list[ConversationMessage],
        max_tokens: int = 4096,
    ) -> str:
        """Build a context string from *messages* within *max_tokens*.

        The method works in three steps:

        1. Reserve space for the system header.
        2. Walk messages from newest to oldest, collecting those that fit.
        3. Prepend a summary placeholder for any omitted older messages.

        Args:
            messages: Conversation history (chronological order).
            max_tokens: Token budget for the assembled context.

        Returns:
            Formatted context string ready to prepend to a prompt.
        """
        if not messages:
            return self._SYSTEM_HEADER

        char_budget = max_tokens * _CHARS_PER_TOKEN
        header_cost = len(self._SYSTEM_HEADER)
        char_budget -= header_cost

        selected: list[ConversationMessage] = []
        for msg in reversed(messages):
            formatted = self._format_message(msg)
            if len(formatted) > char_budget:
                break
            selected.append(msg)
            char_budget -= len(formatted)

        selected.reverse()
        omitted_count = len(messages) - len(selected)

        parts: list[str] = [self._SYSTEM_HEADER]
        if omitted_count > 0:
            parts.append(f"[{omitted_count} earlier message(s) not shown]\n\n")
        parts.extend(self._format_message(msg) for msg in selected)

        return "".join(parts)

    @staticmethod
    def _format_message(msg: ConversationMessage) -> str:
        """Render a single message as a labelled block.

        Args:
            msg: Message to format.

        Returns:
            Formatted string ending with a newline.
        """
        return f"{msg.role.upper()}: {msg.content}\n"
