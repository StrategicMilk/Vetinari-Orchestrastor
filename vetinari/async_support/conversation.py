"""Multi-turn conversation memory with token-aware context windowing."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

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
    """

    role: str
    content: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------


class ConversationStore:
    """Thread-safe in-memory store for multi-turn conversation sessions.

    Obtain the singleton instance via :func:`get_conversation_store`.

    All public methods are safe to call from multiple threads simultaneously.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[ConversationMessage]] = {}
        self._lock = threading.Lock()

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
            ValueError: If *session_id* already exists.
        """
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if sid in self._sessions:
                raise ValueError(f"Session '{sid}' already exists")
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
        """Append a message to an existing session.

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
            metadata=metadata or {},
        )
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' does not exist")
            self._sessions[session_id].append(msg)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_history(self, session_id: str, limit: int = 50) -> list[ConversationMessage]:
        """Return up to *limit* most recent messages for a session.

        Args:
            session_id: Target session ID.
            limit: Maximum number of messages to return.  0 means all.

        Returns:
            List of :class:`ConversationMessage` objects, oldest first.

        Raises:
            KeyError: If *session_id* does not exist.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' does not exist")
            messages = list(self._sessions[session_id])

        if limit and len(messages) > limit:
            return messages[-limit:]
        return messages

    def get_context_window(self, session_id: str, max_tokens: int = 4096) -> list[ConversationMessage]:
        """Return the most recent messages that fit within a token budget.

        Token count is estimated at :data:`_CHARS_PER_TOKEN` characters per
        token.  Messages are included from newest to oldest until the budget
        is exhausted, then returned in chronological order.

        Args:
            session_id: Target session ID.
            max_tokens: Maximum number of tokens to include.

        Returns:
            List of :class:`ConversationMessage` objects fitting within the
            token budget, oldest first.

        Raises:
            KeyError: If *session_id* does not exist.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' does not exist")
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
        """Remove all messages from a session.

        Args:
            session_id: Target session ID.

        Raises:
            KeyError: If *session_id* does not exist.
        """
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' does not exist")
            self._sessions[session_id] = []

    def list_sessions(self) -> list[str]:
        """Return the IDs of all existing sessions.

        Returns:
            Sorted list of session ID strings.
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

    The store is created on first call.  Thread-safe.

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
    """Reset the singleton (for testing only)."""
    global _store
    with _store_lock:
        _store = None


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
        for msg in selected:
            parts.append(self._format_message(msg))

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
