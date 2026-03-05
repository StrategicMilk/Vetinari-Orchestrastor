"""
Auto-Compaction Engine — WS12
===============================
Manages conversation-context compaction for agents running against local LLMs
with limited context windows.

Behaviour
---------
- Tracks token count for each conversation (plan_id or session_id).
- When usage reaches VETINARI_COMPACTION_THRESHOLD (default 80 %) of the
  declared context window, compaction is triggered.
- Compaction strategy:
    1. Keep the last N messages verbatim (default 5).
    2. Summarise older messages into a structured summary block.
    3. Never compact mid-task — waits until the current task completes.
    4. Full transcript is appended to disk before compaction.
- Manual trigger: ``/compact`` slash command in the chat UI calls
  ``CompactionEngine.compact(session_id)``.
- Context meter: ``usage_pct(session_id)`` returns 0.0-1.0 for UI display.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-var driven)
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLD = float(os.environ.get("VETINARI_COMPACTION_THRESHOLD", "0.80"))
_DEFAULT_KEEP_LAST = int(os.environ.get("VETINARI_COMPACTION_KEEP_LAST", "5"))
_DEFAULT_CONTEXT_WINDOW = int(os.environ.get("VETINARI_CONTEXT_WINDOW", "8192"))

# Rough token estimate: 4 chars ≈ 1 token
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single conversation message."""
    role: str          # system | user | assistant
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    task_id: Optional[str] = None

    def token_estimate(self) -> int:
        return max(1, len(self.content) // _CHARS_PER_TOKEN)


@dataclass
class CompactionRecord:
    """Record of a compaction event."""
    session_id: str
    compacted_at: str
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    summary: str


@dataclass
class SessionContext:
    """All context for one session/plan."""
    session_id: str
    context_window: int = _DEFAULT_CONTEXT_WINDOW
    messages: List[Message] = field(default_factory=list)
    compaction_history: List[CompactionRecord] = field(default_factory=list)
    active_task_id: Optional[str] = None  # set while a task is running

    def token_count(self) -> int:
        return sum(m.token_estimate() for m in self.messages)

    def usage_pct(self) -> float:
        return self.token_count() / max(1, self.context_window)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CompactionEngine:
    """Manages per-session context compaction."""

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        keep_last: int = _DEFAULT_KEEP_LAST,
        transcript_dir: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self.keep_last = keep_last
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = Lock()

        # Directory for full transcripts before compaction
        try:
            from vetinari.config import get_data_dir
            self._transcript_dir = Path(transcript_dir or str(get_data_dir() / "transcripts"))
        except Exception:
            self._transcript_dir = Path(transcript_dir or ".vetinari/transcripts")
        self._transcript_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def get_or_create_session(
        self, session_id: str, context_window: int = _DEFAULT_CONTEXT_WINDOW
    ) -> SessionContext:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionContext(
                    session_id=session_id,
                    context_window=context_window,
                )
            return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        task_id: Optional[str] = None,
    ) -> None:
        """Add a message and auto-compact if threshold is exceeded."""
        session = self.get_or_create_session(session_id)
        with self._lock:
            session.messages.append(
                Message(role=role, content=content, task_id=task_id)
            )
            if task_id:
                session.active_task_id = task_id

        if session.usage_pct() >= self.threshold and not session.active_task_id:
            logger.info(
                f"[Compaction] Session {session_id} at "
                f"{session.usage_pct():.0%} — auto-compacting"
            )
            self.compact(session_id)

    def mark_task_complete(self, session_id: str) -> None:
        """Signal that the current task is done (allows auto-compact to fire)."""
        session = self.get_or_create_session(session_id)
        with self._lock:
            session.active_task_id = None
        # Check if we should compact now
        if session.usage_pct() >= self.threshold:
            self.compact(session_id)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(self, session_id: str) -> Optional[CompactionRecord]:
        """
        Compact the session context.

        1. Save full transcript to disk.
        2. Keep the last ``keep_last`` messages verbatim.
        3. Summarise the rest into a single system message.
        """
        session = self.get_or_create_session(session_id)
        with self._lock:
            messages = list(session.messages)

        if len(messages) <= self.keep_last:
            logger.debug(
                f"[Compaction] Session {session_id}: not enough messages to compact"
            )
            return None

        tokens_before = sum(m.token_estimate() for m in messages)

        # Save transcript
        self._save_transcript(session_id, messages)

        # Summarise older messages
        older = messages[: -self.keep_last]
        recent = messages[-self.keep_last :]
        summary_text = self._summarise(older)

        summary_msg = Message(
            role="system",
            content=(
                f"[COMPACTED CONTEXT — {len(older)} messages summarised]\n{summary_text}"
            ),
        )

        new_messages = [summary_msg] + recent
        tokens_after = sum(m.token_estimate() for m in new_messages)

        with self._lock:
            session.messages = new_messages

        record = CompactionRecord(
            session_id=session_id,
            compacted_at=datetime.now().isoformat(),
            messages_before=len(messages),
            messages_after=len(new_messages),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            summary=summary_text[:200],
        )
        with self._lock:
            session.compaction_history.append(record)

        logger.info(
            f"[Compaction] Session {session_id}: "
            f"{tokens_before} → {tokens_after} tokens "
            f"({len(messages)} → {len(new_messages)} messages)"
        )
        return record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarise(self, messages: List[Message]) -> str:
        """
        Produce a structured summary of a message list.
        Tries LLM summarisation; falls back to plain concatenation.
        """
        combined = "\n".join(
            f"[{m.role}]: {m.content[:300]}" for m in messages
        )
        try:
            from vetinari.adapter_manager import get_adapter_manager
            am = get_adapter_manager()
            adapter = am.get_current_adapter() if hasattr(am, "get_current_adapter") else None
            if adapter and hasattr(adapter, "chat"):
                resp = adapter.chat(
                    system=(
                        "Summarise the following conversation segment into a structured "
                        "summary. Include: completed tasks, key decisions, active context, "
                        "and any unresolved issues. Be concise."
                    ),
                    messages=[{"role": "user", "content": combined[:3000]}],
                    max_tokens=512,
                )
                if isinstance(resp, dict):
                    return resp.get("content") or resp.get("text") or combined[:500]
                return str(resp)[:500]
        except Exception as exc:
            logger.debug(f"[Compaction] LLM summarisation failed: {exc}")

        # Plain fallback
        return (
            f"Summary of {len(messages)} messages:\n"
            + combined[:1000]
        )

    def _save_transcript(self, session_id: str, messages: List[Message]) -> None:
        ts = int(time.time())
        filename = self._transcript_dir / f"{session_id}_{ts}.jsonl"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                for m in messages:
                    f.write(
                        json.dumps({
                            "role": m.role,
                            "content": m.content[:2000],
                            "timestamp": m.timestamp,
                            "task_id": m.task_id,
                        })
                        + "\n"
                    )
        except Exception as exc:
            logger.warning(f"[Compaction] Transcript save failed: {exc}")

    # ------------------------------------------------------------------
    # Context meter
    # ------------------------------------------------------------------

    def usage_pct(self, session_id: str) -> float:
        """Return 0.0-1.0 usage fraction for the UI context meter."""
        if session_id not in self._sessions:
            return 0.0
        return self._sessions[session_id].usage_pct()

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Return messages as plain dicts for use with LLM adapters."""
        session = self.get_or_create_session(session_id)
        return [{"role": m.role, "content": m.content} for m in session.messages]

    def session_stats(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self._sessions:
            return {}
        s = self._sessions[session_id]
        return {
            "session_id": session_id,
            "messages": len(s.messages),
            "tokens": s.token_count(),
            "context_window": s.context_window,
            "usage_pct": round(s.usage_pct(), 3),
            "compactions": len(s.compaction_history),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_compaction_instance: Optional[CompactionEngine] = None


def get_compaction_engine() -> CompactionEngine:
    """Return the module-level CompactionEngine singleton."""
    global _compaction_instance
    if _compaction_instance is None:
        _compaction_instance = CompactionEngine()
    return _compaction_instance
