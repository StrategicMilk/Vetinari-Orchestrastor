"""
Context Compaction Manager -- vetinari.memory.compaction

Auto-compaction at configurable context threshold, session tracking,
and transcript save-to-disk.

Usage::

    from vetinari.memory.compaction import get_compaction_manager

    mgr = get_compaction_manager()
    if mgr.should_compact(current_tokens=28000, max_tokens=32768):
        summary = mgr.compact(messages, model_id="qwen3-30b")
    mgr.save_transcript(session_id, messages)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_COMPACT_THRESHOLD = 0.80  # compact when usage >= 80% of context window
DEFAULT_TRANSCRIPT_DIR = ".vetinari/transcripts"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SessionInfo:
    """Tracks a single conversation session."""
    session_id: str
    started_at: float = field(default_factory=time.time)
    message_count: int = 0
    compaction_count: int = 0
    total_tokens_processed: int = 0
    last_compacted_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "message_count": self.message_count,
            "compaction_count": self.compaction_count,
            "total_tokens_processed": self.total_tokens_processed,
            "last_compacted_at": self.last_compacted_at,
        }


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    original_count: int
    compacted_count: int
    summary: str
    tokens_before: int
    tokens_after: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_count": self.original_count,
            "compacted_count": self.compacted_count,
            "summary": self.summary,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# CompactionManager
# ---------------------------------------------------------------------------

class CompactionManager:
    """
    Manages context window compaction, session tracking, and transcript persistence.

    - ``should_compact()`` checks whether token usage exceeds the threshold.
    - ``compact()`` summarises older messages to free context space.
    - ``save_transcript()`` writes full message history to disk.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_COMPACT_THRESHOLD,
        transcript_dir: str = DEFAULT_TRANSCRIPT_DIR,
    ):
        self._threshold = max(0.5, min(threshold, 0.95))
        self._transcript_dir = Path(transcript_dir)
        self._sessions: Dict[str, SessionInfo] = {}
        self._compaction_history: List[CompactionResult] = []

        logger.debug(
            "[CompactionManager] Initialized (threshold=%.0f%%)", self._threshold * 100
        )

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> SessionInfo:
        """Get an existing session or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionInfo(session_id=session_id)
            logger.debug("[CompactionManager] New session: %s", session_id)
        return self._sessions[session_id]

    def list_sessions(self) -> List[SessionInfo]:
        """Return all tracked sessions."""
        return list(self._sessions.values())

    # ------------------------------------------------------------------
    # Compaction logic
    # ------------------------------------------------------------------

    def should_compact(self, current_tokens: int, max_tokens: int) -> bool:
        """
        Return True if context usage has reached the compaction threshold.

        Args:
            current_tokens: Number of tokens currently in the context.
            max_tokens: Maximum context window size for the model.
        """
        if max_tokens <= 0:
            return False
        usage = current_tokens / max_tokens
        return usage >= self._threshold

    def compact(
        self,
        messages: List[Dict[str, Any]],
        model_id: str = "default",
        session_id: str = "default",
        keep_recent: int = 4,
    ) -> CompactionResult:
        """
        Compact a message list by summarising older messages.

        Keeps the system prompt (first message) and the most recent
        ``keep_recent`` messages intact.  Everything in between is
        replaced with a single summary message.

        Args:
            messages: The full message list (each dict has 'role' and 'content').
            model_id: Model identifier (for future LLM-based summarisation).
            session_id: Session to attribute this compaction to.
            keep_recent: Number of recent messages to preserve verbatim.

        Returns:
            CompactionResult with before/after stats and the summary text.
        """
        session = self.get_or_create_session(session_id)
        original_count = len(messages)
        tokens_before = self._estimate_tokens(messages)

        if original_count <= keep_recent + 1:
            # Nothing to compact
            return CompactionResult(
                original_count=original_count,
                compacted_count=original_count,
                summary="No compaction needed",
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        # Separate system prompt, middle messages, and recent messages
        system_msgs = [messages[0]] if messages and messages[0].get("role") == "system" else []
        start_idx = len(system_msgs)
        middle = messages[start_idx: -keep_recent] if keep_recent > 0 else messages[start_idx:]
        recent = messages[-keep_recent:] if keep_recent > 0 else []

        # Generate summary of middle messages
        summary = self._summarise_messages(middle, model_id)

        # Build compacted message list
        compacted = list(system_msgs)
        compacted.append({
            "role": "system",
            "content": f"[Context Summary]\n{summary}",
        })
        compacted.extend(recent)

        tokens_after = self._estimate_tokens(compacted)

        result = CompactionResult(
            original_count=original_count,
            compacted_count=len(compacted),
            summary=summary,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

        # Update session tracking
        session.compaction_count += 1
        session.last_compacted_at = time.time()
        self._compaction_history.append(result)

        # Replace messages in-place
        messages.clear()
        messages.extend(compacted)

        logger.info(
            "[CompactionManager] Compacted %d -> %d messages (tokens: %d -> %d)",
            original_count, len(compacted), tokens_before, tokens_after,
        )

        return result

    def _summarise_messages(self, messages: List[Dict[str, Any]], model_id: str) -> str:
        """
        Summarise a list of messages into a concise context summary.

        Currently uses extractive summarisation; can be upgraded to use
        LLM inference for higher quality summaries.
        """
        if not messages:
            return "No prior context."

        # Try LLM-based summarisation
        try:
            from vetinari.adapters.base import InferenceRequest
            from vetinari.adapter_manager import get_adapter_manager

            content_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    content_parts.append(f"{role}: {content[:200]}")

            joined = "\n".join(content_parts[-20:])  # last 20 messages max
            req = InferenceRequest(
                model_id=model_id,
                prompt=f"Summarise this conversation concisely, preserving key decisions, code, and context:\n\n{joined}",
                system_prompt="You are a conversation summariser. Be concise but preserve all important technical details.",
                max_tokens=512,
                temperature=0.2,
            )
            mgr = get_adapter_manager()
            resp = mgr.infer(req)
            if resp.status == "ok" and resp.output:
                return resp.output
        except Exception:
            pass

        # Fallback: extractive summary
        parts = []
        for msg in messages:
            content = str(msg.get("content", ""))
            if content.strip():
                parts.append(content[:150])
        return " | ".join(parts[-10:]) or "Prior conversation context."

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4

    # ------------------------------------------------------------------
    # Transcript persistence
    # ------------------------------------------------------------------

    def save_transcript(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a full message transcript to disk.

        Args:
            session_id: Session identifier.
            messages: The full message list.
            metadata: Optional metadata to include.

        Returns:
            The file path where the transcript was saved.
        """
        self._transcript_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{timestamp}.json"
        filepath = self._transcript_dir / filename

        session = self._sessions.get(session_id)
        transcript = {
            "session_id": session_id,
            "saved_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "session_info": session.to_dict() if session else None,
            "metadata": metadata or {},
            "messages": messages,
        }

        try:
            filepath.write_text(json.dumps(transcript, indent=2, default=str))
            logger.info("[CompactionManager] Transcript saved: %s", filepath)
        except Exception as e:
            logger.warning("[CompactionManager] Failed to save transcript: %s", e)

        return str(filepath)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_compaction_history(self) -> List[Dict[str, Any]]:
        """Return history of all compaction operations."""
        return [r.to_dict() for r in self._compaction_history]

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "threshold": self._threshold,
            "total_sessions": len(self._sessions),
            "total_compactions": len(self._compaction_history),
            "transcript_dir": str(self._transcript_dir),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_compaction_manager: Optional[CompactionManager] = None


def get_compaction_manager(**kwargs) -> CompactionManager:
    """Get or create the global CompactionManager."""
    global _compaction_manager
    if _compaction_manager is None:
        _compaction_manager = CompactionManager(**kwargs)
    return _compaction_manager


def reset_compaction_manager() -> None:
    """Reset the singleton (useful for testing)."""
    global _compaction_manager
    _compaction_manager = None
