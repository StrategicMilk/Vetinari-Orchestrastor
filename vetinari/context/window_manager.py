"""Context Window Manager (C4).

============================
Tracks token usage per conversation and compresses context when approaching
the model's context window limit.

Strategy:
  - Track tokens per message in the conversation history
  - When usage exceeds 75% of the window, compress the oldest 50% into a summary
  - Stage-boundary compression between pipeline stages
  - Configurable per-model context window sizes

Token estimation uses a simple heuristic (words x 1.3) unless a proper
tokenizer is available.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Known context window sizes ────────────────────────────────────────

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Local models (LM Studio)
    "qwen2.5-coder-7b": 32768,
    "qwen2.5-coder-14b": 32768,
    "qwen2.5-72b": 32768,
    "qwen3-30b-a3b": 32768,
    "qwen2.5-vl-32b": 32768,
    # Cloud models
    "claude-3.5-sonnet": 200000,
    "claude-opus-4": 200000,
    "claude-sonnet-4": 200000,
    "gpt-4o": 128000,
    "gemini-1.5-pro": 1000000,
    # Defaults
    "default": 32768,
}


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using word-based heuristic.

    Average English text: ~1.3 tokens per word.
    Code tends to be ~1.5 tokens per word due to symbols.

    Returns:
        The computed value.
    """
    if not text:
        return 0
    word_count = len(text.split())
    # Heuristic: if text has lots of symbols, use higher multiplier
    symbol_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    multiplier = 1.5 if symbol_ratio > 0.15 else 1.3
    return int(word_count * multiplier)


@dataclass
class ConversationMessage:
    """A single message in the conversation context."""

    role: str  # "system", "user", "assistant"
    content: str
    token_count: int = 0
    is_compressed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


@dataclass
class WindowState:
    """Tracks the state of a context window."""

    model_id: str
    max_tokens: int
    used_tokens: int = 0
    message_count: int = 0
    compressions: int = 0
    compression_ratio: float = 1.0  # ratio of original to compressed


class ContextWindowManager:
    """Manages context window usage and compression.

    Usage::

        mgr = ContextWindowManager(model_id="qwen2.5-coder-14b")
        mgr.add_message("user", "Write a function to sort a list")
        mgr.add_message("assistant", "def sort_list(items): ...")

        if mgr.should_compress():
            mgr.compress()
    """

    def __init__(
        self,
        model_id: str = "default",
        max_context_ratio: float = 0.75,
        compression_target: float = 0.5,
        summary_max_tokens: int = 500,
    ):
        self._model_id = model_id
        self._max_context_ratio = max_context_ratio
        self._compression_target = compression_target
        self._summary_max_tokens = summary_max_tokens
        self._messages: list[ConversationMessage] = []
        self._lock = threading.Lock()

        # Determine window size
        self._window_size = _MODEL_CONTEXT_WINDOWS.get(model_id, _MODEL_CONTEXT_WINDOWS["default"])
        self._threshold = int(self._window_size * self._max_context_ratio)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def used_tokens(self) -> int:
        return sum(m.token_count for m in self._messages)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self._window_size - self.used_tokens)

    @property
    def usage_ratio(self) -> float:
        return self.used_tokens / self._window_size if self._window_size > 0 else 0.0

    def add_message(self, role: str, content: str, **metadata: Any) -> int:
        """Add a message and return its estimated token count.

        Args:
            role: The role.
            content: The content.
            **metadata: Additional metadata key-value pairs for the message.

        Returns:
            The computed value.
        """
        msg = ConversationMessage(role=role, content=content, metadata=metadata)
        with self._lock:
            self._messages.append(msg)
        return msg.token_count

    def should_compress(self) -> bool:
        """Check if compression is needed (usage > threshold)."""
        return self.used_tokens > self._threshold

    def compress(self, summary_fn: Any | None = None) -> int:
        """Compress the oldest 50% of messages into a summary.

        Args:
            summary_fn: Optional callable(text) -> str that produces a
                        summary. If not provided, uses a simple truncation.

        Returns:
            Number of tokens saved.
        """
        with self._lock:
            if len(self._messages) < 3:
                return 0

            # Find the midpoint (compress oldest half)
            midpoint = len(self._messages) // 2
            to_compress = self._messages[:midpoint]
            to_keep = self._messages[midpoint:]

            # Build text to summarize
            original_text = "\n".join(f"[{m.role}]: {m.content}" for m in to_compress)
            original_tokens = sum(m.token_count for m in to_compress)

            # Generate summary
            if summary_fn:
                try:
                    summary = summary_fn(original_text)
                except Exception as e:
                    logger.warning("Summary function failed: %s", e)
                    summary = self._simple_compress(original_text)
            else:
                summary = self._simple_compress(original_text)

            # Create compressed message
            compressed_msg = ConversationMessage(
                role="system",
                content=f"[Compressed context summary]\n{summary}",
                is_compressed=True,
                metadata={"original_messages": len(to_compress)},
            )

            saved = original_tokens - compressed_msg.token_count
            self._messages = [compressed_msg, *to_keep]

            logger.info(
                "Context compressed: %d msgs → 1 summary, saved %d tokens (%.0f%%)",
                len(to_compress),
                saved,
                (saved / original_tokens * 100) if original_tokens > 0 else 0,
            )
            return saved

    def stage_boundary_compress(self, stage_name: str = "") -> int:
        """Compress at pipeline stage boundaries.

        Called between pipeline stages to keep context lean. Only compresses
        if usage is above 50% (lower threshold than regular compression).

        Returns:
            The computed value.
        """
        if self.usage_ratio > 0.5:
            logger.info("Stage boundary compression at '%s'", stage_name)
            return self.compress()
        return 0

    def get_messages(self) -> list[dict[str, str]]:
        """Return messages in the format expected by LLM APIs."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def get_state(self) -> WindowState:
        """Return current window state."""
        return WindowState(
            model_id=self._model_id,
            max_tokens=self._window_size,
            used_tokens=self.used_tokens,
            message_count=len(self._messages),
        )

    def clear(self) -> None:
        """Clear all messages."""
        with self._lock:
            self._messages.clear()

    def _simple_compress(self, text: str) -> str:
        """Simple compression: keep first and last parts, drop middle."""
        words = text.split()
        target_words = self._summary_max_tokens  # rough approximation
        if len(words) <= target_words:
            return text
        half = target_words // 2
        return " ".join(words[:half]) + "\n[...compressed...]\n" + " ".join(words[-half:])

    def to_dict(self) -> dict[str, Any]:
        """Dashboard-friendly state."""
        return {
            "model_id": self._model_id,
            "window_size": self._window_size,
            "used_tokens": self.used_tokens,
            "remaining_tokens": self.remaining_tokens,
            "usage_ratio": round(self.usage_ratio, 3),
            "message_count": len(self._messages),
            "threshold": self._threshold,
            "should_compress": self.should_compress(),
        }


# ── Singleton ─────────────────────────────────────────────────────────

_window_manager: ContextWindowManager | None = None


def get_window_manager(model_id: str = "default") -> ContextWindowManager:
    """Get or create the global context window manager.

    Returns:
        The ContextWindowManager result.
    """
    global _window_manager
    if _window_manager is None:
        _window_manager = ContextWindowManager(model_id=model_id)
    return _window_manager
