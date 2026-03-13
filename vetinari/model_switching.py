"""Mid-session model switching for Vetinari.

Enables graceful model transitions without losing context,
automatic fallback on model failure, and CLI control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelSwitch:
    """Record of a model switch event."""

    timestamp: str
    from_model: str
    to_model: str
    reason: str  # "manual", "fallback", "context_limit", "cost"
    context_preserved: bool = True


@dataclass
class ModelSwitchConfig:
    """Configuration for model switching behavior."""

    fallback_chain: list[str] = field(
        default_factory=lambda: [
            "qwen2.5-coder-32b",
            "qwen2.5-coder-14b",
            "qwen2.5-coder-7b",
        ]
    )
    auto_fallback: bool = True  # Auto-switch on failure
    context_handoff: bool = True  # Summarize context on switch
    max_switches_per_session: int = 10


class ModelSwitcher:
    """Manages mid-session model switching."""

    def __init__(self, config: ModelSwitchConfig = None):
        self._config = config or ModelSwitchConfig()
        self._current_model: str | None = None
        self._history: list[ModelSwitch] = []
        self._switch_count: int = 0

    @property
    def current_model(self) -> str | None:
        return self._current_model

    def set_initial_model(self, model_id: str) -> None:
        self._current_model = model_id

    def switch(self, to_model: str, reason: str = "manual") -> ModelSwitch:
        """Switch to a different model."""
        if self._switch_count >= self._config.max_switches_per_session:
            raise RuntimeError("Max model switches per session exceeded")

        from_model = self._current_model or "none"
        switch = ModelSwitch(
            timestamp=datetime.now().isoformat(),
            from_model=from_model,
            to_model=to_model,
            reason=reason,
            context_preserved=self._config.context_handoff,
        )
        self._current_model = to_model
        self._history.append(switch)
        self._switch_count += 1
        logger.info("Model switch: %s -> %s (reason: %s)", from_model, to_model, reason)
        return switch

    def fallback(self) -> ModelSwitch | None:
        """Switch to next model in fallback chain."""
        if not self._config.auto_fallback:
            return None
        chain = self._config.fallback_chain
        if self._current_model in chain:
            idx = chain.index(self._current_model)
            if idx + 1 < len(chain):
                return self.switch(chain[idx + 1], reason="fallback")
        elif chain:
            return self.switch(chain[0], reason="fallback")
        return None

    def get_history(self) -> list[dict[str, Any]]:
        from dataclasses import asdict

        return [asdict(s) for s in self._history]

    def summarize_context(self, messages: list[dict]) -> str:
        """Summarize conversation context for handoff to new model."""
        if not messages:
            return ""
        # Extract key information for context transfer
        key_points = []
        for msg in messages[-10:]:  # Last 10 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content and len(content) > 50:
                key_points.append(f"[{role}] {content[:200]}...")
            elif content:
                key_points.append(f"[{role}] {content}")
        return "\n".join(key_points)


_switcher: ModelSwitcher | None = None


def get_model_switcher() -> ModelSwitcher:
    global _switcher
    if _switcher is None:
        _switcher = ModelSwitcher()
    return _switcher
