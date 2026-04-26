"""Fake implementations for testing without LLM inference.

Provides lightweight stand-ins for the inference adapter and related
components.  These fakes return deterministic, configurable responses so
tests can exercise pipeline logic without loading a real model or making
network calls.

Example::

    from tests.fakes import FakeInferenceAdapter

    adapter = FakeInferenceAdapter(default_response="Hello, world!")
    result = adapter.generate("Say hello")
    assert result["choices"][0]["text"] == "Hello, world!"
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "FakeEventBus",
    "FakeInferenceAdapter",
    "FakeMemoryStore",
    "FakeModelPool",
]


# ── Fake Inference Adapter ──────────────────────────────────────────────────


class FakeInferenceAdapter:
    """Drop-in fake for inference adapters that returns canned responses.

    Tracks all calls for assertion in tests.  Supports both synchronous
    ``generate`` and streaming ``generate_stream`` interfaces.

    Args:
        default_response: The text returned by default for any prompt.
        responses: Optional mapping of prompt substrings to specific responses.
            If a prompt contains a key from this dict, the mapped response is
            returned instead of *default_response*.
        should_fail: If True, all calls raise RuntimeError (for error-path testing).
    """

    def __init__(
        self,
        default_response: str = "Fake LLM response for testing.",
        responses: dict[str, str] | None = None,
        should_fail: bool = False,
    ) -> None:
        self.default_response = default_response
        self.responses = responses or {}
        self.should_fail = should_fail
        self.call_history: list[dict[str, Any]] = []

    def _resolve_response(self, prompt: str) -> str:
        """Pick the response based on prompt content."""
        for substring, response in self.responses.items():
            if substring in prompt:
                return response
        return self.default_response

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Simulate a completion call and return a fake response dict.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instruction.
            temperature: Sampling temperature (ignored by fake).
            max_tokens: Token limit (ignored by fake).
            **kwargs: Extra arguments (recorded but ignored).

        Returns:
            A dict matching the adapter response format with ``choices``,
            ``usage``, and ``model`` fields.

        Raises:
            RuntimeError: If *should_fail* is True.
        """
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        })

        if self.should_fail:
            msg = "FakeInferenceAdapter configured to fail"
            raise RuntimeError(msg)

        text = self._resolve_response(prompt)
        return {
            "choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(text.split()),
                "total_tokens": len(prompt.split()) + len(text.split()),
            },
            "model": "fake-model-7b",
            "status": "ok",
        }

    def generate_stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Simulate streaming by yielding the response word-by-word.

        Args:
            prompt: The user prompt.
            **kwargs: Extra arguments (forwarded to generate).

        Yields:
            Individual words from the response text.
        """
        result = self.generate(prompt, **kwargs)
        text = result["choices"][0]["text"]
        for word in text.split():
            yield word + " "

    def reset(self) -> None:
        """Clear the call history."""
        self.call_history.clear()


# ── Fake Memory Store ───────────────────────────────────────────────────────


class FakeMemoryStore:
    """In-memory fake for the unified memory store.

    Stores memories as plain dicts in a list.  Supports search by
    substring matching on content.
    """

    def __init__(self) -> None:
        self._memories: list[dict[str, Any]] = []

    def store(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a memory entry and return its ID.

        Args:
            content: The memory content text.
            metadata: Optional metadata to attach.

        Returns:
            A string ID for the stored memory.
        """
        memory_id = f"mem_{len(self._memories):04d}"
        self._memories.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
        })
        return memory_id

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by substring match.

        Args:
            query: Search string.
            limit: Maximum results to return.

        Returns:
            List of matching memory dicts, newest first.
        """
        matches = [m for m in self._memories if query.lower() in m["content"].lower()]
        return matches[:limit]

    def clear(self) -> None:
        """Remove all stored memories."""
        self._memories.clear()

    @property
    def count(self) -> int:
        """Number of memories currently stored."""
        return len(self._memories)


# ── Fake Event Bus ──────────────────────────────────────────────────────────


class FakeEventBus:
    """In-memory fake for the event bus that records all emitted events.

    Supports both ``emit`` and ``on`` for testing event-driven flows
    without the real async machinery.
    """

    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, Any]]] = []
        self._handlers: dict[str, list[Any]] = {}

    def emit(self, event_name: str, data: dict[str, Any] | None = None) -> None:
        """Record an event emission and invoke registered handlers.

        Args:
            event_name: The event type name.
            data: Event payload.
        """
        payload = data or {}
        self.emitted.append((event_name, payload))
        for handler in self._handlers.get(event_name, []):
            try:
                handler(payload)
            except Exception:
                logger.exception("Fake event handler raised for %s", event_name)

    def on(self, event_name: str, handler: Any) -> None:
        """Register a handler for an event type.

        Args:
            event_name: The event type to listen for.
            handler: Callable invoked with the event data dict.
        """
        self._handlers.setdefault(event_name, []).append(handler)

    def reset(self) -> None:
        """Clear all emitted events and registered handlers."""
        self.emitted.clear()
        self._handlers.clear()


# ── Fake Model Pool ────────────────────────────────────────────────────────


class FakeModelPool:
    """Fake model pool that returns predetermined model selections.

    Useful for testing model routing logic without real model discovery.
    """

    def __init__(self, available_models: list[str] | None = None) -> None:
        self.available_models = available_models or [
            "test-model-7b-q4",
            "test-model-3b-q4",
        ]
        self.selection_history: list[str] = []

    def select_model(self, task_type: str = "general", **kwargs: Any) -> str:
        """Select a model for a task type (always returns the first available).

        Args:
            task_type: The type of task (ignored by fake).
            **kwargs: Extra selection criteria (ignored).

        Returns:
            The model identifier string.
        """
        model = self.available_models[0] if self.available_models else "fallback-model"
        self.selection_history.append(model)
        return model

    def list_models(self) -> list[str]:
        """Return all available model identifiers.

        Returns:
            List of model ID strings.
        """
        return list(self.available_models)
