"""Replay replayer — replays previously recorded LLM and tool call events.

Loads a JSONL file written by ReplayRecorder and serves recorded responses
back in sequence so integration tests can exercise agent logic without any
live model calls or tool side-effects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Event type discriminators (must match recorder.py) ───────────────────────
_EVENT_LLM = "llm_call"
_EVENT_TOOL = "tool_call"


class ReplayExhaustedError(Exception):
    """Raised when more calls are made than were recorded in the scenario.

    Args:
        event_type: The type of event that was requested but not available.
        calls_made: How many calls of this type have already been consumed.
    """

    def __init__(self, event_type: str, calls_made: int) -> None:
        super().__init__(
            f"Replay exhausted: requested {event_type!r} call #{calls_made + 1} "
            f"but only {calls_made} were recorded in this scenario."
        )
        self.event_type = event_type
        self.calls_made = calls_made


class ReplayReplayer:
    """Replays JSONL scenarios recorded by ReplayRecorder.

    Maintains separate cursors for LLM and tool events so the two streams can
    be consumed independently. Call ``start_replay()`` before any ``get_next_*``
    calls.  Use ``assert_all_replayed()`` at the end of a test to verify every
    recorded event was consumed.

    Usage::

        replayer = ReplayReplayer()
        replayer.start_replay(Path("scenario.jsonl"))
        response = replayer.get_next_llm_response("What is 2+2?")
        replayer.assert_all_replayed()
    """

    def __init__(self) -> None:
        self._llm_events: list[dict[str, Any]] = []
        self._tool_events: list[dict[str, Any]] = []
        self._llm_cursor: int = 0
        self._tool_cursor: int = 0
        self._scenario_path: Path | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start_replay(self, scenario_path: Path) -> None:
        """Load a JSONL scenario file and reset all cursors.

        Subsequent calls to ``get_next_llm_response`` and
        ``get_next_tool_response`` will serve events from this scenario in
        recorded order.

        Args:
            scenario_path: Path to the JSONL file produced by ReplayRecorder.

        Raises:
            FileNotFoundError: If the scenario file does not exist.
            ValueError: If the file contains malformed JSON on any line.
        """
        scenario_path = Path(scenario_path)
        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Replay scenario not found: {scenario_path}. Generate it with ReplayRecorder.stop_recording()."
            )

        all_events: list[dict[str, Any]] = []
        with scenario_path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    all_events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON on line {lineno} of {scenario_path}: {exc}") from exc

        # Partition into separate queues, preserving recorded sequence order
        self._llm_events = [e for e in all_events if e.get("event_type") == _EVENT_LLM]
        self._tool_events = [e for e in all_events if e.get("event_type") == _EVENT_TOOL]
        self._llm_cursor = 0
        self._tool_cursor = 0
        self._scenario_path = scenario_path

        logger.debug(
            "Replay loaded %d LLM events and %d tool events from %s",
            len(self._llm_events),
            len(self._tool_events),
            scenario_path,
        )

    def get_next_llm_response(self, prompt: str) -> str:
        """Return the next recorded LLM response in sequence.

        The ``prompt`` argument is accepted for call-site symmetry with real
        inference calls but is not used to match events — responses are served
        strictly in recorded order.

        Args:
            prompt: The prompt that would be sent to the model (informational only).

        Returns:
            The recorded response string for this call.

        Raises:
            RuntimeError: If ``start_replay`` has not been called.
            ReplayExhaustedError: If more LLM calls are made than were recorded.
        """
        if self._scenario_path is None:
            raise RuntimeError("Call start_replay() before get_next_llm_response().")

        if self._llm_cursor >= len(self._llm_events):
            raise ReplayExhaustedError(_EVENT_LLM, self._llm_cursor)

        event = self._llm_events[self._llm_cursor]
        self._llm_cursor += 1
        logger.debug(
            "Replay serving LLM event seq=%s (cursor=%d/%d)",
            event.get("seq"),
            self._llm_cursor,
            len(self._llm_events),
        )
        return str(event["response"])

    def get_next_tool_response(self, tool_name: str) -> Any:
        """Return the next recorded tool output in sequence.

        The ``tool_name`` argument is accepted for call-site symmetry but is
        not used for matching — outputs are served in recorded order.

        Args:
            tool_name: The name of the tool being called (informational only).

        Returns:
            The recorded output value for this tool call.

        Raises:
            RuntimeError: If ``start_replay`` has not been called.
            ReplayExhaustedError: If more tool calls are made than were recorded.
        """
        if self._scenario_path is None:
            raise RuntimeError("Call start_replay() before get_next_tool_response().")

        if self._tool_cursor >= len(self._tool_events):
            raise ReplayExhaustedError(_EVENT_TOOL, self._tool_cursor)

        event = self._tool_events[self._tool_cursor]
        self._tool_cursor += 1
        logger.debug(
            "Replay serving tool event seq=%s, tool=%r (cursor=%d/%d)",
            event.get("seq"),
            event.get("tool_name"),
            self._tool_cursor,
            len(self._tool_events),
        )
        return event["output"]

    def assert_all_replayed(self) -> None:
        """Assert that every recorded event was consumed during the test.

        Raises:
            AssertionError: If any LLM or tool events remain unconsumed,
                describing how many were recorded vs consumed of each type.
        """
        llm_remaining = len(self._llm_events) - self._llm_cursor
        tool_remaining = len(self._tool_events) - self._tool_cursor

        if llm_remaining > 0 or tool_remaining > 0:
            parts: list[str] = []
            if llm_remaining > 0:
                parts.append(
                    f"{llm_remaining} LLM event(s) recorded but never consumed "
                    f"(consumed {self._llm_cursor}/{len(self._llm_events)})"
                )
            if tool_remaining > 0:
                parts.append(
                    f"{tool_remaining} tool event(s) recorded but never consumed "
                    f"(consumed {self._tool_cursor}/{len(self._tool_events)})"
                )
            raise AssertionError("Not all replay events were consumed: " + "; ".join(parts))
