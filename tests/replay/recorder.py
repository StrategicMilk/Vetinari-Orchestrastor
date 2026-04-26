"""Replay recorder — captures LLM and tool calls as structured JSONL events.

Records each LLM inference call (prompt, response, model_id, latency_ms) and
each tool invocation (tool_name, input, output) with sequence numbers and
timestamps so a ReplayReplayer can reproduce the exact call sequence later
without touching a live model.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Event type discriminators ─────────────────────────────────────────────────
_EVENT_LLM = "llm_call"
_EVENT_TOOL = "tool_call"


class ReplayRecorder:
    """Records LLM inference calls and tool invocations to a JSONL file.

    Usage::

        recorder = ReplayRecorder(output_dir=tmp_path)
        recorder.start_recording("my_scenario")
        recorder.record_llm_call(
            prompt="What is 2+2?",
            response="4",
            model_id="test-model-7b",
            latency_ms=120,
        )
        path = recorder.stop_recording()
        # path is a .jsonl file with one event per line

    The recorder is thread-safe: multiple threads can call ``record_llm_call``
    and ``record_tool_call`` concurrently.

    Args:
        output_dir: Directory where JSONL scenario files will be written.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._scenario_name: str | None = None
        self._events: list[dict[str, Any]] = []
        self._seq: int = 0
        self._lock = threading.Lock()
        self._is_recording = False

    # ── Public API ────────────────────────────────────────────────────────────

    def start_recording(self, scenario_name: str) -> None:
        """Begin a new recording session under the given scenario name.

        Clears any previously recorded events. Raises ``RuntimeError`` if a
        recording session is already in progress.

        Args:
            scenario_name: Human-readable name for this scenario. Used as the
                stem of the output JSONL file.

        Raises:
            RuntimeError: If ``start_recording`` is called while a session is
                already active.
        """
        with self._lock:
            if self._is_recording:
                raise RuntimeError(
                    f"Recording already in progress for scenario {self._scenario_name!r}. Call stop_recording() first."
                )
            self._scenario_name = scenario_name
            self._events = []
            self._seq = 0
            self._is_recording = True
        logger.debug("Replay recorder started for scenario %r", scenario_name)

    def record_llm_call(
        self,
        prompt: str,
        response: str,
        model_id: str,
        latency_ms: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a single LLM inference call.

        Args:
            prompt: The prompt text sent to the model.
            response: The model's response text.
            model_id: Identifier of the model that produced the response.
            latency_ms: Wall-clock inference time in milliseconds.
            metadata: Optional extra key-value pairs to store with the event.

        Raises:
            RuntimeError: If no recording session is active.
        """
        self._append_event(
            event_type=_EVENT_LLM,
            payload={
                "prompt": prompt,
                "response": response,
                "model_id": model_id,
                "latency_ms": latency_ms,
                "metadata": metadata or {},
            },
        )

    def record_tool_call(
        self,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a single tool invocation.

        Args:
            tool_name: Name of the tool that was called.
            input_data: Input passed to the tool (must be JSON-serializable).
            output_data: Output returned by the tool (must be JSON-serializable).
            metadata: Optional extra key-value pairs to store with the event.

        Raises:
            RuntimeError: If no recording session is active.
        """
        self._append_event(
            event_type=_EVENT_TOOL,
            payload={
                "tool_name": tool_name,
                "input": input_data,
                "output": output_data,
                "metadata": metadata or {},
            },
        )

    def stop_recording(self) -> Path:
        """Flush all recorded events to a JSONL file and end the session.

        Returns:
            Absolute path to the written JSONL file.

        Raises:
            RuntimeError: If no recording session is active.
        """
        with self._lock:
            if not self._is_recording:
                raise RuntimeError("No active recording session. Call start_recording() first.")
            events_snapshot = list(self._events)
            scenario = self._scenario_name
            self._is_recording = False

        out_path = self._output_dir / f"{scenario}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for event in events_snapshot:
                fh.write(json.dumps(event) + "\n")

        logger.debug("Replay recording saved: %s (%d events)", out_path, len(events_snapshot))
        return out_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a single event to the in-memory buffer (thread-safe).

        Args:
            event_type: Discriminator string (_EVENT_LLM or _EVENT_TOOL).
            payload: Event-specific data fields.

        Raises:
            RuntimeError: If no recording session is active.
        """
        with self._lock:
            if not self._is_recording:
                raise RuntimeError("No active recording session. Call start_recording() before recording events.")
            seq = self._seq
            self._seq += 1
            event: dict[str, Any] = {
                "seq": seq,
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
            self._events.append(event)
