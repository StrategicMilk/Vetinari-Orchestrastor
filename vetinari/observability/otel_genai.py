"""OpenTelemetry GenAI Semantic Conventions tracer (P10.1).

Implements the OpenTelemetry GenAI semantic conventions for agent spans,
tool calls, and token accounting.  Works without the OTEL SDK installed —
falls back to in-process recording with JSON export.

Standard attribute names follow the GenAI semantic conventions draft spec:
  https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional OpenTelemetry import ────────────────────────────────────────────

_OTEL_AVAILABLE = False
_otel_trace = None  # type: ignore[assignment]
try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-untyped]

    _OTEL_AVAILABLE = True
    logger.debug("opentelemetry available — GenAI tracer will emit real spans")
except ImportError:
    logger.debug("opentelemetry not installed — using in-process GenAI tracer")

# OTel tracer name following GenAI semantic conventions
_OTEL_TRACER_NAME = "vetinari.genai"


# ── GenAI attribute name constants ───────────────────────────────────────────

ATTR_AGENT_NAME = "gen_ai.agent.name"
ATTR_OPERATION = "gen_ai.operation.name"
ATTR_REQUEST_MODEL = "gen_ai.request.model"
ATTR_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_INPUT = "gen_ai.tool.input"
ATTR_TOOL_OUTPUT = "gen_ai.tool.output"
ATTR_SPAN_STATUS = "gen_ai.span.status"


# ── SpanContext dataclass ────────────────────────────────────────────────────


@dataclass
class SpanContext:
    """Lightweight span record adhering to GenAI semantic conventions.

    Attributes:
        trace_id: Hex trace identifier (32 chars).
        span_id: Hex span identifier (16 chars).
        agent_name: Name of the agent that owns this span.
        operation: Operation name (e.g. ``"chat"``, ``"embeddings"``).
        start_time: Monotonic clock value at span creation.
        attributes: Mutable attribute bag keyed by GenAI convention names.
        events: Ordered list of event dicts recorded on this span.
        _end_time: Set when the span is closed; ``None`` while active.
    """

    trace_id: str
    span_id: str
    agent_name: str
    operation: str
    start_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    _end_time: float | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _close(self, status: str, tokens_used: int) -> None:
        """Finalise this span (called by GenAITracer.end_agent_span)."""
        self._end_time = time.monotonic()
        self.attributes[ATTR_SPAN_STATUS] = status
        if tokens_used:
            existing = self.attributes.get(ATTR_OUTPUT_TOKENS, 0)
            self.attributes[ATTR_OUTPUT_TOKENS] = existing + tokens_used

    @property
    def duration_ms(self) -> float:
        """Elapsed wall-clock time in milliseconds.

        Returns:
            Duration from span start to end (or now if still active).
        """
        end = self._end_time if self._end_time is not None else time.monotonic()
        return (end - self.start_time) * 1_000

    @property
    def is_active(self) -> bool:
        """True while the span has not been ended."""
        return self._end_time is None

    def to_dict(self) -> dict[str, Any]:
        """Serialise span to a JSON-compatible dict.

        Returns:
            Dictionary representation suitable for ``json.dumps``.
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "agent_name": self.agent_name,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self._end_time,
            "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes,
            "events": self.events,
        }


# ── GenAITracer ──────────────────────────────────────────────────────────────


class GenAITracer:
    """Singleton tracer that records GenAI semantic-convention spans.

    Instantiate via :func:`get_genai_tracer` — do not construct directly.

    Example::

        tracer = get_genai_tracer()
        span = tracer.start_agent_span("builder", "chat", model="qwen-32b")
        tracer.record_tool_call(span, "code_search", '{"q": "foo"}', '"bar"')  # noqa: VET034
        tracer.end_agent_span(span, status="ok", tokens_used=512)
        tracer.export_traces("/tmp/traces.json")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: list[SpanContext] = []
        self._active: dict[str, SpanContext] = {}  # span_id -> SpanContext
        self._total_tokens: int = 0
        self._current_trace_id: str = uuid.uuid4().hex

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_agent_span(
        self,
        agent_name: str,
        operation: str,
        model: str = "",
    ) -> SpanContext:
        """Open a new agent span.

        When OpenTelemetry is available, also starts a real OTel span that
        will be exported via the configured OTel exporter.

        Args:
            agent_name: Logical agent name (e.g. ``"builder"``).
            operation: GenAI operation (e.g. ``"chat"``, ``"embeddings"``).
            model: Request model identifier, if known.

        Returns:
            A :class:`SpanContext` representing the open span.
        """
        span_id = uuid.uuid4().hex[:16]
        attrs: dict[str, Any] = {
            ATTR_AGENT_NAME: agent_name,
            ATTR_OPERATION: operation,
        }
        if model:
            attrs[ATTR_REQUEST_MODEL] = model

        ctx = SpanContext(
            trace_id=self._current_trace_id,
            span_id=span_id,
            agent_name=agent_name,
            operation=operation,
            start_time=time.monotonic(),
            attributes=attrs,
        )

        # Bridge to real OTel spans when SDK is available
        if _OTEL_AVAILABLE and _otel_trace is not None:
            tracer = _otel_trace.get_tracer(_OTEL_TRACER_NAME)
            otel_span = tracer.start_span(
                f"gen_ai.{operation}",
                attributes={k: str(v) for k, v in attrs.items()},
            )
            ctx._otel_span = otel_span  # type: ignore[attr-defined]

        with self._lock:
            self._active[span_id] = ctx

        logger.debug(
            "GenAI span started: agent=%s op=%s span_id=%s otel=%s",
            agent_name,
            operation,
            span_id,
            _OTEL_AVAILABLE,
        )
        return ctx

    def end_agent_span(
        self,
        span: SpanContext,
        status: str = "ok",
        tokens_used: int = 0,
    ) -> None:
        """Close a span and move it to the completed list.

        Also ends the corresponding OTel span if one was created.

        Args:
            span: The :class:`SpanContext` returned by :meth:`start_agent_span`.
            status: Completion status — ``"ok"`` or ``"error"``.
            tokens_used: Total tokens consumed (added to output token count).
        """
        if not span.is_active:
            logger.warning("end_agent_span called on already-closed span %s", span.span_id)
            return

        span._close(status=status, tokens_used=tokens_used)

        # End the real OTel span if one was attached
        otel_span = getattr(span, "_otel_span", None)
        if otel_span is not None:
            if tokens_used:
                otel_span.set_attribute(ATTR_OUTPUT_TOKENS, tokens_used)
            otel_span.set_attribute(ATTR_SPAN_STATUS, status)
            otel_span.end()

        with self._lock:
            self._active.pop(span.span_id, None)
            self._spans.append(span)
            self._total_tokens += tokens_used

        logger.debug(
            "GenAI span ended: agent=%s status=%s tokens=%d duration=%.1fms",
            span.agent_name,
            status,
            tokens_used,
            span.duration_ms,
        )

    def record_tool_call(
        self,
        span: SpanContext,
        tool_name: str,
        input_data: str,
        output_data: str,
    ) -> None:
        """Append a tool-call event to an open span.

        Args:
            span: The parent :class:`SpanContext`.
            tool_name: Name of the tool (e.g. ``"code_search"``).
            input_data: Serialised input passed to the tool.
            output_data: Serialised output returned from the tool.
        """
        event: dict[str, Any] = {
            "name": "gen_ai.tool.call",
            "timestamp": time.monotonic(),
            "attributes": {
                ATTR_TOOL_NAME: tool_name,
                ATTR_TOOL_INPUT: input_data[:4096],
                ATTR_TOOL_OUTPUT: output_data[:4096],
            },
        }
        span.events.append(event)
        logger.debug("Tool call recorded: tool=%s span=%s", tool_name, span.span_id)

    def export_traces(self, filepath: str) -> int:
        """Write all completed spans to a JSON file for external ingestion.

        Args:
            filepath: Destination path (created if it does not exist).

        Returns:
            Number of spans exported.

        Raises:
            OSError: If the file cannot be written.
        """
        with self._lock:
            completed = [s.to_dict() for s in self._spans]

        output = {
            "schema_version": "1.0",
            "service": "vetinari",
            "convention": "opentelemetry-genai-semconv",
            "exported_at": time.time(),
            "spans": completed,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)

        logger.info("Exported %d GenAI spans to %s", len(completed), filepath)
        return len(completed)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate tracer statistics.

        Returns:
            Dictionary with ``total_spans``, ``active_spans``, and
            ``total_tokens`` keys.
        """
        with self._lock:
            return {
                "total_spans": len(self._spans),
                "active_spans": len(self._active),
                "total_tokens": self._total_tokens,
            }

    def reset(self) -> None:
        """Clear all recorded spans (intended for testing)."""
        with self._lock:
            self._spans.clear()
            self._active.clear()
            self._total_tokens = 0
            self._current_trace_id = uuid.uuid4().hex


# ── Singleton accessor ───────────────────────────────────────────────────────

_tracer_instance: GenAITracer | None = None
_tracer_lock = threading.Lock()


def get_genai_tracer() -> GenAITracer:
    """Return the process-global :class:`GenAITracer` instance.

    Returns:
        The singleton :class:`GenAITracer`.
    """
    global _tracer_instance
    if _tracer_instance is None:
        with _tracer_lock:
            if _tracer_instance is None:
                _tracer_instance = GenAITracer()
    return _tracer_instance


def reset_genai_tracer() -> None:
    """Reset the singleton (intended for testing)."""
    global _tracer_instance
    with _tracer_lock:
        _tracer_instance = None
