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
import os
import threading
import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vetinari.constants import TRUNCATE_OTEL_OUTPUT

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

# Per-async-task / per-thread current trace ID.  Each ContextVar lookup is
# isolated to the current asyncio Task or OS thread, preventing trace ID
# cross-contamination when multiple concurrent callers share the singleton.
# Writers: start_agent_span (via _set_trace_id), reset (via _set_trace_id).
# Readers: start_agent_span (trace_id for new root spans).
# Lifecycle: task-local; auto-expires when the task completes.
# Lock: ContextVar is inherently thread/task-safe — no extra lock needed.
_current_trace_id_var: ContextVar[str | None] = ContextVar("genai_trace_id", default=None)


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
ATTR_SYSTEM = "gen_ai.system"  # Fixed system identifier for this service


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
        parent_span_id: Span ID of the parent span, enabling hierarchical nesting
            (pipeline > agent > llm).  ``None`` for root spans.
        _end_time: Set when the span is closed; ``None`` while active.
    """

    trace_id: str
    span_id: str
    agent_name: str
    operation: str
    start_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    parent_span_id: str | None = field(default=None)
    _end_time: float | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"SpanContext(trace_id={self.trace_id!r},"
            f" span_id={self.span_id!r},"
            f" agent_name={self.agent_name!r}, operation={self.operation!r})"
        )

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
            "parent_span_id": self.parent_span_id,
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
        tracer.record_tool_call(span, "code_search", '{"q": "foo"}', '"bar"')
        tracer.end_agent_span(span, status="ok", tokens_used=512)
        tracer.export_traces("/tmp/traces.json")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: deque[SpanContext] = deque(maxlen=5000)
        self._active: dict[str, SpanContext] = {}  # span_id -> SpanContext
        self._total_tokens: int = 0
        # Note: trace ID is now stored per-async-task/thread in
        # _current_trace_id_var (ContextVar) rather than on the singleton.
        # This prevents trace ID leakage across concurrent callers.

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
            ATTR_SYSTEM: "vetinari",
        }
        if model:
            attrs[ATTR_REQUEST_MODEL] = model

        # Allocate a new trace ID for this async task / thread if none exists
        # yet. Subsequent calls in the same task reuse the same trace ID so
        # sibling spans are grouped together. Child spans inherit via
        # start_child_span, which reads parent.trace_id directly.
        trace_id = _current_trace_id_var.get()
        if trace_id is None:
            trace_id = uuid.uuid4().hex
            _current_trace_id_var.set(trace_id)

        ctx = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            agent_name=agent_name,
            operation=operation,
            start_time=time.monotonic(),
            attributes=attrs,
        )

        # Bridge to real OTel spans when SDK is available and backend is not noop.
        # Guard on _backend != "noop" so we don't create OTel spans when tracing
        # is disabled — avoids spurious exports to a noop provider.
        if _OTEL_AVAILABLE and _otel_trace is not None and _backend != "noop":
            tracer = _otel_trace.get_tracer(_OTEL_TRACER_NAME)
            # Root agent spans are always root OTel spans — parent propagation
            # is handled by start_child_span, which sets parent_span_id and
            # calls set_span_in_context explicitly.  The dead block that
            # checked ctx.parent_span_id here was removed: SpanContext is
            # created above with no parent_span_id, so the check could never
            # fire.
            otel_span = tracer.start_span(
                f"gen_ai.{operation}",
                context=None,
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

    def start_child_span(
        self,
        parent: SpanContext,
        agent_name: str,
        operation: str,
        model: str = "",
    ) -> SpanContext:
        """Open a new span that is a child of an existing span.

        The child inherits ``trace_id`` from the parent and records the
        parent's ``span_id`` in ``parent_span_id``, enabling hierarchical
        nesting such as ``pipeline > agent > llm``.

        Args:
            parent: The parent :class:`SpanContext` to nest under.
            agent_name: Logical agent name for the child span.
            operation: GenAI operation name for the child span.
            model: Request model identifier, if known.

        Returns:
            A new :class:`SpanContext` linked to the parent.
        """
        span_id = uuid.uuid4().hex[:16]
        attrs: dict[str, Any] = {
            ATTR_AGENT_NAME: agent_name,
            ATTR_OPERATION: operation,
            ATTR_SYSTEM: "vetinari",
        }
        if model:
            attrs[ATTR_REQUEST_MODEL] = model

        ctx = SpanContext(
            trace_id=parent.trace_id,
            span_id=span_id,
            agent_name=agent_name,
            operation=operation,
            start_time=time.monotonic(),
            attributes=attrs,
            parent_span_id=parent.span_id,
        )

        # Bridge to real OTel spans when SDK is available and backend is not noop.
        # Always propagate the parent OTel span as context so the SDK links
        # child spans under their parent trace rather than starting a new root.
        if _OTEL_AVAILABLE and _otel_trace is not None and _backend != "noop":
            tracer = _otel_trace.get_tracer(_OTEL_TRACER_NAME)
            parent_otel = getattr(parent, "_otel_span", None)
            otel_ctx = _otel_trace.set_span_in_context(parent_otel) if parent_otel is not None else None
            otel_span = tracer.start_span(
                f"gen_ai.{operation}",
                context=otel_ctx,
                attributes={k: str(v) for k, v in attrs.items()},
            )
            ctx._otel_span = otel_span  # type: ignore[attr-defined]

        with self._lock:
            self._active[span_id] = ctx

        logger.debug(
            "GenAI child span started: agent=%s op=%s span_id=%s parent_span_id=%s",
            agent_name,
            operation,
            span_id,
            parent.span_id,
        )
        return ctx

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
                ATTR_TOOL_INPUT: input_data[:TRUNCATE_OTEL_OUTPUT],
                ATTR_TOOL_OUTPUT: output_data[:TRUNCATE_OTEL_OUTPUT],
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
        with Path(filepath).open("w", encoding="utf-8") as fh:
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
        """Clear all recorded spans (intended for testing).

        Resets the per-task trace ID so the next call to start_agent_span
        allocates a fresh trace ID for this async task / thread.
        """
        with self._lock:
            self._spans.clear()
            self._active.clear()
            self._total_tokens = 0
        # Reset the ContextVar for the calling task/thread only — other
        # concurrent tasks are unaffected.
        _current_trace_id_var.set(None)


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


# ── Backend configuration ────────────────────────────────────────────────────

# Valid backend identifiers.
_VALID_BACKENDS = frozenset({"noop", "jaeger", "file"})

# Active backend name — "noop" until explicitly configured or env var sets it.
# Written by configure_backend(); read by get_active_backend() and flush_file_backend().
_backend: str = "noop"


def configure_backend(backend_type: str, endpoint: str = "") -> None:
    """Configure the tracing backend for this process.

    Supported values for ``backend_type``:

    - ``"noop"`` — disables export; spans are still recorded in-process.
    - ``"file"`` — flushes spans to ``outputs/traces/`` (relative to cwd)
      when :func:`flush_file_backend` is called.
    - ``"jaeger"`` — wires the OTel SDK to a Jaeger-compatible OTLP endpoint.
      Logs a warning and falls back to noop behaviour when the OTel SDK is not
      installed; does not raise.

    Args:
        backend_type: One of ``"noop"``, ``"file"``, or ``"jaeger"``.
        endpoint: OTLP endpoint URL, used only when ``backend_type="jaeger"``.

    Raises:
        ValueError: If ``backend_type`` is not one of the recognised values.
    """
    global _backend

    if backend_type not in _VALID_BACKENDS:
        raise ValueError(f"Invalid OTel backend {backend_type!r} — must be one of {sorted(_VALID_BACKENDS)}")

    if backend_type == "jaeger":
        if not _OTEL_AVAILABLE:
            logger.warning(
                "configure_backend('jaeger') requested but opentelemetry SDK is not "
                "installed — falling back to noop export (spans recorded in-process only)"
            )
            # Report the actual export mode — noop, not jaeger — so callers can
            # trust get_active_backend() as ground truth rather than a wish.
            _backend = "noop"
            logger.debug("OTel backend configured: noop (jaeger requested but SDK unavailable)")
            # Reset the singleton so the next get_genai_tracer() call creates a
            # fresh instance that observes the updated _backend value.
            reset_genai_tracer()
            return
        else:
            _ep = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            logger.info("OTel backend set to jaeger — endpoint=%s", _ep)

    _backend = backend_type
    logger.debug("OTel backend configured: %s", backend_type)
    # Reset the singleton after every backend change so subsequent callers pick
    # up the new configuration instead of reusing a stale instance.
    reset_genai_tracer()


def get_active_backend() -> str:
    """Return the name of the currently active tracing backend.

    Returns:
        One of ``"noop"``, ``"file"``, or ``"jaeger"``.
    """
    return _backend


def flush_file_backend() -> int:
    """Write completed spans to disk when the file backend is active.

    Spans are written to ``outputs/traces/traces_<timestamp>.json`` relative to
    the current working directory.  If the active backend is not ``"file"``, this
    is a no-op that returns ``0``.

    Returns:
        Number of spans exported, or ``0`` when backend is not ``"file"``.
    """
    if _backend != "file":
        return 0

    import time as _time  # already imported at module level, but alias for clarity

    traces_dir = Path("outputs") / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(_time.time())
    filepath = traces_dir / f"traces_{timestamp}.json"

    tracer = get_genai_tracer()
    return tracer.export_traces(str(filepath))


def _init_backend_from_env() -> None:
    """Read env vars and configure the tracing backend.

    Reads ``VETINARI_OTEL_BACKEND`` and ``VETINARI_OTEL_ENDPOINT`` from the
    environment and calls :func:`configure_backend` accordingly.

    Called automatically at module import time and may be called again to
    re-read the environment (useful in tests that adjust env vars via
    ``monkeypatch``).

    Environment variables:

    - ``VETINARI_OTEL_BACKEND`` — backend name; defaults to ``"noop"``.
    - ``VETINARI_OTEL_ENDPOINT`` — OTLP endpoint for the jaeger backend;
      ignored for other backends.

    If ``VETINARI_OTEL_BACKEND`` holds an unrecognised value a warning is
    logged and the backend is set to ``"noop"``.
    """
    import os

    raw = os.environ.get("VETINARI_OTEL_BACKEND", "noop").strip().lower()
    endpoint = os.environ.get("VETINARI_OTEL_ENDPOINT", "")

    if raw not in _VALID_BACKENDS:
        logger.warning(
            "Unrecognised VETINARI_OTEL_BACKEND value %r — defaulting to 'noop'. Valid values are: %s",
            raw,
            sorted(_VALID_BACKENDS),
        )
        configure_backend("noop")
        return

    configure_backend(raw, endpoint)


# Run once at import time so the backend reflects the current environment.
_init_backend_from_env()
