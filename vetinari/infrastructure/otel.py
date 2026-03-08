"""
OpenTelemetry Integration for Vetinari
========================================

Wraps Vetinari's existing telemetry system with OpenTelemetry-compatible
span creation, trace hierarchy for fan-out patterns, and optional OTLP
export. Gracefully degrades to no-op when opentelemetry is not installed.

Configuration:
    VETINARI_OTEL_ENDPOINT: OTLP endpoint URL (e.g. "http://localhost:4317")
    VETINARI_OTEL_SERVICE_NAME: Service name (default: "vetinari")

Usage:
    from vetinari.infrastructure.otel import get_otel_integration

    otel = get_otel_integration()
    with otel.agent_span("BUILDER", model="qwen3-32b") as span:
        span.set_attribute("tokens", 1500)
        # ... do work ...
"""

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Detect whether opentelemetry is available
_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    _OTEL_AVAILABLE = True
except ImportError:
    otel_trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    Resource = None  # type: ignore

# Optional OTLP exporter
_OTLP_AVAILABLE = False
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _OTLP_AVAILABLE = True
except ImportError:
    OTLPSpanExporter = None  # type: ignore


@dataclass
class SpanRecord:
    """Lightweight record of a completed span for local telemetry integration."""
    span_id: str
    trace_id: str
    parent_span_id: str = ""
    operation_name: str = ""
    agent_type: str = ""
    model: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"        # "ok", "error"
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "agent_type": self.agent_type,
            "model": self.model,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
            "error_message": self.error_message,
        }


class _NoOpSpan:
    """No-op span implementation when OpenTelemetry is not installed."""

    def __init__(self, name: str = ""):
        self._name = name
        self._attributes: Dict[str, Any] = {}
        self._start = time.time()

    def set_attribute(self, key: str, value: Any) -> None:
        self._attributes[key] = value

    def set_status(self, status: Any, description: str = "") -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def end(self) -> None:
        pass


class OTelIntegration:
    """
    OpenTelemetry integration that wraps Vetinari's existing telemetry.

    Provides:
    - Agent span creation with rich attributes
    - Trace hierarchy for fan-out (parallel agent execution) patterns
    - Optional OTLP export to external backends
    - Graceful no-op fallback when opentelemetry is not installed
    - Local span recording for integration with existing TelemetryCollector
    """

    def __init__(
        self,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        self._service_name = service_name or os.environ.get("VETINARI_OTEL_SERVICE_NAME", "vetinari")
        self._endpoint = endpoint or os.environ.get("VETINARI_OTEL_ENDPOINT", "")
        self._tracer = None
        self._provider = None
        self._lock = threading.Lock()
        self._span_records: List[SpanRecord] = []
        self._active_traces: Dict[str, str] = {}  # trace_id -> root span_id
        self._initialized = False

        self._initialize()

    def _initialize(self) -> None:
        """Initialize OpenTelemetry provider and tracer if available."""
        if not _OTEL_AVAILABLE:
            logger.info("OpenTelemetry not installed; OTelIntegration running in no-op mode")
            self._initialized = True
            return

        try:
            resource = Resource.create({"service.name": self._service_name})
            self._provider = TracerProvider(resource=resource)

            # Set up OTLP exporter if endpoint is configured
            if self._endpoint and _OTLP_AVAILABLE:
                exporter = OTLPSpanExporter(endpoint=self._endpoint, insecure=True)
                processor = BatchSpanProcessor(exporter)
                self._provider.add_span_processor(processor)
                logger.info("OTLP exporter configured: %s", self._endpoint)
            elif self._endpoint:
                logger.warning(
                    "OTLP endpoint configured (%s) but opentelemetry-exporter-otlp not installed",
                    self._endpoint,
                )

            otel_trace.set_tracer_provider(self._provider)
            self._tracer = otel_trace.get_tracer(self._service_name, "1.0.0")
            self._initialized = True
            logger.info("OTelIntegration initialized with OpenTelemetry (service=%s)", self._service_name)
        except Exception as exc:
            logger.error("Failed to initialize OpenTelemetry: %s", exc)
            self._tracer = None
            self._initialized = True

    @contextmanager
    def agent_span(
        self,
        agent_type: str,
        model: str = "",
        operation: str = "execute",
        parent_trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """
        Create a span for an agent operation.

        Args:
            agent_type: The agent type (e.g., "BUILDER", "RESEARCHER").
            model: The model being used.
            operation: Operation name (e.g., "execute", "verify").
            parent_trace_id: Optional parent trace for fan-out hierarchy.
            attributes: Additional span attributes.

        Yields:
            A span object (real OTel span or NoOpSpan).
        """
        span_name = f"agent.{agent_type}.{operation}"
        start_time = time.time()
        import uuid
        span_id = uuid.uuid4().hex[:16]
        trace_id = parent_trace_id or uuid.uuid4().hex[:32]

        all_attrs = {
            "vetinari.agent_type": agent_type,
            "vetinari.model": model,
            "vetinari.operation": operation,
        }
        if attributes:
            all_attrs.update(attributes)

        record = SpanRecord(
            span_id=span_id,
            trace_id=trace_id,
            operation_name=span_name,
            agent_type=agent_type,
            model=model,
            start_time=start_time,
            attributes=all_attrs,
        )

        if self._tracer is not None:
            # Real OpenTelemetry span
            span = self._tracer.start_span(span_name)
            for k, v in all_attrs.items():
                try:
                    span.set_attribute(k, str(v) if not isinstance(v, (int, float, bool, str)) else v)
                except Exception:
                    pass
            try:
                yield span
                record.status = "ok"
            except Exception as exc:
                record.status = "error"
                record.error_message = str(exc)
                try:
                    span.record_exception(exc)
                    span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                except Exception:
                    pass
                raise
            finally:
                record.end_time = time.time()
                record.duration_ms = (record.end_time - record.start_time) * 1000
                span.end()
                self._record_span(record)
        else:
            # No-op mode
            noop = _NoOpSpan(span_name)
            for k, v in all_attrs.items():
                noop.set_attribute(k, v)
            try:
                yield noop
                record.status = "ok"
            except Exception as exc:
                record.status = "error"
                record.error_message = str(exc)
                raise
            finally:
                record.end_time = time.time()
                record.duration_ms = (record.end_time - record.start_time) * 1000
                self._record_span(record)

    @contextmanager
    def fan_out_trace(self, operation: str = "fan_out") -> Generator[str, None, None]:
        """
        Create a parent trace for fan-out patterns (parallel agent execution).

        Yields a trace_id that child agent_span calls can reference via parent_trace_id
        to establish a proper trace hierarchy.

        Usage:
            with otel.fan_out_trace("parallel_research") as trace_id:
                with otel.agent_span("RESEARCHER", parent_trace_id=trace_id):
                    ...
                with otel.agent_span("EXPLORER", parent_trace_id=trace_id):
                    ...
        """
        import uuid
        trace_id = uuid.uuid4().hex[:32]
        start_time = time.time()

        logger.debug("Fan-out trace started: %s (%s)", trace_id, operation)
        with self._lock:
            self._active_traces[trace_id] = operation

        try:
            yield trace_id
        finally:
            with self._lock:
                self._active_traces.pop(trace_id, None)
            duration = (time.time() - start_time) * 1000
            logger.debug("Fan-out trace completed: %s (%.1fms)", trace_id, duration)

    def record_token_usage(
        self,
        agent_type: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        """
        Record token usage and bridge to the existing TelemetryCollector.

        Args:
            agent_type: Agent type string.
            model: Model identifier.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            total_tokens: Total token count.
        """
        # Record in existing telemetry
        try:
            from vetinari.telemetry import get_telemetry_collector
            collector = get_telemetry_collector()
            collector.record_adapter_latency(
                provider="vetinari",
                model=model,
                latency_ms=0,
                success=True,
                tokens_used=total_tokens,
            )
        except Exception as exc:
            logger.debug("Could not bridge to TelemetryCollector: %s", exc)

        # Record as OTel event if tracer is available
        if self._tracer is not None:
            try:
                current_span = otel_trace.get_current_span()
                if current_span and current_span.is_recording():
                    current_span.set_attribute("vetinari.tokens.prompt", prompt_tokens)
                    current_span.set_attribute("vetinari.tokens.completion", completion_tokens)
                    current_span.set_attribute("vetinari.tokens.total", total_tokens)
            except Exception:
                pass

    def record_quality_score(self, agent_type: str, task_id: str, score: float) -> None:
        """
        Record a quality score as a span attribute or event.

        Args:
            agent_type: Agent type string.
            task_id: Task identifier.
            score: Quality score (0.0 to 1.0).
        """
        if self._tracer is not None:
            try:
                current_span = otel_trace.get_current_span()
                if current_span and current_span.is_recording():
                    current_span.set_attribute("vetinari.quality_score", score)
                    current_span.set_attribute("vetinari.task_id", task_id)
            except Exception:
                pass
        logger.debug("Quality score recorded: agent=%s task=%s score=%.2f",
                      agent_type, task_id, score)

    def get_span_records(self, limit: int = 100) -> List[SpanRecord]:
        """Return recorded spans for local inspection."""
        with self._lock:
            return list(self._span_records[-limit:])

    def get_active_traces(self) -> Dict[str, str]:
        """Return currently active fan-out traces."""
        with self._lock:
            return dict(self._active_traces)

    def shutdown(self) -> None:
        """Flush and shut down the OTel provider."""
        if self._provider is not None:
            try:
                self._provider.shutdown()
                logger.info("OTelIntegration shut down cleanly")
            except Exception as exc:
                logger.error("Error during OTel shutdown: %s", exc)

    @property
    def is_otel_available(self) -> bool:
        """Whether the real OpenTelemetry SDK is loaded."""
        return _OTEL_AVAILABLE

    @property
    def is_exporting(self) -> bool:
        """Whether spans are being exported to an OTLP backend."""
        return _OTEL_AVAILABLE and _OTLP_AVAILABLE and bool(self._endpoint)

    def _record_span(self, record: SpanRecord) -> None:
        """Store a span record locally for integration with existing telemetry."""
        with self._lock:
            self._span_records.append(record)
            # Keep bounded
            if len(self._span_records) > 10000:
                self._span_records = self._span_records[-5000:]

        # Bridge duration to existing telemetry
        if record.model and record.duration_ms > 0:
            try:
                from vetinari.telemetry import get_telemetry_collector
                collector = get_telemetry_collector()
                collector.record_adapter_latency(
                    provider="vetinari",
                    model=record.model,
                    latency_ms=record.duration_ms,
                    success=(record.status == "ok"),
                )
            except Exception:
                pass


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------

_otel_instance: Optional[OTelIntegration] = None
_otel_lock = threading.Lock()


def get_otel_integration() -> OTelIntegration:
    """Get or create the global OTelIntegration singleton."""
    global _otel_instance
    if _otel_instance is None:
        with _otel_lock:
            if _otel_instance is None:
                _otel_instance = OTelIntegration()
    return _otel_instance


def reset_otel_integration() -> None:
    """Reset the OTel singleton (mainly for testing)."""
    global _otel_instance
    if _otel_instance is not None:
        _otel_instance.shutdown()
    _otel_instance = None
