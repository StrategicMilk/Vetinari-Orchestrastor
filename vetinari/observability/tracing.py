"""Distributed Tracing (C7).

=========================
Optional OpenTelemetry integration with no-op fallback.

Span hierarchy: Pipeline > Stage > Agent > LLM Call.
Attributes: agent_type, mode, task_id, model_id, success, tokens.

When ``opentelemetry`` is installed, real spans are emitted.  Otherwise,
a lightweight no-op tracer records spans to structured logging only.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Try to import OpenTelemetry ───────────────────────────────────────

_OTEL_AVAILABLE = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
    _tracer = trace.get_tracer("vetinari", "0.4.0")
    logger.info("OpenTelemetry tracing enabled")
except ImportError:
    logger.debug("OpenTelemetry not installed — using no-op tracer")


# ── No-op span for when OTel is not available ─────────────────────────


@dataclass
class NoOpSpan:
    """Lightweight span that logs to structured logging."""

    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None
    status: str = "OK"
    events: list[dict[str, Any]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute.

        Args:
            key: The key.
            value: The value.
        """
        self.attributes[key] = value

    def set_status(self, status: str, description: str = "") -> None:
        """Set status.

        Args:
            status: The status.
            description: The description.
        """
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event.

        Args:
            name: The name.
            attributes: The attributes.
        """
        self.events.append(
            {
                "name": name,
                "timestamp": time.monotonic(),
                "attributes": attributes or {},
            }
        )

    def end(self) -> None:
        """End for the current context."""
        self.end_time = time.monotonic()
        duration_ms = (self.end_time - self.start_time) * 1000
        logger.debug(
            "Span[%s] %s duration=%.1fms status=%s attrs=%s",
            self.span_id[:8],
            self.name,
            duration_ms,
            self.status,
            {k: v for k, v in self.attributes.items() if k != "status_description"},
        )


# ── Span context manager ─────────────────────────────────────────────


@contextmanager
def start_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    parent: Any | None = None,
) -> Generator:
    """Start a tracing span.

    Uses OpenTelemetry if available, otherwise uses NoOpSpan with logging.

    Usage::

        with start_span("agent.execute", {"agent_type": "BUILDER"}) as span:
            span.set_attribute("model_id", "qwen-32b")
            result = agent.execute(task)
            span.set_attribute("success", result.success)

    Args:
        name: The name.
        attributes: The attributes.
        parent: The parent.

    Raises:
        Exception: Re-raises any exception raised inside the span body after recording it on the span.
    """
    if _OTEL_AVAILABLE and _tracer:
        with _tracer.start_as_current_span(name, attributes=attributes) as otel_span:
            if attributes:
                for k, v in attributes.items():
                    otel_span.set_attribute(k, v)
            try:
                yield otel_span
            except Exception as e:
                otel_span.set_status(StatusCode.ERROR, str(e))
                otel_span.record_exception(e)
                raise
    else:
        span = NoOpSpan(name=name, attributes=attributes or {})
        try:
            yield span
        except Exception as e:
            span.set_status("ERROR", str(e))
            raise
        finally:
            span.end()


# ── Convenience functions for common span types ───────────────────────


@contextmanager
def pipeline_span(goal: str, plan_id: str = "") -> Generator:
    """Top-level pipeline span.

    Args:
        goal: The goal.
        plan_id: The plan id.
    """
    with start_span(
        "vetinari.pipeline",
        {
            "goal": goal[:200],
            "plan_id": plan_id,
        },
    ) as span:
        yield span


@contextmanager
def stage_span(stage_name: str, stage_number: int = 0) -> Generator:
    """Pipeline stage span.

    Args:
        stage_name: The stage name.
        stage_number: The stage number.
    """
    with start_span(
        f"vetinari.stage.{stage_name}",
        {
            "stage.name": stage_name,
            "stage.number": stage_number,
        },
    ) as span:
        yield span


@contextmanager
def agent_span(
    agent_type: str,
    mode: str = "",
    task_id: str = "",
) -> Generator:
    """Agent execution span.

    Args:
        agent_type: The agent type.
        mode: The mode.
        task_id: The task id.
    """
    with start_span(
        "vetinari.agent.execute",
        {
            "agent.type": agent_type,
            "agent.mode": mode,
            "task.id": task_id,
        },
    ) as span:
        yield span


@contextmanager
def llm_span(
    model_id: str,
    agent_type: str = "",
    max_tokens: int = 0,
) -> Generator:
    """LLM inference call span.

    Args:
        model_id: The model id.
        agent_type: The agent type.
        max_tokens: The max tokens.
    """
    with start_span(
        "vetinari.llm.infer",
        {
            "llm.model_id": model_id,
            "llm.max_tokens": max_tokens,
            "agent.type": agent_type,
        },
    ) as span:
        yield span


def is_otel_available() -> bool:
    """Check if OpenTelemetry is available."""
    return _OTEL_AVAILABLE
