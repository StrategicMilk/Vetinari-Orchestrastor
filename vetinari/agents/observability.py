"""Observability helpers for Vetinari agents.

Provides the _ObservabilitySpan context manager used by the inference mixin
to instrument LLM calls with OpenTelemetry spans when available.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class _ObservabilitySpan:
    """Context manager for observability spans. No-op when OpenLLMetry not installed."""

    def __init__(self, operation: str, metadata: dict[str, Any] | None = None) -> None:
        """Initialise the span context manager.

        Args:
            operation: Name of the operation to record as a span.
            metadata: Optional key/value attributes attached to the span at creation.
        """
        self._operation = operation
        self._metadata = metadata or {}  # noqa: VET112 - empty fallback preserves optional request metadata contract
        self._span = None
        self._start_time: float = 0.0

    def __enter__(self) -> _ObservabilitySpan:
        """Start the span and record the wall-clock start time.

        Returns:
            Self, so callers can use ``as span`` to call ``set_attribute``.
        """
        self._start_time = time.time()
        try:
            from opentelemetry import trace

            tracer = trace.get_tracer("vetinari.agents")
            self._span = tracer.start_span(self._operation)
            for key, value in self._metadata.items():
                self._span.set_attribute(key, str(value))
        except ImportError:
            logger.debug("OpenLLMetry not installed — observability disabled")
        except Exception:
            logger.warning("Observability span creation failed", exc_info=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the span, recording elapsed duration and any exception info.

        Args:
            exc_type: Exception class, or None if no exception was raised.
            exc_val: Exception instance, or None.
            exc_tb: Traceback object, or None.
        """
        duration_ms = (time.time() - self._start_time) * 1000
        if self._span is not None:
            try:
                self._span.set_attribute("duration_ms", duration_ms)
                if exc_type is not None:
                    self._span.set_attribute("error", True)
                    self._span.set_attribute("error.type", exc_type.__name__)
                self._span.end()
            except Exception:
                logger.warning("Observability span end failed", exc_info=True)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the active span.

        No-op when no span is active (e.g. OpenLLMetry not installed).

        Args:
            key: Attribute key.
            value: Attribute value; converted to str before storage.
        """
        if self._span is not None:
            try:
                self._span.set_attribute(key, str(value))
            except Exception:
                logger.warning("Failed to set span attribute %s", key)
