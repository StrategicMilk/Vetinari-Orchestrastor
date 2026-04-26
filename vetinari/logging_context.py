"""Distributed tracing correlation context for structured logging.

Provides contextvars-based trace/span/request/plan ID binding via the
CorrelationContext manager and related helper functions. All bindings are
task-local (async-safe) and do not leak across concurrent tasks.

Usage::

    from vetinari.logging_context import CorrelationContext, get_trace_id

    with CorrelationContext() as ctx:
        logger.info("Executing task")  # trace_id auto-included in logs
        ctx.set_span_id("span_123")
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any

import structlog

# ── Context Variables for Distributed Tracing ─────────────────────────────

_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id_var: ContextVar[str | None] = ContextVar("span_id", default=None)
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_plan_id_var: ContextVar[str | None] = ContextVar("plan_id", default=None)


class CorrelationContext:
    """Context manager for distributed tracing correlation.

    Binds trace/span/request/plan IDs to both stdlib ContextVars and
    structlog's contextvars for automatic inclusion in all log output.

    Usage::

        with CorrelationContext() as ctx:
            logger.info("Starting task")  # trace_id automatically included
            ctx.set_span_id("span_123")
    """

    def __init__(
        self,
        trace_id: str | None = None,
        span_id: str | None = None,
        request_id: str | None = None,
        plan_id: str | None = None,
    ):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id
        self.request_id = request_id
        self.plan_id = plan_id
        self._tokens: list[Any] = []

    def __enter__(self) -> CorrelationContext:
        """Enter context and bind correlation IDs."""
        self._bound_keys: list[str] = []

        token = _trace_id_var.set(self.trace_id)
        self._tokens.append(token)
        self._bound_keys.append("trace_id")

        if self.span_id:
            token = _span_id_var.set(self.span_id)
            self._tokens.append(token)
            self._bound_keys.append("span_id")

        if self.request_id:
            token = _request_id_var.set(self.request_id)
            self._tokens.append(token)
            self._bound_keys.append("request_id")

        if self.plan_id:
            token = _plan_id_var.set(self.plan_id)
            self._tokens.append(token)
            self._bound_keys.append("plan_id")

        # Bind to structlog contextvars
        bindings = {"trace_id": self.trace_id}
        if self.span_id:
            bindings["span_id"] = self.span_id
        if self.request_id:
            bindings["request_id"] = self.request_id
        if self.plan_id:
            bindings["plan_id"] = self.plan_id
        structlog.contextvars.bind_contextvars(**bindings)

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Exit context and reset only the correlation IDs that were bound.

        After resetting ContextVar tokens the outer values are correct, but
        structlog's context still holds the inner values.  We therefore
        re-bind any key whose outer ContextVar is non-None (restoring the
        outer structlog context), then unbind the remainder (keys that had no
        outer value) so structlog stays in sync with the restored ContextVars.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Map bound-key name → its ContextVar so we can read restored values.
        _key_to_var: dict[str, ContextVar[str | None]] = {
            "trace_id": _trace_id_var,
            "span_id": _span_id_var,
            "request_id": _request_id_var,
            "plan_id": _plan_id_var,
        }

        for token in reversed(self._tokens):
            try:
                token.var.reset(token)
            except Exception:
                logger.warning("Failed to reset ContextVar token during context exit", exc_info=True)

        # After resetting all tokens, re-sync structlog with the now-restored
        # ContextVar values.  Keys that have a non-None outer value are
        # re-bound; keys with None (no outer context) are unbound so they
        # disappear from log output correctly.
        if self._bound_keys:
            rebind: dict[str, str] = {}
            unbind_keys: list[str] = []
            for key in self._bound_keys:
                var = _key_to_var.get(key)
                outer_value = var.get() if var is not None else None
                if outer_value is not None:
                    rebind[key] = outer_value
                else:
                    unbind_keys.append(key)
            if rebind:
                structlog.contextvars.bind_contextvars(**rebind)
            if unbind_keys:
                structlog.contextvars.unbind_contextvars(*unbind_keys)

    def set_span_id(self, span_id: str) -> None:
        """Set or update the span ID within this context."""
        self.span_id = span_id
        token = _span_id_var.set(span_id)
        self._tokens.append(token)
        if "span_id" not in self._bound_keys:
            self._bound_keys.append("span_id")
        structlog.contextvars.bind_contextvars(span_id=span_id)

    def set_request_id(self, request_id: str) -> None:
        """Set or update the request ID within this context."""
        self.request_id = request_id
        token = _request_id_var.set(request_id)
        self._tokens.append(token)
        if "request_id" not in self._bound_keys:
            self._bound_keys.append("request_id")
        structlog.contextvars.bind_contextvars(request_id=request_id)

    def set_plan_id(self, plan_id: str) -> None:
        """Set or update the plan ID within this context."""
        self.plan_id = plan_id
        token = _plan_id_var.set(plan_id)
        self._tokens.append(token)
        if "plan_id" not in self._bound_keys:
            self._bound_keys.append("plan_id")
        structlog.contextvars.bind_contextvars(plan_id=plan_id)


def get_trace_id() -> str | None:
    """Read the trace ID bound to the current async task or thread via contextvars."""
    return _trace_id_var.get()


def get_span_id() -> str | None:
    """Read the span ID bound to the current async task or thread via contextvars."""
    return _span_id_var.get()


def get_request_id() -> str | None:
    """Read the request ID bound to the current async task or thread via contextvars."""
    return _request_id_var.get()


def set_request_id(request_id: str) -> Any:
    """Bind a request ID to the current async task or thread via contextvars.

    Intended for use in request middleware where a full ``CorrelationContext``
    context manager is not in scope.  The binding is task-local and does not
    affect other concurrent requests.

    Returns the ``ContextVar`` token so callers can restore the previous value
    with :func:`clear_request_id` — preventing request ID leakage across
    middleware boundaries.

    Args:
        request_id: The request identifier to bind (typically a UUID or the
            value of the ``X-Request-ID`` HTTP header).

    Returns:
        Token that can be passed to :func:`clear_request_id` to restore the
        previous request ID (or ``None`` if there was none).
    """
    token = _request_id_var.set(request_id)
    structlog.contextvars.bind_contextvars(request_id=request_id)
    return token


def clear_request_id(token: Any) -> None:
    """Restore the request ID that existed before the paired :func:`set_request_id` call.

    Resets both the ``ContextVar`` and structlog's context so no stale request
    ID leaks into unrelated log lines emitted after this call.

    Args:
        token: The token returned by :func:`set_request_id`.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        _request_id_var.reset(token)
    except Exception:
        logger.warning(
            "Failed to reset request_id ContextVar token — request ID may persist in logs",
            exc_info=True,
        )
    outer = _request_id_var.get()
    if outer is not None:
        structlog.contextvars.bind_contextvars(request_id=outer)
    else:
        structlog.contextvars.unbind_contextvars("request_id")


def get_plan_id() -> str | None:
    """Read the plan ID bound to the current async task or thread via contextvars."""
    return _plan_id_var.get()
