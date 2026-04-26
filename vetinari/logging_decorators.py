"""Decorators for operation timing and trace correlation.

Provides :func:`timed_operation`, :func:`traced_operation`, and their async
variants for automatic instrumentation of function calls with structured
logging. These decorators handle timing, exception logging, and trace ID
binding automatically.

Usage::

    from vetinari.logging_decorators import timed_operation, traced_operation

    @timed_operation("model_discovery")
    def discover_models():
        ...

    @traced_operation("plan_execution")
    def execute_plan(plan_id):
        logger.info("Step 1")  # trace_id automatically included
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from typing import Any


def get_logger(name: str) -> Any:
    """Lazy import to avoid circular imports.

    Returns:
        StructuredLogger instance for the given name.
    """
    from vetinari.structured_logging import get_logger as _get_logger

    return _get_logger(name)


def get_trace_id() -> str | None:
    """Lazy import to avoid circular imports.

    Returns:
        The current trace ID or None if not set.
    """
    from vetinari.logging_context import get_trace_id as _get_trace_id

    return _get_trace_id()


def CorrelationContext(*args: Any, **kwargs: Any) -> Any:
    """Lazy import to avoid circular imports.

    Returns:
        CorrelationContext manager for distributed tracing.
    """
    from vetinari.logging_context import CorrelationContext as _CorrelationContext

    return _CorrelationContext(*args, **kwargs)


def timed_operation(operation_name: str) -> Callable:
    """Decorator to log timing for operations.

    Usage::

        @timed_operation("model_discovery")
        def discover_models():
            ...

    Args:
        operation_name: Name of the operation to time.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        """Wrap function with operation logging.

        Returns:
            The wrapped function with timing instrumentation.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute function with timing and logging.

            Returns:
                The original function's return value.

            Raises:
                Exception: Re-raises any exception from the wrapped function
                    after logging the failure.
            """
            start_time = datetime.now(timezone.utc)
            log = get_logger(func.__module__)

            try:
                result = func(*args, **kwargs)
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                log.info(
                    "Operation completed: %s",
                    operation_name,
                    event_type="operation_complete",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    success=True,
                )
                return result
            except Exception as e:
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                log.error(
                    "Operation failed: %s",
                    operation_name,
                    event_type="operation_error",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    error=str(e),
                    success=False,
                )
                raise

        return wrapper

    return decorator


def traced_operation(operation_name: str, generate_trace_id: bool = True) -> Callable:
    """Decorator to log operations with automatic trace ID assignment.

    Usage::

        @traced_operation("plan_execution")
        def execute_plan(plan_id):
            logger.info("Step 1")  # trace_id automatically included

    Args:
        operation_name: Name of the operation.
        generate_trace_id: Whether to generate a new trace ID.

    Returns:
        Decorator function.
    """
    import uuid

    def decorator(func: Callable) -> Callable:
        """Wrap function with trace correlation.

        Returns:
            The wrapped function with trace context instrumentation.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute function within correlation context.

            Returns:
                The original function's return value.

            Raises:
                Exception: Re-raises any exception from the wrapped function
                    after logging the failure.
            """
            trace_id = str(uuid.uuid4()) if generate_trace_id else get_trace_id()

            with CorrelationContext(trace_id=trace_id):
                start_time = datetime.now(timezone.utc)
                log = get_logger(func.__module__)

                try:
                    log.info(
                        "Operation started: %s",
                        operation_name,
                        event_type="operation_start",
                        operation=operation_name,
                    )
                    result = func(*args, **kwargs)
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    log.info(
                        "Operation completed: %s",
                        operation_name,
                        event_type="operation_complete",
                        operation=operation_name,
                        duration_ms=duration_ms,
                        success=True,
                    )
                    return result
                except Exception as e:
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    log.error(
                        "Operation failed: %s",
                        operation_name,
                        event_type="operation_error",
                        operation=operation_name,
                        duration_ms=duration_ms,
                        error=str(e),
                        success=False,
                    )
                    raise

        return wrapper

    return decorator


def async_timed_operation(operation_name: str) -> Callable:
    """Decorator to log timing for async operations.

    Async-compatible variant of :func:`timed_operation`.  Uses ``await`` to
    call the wrapped coroutine so the event loop is not blocked.

    Usage::

        @async_timed_operation("inference")
        async def infer(prompt):
            ...

    Args:
        operation_name: Name of the operation to time.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        """Wrap async function with operation logging.

        Returns:
            The wrapped async function with timing instrumentation.
        """

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute async function with timing and logging.

            Returns:
                The original function's return value.

            Raises:
                Exception: Re-raises any exception from the wrapped function
                    after logging the failure.
            """
            start_time = datetime.now(timezone.utc)
            log = get_logger(func.__module__)

            try:
                result = await func(*args, **kwargs)
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                log.info(
                    "Operation completed: %s",
                    operation_name,
                    event_type="operation_complete",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    success=True,
                )
                return result
            except Exception as e:
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                log.error(
                    "Operation failed: %s",
                    operation_name,
                    event_type="operation_error",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    error=str(e),
                    success=False,
                )
                raise

        return wrapper

    return decorator


def async_traced_operation(operation_name: str, generate_trace_id: bool = True) -> Callable:
    """Decorator to log async operations with automatic trace ID assignment.

    Async-compatible variant of :func:`traced_operation`.

    Usage::

        @async_traced_operation("plan_execution")
        async def execute_plan(plan_id):
            logger.info("Step 1")  # trace_id automatically included

    Args:
        operation_name: Name of the operation.
        generate_trace_id: Whether to generate a new trace ID.

    Returns:
        Decorator function.
    """
    import uuid

    def decorator(func: Callable) -> Callable:
        """Wrap async function with trace correlation.

        Returns:
            The wrapped async function with trace context instrumentation.
        """

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute async function within correlation context.

            Returns:
                The original function's return value.

            Raises:
                Exception: Re-raises any exception from the wrapped function
                    after logging the failure.
            """
            trace_id = str(uuid.uuid4()) if generate_trace_id else get_trace_id()

            with CorrelationContext(trace_id=trace_id):
                start_time = datetime.now(timezone.utc)
                log = get_logger(func.__module__)

                try:
                    log.info(
                        "Operation started: %s",
                        operation_name,
                        event_type="operation_start",
                        operation=operation_name,
                    )
                    result = await func(*args, **kwargs)
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    log.info(
                        "Operation completed: %s",
                        operation_name,
                        event_type="operation_complete",
                        operation=operation_name,
                        duration_ms=duration_ms,
                        success=True,
                    )
                    return result
                except Exception as e:
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    log.error(
                        "Operation failed: %s",
                        operation_name,
                        event_type="operation_error",
                        operation=operation_name,
                        duration_ms=duration_ms,
                        error=str(e),
                        success=False,
                    )
                    raise

        return wrapper

    return decorator
