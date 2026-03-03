"""
Structured JSON Logging Module for Vetinari

Provides structured JSON logging with:
- JSON output format
- Configurable log levels
- Context fields (execution_id, task_id, wave_id, plan_id)
- Metric sampling
- Backward compatibility with text logging

Usage:
    from vetinari.structured_logging import get_logger, log_event
    
    logger = get_logger("orchestrator")
    logger.info("Task started", extra={"task_id": "task_1", "execution_id": "exec_abc"})
    
    # Or use the log_event convenience function
    log_event("info", "orchestrator", "Task completed", task_id="task_1")
"""

import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable
from functools import wraps
from enum import Enum
import threading
from contextvars import ContextVar

# Context variables for distributed tracing
_trace_id_var: ContextVar[str] = ContextVar('trace_id', default=None)
_span_id_var: ContextVar[str] = ContextVar('span_id', default=None)
_request_id_var: ContextVar[str] = ContextVar('request_id', default=None)


class CorrelationContext:
    """
    Context manager for distributed tracing correlation.
    
    Automatically generates trace_id and propagates it through async/concurrent calls.
    
    Usage:
        with CorrelationContext() as ctx:
            logger.info("Starting task")  # trace_id automatically included
            ctx.set_span_id("span_123")
            logger.info("In span")  # trace_id and span_id included
    """
    
    def __init__(self, trace_id: Optional[str] = None, span_id: Optional[str] = None, 
                 request_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id
        self.request_id = request_id
        self._tokens = []
    
    def __enter__(self):
        """Enter context and set correlation IDs."""
        token = _trace_id_var.set(self.trace_id)
        self._tokens.append(token)
        
        if self.span_id:
            token = _span_id_var.set(self.span_id)
            self._tokens.append(token)
        
        if self.request_id:
            token = _request_id_var.set(self.request_id)
            self._tokens.append(token)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and reset correlation IDs."""
        for token in reversed(self._tokens):
            _trace_id_var.set(None)
            _span_id_var.set(None)
            _request_id_var.set(None)
    
    def set_span_id(self, span_id: str):
        """Set or update the span ID within this context."""
        self.span_id = span_id
        _span_id_var.set(span_id)
    
    def set_request_id(self, request_id: str):
        """Set or update the request ID within this context."""
        self.request_id = request_id
        _request_id_var.set(request_id)


def get_trace_id() -> Optional[str]:
    """Get the current trace ID from context."""
    return _trace_id_var.get()


def get_span_id() -> Optional[str]:
    """Get the current span ID from context."""
    return _span_id_var.get()


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return _request_id_var.get()


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    Outputs log records as JSON with consistent field schema.
    """
    
    def __init__(self, include_extra: bool = True, include_context: bool = True):
        super().__init__()
        self.include_extra = include_extra
        self.include_context = include_context
        
        # Default context fields
        self._context = {
            "service": "vetinari",
            "version": os.environ.get("VETINARI_VERSION", "unknown")
        }
    
    def set_context(self, **kwargs):
        """Set global context fields that will be included in all logs."""
        self._context.update(kwargs)
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Build the log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add thread and process info
        if threading.current_thread().name != "MainThread":
            log_entry["thread"] = threading.current_thread().name
        log_entry["process_id"] = record.process
        
        # Add distributed tracing context
        trace_id = get_trace_id()
        span_id = get_span_id()
        request_id = get_request_id()
        
        if trace_id:
            log_entry["trace_id"] = trace_id
        if span_id:
            log_entry["span_id"] = span_id
        if request_id:
            log_entry["request_id"] = request_id
        
        # Add context (global + record-specific)
        if self.include_context:
            log_entry["context"] = dict(self._context)
            # Add execution context if available
            execution_id = getattr(record, "execution_id", None)
            task_id = getattr(record, "task_id", None)
            wave_id = getattr(record, "wave_id", None)
            plan_id = getattr(record, "plan_id", None)
            
            if execution_id:
                log_entry["context"]["execution_id"] = execution_id
            if task_id:
                log_entry["context"]["task_id"] = task_id
            if wave_id:
                log_entry["context"]["wave_id"] = wave_id
            if plan_id:
                log_entry["context"]["plan_id"] = plan_id
        
        # Add extra fields
        if self.include_extra and hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add execution time if available
        if hasattr(record, "execution_time_ms"):
            log_entry["execution_time_ms"] = record.execution_time_ms
        
        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter for development.
    """
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


class StructuredLogger:
    """
    Convenience wrapper around Python logging with structured logging support.
    """
    
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger
        self._context = {}
    
    def set_context(self, **kwargs):
        """Set context for this logger."""
        self._context.update(kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with additional context fields."""
        extra = {"extra_fields": dict(self._context)}
        extra["extra_fields"].update(kwargs)
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with full traceback."""
        self._log_with_context(logging.ERROR, message, **kwargs)


# Global state
_loggers: Dict[str, StructuredLogger] = {}
_configured = False
_log_level = logging.INFO
_use_json = True


def _get_log_level() -> int:
    """Get log level from environment variable."""
    level_str = os.environ.get("VETINARI_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_str, logging.INFO)


def _use_structured_logging() -> bool:
    """Check if structured JSON logging is enabled."""
    return os.environ.get("VETINARI_STRUCTURED_LOGGING", "true").lower() in ("1", "true", "yes")


def configure_logging():
    """Configure logging for the application."""
    global _configured, _log_level, _use_json
    
    if _configured:
        return
    
    _log_level = _get_log_level()
    _use_json = _use_structured_logging()
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add handler based on configuration
    handler = logging.StreamHandler(sys.stdout)
    
    if _use_json:
        formatter = StructuredFormatter()
    else:
        formatter = TextFormatter()
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    _configured = True


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically module name)
    
    Returns:
        StructuredLogger instance
    """
    global _loggers
    
    if not _configured:
        configure_logging()
    
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = StructuredLogger(name, logger)
    
    return _loggers[name]


def log_event(
    level: str,
    logger_name: str,
    message: str,
    **context
):
    """
    Convenience function to log a structured event.
    
    Args:
        level: Log level (debug, info, warning, error, critical)
        logger_name: Name of the logger
        message: Log message
        **context: Additional context fields
    """
    logger = get_logger(logger_name)
    level_func = getattr(logger, level.lower(), logger.info)
    level_func(message, **context)


def log_task_start(task_id: str, task_type: str = "generic", **kwargs):
    """Log task start event."""
    logger = get_logger("executor")
    logger.info(
        f"Task started: {task_id}",
        event_type="task_start",
        task_id=task_id,
        task_type=task_type,
        **kwargs
    )


def log_task_complete(task_id: str, execution_time_ms: float, **kwargs):
    """Log task completion event."""
    logger = get_logger("executor")
    logger.info(
        f"Task completed: {task_id}",
        event_type="task_complete",
        task_id=task_id,
        execution_time_ms=execution_time_ms,
        **kwargs
    )


def log_task_error(task_id: str, error: str, **kwargs):
    """Log task error event."""
    logger = get_logger("executor")
    logger.error(
        f"Task failed: {task_id}",
        event_type="task_error",
        task_id=task_id,
        error=error,
        **kwargs
    )


def log_model_discovery(models_found: int, duration_ms: float, **kwargs):
    """Log model discovery event."""
    logger = get_logger("model_pool")
    logger.info(
        f"Model discovery completed: {models_found} models",
        event_type="model_discovery",
        models_found=models_found,
        duration_ms=duration_ms,
        **kwargs
    )


def log_wave_start(wave_id: str, task_count: int, **kwargs):
    """Log wave start event."""
    logger = get_logger("scheduler")
    logger.info(
        f"Wave started: {wave_id}",
        event_type="wave_start",
        wave_id=wave_id,
        task_count=task_count,
        **kwargs
    )


def log_wave_complete(wave_id: str, duration_ms: float, **kwargs):
    """Log wave completion event."""
    logger = get_logger("scheduler")
    logger.info(
        f"Wave completed: {wave_id}",
        event_type="wave_complete",
        wave_id=wave_id,
        duration_ms=duration_ms,
        **kwargs
    )


def log_api_request(endpoint: str, method: str, status_code: int, duration_ms: float, **kwargs):
    """Log API request event."""
    logger = get_logger("web_ui")
    logger.info(
        f"API request: {method} {endpoint}",
        event_type="api_request",
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs
    )


def log_sandbox_execution(execution_id: str, success: bool, duration_ms: float, memory_mb: float, **kwargs):
    """Log sandbox execution event."""
    logger = get_logger("sandbox")
    level = logger.info if success else logger.error
    level(
        f"Sandbox execution {'success' if success else 'failed'}: {execution_id}",
        event_type="sandbox_execution",
        execution_id=execution_id,
        success=success,
        duration_ms=duration_ms,
        memory_mb=memory_mb,
        **kwargs
    )


def timed_operation(operation_name: str):
    """
    Decorator to log timing for operations.
    
    Usage:
        @timed_operation("model_discovery")
        def discover_models():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now(timezone.utc)
            logger = get_logger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.info(
                    f"Operation completed: {operation_name}",
                    event_type="operation_complete",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    success=True
                )
                return result
            except Exception as e:
                duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                
                logger.error(
                    f"Operation failed: {operation_name}",
                    event_type="operation_error",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    error=str(e),
                    success=False
                )
                raise
        
        return wrapper
    return decorator


def traced_operation(operation_name: str, generate_trace_id: bool = True):
    """
    Decorator to log operations with automatic trace ID assignment.
    
    Usage:
        @traced_operation("plan_execution")
        def execute_plan(plan_id):
            logger.info("Step 1")  # trace_id automatically included
            ...
    
    Args:
        operation_name: Name of the operation
        generate_trace_id: Whether to generate a new trace ID for this operation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4()) if generate_trace_id else get_trace_id()
            
            with CorrelationContext(trace_id=trace_id):
                start_time = datetime.now(timezone.utc)
                logger = get_logger(func.__module__)
                
                try:
                    logger.info(
                        f"Operation started: {operation_name}",
                        event_type="operation_start",
                        operation=operation_name
                    )
                    
                    result = func(*args, **kwargs)
                    
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    logger.info(
                        f"Operation completed: {operation_name}",
                        event_type="operation_complete",
                        operation=operation_name,
                        duration_ms=duration_ms,
                        success=True
                    )
                    return result
                except Exception as e:
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    
                    logger.error(
                        f"Operation failed: {operation_name}",
                        event_type="operation_error",
                        operation=operation_name,
                        duration_ms=duration_ms,
                        error=str(e),
                        success=False
                    )
                    raise
        
        return wrapper
    return decorator



class MetricsCollector:
    """
    Simple metrics collector for observability.
    Collects counters and histograms for monitoring.
    """
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, list] = {}
        self._lock = threading.Lock()
    
    def increment(self, metric_name: str, value: int = 1, **tags):
        """Increment a counter metric."""
        key = self._make_key(metric_name, tags)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
    
    def record(self, metric_name: str, value: float, **tags):
        """Record a histogram value."""
        key = self._make_key(metric_name, tags)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
    
    def get_counter(self, metric_name: str, **tags) -> int:
        """Get counter value."""
        key = self._make_key(metric_name, tags)
        with self._lock:
            return self._counters.get(key, 0)
    
    def get_histogram_stats(self, metric_name: str, **tags) -> Optional[Dict[str, float]]:
        """Get histogram statistics."""
        key = self._make_key(metric_name, tags)
        with self._lock:
            values = self._histograms.get(key, [])
            if not values:
                return None
            
            sorted_values = sorted(values)
            return {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "p50": sorted_values[len(sorted_values) // 2],
                "p95": sorted_values[int(len(sorted_values) * 0.95)],
                "p99": sorted_values[int(len(sorted_values) * 0.99)]
            }
    
    def _make_key(self, metric_name: str, tags: Dict[str, str]) -> str:
        """Create a metric key from name and tags."""
        if not tags:
            return metric_name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}{{{tag_str}}}"


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics


# Convenience metric functions
def record_task_duration(task_id: str, duration_ms: float):
    """Record task execution duration."""
    _metrics.record("vetinari.task.duration", duration_ms, task_type="generic")


def increment_task_count(status: str):
    """Increment task count by status."""
    _metrics.increment("vetinari.task.count", status=status)


def record_model_latency(duration_ms: float):
    """Record model inference latency."""
    _metrics.record("vetinari.model.latency", duration_ms)


def increment_api_request(status_code: int):
    """Increment API request counter."""
    _metrics.increment("vetinari.api.request", status=status_code)
