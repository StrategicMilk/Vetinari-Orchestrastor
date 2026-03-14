"""Log Aggregation Integration for Vetinari Dashboard  (Phase 4 Step 4).

Provides a unified interface for exporting structured log records to multiple
centralised logging backends, plus a lightweight in-process log store that the
dashboard can query without any external dependency.

Supported backends
------------------
  file          Write newline-delimited JSON to a local file.
  elasticsearch Send bulk POST requests to an Elasticsearch index.
  splunk        Send events via the Splunk HTTP Event Collector (HEC).
  datadog       Send log entries via the Datadog Logs Intake API.

All network backends degrade gracefully: if the dependency (``requests``) is
missing, or the remote endpoint is unreachable, the error is logged and the
call returns False rather than raising.

Usage
-----
    from vetinari.dashboard.log_aggregator import get_log_aggregator, LogRecord

    agg = get_log_aggregator()
    agg.configure_backend("file", path="logs/audit.jsonl")
    agg.ingest(LogRecord(
        message="Plan approved",
        level="INFO",
        trace_id="abc-123",
        extra={"plan_id": "plan_001", "risk_score": 0.12},
    ))

    # Search in-process store
    results = agg.search(trace_id="abc-123", limit=10)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log record
# ---------------------------------------------------------------------------


@dataclass
class LogRecord:
    """A structured log record ready for aggregation."""

    message: str
    level: str = "INFO"  # DEBUG / INFO / WARNING / ERROR / CRITICAL
    timestamp: float = field(default_factory=time.time)
    trace_id: str | None = None
    span_id: str | None = None
    request_id: str | None = None
    logger_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "level": self.level,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "request_id": self.request_id,
            "logger_name": self.logger_name,
            **self.extra,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class BackendBase:
    """Abstract base for log aggregation backends."""

    name: str = "base"

    def configure(self, **kwargs: Any) -> None:
        """Apply backend-specific configuration."""

    def send(self, records: list[LogRecord]) -> bool:
        """Send a batch of records. Returns True on success.

        Subclasses must override this method.

        Raises:
            NotImplementedError: If the operation fails.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement send()")  # noqa: VET033

    def close(self) -> None:
        """Release any resources held by this backend."""


# ---------------------------------------------------------------------------
# File backend
# ---------------------------------------------------------------------------


class FileBackend(BackendBase):
    """Appends newline-delimited JSON to a local file."""

    name = "file"

    def __init__(self) -> None:
        self._path: str | None = None
        self._lock = threading.Lock()

    def configure(self, path: str = "logs/vetinari_audit.jsonl", **_: Any) -> None:
        """Configure."""
        self._path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def send(self, records: list[LogRecord]) -> bool:
        """Send.

        Returns:
            True if successful, False otherwise.
        """
        if not self._path:
            logger.warning("FileBackend not configured (no path set).")
            return False
        try:
            with self._lock, open(self._path, "a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(rec.to_json() + "\n")
            return True
        except OSError as exc:
            logger.error("FileBackend.send failed: %s", exc)
            return False

    def close(self) -> None:
        """Close for the current context."""
        # noqa: VET031  (intentional: file opened/closed per send call — nothing to release)


# ---------------------------------------------------------------------------
# Elasticsearch backend
# ---------------------------------------------------------------------------


class ElasticsearchBackend(BackendBase):
    """Sends records via the Elasticsearch Bulk API."""

    name = "elasticsearch"

    def __init__(self) -> None:
        self._url: str | None = None
        self._index: str = "vetinari-logs"
        self._headers: dict[str, str] = {"Content-Type": "application/x-ndjson"}

    def configure(
        self,
        url: str = "http://localhost:9200",  # noqa: VET041
        index: str = "vetinari-logs",
        api_key: str | None = None,
        **_: Any,
    ) -> None:
        """Configure.

        Args:
            url: The url.
            index: The index.
            api_key: The api key.
        """
        self._url = url.rstrip("/")
        self._index = index
        if api_key:
            self._headers["Authorization"] = f"ApiKey {api_key}"

    def send(self, records: list[LogRecord]) -> bool:
        """Send.

        Returns:
            True if successful, False otherwise.
        """
        if not self._url:
            logger.warning("ElasticsearchBackend not configured.")
            return False
        try:
            import requests  # optional dependency
        except ImportError:
            logger.error("'requests' package is required for ElasticsearchBackend.")
            return False

        # Build NDJSON bulk body
        lines: list[str] = []
        for rec in records:
            action = json.dumps({"index": {"_index": self._index}})
            lines.append(action)
            lines.append(rec.to_json())
        body = "\n".join(lines) + "\n"

        try:
            resp = requests.post(
                f"{self._url}/_bulk",
                data=body,
                headers=self._headers,
                timeout=10,
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    "ElasticsearchBackend received HTTP %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            return True
        except Exception as exc:
            logger.error("ElasticsearchBackend.send error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Splunk HEC backend
# ---------------------------------------------------------------------------


class SplunkBackend(BackendBase):
    """Sends records via Splunk HTTP Event Collector."""

    name = "splunk"

    def __init__(self) -> None:
        self._url: str | None = None
        self._token: str | None = None
        self._source: str = "vetinari"
        self._sourcetype: str = "_json"

    def configure(
        self,
        url: str = "http://localhost:8088",  # noqa: VET041
        token: str = "",
        source: str = "vetinari",
        sourcetype: str = "_json",
        **_: Any,
    ) -> None:
        """Configure.

        Args:
            url: The url.
            token: The token.
            source: The source.
            sourcetype: The sourcetype.
        """
        self._url = url.rstrip("/")
        self._token = token
        self._source = source
        self._sourcetype = sourcetype

    def send(self, records: list[LogRecord]) -> bool:
        """Send.

        Returns:
            True if successful, False otherwise.
        """
        if not self._url or not self._token:
            logger.warning("SplunkBackend not configured (url/token missing).")
            return False
        try:
            import requests
        except ImportError:
            logger.error("'requests' package is required for SplunkBackend.")
            return False

        headers = {
            "Authorization": f"Splunk {self._token}",
            "Content-Type": "application/json",
        }
        # Each record is a separate HEC event
        body = "".join(
            json.dumps(
                {
                    "time": rec.timestamp,
                    "source": self._source,
                    "sourcetype": self._sourcetype,
                    "event": rec.to_dict(),
                }
            )
            for rec in records
        )
        try:
            resp = requests.post(
                f"{self._url}/services/collector/event",
                data=body,
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                logger.error(
                    "SplunkBackend received HTTP %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            return True
        except Exception as exc:
            logger.error("SplunkBackend.send error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Datadog backend
# ---------------------------------------------------------------------------


class DatadogBackend(BackendBase):
    """Sends records via the Datadog Logs Intake API."""

    name = "datadog"
    _DD_URL = "https://http-intake.logs.datadoghq.com/api/v2/logs"

    def __init__(self) -> None:
        self._api_key: str | None = None
        self._service: str = "vetinari"
        self._ddsource: str = "python"
        self._ddtags: str = ""

    def configure(
        self,
        api_key: str = "",
        service: str = "vetinari",
        ddsource: str = "python",
        ddtags: str = "",
        **_: Any,
    ) -> None:
        """Configure.

        Args:
            api_key: The api key.
            service: The service.
            ddsource: The ddsource.
            ddtags: The ddtags.
        """
        self._api_key = api_key
        self._service = service
        self._ddsource = ddsource
        self._ddtags = ddtags

    def send(self, records: list[LogRecord]) -> bool:
        """Send.

        Returns:
            True if successful, False otherwise.
        """
        if not self._api_key:
            logger.warning("DatadogBackend not configured (api_key missing).")
            return False
        try:
            import requests
        except ImportError:
            logger.error("'requests' package is required for DatadogBackend.")
            return False

        payload = [
            {
                "ddsource": self._ddsource,
                "ddtags": self._ddtags,
                "service": self._service,
                "message": rec.message,
                "status": rec.level,
                **rec.to_dict(),
            }
            for rec in records
        ]
        headers = {
            "DD-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self._DD_URL,
                json=payload,
                headers=headers,
                timeout=10,
            )
            if resp.status_code not in (200, 202):
                logger.error(
                    "DatadogBackend received HTTP %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            return True
        except Exception as exc:
            logger.error("DatadogBackend.send error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Webhook backend (generic HTTP POST)
# ---------------------------------------------------------------------------


class WebhookBackend(BackendBase):
    """POST aggregated logs to an external webhook URL."""

    name = "webhook"

    def __init__(self) -> None:
        self._url: str | None = None
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        self._timeout: int = 10

    def configure(self, url: str = "", headers: dict[str, str] | None = None, timeout: int = 10, **_: Any) -> None:
        """Configure.

        Args:
            url: The url.
            headers: The headers.
            timeout: The timeout.
        """
        self._url = url
        if headers:
            self._headers.update(headers)
        self._timeout = timeout

    def send(self, records: list[LogRecord]) -> bool:
        """Send.

        Returns:
            True if successful, False otherwise.
        """
        if not self._url:
            logger.warning("WebhookBackend not configured (url missing).")
            return False
        try:
            import requests
        except ImportError:
            logger.error("'requests' package is required for WebhookBackend.")
            return False

        payload = [rec.to_dict() for rec in records]
        try:
            resp = requests.post(
                self._url,
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
            if not resp.ok:
                logger.error(
                    "WebhookBackend received HTTP %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
            return True
        except Exception as exc:
            logger.error("WebhookBackend.send error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_CLASSES: dict[str, type] = {
    "file": FileBackend,
    "elasticsearch": ElasticsearchBackend,
    "splunk": SplunkBackend,
    "datadog": DatadogBackend,
    "webhook": WebhookBackend,
}


# ---------------------------------------------------------------------------
# LogAggregator — core class
# ---------------------------------------------------------------------------


class LogAggregator:
    """Central log aggregation hub.

    * Maintains an in-process circular buffer of recent records for dashboard
      search / correlation.
    * Fans out each ingested batch to all configured backends.
    * Thread-safe singleton — obtain via ``get_log_aggregator()``.
    """

    _instance: LogAggregator | None = None
    _class_lock = threading.Lock()
    MAX_BUFFER = 5_000  # maximum records kept in memory

    def __new__(cls) -> LogAggregator:
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._setup()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        self._lock = threading.RLock()
        self._buffer: deque = deque(maxlen=self.MAX_BUFFER)
        self._backends: dict[str, BackendBase] = {}
        self._batch_size: int = 100
        self._pending: list[LogRecord] = []

    # ------------------------------------------------------------------
    # Backend management
    # ------------------------------------------------------------------

    def configure_backend(self, name: str, **kwargs: Any) -> None:
        """Add and configure a named backend.

        Args:
            name: One of ``file``, ``elasticsearch``, ``splunk``, ``datadog``.
            **kwargs: Backend-specific configuration (see each class's
                      ``configure()`` docstring).

        Raises:
            ValueError: If ``name`` is not a recognised backend type.
        """
        cls = _BACKEND_CLASSES.get(name)
        if cls is None:
            raise ValueError(f"Unknown backend '{name}'. Valid options: {sorted(_BACKEND_CLASSES)}")
        with self._lock:
            backend = cls()
            backend.configure(**kwargs)
            self._backends[name] = backend
            logger.info("Configured log aggregation backend: %s", name)

    def remove_backend(self, name: str) -> bool:
        """Remove a backend by name. Returns True if it existed.

        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            b = self._backends.pop(name, None)
            if b:
                b.close()
                logger.info("Removed log aggregation backend: %s", name)
            return b is not None

    def list_backends(self) -> list[str]:
        """List backends.

        Returns:
            The result string.
        """
        with self._lock:
            return list(self._backends.keys())

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, record: LogRecord) -> None:
        """Ingest a single log record.

        The record is appended to the in-process buffer immediately and
        queued for backend dispatch. When the pending queue reaches
        ``_batch_size`` it is flushed automatically.
        """
        with self._lock:
            self._buffer.append(record)
            self._pending.append(record)
            if len(self._pending) >= self._batch_size:
                self._flush_locked()

    def ingest_many(self, records: list[LogRecord]) -> None:
        """Ingest multiple records at once."""
        for rec in records:
            self.ingest(rec)

    def flush(self) -> None:
        """Force-send all pending records to configured backends."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Must be called while holding self._lock."""
        if not self._pending:
            return
        batch = list(self._pending)
        self._pending.clear()
        for name, backend in self._backends.items():
            try:
                ok = backend.send(batch)
                if not ok:
                    logger.warning("Backend '%s' returned False for batch of %d", name, len(batch))
            except Exception as exc:
                logger.error("Backend '%s' raised during send: %s", name, exc)

    # ------------------------------------------------------------------
    # Search / correlation
    # ------------------------------------------------------------------

    def search(
        self,
        trace_id: str | None = None,
        level: str | None = None,
        logger_name: str | None = None,
        message_contains: str | None = None,
        since: float | None = None,  # unix timestamp
        limit: int = 100,
    ) -> list[LogRecord]:
        """Search the in-process buffer.

        All supplied filters are ANDed together.

        Args:
            trace_id:         Match records with this trace_id.
            level:            Match records with this log level (case-insensitive).
            logger_name:      Match records emitted by this logger.
            message_contains: Match records whose message contains this string.
            since:            Only return records with timestamp >= since.
            limit:            Maximum number of records to return (most recent first).

        Returns:
            List of matching LogRecord objects (newest first).
        """
        with self._lock:
            results: list[LogRecord] = []
            for rec in reversed(list(self._buffer)):
                if trace_id and rec.trace_id != trace_id:
                    continue
                if level and rec.level.upper() != level.upper():
                    continue
                if logger_name and rec.logger_name != logger_name:
                    continue
                if message_contains and message_contains not in rec.message:
                    continue
                if since is not None and rec.timestamp < since:
                    continue
                results.append(rec)
                if len(results) >= limit:
                    break
        return results

    def get_trace_records(self, trace_id: str) -> list[LogRecord]:
        """Return all records belonging to a trace, ordered oldest-first.

        Returns:
            List of results.
        """
        with self._lock:
            records = [r for r in self._buffer if r.trace_id == trace_id]
        return sorted(records, key=lambda r: r.timestamp)

    def correlate_span(self, trace_id: str, span_id: str) -> list[LogRecord]:
        """Return all records for a specific span within a trace.

        Args:
            trace_id: The trace id.
            span_id: The span id.

        Returns:
            List of results.
        """
        with self._lock:
            return [r for r in self._buffer if r.trace_id == trace_id and r.span_id == span_id]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get stats.

        Returns:
            The result string.
        """
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "pending": len(self._pending),
                "backends": list(self._backends.keys()),
                "max_buffer": self.MAX_BUFFER,
                "batch_size": self._batch_size,
            }

    def clear_buffer(self) -> None:
        """Discard all buffered records (for tests / memory management)."""
        with self._lock:
            self._buffer.clear()
            self._pending.clear()


# ---------------------------------------------------------------------------
# Python logging handler — bridge from stdlib logging to LogAggregator
# ---------------------------------------------------------------------------


class AggregatorHandler(logging.Handler):
    """A stdlib ``logging.Handler`` that feeds records into the ``LogAggregator``.

    Attach it to any logger to automatically capture structured log output:

        import logging
        from vetinari.dashboard.log_aggregator import AggregatorHandler

        root_logger = logging.getLogger()
        root_logger.addHandler(AggregatorHandler())
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit for the current context."""
        try:
            agg = get_log_aggregator()
            lr = LogRecord(
                message=self.format(record),
                level=record.levelname,
                timestamp=record.created,
                trace_id=getattr(record, "trace_id", None),
                span_id=getattr(record, "span_id", None),
                request_id=getattr(record, "request_id", None),
                logger_name=record.name,
                extra={
                    k: v
                    for k, v in record.__dict__.items()
                    if k not in logging.LogRecord.__dict__
                    and not k.startswith("_")
                    and k
                    not in (
                        "message",
                        "asctime",
                        "args",
                        "msg",
                        "levelname",
                        "levelno",
                        "name",
                        "pathname",
                        "filename",
                        "module",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                        "lineno",
                        "funcName",
                        "created",
                        "msecs",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                    )
                },
            )
            agg.ingest(lr)
        except Exception:  # pragma: no cover
            self.handleError(record)


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


def get_log_aggregator() -> LogAggregator:
    """Return the global LogAggregator singleton."""
    return LogAggregator()


def reset_log_aggregator() -> None:
    """Destroy the singleton (for tests / clean shutdown)."""
    with LogAggregator._class_lock:
        if LogAggregator._instance is not None:
            LogAggregator._instance.flush()
        LogAggregator._instance = None
    logger.debug("LogAggregator singleton reset")


# ---------------------------------------------------------------------------
# SSE (Server-Sent Events) Backend — streams log records to connected clients
# ---------------------------------------------------------------------------


class SSEBackend(BackendBase):
    """Backend that buffers log records for Server-Sent Events (SSE) streaming.

    Dashboard clients connect via an SSE endpoint and receive real-time log
    updates. Records are kept in a bounded deque; older entries are discarded
    when the buffer fills.
    """

    name = "sse"

    def __init__(self) -> None:
        self._buffer: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

    def configure(self, max_buffer: int = 1000, **_: Any) -> None:
        """Configure."""
        with self._lock:
            self._buffer = deque(maxlen=max_buffer)

    def send(self, records: list[LogRecord]) -> bool:
        """Send.

        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            for rec in records:
                self._buffer.append(rec)
        return True

    def get_recent(self, limit: int = 50) -> list[LogRecord]:
        """Return the most recent records (newest last).

        Returns:
            List of results.
        """
        with self._lock:
            items = list(self._buffer)
        return items[-limit:]

    def close(self) -> None:
        """Close for the current context."""
        with self._lock:
            self._buffer.clear()


_sse_backend_instance: SSEBackend | None = None
_sse_lock = threading.Lock()


def get_sse_backend() -> SSEBackend:
    """Return the global SSEBackend singleton.

    Returns:
        The SSEBackend result.
    """
    global _sse_backend_instance
    if _sse_backend_instance is None:
        with _sse_lock:
            if _sse_backend_instance is None:
                _sse_backend_instance = SSEBackend()
    return _sse_backend_instance


def reset_sse_backend() -> None:
    """Destroy the SSEBackend singleton (for tests)."""
    global _sse_backend_instance
    with _sse_lock:
        if _sse_backend_instance is not None:
            _sse_backend_instance.close()
        _sse_backend_instance = None
