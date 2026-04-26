"""Log Aggregation Backend Implementations for the Vetinari Dashboard.

Contains all concrete backend classes that ship log records to external
systems, plus the SSEBackend for real-time dashboard streaming.

Supported backends
------------------
  file     Write newline-delimited JSON to a local file.
  datadog  Send log entries via the Datadog Logs Intake API.
  webhook  POST aggregated logs to an arbitrary HTTP endpoint.
  sse      Buffer records for Server-Sent Events streaming clients.

All network backends degrade gracefully: if the dependency (``requests``) is
missing, or the remote endpoint is unreachable, the error is logged and the
call returns False rather than raising.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any

from vetinari.constants import DATADOG_LOGS_URL, LOG_BACKEND_BUFFER_SIZE, LOG_BACKEND_TIMEOUT, LOGS_DIR
from vetinari.dashboard.log_aggregator import BackendBase, LogRecord
from vetinari.http import create_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File backend
# ---------------------------------------------------------------------------


class FileBackend(BackendBase):
    """Appends newline-delimited JSON to a local file."""

    name = "file"

    def __init__(self) -> None:
        self._path: str | None = None
        self._lock = threading.Lock()

    def configure(self, path: str = str(LOGS_DIR / "vetinari_audit.jsonl"), **_: Any) -> None:
        """Configure the file backend.

        Args:
            path: Filesystem path of the output JSONL file.  Parent
                directories are created automatically.
            **_: Ignored extra keyword arguments.
        """
        self._path = path
        Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)

    def send(self, records: list[LogRecord]) -> bool:
        """Append records to the configured file.

        Args:
            records: Batch of log records to write.

        Returns:
            True on success, False if the backend is unconfigured or an
            OSError occurs.
        """
        if not self._path:
            logger.warning("FileBackend not configured (no path set).")
            return False
        try:
            with self._lock, Path(self._path).open("a", encoding="utf-8") as fh:
                fh.writelines(rec.to_json() + "\n" for rec in records)
            return True
        except OSError as exc:
            logger.error("FileBackend.send failed: %s", exc)
            return False

    def close(self) -> None:
        """Release resources held by this backend.

        File handles are opened and closed per ``send()`` call, so there is
        nothing to release here.
        """
        # noqa: VET031  (intentional: file opened/closed per send call — nothing to release)


# ---------------------------------------------------------------------------
# Datadog backend
# ---------------------------------------------------------------------------


class DatadogBackend(BackendBase):
    """Sends records via the Datadog Logs Intake API."""

    name = "datadog"
    _DD_URL = DATADOG_LOGS_URL

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
        """Configure the Datadog backend.

        Args:
            api_key: Datadog API key for authentication.
            service: Service name tag applied to every log entry.
            ddsource: Source tag (language / integration name).
            ddtags: Comma-separated key:value tag string.
            **_: Ignored extra keyword arguments.
        """
        self._api_key = api_key
        self._service = service
        self._ddsource = ddsource
        self._ddtags = ddtags

    def send(self, records: list[LogRecord]) -> bool:
        """Send records to Datadog Logs Intake API.

        Args:
            records: Batch of log records to ship.

        Returns:
            True on success, False if the backend is unconfigured, the
            ``requests`` package is missing, or the HTTP call fails.
        """
        if not self._api_key:
            logger.warning("DatadogBackend not configured (api_key missing).")
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
            with create_session() as session:
                resp = session.post(
                    self._DD_URL,
                    json=payload,
                    headers=headers,
                    timeout=LOG_BACKEND_TIMEOUT,
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
        self._timeout: int = LOG_BACKEND_TIMEOUT

    def configure(
        self,
        url: str = "",
        headers: dict[str, str] | None = None,
        timeout: int = LOG_BACKEND_TIMEOUT,
        **_: Any,
    ) -> None:
        """Configure the webhook backend.

        Args:
            url: Target webhook URL.
            headers: Optional extra HTTP headers to merge into each request.
            timeout: Request timeout in seconds.
            **_: Ignored extra keyword arguments.
        """
        self._url = url
        if headers:
            self._headers.update(headers)
        self._timeout = timeout

    def send(self, records: list[LogRecord]) -> bool:
        """POST records as a JSON array to the configured webhook.

        Args:
            records: Batch of log records to ship.

        Returns:
            True on success, False if the backend is unconfigured, the
            ``requests`` package is missing, or the HTTP call fails.
        """
        if not self._url:
            logger.warning("WebhookBackend not configured (url missing).")
            return False
        payload = [rec.to_dict() for rec in records]
        try:
            with create_session() as session:
                resp = session.post(
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
        self._buffer: deque = deque(maxlen=LOG_BACKEND_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._clients: list[queue.Queue] = []
        self._clients_lock = threading.Lock()

    def configure(self, max_buffer: int = LOG_BACKEND_BUFFER_SIZE, **_: Any) -> None:
        """Configure the SSE backend.

        Args:
            max_buffer: Maximum number of log records to retain in memory.
            **_: Ignored extra keyword arguments.
        """
        with self._lock:
            self._buffer = deque(maxlen=max_buffer)

    def add_client(self, q: queue.Queue) -> None:
        """Register a new SSE client queue for real-time log delivery.

        Args:
            q: Queue that will receive serialised log records as they arrive.
        """
        with self._clients_lock:
            self._clients.append(q)

    def remove_client(self, q: queue.Queue) -> None:
        """Unregister a previously-registered SSE client queue.

        Args:
            q: The queue to remove from the client list.
        """
        import contextlib

        with self._clients_lock, contextlib.suppress(ValueError):
            self._clients.remove(q)

    def send(self, records: list[LogRecord]) -> bool:
        """Buffer records for SSE consumption and forward to connected clients.

        Args:
            records: Batch of log records to buffer.

        Returns:
            Always True — buffering does not fail.
        """
        with self._lock:
            for rec in records:
                self._buffer.append(rec)
        # Fan out to connected SSE clients.
        # Use rec.to_json() — json.dumps(dataclass) with default=str produces a
        # repr blob rather than a proper JSON object, so the SSE consumer cannot
        # parse it.
        with self._clients_lock:
            dead: list[queue.Queue] = []
            for client_q in self._clients:
                for rec in records:
                    try:
                        client_q.put_nowait(rec.to_json())
                    except Exception:
                        dead.append(client_q)
                        break
            import contextlib

            for dq in dead:
                with contextlib.suppress(ValueError):
                    self._clients.remove(dq)
        return True

    def get_recent(self, limit: int = 50) -> list[LogRecord]:
        """Return the most recent records (newest last).

        Args:
            limit: Maximum number of records to return.

        Returns:
            Up to ``limit`` most recent log records, ordered oldest-first.
        """
        with self._lock:
            items = list(self._buffer)
        return items[-limit:]

    def close(self) -> None:
        """Discard all buffered records and disconnect all SSE clients.

        Clears both the in-memory record buffer and the connected client queue
        list so that subsequent calls to send() and get_recent() return empty
        results and no stale queue references are held after shutdown.
        """
        with self._lock:
            self._buffer.clear()
        # Clear the client list under its own lock so in-flight send() calls
        # that hold _clients_lock see a consistent empty list after close().
        with self._clients_lock:
            self._clients.clear()


# ---------------------------------------------------------------------------
# SSE singleton helpers
# ---------------------------------------------------------------------------

_sse_backend_instance: SSEBackend | None = None
_sse_lock = threading.Lock()


def get_sse_backend() -> SSEBackend:
    """Return the global SSEBackend singleton.

    Returns:
        The process-wide SSEBackend instance, created on first call.
    """
    global _sse_backend_instance
    if _sse_backend_instance is None:
        with _sse_lock:
            if _sse_backend_instance is None:
                _sse_backend_instance = SSEBackend()
    return _sse_backend_instance


def reset_sse_backend() -> None:
    """Destroy the SSEBackend singleton (for tests / clean shutdown)."""
    global _sse_backend_instance
    with _sse_lock:
        if _sse_backend_instance is not None:
            _sse_backend_instance.close()
        _sse_backend_instance = None
