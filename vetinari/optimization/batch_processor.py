"""Batch API processing for Vetinari (P10.7).

Queues LLM inference requests and dispatches them in batches to reduce
per-request overhead.  Supports priority ordering (high/normal/low) and
provides per-request callbacks.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_FLUSH_INTERVAL: float = 0.5  # seconds
_DEFAULT_MAX_BATCH_SIZE: int = 10

_instance: Optional[BatchProcessor] = None
_instance_lock: threading.Lock = threading.Lock()


class Priority(str, Enum):
    """Request priority levels for the batch queue."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

    @property
    def _order(self) -> int:
        return {"high": 0, "normal": 1, "low": 2}[self.value]

    def __lt__(self, other: Priority) -> bool:  # type: ignore[override]
        return self._order < other._order


@dataclass
class BatchRequest:
    """A single request submitted to the :class:`BatchProcessor`.

    Attributes:
        request_id: Unique identifier for the request.
        model_id: The model to use for inference.
        prompt: The user prompt text.
        system_prompt: Optional system prompt to prepend.
        priority: Scheduling priority (high/normal/low).
        callback: Optional callable invoked with the :class:`BatchResult` when done.
    """

    request_id: str
    model_id: str
    prompt: str
    system_prompt: str = ""
    priority: Priority = Priority.NORMAL
    callback: Optional[Callable[[BatchResult], None]] = field(default=None, repr=False)


@dataclass
class BatchResult:
    """Result for a single request after batch processing.

    Attributes:
        request_id: The ID matching the original :class:`BatchRequest`.
        output: Model output text (empty string on failure).
        tokens_used: Total tokens consumed (prompt + completion).
        latency_ms: Wall-clock latency in milliseconds.
        status: ``"ok"`` on success, ``"error"`` on failure.
    """

    request_id: str
    output: str
    tokens_used: int
    latency_ms: float
    status: str  # "ok" | "error"


# Internal queue item: (priority_order, enqueue_time, BatchRequest)
_QueueItem = tuple[int, float, BatchRequest]


class BatchProcessor:
    """Thread-safe priority batch processor for LLM inference requests.

    Requests are queued and dispatched in order of priority (high first).
    Within the same priority level, FIFO ordering is preserved.

    Args:
        flush_interval: Seconds between automatic flushes (informational; callers
                        drive flushing via :meth:`flush`).
        max_batch_size: Default maximum requests per :meth:`flush` call.
    """

    def __init__(
        self,
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        self._flush_interval = flush_interval
        self._max_batch_size = max_batch_size
        self._lock = threading.Lock()
        self._queue: list[_QueueItem] = []
        self._total_processed: int = 0
        self._batch_sizes: list[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, request: BatchRequest) -> str:
        """Add a request to the batch queue.

        Args:
            request: The :class:`BatchRequest` to enqueue.

        Returns:
            The ``request_id`` of the enqueued request (same as ``request.request_id``).
        """
        with self._lock:
            item: _QueueItem = (request.priority._order, time.monotonic(), request)
            self._queue.append(item)
            # Keep sorted by (priority_order, enqueue_time)
            self._queue.sort(key=lambda x: (x[0], x[1]))
        logger.debug("BatchProcessor: queued request %s (priority=%s)", request.request_id, request.priority)
        return request.request_id

    def flush(self, max_batch_size: int = 0) -> list[BatchResult]:
        """Process up to ``max_batch_size`` queued requests and return results.

        Requests are processed in priority order (high before normal before low).
        This implementation simulates inference synchronously; integrate with a
        real adapter by overriding :meth:`_invoke`.

        Args:
            max_batch_size: Maximum requests to process in this flush.  Defaults
                            to the instance's ``max_batch_size`` if 0.

        Returns:
            List of :class:`BatchResult` objects, one per processed request.
        """
        if max_batch_size <= 0:
            max_batch_size = self._max_batch_size

        with self._lock:
            batch = self._queue[:max_batch_size]
            self._queue = self._queue[max_batch_size:]

        if not batch:
            return []

        results: list[BatchResult] = []
        for _, _, req in batch:
            result = self._invoke(req)
            results.append(result)
            if req.callback is not None:
                try:
                    req.callback(result)
                except Exception:
                    logger.exception("BatchProcessor: callback raised for request %s", req.request_id)

        with self._lock:
            self._total_processed += len(results)
            self._batch_sizes.append(len(results))

        logger.debug("BatchProcessor: flushed %d requests", len(results))
        return results

    def get_stats(self) -> dict:
        """Return processing statistics.

        Returns:
            Dictionary with keys: ``queue_depth``, ``total_processed``,
            ``avg_batch_size``.
        """
        with self._lock:
            avg = sum(self._batch_sizes) / len(self._batch_sizes) if self._batch_sizes else 0.0
            return {
                "queue_depth": len(self._queue),
                "total_processed": self._total_processed,
                "avg_batch_size": avg,
            }

    def clear(self) -> None:
        """Discard all queued requests and reset statistics."""
        with self._lock:
            self._queue.clear()
            self._total_processed = 0
            self._batch_sizes.clear()

    # ------------------------------------------------------------------
    # Extension point
    # ------------------------------------------------------------------

    def _invoke(self, request: BatchRequest) -> BatchResult:
        """Simulate or dispatch a single inference request.

        Override this method to integrate with a real LLM adapter.

        Args:
            request: The request to process.

        Returns:
            A :class:`BatchResult` with the model output.
        """
        start = time.monotonic()
        # Default implementation — subclasses should override to call a real LLM adapter
        tokens_used = max(1, len(request.prompt) // 4)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return BatchResult(
            request_id=request.request_id,
            output=f"[batch-stub] response for {request.request_id}",
            tokens_used=tokens_used,
            latency_ms=elapsed_ms,
            status="ok",
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_request(
    prompt: str,
    model_id: str = "default",
    system_prompt: str = "",
    priority: Priority = Priority.NORMAL,
    callback: Optional[Callable[[BatchResult], None]] = None,
) -> BatchRequest:
    """Create a :class:`BatchRequest` with an auto-generated ID.

    Args:
        prompt: User prompt text.
        model_id: Model identifier string.
        system_prompt: Optional system prompt.
        priority: Scheduling priority.
        callback: Optional completion callback.

    Returns:
        A new :class:`BatchRequest`.
    """
    return BatchRequest(
        request_id=str(uuid.uuid4()),
        model_id=model_id,
        prompt=prompt,
        system_prompt=system_prompt,
        priority=priority,
        callback=callback,
    )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


def get_batch_processor(
    flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
    max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
) -> BatchProcessor:
    """Return the module-level singleton :class:`BatchProcessor`.

    Args:
        flush_interval: Flush interval in seconds (used on first creation only).
        max_batch_size: Default batch size (used on first creation only).

    Returns:
        The singleton :class:`BatchProcessor` instance.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = BatchProcessor(flush_interval=flush_interval, max_batch_size=max_batch_size)
    return _instance


def reset_batch_processor() -> None:
    """Destroy the singleton so the next call to ``get_batch_processor`` creates a fresh one.

    Intended for use in tests only.
    """
    global _instance
    with _instance_lock:
        _instance = None
