"""Continuous Batching (C17).

==========================
Thread-safe inference request batching for local in-process inference.

Collects inference requests into batches and dispatches them together
every ``max_wait_ms`` or when ``max_batch_size`` is reached.

Config keys: batching.enabled, batching.max_batch_size, batching.max_wait_ms
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vetinari.constants import INFERENCE_BATCHER_QUEUE_SIZE, THREAD_JOIN_TIMEOUT
from vetinari.exceptions import InferenceError
from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single inference request in the batch queue."""

    request_id: str
    model_id: str
    prompt: str
    system_prompt: str = ""
    max_tokens: int = 2048
    temperature: float = 0.3
    callback: Callable | None = None
    result: str | None = None
    error: str | None = None
    event: threading.Event = field(default_factory=threading.Event)

    def __repr__(self) -> str:
        return (
            f"BatchRequest(request_id={self.request_id!r}, model_id={self.model_id!r}, max_tokens={self.max_tokens!r})"
        )


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for the inference batcher."""

    enabled: bool = False
    max_batch_size: int = 8
    max_wait_ms: float = 100.0  # dispatch every 100ms
    models_dir: str = ""

    def __repr__(self) -> str:
        return f"BatchConfig(enabled={self.enabled!r}, max_batch_size={self.max_batch_size!r}, max_wait_ms={self.max_wait_ms!r})"


class InferenceBatcher:
    """Collects and batches inference requests.

    When ``submit()`` is called, the request is queued. A background
    thread dispatches batches using the local in-process inference adapter
    either when the batch is full or the wait timer expires.
    """

    def __init__(self, config: BatchConfig | None = None):
        cfg = config or BatchConfig()
        if not cfg.models_dir:
            from dataclasses import replace

            cfg = replace(cfg, models_dir=os.environ.get("VETINARI_MODELS_DIR", ""))
        self._config = cfg
        self._queue: queue.Queue[BatchRequest] = queue.Queue(maxsize=INFERENCE_BATCHER_QUEUE_SIZE)
        self._running = False
        self._thread: threading.Thread | None = None
        self._total_batches = 0
        self._total_requests = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether continuous batching is enabled in the current configuration."""
        return self._config.enabled

    def start(self) -> None:
        """Start the background dispatch thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="InferenceBatcher")
        self._thread.start()
        logger.info(
            "InferenceBatcher started (batch_size=%d, wait_ms=%.0f)",
            self._config.max_batch_size,
            self._config.max_wait_ms,
        )

    def stop(self) -> None:
        """Stop the dispatch thread.

        Sets the running flag and places a sentinel value into the queue so the
        worker unblocks immediately rather than waiting up to ``max_wait_ms``.
        """
        self._running = False
        # Sentinel wakes the worker so it exits without waiting for a real request.
        self._queue.put(None)  # type: ignore[arg-type]
        if self._thread:
            self._thread.join(timeout=THREAD_JOIN_TIMEOUT)

    def submit(self, request: BatchRequest, timeout: float = 30.0) -> str:
        """Submit an inference request and wait for the result.

        If batching is disabled, dispatches immediately (synchronous).

        Args:
            request: The request.
            timeout: The timeout.

        Returns:
            The generated text output from the inference model.  Empty string
            if batching is enabled but the model produced no output.

        Raises:
            InferenceError: If the inference call fails or the batch marks the
                request as errored.
        """
        if not self._config.enabled:
            return self._dispatch_single(request)

        if not self._running:
            self.start()

        self._queue.put(request)
        request.event.wait(timeout=timeout)

        if request.error:
            raise InferenceError(f"Batch inference failed: {request.error}")
        return request.result or ""

    def _dispatch_loop(self) -> None:
        """Background thread: collect and dispatch batches."""
        while self._running:
            batch: list[BatchRequest] = []
            deadline = time.monotonic() + self._config.max_wait_ms / 1000.0

            # Collect up to max_batch_size or until deadline
            while len(batch) < self._config.max_batch_size:
                remaining = max(0, deadline - time.monotonic())
                try:
                    req = self._queue.get(timeout=remaining)
                    if req is None:
                        # Sentinel placed by stop() — exit immediately.
                        return
                    batch.append(req)
                except queue.Empty:
                    break

            if batch:
                self._dispatch_batch(batch)

    def _dispatch_batch(self, batch: list[BatchRequest]) -> None:
        """Dispatch a batch of requests via local in-process inference."""
        with self._lock:
            self._total_batches += 1
            self._total_requests += len(batch)

        # Group by model for efficient batching
        by_model: dict[str, list[BatchRequest]] = {}
        for req in batch:
            by_model.setdefault(req.model_id, []).append(req)

        for model_id, requests in by_model.items():
            try:
                from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

                adapter = LocalInferenceAdapter()
                for req in requests:
                    try:
                        result = adapter.chat(
                            model_id=model_id or "default",
                            system_prompt=req.system_prompt,
                            input_text=req.prompt,
                        )
                        req.result = result.get("output", "")
                    except Exception as e:
                        req.error = str(e)
                    finally:
                        req.event.set()
            except Exception as e:
                for req in requests:
                    req.error = str(e)
                    req.event.set()

    def _dispatch_single(self, request: BatchRequest) -> str:
        """Synchronous single-request dispatch (batching disabled)."""
        try:
            from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

            adapter = LocalInferenceAdapter()
            result = adapter.chat(
                model_id=request.model_id or "default",
                system_prompt=request.system_prompt,
                input_text=request.prompt,
            )
            return result.get("output", "")
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        """Return runtime statistics for the inference batcher.

        Returns:
            Dictionary containing enabled state, running status, total batches
            dispatched, total requests processed, average batch size, and
            current queue depth.
        """
        return {
            "enabled": self._config.enabled,
            StatusEnum.RUNNING.value: self._running,
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": (self._total_requests / max(self._total_batches, 1)),
            "queue_size": self._queue.qsize(),
        }


# ── Singleton ─────────────────────────────────────────────────────────

_batcher: InferenceBatcher | None = None
_batcher_lock = threading.Lock()


def get_inference_batcher(config: BatchConfig | None = None) -> InferenceBatcher:
    """Get inference batcher.

    Returns:
        The process-wide InferenceBatcher singleton.  If it does not yet
        exist, it is created with ``config`` (or default settings if None).
        Subsequent calls ignore ``config`` and return the existing instance.
    """
    global _batcher
    if _batcher is None:
        with _batcher_lock:
            if _batcher is None:
                _batcher = InferenceBatcher(config)
    return _batcher
