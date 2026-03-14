"""Continuous Batching (C17).

==========================
Thread-safe inference request batching for LM Studio vLLM backend.

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


@dataclass
class BatchConfig:
    """Configuration for the inference batcher."""

    enabled: bool = False
    max_batch_size: int = 8
    max_wait_ms: float = 100.0  # dispatch every 100ms
    lmstudio_host: str = ""


class InferenceBatcher:
    """Collects and batches inference requests.

    When ``submit()`` is called, the request is queued. A background
    thread dispatches batches to LM Studio either when the batch is
    full or the wait timer expires.
    """

    def __init__(self, config: BatchConfig | None = None):
        self._config = config or BatchConfig()
        if not self._config.lmstudio_host:
            self._config.lmstudio_host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")  # noqa: VET041
        self._queue: queue.Queue[BatchRequest] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._total_batches = 0
        self._total_requests = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
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
        """Stop the dispatch thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, request: BatchRequest, timeout: float = 30.0) -> str:
        """Submit an inference request and wait for the result.

        If batching is disabled, dispatches immediately (synchronous).

        Args:
            request: The request.
            timeout: The timeout.

        Returns:
            The result string.

        Raises:
            RuntimeError: If the operation fails.
        """
        if not self._config.enabled:
            return self._dispatch_single(request)

        if not self._running:
            self.start()

        self._queue.put(request)
        request.event.wait(timeout=timeout)

        if request.error:
            raise RuntimeError(f"Batch inference failed: {request.error}")
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
                    batch.append(req)
                except queue.Empty:
                    break

            if batch:
                self._dispatch_batch(batch)

    def _dispatch_batch(self, batch: list[BatchRequest]) -> None:
        """Dispatch a batch of requests to LM Studio."""
        with self._lock:
            self._total_batches += 1
            self._total_requests += len(batch)

        # Group by model for efficient batching
        by_model: dict[str, list[BatchRequest]] = {}
        for req in batch:
            by_model.setdefault(req.model_id, []).append(req)

        for model_id, requests in by_model.items():
            try:
                import requests as http_requests

                # LM Studio doesn't support true batch API yet,
                # so we send individual requests but benefit from
                # connection pooling and reduced queue overhead
                session = http_requests.Session()
                for req in requests:
                    try:
                        resp = session.post(
                            f"{self._config.lmstudio_host}/v1/chat/completions",
                            json={
                                "model": model_id or "default",
                                "messages": [
                                    {"role": "system", "content": req.system_prompt},
                                    {"role": "user", "content": req.prompt},
                                ],
                                "max_tokens": req.max_tokens,
                                "temperature": req.temperature,
                            },
                            timeout=120,
                        )
                        resp.raise_for_status()
                        req.result = resp.json()["choices"][0]["message"]["content"]
                    except Exception as e:
                        req.error = str(e)
                    finally:
                        req.event.set()
                session.close()
            except Exception as e:
                for req in requests:
                    req.error = str(e)
                    req.event.set()

    def _dispatch_single(self, request: BatchRequest) -> str:
        """Synchronous single-request dispatch (batching disabled)."""
        try:
            import requests as http_requests

            resp = http_requests.post(
                f"{self._config.lmstudio_host}/v1/chat/completions",
                json={
                    "model": request.model_id or "default",
                    "messages": [
                        {"role": "system", "content": request.system_prompt},
                        {"role": "user", "content": request.prompt},
                    ],
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}") from e

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": (self._total_requests / max(self._total_batches, 1)),
            "queue_size": self._queue.qsize(),
        }


# ── Singleton ─────────────────────────────────────────────────────────

_batcher: InferenceBatcher | None = None


def get_inference_batcher(config: BatchConfig | None = None) -> InferenceBatcher:
    """Get inference batcher.

    Returns:
        The InferenceBatcher result.
    """
    global _batcher
    if _batcher is None:
        _batcher = InferenceBatcher(config)
    return _batcher
