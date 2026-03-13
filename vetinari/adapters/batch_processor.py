"""Batch Processor — vetinari.adapters.batch_processor.

Queues non-urgent inference requests and flushes them in batches to provider
batch endpoints (typically 50% cost discount vs. synchronous API calls).

Supported providers
-------------------
- Anthropic  — uses /v1/messages/batches  (up to 10 000 requests, 50% discount)
- OpenAI     — uses /v1/batches            (up to 50 000 requests, 50% discount)

For providers without batch endpoints the processor falls back to sequential
synchronous calls so callers always get a result.

Usage
-----
    from vetinari.adapters.batch_processor import BatchProcessor, get_batch_processor

    bp = get_batch_processor()

    # Enqueue (non-blocking)
    future = bp.enqueue(request, provider="anthropic")

    # Wait for result
    response = future.result(timeout=300)

    # Or flush immediately
    results = bp.flush()

Configuration
-------------
- ``BATCH_FLUSH_INTERVAL``  — seconds between automatic flushes (default 60)
- ``BATCH_MAX_SIZE``        — max requests per batch before forced flush (default 100)
- ``BATCH_ENABLED``         — set to "0" to disable batching (falls back to sync)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_FLUSH_INTERVAL = float(os.environ.get("BATCH_FLUSH_INTERVAL", "60"))
_MAX_BATCH_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "100"))
_BATCH_ENABLED = os.environ.get("BATCH_ENABLED", "1").lower() not in ("0", "false", "no")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BatchItem:
    """A single queued inference request."""

    item_id: str
    provider: str
    request: Any  # vetinari.adapters.base.InferenceRequest
    future: Future[Any]
    enqueued_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result for a single batch item."""

    item_id: str
    response: Any | None  # vetinari.adapters.base.InferenceResponse
    error: str | None = None
    cached: bool = False

    @property
    def success(self) -> bool:
        return self.error is None and self.response is not None


# ---------------------------------------------------------------------------
# Provider batch backends
# ---------------------------------------------------------------------------


class _AnthropicBatchBackend:
    """Submits batches to Anthropic /v1/messages/batches."""

    def __init__(self, api_key: str, api_version: str = "2023-06-01"):
        self._api_key = api_key
        self._api_version = api_version
        self._base_url = "https://api.anthropic.com/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
            "anthropic-beta": "message-batches-2024-09-24",
            "Content-Type": "application/json",
        }

    def submit(self, items: list[BatchItem]) -> dict[str, BatchResult]:
        """Submit batch to Anthropic. Returns mapping item_id -> BatchResult."""
        import requests as _req  # local import to avoid module-level dependency

        batch_requests = []
        for item in items:
            req = item.request
            payload: dict[str, Any] = {
                "model": req.model_id,
                "max_tokens": req.max_tokens,
                "messages": [{"role": "user", "content": req.prompt}],
                "temperature": req.temperature,
                "top_p": req.top_p,
                "top_k": req.top_k,
            }
            if req.system_prompt:
                payload["system"] = req.system_prompt
            if req.stop_sequences:
                payload["stop_sequences"] = req.stop_sequences

            batch_requests.append(
                {
                    "custom_id": item.item_id,
                    "params": payload,
                }
            )

        try:
            response = _req.post(
                f"{self._base_url}/messages/batches",
                headers=self._headers(),
                json={"requests": batch_requests},
                timeout=60,
            )
            response.raise_for_status()
            batch_data = response.json()
            batch_id = batch_data["id"]
            logger.info("[Anthropic] Batch submitted: %s (%d items)", batch_id, len(items))

            # Poll for results
            results = self._poll_results(batch_id, item_ids={i.item_id for i in items})
            return results

        except Exception as exc:
            logger.error("[Anthropic] Batch submission failed: %s — falling back to sync", exc)
            # Return error results for all items
            return {item.item_id: BatchResult(item_id=item.item_id, response=None, error=str(exc)) for item in items}

    def _poll_results(
        self, batch_id: str, item_ids: set, poll_interval: float = 5.0, timeout: float = 600.0
    ) -> dict[str, BatchResult]:
        """Poll Anthropic batch until complete."""
        import requests as _req

        from vetinari.adapters.base import InferenceResponse  # local import

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = _req.get(
                    f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
                    headers=self._headers(),
                    timeout=30,
                )
                resp.raise_for_status()
                status_data = resp.json()

                if status_data.get("processing_status") == "ended":
                    # Fetch results
                    results_resp = _req.get(
                        f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results",
                        headers=self._headers(),
                        timeout=60,
                    )
                    results_resp.raise_for_status()
                    batch_results: dict[str, BatchResult] = {}
                    for line in results_resp.text.strip().splitlines():
                        import json

                        item_result = json.loads(line)
                        item_id = item_result.get("custom_id", "")
                        if item_result.get("result", {}).get("type") == "succeeded":
                            msg = item_result["result"]["message"]
                            output = ""
                            if msg.get("content"):
                                output = msg["content"][0].get("text", "")
                            tokens = msg.get("usage", {}).get("input_tokens", 0) + msg.get("usage", {}).get(
                                "output_tokens", 0
                            )
                            cached = msg.get("usage", {}).get("cache_read_input_tokens", 0) > 0
                            inf_resp = InferenceResponse(
                                model_id=msg.get("model", ""),
                                output=output,
                                latency_ms=0,
                                tokens_used=tokens,
                                status="ok",
                            )
                            batch_results[item_id] = BatchResult(item_id=item_id, response=inf_resp, cached=cached)
                        else:
                            err = str(item_result.get("result", {}).get("error", "unknown"))
                            batch_results[item_id] = BatchResult(item_id=item_id, response=None, error=err)
                    return batch_results
            except Exception as poll_exc:
                logger.warning("[Anthropic] Batch poll error: %s", poll_exc)

            time.sleep(poll_interval)

        # Timeout
        return {
            item_id: BatchResult(item_id=item_id, response=None, error="Batch poll timeout") for item_id in item_ids
        }


class _OpenAIBatchBackend:
    """Submits batches to OpenAI /v1/batches."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._base_url = "https://api.openai.com/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def submit(self, items: list[BatchItem]) -> dict[str, BatchResult]:
        """Submit batch to OpenAI. Returns mapping item_id -> BatchResult."""
        import io
        import json

        import requests as _req

        # Build JSONL file
        jsonl_lines = []
        for item in items:
            req = item.request
            messages = []
            if req.system_prompt:
                messages.append({"role": "system", "content": req.system_prompt})
            messages.append({"role": "user", "content": req.prompt})

            body: dict[str, Any] = {
                "model": req.model_id,
                "messages": messages,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "max_tokens": req.max_tokens,
            }
            if req.stop_sequences:
                body["stop"] = req.stop_sequences[:4]

            jsonl_lines.append(
                json.dumps(
                    {
                        "custom_id": item.item_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
            )

        jsonl_content = "\n".join(jsonl_lines).encode("utf-8")

        try:
            # Upload file
            upload_resp = _req.post(
                f"{self._base_url}/files",
                headers={"Authorization": f"Bearer {self._api_key}"},
                files={"file": ("batch.jsonl", io.BytesIO(jsonl_content), "application/jsonl")},
                data={"purpose": "batch"},
                timeout=60,
            )
            upload_resp.raise_for_status()
            file_id = upload_resp.json()["id"]

            # Create batch
            batch_resp = _req.post(
                f"{self._base_url}/batches",
                headers=self._headers(),
                json={
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
                timeout=30,
            )
            batch_resp.raise_for_status()
            batch_id = batch_resp.json()["id"]
            logger.info("[OpenAI] Batch submitted: %s (%d items)", batch_id, len(items))

            return self._poll_results(batch_id, item_ids={i.item_id for i in items})

        except Exception as exc:
            logger.error("[OpenAI] Batch submission failed: %s — falling back to sync", exc)
            return {item.item_id: BatchResult(item_id=item.item_id, response=None, error=str(exc)) for item in items}

    def _poll_results(
        self, batch_id: str, item_ids: set, poll_interval: float = 10.0, timeout: float = 86400.0
    ) -> dict[str, BatchResult]:
        """Poll OpenAI batch until complete."""
        import json

        import requests as _req

        from vetinari.adapters.base import InferenceResponse

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = _req.get(
                    f"{self._base_url}/batches/{batch_id}",
                    headers=self._headers(),
                    timeout=30,
                )
                resp.raise_for_status()
                batch_data = resp.json()
                status = batch_data.get("status", "")

                if status == "completed":
                    output_file_id = batch_data.get("output_file_id")
                    if not output_file_id:
                        break

                    file_resp = _req.get(
                        f"{self._base_url}/files/{output_file_id}/content",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        timeout=120,
                    )
                    file_resp.raise_for_status()

                    batch_results: dict[str, BatchResult] = {}
                    for line in file_resp.text.strip().splitlines():
                        item_result = json.loads(line)
                        item_id = item_result.get("custom_id", "")
                        resp_body = item_result.get("response", {}).get("body", {})
                        if resp_body.get("choices"):
                            output = resp_body["choices"][0].get("message", {}).get("content", "")
                            tokens = resp_body.get("usage", {}).get("total_tokens", 0)
                            cached = (
                                resp_body.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0
                            )
                            inf_resp = InferenceResponse(
                                model_id=resp_body.get("model", ""),
                                output=output,
                                latency_ms=0,
                                tokens_used=tokens,
                                status="ok",
                            )
                            batch_results[item_id] = BatchResult(item_id=item_id, response=inf_resp, cached=cached)
                        else:
                            err = str(item_result.get("error", "unknown"))
                            batch_results[item_id] = BatchResult(item_id=item_id, response=None, error=err)
                    return batch_results

                elif status in ("failed", "expired", "cancelled"):
                    break

            except Exception as poll_exc:
                logger.warning("[OpenAI] Batch poll error: %s", poll_exc)

            time.sleep(poll_interval)

        return {
            item_id: BatchResult(item_id=item_id, response=None, error="Batch poll timeout/failed")
            for item_id in item_ids
        }


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------


class BatchProcessor:
    """Queues non-urgent inference requests and flushes them in batches.

    Batching provides ~50% cost discount on supported providers.
    For unsupported providers, falls back to sequential sync calls.

    Thread-safe. The auto-flush background thread is started lazily on first enqueue.
    """

    def __init__(
        self,
        flush_interval: float = _FLUSH_INTERVAL,
        max_batch_size: int = _MAX_BATCH_SIZE,
        enabled: bool = _BATCH_ENABLED,
    ):
        self.flush_interval = flush_interval
        self.max_batch_size = max_batch_size
        self.enabled = enabled

        self._queue: dict[str, list[BatchItem]] = {}  # provider -> items
        self._lock = threading.Lock()
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._backends: dict[str, Any] = {}

        logger.info(
            "BatchProcessor initialized (enabled=%s, flush_interval=%.0fs, max_batch=%d)",
            enabled,
            flush_interval,
            max_batch_size,
        )

    def register_backend(self, provider: str, backend: Any) -> None:
        """Register a batch backend for a provider."""
        self._backends[provider] = backend
        logger.debug("Registered batch backend for provider: %s", provider)

    def register_anthropic(self, api_key: str, api_version: str = "2023-06-01") -> None:
        """Convenience method to register Anthropic batch backend."""
        self.register_backend("anthropic", _AnthropicBatchBackend(api_key, api_version))

    def register_openai(self, api_key: str) -> None:
        """Convenience method to register OpenAI batch backend."""
        self.register_backend("openai", _OpenAIBatchBackend(api_key))

    def enqueue(self, request: Any, provider: str = "anthropic", metadata: dict[str, Any] | None = None) -> Future[Any]:
        """Enqueue a request for batch processing.

        Args:
            request: InferenceRequest to process.
            provider: Provider name ("anthropic", "openai", etc.).
            metadata: Optional extra metadata for the batch item.

        Returns:
            Future that resolves to an InferenceResponse.
        """
        future: Future[Any] = Future()

        if not self.enabled:
            # Fall through to sync immediately
            self._execute_sync(request, provider, future)
            return future

        item = BatchItem(
            item_id=str(uuid.uuid4()),
            provider=provider,
            request=request,
            future=future,
            metadata=metadata or {},
        )

        with self._lock:
            if provider not in self._queue:
                self._queue[provider] = []
            self._queue[provider].append(item)
            queue_len = len(self._queue[provider])

        logger.debug("Enqueued request %s for %s (queue_len=%d)", item.item_id, provider, queue_len)

        # Auto-flush if batch is full
        if queue_len >= self.max_batch_size:
            threading.Thread(target=self._flush_provider, args=(provider,), daemon=True).start()

        # Start background flush thread if not running
        self._ensure_flush_thread()

        return future

    def flush(self, provider: str | None = None) -> int:
        """Flush queued requests immediately.

        Args:
            provider: Specific provider to flush, or None for all.

        Returns:
            Number of items flushed.
        """
        if provider:
            return self._flush_provider(provider)
        else:
            total = 0
            with self._lock:
                providers = list(self._queue.keys())
            for p in providers:
                total += self._flush_provider(p)
            return total

    def _flush_provider(self, provider: str) -> int:
        """Flush all queued items for a provider. Returns count flushed."""
        with self._lock:
            items = self._queue.pop(provider, [])

        if not items:
            return 0

        logger.info("Flushing %d items for provider: %s", len(items), provider)

        backend = self._backends.get(provider)
        if backend is None:
            # No batch backend — fall back to sequential sync
            logger.debug("No batch backend for %s, running sync fallback", provider)
            for item in items:
                self._execute_sync(item.request, provider, item.future)
            return len(items)

        try:
            results = backend.submit(items)
            for item in items:
                result = results.get(item.item_id)
                if result and result.success:
                    item.future.set_result(result.response)
                else:
                    err = result.error if result else "No result returned"
                    # Return error response rather than exception so callers can handle gracefully
                    from vetinari.adapters.base import InferenceResponse

                    item.future.set_result(
                        InferenceResponse(
                            model_id=item.request.model_id,
                            output="",
                            latency_ms=0,
                            tokens_used=0,
                            status="error",
                            error=err,
                        )
                    )
        except Exception as exc:
            logger.error("Batch flush failed for %s: %s", provider, exc)
            from vetinari.adapters.base import InferenceResponse

            for item in items:
                if not item.future.done():
                    item.future.set_result(
                        InferenceResponse(
                            model_id=item.request.model_id,
                            output="",
                            latency_ms=0,
                            tokens_used=0,
                            status="error",
                            error=str(exc),
                        )
                    )

        return len(items)

    def _execute_sync(self, request: Any, provider: str, future: Future[Any]) -> None:
        """Execute a single request synchronously and resolve the future."""
        try:
            from vetinari.adapter_manager import get_adapter_manager

            manager = get_adapter_manager()
            adapter = manager.get_adapter(provider)
            if adapter:
                response = adapter.infer(request)
            else:
                from vetinari.adapters.base import InferenceResponse

                response = InferenceResponse(
                    model_id=request.model_id,
                    output="",
                    latency_ms=0,
                    tokens_used=0,
                    status="error",
                    error=f"No adapter for provider: {provider}",
                )
            future.set_result(response)
        except Exception as exc:
            from vetinari.adapters.base import InferenceResponse

            future.set_result(
                InferenceResponse(
                    model_id=getattr(request, "model_id", "unknown"),
                    output="",
                    latency_ms=0,
                    tokens_used=0,
                    status="error",
                    error=str(exc),
                )
            )

    def _ensure_flush_thread(self) -> None:
        """Start background flush thread if not already running."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return

        self._stop_event.clear()

        def _auto_flush() -> None:
            logger.info("BatchProcessor auto-flush thread started (interval=%.0fs)", self.flush_interval)
            while not self._stop_event.wait(self.flush_interval):
                try:
                    self.flush()
                except Exception as exc:
                    logger.error("Auto-flush error: %s", exc)
            logger.info("BatchProcessor auto-flush thread stopped")

        self._flush_thread = threading.Thread(target=_auto_flush, daemon=True, name="batch-processor-flush")
        self._flush_thread.start()

    def stop(self) -> None:
        """Stop the background flush thread and flush remaining items."""
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
        # Final flush
        self.flush()

    def get_queue_stats(self) -> dict[str, Any]:
        """Return current queue statistics."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "providers": {p: len(items) for p, items in self._queue.items()},
                "total_queued": sum(len(items) for items in self._queue.values()),
                "flush_interval": self.flush_interval,
                "max_batch_size": self.max_batch_size,
                "backends_registered": list(self._backends.keys()),
                "flush_thread_active": self._flush_thread is not None and self._flush_thread.is_alive(),
            }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_batch_processor: BatchProcessor | None = None
_bp_lock = threading.Lock()


def get_batch_processor() -> BatchProcessor:
    """Get or create the global BatchProcessor instance."""
    global _batch_processor
    if _batch_processor is None:
        with _bp_lock:
            if _batch_processor is None:
                _batch_processor = BatchProcessor()
    return _batch_processor


def reset_batch_processor() -> None:
    """Reset the global BatchProcessor (useful for testing)."""
    global _batch_processor
    with _bp_lock:
        if _batch_processor is not None:
            _batch_processor.stop()
        _batch_processor = None
