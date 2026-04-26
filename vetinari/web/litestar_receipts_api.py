"""Litestar handlers for the WorkReceipt read API and SSE stream.

The Control Center consumes three endpoints from this module:

- ``GET /api/projects/{project_id}/receipts`` lists a project's receipts
  with pagination and optional ``kind`` / ``awaiting`` / ``since`` filters.
- ``GET /api/attention`` returns awaiting receipts across all projects so
  the Attention track can surface user-blocking work.
- ``GET /api/projects/{project_id}/receipts/stream`` opens an SSE channel
  fed by ``ReceiptAppended`` events so the UI can update without polling.

All routes are read-only -- mutating receipts (e.g., resolving an
``awaiting_user`` block) lives in a separate flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import AsyncGenerator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vetinari.events import EventBus, get_event_bus
from vetinari.receipts import (
    ReceiptAppended,
    WorkReceipt,
    WorkReceiptKind,
    WorkReceiptStore,
)

logger = logging.getLogger(__name__)

try:
    from litestar import get
    from litestar.response import Response, ServerSentEvent

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# -- Module-level SSE state --------------------------------------------------
# Who writes: ``stream_receipts`` adds queues; ``_dispatch_receipt`` writes events.
# Who reads: ``_generate_receipt_events`` drains its own queue; cleanup runs in
# the generator's ``finally`` so disconnected clients leak nothing.
# Lock: ``_clients_lock`` covers the dict + its inner sets.

_SSE_KEEPALIVE_TIMEOUT = 30.0
_SSE_MAX_IDLE_CYCLES = 120  # ~60 minutes at 30s
_SSE_PER_CLIENT_QUEUE_SIZE = 256
_MAX_SSE_CONNECTIONS = 50

_clients_lock = threading.Lock()
_clients_by_project: dict[str, set[queue.Queue]] = {}
_sse_connection_count = 0
_subscription_id: str | None = None
_subscription_lock = threading.Lock()


def _ensure_subscribed(bus: EventBus | None = None) -> None:
    """Subscribe to ``ReceiptAppended`` events once per process.

    The handler is registered on first use so import does not fire any
    side effects.
    """
    global _subscription_id
    if _subscription_id is not None:
        return
    with _subscription_lock:
        if _subscription_id is not None:
            return
        active_bus = bus if bus is not None else get_event_bus()
        _subscription_id = active_bus.subscribe(ReceiptAppended, _dispatch_receipt)


def _dispatch_receipt(event: ReceiptAppended) -> None:
    """Push a ``ReceiptAppended`` event into every interested client queue.

    Drops events on full queues (logs WARNING) per the SSE contract: events
    must never silently disappear and full-queue drops must be observable.
    """
    payload = json.dumps({
        "project_id": event.project_id,
        "receipt_id": event.receipt_id,
        "kind": event.kind,
        "passed": event.passed,
        "awaiting_user": event.awaiting_user,
    })

    with _clients_lock:
        per_project = list(_clients_by_project.get(event.project_id, ()))

    for q in per_project:
        try:
            q.put_nowait(payload)
        except queue.Full:
            logger.warning(
                "Receipts SSE queue full for project=%s — dropping event %s",
                event.project_id,
                event.receipt_id,
            )


def _add_client(project_id: str, q: queue.Queue) -> None:
    """Register a per-client queue for a project's receipt stream."""
    with _clients_lock:
        _clients_by_project.setdefault(project_id, set()).add(q)


def _remove_client(project_id: str, q: queue.Queue) -> None:
    """Remove a client queue when the SSE generator exits."""
    with _clients_lock:
        s = _clients_by_project.get(project_id)
        if s and q in s:
            s.discard(q)
            if not s:
                _clients_by_project.pop(project_id, None)


def _store(repo_root: Path | None = None) -> WorkReceiptStore:
    """Construct a ``WorkReceiptStore`` honouring an optional override."""
    return WorkReceiptStore(repo_root=repo_root)


def _kind_safe(value: str | None) -> WorkReceiptKind | None:
    """Coerce a string to a ``WorkReceiptKind`` or return ``None`` on miss.

    Used as an input-validation predicate by ``list_project_receipts``;
    the caller distinguishes ``None`` (no filter requested) from an
    invalid value via the original argument, so silent ``None`` here is
    correct -- the caller surfaces the 400 error message.
    """
    if value is None:
        return None
    try:
        return WorkReceiptKind(value)
    except ValueError:
        logger.debug("Unknown WorkReceiptKind value %r — caller will surface 400", value)
        return None


def _receipt_to_dict(receipt: WorkReceipt) -> dict[str, Any]:
    """Convert a ``WorkReceipt`` to a JSON-friendly dict for API responses.

    Mirrors the JSONL on-disk shape so the Control Center can decode the
    same record format whether it came from the API or the SSE channel.
    """
    return {
        "receipt_id": receipt.receipt_id,
        "project_id": receipt.project_id,
        "agent_id": receipt.agent_id,
        "agent_type": receipt.agent_type.value,
        "kind": receipt.kind.value,
        "started_at_utc": receipt.started_at_utc,
        "finished_at_utc": receipt.finished_at_utc,
        "inputs_summary": receipt.inputs_summary,
        "outputs_summary": receipt.outputs_summary,
        "outcome": {
            "passed": receipt.outcome.passed,
            "score": receipt.outcome.score,
            "basis": receipt.outcome.basis.value,
            "issues": list(receipt.outcome.issues),
            "suggestions": list(receipt.outcome.suggestions),
            "provenance": (asdict(receipt.outcome.provenance) if receipt.outcome.provenance is not None else None),
        },
        "awaiting_user": receipt.awaiting_user,
        "awaiting_reason": receipt.awaiting_reason,
        "linked_claim_ids": list(receipt.linked_claim_ids),
    }


def _filter_and_paginate(
    receipts: list[WorkReceipt],
    *,
    kind: WorkReceiptKind | None,
    awaiting: bool | None,
    since: str | None,
    limit: int,
    offset: int,
) -> tuple[list[WorkReceipt], int]:
    """Apply filters and paginate a list of receipts.

    Returns the page slice and the post-filter total so the API can return
    pagination metadata without re-walking the file.
    """
    filtered = receipts
    if kind is not None:
        filtered = [r for r in filtered if r.kind is kind]
    if awaiting is not None:
        filtered = [r for r in filtered if r.awaiting_user is awaiting]
    if since:
        filtered = [r for r in filtered if r.finished_at_utc >= since]

    total = len(filtered)
    page = filtered[offset : offset + limit]
    return page, total


async def _generate_receipt_events(
    project_id: str,
    q: queue.Queue,
) -> AsyncGenerator[dict[str, Any] | str, None]:
    """Stream ``data`` frames or ``comment`` keepalives until disconnect.

    Polls ``q`` without blocking the event loop. On client disconnect or
    idle-timeout the queue is removed in ``finally`` so resources do not
    leak (anti-pattern: SSE queue leaks).
    """
    idle_cycles = 0
    try:
        loop = asyncio.get_running_loop()
        while True:
            try:
                data = q.get_nowait()
                yield {"data": data}
                idle_cycles = 0
            except queue.Empty:
                deadline = loop.time() + _SSE_KEEPALIVE_TIMEOUT
                while True:
                    try:
                        data = q.get_nowait()
                        yield {"data": data}
                        idle_cycles = 0
                        break
                    except queue.Empty:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            idle_cycles += 1
                            if idle_cycles >= _SSE_MAX_IDLE_CYCLES:
                                logger.info(
                                    "Receipts SSE for project=%s idle %d cycles — disconnecting",
                                    project_id,
                                    idle_cycles,
                                )
                                return
                            yield {"comment": "keepalive"}
                            break
                        await asyncio.sleep(min(0.1, remaining))
    except GeneratorExit:  # noqa: VET022 — expected on client disconnect
        pass
    finally:
        _remove_client(project_id, q)
        with _clients_lock:
            global _sse_connection_count
            _sse_connection_count -= 1


def create_receipts_api_handlers() -> list[Any]:
    """Build the receipts read-API and SSE handlers.

    Returns:
        List of three Litestar route handler callables, or an empty list
        when Litestar is not available so the caller can extend its
        handler list without an import-time dependency.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    @get("/api/projects/{project_id:str}/receipts", sync_to_thread=False)
    def list_project_receipts(
        project_id: str,
        kind: str | None = None,
        awaiting: bool | None = None,
        since: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any] | Response:
        """Return a project's receipts with filtering and pagination.

        Args:
            project_id: Project identifier path parameter.
            kind: Optional ``WorkReceiptKind`` value filter.
            awaiting: When True, only receipts with ``awaiting_user=True``;
                False excludes them; ``None`` includes all.
            since: Optional ISO-8601 timestamp lower bound (inclusive) on
                ``finished_at_utc``.
            limit: Page size (default 100, capped at 500).
            offset: Page offset.

        Returns:
            JSON dict with ``project_id``, ``total``, ``offset``, ``limit``
            and a ``receipts`` array.
        """
        if limit <= 0 or limit > 500:
            limit = 100
        if offset < 0:
            offset = 0

        kind_value = _kind_safe(kind)
        if kind is not None and kind_value is None:
            return Response(
                content={"error": f"unknown kind: {kind!r}"},
                status_code=400,
            )

        try:
            receipts = list(_store().iter_receipts(project_id))
        except Exception:
            logger.warning("Could not read receipts for project=%s", project_id, exc_info=True)
            return Response(content={"error": "receipts unavailable"}, status_code=503)

        page, total = _filter_and_paginate(
            receipts,
            kind=kind_value,
            awaiting=awaiting,
            since=since,
            limit=limit,
            offset=offset,
        )

        return {
            "project_id": project_id,
            "total": total,
            "offset": offset,
            "limit": limit,
            "receipts": [_receipt_to_dict(r) for r in page],
        }

    @get("/api/attention", sync_to_thread=False)
    def list_attention() -> dict[str, Any] | Response:
        """Return all currently awaiting receipts across all projects.

        Walks every project under ``outputs/receipts/`` and returns the
        receipts with ``awaiting_user=True`` so the Attention track can
        list every project blocked on the user.

        Returns:
            JSON dict with a ``count`` and an ``items`` array of
            awaiting receipts.
        """
        store = _store()
        receipts_root = store.receipts_root

        items: list[dict[str, Any]] = []
        if receipts_root.exists():
            for project_dir in sorted(receipts_root.iterdir()):
                if not project_dir.is_dir():
                    continue
                try:
                    awaiting = store.find_awaiting(project_dir.name)
                except Exception:
                    logger.warning(
                        "Could not read awaiting receipts for project=%s",
                        project_dir.name,
                        exc_info=True,
                    )
                    continue
                items.extend(_receipt_to_dict(r) for r in awaiting)

        return {
            "count": len(items),
            "items": items,
        }

    @get("/api/projects/{project_id:str}/receipts/stream", sync_to_thread=False)
    def stream_receipts(project_id: str) -> Response | ServerSentEvent:
        """Open an SSE stream of ``receipt.appended`` events for a project.

        Args:
            project_id: Project identifier path parameter.

        Returns:
            ``ServerSentEvent`` streaming response, ``429`` when the cap
            is reached.
        """
        global _sse_connection_count

        # Subscribe outside the lock — _ensure_subscribed has its own lock and
        # publishes events that need the bus to be live before the queue exists.
        _ensure_subscribed()
        q: queue.Queue = queue.Queue(maxsize=_SSE_PER_CLIENT_QUEUE_SIZE)

        # Atomicize counter + queue registration so the cleanup path always
        # has both a counter increment AND a queue to remove (no TOCTOU
        # between the two — fixes MEDIUM finding from review).
        with _clients_lock:
            if _sse_connection_count >= _MAX_SSE_CONNECTIONS:
                logger.warning(
                    "Receipts SSE connection limit (%d) reached — rejecting new client",
                    _MAX_SSE_CONNECTIONS,
                )
                return Response(
                    content={"error": "Too many SSE connections"},
                    status_code=429,
                    headers={"Cache-Control": "no-cache"},
                )
            _sse_connection_count += 1
            _clients_by_project.setdefault(project_id, set()).add(q)

        return ServerSentEvent(
            _generate_receipt_events(project_id, q),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return [list_project_receipts, list_attention, stream_receipts]


__all__ = [
    "create_receipts_api_handlers",
]
