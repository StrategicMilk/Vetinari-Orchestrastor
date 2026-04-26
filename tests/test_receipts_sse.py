"""SSE-specific contract tests for the receipts stream.

Covers:
- ``ReceiptAppended`` events dispatched to the right per-project queues.
- The async generator yields a complete frame, never a prefix or length-
  only assertion (anti-pattern: prefix-only assertions).
- The generator's ``finally`` block removes the queue (anti-pattern: SSE
  queue leaks) and decrements the connection counter.
- Queue-full drops emit a WARNING with the project + receipt id (anti-
  pattern: events silently dropped).
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
from vetinari.events import EventBus, get_event_bus, reset_event_bus
from vetinari.receipts import WorkReceipt, WorkReceiptKind, WorkReceiptStore
from vetinari.types import AgentType, EvidenceBasis

pytest.importorskip("litestar", reason="Litestar required for SSE tests")


def _outcome() -> OutcomeSignal:
    return OutcomeSignal(
        passed=True,
        score=0.9,
        basis=EvidenceBasis.TOOL_EVIDENCE,
        tool_evidence=(ToolEvidence(tool_name="pytest", command="pytest -q", exit_code=0, passed=True),),
        provenance=Provenance(
            source="vetinari.receipts.tests",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            tool_name="pytest",
        ),
    )


def _receipt(project_id: str = "proj-sse") -> WorkReceipt:
    return WorkReceipt(
        project_id=project_id,
        agent_id="worker-001",
        agent_type=AgentType.WORKER,
        kind=WorkReceiptKind.WORKER_TASK,
        outcome=_outcome(),
        inputs_summary="sse test input",
        outputs_summary="sse test output",
    )


@pytest.fixture
def isolated_bus() -> EventBus:
    return EventBus()


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reset SSE module state between tests."""
    from vetinari.receipts import store as store_module
    from vetinari.web import litestar_receipts_api as api_mod

    monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)
    monkeypatch.setattr(api_mod, "_subscription_id", None)
    monkeypatch.setattr(api_mod, "_clients_by_project", {})
    monkeypatch.setattr(api_mod, "_sse_connection_count", 0)
    reset_event_bus()
    yield
    reset_event_bus()


class TestDispatch:
    """ReceiptAppended events dispatch to the right queue."""

    def test_event_reaches_per_project_queue(self, tmp_path: Path) -> None:
        from vetinari.web import litestar_receipts_api as api_mod

        api_mod._ensure_subscribed(bus=get_event_bus())
        client_q: queue.Queue = queue.Queue(maxsize=8)
        api_mod._add_client("proj-1", client_q)

        store = WorkReceiptStore(repo_root=tmp_path, event_bus=get_event_bus())
        receipt = _receipt(project_id="proj-1")
        store.append(receipt)

        payload = json.loads(client_q.get(timeout=2.0))
        assert payload["receipt_id"] == receipt.receipt_id
        assert payload["project_id"] == "proj-1"

    def test_other_project_does_not_receive_event(self, tmp_path: Path) -> None:
        from vetinari.web import litestar_receipts_api as api_mod

        api_mod._ensure_subscribed(bus=get_event_bus())
        listener_q: queue.Queue = queue.Queue(maxsize=8)
        api_mod._add_client("proj-listener", listener_q)

        store = WorkReceiptStore(repo_root=tmp_path, event_bus=get_event_bus())
        store.append(_receipt(project_id="proj-other"))

        with pytest.raises(queue.Empty):
            listener_q.get(timeout=0.5)


class TestGeneratorFrames:
    """The generator must yield full data frames, not prefixes."""

    def test_generator_yields_complete_data_frame(self) -> None:
        from vetinari.web import litestar_receipts_api as api_mod

        client_q: queue.Queue = queue.Queue(maxsize=8)
        client_q.put_nowait(json.dumps({"receipt_id": "rcpt-001", "kind": "worker_task"}))

        async def _drain() -> dict[str, object]:
            gen = api_mod._generate_receipt_events("proj-x", client_q)
            try:
                async for item in gen:
                    return item
                raise AssertionError("generator yielded nothing")
            finally:
                await gen.aclose()

        frame = asyncio.run(_drain())
        assert isinstance(frame, dict)
        assert "data" in frame
        # Anti-pattern Prefix-only assertions: parse the JSON and assert
        # specific fields rather than ``startswith("data:")`` only.
        decoded = json.loads(frame["data"])
        assert decoded == {"receipt_id": "rcpt-001", "kind": "worker_task"}


class TestCleanup:
    """The generator's ``finally`` removes the queue from the registry."""

    def test_cleanup_removes_queue_on_close(self, tmp_path: Path) -> None:
        from vetinari.web import litestar_receipts_api as api_mod

        client_q: queue.Queue = queue.Queue(maxsize=8)
        api_mod._add_client("proj-cleanup", client_q)
        # Mimic stream_receipts having incremented the counter.
        api_mod._sse_connection_count = 1

        async def _close() -> None:
            gen = api_mod._generate_receipt_events("proj-cleanup", client_q)
            client_q.put_nowait(json.dumps({"event": "frame"}))
            await gen.__anext__()
            await gen.aclose()

        asyncio.run(_close())

        assert "proj-cleanup" not in api_mod._clients_by_project
        assert api_mod._sse_connection_count == 0


class TestQueueFullLogging:
    """Queue-full drops must log WARNING (anti-pattern: events dropped)."""

    def test_queue_full_drop_logs_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from vetinari.web import litestar_receipts_api as api_mod

        api_mod._ensure_subscribed(bus=get_event_bus())
        # Tiny queue so a single append fills it; the second triggers drop.
        full_q: queue.Queue = queue.Queue(maxsize=1)
        full_q.put_nowait("seed")  # pre-fill so ANY new event is dropped
        api_mod._add_client("proj-full", full_q)

        store = WorkReceiptStore(repo_root=tmp_path, event_bus=get_event_bus())
        with caplog.at_level(logging.WARNING, logger="vetinari.web.litestar_receipts_api"):
            store.append(_receipt(project_id="proj-full"))

        msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        # Anti-pattern Events silently dropped: the drop must produce a
        # WARNING that names the project so an operator can correlate it.
        assert any("Receipts SSE queue full" in m and "proj-full" in m for m in msgs)
