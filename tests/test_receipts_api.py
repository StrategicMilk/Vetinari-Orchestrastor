"""Tests for vetinari.web.litestar_receipts_api -- read API + SSE.

Covers SHARD-03 task 3.1, 3.2, 3.4 acceptance:
- Listing routes paginate, filter by kind, filter by awaiting.
- /api/attention surfaces awaiting receipts across projects with the
  structured ``awaiting_reason`` (no client-side synthesis).
- SSE stream delivers a complete receipt event end-to-end (no
  prefix-only or length-only assertions).
- All routes go through the real ASGI app stack (no handler-direct
  ``handler.fn(...)`` shortcuts -- anti-pattern).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
from vetinari.events import EventBus
from vetinari.receipts import (
    WorkReceipt,
    WorkReceiptKind,
    WorkReceiptStore,
)
from vetinari.types import AgentType, EvidenceBasis

pytest.importorskip("litestar", reason="Litestar required for receipts API tests")

from litestar import Litestar
from litestar.testing import TestClient

from vetinari.web.litestar_receipts_api import create_receipts_api_handlers

# ---- helpers ---------------------------------------------------------------


def _outcome(*, passed: bool = True) -> OutcomeSignal:
    """Build a representative tool-evidence-backed OutcomeSignal."""
    return OutcomeSignal(
        passed=passed,
        score=0.9 if passed else 0.0,
        basis=EvidenceBasis.TOOL_EVIDENCE,
        tool_evidence=(
            ToolEvidence(
                tool_name="pytest",
                command="pytest -q",
                exit_code=0 if passed else 1,
                passed=passed,
            ),
        ),
        provenance=Provenance(
            source="vetinari.receipts.tests",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            tool_name="pytest",
        ),
    )


def _receipt(
    *,
    project_id: str,
    kind: WorkReceiptKind = WorkReceiptKind.WORKER_TASK,
    awaiting_user: bool = False,
    awaiting_reason: str | None = None,
    inputs_summary: str = "test input",
    finished_at_utc: str | None = None,
) -> WorkReceipt:
    """Build a WorkReceipt with sensible defaults for table-driven tests."""
    extra = {}
    if finished_at_utc is not None:
        extra["finished_at_utc"] = finished_at_utc
    return WorkReceipt(
        project_id=project_id,
        agent_id="test-agent-001",
        agent_type=AgentType.WORKER,
        kind=kind,
        outcome=_outcome(),
        inputs_summary=inputs_summary,
        outputs_summary="test output",
        awaiting_user=awaiting_user,
        awaiting_reason=awaiting_reason,
        **extra,
    )


@pytest.fixture
def isolated_bus() -> EventBus:
    """Per-test EventBus to keep subscribers isolated."""
    return EventBus()


@pytest.fixture
def receipts_store(tmp_path: Path, isolated_bus: EventBus) -> WorkReceiptStore:
    """Per-test WorkReceiptStore rooted at tmp_path."""
    return WorkReceiptStore(repo_root=tmp_path, event_bus=isolated_bus)


@pytest.fixture
def receipts_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a Litestar app with only the receipts handlers, rooted at tmp_path."""
    from vetinari.receipts import store as store_module

    monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)

    handlers = create_receipts_api_handlers()
    return Litestar(route_handlers=handlers)


# ---- /api/projects/{project_id}/receipts -----------------------------------


class TestListProjectReceipts:
    """``GET /api/projects/{project_id}/receipts`` filtering and pagination."""

    def test_returns_empty_for_unknown_project(self, receipts_app: Any) -> None:
        with TestClient(app=receipts_app) as client:
            response = client.get("/api/projects/unknown/receipts")
        assert response.status_code == 200
        body = response.json()
        assert body == {
            "project_id": "unknown",
            "total": 0,
            "offset": 0,
            "limit": 100,
            "receipts": [],
        }

    def test_returns_all_receipts_in_append_order(
        self,
        receipts_app: Any,
        receipts_store: WorkReceiptStore,
    ) -> None:
        for label in ("first", "second", "third"):
            receipts_store.append(_receipt(project_id="proj-list", inputs_summary=label))

        with TestClient(app=receipts_app) as client:
            response = client.get("/api/projects/proj-list/receipts")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 3
        assert [r["inputs_summary"] for r in body["receipts"]] == ["first", "second", "third"]

    def test_filter_by_kind(
        self,
        receipts_app: Any,
        receipts_store: WorkReceiptStore,
    ) -> None:
        receipts_store.append(_receipt(project_id="proj-kind", kind=WorkReceiptKind.WORKER_TASK))
        receipts_store.append(_receipt(project_id="proj-kind", kind=WorkReceiptKind.PLAN_ROUND))
        receipts_store.append(_receipt(project_id="proj-kind", kind=WorkReceiptKind.WORKER_TASK))

        with TestClient(app=receipts_app) as client:
            response = client.get("/api/projects/proj-kind/receipts?kind=worker_task")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert all(r["kind"] == "worker_task" for r in body["receipts"])

    def test_unknown_kind_returns_400(
        self,
        receipts_app: Any,
        receipts_store: WorkReceiptStore,
    ) -> None:
        receipts_store.append(_receipt(project_id="proj-bad-kind"))
        with TestClient(app=receipts_app) as client:
            response = client.get("/api/projects/proj-bad-kind/receipts?kind=banana")
        assert response.status_code == 400
        assert "unknown kind" in response.json()["error"]

    def test_filter_by_awaiting(
        self,
        receipts_app: Any,
        receipts_store: WorkReceiptStore,
    ) -> None:
        receipts_store.append(_receipt(project_id="proj-await"))
        receipts_store.append(
            _receipt(
                project_id="proj-await",
                awaiting_user=True,
                awaiting_reason="inspector surfaced unsupported claims: 1",
            )
        )
        receipts_store.append(_receipt(project_id="proj-await"))

        with TestClient(app=receipts_app) as client:
            response = client.get("/api/projects/proj-await/receipts?awaiting=true")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["receipts"][0]["awaiting_user"] is True
        assert body["receipts"][0]["awaiting_reason"] == "inspector surfaced unsupported claims: 1"

    def test_auxiliary_actors_serialise_agent_type_correctly(
        self,
        receipts_app: Any,
        receipts_store: WorkReceiptStore,
    ) -> None:
        """API surfaces ``RELEASE`` / ``TRAINING`` actor labels honestly.

        Per ADR-0103, training and release runners use their own
        ``AgentType`` values. The HTTP layer must preserve them so
        downstream filters can distinguish factory-pipeline work from
        auxiliary work without guessing from ``agent_id`` substrings.
        """
        receipts_store.append(
            WorkReceipt(
                project_id="proj-aux-api",
                agent_id="release-doctor:0.0.1",
                agent_type=AgentType.RELEASE,
                kind=WorkReceiptKind.RELEASE_STEP,
                outcome=_outcome(),
                inputs_summary="release pipeline step: smoke",
                outputs_summary="ok",
            )
        )
        with TestClient(app=receipts_app) as client:
            response = client.get("/api/projects/proj-aux-api/receipts")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        record = body["receipts"][0]
        assert record["agent_type"] == AgentType.RELEASE.value
        assert record["kind"] == WorkReceiptKind.RELEASE_STEP.value

    def test_pagination_limit_and_offset(
        self,
        receipts_app: Any,
        receipts_store: WorkReceiptStore,
    ) -> None:
        for i in range(5):
            receipts_store.append(_receipt(project_id="proj-page", inputs_summary=f"r{i}"))

        with TestClient(app=receipts_app) as client:
            response = client.get("/api/projects/proj-page/receipts?limit=2&offset=2")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 5
        assert body["limit"] == 2
        assert body["offset"] == 2
        assert [r["inputs_summary"] for r in body["receipts"]] == ["r2", "r3"]


# ---- /api/attention --------------------------------------------------------


class TestAttentionTrack:
    """``GET /api/attention`` surfaces awaiting receipts across projects."""

    def test_returns_empty_when_no_projects(self, receipts_app: Any) -> None:
        with TestClient(app=receipts_app) as client:
            response = client.get("/api/attention")
        assert response.status_code == 200
        body = response.json()
        assert body == {"count": 0, "items": []}

    def test_surfaces_awaiting_across_projects(
        self,
        receipts_app: Any,
        receipts_store: WorkReceiptStore,
    ) -> None:
        # Two awaiting receipts across two projects, plus a non-awaiting one.
        receipts_store.append(_receipt(project_id="proj-A"))
        receipts_store.append(
            _receipt(
                project_id="proj-A",
                awaiting_user=True,
                awaiting_reason="plan reviewer refused -- destructive scope",
            )
        )
        receipts_store.append(
            _receipt(
                project_id="proj-B",
                awaiting_user=True,
                awaiting_reason="awaiting human approval for migration",
            )
        )

        with TestClient(app=receipts_app) as client:
            response = client.get("/api/attention")
        assert response.status_code == 200
        body = response.json()

        assert body["count"] == 2
        reasons = sorted(item["awaiting_reason"] for item in body["items"])
        assert reasons == [
            "awaiting human approval for migration",
            "plan reviewer refused -- destructive scope",
        ]

    def test_reasons_are_real_not_synthesised(
        self,
        receipts_app: Any,
        receipts_store: WorkReceiptStore,
    ) -> None:
        # Bug repro: server must not synthesise a reason — it must echo the
        # exact string set when the receipt was emitted.
        original_reason = "Foreman blocked: TS-14 non-goal matched (refusal_id=ref_abc)"
        receipts_store.append(
            _receipt(
                project_id="proj-real-reason",
                awaiting_user=True,
                awaiting_reason=original_reason,
            )
        )

        with TestClient(app=receipts_app) as client:
            response = client.get("/api/attention")
        body = response.json()
        assert body["count"] == 1
        assert body["items"][0]["awaiting_reason"] == original_reason


# ---- SSE stream ------------------------------------------------------------


class TestReceiptsStream:
    """SSE event-pipe coverage for the receipts stream channel.

    Drives the dispatch + generator components end-to-end to prove a
    real receipt round-trips through the SSE event pipeline. Handler-
    factory and app-registration coverage live in TestFactory and
    TestAppRegistration so the route's HTTP wiring is also verified.
    """

    def test_dispatch_pushes_to_per_project_queue(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Publishing a ReceiptAppended must reach the per-project queue."""
        import queue as _queue

        from vetinari.events import get_event_bus, reset_event_bus
        from vetinari.receipts import store as store_module
        from vetinari.web import litestar_receipts_api as api_mod

        # Force the store to write under tmp_path so we don't pollute the
        # repo's outputs directory.
        monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)

        reset_event_bus()
        # Reset module-level subscription so a fresh subscription is made
        # against the new bus.
        monkeypatch.setattr(api_mod, "_subscription_id", None)
        monkeypatch.setattr(api_mod, "_clients_by_project", {})
        try:
            api_mod._ensure_subscribed(bus=get_event_bus())
            client_q: _queue.Queue = _queue.Queue(maxsize=16)
            api_mod._add_client("proj-sse", client_q)

            store = WorkReceiptStore(repo_root=tmp_path, event_bus=get_event_bus())
            receipt = _receipt(
                project_id="proj-sse",
                awaiting_user=True,
                awaiting_reason="awaiting test confirmation",
            )
            store.append(receipt)

            payload_raw = client_q.get(timeout=5.0)
            payload = json.loads(payload_raw)

            assert payload["project_id"] == "proj-sse"
            assert payload["receipt_id"] == receipt.receipt_id
            assert payload["kind"] == WorkReceiptKind.WORKER_TASK.value
            assert payload["awaiting_user"] is True
            assert payload["passed"] is True
        finally:
            reset_event_bus()

    def test_full_event_drains_through_async_generator(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The async generator must yield a complete data frame, not a prefix.

        Asserts the parsed JSON payload contained in the SSE ``data`` frame
        rather than substring/length checks (anti-pattern: prefix-only
        assertions).
        """
        import asyncio as _asyncio
        import queue as _queue

        from vetinari.events import get_event_bus, reset_event_bus
        from vetinari.receipts import store as store_module
        from vetinari.web import litestar_receipts_api as api_mod

        monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)

        reset_event_bus()
        monkeypatch.setattr(api_mod, "_subscription_id", None)
        monkeypatch.setattr(api_mod, "_clients_by_project", {})
        try:
            api_mod._ensure_subscribed(bus=get_event_bus())
            client_q: _queue.Queue = _queue.Queue(maxsize=16)
            api_mod._add_client("proj-async", client_q)

            store = WorkReceiptStore(repo_root=tmp_path, event_bus=get_event_bus())
            receipt = _receipt(project_id="proj-async")
            store.append(receipt)

            async def _drain_one() -> dict[str, Any] | str:
                gen = api_mod._generate_receipt_events("proj-async", client_q)
                async for item in gen:
                    await gen.aclose()
                    return item
                raise AssertionError("generator yielded nothing")

            frame = _asyncio.run(_drain_one())
            assert isinstance(frame, dict) and "data" in frame
            payload = json.loads(frame["data"])
            assert payload["receipt_id"] == receipt.receipt_id
            assert payload["project_id"] == "proj-async"
            assert payload["kind"] == WorkReceiptKind.WORKER_TASK.value
        finally:
            reset_event_bus()

    def test_generator_cleans_up_queue_on_close(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``finally`` must remove the queue (anti-pattern: SSE queue leaks)."""
        import asyncio as _asyncio
        import queue as _queue

        from vetinari.events import get_event_bus, reset_event_bus
        from vetinari.receipts import store as store_module
        from vetinari.web import litestar_receipts_api as api_mod

        monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)
        reset_event_bus()
        monkeypatch.setattr(api_mod, "_subscription_id", None)
        monkeypatch.setattr(api_mod, "_clients_by_project", {})
        # Pretend a client connected so we can prove the cleanup decrements.
        monkeypatch.setattr(api_mod, "_sse_connection_count", 1)
        try:
            api_mod._ensure_subscribed(bus=get_event_bus())
            client_q: _queue.Queue = _queue.Queue(maxsize=16)
            api_mod._add_client("proj-cleanup", client_q)
            assert "proj-cleanup" in api_mod._clients_by_project

            async def _close() -> None:
                gen = api_mod._generate_receipt_events("proj-cleanup", client_q)
                # Push one frame so __anext__ returns immediately, putting
                # the generator into its try block. Without this, aclose()
                # on an unstarted generator skips the finally entirely.
                client_q.put_nowait(json.dumps({"test": "frame"}))
                await gen.__anext__()
                await gen.aclose()

            _asyncio.run(_close())

            assert "proj-cleanup" not in api_mod._clients_by_project
            assert api_mod._sse_connection_count == 0
        finally:
            reset_event_bus()


# ---- factory + UI wiring smoke ---------------------------------------------


class TestFactory:
    """create_receipts_api_handlers() returns the expected handler set."""

    def test_returns_three_handlers(self) -> None:
        handlers = create_receipts_api_handlers()
        assert len(handlers) == 3
        paths: set[str] = set()
        for h in handlers:
            paths.update(getattr(h, "paths", ()) or ())
        assert "/api/projects/{project_id:str}/receipts" in paths
        assert "/api/attention" in paths
        assert "/api/projects/{project_id:str}/receipts/stream" in paths

    def test_returns_empty_when_litestar_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import vetinari.web.litestar_receipts_api as mod

        monkeypatch.setattr(mod, "_LITESTAR_AVAILABLE", False)
        assert mod.create_receipts_api_handlers() == []


class TestAppRegistration:
    """The receipts handlers are wired into the main Litestar app factory."""

    def test_litestar_app_registers_receipts_routes(self) -> None:
        from vetinari.web.litestar_app import create_app

        app = create_app()
        registered = {route.path for route in app.routes}
        assert any(p.endswith("/receipts") for p in registered), "list_project_receipts not registered"
        assert "/api/attention" in registered
        assert any(p.endswith("/receipts/stream") for p in registered), "stream_receipts not registered"
