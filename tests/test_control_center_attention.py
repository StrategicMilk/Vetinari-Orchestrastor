"""Control Center attention-track tests (SHARD-03 task 3.4).

These tests focus on the Control Center contract — what the UI renders
when there are zero receipts (no synthetic placeholders) and what it
sees when projects raise structured ``awaiting_reason`` blockers.

The Svelte component itself is exercised in the browser; here we cover
the API surface the component subscribes to and the file-system shape
the Svelte module expects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
from vetinari.events import EventBus
from vetinari.receipts import WorkReceipt, WorkReceiptKind, WorkReceiptStore
from vetinari.types import AgentType, EvidenceBasis

pytest.importorskip("litestar", reason="Litestar required for Control Center tests")

from litestar import Litestar
from litestar.testing import TestClient

from vetinari.web.litestar_receipts_api import create_receipts_api_handlers


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


def _awaiting_receipt(project_id: str, reason: str) -> WorkReceipt:
    return WorkReceipt(
        project_id=project_id,
        agent_id="inspector-001",
        agent_type=AgentType.INSPECTOR,
        kind=WorkReceiptKind.INSPECTOR_PASS,
        outcome=_outcome(),
        inputs_summary="ran inspector pass",
        outputs_summary="inspector found 1 unsupported claim",
        awaiting_user=True,
        awaiting_reason=reason,
    )


@pytest.fixture
def receipts_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from vetinari.receipts import store as store_module

    monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)
    return Litestar(route_handlers=create_receipts_api_handlers())


class TestEmptyState:
    """Zero receipts must produce an empty list, not a placeholder."""

    def test_empty_attention_has_no_synthetic_items(self, receipts_app: Any) -> None:
        with TestClient(app=receipts_app) as client:
            response = client.get("/api/attention")
        body = response.json()
        # Anti-pattern Fallback as success: the API must NOT invent a "no
        # work yet" placeholder receipt; the empty list is the truthful
        # answer and the UI shows "No attention required" from it.
        assert body == {"count": 0, "items": []}

    def test_project_with_zero_receipts_has_empty_list(self, receipts_app: Any) -> None:
        with TestClient(app=receipts_app) as client:
            response = client.get("/api/projects/no-such-project/receipts")
        body = response.json()
        assert body["receipts"] == []
        assert body["total"] == 0


class TestAttentionAcrossProjects:
    """Two awaiting receipts on two different projects both surface."""

    def test_attention_surfaces_two_projects(
        self,
        receipts_app: Any,
        tmp_path: Path,
    ) -> None:
        bus = EventBus()
        store = WorkReceiptStore(repo_root=tmp_path, event_bus=bus)
        store.append(_awaiting_receipt("proj-A", "Foreman: ambiguous goal — clarify intent"))
        store.append(_awaiting_receipt("proj-B", "Inspector: 2 unsupported claims"))

        with TestClient(app=receipts_app) as client:
            response = client.get("/api/attention")
        body = response.json()

        assert body["count"] == 2
        projects = sorted(item["project_id"] for item in body["items"])
        assert projects == ["proj-A", "proj-B"]
        reasons = sorted(item["awaiting_reason"] for item in body["items"])
        assert reasons == [
            "Foreman: ambiguous goal — clarify intent",
            "Inspector: 2 unsupported claims",
        ]


class TestSvelteComponentInPlace:
    """The Svelte AttentionTrack component file is registered in the UI tree.

    This is a lightweight wiring check — the Python suite cannot drive a
    full Svelte mount, but it can verify the file exists and Dashboard
    imports it. Real browser rendering must be exercised via the dev
    server before release.
    """

    def test_attention_track_component_exists(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        path = repo_root / "ui" / "svelte" / "src" / "components" / "dashboard" / "AttentionTrack.svelte"
        assert path.exists(), "AttentionTrack.svelte not present"
        src = path.read_text(encoding="utf-8")
        assert "api.listAttention" in src, "AttentionTrack must call /api/attention via api.js wrapper"
        assert "No attention required" in src, "empty-state copy missing"
        assert "awaiting_reason" in src, "structured reason field must be displayed"

    def test_dashboard_view_renders_attention_track(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        dashboard = (repo_root / "ui" / "svelte" / "src" / "views" / "Dashboard.svelte").read_text(encoding="utf-8")
        assert "import AttentionTrack" in dashboard
        assert "<AttentionTrack" in dashboard

    def test_api_wrapper_exposes_listAttention(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        api_js = (repo_root / "ui" / "svelte" / "src" / "lib" / "api.js").read_text(encoding="utf-8")
        assert "export function listAttention" in api_js
        assert "/attention" in api_js


class TestProjectReceiptStripInPlace:
    """The Svelte ProjectReceiptStrip component is wired into project cards.

    SHARD-03 task 3.3: project cards must show counts by WorkReceiptKind
    and the spec'd ``"No work recorded yet"`` empty-state. This is a
    source-text tripwire because the Python test suite cannot drive a
    Svelte mount; real rendering verification belongs in a browser
    smoke test against the dev server.
    """

    def test_project_receipt_strip_component_exists(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        path = repo_root / "ui" / "svelte" / "src" / "components" / "projects" / "ProjectReceiptStrip.svelte"
        assert path.exists(), "ProjectReceiptStrip.svelte not present"
        src = path.read_text(encoding="utf-8")
        assert "api.listProjectReceipts" in src, (
            "ProjectReceiptStrip must call /api/projects/{id}/receipts via api.js wrapper"
        )
        # Anti-pattern Fallback as success — the empty-state copy is the
        # truthful answer when a project has zero receipts. The exact
        # phrase is the SHARD-03 contract.
        assert "No work recorded yet" in src, "empty-state copy missing"
        # The strip displays counts for each of the five WorkReceiptKind values.
        for kind in ("plan_round", "worker_task", "inspector_pass", "training_step", "release_step"):
            assert kind in src, f"strip must reference WorkReceiptKind.{kind}"

    def test_projects_view_renders_receipt_strip(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        view = (repo_root / "ui" / "svelte" / "src" / "views" / "ProjectsView.svelte").read_text(encoding="utf-8")
        assert "import ProjectReceiptStrip" in view
        assert "<ProjectReceiptStrip" in view

    def test_api_wrapper_exposes_listProjectReceipts(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        api_js = (repo_root / "ui" / "svelte" / "src" / "lib" / "api.js").read_text(encoding="utf-8")
        assert "export function listProjectReceipts" in api_js
        assert "/projects/" in api_js and "/receipts" in api_js
