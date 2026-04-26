"""Mounted request-level governance tests for memory and autonomy routes.

Proves that every handler in litestar_memory_api and litestar_autonomy_api
returns a structured JSON error (status 500, body {status: "error", ...})
when its backing subsystem raises, rather than letting the exception bubble
into Litestar's generic 500 handler which may return a non-JSON traceback.

All tests go through the full Litestar HTTP stack via TestClient so that
framework-level routing, middleware, and serialisation are exercised.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

_ADMIN_TOKEN = "test-autonomy-admin"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def litestar_app():
    """Full Litestar app with heavy subsystems patched out.

    Returns:
        A Litestar application instance with shutdown signal registration
        suppressed so the fixture can run outside the main thread.
    """
    with (
        patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
        patch("vetinari.web.litestar_app._register_shutdown_handlers"),
    ):
        from vetinari.web.litestar_app import create_app

        yield create_app(debug=True)


# ---------------------------------------------------------------------------
# TestMemoryGovernance — all 7 memory route handlers
# ---------------------------------------------------------------------------


class TestMemoryGovernance:
    """Verify every memory route returns a structured 500 when the store raises."""

    # -- GET routes (parametrized — identical error-path structure) ----------

    @pytest.mark.parametrize(
        "url",
        [
            "/api/v1/memory",
            "/api/v1/memory/search?q=hello",
            "/api/v1/memory/sessions",
            "/api/v1/memory/stats",
        ],
        ids=["list_memories", "search_memories", "get_session_info", "get_memory_stats"],
    )
    def test_get_routes_return_500_json_when_store_raises(self, litestar_app, url: str):
        """GET memory routes return structured JSON 500 when the store raises RuntimeError.

        Args:
            litestar_app: Full Litestar application fixture.
            url: URL path (with any required query params) to GET.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.web.litestar_memory_api._get_store",
            side_effect=RuntimeError("store unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.get(url)

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code} for {url}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"
        assert isinstance(body["error"], str) and body["error"], f"Expected non-empty error message in body: {body}"

    # -- POST /api/v1/memory -------------------------------------------------

    def test_store_memory_returns_500_when_store_raises(self, litestar_app):
        """POST /api/v1/memory returns structured JSON 500 when the store raises.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.web.litestar_memory_api._get_store",
            side_effect=RuntimeError("store unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/v1/memory",
                    json={"content": "test content", "entry_type": "feedback"},
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- PUT /api/v1/memory/{entry_id} ---------------------------------------

    def test_update_memory_returns_500_when_store_raises(self, litestar_app):
        """PUT /api/v1/memory/{entry_id} returns structured JSON 500 when the store raises.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.web.litestar_memory_api._get_store",
            side_effect=RuntimeError("store unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.put(
                    "/api/v1/memory/mem-001",
                    json={"content": "updated content"},
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- DELETE /api/v1/memory/{entry_id} ------------------------------------

    def test_forget_memory_returns_500_when_store_raises(self, litestar_app):
        """DELETE /api/v1/memory/{entry_id} returns structured JSON 500 when the store raises.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.web.litestar_memory_api._get_store",
            side_effect=RuntimeError("store unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.delete(
                    "/api/v1/memory/mem-001",
                    headers={"X-Requested-With": "XMLHttpRequest"},
                )

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- Happy-path smoke tests (store works) --------------------------------

    def test_list_memories_returns_200_when_store_works(self, litestar_app):
        """GET /api/v1/memory returns 200 with items list when the store works.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        mock_store = MagicMock()
        mock_store.timeline.return_value = []

        with patch("vetinari.web.litestar_memory_api._get_store", return_value=mock_store):
            with TestClient(app=litestar_app) as client:
                resp = client.get("/api/v1/memory")

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert "items" in body, f"Expected 'items' key, got: {body}"
        assert body["items"] == []

    def test_search_memories_returns_400_when_query_missing(self, litestar_app):
        """GET /api/v1/memory/search without q= returns 400 (validation before store call).

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with TestClient(app=litestar_app) as client:
            resp = client.get("/api/v1/memory/search")

        assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error"


# ---------------------------------------------------------------------------
# TestAutonomyGovernance — all autonomy routes that need try/except
# ---------------------------------------------------------------------------


class TestAutonomyGovernance:
    """Verify every autonomy route returns a structured 500 when its subsystem raises."""

    # -- autonomy_status (GET /api/v1/autonomy/status) -----------------------

    def test_autonomy_status_returns_500_when_load_raises(self, litestar_app):
        """GET /api/v1/autonomy/status returns structured JSON 500 when policy load raises.

        Patches the module-level cache to None and the YAML loader to raise so
        that _load_policies() propagates the exception through autonomy_status.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        import vetinari.web.litestar_autonomy_api as auto_mod

        original_cache = auto_mod._policies_cache
        auto_mod._policies_cache = None
        try:
            with patch.object(auto_mod, "_load_policies", side_effect=RuntimeError("disk error")):
                with TestClient(app=litestar_app) as client:
                    resp = client.get("/api/v1/autonomy/status", headers=_ADMIN_HEADERS)
        finally:
            auto_mod._policies_cache = original_cache

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- list_pending_promotions (GET /api/v1/autonomy/promotions/pending) ---

    def test_list_pending_promotions_returns_500_when_governor_raises(self, litestar_app):
        """GET /api/v1/autonomy/promotions/pending returns 500 when governor raises.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.autonomy.governor.get_governor",
            side_effect=RuntimeError("governor unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.get("/api/v1/autonomy/promotions/pending", headers=_ADMIN_HEADERS)

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- veto_promotion (POST /api/v1/autonomy/promotions/{action_type}/veto) -

    def test_veto_promotion_returns_500_when_governor_raises(self, litestar_app):
        """POST /api/v1/autonomy/promotions/{action_type}/veto returns 500 when governor raises.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.autonomy.governor.get_governor",
            side_effect=RuntimeError("governor unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/v1/autonomy/promotions/code_generation/veto",
                    headers={**_ADMIN_HEADERS, "X-Requested-With": "XMLHttpRequest"},
                )

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- undo_action (POST /api/v1/undo/{action_id}) -------------------------

    def test_undo_action_returns_500_when_registry_raises(self, litestar_app):
        """POST /api/v1/undo/{action_id} returns 500 when rollback registry raises.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.autonomy.rollback.get_rollback_registry",
            side_effect=RuntimeError("registry unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.post(
                    "/api/v1/undo/undo_abc123def456",
                    headers={**_ADMIN_HEADERS, "X-Requested-With": "XMLHttpRequest"},
                )

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- rollback_history (GET /api/v1/autonomy/rollback/history) ------------

    def test_rollback_history_returns_500_when_registry_raises(self, litestar_app):
        """GET /api/v1/autonomy/rollback/history returns 500 when registry raises.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.autonomy.rollback.get_rollback_registry",
            side_effect=RuntimeError("registry unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.get("/api/v1/autonomy/rollback/history", headers=_ADMIN_HEADERS)

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- trust_status (GET /api/v1/autonomy/trust) ---------------------------

    def test_trust_status_returns_500_when_governor_raises(self, litestar_app):
        """GET /api/v1/autonomy/trust returns 500 when governor raises.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        with patch(
            "vetinari.autonomy.governor.get_governor",
            side_effect=RuntimeError("governor unavailable"),
        ):
            with TestClient(app=litestar_app) as client:
                resp = client.get("/api/v1/autonomy/trust", headers=_ADMIN_HEADERS)

        assert resp.status_code == 500, f"Expected 500, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert body.get("status") == "error", f"Expected status='error', got: {body}"

    # -- Happy-path smoke: autonomy/status works when policies load ----------

    def test_autonomy_status_returns_200_when_policies_load(self, litestar_app):
        """GET /api/v1/autonomy/status returns 200 with summary dict when healthy.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        import vetinari.web.litestar_autonomy_api as auto_mod
        from vetinari.web.litestar_autonomy_api import DEFAULT_POLICIES

        original_cache = auto_mod._policies_cache
        auto_mod._policies_cache = list(DEFAULT_POLICIES)
        try:
            with TestClient(app=litestar_app) as client:
                resp = client.get("/api/v1/autonomy/status", headers=_ADMIN_HEADERS)
        finally:
            auto_mod._policies_cache = original_cache

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert isinstance(body["summary"], dict), f"Expected summary dict, got: {body}"
        assert sum(body["summary"].values()) == len(DEFAULT_POLICIES), (
            f"Expected summary counts to add up to total policies, got: {body}"
        )
        assert body["total"] == len(DEFAULT_POLICIES)

    # -- Autonomy list_policies happy path -----------------------------------

    def test_list_policies_returns_200_when_policies_load(self, litestar_app):
        """GET /api/v1/autonomy/policies returns 200 with policies list when healthy.

        Args:
            litestar_app: Full Litestar application fixture.
        """
        from litestar.testing import TestClient

        import vetinari.web.litestar_autonomy_api as auto_mod
        from vetinari.web.litestar_autonomy_api import DEFAULT_POLICIES

        original_cache = auto_mod._policies_cache
        auto_mod._policies_cache = list(DEFAULT_POLICIES)
        try:
            with TestClient(app=litestar_app) as client:
                resp = client.get("/api/v1/autonomy/policies", headers=_ADMIN_HEADERS)
        finally:
            auto_mod._policies_cache = original_cache

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
        body = resp.json()
        assert len(body["policies"]) == len(DEFAULT_POLICIES), f"Expected one entry per default policy, got: {body}"
        assert body["count"] == len(DEFAULT_POLICIES)
