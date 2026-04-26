"""Governance tests for decisions and model-discovery read routes.

Proves that:
1. The decisions API returns 400 on invalid query params (unknown type,
   out-of-range confidence_min, malformed since_iso) instead of silently
   returning empty lists.
2. The decisions API returns 503 when the journal subsystem is unavailable.
3. ``_CONFIDENCE_VALUES`` treats ``"low"`` (0.25) and ``"very_low"`` (0.0)
   as distinct values so confidence filtering is correct.
4. The model-discovery ``/api/v1/models`` and ``/api/v1/discover`` routes
   return 503 when ``_get_models_cached`` raises  -  not a degraded 200.

All tests go through the full Litestar HTTP stack via TestClient.

Patching note  -  model-discovery handlers:
    The ``_isolate_vetinari_modules`` autouse fixture (conftest.py) wipes ALL
    ``vetinari.*`` entries from ``sys.modules`` before each test file's first
    test runs, then restores the pre-test-file baseline.  This means a string-
    based patch such as ``patch("vetinari.web.litestar_models_discovery._get_models_cached")``
    resolves a DIFFERENT module object than the one the route handlers reference
    (their ``__globals__`` points to the original module dict from app-creation
    time).

    The fix: capture the live module reference inside the module-scoped ``app``
    fixture (which runs BEFORE the function-scoped ``_isolate_vetinari_modules``
    fixture fires) and expose it via the ``discovery_mod`` fixture.  Tests then
    use ``patch.object(discovery_mod, "_get_models_cached", ...)`` which
    operates on the correct object.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module when Litestar is not installed.

# Module-level dict for storing live web module references captured at
# app-creation time.  Must be populated inside the module-scoped ``app``
# fixture BEFORE ``_isolate_vetinari_modules`` runs.
_web_module_refs: dict[str, object] = {}


# ---------------------------------------------------------------------------
# App / client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Litestar app with shutdown side-effects suppressed.

    Scoped to module so the Litestar app object is only built once; each test
    creates its own TestClient context so connection state does not leak.

    Also populates the module-level ``_web_module_refs`` dict with the live
    ``litestar_models_discovery`` module object captured while it is still in
    ``sys.modules``.  This must happen here (inside the module-scoped ``app``
    fixture) because the autouse function-scoped ``_isolate_vetinari_modules``
    fixture wipes ``sys.modules`` before any test body runs.

    Side effects:
        - Populates ``_web_module_refs["discovery"]`` for use by ``discovery_mod``.

    Returns:
        A Litestar application instance.
    """
    with patch("vetinari.web.litestar_app._register_shutdown_handlers"):
        from vetinari.web.litestar_app import create_app

        litestar_app = create_app(debug=True)

    # Capture NOW  -  before _isolate_vetinari_modules runs for the first test.
    _web_module_refs["discovery"] = sys.modules.get("vetinari.web.litestar_models_discovery")
    return litestar_app


@pytest.fixture(scope="module")
def discovery_mod(app):
    """Return the live ``litestar_models_discovery`` module bound at app-creation time.

    String patches on ``vetinari.web.litestar_models_discovery._get_models_cached``
    fail after ``_isolate_vetinari_modules`` wipes ``sys.modules`` because they
    resolve a different module object than the one the route handlers reference
    via their ``__globals__``.

    The reference is captured inside the ``app`` fixture (before any
    function-scoped fixture can wipe ``sys.modules``) and stored in the
    module-level ``_web_module_refs`` dict, making it immune to test ordering.

    Args:
        app: The Litestar application instance (module-scoped).

    Returns:
        The ``litestar_models_discovery`` module object used by the app's route
        handlers.
    """
    return _web_module_refs.get("discovery")


@pytest.fixture
def client(app):
    """TestClient bound to the shared Litestar app.

    Yields:
        A live TestClient for the duration of one test.
    """
    from litestar.testing import TestClient

    with TestClient(app) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_503_error(response: object) -> None:
    """Assert that *response* is a bounded 503 with ``status: error`` envelope.

    Args:
        response: HTTP response from the TestClient.
    """
    assert response.status_code == 503, f"Expected 503, got {response.status_code}. Body: {response.text[:400]}"
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


def _assert_400_error(response: object) -> None:
    """Assert that *response* is a 400 with ``status: error`` envelope.

    Args:
        response: HTTP response from the TestClient.
    """
    assert response.status_code == 400, f"Expected 400, got {response.status_code}. Body: {response.text[:400]}"
    body = response.json()
    assert body.get("status") == "error", f"Expected envelope status='error', got {body.get('status')!r}. Body: {body}"


# ---------------------------------------------------------------------------
# _CONFIDENCE_VALUES correctness
# ---------------------------------------------------------------------------


class TestConfidenceValues:
    """Unit test for _CONFIDENCE_VALUES dict in litestar_decisions_api."""

    def test_low_and_very_low_are_distinct(self) -> None:
        """'low' (0.25) and 'very_low' (0.0) must be distinct confidence values.

        Guards against the prior bug where both mapped to 0.0, making it
        impossible to filter out very_low while keeping low records.
        """
        from vetinari.web.litestar_decisions_api import _CONFIDENCE_VALUES

        assert "low" in _CONFIDENCE_VALUES, "'low' key missing from _CONFIDENCE_VALUES"
        assert "very_low" in _CONFIDENCE_VALUES, "'very_low' key missing from _CONFIDENCE_VALUES"
        assert _CONFIDENCE_VALUES["low"] != _CONFIDENCE_VALUES["very_low"], (
            f"'low' ({_CONFIDENCE_VALUES['low']}) and 'very_low' "
            f"({_CONFIDENCE_VALUES['very_low']}) must have distinct values"
        )
        assert _CONFIDENCE_VALUES["low"] == 0.25, f"'low' should map to 0.25, got {_CONFIDENCE_VALUES['low']}"
        assert _CONFIDENCE_VALUES["very_low"] == 0.0, (
            f"'very_low' should map to 0.0, got {_CONFIDENCE_VALUES['very_low']}"
        )

    def test_high_medium_ordering(self) -> None:
        """Confidence values must be strictly ordered: high > medium > low > very_low."""
        from vetinari.web.litestar_decisions_api import _CONFIDENCE_VALUES

        assert _CONFIDENCE_VALUES["high"] > _CONFIDENCE_VALUES["medium"], "high must be greater than medium"
        assert _CONFIDENCE_VALUES["medium"] > _CONFIDENCE_VALUES["low"], "medium must be greater than low"
        assert _CONFIDENCE_VALUES["low"] > _CONFIDENCE_VALUES["very_low"], "low must be greater than very_low"


# ---------------------------------------------------------------------------
# Decisions API  -  query param validation
# ---------------------------------------------------------------------------


class TestDecisionsUnknownType:
    """GET /api/v1/decisions?type=<unknown>  -  must return 400, not empty 200."""

    def test_unknown_type_returns_400(self, client: object) -> None:
        """Unknown decision type query param returns 400  -  not an empty list."""
        resp = client.get("/api/v1/decisions?type=nonexistent_decision_type_xyz")
        _assert_400_error(resp)
        body = resp.json()
        # Error message must mention the invalid type
        assert "nonexistent_decision_type_xyz" in body.get("error", ""), (
            "Error message should echo back the invalid type value"
        )

    def test_valid_type_does_not_return_400(self, client: object) -> None:
        """Valid decision types do not trigger 400 (they may 503 if journal absent)."""
        # Mock the journal so we get a real response rather than 503
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = []
        with patch("vetinari.observability.decision_journal.get_decision_journal", return_value=mock_journal):
            resp = client.get("/api/v1/decisions?type=model_selection")
        assert resp.status_code == 200, f"Valid type should return 200. Got: {resp.text[:400]}"
        body = resp.json()
        assert body["status"] == "ok"
        assert body["decisions"] == []
        assert body["total"] == 0


class TestDecisionsConfidenceMin:
    """GET /api/v1/decisions?confidence_min=<value>  -  validation."""

    def test_non_float_confidence_min_returns_400(self, client: object) -> None:
        """Non-numeric confidence_min returns 400."""
        resp = client.get("/api/v1/decisions?confidence_min=high")
        _assert_400_error(resp)

    def test_confidence_min_above_1_returns_400(self, client: object) -> None:
        """confidence_min > 1.0 returns 400."""
        resp = client.get("/api/v1/decisions?confidence_min=1.5")
        _assert_400_error(resp)

    def test_confidence_min_below_0_returns_400(self, client: object) -> None:
        """confidence_min < 0.0 returns 400."""
        resp = client.get("/api/v1/decisions?confidence_min=-0.1")
        _assert_400_error(resp)

    @pytest.mark.parametrize("value", ["0.0", "0.5", "1.0", "0.25"])
    def test_valid_confidence_min_does_not_return_400(self, client: object, value: str) -> None:
        """Valid confidence_min float values in [0.0, 1.0] do not trigger 400."""
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = []
        with patch("vetinari.observability.decision_journal.get_decision_journal", return_value=mock_journal):
            resp = client.get(f"/api/v1/decisions?confidence_min={value}")
        assert resp.status_code != 400, f"Valid confidence_min={value!r} should not return 400. Body: {resp.text[:400]}"


class TestDecisionsSinceIso:
    """GET /api/v1/decisions?since_iso=<value>  -  ISO 8601 validation."""

    def test_invalid_since_iso_returns_400(self, client: object) -> None:
        """Malformed since_iso returns 400."""
        resp = client.get("/api/v1/decisions?since_iso=not-a-date")
        _assert_400_error(resp)

    def test_partial_date_returns_400(self, client: object) -> None:
        """Partial date string (no time component) that fromisoformat rejects returns 400."""
        # Python's fromisoformat accepts 'YYYY-MM-DD' since 3.7  -  use a clearly invalid format
        resp = client.get("/api/v1/decisions?since_iso=2024/01/15")
        _assert_400_error(resp)

    def test_valid_iso_does_not_return_400(self, client: object) -> None:
        """Valid ISO 8601 datetime does not trigger 400."""
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = []
        with patch("vetinari.observability.decision_journal.get_decision_journal", return_value=mock_journal):
            resp = client.get("/api/v1/decisions?since_iso=2024-01-15T10:30:00")
        assert resp.status_code != 400, f"Valid ISO datetime should not return 400. Body: {resp.text[:400]}"


# ---------------------------------------------------------------------------
# Decisions API  -  journal subsystem unavailable
# ---------------------------------------------------------------------------


class TestDecisionsJournalUnavailable:
    """GET /api/v1/decisions  -  503 when journal subsystem unavailable."""

    def test_journal_module_unavailable_returns_503(self, client: object) -> None:
        """Decisions returns 503 when decision_journal module cannot be imported."""
        import sys

        orig = sys.modules.get("vetinari.observability.decision_journal")
        sys.modules["vetinari.observability.decision_journal"] = None  # type: ignore[assignment]
        try:
            resp = client.get("/api/v1/decisions")
        finally:
            if orig is None:
                sys.modules.pop("vetinari.observability.decision_journal", None)
            else:
                sys.modules["vetinari.observability.decision_journal"] = orig
        _assert_503_error(resp)

    def test_get_decision_journal_raises_returns_503(self, client: object) -> None:
        """Decisions returns 503 when get_decision_journal() raises."""
        with patch(
            "vetinari.observability.decision_journal.get_decision_journal",
            side_effect=RuntimeError("SQLite locked"),
        ):
            resp = client.get("/api/v1/decisions")
        _assert_503_error(resp)

    def test_journal_unavailable_not_empty_list_200(self, client: object) -> None:
        """Decisions returns 503, not ``{"decisions":[], "total":0}`` on failure.

        Guards against the 'unavailable-dependency pass-through' anti-pattern.
        """
        with patch(
            "vetinari.observability.decision_journal.get_decision_journal",
            side_effect=OSError("journal file deleted"),
        ):
            resp = client.get("/api/v1/decisions")
        # Must not silently return a 200 with empty list
        assert resp.status_code != 200, "Journal unavailability must NOT be masked as a 200 empty list"
        _assert_503_error(resp)


# ---------------------------------------------------------------------------
# Model-discovery routes
# ---------------------------------------------------------------------------


class TestApiModels:
    """GET /api/v1/models  -  503 when _get_models_cached raises."""

    def test_models_cached_raises_returns_503(self, client: object, discovery_mod: object) -> None:
        """Discovery /api/v1/models returns 503 when _get_models_cached raises.

        Uses ``patch.object`` on the live module reference captured at
        app-creation time.  String patches resolve a different module object
        after ``_isolate_vetinari_modules`` wipes ``sys.modules``.
        """
        with patch.object(
            discovery_mod,
            "_get_models_cached",
            side_effect=RuntimeError("model scan failed"),
        ):
            resp = client.get("/api/v1/models")
        _assert_503_error(resp)

    def test_models_cached_raises_not_degraded_200(self, client: object, discovery_mod: object) -> None:
        """Discovery /api/v1/models must not return 200 with empty models on failure.

        Guards against the 'unavailable-dependency pass-through' anti-pattern
        where ``{"models":[], "count":0, "cached":true}`` hides the failure.
        """
        with patch.object(
            discovery_mod,
            "_get_models_cached",
            side_effect=RuntimeError("model scan failed"),
        ):
            resp = client.get("/api/v1/models")
        assert resp.status_code != 200, "Discovery failure must NOT be masked as a 200 empty-models response"
        _assert_503_error(resp)

    def test_models_cached_success_returns_200(self, client: object, discovery_mod: object) -> None:
        """Discovery /api/v1/models returns 200 when models are available.

        The response is wrapped in the standard success envelope  -  data lives
        at ``body["data"]["count"]``, not at the top level.
        """
        with patch.object(
            discovery_mod,
            "_get_models_cached",
            return_value=[{"name": "test-model", "capabilities": ["chat"]}],
        ):
            resp = client.get("/api/v1/models")
        assert resp.status_code == 200, f"Expected 200 on success, got {resp.status_code}. Body: {resp.text[:400]}"
        body = resp.json()
        data = body.get("data", {})
        assert data.get("count") == 1, f"Expected count=1, got {data.get('count')}. Body: {body}"


class TestApiDiscover:
    """GET /api/v1/discover  -  503 when _get_models_cached raises."""

    def test_discover_raises_returns_503(self, client: object, discovery_mod: object) -> None:
        """Discover endpoint returns 503 when _get_models_cached raises.

        Uses ``patch.object`` on the live module reference captured at
        app-creation time.  String patches resolve a different module object
        after ``_isolate_vetinari_modules`` wipes ``sys.modules``.
        """
        with patch.object(
            discovery_mod,
            "_get_models_cached",
            side_effect=RuntimeError("GGUF dir unreadable"),
        ):
            resp = client.get("/api/v1/discover")
        _assert_503_error(resp)

    def test_discover_raises_not_degraded_200(self, client: object, discovery_mod: object) -> None:
        """Discover endpoint must not return 200 with empty models on failure.

        Guards against the 'unavailable-dependency pass-through' anti-pattern
        where ``{"discovered":0, "models":[], "status":"ok"}`` hides the failure.
        """
        with patch.object(
            discovery_mod,
            "_get_models_cached",
            side_effect=RuntimeError("GGUF dir unreadable"),
        ):
            resp = client.get("/api/v1/discover")
        assert resp.status_code != 200, "Discovery failure must NOT be masked as a 200 empty-discovered response"
        _assert_503_error(resp)

    def test_discover_success_returns_200(self, client: object, discovery_mod: object) -> None:
        """Discover endpoint returns 200 when models are found."""
        with patch.object(
            discovery_mod,
            "_get_models_cached",
            return_value=[{"name": "gguf-model-q4", "capabilities": ["code_gen"]}],
        ):
            resp = client.get("/api/v1/discover")
        assert resp.status_code == 200, f"Expected 200 on success, got {resp.status_code}. Body: {resp.text[:400]}"
        body = resp.json()
        assert body.get("discovered") == 1, f"Expected discovered=1, got {body.get('discovered')}. Body: {body}"
        assert body.get("status") == "ok", f"Expected status='ok', got {body.get('status')!r}"
