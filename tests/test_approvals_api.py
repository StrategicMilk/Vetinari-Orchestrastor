"""Tests for vetinari.web.approvals_api — approval queue and trust API handlers.

Covers:
- GET /api/v1/approvals/pending returns list of pending actions
- POST /api/v1/approvals/{action_id}/approve returns success/failure
- POST /api/v1/approvals/{action_id}/reject with reason returns success/failure
- GET /api/v1/decisions/log returns audit log entries
- GET /api/v1/decisions/history returns unified decision history from both stores
- GET /api/v1/autonomy/trust-status returns trust metric dict
- GET /api/v1/autonomy/promotions returns promotion suggestion list
- POST /api/v1/autonomy/promote/{action_type} applies or rejects promotion
- POST /api/v1/autonomy/veto/{action_type} sets a promotion veto
- DELETE /api/v1/autonomy/veto/{action_type} clears a promotion veto
- GET /api/v1/autonomy/vetoes returns the currently vetoed action types
- admin_guard is enforced on all GET routes (unauthenticated client gets 403)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vetinari.autonomy.approval_queue import DecisionLogEntry, PendingAction
from vetinari.autonomy.governor import PromotionSuggestion
from vetinari.observability.decision_journal import DecisionRecord
from vetinari.types import AutonomyLevel, ConfidenceLevel, DecisionType, PermissionDecision
from vetinari.web.approvals_api import create_approvals_handlers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pending(action_id: str = "act_abc123", action_type: str = "param_tuning") -> PendingAction:
    """Create a minimal PendingAction for test assertions."""
    return PendingAction(
        action_id=action_id,
        action_type=action_type,
        details={"key": "value"},
        confidence=0.85,
        status="pending",
        created_at="2026-04-01T00:00:00+00:00",
    )


def _make_log_entry(
    action_id: str = "dec_xyz789",
    action_type: str = "model_substitution",
) -> DecisionLogEntry:
    """Create a minimal DecisionLogEntry for test assertions."""
    return DecisionLogEntry(
        action_id=action_id,
        action_type=action_type,
        autonomy_level=AutonomyLevel.L3_ACT_LOG.value,
        decision=PermissionDecision.APPROVE.value,
        confidence=0.9,
        outcome="success",
        timestamp="2026-04-01T00:00:00+00:00",
    )


def _make_promotion(action_type: str = "prompt_optimization") -> PromotionSuggestion:
    """Create a minimal PromotionSuggestion for test assertions."""
    return PromotionSuggestion(
        action_type=action_type,
        current_level=AutonomyLevel.L2_ACT_REPORT,
        suggested_level=AutonomyLevel.L3_ACT_LOG,
        success_rate=0.97,
        total_actions=55,
    )


def _make_decision_record(
    decision_id: str = "dec_journal001",
    decision_type: DecisionType = DecisionType.ROUTING,
    confidence_score: float = 0.75,
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM,
) -> DecisionRecord:
    """Create a minimal DecisionRecord for test assertions."""
    return DecisionRecord(
        decision_id=decision_id,
        decision_type=decision_type,
        description="Route to Worker agent",
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        confidence_factors={"mean_logprob": -0.9},
        action_taken="routed_to_worker",
        context={},
        outcome="",
        timestamp="2026-04-01T01:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def approvals_app() -> Any:
    """Minimal Litestar app with only the approvals routes, guard patched to pass.

    Patches admin_guard to a no-op so tests focus on handler logic rather than
    authentication.
    """
    from litestar import Litestar

    def _pass_guard(connection: Any, _handler: Any = None, **kwargs: Any) -> None:
        return None

    with patch("vetinari.web.approvals_api.admin_guard", _pass_guard):
        handlers = create_approvals_handlers()
    return Litestar(route_handlers=handlers)


@pytest.fixture
def unauthenticated_approvals_app(monkeypatch: Any) -> Any:
    """Minimal Litestar app with approvals routes and the REAL admin_guard active.

    Used to prove that admin_guard actually enforces 401 on GET routes when
    no credentials are provided.

    Sets VETINARI_ADMIN_TOKEN so the guard does NOT fall back to IP-based auth,
    forcing the test to provide an explicit token header (which it doesn't).
    """
    from litestar import Litestar

    # Set a fake admin token to disable IP-based fallback auth
    monkeypatch.setenv("VETINARI_ADMIN_TOKEN", "test-admin-secret-key")

    handlers = create_approvals_handlers()
    return Litestar(route_handlers=handlers)


# ---------------------------------------------------------------------------
# Smoke test: create_approvals_handlers returns handlers when Litestar present
# ---------------------------------------------------------------------------


class TestCreateHandlers:
    """create_approvals_handlers() returns the expected handler set."""

    def test_returns_eleven_handlers_when_litestar_available(self) -> None:
        """Eleven route handlers are returned when Litestar is installed."""
        handlers = create_approvals_handlers()
        if handlers:
            assert len(handlers) == 11

    def test_returns_empty_list_when_litestar_unavailable(self, monkeypatch: Any) -> None:
        """An empty list is returned gracefully when Litestar is missing."""
        import vetinari.web.approvals_api as mod

        monkeypatch.setattr(mod, "_LITESTAR_AVAILABLE", False)
        handlers = mod.create_approvals_handlers()
        assert handlers == []


# ---------------------------------------------------------------------------
# admin_guard enforcement: all GET routes must return 403 without credentials
# ---------------------------------------------------------------------------


class TestAdminGuardEnforcement:
    """Prove that admin_guard is mounted on every GET route.

    Each test hits the real app stack with no auth token and asserts 401.
    Litestar maps ``NotAuthorizedException`` to HTTP 401.
    """

    GET_ROUTES = [
        "/api/v1/approvals/pending",
        "/api/v1/decisions/log",
        "/api/v1/decisions/history",
        "/api/v1/autonomy/trust-status",
        "/api/v1/autonomy/promotions",
        "/api/v1/autonomy/vetoes",
    ]

    @pytest.mark.parametrize("path", GET_ROUTES)
    def test_get_route_requires_auth(self, unauthenticated_approvals_app: Any, path: str) -> None:
        """GET request without auth credentials must return 401 Unauthorized.

        Litestar translates ``NotAuthorizedException`` raised by ``admin_guard``
        to HTTP 401.  The guard is wired into every GET route handler via the
        ``guards=[admin_guard]`` parameter on each ``@get`` decorator.
        """
        from litestar.testing import TestClient

        with TestClient(app=unauthenticated_approvals_app) as client:
            response = client.get(path)
        assert response.status_code == 401, (
            f"Expected 401 for {path} without auth, got {response.status_code}"
        )


# ---------------------------------------------------------------------------
# GET /api/v1/approvals/pending
# ---------------------------------------------------------------------------


class TestListPendingApprovals:
    """list_pending_approvals() delegates to the approval queue singleton."""

    def test_returns_list_of_pending_dicts(self, approvals_app: Any) -> None:
        """Handler returns a list with the correct fields for each pending action."""
        from litestar.testing import TestClient

        pending = [_make_pending("act_001", "param_tuning"), _make_pending("act_002", "model_swap")]
        mock_queue = MagicMock()
        mock_queue.get_pending.return_value = pending

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.get("/api/v1/approvals/pending")

        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["action_id"] == "act_001"
        assert result[0]["action_type"] == "param_tuning"
        assert result[1]["action_id"] == "act_002"

    def test_returns_empty_list_when_nothing_pending(self, approvals_app: Any) -> None:
        """Handler returns an empty list when the queue has no pending actions."""
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.get_pending.return_value = []

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.get("/api/v1/approvals/pending")

        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# POST /api/v1/approvals/{action_id}/approve
# ---------------------------------------------------------------------------


class TestApproveAction:
    """approve_action() calls queue.approve() and returns the right shape."""

    def test_approve_success_returns_true(self, approvals_app: Any) -> None:
        """Returns success=True and the action_id when approve() succeeds."""
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.approve.return_value = True

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.post("/api/v1/approvals/act_abc123/approve")

        assert response.status_code == 201
        result = response.json()
        mock_queue.approve.assert_called_once_with("act_abc123")
        assert result["success"] is True
        assert result["action_id"] == "act_abc123"

    def test_approve_failure_returns_404(self, approvals_app: Any) -> None:
        """Returns HTTP 404 when approve() returns False (not found/not pending)."""
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.approve.return_value = False

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.post("/api/v1/approvals/act_missing/approve")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/approvals/{action_id}/reject
# ---------------------------------------------------------------------------


class TestRejectAction:
    """reject_action() calls queue.reject() with reason and returns the right shape."""

    def test_reject_success_returns_true(self, approvals_app: Any) -> None:
        """Returns success=True when reject() succeeds."""
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.reject.return_value = True

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.post(
                    "/api/v1/approvals/act_abc123/reject",
                    json={"reason": "too risky"},
                )

        assert response.status_code == 201
        result = response.json()
        mock_queue.reject.assert_called_once_with("act_abc123", reason="too risky")
        assert result["success"] is True

    def test_reject_no_reason_body_defaults_to_empty(self, approvals_app: Any) -> None:
        """reject() is called with reason='' when no body is provided."""
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.reject.return_value = True

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.post("/api/v1/approvals/act_abc123/reject")

        assert response.status_code == 201
        mock_queue.reject.assert_called_once_with("act_abc123", reason="")

    def test_reject_failure_returns_404(self, approvals_app: Any) -> None:
        """Returns HTTP 404 when the action is not found or not pending."""
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.reject.return_value = False

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.post("/api/v1/approvals/act_missing/reject")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/decisions/log
# ---------------------------------------------------------------------------


class TestGetDecisionLog:
    """get_decision_log() returns audit log entries with correct shape."""

    def test_returns_entries_and_total(self, approvals_app: Any) -> None:
        """Handler returns total count and a list of entry dicts."""
        from litestar.testing import TestClient

        entries = [_make_log_entry("dec_001"), _make_log_entry("dec_002")]
        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = entries

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.get("/api/v1/decisions/log")

        assert response.status_code == 200
        result = response.json()
        assert result["total"] == 2
        assert len(result["entries"]) == 2
        assert result["entries"][0]["action_id"] == "dec_001"

    def test_passes_action_type_filter(self, approvals_app: Any) -> None:
        """action_type query parameter is forwarded to get_decision_log()."""
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = []

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
                response = client.get(
                    "/api/v1/decisions/log",
                    params={"action_type": "model_substitution", "limit": "50"},
                )

        assert response.status_code == 200
        mock_queue.get_decision_log.assert_called_once_with(action_type="model_substitution", limit=50)


# ---------------------------------------------------------------------------
# GET /api/v1/autonomy/trust-status
# ---------------------------------------------------------------------------


class TestGetTrustStatus:
    """get_trust_status() returns governor trust metrics."""

    def test_returns_trust_status_dict(self, approvals_app: Any) -> None:
        """Handler returns action_count and nested trust_status dict."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.get_trust_status.return_value = {
            "param_tuning": {
                "total_actions": 60,
                "success_rate": 0.97,
                "current_level": "L3",
            }
        }

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.get("/api/v1/autonomy/trust-status")

        assert response.status_code == 200
        result = response.json()
        assert result["action_count"] == 1
        assert "param_tuning" in result["trust_status"]
        assert result["trust_status"]["param_tuning"]["success_rate"] == 0.97


# ---------------------------------------------------------------------------
# GET /api/v1/autonomy/promotions
# ---------------------------------------------------------------------------


class TestGetPromotions:
    """get_promotions() returns promotion suggestions."""

    def test_returns_promotions_list(self, approvals_app: Any) -> None:
        """Handler returns count and list of promotion suggestion dicts."""
        from litestar.testing import TestClient

        suggestions = [_make_promotion("prompt_opt"), _make_promotion("model_select")]
        mock_governor = MagicMock()
        mock_governor.suggest_promotions.return_value = suggestions

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.get("/api/v1/autonomy/promotions")

        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 2
        action_types = {p["action_type"] for p in result["promotions"]}
        assert action_types == {"prompt_opt", "model_select"}
        assert result["promotions"][0]["current_level"] == AutonomyLevel.L2_ACT_REPORT.value

    def test_returns_empty_when_none_eligible(self, approvals_app: Any) -> None:
        """Handler returns count=0 and empty list when no promotions are ready."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.suggest_promotions.return_value = []

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.get("/api/v1/autonomy/promotions")

        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 0
        assert result["promotions"] == []


# ---------------------------------------------------------------------------
# POST /api/v1/autonomy/promote/{action_type}
# ---------------------------------------------------------------------------


class TestApplyPromotion:
    """apply_promotion() calls governor.apply_promotion() and returns correct shape."""

    def test_successful_promotion_returns_true(self, approvals_app: Any) -> None:
        """Returns success=True and action_type when promotion is applied."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.apply_promotion.return_value = True

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.post("/api/v1/autonomy/promote/param_tuning")

        assert response.status_code == 201
        result = response.json()
        mock_governor.apply_promotion.assert_called_once_with("param_tuning")
        assert result["success"] is True
        assert result["action_type"] == "param_tuning"

    def test_ineligible_promotion_returns_422(self, approvals_app: Any) -> None:
        """Returns HTTP 422 when the action type is not eligible for promotion."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.apply_promotion.return_value = False

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.post("/api/v1/autonomy/promote/new_action_type")

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/decisions/history
# ---------------------------------------------------------------------------


class TestGetDecisionHistory:
    """get_decision_history() merges approval-queue and journal entries."""

    def test_returns_entries_from_both_sources(self, approvals_app: Any) -> None:
        """Entries from both the approval queue and decision journal appear in output."""
        from litestar.testing import TestClient

        aq_entry = _make_log_entry("dec_aq1", "param_tuning")
        journal_entry = _make_decision_record("dec_j1", DecisionType.ROUTING, 0.75, ConfidenceLevel.MEDIUM)

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = [aq_entry]
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = [journal_entry]

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                response = client.get("/api/v1/decisions/history")

        assert response.status_code == 200
        result = response.json()
        assert result["total"] == 2
        sources = {e["source"] for e in result["entries"]}
        assert sources == {"autonomy", "pipeline"}

    def test_autonomy_entry_has_confidence_level(self, approvals_app: Any) -> None:
        """Approval-queue entries have confidence_level computed from the stored float."""
        from litestar.testing import TestClient

        aq_entry = _make_log_entry("dec_aq1", "param_tuning")
        # confidence=0.9 in _make_log_entry → HIGH

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = [aq_entry]
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = []

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                response = client.get("/api/v1/decisions/history")

        assert response.status_code == 200
        result = response.json()
        entry = result["entries"][0]
        assert entry["confidence_level"] == ConfidenceLevel.HIGH.value
        assert entry["source"] == "autonomy"
        assert entry["is_fallback"] is False

    def test_zero_confidence_flagged_as_fallback(self, approvals_app: Any) -> None:
        """An approval-queue entry with confidence=0.0 is flagged as is_fallback=True."""
        from litestar.testing import TestClient

        aq_entry = DecisionLogEntry(
            action_id="dec_fallback",
            action_type="model_sub",
            autonomy_level=AutonomyLevel.L1_SUGGEST.value,
            decision=PermissionDecision.DEFER.value,
            confidence=0.0,
            outcome="",
            timestamp="2026-04-01T00:00:00+00:00",
        )

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = [aq_entry]
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = []

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                response = client.get("/api/v1/decisions/history")

        assert response.status_code == 200
        result = response.json()
        entry = result["entries"][0]
        assert entry["is_fallback"] is True
        assert entry["confidence_level"] == ConfidenceLevel.VERY_LOW.value

    def test_entries_sorted_most_recent_first(self, approvals_app: Any) -> None:
        """Merged entries are returned in descending timestamp order."""
        from litestar.testing import TestClient

        older_entry = DecisionLogEntry(
            action_id="dec_old",
            action_type="param_tuning",
            autonomy_level=AutonomyLevel.L3_ACT_LOG.value,
            decision=PermissionDecision.APPROVE.value,
            confidence=0.9,
            outcome="",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        newer_journal_with_ts = DecisionRecord(
            decision_id="dec_new",
            decision_type=DecisionType.ROUTING,
            description="",
            confidence_score=0.75,
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_factors={"mean_logprob": -0.9},
            action_taken="routed",
            context={},
            outcome="",
            timestamp="2026-04-01T00:00:00+00:00",
        )

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = [older_entry]
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = [newer_journal_with_ts]

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                response = client.get("/api/v1/decisions/history")

        assert response.status_code == 200
        result = response.json()
        assert result["entries"][0]["id"] == "dec_new"
        assert result["entries"][1]["id"] == "dec_old"

    def test_limit_applies_to_merged_result(self, approvals_app: Any) -> None:
        """The limit parameter caps the total number of entries returned."""
        from litestar.testing import TestClient

        aq_entries = [_make_log_entry(f"dec_{i}", "act") for i in range(5)]
        journal_entries = [_make_decision_record(f"dec_j{i}", confidence_score=0.75) for i in range(5)]

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = aq_entries
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = journal_entries

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                response = client.get("/api/v1/decisions/history", params={"limit": "3"})

        assert response.status_code == 200
        result = response.json()
        assert result["total"] == 3
        assert len(result["entries"]) == 3

    def test_negative_limit_clamped_to_one(self, approvals_app: Any) -> None:
        """Negative limit is clamped to 1 (minimum).

        The endpoint documents that limit is "clamped to 1-500". A negative
        or zero limit is invalid but should not crash — it should clamp to 1.
        """
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = [_make_log_entry("dec_001")]
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = []

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                response = client.get("/api/v1/decisions/history", params={"limit": "-5"})

        assert response.status_code == 200
        result = response.json()
        # Limit was clamped to 1, so we get at most 1 entry
        assert len(result["entries"]) == 1
        # Verify the mock was called with limit=1 (clamped)
        mock_queue.get_decision_log.assert_called_once()
        call_kwargs = mock_queue.get_decision_log.call_args[1]
        assert call_kwargs["limit"] == 1

    def test_overlarge_limit_clamped_to_500(self, approvals_app: Any) -> None:
        """Limit larger than 500 is clamped to 500 (maximum).

        The endpoint documents that limit is "clamped to 1-500". A limit
        larger than 500 is clamped to prevent unbounded history dumps.
        """
        from litestar.testing import TestClient

        # Create 10 entries to exceed the clamped limit
        aq_entries = [_make_log_entry(f"dec_{i}", "act") for i in range(6)]
        journal_entries = [_make_decision_record(f"dec_j{i}", confidence_score=0.75) for i in range(4)]

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = aq_entries
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = journal_entries

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                # Request limit=9999 (much larger than max of 500)
                response = client.get("/api/v1/decisions/history", params={"limit": "9999"})

        assert response.status_code == 200
        result = response.json()
        # Limit was clamped to 500, and we have 10 total entries, so we get all 10
        assert result["total"] == 10
        # Verify the mocks were called with limit=500 (clamped)
        mock_queue.get_decision_log.assert_called_once()
        call_kwargs = mock_queue.get_decision_log.call_args[1]
        assert call_kwargs["limit"] == 500

    def test_unknown_decision_type_returns_400(self, approvals_app: Any) -> None:
        """Invalid decision_type parameter returns HTTP 400.

        The endpoint accepts optional decision_type filter as an enum. If an
        unrecognized decision_type value is passed, the endpoint rejects the
        request with 400 Bad Request.
        """
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = []
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = []

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                # Pass an invalid decision_type that doesn't match any DecisionType enum value
                response = client.get("/api/v1/decisions/history", params={"decision_type": "invalid_type"})

        assert response.status_code == 400
        result = response.json()
        # Error response should include the invalid value in the message (check both 'error' and 'detail' keys)
        error_msg = result.get("error", "") or result.get("detail", "")
        assert "invalid_type" in error_msg.lower()

    def test_unknown_confidence_level_returns_400(self, approvals_app: Any) -> None:
        """Invalid confidence_level parameter returns HTTP 400.

        The endpoint accepts optional confidence_level filter as an enum. If an
        unrecognized confidence_level value is passed, the endpoint rejects the
        request with 400 Bad Request.
        """
        from litestar.testing import TestClient

        mock_queue = MagicMock()
        mock_queue.get_decision_log.return_value = []
        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = []

        with TestClient(app=approvals_app) as client:
            with (
                patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue),
                patch(
                    "vetinari.observability.decision_journal.get_decision_journal",
                    return_value=mock_journal,
                ),
            ):
                # Pass an invalid confidence_level that doesn't match any ConfidenceLevel enum value
                response = client.get("/api/v1/decisions/history", params={"confidence_level": "unknown_level"})

        assert response.status_code == 400
        result = response.json()
        # Error response should include the invalid value in the message (check both 'error' and 'detail' keys)
        error_msg = result.get("error", "") or result.get("detail", "")
        assert "unknown_level" in error_msg.lower()


# ---------------------------------------------------------------------------
# POST /api/v1/autonomy/veto/{action_type}
# ---------------------------------------------------------------------------


class TestVetoPromotion:
    """veto_promotion() calls governor.veto_promotion() and returns correct shape."""

    def test_veto_returns_success_and_vetoed_true(self, approvals_app: Any) -> None:
        """Returns success=True and vetoed=True when veto_promotion() succeeds."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.veto_promotion.return_value = True

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.post("/api/v1/autonomy/veto/model_selection")

        assert response.status_code == 201
        result = response.json()
        mock_governor.veto_promotion.assert_called_once_with("model_selection")
        assert result["success"] is True
        assert result["action_type"] == "model_selection"
        assert result["vetoed"] is True


# ---------------------------------------------------------------------------
# DELETE /api/v1/autonomy/veto/{action_type}
# ---------------------------------------------------------------------------


class TestClearVeto:
    """clear_veto() calls governor.clear_veto() and returns correct shape."""

    def test_clear_existing_veto_returns_success(self, approvals_app: Any) -> None:
        """Returns success=True and vetoed=False when a veto is cleared."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.clear_veto.return_value = True

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.delete("/api/v1/autonomy/veto/model_selection")

        assert response.status_code == 200
        result = response.json()
        mock_governor.clear_veto.assert_called_once_with("model_selection")
        assert result["success"] is True
        assert result["vetoed"] is False

    def test_clear_nonexistent_veto_returns_404(self, approvals_app: Any) -> None:
        """Returns HTTP 404 when no veto exists for the given action type."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.clear_veto.return_value = False

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.delete("/api/v1/autonomy/veto/unknown_action")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/autonomy/vetoes
# ---------------------------------------------------------------------------


class TestListVetoes:
    """list_vetoes() returns the currently vetoed action types."""

    def test_returns_sorted_vetoed_types(self, approvals_app: Any) -> None:
        """Handler returns sorted list of vetoed action types and count."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.get_vetoed_actions.return_value = frozenset({"model_selection", "code_review"})

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.get("/api/v1/autonomy/vetoes")

        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 2
        assert result["vetoed_action_types"] == ["code_review", "model_selection"]

    def test_returns_empty_when_no_vetoes(self, approvals_app: Any) -> None:
        """Handler returns count=0 and empty list when no vetoes are active."""
        from litestar.testing import TestClient

        mock_governor = MagicMock()
        mock_governor.get_vetoed_actions.return_value = frozenset()

        with TestClient(app=approvals_app) as client:
            with patch("vetinari.autonomy.governor.get_governor", return_value=mock_governor):
                response = client.get("/api/v1/autonomy/vetoes")

        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 0
        assert result["vetoed_action_types"] == []
