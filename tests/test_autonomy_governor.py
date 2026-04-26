"""Tests for vetinari.autonomy.governor — five-level policy engine and progressive trust."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vetinari.autonomy.governor import (
    AutonomyGovernor,
    PendingPromotion,
    PermissionResult,
    get_governor,
    reset_governor,
)
from vetinari.types import AutonomyLevel, AutonomyMode, DomainCareLevel, PermissionDecision

# -- Helpers ------------------------------------------------------------------


def _write_policy(tmp_path: Path, actions: dict, **extra: object) -> Path:
    """Write an autonomy_policies.yaml to tmp_path and return its path.

    Extra top-level keys (global_autonomy_mode, domain_care_levels, defaults)
    can be passed as keyword arguments.
    """
    policy_path = tmp_path / "autonomy_policies.yaml"
    data: dict = {"actions": actions, **extra}
    policy_path.write_text(yaml.dump(data), encoding="utf-8")
    return policy_path


# -- Level-to-decision mapping ------------------------------------------------


class TestLevelToDecision:
    """All five autonomy levels map to the correct PermissionDecision."""

    def test_l0_manual_returns_deny(self, tmp_path: Path) -> None:
        """L0 (manual) must never grant autonomous permission."""
        policy_path = _write_policy(tmp_path, {"action_a": {"level": "L0"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        result = governor.request_permission("action_a")
        assert result == PermissionDecision.DENY

    def test_l1_suggest_returns_defer(self, tmp_path: Path) -> None:
        """L1 (suggest) routes to the human approval queue."""
        policy_path = _write_policy(tmp_path, {"action_b": {"level": "L1"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        result = governor.request_permission("action_b")
        assert result == PermissionDecision.DEFER

    @pytest.mark.parametrize("level", ["L2", "L3", "L4"])
    def test_l2_l4_return_approve(self, tmp_path: Path, level: str) -> None:
        """L2 through L4 all approve autonomous execution."""
        policy_path = _write_policy(tmp_path, {"action_c": {"level": level}})
        governor = AutonomyGovernor(policy_path=policy_path)
        result = governor.request_permission("action_c")
        assert result == PermissionDecision.APPROVE


# -- Policy loading -----------------------------------------------------------


class TestPolicyLoading:
    """YAML policy files are loaded correctly."""

    def test_known_action_uses_configured_level(self, tmp_path: Path) -> None:
        """request_permission uses the level from YAML, not the default."""
        policy_path = _write_policy(tmp_path, {"param_tuning": {"level": "L3"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        assert governor.request_permission("param_tuning") == PermissionDecision.APPROVE

    def test_missing_policy_file_uses_default(self, tmp_path: Path) -> None:
        """When policy file is absent, all unknown actions default to L1 (DEFER)."""
        governor = AutonomyGovernor(policy_path=tmp_path / "nonexistent.yaml")
        result = governor.request_permission("any_action")
        assert result == PermissionDecision.DEFER

    def test_unknown_action_type_uses_default_level(self, tmp_path: Path) -> None:
        """An action type not in the policy file uses the configured default."""
        policy_path = _write_policy(tmp_path, {"known_action": {"level": "L4"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        # "mystery_action" is not in the YAML — should fall back to default L1
        result = governor.request_permission("mystery_action")
        assert result == PermissionDecision.DEFER


# -- max_change_pct enforcement -----------------------------------------------


class TestMaxChangePct:
    """Actions that exceed the max_change_pct limit are deferred regardless of level."""

    def test_change_within_limit_proceeds(self, tmp_path: Path) -> None:
        """A change within the configured limit is approved at L3."""
        policy_path = _write_policy(
            tmp_path,
            {"prompt_opt": {"level": "L3", "max_change_pct": 30.0}},
        )
        governor = AutonomyGovernor(policy_path=policy_path)
        result = governor.request_permission("prompt_opt", details={"change_pct": 20.0})
        assert result == PermissionDecision.APPROVE

    def test_change_exceeds_limit_defers(self, tmp_path: Path) -> None:
        """A change that exceeds max_change_pct is deferred even at L3."""
        policy_path = _write_policy(
            tmp_path,
            {"prompt_opt": {"level": "L3", "max_change_pct": 30.0}},
        )
        governor = AutonomyGovernor(policy_path=policy_path)
        result = governor.request_permission("prompt_opt", details={"change_pct": 50.0})
        assert result == PermissionDecision.DEFER


# -- Progressive Trust Engine -------------------------------------------------


class TestProgressiveTrustEngine:
    """record_outcome() and suggest_promotions() implement the trust engine."""

    def test_suggest_promotions_empty_before_threshold(self, tmp_path: Path) -> None:
        """No promotion is suggested before 50 actions."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        for _ in range(49):
            governor.record_outcome("act", success=True)
        assert governor.suggest_promotions() == []

    def test_suggest_promotions_after_95pct_success(self, tmp_path: Path) -> None:
        """95%+ success rate over 50+ actions produces a promotion suggestion."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        for _ in range(50):
            governor.record_outcome("act", success=True)
        suggestions = governor.suggest_promotions()
        assert len(suggestions) >= 1
        suggestion = next(s for s in suggestions if s.action_type == "act")
        assert suggestion.current_level == AutonomyLevel.L2_ACT_REPORT
        assert suggestion.suggested_level == AutonomyLevel.L3_ACT_LOG

    def test_auto_demotion_after_3_consecutive_failures(self, tmp_path: Path) -> None:
        """Three consecutive failures trigger an immediate demotion."""
        policy_path = _write_policy(tmp_path, {"risky_act": {"level": "L3"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        for _ in range(3):
            governor.record_outcome("risky_act", success=False)
        policy = governor.get_policy("risky_act")
        assert policy.level == AutonomyLevel.L2_ACT_REPORT

    def test_apply_promotion_resets_counters(self, tmp_path: Path) -> None:
        """apply_promotion() resets trust counters so the clock restarts at new level."""
        policy_path = _write_policy(tmp_path, {"safe_act": {"level": "L1"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        for _ in range(50):
            governor.record_outcome("safe_act", success=True)
        applied = governor.apply_promotion("safe_act")
        assert applied is True
        record = governor._trust_records["safe_act"]
        assert record.total_actions == 0
        assert record.successful_actions == 0
        assert record.consecutive_failures == 0


# -- request_permission_full --------------------------------------------------


class TestRequestPermissionFull:
    """request_permission_full() enqueues when deferred and logs all decisions."""

    def test_defer_returns_action_id(self, tmp_path: Path) -> None:
        """L1 actions produce a PermissionResult with a non-None action_id."""
        from unittest.mock import MagicMock

        policy_path = _write_policy(tmp_path, {"action_b": {"level": "L1"}})
        governor = AutonomyGovernor(policy_path=policy_path)

        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = "act_test1234"

        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = governor.request_permission_full("action_b", confidence=0.8)

        assert isinstance(result, PermissionResult)
        assert result.decision == PermissionDecision.DEFER
        assert result.action_id == "act_test1234"
        mock_queue.enqueue.assert_called_once_with("action_b", details=None, confidence=0.8)
        mock_queue.log_decision.assert_called_once()

    def test_approve_has_no_action_id(self, tmp_path: Path) -> None:
        """L3 actions produce a PermissionResult with action_id=None."""
        from unittest.mock import MagicMock

        policy_path = _write_policy(tmp_path, {"act": {"level": "L3"}})
        governor = AutonomyGovernor(policy_path=policy_path)

        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = governor.request_permission_full("act")

        assert isinstance(result, PermissionResult)
        assert result.decision == PermissionDecision.APPROVE
        assert result.action_id is None
        mock_queue.enqueue.assert_not_called()
        # Audit log must still be written for APPROVE decisions
        mock_queue.log_decision.assert_called_once()

    def test_deny_has_no_action_id(self, tmp_path: Path) -> None:
        """L0 actions produce a PermissionResult with action_id=None."""
        from unittest.mock import MagicMock

        policy_path = _write_policy(tmp_path, {"blocked_act": {"level": "L0"}})
        governor = AutonomyGovernor(policy_path=policy_path)

        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = governor.request_permission_full("blocked_act")

        assert isinstance(result, PermissionResult)
        assert result.decision == PermissionDecision.DENY
        assert result.action_id is None
        mock_queue.enqueue.assert_not_called()

    def test_audit_log_called_for_every_decision(self, tmp_path: Path) -> None:
        """log_decision() is called regardless of the permission level (APPROVE, DENY, DEFER)."""
        from unittest.mock import MagicMock

        policy_path = _write_policy(tmp_path, {"a": {"level": "L0"}, "b": {"level": "L1"}, "c": {"level": "L4"}})
        governor = AutonomyGovernor(policy_path=policy_path)

        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = "act_dummy"

        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            governor.request_permission_full("a")
            governor.request_permission_full("b")
            governor.request_permission_full("c")

        assert mock_queue.log_decision.call_count == 3


# -- Veto mechanism -----------------------------------------------------------


class TestVeto:
    """veto_promotion() and clear_veto() block and restore promotion eligibility."""

    def _make_eligible_governor(self, tmp_path: Path, action_type: str) -> AutonomyGovernor:
        """Build a governor with an action type that has met promotion criteria."""
        policy_path = _write_policy(tmp_path, {action_type: {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        for _ in range(50):
            governor.record_outcome(action_type, success=True)
        return governor

    def test_veto_blocks_suggestion(self, tmp_path: Path) -> None:
        """A vetoed action type does not appear in suggest_promotions()."""
        governor = self._make_eligible_governor(tmp_path, "safe_act")
        # Confirm eligible before veto
        assert any(s.action_type == "safe_act" for s in governor.suggest_promotions())
        governor.veto_promotion("safe_act")
        suggestions = governor.suggest_promotions()
        assert not any(s.action_type == "safe_act" for s in suggestions)

    def test_veto_blocks_apply_promotion(self, tmp_path: Path) -> None:
        """apply_promotion() returns False when the action type is vetoed."""
        governor = self._make_eligible_governor(tmp_path, "safe_act")
        governor.veto_promotion("safe_act")
        result = governor.apply_promotion("safe_act")
        assert result is False

    def test_clear_veto_restores_eligibility(self, tmp_path: Path) -> None:
        """After clear_veto(), the action type reappears in suggest_promotions()."""
        governor = self._make_eligible_governor(tmp_path, "safe_act")
        governor.veto_promotion("safe_act")
        assert governor.clear_veto("safe_act") is True
        suggestions = governor.suggest_promotions()
        assert any(s.action_type == "safe_act" for s in suggestions)

    def test_clear_veto_returns_false_when_no_veto(self, tmp_path: Path) -> None:
        """clear_veto() returns False when no veto was set for the action type."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        assert governor.clear_veto("act") is False

    def test_veto_always_returns_true(self, tmp_path: Path) -> None:
        """veto_promotion() always returns True (applying a veto never fails)."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        assert governor.veto_promotion("act") is True

    def test_get_vetoed_actions_empty_by_default(self, tmp_path: Path) -> None:
        """get_vetoed_actions() returns an empty frozenset when no vetoes are set."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        assert governor.get_vetoed_actions() == frozenset()

    def test_get_vetoed_actions_reflects_veto_calls(self, tmp_path: Path) -> None:
        """get_vetoed_actions() grows as veto_promotion() is called."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        governor.veto_promotion("model_selection")
        assert governor.get_vetoed_actions() == frozenset({"model_selection"})
        governor.veto_promotion("code_review")
        assert governor.get_vetoed_actions() == frozenset({"model_selection", "code_review"})

    def test_get_vetoed_actions_shrinks_after_clear(self, tmp_path: Path) -> None:
        """get_vetoed_actions() shrinks after clear_veto() removes an entry."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        governor.veto_promotion("model_selection")
        governor.veto_promotion("code_review")
        governor.clear_veto("model_selection")
        assert governor.get_vetoed_actions() == frozenset({"code_review"})

    def test_get_vetoed_actions_returns_frozenset(self, tmp_path: Path) -> None:
        """get_vetoed_actions() returns a frozenset (immutable snapshot)."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        governor.veto_promotion("act")
        result = governor.get_vetoed_actions()
        assert isinstance(result, frozenset)
        # Mutating the internal set afterwards must not affect the returned snapshot
        governor.clear_veto("act")
        assert "act" in result  # snapshot is unchanged


# -- rollback_on_regression ---------------------------------------------------


class TestRollbackOnRegression:
    """rollback_on_regression policy flag triggers immediate demotion on first failure."""

    def test_rollback_on_first_failure(self, tmp_path: Path) -> None:
        """When rollback_on_regression=True, a single failure demotes immediately."""
        policy_path = _write_policy(
            tmp_path,
            {"sensitive_act": {"level": "L3", "rollback_on_regression": True}},
        )
        governor = AutonomyGovernor(policy_path=policy_path)
        governor.record_outcome("sensitive_act", success=False)
        policy = governor.get_policy("sensitive_act")
        assert policy.level == AutonomyLevel.L2_ACT_REPORT

    def test_no_rollback_without_flag(self, tmp_path: Path) -> None:
        """When rollback_on_regression=False, a single failure does NOT demote."""
        policy_path = _write_policy(
            tmp_path,
            {"normal_act": {"level": "L3", "rollback_on_regression": False}},
        )
        governor = AutonomyGovernor(policy_path=policy_path)
        governor.record_outcome("normal_act", success=False)
        policy = governor.get_policy("normal_act")
        # Single failure should not demote — still at L3
        assert policy.level == AutonomyLevel.L3_ACT_LOG

    def test_no_rollback_default_behavior(self, tmp_path: Path) -> None:
        """Default policy (rollback_on_regression not set) requires 3 failures to demote."""
        policy_path = _write_policy(tmp_path, {"plain_act": {"level": "L3"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        # Two failures — should not demote yet
        governor.record_outcome("plain_act", success=False)
        governor.record_outcome("plain_act", success=False)
        assert governor.get_policy("plain_act").level == AutonomyLevel.L3_ACT_LOG
        # Third failure — should demote
        governor.record_outcome("plain_act", success=False)
        assert governor.get_policy("plain_act").level == AutonomyLevel.L2_ACT_REPORT


# -- check_pending_promotions -------------------------------------------------


class TestCheckPendingPromotions:
    """check_pending_promotions() surfaces eligible promotions for operator review."""

    def test_check_pending_applies_expired(self, tmp_path: Path) -> None:
        """check_pending_promotions() applies promotions whose veto window has expired."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        for _ in range(50):
            governor.record_outcome("act", success=True)
        # Force the veto deadline into the past so check_pending finds it
        pending_promo = governor.get_pending_promotions().get("act")
        assert pending_promo is not None
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        pending_promo.veto_deadline = past
        applied = governor.check_pending_promotions()
        assert applied == ["act"]
        # Verify the policy was actually promoted
        assert governor.get_policy("act").level == AutonomyLevel.L3_ACT_LOG

    def test_check_pending_empty_when_no_eligible(self, tmp_path: Path) -> None:
        """check_pending_promotions() returns an empty list when no actions qualify."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        # Only 10 actions — far below the 50-action threshold
        for _ in range(10):
            governor.record_outcome("act", success=True)
        assert governor.check_pending_promotions() == []

    def test_check_pending_excludes_vetoed(self, tmp_path: Path) -> None:
        """check_pending_promotions() respects vetoes — vetoed actions are excluded."""
        policy_path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        governor = AutonomyGovernor(policy_path=policy_path)
        for _ in range(50):
            governor.record_outcome("act", success=True)
        governor.veto_promotion("act")
        assert governor.check_pending_promotions() == []


# -- Singleton ----------------------------------------------------------------


class TestSingleton:
    """get_governor() returns a consistent singleton."""

    def test_get_governor_returns_instance(self) -> None:
        """get_governor() returns an AutonomyGovernor instance."""
        # Reset singleton for test isolation
        import vetinari.autonomy.governor as gov_module

        original = gov_module._governor
        gov_module._governor = None
        try:
            instance = get_governor()
            assert isinstance(instance, AutonomyGovernor)
        finally:
            gov_module._governor = original

    def test_get_governor_same_instance_on_repeat_call(self) -> None:
        """get_governor() returns the same object on repeated calls."""
        import vetinari.autonomy.governor as gov_module

        original = gov_module._governor
        gov_module._governor = None
        try:
            first = get_governor()
            second = get_governor()
            assert first is second
        finally:
            gov_module._governor = original


# -- 13.1: Global Autonomy Mode -----------------------------------------------


class TestAutonomyMode:
    """AutonomyMode enum, config loading, and per-mode risk defaults."""

    def test_governor_loads_mode_from_yaml(self, tmp_path: Path) -> None:
        """Governor reads global_autonomy_mode from YAML and stores it."""
        path = _write_policy(tmp_path, {}, global_autonomy_mode="aggressive")
        gov = AutonomyGovernor(policy_path=path)
        assert gov.get_autonomy_mode() == AutonomyMode.AGGRESSIVE

    def test_governor_defaults_to_balanced(self, tmp_path: Path) -> None:
        """When YAML omits global_autonomy_mode, governor defaults to BALANCED."""
        path = _write_policy(tmp_path, {})
        gov = AutonomyGovernor(policy_path=path)
        assert gov.get_autonomy_mode() == AutonomyMode.BALANCED

    def test_set_autonomy_mode(self, tmp_path: Path) -> None:
        """set_autonomy_mode() changes the active mode."""
        path = _write_policy(tmp_path, {})
        gov = AutonomyGovernor(policy_path=path)
        gov.set_autonomy_mode(AutonomyMode.CONSERVATIVE)
        assert gov.get_autonomy_mode() == AutonomyMode.CONSERVATIVE

    def test_conservative_mode_defaults(self, tmp_path: Path) -> None:
        """CONSERVATIVE: risky=L1, medium=L2, safe=L3."""
        path = _write_policy(tmp_path, {}, global_autonomy_mode="conservative")
        gov = AutonomyGovernor(policy_path=path)
        assert gov.get_mode_default("risky") == AutonomyLevel.L1_SUGGEST
        assert gov.get_mode_default("medium") == AutonomyLevel.L2_ACT_REPORT
        assert gov.get_mode_default("safe") == AutonomyLevel.L3_ACT_LOG

    def test_balanced_mode_defaults(self, tmp_path: Path) -> None:
        """BALANCED: risky=L2, medium=L3, safe=L4."""
        path = _write_policy(tmp_path, {}, global_autonomy_mode="balanced")
        gov = AutonomyGovernor(policy_path=path)
        assert gov.get_mode_default("risky") == AutonomyLevel.L2_ACT_REPORT
        assert gov.get_mode_default("medium") == AutonomyLevel.L3_ACT_LOG
        assert gov.get_mode_default("safe") == AutonomyLevel.L4_FULL_AUTO

    def test_aggressive_mode_defaults(self, tmp_path: Path) -> None:
        """AGGRESSIVE: risky=L3, medium=L4, safe=L4."""
        path = _write_policy(tmp_path, {}, global_autonomy_mode="aggressive")
        gov = AutonomyGovernor(policy_path=path)
        assert gov.get_mode_default("risky") == AutonomyLevel.L3_ACT_LOG
        assert gov.get_mode_default("medium") == AutonomyLevel.L4_FULL_AUTO
        assert gov.get_mode_default("safe") == AutonomyLevel.L4_FULL_AUTO

    def test_invalid_mode_in_yaml_defaults_to_balanced(self, tmp_path: Path) -> None:
        """An unrecognised mode string in YAML falls back to BALANCED."""
        path = _write_policy(tmp_path, {}, global_autonomy_mode="yolo")
        gov = AutonomyGovernor(policy_path=path)
        assert gov.get_autonomy_mode() == AutonomyMode.BALANCED


# -- 13.2: Confidence-Based Routing -------------------------------------------


class TestConfidenceRouting:
    """request_permission_full() maps confidence to autonomy level dynamically."""

    def _make_gov(self, tmp_path: Path, mode: str, actions: dict | None = None) -> AutonomyGovernor:
        path = _write_policy(tmp_path, actions or {}, global_autonomy_mode=mode)
        return AutonomyGovernor(policy_path=path)

    def test_high_confidence_balanced_approves(self, tmp_path: Path) -> None:
        """Confidence >= 0.85 in BALANCED mode -> L4 (APPROVE)."""
        gov = self._make_gov(tmp_path, "balanced", {"act": {"level": "L4"}})
        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("act", confidence=0.9)
        assert result.decision == PermissionDecision.APPROVE
        assert result.level == AutonomyLevel.L4_FULL_AUTO

    def test_low_confidence_balanced_defers(self, tmp_path: Path) -> None:
        """Confidence < 0.4 in BALANCED mode -> L1 (DEFER)."""
        gov = self._make_gov(tmp_path, "balanced", {"act": {"level": "L4"}})
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = "act_123"
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("act", confidence=0.3)
        assert result.decision == PermissionDecision.DEFER
        assert result.level == AutonomyLevel.L1_SUGGEST

    def test_medium_confidence_conservative_reports(self, tmp_path: Path) -> None:
        """Confidence 0.6-0.84 in CONSERVATIVE mode -> L2 (APPROVE, act & report)."""
        gov = self._make_gov(tmp_path, "conservative", {"act": {"level": "L4"}})
        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("act", confidence=0.7)
        assert result.decision == PermissionDecision.APPROVE
        assert result.level == AutonomyLevel.L2_ACT_REPORT

    def test_confidence_never_exceeds_policy_ceiling(self, tmp_path: Path) -> None:
        """High confidence cannot exceed the action's configured policy level."""
        # Policy is L2, but confidence=0.9 in BALANCED would suggest L4
        # Result should be min(L4, L2) = L2
        gov = self._make_gov(tmp_path, "balanced", {"limited": {"level": "L2"}})
        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("limited", confidence=0.95)
        assert result.level == AutonomyLevel.L2_ACT_REPORT

    def test_zero_confidence_uses_policy_level(self, tmp_path: Path) -> None:
        """When confidence is 0.0 (default), the policy level is used as-is."""
        gov = self._make_gov(tmp_path, "balanced", {"act": {"level": "L3"}})
        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("act", confidence=0.0)
        assert result.level == AutonomyLevel.L3_ACT_LOG
        assert result.decision == PermissionDecision.APPROVE

    @pytest.mark.parametrize(
        ("confidence", "expected_band"),
        [
            (0.95, "high"),
            (0.85, "high"),
            (0.7, "medium"),
            (0.6, "medium"),
            (0.5, "low"),
            (0.4, "low"),
            (0.3, "very_low"),
            (0.0, "very_low"),
        ],
    )
    def test_confidence_to_band_boundaries(self, confidence: float, expected_band: str) -> None:
        """_confidence_to_band maps correctly at all boundaries."""
        assert AutonomyGovernor._confidence_to_band(confidence) == expected_band


# -- 13.3: Auto-Promotion with Veto -------------------------------------------


class TestAutoPromotionVeto:
    """auto_promote(), check_pending_promotions(), and veto_promotion()."""

    def _make_eligible_gov(self, tmp_path: Path) -> AutonomyGovernor:
        """Create a governor where 'act' has 50 successes (eligible for promotion)."""
        path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        gov = AutonomyGovernor(policy_path=path)
        # Build up trust without triggering auto_promote (patch it out)
        with patch.object(gov, "auto_promote"):
            for _ in range(50):
                gov.record_outcome("act", success=True)
        return gov

    def test_auto_promote_creates_pending(self, tmp_path: Path) -> None:
        """auto_promote creates a PendingPromotion with correct fields."""
        gov = self._make_eligible_gov(tmp_path)
        pending = gov.auto_promote("act")
        assert pending is not None
        assert isinstance(pending, PendingPromotion)
        assert pending.action_type == "act"
        assert pending.current_level == AutonomyLevel.L2_ACT_REPORT
        assert pending.new_level == AutonomyLevel.L3_ACT_LOG

    def test_auto_promote_sets_veto_deadline(self, tmp_path: Path) -> None:
        """Veto deadline is ~1 hour after proposed_at."""
        gov = self._make_eligible_gov(tmp_path)
        pending = gov.auto_promote("act")
        assert pending is not None
        proposed = datetime.fromisoformat(pending.proposed_at)
        deadline = datetime.fromisoformat(pending.veto_deadline)
        delta = deadline - proposed
        assert timedelta(minutes=59) <= delta <= timedelta(minutes=61)

    def test_auto_promote_idempotent(self, tmp_path: Path) -> None:
        """Calling auto_promote twice for the same action returns None the second time."""
        gov = self._make_eligible_gov(tmp_path)
        first = gov.auto_promote("act")
        second = gov.auto_promote("act")
        assert first is not None
        assert second is None

    def test_veto_cancels_pending(self, tmp_path: Path) -> None:
        """veto_promotion removes the pending promotion but preserves trust history."""
        gov = self._make_eligible_gov(tmp_path)
        gov.auto_promote("act")
        vetoed = gov.veto_promotion("act")
        assert vetoed is True
        assert gov.get_pending_promotions() == {}
        # Trust history preserved — veto blocks without erasing history
        record = gov._trust_records["act"]
        assert record.total_actions > 0
        assert record.successful_actions > 0

    def test_veto_returns_true_for_unknown(self, tmp_path: Path) -> None:
        """veto_promotion returns True even when no pending promotion exists.

        Session 25: veto_promotion always returns True — the permanent veto
        is applied regardless of whether a pending promotion exists.
        """
        path = _write_policy(tmp_path, {})
        gov = AutonomyGovernor(policy_path=path)
        assert gov.veto_promotion("nonexistent") is True
        # Permanent veto is recorded
        assert "nonexistent" in gov.get_vetoed_actions()

    def test_check_pending_applies_expired(self, tmp_path: Path) -> None:
        """check_pending_promotions applies promotions whose veto window expired."""
        gov = self._make_eligible_gov(tmp_path)
        gov.auto_promote("act")

        # Manually backdate the veto deadline to the past
        pending = gov._pending_promotions["act"]
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        gov._pending_promotions["act"] = PendingPromotion(
            action_type=pending.action_type,
            current_level=pending.current_level,
            new_level=pending.new_level,
            proposed_at=pending.proposed_at,
            veto_deadline=past.isoformat(),
        )

        promoted = gov.check_pending_promotions()
        assert "act" in promoted
        assert gov.get_policy("act").level == AutonomyLevel.L3_ACT_LOG
        assert gov.get_pending_promotions() == {}

    def test_check_pending_skips_non_expired(self, tmp_path: Path) -> None:
        """Promotions within the veto window are not applied."""
        gov = self._make_eligible_gov(tmp_path)
        gov.auto_promote("act")
        # Deadline is in the future — should not apply
        promoted = gov.check_pending_promotions()
        assert promoted == []
        assert gov.get_policy("act").level == AutonomyLevel.L2_ACT_REPORT

    def test_vetoed_promotion_not_applied(self, tmp_path: Path) -> None:
        """A vetoed promotion is never applied even after the original deadline passes."""
        gov = self._make_eligible_gov(tmp_path)
        gov.auto_promote("act")
        gov.veto_promotion("act")
        # Even if we force-check, nothing happens
        promoted = gov.check_pending_promotions()
        assert promoted == []
        assert gov.get_policy("act").level == AutonomyLevel.L2_ACT_REPORT

    def test_record_outcome_triggers_auto_promote(self, tmp_path: Path) -> None:
        """record_outcome calls auto_promote when trust criteria are met."""
        path = _write_policy(tmp_path, {"act": {"level": "L2"}})
        gov = AutonomyGovernor(policy_path=path)
        for _ in range(50):
            gov.record_outcome("act", success=True)
        # After 50 successes, auto_promote should have created a pending promotion
        pending = gov.get_pending_promotions()
        assert "act" in pending

    def test_get_pending_promotions_returns_copy(self, tmp_path: Path) -> None:
        """get_pending_promotions returns a dict copy, not the internal reference."""
        gov = self._make_eligible_gov(tmp_path)
        gov.auto_promote("act")
        result = gov.get_pending_promotions()
        result.clear()
        # Internal state should be unaffected
        assert len(gov.get_pending_promotions()) == 1


# -- 13.5: Domain Care Levels ------------------------------------------------


class TestDomainCareLevels:
    """Per-domain care level overrides in request_permission_full()."""

    def _make_gov_with_domains(self, tmp_path: Path, mode: str = "aggressive") -> AutonomyGovernor:
        """Create a governor with domain care levels and an L4 action."""
        path = _write_policy(
            tmp_path,
            {"act": {"level": "L4"}},
            global_autonomy_mode=mode,
            domain_care_levels={
                "code-generation": "review",
                "testing": "auto",
                "deployment": "review",
            },
        )
        return AutonomyGovernor(policy_path=path)

    def test_review_domain_forces_defer_in_aggressive(self, tmp_path: Path) -> None:
        """Domain marked 'review' forces L1 (DEFER) even in AGGRESSIVE mode with high confidence."""
        gov = self._make_gov_with_domains(tmp_path, mode="aggressive")
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = "act_xyz"
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("act", confidence=0.99, domain="deployment")
        assert result.decision == PermissionDecision.DEFER
        assert result.level == AutonomyLevel.L1_SUGGEST

    def test_auto_domain_follows_confidence_routing(self, tmp_path: Path) -> None:
        """Domain marked 'auto' follows normal confidence-based routing."""
        gov = self._make_gov_with_domains(tmp_path, mode="balanced")
        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("act", confidence=0.9, domain="testing")
        assert result.decision == PermissionDecision.APPROVE

    def test_unknown_domain_uses_global_mode(self, tmp_path: Path) -> None:
        """Unknown domains fall back to global mode — no domain override."""
        gov = self._make_gov_with_domains(tmp_path, mode="balanced")
        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("act", confidence=0.9, domain="analytics")
        # No domain override, confidence=0.9 in BALANCED -> L4 (capped by policy L4)
        assert result.decision == PermissionDecision.APPROVE
        assert result.level == AutonomyLevel.L4_FULL_AUTO

    def test_no_domain_param_uses_global_mode(self, tmp_path: Path) -> None:
        """When domain is None, no domain override is applied."""
        gov = self._make_gov_with_domains(tmp_path, mode="aggressive")
        mock_queue = MagicMock()
        with patch("vetinari.autonomy.approval_queue.get_approval_queue", return_value=mock_queue):
            result = gov.request_permission_full("act", confidence=0.9)
        assert result.decision == PermissionDecision.APPROVE

    def test_get_domain_care_level_returns_review(self, tmp_path: Path) -> None:
        """get_domain_care_level returns REVIEW for configured review domains."""
        gov = self._make_gov_with_domains(tmp_path)
        assert gov.get_domain_care_level("code-generation") == DomainCareLevel.REVIEW

    def test_get_domain_care_level_returns_auto(self, tmp_path: Path) -> None:
        """get_domain_care_level returns AUTO for configured auto domains."""
        gov = self._make_gov_with_domains(tmp_path)
        assert gov.get_domain_care_level("testing") == DomainCareLevel.AUTO

    def test_get_domain_care_level_returns_none_for_unknown(self, tmp_path: Path) -> None:
        """get_domain_care_level returns None for unconfigured domains."""
        gov = self._make_gov_with_domains(tmp_path)
        assert gov.get_domain_care_level("analytics") is None

    def test_domain_care_levels_loaded_from_yaml(self, tmp_path: Path) -> None:
        """Governor loads all domain_care_levels entries from YAML config."""
        gov = self._make_gov_with_domains(tmp_path)
        assert gov.get_domain_care_level("code-generation") == DomainCareLevel.REVIEW
        assert gov.get_domain_care_level("testing") == DomainCareLevel.AUTO
        assert gov.get_domain_care_level("deployment") == DomainCareLevel.REVIEW
