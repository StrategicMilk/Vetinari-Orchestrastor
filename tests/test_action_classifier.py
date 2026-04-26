"""Tests for vetinari.analytics.action_classifier.

Covers the three governance tiers, the override mechanism, and the
default behaviour for unknown action types.
"""

from __future__ import annotations

import threading

import pytest

from vetinari.analytics.action_classifier import (
    ALLOW_ACTIONS,
    DENY_ACTIONS,
    FOREMAN_REVIEW_ACTIONS,
    ActionClassification,
    ActionClassifier,
    get_action_classifier,
    reset_action_classifier,
)
from vetinari.types import ActionTier, AgentType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier() -> ActionClassifier:
    """Fresh ActionClassifier for each test."""
    return ActionClassifier()


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Ensure the module singleton is clean before and after every test."""
    reset_action_classifier()
    yield
    reset_action_classifier()


# ---------------------------------------------------------------------------
# ActionClassification dataclass
# ---------------------------------------------------------------------------


class TestActionClassification:
    def test_frozen(self) -> None:
        """Frozen dataclass must reject attribute mutation."""
        ac = ActionClassification(
            tier=ActionTier.ALLOW,
            action_type="read_file",
            rationale="safe",
        )
        with pytest.raises((AttributeError, TypeError)):
            ac.tier = ActionTier.DENY  # type: ignore[misc]

    def test_default_requires_foreman_review(self) -> None:
        ac = ActionClassification(
            tier=ActionTier.ALLOW,
            action_type="query",
            rationale="ok",
        )
        assert ac.requires_foreman_review is False

    def test_repr_includes_tier_and_action(self) -> None:
        ac = ActionClassification(
            tier=ActionTier.REQUIRE_APPROVAL,
            action_type="deploy_to_production",
            rationale="needs review",
            requires_foreman_review=True,
        )
        text = repr(ac)
        assert "require_approval" in text
        assert "deploy_to_production" in text
        assert "foreman-review" in text


# ---------------------------------------------------------------------------
# ALLOW tier
# ---------------------------------------------------------------------------


class TestAllowTier:
    def test_allow_action_classified_as_allow(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("read_file", AgentType.WORKER.value)
        assert result.tier is ActionTier.ALLOW

    def test_allow_action_no_foreman_review(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("status_check", AgentType.FOREMAN.value)
        assert result.requires_foreman_review is False

    def test_all_allow_actions_classify_correctly(self, classifier: ActionClassifier) -> None:
        """Every entry in ALLOW_ACTIONS must return ActionTier.ALLOW."""
        for action in ALLOW_ACTIONS:
            result = classifier.classify(action, AgentType.WORKER.value)
            assert result.tier is ActionTier.ALLOW, f"Expected ALLOW for {action!r}"

    def test_allow_rationale_is_non_empty(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("query", AgentType.INSPECTOR.value)
        assert len(result.rationale) > 10

    def test_allow_action_type_preserved(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("search", AgentType.WORKER.value)
        assert result.action_type == "search"


# ---------------------------------------------------------------------------
# DENY tier
# ---------------------------------------------------------------------------


class TestDenyTier:
    def test_deny_action_classified_as_deny(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("delete_production_data", AgentType.WORKER.value)
        assert result.tier is ActionTier.DENY

    def test_all_deny_actions_classify_correctly(self, classifier: ActionClassifier) -> None:
        """Every entry in DENY_ACTIONS must return ActionTier.DENY."""
        for action in DENY_ACTIONS:
            result = classifier.classify(action, AgentType.WORKER.value)
            assert result.tier is ActionTier.DENY, f"Expected DENY for {action!r}"

    def test_deny_is_absolute_even_with_override(self, classifier: ActionClassifier) -> None:
        """DENY_ACTIONS cannot be overridden to ALLOW or REQUIRE_APPROVAL."""
        overrides = {"delete_production_data": ActionTier.ALLOW}
        result = classifier.classify_with_override(
            "delete_production_data", AgentType.WORKER.value, overrides=overrides
        )
        assert result.tier is ActionTier.DENY

    def test_deny_rationale_explains_absolute_block(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("bypass_guardrails", AgentType.WORKER.value)
        assert "deny" in result.rationale.lower() or "absolute" in result.rationale.lower()

    def test_deny_does_not_set_foreman_review(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("modify_credentials", AgentType.FOREMAN.value)
        assert result.requires_foreman_review is False


# ---------------------------------------------------------------------------
# REQUIRE_APPROVAL tier (default and foreman-review variant)
# ---------------------------------------------------------------------------


class TestRequireApprovalTier:
    def test_unknown_action_defaults_to_require_approval(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("totally_unknown_action_xyz", AgentType.WORKER.value)
        assert result.tier is ActionTier.REQUIRE_APPROVAL

    def test_unknown_action_no_foreman_review(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("mystery_operation", AgentType.WORKER.value)
        assert result.requires_foreman_review is False

    def test_foreman_review_actions_require_approval_with_flag(self, classifier: ActionClassifier) -> None:
        """Actions in FOREMAN_REVIEW_ACTIONS get REQUIRE_APPROVAL + foreman flag."""
        for action in FOREMAN_REVIEW_ACTIONS:
            result = classifier.classify(action, AgentType.WORKER.value)
            assert result.tier is ActionTier.REQUIRE_APPROVAL, f"Expected REQUIRE_APPROVAL for {action!r}"
            assert result.requires_foreman_review is True, f"Expected requires_foreman_review=True for {action!r}"

    def test_rationale_mentions_unknown_default(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("something_new", AgentType.WORKER.value)
        assert "unknown" in result.rationale.lower() or "default" in result.rationale.lower()


# ---------------------------------------------------------------------------
# Override mechanism
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_override_allow_action_to_require_approval(self, classifier: ActionClassifier) -> None:
        overrides = {"read_file": ActionTier.REQUIRE_APPROVAL}
        result = classifier.classify_with_override("read_file", AgentType.WORKER.value, overrides=overrides)
        assert result.tier is ActionTier.REQUIRE_APPROVAL

    def test_override_unknown_action_to_allow(self, classifier: ActionClassifier) -> None:
        overrides = {"my_custom_action": ActionTier.ALLOW}
        result = classifier.classify_with_override("my_custom_action", AgentType.WORKER.value, overrides=overrides)
        assert result.tier is ActionTier.ALLOW

    def test_override_rationale_mentions_override(self, classifier: ActionClassifier) -> None:
        overrides = {"read_file": ActionTier.REQUIRE_APPROVAL}
        result = classifier.classify_with_override("read_file", AgentType.WORKER.value, overrides=overrides)
        assert "override" in result.rationale.lower()

    def test_none_overrides_falls_back_to_default(self, classifier: ActionClassifier) -> None:
        result = classifier.classify_with_override("read_file", AgentType.WORKER.value, overrides=None)
        assert result.tier is ActionTier.ALLOW

    def test_empty_overrides_falls_back_to_default(self, classifier: ActionClassifier) -> None:
        result = classifier.classify_with_override("read_file", AgentType.WORKER.value, overrides={})
        assert result.tier is ActionTier.ALLOW

    def test_override_preserves_action_type(self, classifier: ActionClassifier) -> None:
        overrides = {"search": ActionTier.REQUIRE_APPROVAL}
        result = classifier.classify_with_override("search", AgentType.INSPECTOR.value, overrides=overrides)
        assert result.action_type == "search"

    def test_override_foreman_review_action_to_allow(self, classifier: ActionClassifier) -> None:
        """Overriding a FOREMAN_REVIEW_ACTIONS entry to ALLOW should succeed."""
        action = next(iter(FOREMAN_REVIEW_ACTIONS))
        overrides = {action: ActionTier.ALLOW}
        result = classifier.classify_with_override(action, AgentType.WORKER.value, overrides=overrides)
        assert result.tier is ActionTier.ALLOW


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_action_classifier_returns_same_instance(self) -> None:
        a = get_action_classifier()
        b = get_action_classifier()
        assert a is b

    def test_reset_creates_new_instance(self) -> None:
        a = get_action_classifier()
        reset_action_classifier()
        b = get_action_classifier()
        assert a is not b

    def test_singleton_is_action_classifier(self) -> None:
        assert isinstance(get_action_classifier(), ActionClassifier)

    def test_concurrent_access_returns_same_instance(self) -> None:
        """All threads must receive the same singleton instance."""
        results: list[ActionClassifier] = []
        lock = threading.Lock()

        def fetch() -> None:
            instance = get_action_classifier()
            with lock:
                results.append(instance)

        threads = [threading.Thread(target=fetch) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# Context parameter
# ---------------------------------------------------------------------------


class TestContextParameter:
    def test_context_does_not_change_tier(self, classifier: ActionClassifier) -> None:
        """Passing context metadata must not alter the tier assignment."""
        ctx = {"project_id": "proj-1", "environment": "production"}
        result_with = classifier.classify("read_file", AgentType.WORKER.value, context=ctx)
        result_without = classifier.classify("read_file", AgentType.WORKER.value, context=None)
        assert result_with.tier == result_without.tier

    def test_context_none_is_accepted(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("status_check", AgentType.FOREMAN.value, context=None)
        assert result.tier is ActionTier.ALLOW


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


class TestInputNormalisation:
    def test_action_with_leading_trailing_spaces(self, classifier: ActionClassifier) -> None:
        """Whitespace around an allow-listed action must still produce ALLOW."""
        result = classifier.classify("  read_file  ", AgentType.WORKER.value)
        assert result.tier is ActionTier.ALLOW

    def test_uppercase_allow_action(self, classifier: ActionClassifier) -> None:
        """Classification must be case-insensitive for the tier lookup."""
        result = classifier.classify("READ_FILE", AgentType.WORKER.value)
        assert result.tier is ActionTier.ALLOW

    def test_uppercase_deny_action(self, classifier: ActionClassifier) -> None:
        result = classifier.classify("DELETE_PRODUCTION_DATA", AgentType.WORKER.value)
        assert result.tier is ActionTier.DENY
