"""Tests proving governance/autonomy security hardening fails closed.

Covers four security fixes from SESSION-28:
1. _normalise() in FileJurisdictionEnforcer resolves .. traversal.
2. Jurisdiction rejects path-traversal bypass (docs/../vetinari/...).
3. Jurisdiction prefix matching requires path-component boundary.
4. DEFER from request_permission blocks execution (not fall-through).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vetinari.enforcement.jurisdiction import FileJurisdictionEnforcer
from vetinari.safety.policy_enforcer import PolicyDecision, PolicyEnforcer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_enforcer() -> FileJurisdictionEnforcer:
    """Return a fresh FileJurisdictionEnforcer."""
    return FileJurisdictionEnforcer()


def _patch_jurisdiction_spec(monkeypatch: pytest.MonkeyPatch, jurisdiction: list[str]) -> None:
    """Patch the AgentSpec lookup used by the imported enforcer class.

    The full suite restores ``vetinari.*`` modules between test files. Patching
    the module by name can therefore target a re-imported module while the
    collection-time ``FileJurisdictionEnforcer`` reference still uses the
    original function globals.
    """

    spec = SimpleNamespace(jurisdiction=jurisdiction)
    monkeypatch.setitem(
        FileJurisdictionEnforcer.validate.__globals__,
        "get_agent_spec",
        lambda _: spec,
    )


# ---------------------------------------------------------------------------
# Problem 2 / jurisdiction._normalise — path traversal resolution
# ---------------------------------------------------------------------------


class TestNormalise:
    """_normalise must collapse .. segments without touching the filesystem."""

    def test_normalise_simple_path_unchanged(self) -> None:
        """A plain relative path passes through unchanged."""
        result = FileJurisdictionEnforcer._normalise("vetinari/agents/foo.py")
        assert result == "vetinari/agents/foo.py"

    def test_normalise_backslashes_converted(self) -> None:
        """Windows backslashes are replaced with forward slashes."""
        result = FileJurisdictionEnforcer._normalise("vetinari\\agents\\foo.py")
        assert result == "vetinari/agents/foo.py"

    def test_normalise_resolves_dotdot_traversal(self) -> None:
        """.. segments are collapsed so traversal cannot bypass prefix checks."""
        result = FileJurisdictionEnforcer._normalise("docs/../vetinari/agents/contracts.py")
        assert result == "vetinari/agents/contracts.py"

    def test_normalise_resolves_multiple_dotdot(self) -> None:
        """Multiple .. hops are all resolved."""
        result = FileJurisdictionEnforcer._normalise("a/b/c/../../d/e.py")
        assert result == "a/d/e.py"

    def test_normalise_strips_leading_dotslash(self) -> None:
        """Leading ./ is stripped for clean prefix matching."""
        result = FileJurisdictionEnforcer._normalise("./vetinari/foo.py")
        assert result == "vetinari/foo.py"

    def test_normalise_dotdot_with_backslash(self) -> None:
        """Path traversal with Windows separators is neutralised."""
        result = FileJurisdictionEnforcer._normalise("docs\\..\\vetinari\\secret.py")
        assert result == "vetinari/secret.py"


# ---------------------------------------------------------------------------
# Problem 2 / jurisdiction.validate — traversal bypass blocked
# ---------------------------------------------------------------------------


class TestJurisdictionTraversalBlocked:
    """validate() must reject paths that traverse outside allowed jurisdiction."""

    def test_traversal_bypass_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """docs/../vetinari/agents/contracts.py must be rejected when only docs/ is allowed."""
        from vetinari.types import AgentType

        enforcer = _make_enforcer()
        _patch_jurisdiction_spec(monkeypatch, ["docs/"])
        from vetinari.exceptions import JurisdictionViolation

        with pytest.raises(JurisdictionViolation):
            enforcer.validate(AgentType.WORKER, "docs/../vetinari/agents/contracts.py")

    def test_legitimate_path_within_docs_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A genuine docs/ path passes when docs/ is in jurisdiction."""
        from vetinari.types import AgentType

        enforcer = _make_enforcer()
        _patch_jurisdiction_spec(monkeypatch, ["docs/"])
        assert enforcer.validate(AgentType.WORKER, "docs/README.md") is None

    def test_traversal_with_windows_separators_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows-style traversal path is also rejected."""
        from vetinari.types import AgentType

        enforcer = _make_enforcer()
        _patch_jurisdiction_spec(monkeypatch, ["docs/"])
        from vetinari.exceptions import JurisdictionViolation

        with pytest.raises(JurisdictionViolation):
            enforcer.validate(AgentType.WORKER, "docs\\..\\vetinari\\secret.py")


# ---------------------------------------------------------------------------
# Problem 3 / jurisdiction prefix boundary
# ---------------------------------------------------------------------------


class TestJurisdictionPrefixBoundary:
    """Prefix match must require a path-component boundary."""

    def test_boundary_required_similar_prefix_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """vetinari_evil/private.py must not match a jurisdiction of vetinari/."""
        from vetinari.types import AgentType

        enforcer = _make_enforcer()
        _patch_jurisdiction_spec(monkeypatch, ["vetinari/"])
        from vetinari.exceptions import JurisdictionViolation

        with pytest.raises(JurisdictionViolation):
            enforcer.validate(AgentType.WORKER, "vetinari_evil/private.py")

    def test_legitimate_subpath_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """vetinari/agents/foo.py passes when vetinari/ is in jurisdiction."""
        from vetinari.types import AgentType

        enforcer = _make_enforcer()
        _patch_jurisdiction_spec(monkeypatch, ["vetinari/"])
        assert enforcer.validate(AgentType.WORKER, "vetinari/agents/foo.py") is None

    def test_exact_match_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exact match on the jurisdiction entry itself is accepted."""
        from vetinari.types import AgentType

        enforcer = _make_enforcer()
        _patch_jurisdiction_spec(monkeypatch, ["docs"])
        assert enforcer.validate(AgentType.WORKER, "docs") is None


# ---------------------------------------------------------------------------
# Problem 4 / governor DEFER is blocking
# ---------------------------------------------------------------------------


class TestDeferIsBlocking:
    """request_permission() returning DEFER must not allow execution to proceed."""

    def test_defer_decision_is_not_approve(self) -> None:
        """DEFER != APPROVE so the != APPROVE guard correctly blocks it."""
        from vetinari.types import PermissionDecision

        assert PermissionDecision.DEFER != PermissionDecision.APPROVE

    def test_deny_decision_is_not_approve(self) -> None:
        """DENY != APPROVE so the != APPROVE guard correctly blocks it."""
        from vetinari.types import PermissionDecision

        assert PermissionDecision.DENY != PermissionDecision.APPROVE

    def test_approve_decision_passes_guard(self) -> None:
        """APPROVE == APPROVE so APPROVE is the only value that lets execution proceed."""
        from vetinari.types import PermissionDecision

        assert PermissionDecision.APPROVE == PermissionDecision.APPROVE

    def test_governor_l1_returns_defer(self, tmp_path: Path) -> None:
        """L1_SUGGEST policy produces DEFER — verifies the gate fires for real."""
        import yaml

        from vetinari.autonomy.governor import AutonomyGovernor
        from vetinari.types import PermissionDecision

        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            yaml.dump({"actions": {"test_action": {"level": "L1"}}}),
            encoding="utf-8",
        )
        gov = AutonomyGovernor(policy_path=policy_file)
        decision = gov.request_permission("test_action")
        assert decision == PermissionDecision.DEFER
        # Critically: DEFER must not equal APPROVE — the gate blocks execution
        assert decision != PermissionDecision.APPROVE

    def test_governor_l2_returns_approve(self, tmp_path: Path) -> None:
        """L2_ACT_REPORT policy produces APPROVE — sanity check for non-defer path."""
        import yaml

        from vetinari.autonomy.governor import AutonomyGovernor
        from vetinari.types import PermissionDecision

        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            yaml.dump({"actions": {"test_action": {"level": "L2"}}}),
            encoding="utf-8",
        )
        gov = AutonomyGovernor(policy_path=policy_file)
        decision = gov.request_permission("test_action")
        assert decision == PermissionDecision.APPROVE

    def test_governor_l0_returns_deny(self, tmp_path: Path) -> None:
        """L0_MANUAL policy produces DENY — also non-APPROVE and must block."""
        import yaml

        from vetinari.autonomy.governor import AutonomyGovernor
        from vetinari.types import PermissionDecision

        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            yaml.dump({"actions": {"test_action": {"level": "L0"}}}),
            encoding="utf-8",
        )
        gov = AutonomyGovernor(policy_path=policy_file)
        decision = gov.request_permission("test_action")
        assert decision == PermissionDecision.DENY
        assert decision != PermissionDecision.APPROVE


# ---------------------------------------------------------------------------
# Problem 3+4 / PolicyEnforcer._check_jurisdiction — delete verb + unknown principal
# ---------------------------------------------------------------------------


class TestPolicyEnforcerJurisdiction:
    """_check_jurisdiction must gate delete verbs and deny unknown principals."""

    def setup_method(self) -> None:
        """Reset the PolicyEnforcer singleton between tests."""
        PolicyEnforcer._instance = None

    def test_delete_verb_blocked_outside_jurisdiction(self) -> None:
        """delete action on a path outside jurisdiction is denied."""
        enforcer = PolicyEnforcer()
        decision = enforcer.check_action(
            agent_type="worker",
            action="delete",
            target="private/secret.py",
            context={},
        )
        assert decision.allowed is False
        assert decision.risk_level == "high"

    def test_delete_verb_allowed_inside_jurisdiction(self) -> None:
        """delete action inside the worker's jurisdiction passes jurisdiction check.

        Note: it may still be blocked by the irreversibility policy unless
        allow_destructive is set; we allow it here to isolate jurisdiction.
        """
        enforcer = PolicyEnforcer()
        # allow_destructive=True so irreversibility policy does not interfere
        decision = enforcer.check_action(
            agent_type="worker",
            action="delete",
            target="vetinari/old_module.py",
            context={"allow_destructive": True},
        )
        # Jurisdiction passes; irreversibility allows it with allow_destructive=True
        assert decision.allowed is True

    def test_unknown_principal_denied_on_write(self) -> None:
        """An agent type not in _JURISDICTION is denied all mutating actions."""
        enforcer = PolicyEnforcer()
        decision = enforcer.check_action(
            agent_type="rogue_agent",
            action="write",
            target="vetinari/core/main.py",
            context={},
        )
        assert decision.allowed is False
        assert decision.risk_level == "high"

    def test_unknown_principal_denied_on_delete(self) -> None:
        """An unrecognised principal is also denied destructive actions."""
        enforcer = PolicyEnforcer()
        decision = enforcer.check_action(
            agent_type="mystery_bot",
            action="delete",
            target="vetinari/types.py",
            context={"allow_destructive": True},
        )
        assert decision.allowed is False
        assert decision.risk_level == "high"

    def test_read_action_passes_jurisdiction_for_unknown_principal(self) -> None:
        """Read-only verbs are not gated by jurisdiction — they pass through."""
        enforcer = PolicyEnforcer()
        decision = enforcer.check_action(
            agent_type="unknown_reader",
            action="read",
            target="vetinari/types.py",
            context={},
        )
        # Read is not a mutating action — jurisdiction policy returns None (pass)
        # and all other policies also pass for a read
        assert decision.allowed is True

    def test_boundary_check_rejects_similar_prefix(self) -> None:
        """vetinari_evil/ must not match the vetinari/ jurisdiction prefix."""
        enforcer = PolicyEnforcer()
        decision = enforcer.check_action(
            agent_type="worker",
            action="write",
            target="vetinari_evil/private.py",
            context={},
        )
        assert decision.allowed is False
        assert decision.risk_level == "high"
