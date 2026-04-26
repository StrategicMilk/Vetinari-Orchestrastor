"""Tests for the vetinari.enforcement runtime constraint package.

Covers all four enforcers with happy-path, error-path, and edge-case scenarios.
"""

from __future__ import annotations

import pytest

from vetinari.enforcement import (
    AgentCapabilityEnforcer,
    DelegationDepthValidator,
    FileJurisdictionEnforcer,
    QualityGateEnforcer,
    enforce_all,
)
from vetinari.exceptions import (
    CapabilityNotAvailable,
    DelegationDepthExceeded,
    JurisdictionViolation,
    QualityGateFailed,
)
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# DelegationDepthValidator
# ---------------------------------------------------------------------------


class TestDelegationDepthValidator:
    """Tests for DelegationDepthValidator."""

    def test_valid_depth_passes(self):
        """Depth within limit should not raise."""
        # WORKER has max_delegation_depth=3; depth=1 is allowed.
        validator = DelegationDepthValidator()
        result = validator.validate(AgentType.WORKER, current_depth=1)
        assert result is None  # validate() returns None on success

    def test_depth_at_exact_limit_passes(self):
        """Depth equal to the limit should not raise."""
        validator = DelegationDepthValidator()
        # WORKER max_delegation_depth=3
        result = validator.validate(AgentType.WORKER, current_depth=2)
        assert result is None  # validate() returns None on success

    def test_depth_zero_always_passes(self):
        """Depth of zero should always pass for any agent."""
        validator = DelegationDepthValidator()
        result_worker = validator.validate(AgentType.WORKER, current_depth=0)
        result_foreman = validator.validate(AgentType.FOREMAN, current_depth=0)
        assert result_worker is None
        assert result_foreman is None

    def test_depth_exceeds_limit_raises(self):
        """Depth beyond the limit should raise DelegationDepthExceeded."""
        validator = DelegationDepthValidator()
        # WORKER max_delegation_depth=3; depth=3 exceeds it.
        with pytest.raises(DelegationDepthExceeded) as exc_info:
            validator.validate(AgentType.WORKER, current_depth=4)
        error_msg = str(exc_info.value)
        assert AgentType.WORKER.value in error_msg
        assert "4" in error_msg
        assert "3" in error_msg

    def test_planner_higher_limit(self):
        """PLANNER has a higher limit (5); depth=5 should pass."""
        validator = DelegationDepthValidator()
        result = validator.validate(AgentType.FOREMAN, current_depth=5)
        assert result is None  # validate() returns None on success

    def test_planner_exceeds_limit_raises(self):
        """PLANNER depth=6 exceeds its max of 5."""
        validator = DelegationDepthValidator()
        with pytest.raises(DelegationDepthExceeded):
            validator.validate(AgentType.FOREMAN, current_depth=6)

    def test_exception_carries_context(self):
        """Exception context dict should include agent_type, current_depth, max_depth."""
        validator = DelegationDepthValidator()
        with pytest.raises(DelegationDepthExceeded) as exc_info:
            validator.validate(AgentType.WORKER, current_depth=10)
        exc = exc_info.value
        assert exc.context.get("agent_type") == AgentType.WORKER.value
        assert exc.context.get("current_depth") == 10
        assert exc.context.get("max_depth") == 3


# ---------------------------------------------------------------------------
# QualityGateEnforcer
# ---------------------------------------------------------------------------


class TestQualityGateEnforcer:
    """Tests for QualityGateEnforcer."""

    def test_score_above_threshold_passes(self):
        """Score above the threshold should not raise."""
        enforcer = QualityGateEnforcer()
        # WORKER threshold=0.7; score=0.9 is fine.
        result = enforcer.validate(AgentType.WORKER, quality_score=0.9)
        assert result is None  # validate() returns None on success

    def test_score_at_exact_threshold_passes(self):
        """Score exactly at the threshold should not raise."""
        enforcer = QualityGateEnforcer()
        # WORKER threshold=QUALITY_GATE_HIGH (0.70)
        result = enforcer.validate(AgentType.WORKER, quality_score=0.70)
        assert result is None  # validate() returns None on success

    def test_score_below_threshold_raises(self):
        """Score below the threshold should raise QualityGateFailed."""
        enforcer = QualityGateEnforcer()
        # WORKER threshold=QUALITY_GATE_HIGH (0.70); score=0.5 is too low.
        with pytest.raises(QualityGateFailed) as exc_info:
            enforcer.validate(AgentType.WORKER, quality_score=0.5)
        error_msg = str(exc_info.value)
        assert AgentType.WORKER.value in error_msg
        assert "0.500" in error_msg
        assert "0.700" in error_msg

    def test_foreman_higher_threshold(self):
        """FOREMAN has threshold=AGENT_QUALITY_GATE_STRICT (0.75); score=0.85 should pass."""
        enforcer = QualityGateEnforcer()
        result = enforcer.validate(AgentType.FOREMAN, quality_score=0.85)
        assert result is None  # validate() returns None on success

    def test_foreman_below_threshold_raises(self):
        """FOREMAN score=0.7 is below threshold=AGENT_QUALITY_GATE_STRICT (0.75)."""
        enforcer = QualityGateEnforcer()
        with pytest.raises(QualityGateFailed):
            enforcer.validate(AgentType.FOREMAN, quality_score=0.7)

    def test_exception_carries_context(self):
        """Exception context dict should include agent_type, quality_score, threshold."""
        enforcer = QualityGateEnforcer()
        with pytest.raises(QualityGateFailed) as exc_info:
            enforcer.validate(AgentType.INSPECTOR, quality_score=0.1)
        exc = exc_info.value
        assert exc.context.get("agent_type") == AgentType.INSPECTOR.value
        assert exc.context.get("quality_score") == pytest.approx(0.1)
        assert exc.context.get("threshold") == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# FileJurisdictionEnforcer
# ---------------------------------------------------------------------------


class TestFileJurisdictionEnforcer:
    """Tests for FileJurisdictionEnforcer."""

    def test_file_within_jurisdiction_passes(self):
        """A file matching a jurisdiction prefix should not raise."""
        enforcer = FileJurisdictionEnforcer()
        # QUALITY has jurisdiction including "tests/"
        result = enforcer.validate(AgentType.INSPECTOR, file_path="tests/test_foo.py")
        assert result is None  # validate() returns None on success

    def test_file_in_nested_path_passes(self):
        """A file in a subdirectory of a jurisdiction entry should pass."""
        enforcer = FileJurisdictionEnforcer()
        # OPERATIONS has "docs/"
        result = enforcer.validate(AgentType.WORKER, file_path="docs/architecture/overview.md")
        assert result is None  # validate() returns None on success

    def test_file_outside_jurisdiction_raises(self):
        """A file not matching any jurisdiction prefix should raise JurisdictionViolation."""
        enforcer = FileJurisdictionEnforcer()
        # INSPECTOR jurisdiction is quality_agent.py and tests/ — not "vetinari/core/"
        with pytest.raises(JurisdictionViolation) as exc_info:
            enforcer.validate(AgentType.INSPECTOR, file_path="vetinari/core/main.py")
        error_msg = str(exc_info.value)
        assert AgentType.INSPECTOR.value in error_msg
        assert "vetinari/core/main.py" in error_msg

    def test_empty_jurisdiction_allows_all_files(self):
        """An agent with an empty jurisdiction list should permit any file."""
        from vetinari.agents.contracts import AGENT_REGISTRY, AgentSpec

        enforcer = FileJurisdictionEnforcer()
        # Override the registry temporarily with an empty-jurisdiction spec
        original = AGENT_REGISTRY.get(AgentType.WORKER)
        empty_spec = AgentSpec(
            agent_type=AgentType.WORKER,
            name="Test",
            description="Test spec with empty jurisdiction",
            default_model="test",
            jurisdiction=[],
        )
        AGENT_REGISTRY[AgentType.WORKER] = empty_spec
        try:
            result = enforcer.validate(AgentType.WORKER, file_path="any/random/path.py")
            assert result is None  # empty jurisdiction allows any file, validate() returns None
        finally:
            if original is not None:
                AGENT_REGISTRY[AgentType.WORKER] = original

    def test_windows_backslash_path_normalised(self):
        """Windows-style backslash paths should be normalised before comparison."""
        enforcer = FileJurisdictionEnforcer()
        # QUALITY has "tests/" — Windows path should match after normalisation.
        result = enforcer.validate(AgentType.INSPECTOR, file_path=r"tests\test_foo.py")
        assert result is None  # validate() returns None on success

    def test_exception_carries_context(self):
        """Exception context dict should include agent_type, file_path, jurisdiction."""
        enforcer = FileJurisdictionEnforcer()
        with pytest.raises(JurisdictionViolation) as exc_info:
            enforcer.validate(AgentType.INSPECTOR, file_path="vetinari/core/main.py")
        exc = exc_info.value
        assert exc.context.get("agent_type") == AgentType.INSPECTOR.value
        assert exc.context.get("file_path") == "vetinari/core/main.py"
        assert isinstance(exc.context.get("jurisdiction"), list)

    def test_exact_jurisdiction_file_path_passes(self):
        """A file path that exactly equals a jurisdiction entry should pass."""
        enforcer = FileJurisdictionEnforcer()
        # PLANNER jurisdiction includes "vetinari/agents/contracts.py"
        result = enforcer.validate(AgentType.FOREMAN, file_path="vetinari/agents/contracts.py")
        assert result is None  # validate() returns None on success


# ---------------------------------------------------------------------------
# AgentCapabilityEnforcer
# ---------------------------------------------------------------------------


class TestAgentCapabilityEnforcer:
    """Tests for AgentCapabilityEnforcer."""

    def test_available_capability_passes(self):
        """A capability in the agent's list should not raise."""
        enforcer = AgentCapabilityEnforcer()
        # WORKER has "code_scaffolding"
        result = enforcer.validate(AgentType.WORKER, required_capability="code_scaffolding")
        assert result is None  # validate() returns None on success

    def test_unavailable_capability_raises(self):
        """A capability absent from the agent's list should raise CapabilityNotAvailable."""
        enforcer = AgentCapabilityEnforcer()
        # FOREMAN does not have "code_review"
        with pytest.raises(CapabilityNotAvailable) as exc_info:
            enforcer.validate(AgentType.FOREMAN, required_capability="code_review")
        error_msg = str(exc_info.value)
        assert AgentType.FOREMAN.value in error_msg
        assert "code_review" in error_msg

    def test_oracle_capability_passes(self):
        """CONSOLIDATED_ORACLE should allow its own capability."""
        enforcer = AgentCapabilityEnforcer()
        result = enforcer.validate(
            AgentType.WORKER,
            required_capability="architecture_decision_support",
        )
        assert result is None  # validate() returns None on success

    def test_planner_capability_passes(self):
        """PLANNER should allow 'goal_decomposition'."""
        enforcer = AgentCapabilityEnforcer()
        result = enforcer.validate(AgentType.FOREMAN, required_capability="goal_decomposition")
        assert result is None  # validate() returns None on success

    def test_wrong_agent_for_security_audit_raises(self):
        """PLANNER does not have 'security_audit'."""
        enforcer = AgentCapabilityEnforcer()
        with pytest.raises(CapabilityNotAvailable):
            enforcer.validate(AgentType.FOREMAN, required_capability="security_audit")

    def test_exception_carries_context(self):
        """Exception context dict should include agent_type and required_capability."""
        enforcer = AgentCapabilityEnforcer()
        with pytest.raises(CapabilityNotAvailable) as exc_info:
            enforcer.validate(AgentType.WORKER, required_capability="nonexistent_cap")
        exc = exc_info.value
        assert exc.context.get("agent_type") == AgentType.WORKER.value
        assert exc.context.get("required_capability") == "nonexistent_cap"
        assert isinstance(exc.context.get("available_capabilities"), list)

    def test_empty_capability_string_raises(self):
        """An empty string capability is not in any agent's list."""
        enforcer = AgentCapabilityEnforcer()
        with pytest.raises(CapabilityNotAvailable):
            enforcer.validate(AgentType.WORKER, required_capability="")


# ---------------------------------------------------------------------------
# enforce_all convenience function
# ---------------------------------------------------------------------------


class TestEnforceAll:
    """Tests for the enforce_all() convenience function."""

    def test_all_valid_passes_silently(self):
        """All checks passing should return without raising."""
        result = enforce_all(
            AgentType.INSPECTOR,
            current_depth=1,
            quality_score=0.9,
            file_path="tests/test_enforcement.py",
            required_capability="code_review",
        )
        assert result is None  # enforce_all() returns None when all checks pass

    def test_depth_violation_detected(self):
        """enforce_all should surface DelegationDepthExceeded."""
        with pytest.raises(DelegationDepthExceeded):
            enforce_all(AgentType.WORKER, current_depth=99)

    def test_quality_violation_detected(self):
        """enforce_all should surface QualityGateFailed."""
        with pytest.raises(QualityGateFailed):
            enforce_all(AgentType.WORKER, quality_score=0.0)

    def test_jurisdiction_violation_detected(self):
        """enforce_all should surface JurisdictionViolation."""
        with pytest.raises(JurisdictionViolation):
            enforce_all(AgentType.WORKER, file_path="vetinari/core/main.py")

    def test_capability_violation_detected(self):
        """enforce_all should surface CapabilityNotAvailable."""
        with pytest.raises(CapabilityNotAvailable):
            enforce_all(AgentType.FOREMAN, required_capability="security_audit")

    def test_no_args_passes_silently(self):
        """Calling enforce_all with no optional checks should pass silently."""
        result = enforce_all(AgentType.FOREMAN)
        assert result is None  # enforce_all() returns None when no checks are requested

    def test_depth_and_quality_collected_as_composite(self):
        """Multiple violations should be collected into a CompositeEnforcementError."""
        from vetinari.exceptions import CompositeEnforcementError

        with pytest.raises(CompositeEnforcementError) as exc_info:
            enforce_all(
                AgentType.WORKER,
                current_depth=99,
                quality_score=0.0,  # both are violations — collected into composite
            )
        assert len(exc_info.value.violations) == 2

    def test_extra_kwargs_ignored(self):
        """Unknown kwargs should be accepted and ignored without error."""
        result = enforce_all(AgentType.FOREMAN, unknown_field="something", another_key=42)
        assert result is None  # enforce_all() returns None when all checks pass
