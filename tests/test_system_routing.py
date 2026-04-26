"""Tests for System 1/System 2 dual-process routing (US-009).

Verifies that simple tasks route to System 1 (fast path) and complex tasks
route to System 2 (full pipeline), with correct Inspector bypass safety rules.
"""

from __future__ import annotations

import pytest

from vetinari.routing.system_router import (
    InspectorBypassCheck,
    ModelTier,
    SystemDecision,
    SystemType,
    check_inspector_bypass_safety,
    get_system_routing_stats,
    reset_system_routing_stats,
    route_system,
)


class TestSimpleTaskRoutesSystem1:
    """Simple tasks (EXPRESS tier or SIMPLE complexity + high confidence) → System 1."""

    def setup_method(self) -> None:
        reset_system_routing_stats()

    def teardown_method(self) -> None:
        reset_system_routing_stats()

    def test_express_tier_routes_system1(self) -> None:
        """EXPRESS tier with high confidence should produce System 1 routing."""
        decision = route_system("fix typo", intake_tier="express", confidence=0.9)
        assert decision.system_type == SystemType.SYSTEM_1

    def test_simple_complexity_routes_system1(self) -> None:
        """SIMPLE complexity with high confidence should produce System 1 routing."""
        decision = route_system("rename variable", complexity="simple", confidence=0.9)
        assert decision.system_type == SystemType.SYSTEM_1

    def test_system1_skips_foreman(self) -> None:
        """System 1 decisions must set skip_foreman=True (no planning stage)."""
        decision = route_system("fix typo", intake_tier="express", confidence=0.9)
        assert decision.skip_foreman is True

    def test_system1_uses_small_model(self) -> None:
        """System 1 decisions must select the SMALL model tier."""
        decision = route_system("rename variable", complexity="simple", confidence=0.9)
        assert decision.model_tier == ModelTier.SMALL


class TestComplexTaskRoutesSystem2:
    """Complex, custom, or low-confidence tasks → System 2 (full pipeline)."""

    def setup_method(self) -> None:
        reset_system_routing_stats()

    def teardown_method(self) -> None:
        reset_system_routing_stats()

    def test_complex_task_routes_system2(self) -> None:
        """COMPLEX complexity should always produce System 2 routing."""
        decision = route_system(
            "architect a distributed multi-service migration with backwards compatibility",
            complexity="complex",
        )
        assert decision.system_type == SystemType.SYSTEM_2

    def test_custom_tier_routes_system2(self) -> None:
        """CUSTOM intake tier should always produce System 2 routing."""
        decision = route_system("build something custom", intake_tier="custom")
        assert decision.system_type == SystemType.SYSTEM_2

    def test_low_confidence_routes_system2(self) -> None:
        """Confidence below LOW_CONFIDENCE_THRESHOLD should produce System 2 routing."""
        decision = route_system("unclear task", confidence=0.3)
        assert decision.system_type == SystemType.SYSTEM_2

    def test_system2_uses_large_model_for_complex(self) -> None:
        """COMPLEX tasks must select the LARGE model tier."""
        decision = route_system(
            "architect a distributed multi-service migration with backwards compatibility",
            complexity="complex",
        )
        assert decision.model_tier == ModelTier.LARGE

    def test_system2_full_pipeline(self) -> None:
        """System 2 decisions must not skip Foreman or Inspector."""
        decision = route_system(
            "architect a distributed multi-service migration with backwards compatibility",
            complexity="complex",
        )
        assert decision.skip_foreman is False
        assert decision.skip_inspector is False


class TestInspectorBypassSafety:
    """US-001 truth table: Inspector bypass is allowed only when all conditions met."""

    def test_bypass_allowed_when_all_conditions_met(self) -> None:
        """All conditions satisfied → allowed=True."""
        result = check_inspector_bypass_safety(
            "update readme",
            confidence=0.95,
            prior_successes=5,
            involves_code_generation=False,
            autonomy_level=3,
        )
        assert result.allowed is True

    def test_bypass_denied_for_security_tasks(self) -> None:
        """Description containing 'security' must be denied."""
        result = check_inspector_bypass_safety(
            "security audit",
            confidence=0.95,
            prior_successes=5,
            involves_code_generation=False,
            autonomy_level=3,
        )
        assert result.allowed is False
        assert any("security" in flag for flag in result.safety_flags)

    def test_bypass_denied_for_code_generation(self) -> None:
        """Code generation tasks must be denied regardless of other conditions."""
        result = check_inspector_bypass_safety(
            "update readme",
            confidence=0.95,
            prior_successes=5,
            involves_code_generation=True,
            autonomy_level=3,
        )
        assert result.allowed is False

    def test_bypass_denied_for_low_confidence(self) -> None:
        """Confidence below threshold must be denied."""
        result = check_inspector_bypass_safety(
            "update readme",
            confidence=0.5,
            prior_successes=5,
            involves_code_generation=False,
            autonomy_level=3,
        )
        assert result.allowed is False

    def test_bypass_denied_for_low_autonomy(self) -> None:
        """Autonomy level below L2 must be denied."""
        result = check_inspector_bypass_safety(
            "update readme",
            confidence=0.95,
            prior_successes=5,
            involves_code_generation=False,
            autonomy_level=1,
        )
        assert result.allowed is False

    def test_bypass_denied_insufficient_history(self) -> None:
        """Fewer than _MIN_PRIOR_SUCCESSES prior successes must be denied."""
        result = check_inspector_bypass_safety(
            "update readme",
            confidence=0.95,
            prior_successes=1,
            involves_code_generation=False,
            autonomy_level=3,
        )
        assert result.allowed is False


class TestSystemRoutingTracking:
    """Routing decisions are tracked for analytics and continuous improvement."""

    def setup_method(self) -> None:
        reset_system_routing_stats()

    def teardown_method(self) -> None:
        reset_system_routing_stats()

    def test_tracking_records_decisions(self) -> None:
        """Each route_system() call should appear in the stats total."""
        route_system("fix typo", intake_tier="express", confidence=0.9)
        route_system("build complex system", complexity="complex", confidence=0.9)
        stats = get_system_routing_stats()
        assert stats["total"] >= 2

    def test_reset_clears_stats(self) -> None:
        """reset_system_routing_stats() must zero out all counters."""
        route_system("fix typo", intake_tier="express", confidence=0.9)
        reset_system_routing_stats()
        stats = get_system_routing_stats()
        assert stats["total"] == 0
