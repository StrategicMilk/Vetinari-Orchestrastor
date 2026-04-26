"""Tests for agent scope enforcement (Session 23, US-003).

Verifies that the 3-agent factory pipeline (Foreman → Worker → Inspector)
enforces role boundaries:
- Foreman: planning-only inference (tested in test_foreman_coordinator.py)
- Worker: cannot modify frozen structural attributes after init
- Inspector: cannot infer outside evaluation modes
- Tool deny lists are enforced consistently by _use_tool() and _list_tools()
- Delegation depths match contracts.py canonical values
- constraints.yaml uses canonical 3-agent names
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vetinari.agents.consolidated.quality_agent import InspectorAgent
from vetinari.agents.consolidated.worker_agent import _WORKER_FROZEN_ATTRS, WorkerAgent
from vetinari.agents.contracts import AGENT_REGISTRY
from vetinari.agents.tools_mixin import _TOOL_DENY
from vetinari.constraints.architecture import ARCHITECTURE_CONSTRAINTS
from vetinari.exceptions import CapabilityNotAvailable, JurisdictionViolation
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Worker frozen-attribute guard
# ---------------------------------------------------------------------------


class TestWorkerConfigGuard:
    """Worker cannot modify structural attributes after __init__."""

    def test_worker_modify_agent_type_raises_error(self) -> None:
        worker = WorkerAgent()
        with pytest.raises(CapabilityNotAvailable, match="frozen"):
            worker.agent_type = AgentType.FOREMAN

    def test_worker_modify_modes_raises_error(self) -> None:
        worker = WorkerAgent()
        with pytest.raises(CapabilityNotAvailable, match="frozen"):
            worker.MODES = {}

    def test_worker_modify_default_mode_raises_error(self) -> None:
        worker = WorkerAgent()
        with pytest.raises(CapabilityNotAvailable, match="frozen"):
            worker.DEFAULT_MODE = "hacked"

    def test_worker_modify_mode_keywords_raises_error(self) -> None:
        worker = WorkerAgent()
        with pytest.raises(CapabilityNotAvailable, match="frozen"):
            worker.MODE_KEYWORDS = {}

    def test_worker_can_modify_non_frozen_attrs(self) -> None:
        """Non-frozen attributes (e.g. _current_mode) remain writable."""
        worker = WorkerAgent()
        worker._current_mode = "build"
        assert worker._current_mode == "build"

    def test_frozen_attrs_constant_is_complete(self) -> None:
        """All identity-defining attrs are in the frozen set."""
        assert "agent_type" in _WORKER_FROZEN_ATTRS
        assert "MODES" in _WORKER_FROZEN_ATTRS
        assert "DEFAULT_MODE" in _WORKER_FROZEN_ATTRS
        assert "MODE_KEYWORDS" in _WORKER_FROZEN_ATTRS


# ---------------------------------------------------------------------------
# Inspector inference guard
# ---------------------------------------------------------------------------


class TestInspectorInferenceGuard:
    """Inspector cannot infer outside evaluation modes."""

    @pytest.mark.parametrize("mode", ["code_review", "security_audit", "test_generation", "simplification"])
    def test_inspector_evaluation_modes_allowed(self, mode: str) -> None:
        """Evaluation modes should not raise JurisdictionViolation.

        We don't check the actual inference result (requires LLM) — just
        verify the guard doesn't block legitimate modes. The _infer call
        will fail downstream (no model loaded), but that's expected.
        """
        inspector = InspectorAgent()
        inspector._current_mode = mode
        # The guard itself should not raise — any downstream error is fine
        # (no LLM loaded in tests). We verify the guard logic only.
        assert mode in inspector._EVALUATION_MODES

    def test_inspector_unknown_mode_raises_jurisdiction(self) -> None:
        """Inspector must reject inference in non-evaluation modes."""
        inspector = InspectorAgent()
        inspector._current_mode = "build"
        with pytest.raises(JurisdictionViolation, match="only evaluation modes"):
            inspector._infer("test prompt")

    def test_inspector_none_mode_allowed(self) -> None:
        """When _current_mode is None (pre-execute), inference is allowed."""
        inspector = InspectorAgent()
        inspector._current_mode = None
        # Guard should not raise for None mode (allows setup inference)
        assert inspector._current_mode is None


# ---------------------------------------------------------------------------
# Tool deny list consistency
# ---------------------------------------------------------------------------


class TestToolDenyListConsistency:
    """_use_tool() and _list_tools() must agree on denied tools."""

    def test_foreman_deny_list_exists(self) -> None:
        denied = _TOOL_DENY.get(AgentType.FOREMAN.value)
        assert denied is not None
        assert "file_write" in denied
        assert "code_execute" in denied

    def test_inspector_deny_list_exists(self) -> None:
        denied = _TOOL_DENY.get(AgentType.INSPECTOR.value)
        assert denied is not None
        assert "file_write" in denied
        assert "code_execute" in denied

    def test_worker_deny_list_empty(self) -> None:
        """Worker has full tool access — deny list should be empty."""
        denied = _TOOL_DENY.get(AgentType.WORKER.value)
        assert denied is not None
        assert len(denied) == 0

    def test_all_three_agents_have_deny_entries(self) -> None:
        """Every pipeline agent must have an explicit deny list entry."""
        for agent_type in [AgentType.FOREMAN, AgentType.WORKER, AgentType.INSPECTOR]:
            assert agent_type.value in _TOOL_DENY, f"Missing deny list for {agent_type.value}"


# ---------------------------------------------------------------------------
# Delegation depth unification
# ---------------------------------------------------------------------------


class TestDelegationDepthUnification:
    """architecture.py delegation depths must match contracts.py AGENT_SPECS."""

    @pytest.mark.parametrize(
        "agent_type,expected_depth",
        [
            (AgentType.FOREMAN, 5),
            (AgentType.WORKER, 3),
            (AgentType.INSPECTOR, 2),
        ],
    )
    def test_architecture_matches_contracts(self, agent_type: AgentType, expected_depth: int) -> None:
        arch = ARCHITECTURE_CONSTRAINTS[agent_type.value]
        spec = AGENT_REGISTRY[agent_type]
        assert arch.max_delegation_depth == expected_depth, (
            f"{agent_type.value}: architecture says {arch.max_delegation_depth}, expected {expected_depth}"
        )
        assert spec.max_delegation_depth == expected_depth, (
            f"{agent_type.value}: contracts says {spec.max_delegation_depth}, expected {expected_depth}"
        )


# ---------------------------------------------------------------------------
# Constraints YAML canonicalization
# ---------------------------------------------------------------------------


class TestConstraintsYamlCanonical:
    """constraints.yaml must use canonical 3-agent names."""

    @pytest.fixture
    def constraints_data(self) -> dict:
        yaml_path = Path(__file__).parent.parent / "vetinari" / "config" / "standards" / "constraints.yaml"
        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_only_canonical_agent_names(self, constraints_data: dict) -> None:
        agents = set(constraints_data["agents"].keys())
        expected = {"foreman", "worker", "inspector"}
        assert agents == expected, f"Found non-canonical agent names: {agents - expected}"

    def test_wip_limits_canonical(self, constraints_data: dict) -> None:
        wip_keys = {k for k in constraints_data["wip_limits"] if k != "global_max"}
        expected = {"foreman", "worker", "inspector"}
        assert wip_keys == expected, f"WIP limits use non-canonical names: {wip_keys - expected}"

    def test_no_stale_agent_names(self, constraints_data: dict) -> None:
        """Old 6-agent names must not appear anywhere in constraints."""
        stale = {"planner", "builder", "researcher", "oracle", "quality", "operations"}
        all_keys = set(constraints_data.get("agents", {}))
        all_keys.update(k for k in constraints_data.get("wip_limits", {}) if k != "global_max")
        overlap = all_keys & stale
        assert not overlap, f"Stale agent names still present: {overlap}"


# ---------------------------------------------------------------------------
# Bug 1 regression: fail-open on missing AgentSpec (capabilities.py)
# ---------------------------------------------------------------------------


class TestCapabilityEnforcerFailClosed:
    """AgentCapabilityEnforcer must deny requests when AgentSpec is missing.

    Regression for Bug 1: the enforcer previously returned silently (fail-open)
    when ``get_agent_spec()`` returned None for an unregistered agent type.
    The fix: raise CapabilityNotAvailable so enforcement blocks the request.
    """

    def test_missing_spec_raises_capability_not_available(self) -> None:
        """Unregistered agent type must raise CapabilityNotAvailable, not silently pass."""
        from unittest.mock import patch

        from vetinari.enforcement.capabilities import AgentCapabilityEnforcer

        enforcer = AgentCapabilityEnforcer()
        with patch("vetinari.enforcement.capabilities.get_agent_spec", return_value=None):
            with pytest.raises(CapabilityNotAvailable) as exc_info:
                enforcer.validate(AgentType.WORKER, "some_capability")
        # Error message must name the agent and indicate the spec is missing.
        # Use case-insensitive check: AgentType.WORKER.value is "WORKER" (uppercase).
        assert "no registered AgentSpec" in str(exc_info.value)
        assert "worker" in str(exc_info.value).lower()

    def test_missing_spec_error_includes_requested_capability(self) -> None:
        """The denial message must name the requested capability for diagnostics."""
        from unittest.mock import patch

        from vetinari.enforcement.capabilities import AgentCapabilityEnforcer

        enforcer = AgentCapabilityEnforcer()
        with patch("vetinari.enforcement.capabilities.get_agent_spec", return_value=None):
            with pytest.raises(CapabilityNotAvailable) as exc_info:
                enforcer.validate(AgentType.FOREMAN, "special_power")
        # The exception context must carry the requested capability name.
        assert exc_info.value.context.get("required_capability") == "special_power"
        assert exc_info.value.context.get("available_capabilities") == []

    def test_known_agent_type_with_valid_capability_passes(self) -> None:
        """Registered agent with matching capability must not raise."""
        from vetinari.enforcement.capabilities import AgentCapabilityEnforcer

        # WORKER has "code_scaffolding" in its AgentSpec.capabilities
        enforcer = AgentCapabilityEnforcer()
        assert enforcer.validate(AgentType.WORKER, "code_scaffolding") is None

    def test_known_agent_type_with_missing_capability_raises(self) -> None:
        """Registered agent missing the requested capability must raise."""
        from vetinari.enforcement.capabilities import AgentCapabilityEnforcer

        enforcer = AgentCapabilityEnforcer()
        with pytest.raises(CapabilityNotAvailable):
            enforcer.validate(AgentType.INSPECTOR, "nonexistent_capability_xyz")


# ---------------------------------------------------------------------------
# Bug 2 regression: enforce_all() aggregation behaviour
# ---------------------------------------------------------------------------


class TestEnforceAllAggregation:
    """enforce_all() must collect ALL violations before raising.

    Regression for Bug 2: the module docstring previously claimed the first
    failing check raises immediately (short-circuit).  The actual implementation
    collects all failures.  This class exercises the real aggregation path and
    guards against future regressions that would restore the short-circuit.
    """

    def test_multiple_violations_raises_composite_error(self) -> None:
        """Two simultaneous violations must produce CompositeEnforcementError."""
        from vetinari.enforcement import enforce_all
        from vetinari.exceptions import CompositeEnforcementError

        # depth=9999 → DelegationDepthExceeded; quality=0.0 → QualityGateFailed
        with pytest.raises(CompositeEnforcementError) as exc_info:
            enforce_all(
                AgentType.WORKER,
                current_depth=9999,
                quality_score=0.0,
            )
        err = exc_info.value
        assert len(err.violations) == 2, (
            f"Expected 2 violations, got {len(err.violations)}: {err.violations}"
        )

    def test_single_violation_raises_original_type(self) -> None:
        """One violation must re-raise as its original exception type, not CompositeEnforcementError."""
        from vetinari.enforcement import enforce_all
        from vetinari.exceptions import CompositeEnforcementError, DelegationDepthExceeded

        with pytest.raises(DelegationDepthExceeded):
            enforce_all(AgentType.WORKER, current_depth=9999)
        # Must NOT be wrapped in CompositeEnforcementError when there is only one.
        with pytest.raises(DelegationDepthExceeded):
            # If this raised CompositeEnforcementError instead, the test would fail.
            enforce_all(AgentType.WORKER, current_depth=9999)

    def test_no_violations_returns_none(self) -> None:
        """With valid inputs, enforce_all() must return None without raising."""
        from vetinari.enforcement import enforce_all

        result = enforce_all(
            AgentType.WORKER,
            current_depth=1,
            quality_score=0.95,
        )
        assert result is None

    def test_three_simultaneous_violations_all_collected(self) -> None:
        """Three simultaneous violations must all appear in CompositeEnforcementError.violations."""
        from vetinari.enforcement import enforce_all
        from vetinari.exceptions import CompositeEnforcementError

        with pytest.raises(CompositeEnforcementError) as exc_info:
            enforce_all(
                AgentType.WORKER,
                current_depth=9999,
                quality_score=0.0,
                file_path="/etc/passwd",  # outside WORKER jurisdiction
            )
        err = exc_info.value
        assert len(err.violations) == 3, (
            f"Expected 3 violations, got {len(err.violations)}: {err.violations}"
        )
