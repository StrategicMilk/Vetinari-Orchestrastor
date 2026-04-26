"""Tests for vetinari/agents/interfaces.py.

Validates the 3-agent interface model: FOREMAN, WORKER, and INSPECTOR.
"""

from __future__ import annotations

import json

import pytest

from vetinari.agents.interfaces import (
    AGENT_INTERFACES,
    FOREMAN_INTERFACE,
    INSPECTOR_INTERFACE,
    WORKER_INTERFACE,
    AgentInterface,
    Capability,
    CapabilityType,
    get_agent_interface,
)
from vetinari.types import AgentType

# ===========================================================================
# CapabilityType enum
# ===========================================================================


class TestCapabilityTypeEnum:
    """CapabilityType enum values."""

    def test_has_discovery(self):
        assert CapabilityType.DISCOVERY.value == "discovery"

    def test_has_analysis(self):
        assert CapabilityType.ANALYSIS.value == "analysis"

    def test_has_synthesis(self):
        assert CapabilityType.SYNTHESIS.value == "synthesis"

    def test_has_generation(self):
        assert CapabilityType.GENERATION.value == "generation"

    def test_has_verification(self):
        assert CapabilityType.VERIFICATION.value == "verification"

    def test_has_documentation(self):
        assert CapabilityType.DOCUMENTATION.value == "documentation"

    def test_has_optimization(self):
        assert CapabilityType.OPTIMIZATION.value == "optimization"

    def test_has_testing(self):
        assert CapabilityType.TESTING.value == "testing"

    def test_has_governance(self):
        assert CapabilityType.GOVERNANCE.value == "governance"

    def test_total_member_count(self):
        assert len(CapabilityType) == 9

    def test_members_are_enum_instances(self):
        for member in CapabilityType:
            assert isinstance(member, CapabilityType)

    def test_lookup_by_value(self):
        assert CapabilityType("discovery") is CapabilityType.DISCOVERY

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="not a valid CapabilityType"):
            CapabilityType("nonexistent")


# ===========================================================================
# Capability dataclass
# ===========================================================================


class TestCapabilityDataclass:
    """Capability dataclass construction and behaviour."""

    def _make(self, **kw):
        defaults = {
            "name": "test_cap",
            "type": CapabilityType.ANALYSIS,
            "description": "A test capability",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        }
        defaults.update(kw)
        return Capability(**defaults)

    def test_basic_construction(self):
        cap = self._make()
        assert cap.name == "test_cap"

    def test_default_version(self):
        cap = self._make()
        assert cap.version == "1.0.0"

    def test_default_deprecated_false(self):
        cap = self._make()
        assert not cap.deprecated

    def test_custom_version(self):
        cap = self._make(version="2.3.1")
        assert cap.version == "2.3.1"

    def test_custom_deprecated(self):
        cap = self._make(deprecated=True)
        assert cap.deprecated

    def test_type_stored_correctly(self):
        cap = self._make(type=CapabilityType.GENERATION)
        assert cap.type is CapabilityType.GENERATION

    def test_to_dict_returns_dict(self):
        cap = self._make()
        assert isinstance(cap.to_dict(), dict)

    def test_to_dict_name(self):
        cap = self._make(name="foo")
        assert cap.to_dict()["name"] == "foo"

    def test_to_dict_type_is_string(self):
        cap = self._make(type=CapabilityType.SYNTHESIS)
        assert cap.to_dict()["type"] == "synthesis"

    def test_to_dict_description(self):
        cap = self._make(description="hello")
        assert cap.to_dict()["description"] == "hello"

    def test_to_dict_input_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        cap = self._make(input_schema=schema)
        assert cap.to_dict()["input_schema"] == schema

    def test_to_dict_output_schema(self):
        schema = {"type": "object"}
        cap = self._make(output_schema=schema)
        assert cap.to_dict()["output_schema"] == schema

    def test_to_dict_version(self):
        cap = self._make(version="3.0.0")
        assert cap.to_dict()["version"] == "3.0.0"

    def test_to_dict_deprecated_true(self):
        cap = self._make(deprecated=True)
        assert cap.to_dict()["deprecated"]

    def test_to_dict_deprecated_false_default(self):
        cap = self._make()
        assert not cap.to_dict()["deprecated"]

    def test_to_dict_all_required_keys(self):
        cap = self._make()
        d = cap.to_dict()
        for key in ("name", "type", "description", "input_schema", "output_schema", "version", "deprecated"):
            assert key in d, f"Missing key: {key}"


# ===========================================================================
# AgentInterface methods
# ===========================================================================


class TestAgentInterfaceMethods:
    """AgentInterface.get_capability / has_capability / to_dict."""

    def _make_cap(self, name):
        return Capability(
            name=name,
            type=CapabilityType.ANALYSIS,
            description=f"Capability {name}",
            input_schema={},
            output_schema={},
        )

    def _make_interface(self, caps=None):
        return AgentInterface(
            agent_name="TestAgent",
            agent_type="TEST",
            version="1.0.0",
            capabilities=caps or [],
            required_context=["ctx_a"],
            error_codes={"E001": "Some error"},
        )

    def test_get_capability_returns_capability(self):
        cap = self._make_cap("alpha")
        iface = self._make_interface([cap])
        result = iface.get_capability("alpha")
        assert result is cap

    def test_get_capability_returns_none_for_missing(self):
        iface = self._make_interface([self._make_cap("alpha")])
        assert iface.get_capability("beta") is None

    def test_get_capability_empty_list(self):
        iface = self._make_interface([])
        assert iface.get_capability("anything") is None

    def test_get_capability_first_match(self):
        cap1 = self._make_cap("dup")
        cap2 = self._make_cap("dup")
        iface = self._make_interface([cap1, cap2])
        assert iface.get_capability("dup") is cap1

    def test_get_capability_multiple_caps(self):
        caps = [self._make_cap(n) for n in ("a", "b", "c")]
        iface = self._make_interface(caps)
        assert iface.get_capability("b") is caps[1]

    def test_get_capability_last_in_list(self):
        caps = [self._make_cap(n) for n in ("x", "y", "z")]
        iface = self._make_interface(caps)
        assert iface.get_capability("z") is caps[2]

    def test_has_capability_true(self):
        iface = self._make_interface([self._make_cap("do_thing")])
        assert iface.has_capability("do_thing")

    def test_has_capability_false(self):
        iface = self._make_interface([self._make_cap("do_thing")])
        assert not iface.has_capability("other")

    def test_has_capability_empty(self):
        iface = self._make_interface()
        assert not iface.has_capability("x")

    def test_has_capability_multiple_true(self):
        caps = [self._make_cap(n) for n in ("a", "b", "c")]
        iface = self._make_interface(caps)
        assert iface.has_capability("c")

    def test_has_capability_returns_bool(self):
        iface = self._make_interface([self._make_cap("cap")])
        result = iface.has_capability("cap")
        assert isinstance(result, bool)

    def test_to_dict_agent_name(self):
        iface = self._make_interface()
        assert iface.to_dict()["agent_name"] == "TestAgent"

    def test_to_dict_agent_type(self):
        iface = self._make_interface()
        assert iface.to_dict()["agent_type"] == "TEST"

    def test_to_dict_version(self):
        iface = self._make_interface()
        assert iface.to_dict()["version"] == "1.0.0"

    def test_to_dict_capabilities_list(self):
        iface = self._make_interface([self._make_cap("cap1")])
        assert isinstance(iface.to_dict()["capabilities"], list)
        assert len(iface.to_dict()["capabilities"]) == 1

    def test_to_dict_capabilities_empty(self):
        iface = self._make_interface()
        assert iface.to_dict()["capabilities"] == []

    def test_to_dict_required_context(self):
        iface = self._make_interface()
        assert iface.to_dict()["required_context"] == ["ctx_a"]

    def test_to_dict_error_codes(self):
        iface = self._make_interface()
        assert iface.to_dict()["error_codes"] == {"E001": "Some error"}

    def test_to_dict_capabilities_nested_name(self):
        cap = self._make_cap("nested_cap")
        iface = self._make_interface([cap])
        caps_list = iface.to_dict()["capabilities"]
        assert caps_list[0]["name"] == "nested_cap"

    def test_to_dict_all_keys(self):
        iface = self._make_interface()
        for key in ("agent_name", "agent_type", "version", "capabilities", "required_context", "error_codes"):
            assert key in iface.to_dict(), f"Missing key: {key}"

    def test_to_dict_multiple_caps_serialised(self):
        caps = [self._make_cap(n) for n in ("p", "q")]
        iface = self._make_interface(caps)
        assert len(iface.to_dict()["capabilities"]) == 2


# ===========================================================================
# get_agent_interface factory
# ===========================================================================


class TestGetAgentInterface:
    """get_agent_interface factory function."""

    def test_foreman_returned(self):
        result = get_agent_interface(AgentType.FOREMAN.value)
        assert isinstance(result, AgentInterface)
        assert result.agent_name == "Foreman"

    def test_worker_returned(self):
        result = get_agent_interface(AgentType.WORKER.value)
        assert isinstance(result, AgentInterface)
        assert result.agent_name == "Worker"

    def test_inspector_returned(self):
        result = get_agent_interface(AgentType.INSPECTOR.value)
        assert isinstance(result, AgentInterface)
        assert result.agent_name == "Inspector"

    def test_unknown_returns_none(self):
        assert get_agent_interface("DOES_NOT_EXIST") is None

    def test_lowercase_returns_none(self):
        assert get_agent_interface("worker") is None

    def test_empty_string_returns_none(self):
        assert get_agent_interface("") is None

    def test_registry_contains_three_entries(self):
        assert len(AGENT_INTERFACES) == 3

    def test_all_registered_types_return_non_none(self):
        for key in AGENT_INTERFACES:
            result = get_agent_interface(key)
            assert isinstance(result, AgentInterface), f"Expected AgentInterface for key {key!r}"

    def test_returns_same_object_as_constant_foreman(self):
        assert get_agent_interface(AgentType.FOREMAN.value) is FOREMAN_INTERFACE

    def test_returns_same_object_as_constant_worker(self):
        assert get_agent_interface(AgentType.WORKER.value) is WORKER_INTERFACE

    def test_returns_same_object_as_constant_inspector(self):
        assert get_agent_interface(AgentType.INSPECTOR.value) is INSPECTOR_INTERFACE


# ===========================================================================
# Pre-built AgentInterface constants
# ===========================================================================


class TestModuleInterfaceConstants:
    """Smoke-tests for the pre-built AgentInterface constants."""

    def test_foreman_agent_type(self):
        assert FOREMAN_INTERFACE.agent_type == AgentType.FOREMAN.value

    def test_foreman_has_goal_decomposition(self):
        assert FOREMAN_INTERFACE.has_capability("goal_decomposition")

    def test_foreman_has_task_sequencing(self):
        assert FOREMAN_INTERFACE.has_capability("task_sequencing")

    def test_foreman_has_user_clarification(self):
        assert FOREMAN_INTERFACE.has_capability("user_clarification")

    def test_foreman_has_plan_consolidation(self):
        assert FOREMAN_INTERFACE.has_capability("plan_consolidation")

    def test_foreman_has_context_management(self):
        assert FOREMAN_INTERFACE.has_capability("context_management")

    def test_foreman_has_dependency_resolution(self):
        assert FOREMAN_INTERFACE.has_capability("dependency_resolution")

    def test_foreman_required_context_has_goal(self):
        assert "goal" in FOREMAN_INTERFACE.required_context

    def test_worker_agent_type(self):
        assert WORKER_INTERFACE.agent_type == AgentType.WORKER.value

    def test_worker_has_build(self):
        assert WORKER_INTERFACE.has_capability("build")

    def test_worker_has_code_discovery(self):
        assert WORKER_INTERFACE.has_capability("code_discovery")

    def test_worker_has_domain_research(self):
        assert WORKER_INTERFACE.has_capability("domain_research")

    def test_worker_has_architecture_decision(self):
        assert WORKER_INTERFACE.has_capability("architecture_decision")

    def test_worker_has_synthesis(self):
        assert WORKER_INTERFACE.has_capability("synthesis")

    def test_worker_has_documentation_generation(self):
        assert WORKER_INTERFACE.has_capability("documentation_generation")

    def test_worker_build_capability_type(self):
        cap = WORKER_INTERFACE.get_capability("build")
        assert cap.type is CapabilityType.GENERATION

    def test_worker_code_discovery_capability_type(self):
        cap = WORKER_INTERFACE.get_capability("code_discovery")
        assert cap.type is CapabilityType.DISCOVERY

    def test_inspector_agent_type(self):
        assert INSPECTOR_INTERFACE.agent_type == AgentType.INSPECTOR.value

    def test_inspector_has_code_review(self):
        assert INSPECTOR_INTERFACE.has_capability("code_review")

    def test_inspector_has_security_audit(self):
        assert INSPECTOR_INTERFACE.has_capability("security_audit")

    def test_inspector_has_test_generation(self):
        assert INSPECTOR_INTERFACE.has_capability("test_generation")

    def test_inspector_has_code_simplification(self):
        assert INSPECTOR_INTERFACE.has_capability("code_simplification")

    def test_inspector_code_review_type(self):
        cap = INSPECTOR_INTERFACE.get_capability("code_review")
        assert cap.type is CapabilityType.VERIFICATION

    def test_inspector_security_audit_type(self):
        cap = INSPECTOR_INTERFACE.get_capability("security_audit")
        assert cap.type is CapabilityType.GOVERNANCE

    def test_all_interfaces_have_version(self):
        for iface in AGENT_INTERFACES.values():
            assert iface.version

    def test_all_interfaces_serialisable_via_json(self):
        for name, iface in AGENT_INTERFACES.items():
            try:
                serialised = json.dumps(iface.to_dict())
            except (TypeError, ValueError) as exc:
                self.fail(f"Interface {name} not JSON-serialisable: {exc}")
            assert isinstance(serialised, str), f"Interface {name} must serialise to a JSON string"
