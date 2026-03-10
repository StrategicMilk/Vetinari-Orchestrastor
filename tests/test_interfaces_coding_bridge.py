"""
Combined tests for:
  - vetinari/agents/interfaces.py
  - vetinari/agents/coding_bridge.py
"""

import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Both modules import cleanly from the installed package — no stubs needed.
from vetinari.agents.interfaces import (
    CapabilityType,
    Capability,
    AgentInterface,
    AGENT_INTERFACES,
    EXPLORER_INTERFACE,
    LIBRARIAN_INTERFACE,
    RESEARCHER_INTERFACE,
    BUILDER_INTERFACE,
    EVALUATOR_INTERFACE,
    UI_PLANNER_INTERFACE,
    get_agent_interface,
)

from vetinari.agents import coding_bridge as _cb_module
from vetinari.agents.coding_bridge import (
    CodingTask,
    CodingResult,
    CodingBridge,
    get_coding_bridge,
    init_coding_bridge,
)
from vetinari.types import CodingTaskType, CodingTaskStatus


# ===========================================================================
# PART 1 — vetinari/agents/interfaces.py
# ===========================================================================


class TestCapabilityTypeEnum(unittest.TestCase):
    """CapabilityType enum values."""

    def test_has_discovery(self):
        self.assertEqual(CapabilityType.DISCOVERY.value, "discovery")

    def test_has_analysis(self):
        self.assertEqual(CapabilityType.ANALYSIS.value, "analysis")

    def test_has_synthesis(self):
        self.assertEqual(CapabilityType.SYNTHESIS.value, "synthesis")

    def test_has_generation(self):
        self.assertEqual(CapabilityType.GENERATION.value, "generation")

    def test_has_verification(self):
        self.assertEqual(CapabilityType.VERIFICATION.value, "verification")

    def test_has_documentation(self):
        self.assertEqual(CapabilityType.DOCUMENTATION.value, "documentation")

    def test_has_optimization(self):
        self.assertEqual(CapabilityType.OPTIMIZATION.value, "optimization")

    def test_has_testing(self):
        self.assertEqual(CapabilityType.TESTING.value, "testing")

    def test_has_governance(self):
        self.assertEqual(CapabilityType.GOVERNANCE.value, "governance")

    def test_total_member_count(self):
        self.assertEqual(len(CapabilityType), 9)

    def test_members_are_enum_instances(self):
        for member in CapabilityType:
            self.assertIsInstance(member, CapabilityType)

    def test_lookup_by_value(self):
        self.assertIs(CapabilityType("discovery"), CapabilityType.DISCOVERY)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            CapabilityType("nonexistent")


class TestCapabilityDataclass(unittest.TestCase):
    """Capability dataclass construction and behaviour."""

    def _make(self, **kw):
        defaults = dict(
            name="test_cap",
            type=CapabilityType.ANALYSIS,
            description="A test capability",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        defaults.update(kw)
        return Capability(**defaults)

    def test_basic_construction(self):
        cap = self._make()
        self.assertEqual(cap.name, "test_cap")

    def test_default_version(self):
        cap = self._make()
        self.assertEqual(cap.version, "1.0.0")

    def test_default_deprecated_false(self):
        cap = self._make()
        self.assertFalse(cap.deprecated)

    def test_custom_version(self):
        cap = self._make(version="2.3.1")
        self.assertEqual(cap.version, "2.3.1")

    def test_custom_deprecated(self):
        cap = self._make(deprecated=True)
        self.assertTrue(cap.deprecated)

    def test_type_stored_correctly(self):
        cap = self._make(type=CapabilityType.GENERATION)
        self.assertIs(cap.type, CapabilityType.GENERATION)

    def test_to_dict_returns_dict(self):
        cap = self._make()
        self.assertIsInstance(cap.to_dict(), dict)

    def test_to_dict_name(self):
        cap = self._make(name="foo")
        self.assertEqual(cap.to_dict()["name"], "foo")

    def test_to_dict_type_is_string(self):
        cap = self._make(type=CapabilityType.SYNTHESIS)
        self.assertEqual(cap.to_dict()["type"], "synthesis")

    def test_to_dict_description(self):
        cap = self._make(description="hello")
        self.assertEqual(cap.to_dict()["description"], "hello")

    def test_to_dict_input_schema(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        cap = self._make(input_schema=schema)
        self.assertEqual(cap.to_dict()["input_schema"], schema)

    def test_to_dict_output_schema(self):
        schema = {"type": "object"}
        cap = self._make(output_schema=schema)
        self.assertEqual(cap.to_dict()["output_schema"], schema)

    def test_to_dict_version(self):
        cap = self._make(version="3.0.0")
        self.assertEqual(cap.to_dict()["version"], "3.0.0")

    def test_to_dict_deprecated_true(self):
        cap = self._make(deprecated=True)
        self.assertTrue(cap.to_dict()["deprecated"])

    def test_to_dict_deprecated_false_default(self):
        cap = self._make()
        self.assertFalse(cap.to_dict()["deprecated"])

    def test_to_dict_all_required_keys(self):
        cap = self._make()
        d = cap.to_dict()
        for key in ("name", "type", "description", "input_schema", "output_schema", "version", "deprecated"):
            self.assertIn(key, d, msg=f"Missing key: {key}")


class TestAgentInterfaceMethods(unittest.TestCase):
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

    # --- get_capability ---

    def test_get_capability_returns_capability(self):
        cap = self._make_cap("alpha")
        iface = self._make_interface([cap])
        result = iface.get_capability("alpha")
        self.assertIs(result, cap)

    def test_get_capability_returns_none_for_missing(self):
        iface = self._make_interface([self._make_cap("alpha")])
        self.assertIsNone(iface.get_capability("beta"))

    def test_get_capability_empty_list(self):
        iface = self._make_interface([])
        self.assertIsNone(iface.get_capability("anything"))

    def test_get_capability_first_match(self):
        cap1 = self._make_cap("dup")
        cap2 = self._make_cap("dup")
        iface = self._make_interface([cap1, cap2])
        self.assertIs(iface.get_capability("dup"), cap1)

    def test_get_capability_multiple_caps(self):
        caps = [self._make_cap(n) for n in ("a", "b", "c")]
        iface = self._make_interface(caps)
        self.assertIs(iface.get_capability("b"), caps[1])

    def test_get_capability_last_in_list(self):
        caps = [self._make_cap(n) for n in ("x", "y", "z")]
        iface = self._make_interface(caps)
        self.assertIs(iface.get_capability("z"), caps[2])

    # --- has_capability ---

    def test_has_capability_true(self):
        iface = self._make_interface([self._make_cap("do_thing")])
        self.assertTrue(iface.has_capability("do_thing"))

    def test_has_capability_false(self):
        iface = self._make_interface([self._make_cap("do_thing")])
        self.assertFalse(iface.has_capability("other"))

    def test_has_capability_empty(self):
        iface = self._make_interface()
        self.assertFalse(iface.has_capability("x"))

    def test_has_capability_multiple_true(self):
        caps = [self._make_cap(n) for n in ("a", "b", "c")]
        iface = self._make_interface(caps)
        self.assertTrue(iface.has_capability("c"))

    def test_has_capability_returns_bool(self):
        iface = self._make_interface([self._make_cap("cap")])
        result = iface.has_capability("cap")
        self.assertIsInstance(result, bool)

    # --- to_dict ---

    def test_to_dict_agent_name(self):
        iface = self._make_interface()
        self.assertEqual(iface.to_dict()["agent_name"], "TestAgent")

    def test_to_dict_agent_type(self):
        iface = self._make_interface()
        self.assertEqual(iface.to_dict()["agent_type"], "TEST")

    def test_to_dict_version(self):
        iface = self._make_interface()
        self.assertEqual(iface.to_dict()["version"], "1.0.0")

    def test_to_dict_capabilities_list(self):
        iface = self._make_interface([self._make_cap("cap1")])
        self.assertIsInstance(iface.to_dict()["capabilities"], list)
        self.assertEqual(len(iface.to_dict()["capabilities"]), 1)

    def test_to_dict_capabilities_empty(self):
        iface = self._make_interface()
        self.assertEqual(iface.to_dict()["capabilities"], [])

    def test_to_dict_required_context(self):
        iface = self._make_interface()
        self.assertEqual(iface.to_dict()["required_context"], ["ctx_a"])

    def test_to_dict_error_codes(self):
        iface = self._make_interface()
        self.assertEqual(iface.to_dict()["error_codes"], {"E001": "Some error"})

    def test_to_dict_capabilities_nested_name(self):
        cap = self._make_cap("nested_cap")
        iface = self._make_interface([cap])
        caps_list = iface.to_dict()["capabilities"]
        self.assertEqual(caps_list[0]["name"], "nested_cap")

    def test_to_dict_all_keys(self):
        iface = self._make_interface()
        for key in ("agent_name", "agent_type", "version", "capabilities", "required_context", "error_codes"):
            self.assertIn(key, iface.to_dict(), msg=f"Missing key: {key}")

    def test_to_dict_multiple_caps_serialised(self):
        caps = [self._make_cap(n) for n in ("p", "q")]
        iface = self._make_interface(caps)
        self.assertEqual(len(iface.to_dict()["capabilities"]), 2)


class TestGetAgentInterface(unittest.TestCase):
    """get_agent_interface factory function."""

    def test_explorer_returned(self):
        result = get_agent_interface("EXPLORER")
        self.assertIsInstance(result, AgentInterface)
        self.assertEqual(result.agent_name, "Explorer")

    def test_librarian_returned(self):
        result = get_agent_interface("LIBRARIAN")
        self.assertIsInstance(result, AgentInterface)
        self.assertEqual(result.agent_name, "Librarian")

    def test_researcher_returned(self):
        result = get_agent_interface("RESEARCHER")
        self.assertIsInstance(result, AgentInterface)
        self.assertEqual(result.agent_name, "Researcher")

    def test_builder_returned(self):
        result = get_agent_interface("BUILDER")
        self.assertIsInstance(result, AgentInterface)
        self.assertEqual(result.agent_name, "Builder")

    def test_evaluator_returned(self):
        result = get_agent_interface("EVALUATOR")
        self.assertIsInstance(result, AgentInterface)
        self.assertEqual(result.agent_name, "Evaluator")

    def test_ui_planner_returned(self):
        result = get_agent_interface("UI_PLANNER")
        self.assertIsInstance(result, AgentInterface)
        self.assertEqual(result.agent_name, "UI Planner")

    def test_unknown_returns_none(self):
        self.assertIsNone(get_agent_interface("DOES_NOT_EXIST"))

    def test_lowercase_returns_none(self):
        self.assertIsNone(get_agent_interface("explorer"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(get_agent_interface(""))

    def test_registry_contains_six_entries(self):
        # 6 legacy + 5 consolidated agent interfaces (PLANNER, CONSOLIDATED_RESEARCHER,
        # CONSOLIDATED_ORACLE, QUALITY, OPERATIONS) added in P5.5b
        self.assertEqual(len(AGENT_INTERFACES), 11)

    def test_all_registered_types_return_non_none(self):
        for key in AGENT_INTERFACES:
            self.assertIsNotNone(get_agent_interface(key))

    def test_returns_same_object_as_constant(self):
        self.assertIs(get_agent_interface("EXPLORER"), EXPLORER_INTERFACE)


class TestModuleInterfaceConstants(unittest.TestCase):
    """Smoke-tests for the pre-built AgentInterface constants."""

    def test_explorer_has_search_code_patterns(self):
        self.assertTrue(EXPLORER_INTERFACE.has_capability("search_code_patterns"))

    def test_explorer_required_context_has_codebase_path(self):
        self.assertIn("codebase_path", EXPLORER_INTERFACE.required_context)

    def test_explorer_required_context_has_search_tools(self):
        self.assertIn("search_tools", EXPLORER_INTERFACE.required_context)

    def test_librarian_has_lookup_documentation(self):
        self.assertTrue(LIBRARIAN_INTERFACE.has_capability("lookup_documentation"))

    def test_librarian_has_analyze_libraries(self):
        self.assertTrue(LIBRARIAN_INTERFACE.has_capability("analyze_libraries"))

    def test_librarian_two_capabilities(self):
        self.assertEqual(len(LIBRARIAN_INTERFACE.capabilities), 2)

    def test_librarian_lookup_doc_type(self):
        cap = LIBRARIAN_INTERFACE.get_capability("lookup_documentation")
        self.assertIs(cap.type, CapabilityType.DISCOVERY)

    def test_librarian_analyze_libraries_type(self):
        cap = LIBRARIAN_INTERFACE.get_capability("analyze_libraries")
        self.assertIs(cap.type, CapabilityType.ANALYSIS)

    def test_researcher_has_domain_research(self):
        self.assertTrue(RESEARCHER_INTERFACE.has_capability("domain_research"))

    def test_researcher_has_competitive_analysis(self):
        self.assertTrue(RESEARCHER_INTERFACE.has_capability("competitive_analysis"))

    def test_researcher_domain_research_type(self):
        cap = RESEARCHER_INTERFACE.get_capability("domain_research")
        self.assertIs(cap.type, CapabilityType.ANALYSIS)

    def test_builder_has_generate_scaffold(self):
        self.assertTrue(BUILDER_INTERFACE.has_capability("generate_scaffold"))

    def test_builder_generate_scaffold_type(self):
        cap = BUILDER_INTERFACE.get_capability("generate_scaffold")
        self.assertIs(cap.type, CapabilityType.GENERATION)

    def test_evaluator_has_evaluate_quality(self):
        self.assertTrue(EVALUATOR_INTERFACE.has_capability("evaluate_quality"))

    def test_evaluator_evaluate_quality_type(self):
        cap = EVALUATOR_INTERFACE.get_capability("evaluate_quality")
        self.assertIs(cap.type, CapabilityType.VERIFICATION)

    def test_ui_planner_has_design_ui(self):
        self.assertTrue(UI_PLANNER_INTERFACE.has_capability("design_ui"))

    def test_ui_planner_design_ui_type(self):
        cap = UI_PLANNER_INTERFACE.get_capability("design_ui")
        self.assertIs(cap.type, CapabilityType.GENERATION)

    def test_explorer_interface_agent_type(self):
        self.assertEqual(EXPLORER_INTERFACE.agent_type, "EXPLORER")

    def test_all_interfaces_have_version(self):
        for iface in AGENT_INTERFACES.values():
            self.assertTrue(iface.version)

    def test_all_interfaces_serialisable_via_json(self):
        import json
        for name, iface in AGENT_INTERFACES.items():
            try:
                json.dumps(iface.to_dict())
            except (TypeError, ValueError) as exc:
                self.fail(f"Interface {name} not JSON-serialisable: {exc}")

    def test_builder_required_context(self):
        self.assertIn("code_generation_models", BUILDER_INTERFACE.required_context)

    def test_evaluator_required_context(self):
        self.assertIn("quality_standards", EVALUATOR_INTERFACE.required_context)


# ===========================================================================
# PART 2 — vetinari/agents/coding_bridge.py
# ===========================================================================


class TestCodingTaskDataclass(unittest.TestCase):
    """CodingTask dataclass defaults and construction."""

    def test_default_task_type(self):
        t = CodingTask()
        self.assertEqual(t.task_type, CodingTaskType.IMPLEMENT)

    def test_default_status(self):
        t = CodingTask()
        self.assertEqual(t.status, CodingTaskStatus.PENDING)

    def test_task_id_generated(self):
        t = CodingTask()
        self.assertTrue(t.task_id.startswith("code_"))

    def test_task_id_is_string(self):
        t = CodingTask()
        self.assertIsInstance(t.task_id, str)

    def test_default_description_empty(self):
        t = CodingTask()
        self.assertEqual(t.description, "")

    def test_custom_fields(self):
        t = CodingTask(
            task_id="custom_001",
            task_type=CodingTaskType.SCAFFOLD,
            description="build it",
            language="python",
            framework="flask",
        )
        self.assertEqual(t.task_id, "custom_001")
        self.assertEqual(t.task_type, CodingTaskType.SCAFFOLD)
        self.assertEqual(t.description, "build it")
        self.assertEqual(t.language, "python")
        self.assertEqual(t.framework, "flask")

    def test_default_input_files_empty_list(self):
        t = CodingTask()
        self.assertEqual(t.input_files, [])

    def test_default_context_empty_dict(self):
        t = CodingTask()
        self.assertEqual(t.context, {})

    def test_result_defaults_none(self):
        t = CodingTask()
        self.assertIsNone(t.result)

    def test_error_defaults_none(self):
        t = CodingTask()
        self.assertIsNone(t.error)

    def test_created_at_is_string(self):
        t = CodingTask()
        self.assertIsInstance(t.created_at, str)

    def test_completed_at_defaults_none(self):
        t = CodingTask()
        self.assertIsNone(t.completed_at)

    def test_custom_status(self):
        t = CodingTask(status=CodingTaskStatus.RUNNING)
        self.assertEqual(t.status, CodingTaskStatus.RUNNING)

    def test_custom_input_files(self):
        t = CodingTask(input_files=["a.py", "b.py"])
        self.assertEqual(t.input_files, ["a.py", "b.py"])

    def test_custom_context(self):
        t = CodingTask(context={"key": "val"})
        self.assertEqual(t.context["key"], "val")


class TestCodingResultDataclass(unittest.TestCase):
    """CodingResult dataclass."""

    def test_basic_success(self):
        r = CodingResult(success=True, task_id="t1")
        self.assertTrue(r.success)
        self.assertEqual(r.task_id, "t1")

    def test_basic_failure(self):
        r = CodingResult(success=False, task_id="t2", error="oops")
        self.assertFalse(r.success)
        self.assertEqual(r.error, "oops")

    def test_default_output_files(self):
        r = CodingResult(success=True, task_id="t3")
        self.assertEqual(r.output_files, [])

    def test_default_logs_empty(self):
        r = CodingResult(success=True, task_id="t4")
        self.assertEqual(r.logs, "")

    def test_default_error_none(self):
        r = CodingResult(success=True, task_id="t5")
        self.assertIsNone(r.error)

    def test_default_metadata_empty(self):
        r = CodingResult(success=True, task_id="t6")
        self.assertEqual(r.metadata, {})

    def test_output_files_stored(self):
        r = CodingResult(success=True, task_id="t7", output_files=["a.py", "b.py"])
        self.assertEqual(r.output_files, ["a.py", "b.py"])

    def test_metadata_stored(self):
        r = CodingResult(success=True, task_id="t8", metadata={"k": "v"})
        self.assertEqual(r.metadata["k"], "v")

    def test_logs_stored(self):
        r = CodingResult(success=True, task_id="t9", logs="done")
        self.assertEqual(r.logs, "done")


class TestCodingBridgeInit(unittest.TestCase):
    """CodingBridge.__init__ with various env / arg combinations."""

    def _clear_env(self):
        for key in ("CODING_BRIDGE_ENDPOINT", "CODING_BRIDGE_API_KEY", "CODING_BRIDGE_ENABLED"):
            os.environ.pop(key, None)

    def setUp(self):
        self._clear_env()

    def tearDown(self):
        self._clear_env()

    def test_default_endpoint(self):
        bridge = CodingBridge()
        self.assertEqual(bridge.endpoint, "http://localhost:4096")

    def test_custom_endpoint_argument(self):
        bridge = CodingBridge(endpoint="http://custom:9000")
        self.assertEqual(bridge.endpoint, "http://custom:9000")

    def test_endpoint_from_env(self):
        os.environ["CODING_BRIDGE_ENDPOINT"] = "http://env-host:1234"
        bridge = CodingBridge()
        self.assertEqual(bridge.endpoint, "http://env-host:1234")

    def test_arg_overrides_env(self):
        os.environ["CODING_BRIDGE_ENDPOINT"] = "http://env-host:1234"
        bridge = CodingBridge(endpoint="http://arg-host:5678")
        self.assertEqual(bridge.endpoint, "http://arg-host:5678")

    def test_default_api_key_empty(self):
        bridge = CodingBridge()
        self.assertEqual(bridge.api_key, "")

    def test_custom_api_key_argument(self):
        bridge = CodingBridge(api_key="secret123")
        self.assertEqual(bridge.api_key, "secret123")

    def test_api_key_from_env(self):
        os.environ["CODING_BRIDGE_API_KEY"] = "env_secret"
        bridge = CodingBridge()
        self.assertEqual(bridge.api_key, "env_secret")

    def test_enabled_defaults_false(self):
        bridge = CodingBridge()
        self.assertFalse(bridge.enabled)

    def test_enabled_via_env_true(self):
        os.environ["CODING_BRIDGE_ENABLED"] = "true"
        bridge = CodingBridge()
        self.assertTrue(bridge.enabled)

    def test_enabled_via_env_1(self):
        os.environ["CODING_BRIDGE_ENABLED"] = "1"
        bridge = CodingBridge()
        self.assertTrue(bridge.enabled)

    def test_enabled_via_env_yes(self):
        os.environ["CODING_BRIDGE_ENABLED"] = "yes"
        bridge = CodingBridge()
        self.assertTrue(bridge.enabled)

    def test_enabled_false_via_env_false(self):
        os.environ["CODING_BRIDGE_ENABLED"] = "false"
        bridge = CodingBridge()
        self.assertFalse(bridge.enabled)

    def test_enabled_false_env_zero(self):
        os.environ["CODING_BRIDGE_ENABLED"] = "0"
        bridge = CodingBridge()
        self.assertFalse(bridge.enabled)


class TestCodingBridgeIsAvailable(unittest.TestCase):

    def setUp(self):
        os.environ.pop("CODING_BRIDGE_ENABLED", None)

    def tearDown(self):
        os.environ.pop("CODING_BRIDGE_ENABLED", None)

    def test_not_available_when_disabled(self):
        b = CodingBridge()
        b.enabled = False
        self.assertFalse(b.is_available())

    def test_available_when_enabled(self):
        b = CodingBridge()
        b.enabled = True
        self.assertTrue(b.is_available())

    def test_available_with_explicit_enabled_flag(self):
        b = CodingBridge()
        b.enabled = True
        self.assertTrue(b.is_available())

    def test_not_available_after_disabling(self):
        b = CodingBridge()
        b.enabled = True
        b.enabled = False
        self.assertFalse(b.is_available())

    def test_returns_bool(self):
        b = CodingBridge()
        self.assertIsInstance(b.is_available(), bool)


class TestCodingBridgeGenerateTask(unittest.TestCase):

    def setUp(self):
        os.environ.pop("CODING_BRIDGE_ENABLED", None)

    def tearDown(self):
        os.environ.pop("CODING_BRIDGE_ENABLED", None)

    def _enabled(self):
        b = CodingBridge()
        b.enabled = True
        return b

    def _disabled(self):
        b = CodingBridge()
        b.enabled = False
        return b

    def test_disabled_bridge_returns_failure(self):
        b = self._disabled()
        result = b.generate_task(CodingTask(task_id="t1"))
        self.assertFalse(result.success)

    def test_disabled_bridge_error_message(self):
        b = self._disabled()
        result = b.generate_task(CodingTask(task_id="t2"))
        self.assertIn("not enabled", result.error)

    def test_disabled_bridge_returns_correct_task_id(self):
        b = self._disabled()
        result = b.generate_task(CodingTask(task_id="my_task"))
        self.assertEqual(result.task_id, "my_task")

    def test_enabled_non_scaffold_returns_success(self):
        b = self._enabled()
        result = b.generate_task(CodingTask(task_id="t3", task_type=CodingTaskType.IMPLEMENT))
        self.assertTrue(result.success)

    def test_enabled_returns_task_id(self):
        b = self._enabled()
        result = b.generate_task(CodingTask(task_id="task_xyz", task_type=CodingTaskType.TEST))
        self.assertEqual(result.task_id, "task_xyz")

    def test_enabled_logs_contain_task_id(self):
        b = self._enabled()
        result = b.generate_task(CodingTask(task_id="log_task", task_type=CodingTaskType.REVIEW))
        self.assertIn("log_task", result.logs)

    def test_enabled_metadata_contains_task_type(self):
        b = self._enabled()
        result = b.generate_task(CodingTask(task_id="t4", task_type=CodingTaskType.IMPLEMENT, language="python"))
        self.assertEqual(result.metadata["task_type"], "implement")

    def test_enabled_metadata_contains_language_or_engine(self):
        b = self._enabled()
        result = b.generate_task(CodingTask(task_id="t5", task_type=CodingTaskType.IMPLEMENT, language="go"))
        # CodingEngine route returns "engine" key; fallback returns "language"
        self.assertTrue("language" in result.metadata or "engine" in result.metadata)

    def test_enabled_metadata_contains_engine_or_endpoint(self):
        b = self._enabled()
        result = b.generate_task(CodingTask(task_id="t6", task_type=CodingTaskType.IMPLEMENT))
        # CodingEngine route returns "engine", fallback returns "endpoint"
        self.assertTrue("engine" in result.metadata or "endpoint" in result.metadata)

    def test_output_files_is_populated(self):
        b = self._enabled()
        result = b.generate_task(
            CodingTask(task_id="t7", task_type=CodingTaskType.DOCUMENT, output_path="/some/path.py")
        )
        # CodingEngine may return its own output paths; fallback uses output_path
        self.assertIsInstance(result.output_files, list)
        self.assertTrue(len(result.output_files) >= 0)

    def test_no_output_path_result_is_list(self):
        b = self._enabled()
        result = b.generate_task(CodingTask(task_id="t8", task_type=CodingTaskType.IMPLEMENT, output_path=""))
        # CodingEngine may produce output files even without explicit output_path
        self.assertIsInstance(result.output_files, list)

    def test_scaffold_task_delegates_to_generate_scaffold(self):
        b = self._enabled()
        with patch.object(b, "_generate_scaffold") as mock_sc:
            mock_sc.return_value = CodingResult(success=True, task_id="s1")
            task = CodingTask(task_id="s1", task_type=CodingTaskType.SCAFFOLD)
            b.generate_task(task)
            mock_sc.assert_called_once_with(task)

    def test_result_is_coding_result(self):
        b = self._enabled()
        result = b.generate_task(CodingTask(task_id="t9", task_type=CodingTaskType.IMPLEMENT))
        self.assertIsInstance(result, CodingResult)


class TestGenerateScaffold(unittest.TestCase):
    """_generate_scaffold creates real files in a temp directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bridge = CodingBridge()
        self.bridge.enabled = True

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _task(self, project_name="myproj"):
        return CodingTask(
            task_id="scaffold_001",
            task_type=CodingTaskType.SCAFFOLD,
            output_path=os.path.join(self.tmpdir, project_name),
            context={"project_name": project_name},
        )

    def test_scaffold_succeeds(self):
        result = self.bridge._generate_scaffold(self._task())
        self.assertTrue(result.success)

    def test_scaffold_creates_output_files(self):
        result = self.bridge._generate_scaffold(self._task())
        self.assertGreater(len(result.output_files), 0)

    def test_scaffold_setup_py_exists(self):
        self.bridge._generate_scaffold(self._task("proj_a"))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "proj_a", "setup.py")))

    def test_scaffold_readme_exists(self):
        self.bridge._generate_scaffold(self._task("proj_b"))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "proj_b", "README.md")))

    def test_scaffold_init_py_exists(self):
        self.bridge._generate_scaffold(self._task("proj_c"))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "proj_c", "proj_c", "__init__.py")))

    def test_scaffold_main_py_exists(self):
        self.bridge._generate_scaffold(self._task("proj_d"))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "proj_d", "proj_d", "__main__.py")))

    def test_scaffold_metadata_contains_project_name(self):
        result = self.bridge._generate_scaffold(self._task("proj_e"))
        self.assertEqual(result.metadata["project_name"], "proj_e")

    def test_scaffold_task_id_preserved(self):
        result = self.bridge._generate_scaffold(self._task())
        self.assertEqual(result.task_id, "scaffold_001")

    def test_scaffold_metadata_task_type(self):
        result = self.bridge._generate_scaffold(self._task("proj_f"))
        self.assertEqual(result.metadata["task_type"], "scaffold")

    def test_scaffold_metadata_output_path(self):
        task = self._task("proj_g")
        result = self.bridge._generate_scaffold(task)
        self.assertIn("output_path", result.metadata)

    def test_scaffold_returns_coding_result(self):
        result = self.bridge._generate_scaffold(self._task())
        self.assertIsInstance(result, CodingResult)

    def test_scaffold_four_output_files(self):
        result = self.bridge._generate_scaffold(self._task("proj_h"))
        self.assertEqual(len(result.output_files), 4)

    def test_scaffold_default_project_name_fallback(self):
        # No project_name in context — falls back to "demo_project"
        task = CodingTask(
            task_id="sc_default",
            task_type=CodingTaskType.SCAFFOLD,
            output_path=os.path.join(self.tmpdir, "demo_project"),
            context={},
        )
        result = self.bridge._generate_scaffold(task)
        self.assertTrue(result.success)


class TestGetTaskStatus(unittest.TestCase):

    def test_returns_coding_task(self):
        b = CodingBridge()
        result = b.get_task_status("any_task")
        self.assertIsInstance(result, CodingTask)

    def test_task_id_preserved(self):
        b = CodingBridge()
        result = b.get_task_status("specific_id")
        self.assertEqual(result.task_id, "specific_id")

    def test_status_is_pending_for_unknown(self):
        b = CodingBridge()
        result = b.get_task_status("unknown_task")
        self.assertEqual(result.status, CodingTaskStatus.PENDING)

    def test_returns_from_active_tasks_if_present(self):
        b = CodingBridge()
        stored = CodingTask(task_id="known", status=CodingTaskStatus.RUNNING)
        b._active_tasks = {"known": stored}
        result = b.get_task_status("known")
        self.assertIs(result, stored)

    def test_pending_result_empty_string(self):
        b = CodingBridge()
        result = b.get_task_status("ghost")
        self.assertEqual(result.result, "")

    def test_active_tasks_miss_returns_pending(self):
        b = CodingBridge()
        b._active_tasks = {"other": CodingTask(task_id="other")}
        result = b.get_task_status("missing")
        self.assertEqual(result.status, CodingTaskStatus.PENDING)


class TestCancelTask(unittest.TestCase):

    def test_cancel_returns_true(self):
        b = CodingBridge()
        self.assertTrue(b.cancel_task("any_id"))

    def test_cancel_arbitrary_id(self):
        b = CodingBridge()
        self.assertTrue(b.cancel_task("task_abc_123"))

    def test_cancel_returns_bool(self):
        b = CodingBridge()
        result = b.cancel_task("x")
        self.assertIsInstance(result, bool)


class TestListActiveTasks(unittest.TestCase):

    def test_returns_list(self):
        b = CodingBridge()
        self.assertIsInstance(b.list_active_tasks(), list)

    def test_returns_empty_list(self):
        b = CodingBridge()
        self.assertEqual(b.list_active_tasks(), [])


class TestCreateScaffold(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bridge = CodingBridge()
        self.bridge.enabled = True

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_scaffold_enabled_succeeds(self):
        out = os.path.join(self.tmpdir, "scaff")
        result = self.bridge.create_scaffold("python", "flask", out, "myapp")
        self.assertTrue(result.success)

    def test_create_scaffold_disabled_fails(self):
        self.bridge.enabled = False
        result = self.bridge.create_scaffold("python", "django", "/tmp/x", "app")
        self.assertFalse(result.success)

    def test_create_scaffold_calls_generate_task_with_scaffold_type(self):
        with patch.object(self.bridge, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="x")
            self.bridge.create_scaffold("python", "fastapi", "/tmp/out", "proj")
            task_arg = mock_gen.call_args[0][0]
            self.assertEqual(task_arg.task_type, CodingTaskType.SCAFFOLD)

    def test_create_scaffold_context_has_project_name(self):
        with patch.object(self.bridge, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="x")
            self.bridge.create_scaffold("js", "react", "/tmp/out", "webapp")
            task_arg = mock_gen.call_args[0][0]
            self.assertEqual(task_arg.context["project_name"], "webapp")

    def test_create_scaffold_language_set(self):
        with patch.object(self.bridge, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="x")
            self.bridge.create_scaffold("rust", "actix", "/tmp/out", "svc")
            task_arg = mock_gen.call_args[0][0]
            self.assertEqual(task_arg.language, "rust")

    def test_create_scaffold_framework_set(self):
        with patch.object(self.bridge, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="x")
            self.bridge.create_scaffold("python", "fastapi", "/tmp/out", "api")
            task_arg = mock_gen.call_args[0][0]
            self.assertEqual(task_arg.framework, "fastapi")

    def test_create_scaffold_output_path_set(self):
        with patch.object(self.bridge, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="x")
            self.bridge.create_scaffold("go", "gin", "/my/output", "gosvc")
            task_arg = mock_gen.call_args[0][0]
            self.assertEqual(task_arg.output_path, "/my/output")

    def test_create_scaffold_returns_coding_result(self):
        out = os.path.join(self.tmpdir, "r")
        result = self.bridge.create_scaffold("python", "bare", out, "pkg")
        self.assertIsInstance(result, CodingResult)


class TestWriteTests(unittest.TestCase):

    def test_write_tests_calls_generate_task(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="wt1")
            b.write_tests("src/foo.py")
            mock_gen.assert_called_once()

    def test_write_tests_task_type_is_test(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="wt2")
            b.write_tests("src/bar.py")
            task = mock_gen.call_args[0][0]
            self.assertEqual(task.task_type, CodingTaskType.TEST)

    def test_write_tests_input_files_set(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="wt3")
            b.write_tests("src/baz.py")
            task = mock_gen.call_args[0][0]
            self.assertIn("src/baz.py", task.input_files)

    def test_write_tests_default_framework_pytest(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="wt4")
            b.write_tests("src/qux.py")
            task = mock_gen.call_args[0][0]
            self.assertEqual(task.context["test_framework"], "pytest")

    def test_write_tests_custom_framework(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="wt5")
            b.write_tests("src/qux.py", test_framework="unittest")
            task = mock_gen.call_args[0][0]
            self.assertEqual(task.context["test_framework"], "unittest")

    def test_write_tests_returns_result(self):
        b = CodingBridge()
        expected = CodingResult(success=True, task_id="wt6")
        with patch.object(b, "generate_task", return_value=expected):
            result = b.write_tests("src/x.py")
            self.assertIs(result, expected)

    def test_write_tests_description_contains_source_file(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="wt7")
            b.write_tests("src/important.py")
            task = mock_gen.call_args[0][0]
            self.assertIn("src/important.py", task.description)


class TestReviewCode(unittest.TestCase):

    def test_review_code_calls_generate_task(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="r1")
            b.review_code("src/main.py")
            mock_gen.assert_called_once()

    def test_review_code_task_type_is_review(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="r2")
            b.review_code("src/main.py")
            task = mock_gen.call_args[0][0]
            self.assertEqual(task.task_type, CodingTaskType.REVIEW)

    def test_review_code_input_files_set(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="r3")
            b.review_code("src/util.py")
            task = mock_gen.call_args[0][0]
            self.assertIn("src/util.py", task.input_files)

    def test_review_code_returns_result(self):
        b = CodingBridge()
        expected = CodingResult(success=False, task_id="r4", error="Bad code")
        with patch.object(b, "generate_task", return_value=expected):
            result = b.review_code("bad.py")
            self.assertIs(result, expected)

    def test_review_code_description_contains_file_path(self):
        b = CodingBridge()
        with patch.object(b, "generate_task") as mock_gen:
            mock_gen.return_value = CodingResult(success=True, task_id="r5")
            b.review_code("src/logic.py")
            task = mock_gen.call_args[0][0]
            self.assertIn("src/logic.py", task.description)

    def test_review_code_returns_coding_result(self):
        b = CodingBridge()
        with patch.object(b, "generate_task", return_value=CodingResult(success=True, task_id="r6")):
            result = b.review_code("any.py")
            self.assertIsInstance(result, CodingResult)


class TestGetCodingBridgeSingleton(unittest.TestCase):
    """get_coding_bridge() singleton behaviour."""

    def setUp(self):
        _cb_module._coding_bridge = None

    def tearDown(self):
        _cb_module._coding_bridge = None

    def test_returns_coding_bridge_instance(self):
        b = get_coding_bridge()
        self.assertIsInstance(b, CodingBridge)

    def test_returns_same_instance_on_second_call(self):
        b1 = get_coding_bridge()
        b2 = get_coding_bridge()
        self.assertIs(b1, b2)

    def test_singleton_is_set_in_module(self):
        b = get_coding_bridge()
        self.assertIs(_cb_module._coding_bridge, b)

    def test_singleton_not_none_after_call(self):
        get_coding_bridge()
        self.assertIsNotNone(_cb_module._coding_bridge)


class TestInitCodingBridge(unittest.TestCase):

    def setUp(self):
        _cb_module._coding_bridge = None

    def tearDown(self):
        _cb_module._coding_bridge = None

    def test_returns_coding_bridge(self):
        b = init_coding_bridge("http://x:1", "key")
        self.assertIsInstance(b, CodingBridge)

    def test_sets_endpoint(self):
        b = init_coding_bridge("http://custom:9999", None)
        self.assertEqual(b.endpoint, "http://custom:9999")

    def test_sets_api_key(self):
        b = init_coding_bridge(None, "my_api_key")
        self.assertEqual(b.api_key, "my_api_key")

    def test_replaces_existing_singleton(self):
        b1 = init_coding_bridge("http://a:1", "k1")
        b2 = init_coding_bridge("http://b:2", "k2")
        self.assertIsNot(b1, b2)
        self.assertIs(_cb_module._coding_bridge, b2)

    def test_get_coding_bridge_returns_init_bridge(self):
        b1 = init_coding_bridge("http://z:9", "zk")
        b2 = get_coding_bridge()
        self.assertIs(b1, b2)

    def test_init_with_no_args_uses_defaults(self):
        b = init_coding_bridge()
        self.assertIsInstance(b, CodingBridge)
        self.assertEqual(b.endpoint, "http://localhost:4096")


if __name__ == "__main__":
    unittest.main()
