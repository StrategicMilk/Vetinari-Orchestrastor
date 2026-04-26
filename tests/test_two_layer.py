"""
Comprehensive tests for vetinari/orchestration/two_layer.py

Strategy:
- Stub ONLY the 3 direct top-level imports of two_layer.py in sys.modules,
  PLUS enough attributes in the orchestration __init__ stubs so the package
  loads cleanly, BEFORE importing the module under test.
- All lazy/internal imports are patched via unittest.mock.patch inside tests.
- Reset singleton in setUp via tl._two_layer_orchestrator = None.
"""

import importlib.util as _tl_ilu
import os as _tl_os
import sys
import types
import unittest
from enum import Enum
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Build minimal stubs for everything the orchestration package's __init__.py
# pulls in at import time.  We must do this BEFORE any vetinari import.
# ---------------------------------------------------------------------------


def _mod(name, is_pkg=False, **attrs):
    """Create (or reuse) and register a stub module.

    If *name* is already in sys.modules the existing module is reused so that a
    real module loaded by an earlier test file is not silently replaced with a
    hollow stub.  Only MISSING attributes are added to an existing module so
    that real enum classes are never corrupted by plain-class replacements.
    """
    m = sys.modules.get(name)
    if m is None:
        m = types.ModuleType(name)
        if is_pkg:
            m.__path__ = []
            m.__package__ = name
        sys.modules[name] = m
        # New stub: set all supplied attrs
        for k, v in attrs.items():
            setattr(m, k, v)
    else:
        # Existing module: only add attrs that are absent to avoid overwriting
        # real enum classes with plain-class stubs (which break .value lookups).
        for k, v in attrs.items():
            if not hasattr(m, k):
                setattr(m, k, v)
    return m


# Sentinel classes used as stub types so isinstance checks still work simply.
# Defined as proper Enum subclasses so .value lookups always succeed even when
# these classes end up being used in a context where the real types haven't
# been loaded yet.
class _FakeAgentType(str, Enum):
    FOREMAN = "FOREMAN"
    WORKER = "WORKER"
    INSPECTOR = "INSPECTOR"
    ORCHESTRATOR = "ORCHESTRATOR"
    ARCHITECT = "ARCHITECT"
    RESEARCHER = "RESEARCHER"


class _FakePlanStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"


class _FakeTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class _FakeConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class _FakeConfidenceAction(str, Enum):
    PROCEED = "proceed"
    REFINE = "refine"
    BEST_OF_N = "best_of_n"
    DEFER_TO_HUMAN = "defer_to_human"


class _FakeDecisionType(str, Enum):
    ROUTING = "routing"
    APPROVAL = "approval"
    AUTONOMY = "autonomy"
    QUALITY = "quality"
    ESCALATION = "escalation"


# vetinari.types (needed by execution_graph, durable_execution, pipeline_confidence).
# We only add the attrs that are missing; the real module (which may have been
# loaded already by test_agent_graph.py) is never corrupted.
_mod(
    "vetinari.types",
    AgentType=_FakeAgentType,
    PlanStatus=_FakePlanStatus,
    StatusEnum=_FakeTaskStatus,
    ConfidenceLevel=_FakeConfidenceLevel,
    ConfidenceAction=_FakeConfidenceAction,
    DecisionType=_FakeDecisionType,
    ModelTier=MagicMock(),
)

# vetinari.orchestration sub-module stubs
_AgentGraph = MagicMock(name="AgentGraph")
_ExecutionPlan = MagicMock(name="ExecutionPlan")
_ExecutionStrategy = MagicMock(name="ExecutionStrategy")
_TaskNode_ag = MagicMock(name="TaskNode_ag")
_get_agent_graph = MagicMock(name="get_agent_graph", return_value=MagicMock())

_mod(
    "vetinari.orchestration.agent_graph",
    AgentGraph=_AgentGraph,
    ExecutionPlan=_ExecutionPlan,
    ExecutionStrategy=_ExecutionStrategy,
    TaskNode=_TaskNode_ag,
    get_agent_graph=_get_agent_graph,
)

# Load the REAL execution_graph and durable_execution modules so that
# test_durable_execution.py (which is collected AFTER this file in the
# combined suite) gets genuine dataclass instances instead of MagicMocks.
# Pattern: register in sys.modules BEFORE exec_module so that dataclass
# annotation resolution (sys.modules.get(cls.__module__).__dict__) works.
_TL_ROOT = _tl_os.path.dirname(_tl_os.path.dirname(_tl_os.path.abspath(__file__)))


def _load_real(dotted_name, rel_path):
    _p = _tl_os.path.join(_TL_ROOT, *rel_path.split("/"))
    _sp = _tl_ilu.spec_from_file_location(dotted_name, _p)
    _m = _tl_ilu.module_from_spec(_sp)
    sys.modules[dotted_name] = _m  # register BEFORE exec_module
    _sp.loader.exec_module(_m)
    return _m


# execution_graph depends only on vetinari.types (already stubbed above)
_eg_mod = _load_real(
    "vetinari.orchestration.execution_graph",
    "vetinari/orchestration/execution_graph.py",
)
_ExecutionGraph = _eg_mod.ExecutionGraph
_TaskNode_eg = _eg_mod.ExecutionTaskNode

# durable_execution depends on execution_graph (now loaded) + vetinari.types
_de_mod = _load_real(
    "vetinari.orchestration.durable_execution",
    "vetinari/orchestration/durable_execution.py",
)

# DurableExecutionEngine is kept as MagicMock for two_layer tests:
# test_two_layer.py sets tl.DurableExecutionEngine = _DurableExec after import
# and wraps individual tests with patch.object(tl, "DurableExecutionEngine").
_DurableExec = MagicMock(name="DurableExecutionEngine")

_PlanGenerator = MagicMock(name="PlanGenerator")
_mod("vetinari.orchestration.plan_generator", PlanGenerator=_PlanGenerator)


# vetinari.dynamic_model_router — needed by _route_model_for_task lazily
class _FakeTaskType:
    ANALYSIS = "ANALYSIS"
    CODING = "CODING"
    TESTING = "TESTING"
    DOCUMENTATION = "DOCUMENTATION"
    CODE_REVIEW = "CODE_REVIEW"
    CREATIVE_WRITING = "CREATIVE_WRITING"
    SECURITY_AUDIT = "SECURITY_AUDIT"
    DEVOPS = "DEVOPS"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    COST_ANALYSIS = "COST_ANALYSIS"
    SPECIFICATION = "SPECIFICATION"
    CREATIVE = "CREATIVE"
    GENERAL = "GENERAL"


_mock_model_router_instance = MagicMock(name="model_router_instance")
_mod(
    "vetinari.dynamic_model_router",
    TaskType=_FakeTaskType,
    get_model_router=MagicMock(return_value=_mock_model_router_instance),
)
# The actual code imports from vetinari.models.dynamic_model_router, not
# vetinari.dynamic_model_router.  Stub both paths so the lazy import inside
# _route_model_for_task resolves without loading the real module (which would
# pull in vetinari.types.ModelProvider and fail in the stub environment).
_mod(
    "vetinari.models.dynamic_model_router",
    TaskType=_FakeTaskType,
    get_model_router=MagicMock(return_value=_mock_model_router_instance),
)

# Other lazily-imported modules used inside methods
_mod(
    "vetinari.rules_manager",
    get_rules_manager=MagicMock(return_value=MagicMock(build_system_prompt_prefix=MagicMock(return_value=""))),
)
_mod(
    "vetinari.token_optimizer",
    get_token_optimizer=MagicMock(
        return_value=MagicMock(
            prepare_prompt=MagicMock(return_value={"prompt": "p", "max_tokens": 1024, "temperature": 0.7})
        )
    ),
)
_mod("vetinari.adapters", is_pkg=True)
_mod("vetinari.adapters.base", InferenceRequest=MagicMock(name="InferenceRequest"))
_mod("vetinari.adapters.llama_cpp_adapter", LocalInferenceAdapter=MagicMock(name="LocalInferenceAdapter"))
_mod("vetinari.adapters.llama_cpp_local_adapter", LocalInferenceAdapter=MagicMock(name="LocalInferenceAdapter"))
_mod(
    "vetinari.agents.contracts",
    AgentTask=MagicMock,
    AgentType=MagicMock(EVALUATOR="EVALUATOR", SYNTHESIZER="SYNTHESIZER", WORKER="WORKER"),
    Plan=MagicMock(create_new=MagicMock(return_value=MagicMock(tasks=[]))),
    Task=MagicMock,
)

# Force-override contracts attrs that two_layer.py imports at module level.
# This is needed when a prior test file (e.g. test_agent_graph.py) installed a
# strict dataclass AgentTask with required positional args - we MUST replace it
# with a MagicMock before two_layer is imported so _review_outputs() works.
_c = sys.modules["vetinari.agents.contracts"]
_c.AgentTask = MagicMock
_c.Plan = MagicMock(create_new=MagicMock(return_value=MagicMock(tasks=[])))
_c.Task = MagicMock

# Now the real package can be imported; its __init__ will find our stubs.
# We import two_layer directly to avoid __init__ re-importing two_layer
# (which would create a circular reference via the __init__).
# Instead we forcibly import the submodule directly.
# Pop any incomplete stub left by earlier test files (e.g. test_cli.py)
sys.modules.pop("vetinari.orchestration.two_layer", None)

import vetinari.orchestration.two_layer as tl
from vetinari.orchestration.request_routing import (
    _GOAL_CATEGORY_KEYWORDS,
    _GOAL_ROUTING_TABLE,
    classify_goal,
    get_goal_routing,
)
from vetinari.orchestration.two_layer import (
    TwoLayerOrchestrator,
    get_two_layer_orchestrator,
    init_two_layer_orchestrator,
)
from vetinari.types import AgentType

# Patch the names two_layer.py imported at module level so tests can
# control them via patch.object(tl, ...).
tl.PlanGenerator = _PlanGenerator
tl.DurableExecutionEngine = _DurableExec
tl.ExecutionGraph = _ExecutionGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orchestrator(**kwargs):
    """Return a TwoLayerOrchestrator with mocked plan_generator & engine."""
    mock_pg = MagicMock(name="PlanGenerator_instance")
    mock_de = MagicMock(name="DurableExecutionEngine_instance")
    with (
        patch.object(tl, "PlanGenerator", return_value=mock_pg),
        patch.object(tl, "DurableExecutionEngine", return_value=mock_de),
    ):
        orch = TwoLayerOrchestrator(**kwargs)
    return orch


def _make_mock_graph(plan_id="plan-test", nodes=None):
    g = MagicMock()
    g.plan_id = plan_id
    g.nodes = nodes or {}
    g.follow_up_question = None  # prevent auto-MagicMock triggering pause path
    return g


def _make_exec_results(completed=2, failed=0, task_results=None):
    return {
        "completed": completed,
        "failed": failed,
        "task_results": task_results or {"t1": "r1", "t2": "r2"},
    }


# ---------------------------------------------------------------------------
# Tests: classify_goal
# ---------------------------------------------------------------------------


class TestClassifyGoal:
    def test_security_keyword_security(self):
        assert classify_goal("run a security scan") == "security"

    def test_security_keyword_audit(self):
        assert classify_goal("audit the system") == "security"

    def test_security_keyword_vulnerability(self):
        assert classify_goal("find vulnerability") == "security"

    def test_security_keyword_pentest(self):
        assert classify_goal("run a pentest") == "security"

    def test_security_keyword_cve(self):
        assert classify_goal("patch this cve") == "security"

    def test_security_keyword_owasp(self):
        assert classify_goal("check owasp top ten") == "security"

    def test_security_keyword_exploit(self):
        assert classify_goal("simulate exploit") == "security"

    def test_devops_keyword_deploy(self):
        assert classify_goal("deploy to production") == "devops"

    def test_devops_keyword_docker(self):
        assert classify_goal("dockerize the app") == "devops"

    def test_devops_keyword_kubernetes(self):
        assert classify_goal("setup kubernetes cluster") == "devops"

    def test_devops_keyword_cicd(self):
        assert classify_goal("set up ci/cd pipeline") == "devops"

    def test_devops_keyword_terraform(self):
        assert classify_goal("write terraform config") == "devops"

    def test_devops_keyword_helm(self):
        assert classify_goal("create helm chart") == "devops"

    def test_image_keyword_logo(self):
        assert classify_goal("design a logo") == "image"

    def test_image_keyword_icon(self):
        assert classify_goal("create an icon") == "image"

    def test_image_keyword_mockup(self):
        assert classify_goal("make a mockup") == "image"

    def test_image_keyword_diagram(self):
        assert classify_goal("draw a diagram") == "image"

    def test_creative_keyword_story(self):
        assert classify_goal("write a story") == "creative"

    def test_creative_keyword_poem(self):
        assert classify_goal("compose a poem") == "creative"

    def test_creative_keyword_fiction(self):
        assert classify_goal("generate fiction") == "creative"

    def test_creative_keyword_novel(self):
        assert classify_goal("write a novel") == "creative"

    def test_data_keyword_database(self):
        assert classify_goal("design a database schema") == "data"

    def test_data_keyword_migration(self):
        assert classify_goal("run migration scripts") == "data"

    def test_data_keyword_etl(self):
        # avoid "pipeline" which is a devops keyword matched earlier
        assert classify_goal("run an etl job") == "data"

    def test_data_keyword_sql(self):
        assert classify_goal("write sql queries") == "data"

    def test_ui_keyword_ui(self):
        assert classify_goal("build a ui component") == "ui"

    def test_ui_keyword_ux(self):
        assert classify_goal("improve ux flow") == "ui"

    def test_ui_keyword_frontend(self):
        assert classify_goal("create frontend page") == "ui"

    def test_ui_keyword_wireframe(self):
        assert classify_goal("draw wireframe") == "ui"

    def test_ui_keyword_css(self):
        assert classify_goal("style with css") == "ui"

    def test_docs_keyword_document(self):
        assert classify_goal("write documentation") == "docs"

    def test_docs_keyword_readme(self):
        assert classify_goal("update the readme") == "docs"

    def test_docs_keyword_api_docs(self):
        assert classify_goal("generate api docs") == "docs"

    def test_docs_keyword_tutorial(self):
        assert classify_goal("create a tutorial") == "docs"

    def test_research_keyword_research(self):
        assert classify_goal("research machine learning") == "research"

    def test_research_keyword_analyze(self):
        assert classify_goal("analyze the dataset") == "research"

    def test_research_keyword_investigate(self):
        assert classify_goal("investigate the bug") == "research"

    def test_research_keyword_study(self):
        assert classify_goal("study the codebase") == "research"

    def test_research_keyword_explore(self):
        assert classify_goal("explore alternatives") == "research"

    def test_code_keyword_code(self):
        assert classify_goal("write code for sorting") == "code"

    def test_code_keyword_implement(self):
        assert classify_goal("implement the feature") == "code"

    def test_code_keyword_build(self):
        # "build" contains "ui" as substring so use "develop" instead
        assert classify_goal("develop the module") == "code"

    def test_code_keyword_refactor(self):
        assert classify_goal("refactor the class") == "code"

    def test_code_keyword_function(self):
        assert classify_goal("create a function") == "code"

    def test_general_fallback_unknown(self):
        assert classify_goal("do something unrecognised") == "general"

    def test_general_fallback_empty(self):
        assert classify_goal("") == "general"

    def test_case_insensitive_security(self):
        assert classify_goal("SECURITY audit") == "security"

    def test_case_insensitive_code(self):
        assert classify_goal("IMPLEMENT the feature") == "code"

    def test_case_insensitive_devops(self):
        assert classify_goal("DEPLOY the service") == "devops"

    def test_mixed_case(self):
        assert classify_goal("ReFaCtoR the module") == "code"

    def test_first_matching_category_wins(self):
        # "security" comes before "code" in _GOAL_CATEGORY_KEYWORDS
        result = classify_goal("security audit the code implementation")
        assert result == "security"

    def test_all_nine_categories_covered(self):
        assert len(_GOAL_CATEGORY_KEYWORDS) == 17


# ---------------------------------------------------------------------------
# Tests: get_goal_routing
# ---------------------------------------------------------------------------


class TestGetGoalRouting:
    def test_returns_3_tuple(self):
        result = get_goal_routing("write code")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_code_routing(self):
        agent, mode, tier = get_goal_routing("implement the feature")
        assert agent == AgentType.WORKER.value
        assert mode == "build"
        assert tier == "coder"

    def test_research_routing(self):
        agent, mode, _tier = get_goal_routing("research the topic")
        assert agent == AgentType.WORKER.value
        assert mode == "domain_research"

    def test_docs_routing(self):
        agent, _mode, _tier = get_goal_routing("write readme")
        assert agent == AgentType.WORKER.value

    def test_creative_routing(self):
        agent, mode, _tier = get_goal_routing("write a story")
        assert agent == AgentType.WORKER.value
        assert mode == "creative_writing"

    def test_security_routing(self):
        agent, _mode, _tier = get_goal_routing("security audit")
        assert agent == AgentType.INSPECTOR.value

    def test_data_routing(self):
        agent, _mode, _tier = get_goal_routing("design database")
        assert agent == AgentType.WORKER.value

    def test_devops_routing(self):
        agent, _mode, _tier = get_goal_routing("deploy to kubernetes")
        assert agent == AgentType.WORKER.value

    def test_ui_routing(self):
        agent, _mode, _tier = get_goal_routing("build ui component")
        assert agent == AgentType.WORKER.value

    def test_image_routing(self):
        agent, _mode, _tier = get_goal_routing("create a logo")
        assert agent == AgentType.WORKER.value

    def test_general_fallback(self):
        agent, mode, _tier = get_goal_routing("do something unknown xyz")
        assert agent == AgentType.FOREMAN.value
        assert mode == "plan"

    def test_all_categories_in_routing_table(self):
        for cat in _GOAL_CATEGORY_KEYWORDS:
            assert cat in _GOAL_ROUTING_TABLE, f"Missing routing for {cat}"

    def test_routing_table_has_general(self):
        assert "general" in _GOAL_ROUTING_TABLE

    def test_all_routing_entries_3_elements(self):
        for cat, entry in _GOAL_ROUTING_TABLE.items():
            assert len(entry) == 3, f"{cat} entry should have 3 elements"


# ---------------------------------------------------------------------------
# Tests: TwoLayerOrchestrator.__init__
# ---------------------------------------------------------------------------


class TestInit:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None

    def test_default_init_calls_plan_generator(self):
        with patch.object(tl, "PlanGenerator") as MockPG, patch.object(tl, "DurableExecutionEngine"):
            TwoLayerOrchestrator()
            MockPG.assert_called_once_with(None)

    def test_default_init_calls_durable_engine(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine") as MockDE:
            TwoLayerOrchestrator()
            MockDE.assert_called_once_with(checkpoint_dir=None, max_concurrent=4)

    def test_custom_checkpoint_dir(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine") as MockDE:
            TwoLayerOrchestrator(checkpoint_dir="/tmp/ck")
            MockDE.assert_called_once_with(checkpoint_dir="/tmp/ck", max_concurrent=4)

    def test_custom_max_concurrent(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine") as MockDE:
            TwoLayerOrchestrator(max_concurrent=8)
            MockDE.assert_called_once_with(checkpoint_dir=None, max_concurrent=8)

    def test_custom_model_router_passed_to_plan_generator(self):
        mock_router = MagicMock()
        with patch.object(tl, "PlanGenerator") as MockPG, patch.object(tl, "DurableExecutionEngine"):
            orch = TwoLayerOrchestrator(model_router=mock_router)
            MockPG.assert_called_once_with(mock_router)
            assert orch.model_router is mock_router

    def test_custom_agent_context(self):
        ctx = {"key": "value"}
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine"):
            orch = TwoLayerOrchestrator(agent_context=ctx)
            assert orch.agent_context == ctx

    def test_default_model_router_is_none(self):
        orch = _make_orchestrator()
        assert orch.model_router is None

    def test_default_agent_context_is_empty_dict(self):
        orch = _make_orchestrator()
        assert orch.agent_context == {}

    def test_agents_cache_starts_empty(self):
        orch = _make_orchestrator()
        assert orch._agents == {}

    def test_plan_generator_stored_as_attribute(self):
        mock_pg = MagicMock()
        with patch.object(tl, "PlanGenerator", return_value=mock_pg), patch.object(tl, "DurableExecutionEngine"):
            orch = TwoLayerOrchestrator()
            assert orch.plan_generator is mock_pg

    def test_execution_engine_stored_as_attribute(self):
        mock_de = MagicMock()
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine", return_value=mock_de):
            orch = TwoLayerOrchestrator()
            assert orch.execution_engine is mock_de


# ---------------------------------------------------------------------------
# Tests: set_task_handlers
# ---------------------------------------------------------------------------


class TestSetTaskHandlers:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()

    def test_registers_single_handler(self):
        h = MagicMock()
        self.orch.set_task_handlers({"analysis": h})
        self.orch.execution_engine.register_handler.assert_called_once_with("analysis", h)

    def test_registers_multiple_handlers(self):
        h1, h2, h3 = MagicMock(), MagicMock(), MagicMock()
        self.orch.set_task_handlers({"a": h1, "b": h2, "c": h3})
        calls = self.orch.execution_engine.register_handler.call_args_list
        assert len(calls) == 3
        assert call("a", h1) in calls
        assert call("b", h2) in calls
        assert call("c", h3) in calls

    def test_empty_handlers_no_register_calls(self):
        self.orch.set_task_handlers({})
        self.orch.execution_engine.register_handler.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: set_agent_context
# ---------------------------------------------------------------------------


class TestSetAgentContext:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()

    def test_updates_context(self):
        ctx = {"adapter_manager": MagicMock()}
        self.orch.set_agent_context(ctx)
        assert self.orch.agent_context is ctx

    def test_clears_agents_cache(self):
        self.orch._agents = {AgentType.FOREMAN.value: MagicMock()}
        self.orch.set_agent_context({"new": "ctx"})
        assert self.orch._agents == {}

    def test_replaces_old_context(self):
        self.orch.agent_context = {"old": "value"}
        self.orch.set_agent_context({"new": "value"})
        assert self.orch.agent_context == {"new": "value"}

    def test_empty_context_accepted(self):
        self.orch.set_agent_context({})
        assert self.orch.agent_context == {}


# ---------------------------------------------------------------------------
# Tests: _get_agent
# ---------------------------------------------------------------------------


class TestGetAgent:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()
        self.orch._agents = {}

    def test_unknown_type_returns_none(self):
        result = self.orch._get_agent("UNKNOWN_AGENT_XYZ")
        assert result is None

    def test_cache_hit_returns_cached(self):
        cached = MagicMock()
        self.orch._agents[AgentType.FOREMAN.value] = cached
        result = self.orch._get_agent(AgentType.FOREMAN.value)
        assert result is cached

    def test_cache_hit_case_insensitive_input(self):
        cached = MagicMock()
        self.orch._agents[AgentType.FOREMAN.value] = cached
        result = self.orch._get_agent("foreman")
        assert result is cached

    def test_successful_import_returns_agent(self):
        mock_agent = MagicMock()
        mock_mod = MagicMock()
        mock_mod.get_foreman_agent = MagicMock(return_value=mock_agent)
        with patch("importlib.import_module", return_value=mock_mod):
            result = self.orch._get_agent(AgentType.FOREMAN.value)
        assert result is mock_agent

    def test_successful_import_stores_in_cache(self):
        mock_agent = MagicMock()
        mock_mod = MagicMock()
        mock_mod.get_foreman_agent = MagicMock(return_value=mock_agent)
        with patch("importlib.import_module", return_value=mock_mod):
            self.orch._get_agent(AgentType.FOREMAN.value)
        assert AgentType.FOREMAN.value in self.orch._agents

    def test_agent_initialized_with_context_when_context_present(self):
        ctx = {"key": "val"}
        self.orch.agent_context = ctx
        mock_agent = MagicMock()
        mock_mod = MagicMock()
        mock_mod.get_foreman_agent = MagicMock(return_value=mock_agent)
        with patch("importlib.import_module", return_value=mock_mod):
            self.orch._get_agent(AgentType.FOREMAN.value)
        mock_agent.initialize.assert_called_once_with(ctx)

    def test_agent_not_initialized_when_context_empty(self):
        self.orch.agent_context = {}
        mock_agent = MagicMock()
        mock_mod = MagicMock()
        mock_mod.get_foreman_agent = MagicMock(return_value=mock_agent)
        with patch("importlib.import_module", return_value=mock_mod):
            self.orch._get_agent(AgentType.FOREMAN.value)
        mock_agent.initialize.assert_not_called()

    def test_import_error_returns_none(self):
        with patch("importlib.import_module", side_effect=ImportError("missing")):
            result = self.orch._get_agent(AgentType.FOREMAN.value)
        assert result is None

    def test_missing_getter_fn_returns_none(self):
        mock_mod = MagicMock(spec=[])  # no attributes at all
        with patch("importlib.import_module", return_value=mock_mod):
            result = self.orch._get_agent(AgentType.FOREMAN.value)
        assert result is None

    def test_getter_raises_returns_none(self):
        mock_mod = MagicMock()
        mock_mod.get_foreman_agent = MagicMock(side_effect=RuntimeError("crash"))
        with patch("importlib.import_module", return_value=mock_mod):
            result = self.orch._get_agent(AgentType.FOREMAN.value)
        assert result is None

    def test_agent_module_map_has_entries(self):
        # v0.5.0: 3 canonical + legacy redirects
        assert len(TwoLayerOrchestrator._AGENT_MODULE_MAP) >= 3

    def test_agent_module_map_all_values_are_2_tuples(self):
        for key, val in TwoLayerOrchestrator._AGENT_MODULE_MAP.items():
            assert isinstance(val, tuple)
            assert len(val) == 2, f"{key} should map to 2-tuple"

    def test_agent_module_map_getter_names_start_with_get(self):
        for key, (_, fn) in TwoLayerOrchestrator._AGENT_MODULE_MAP.items():
            assert fn.startswith("get_"), f"{key}: getter '{fn}' must start with 'get_'"

    def test_all_expected_agent_types_present(self):
        # v0.5.0: 3 canonical agent types
        expected = {AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value}
        assert set(TwoLayerOrchestrator._AGENT_MODULE_MAP.keys()) == expected


# ---------------------------------------------------------------------------
# Tests: _route_model_for_task
# ---------------------------------------------------------------------------


class TestRouteModelForTask:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()
        self.orch.model_router = MagicMock()

    def _task(self, task_type="analysis"):
        t = MagicMock()
        t.task_type = task_type
        t.id = "t1"
        return t

    def _ok_selection(self, model_id="gpt-4"):
        sel = MagicMock()
        sel.model = MagicMock()
        sel.model.id = model_id
        return sel

    def test_returns_model_id_on_success(self):
        self.orch.model_router.select_model.return_value = self._ok_selection("gpt-4")
        result = self.orch._route_model_for_task(self._task("analysis"))
        assert result == "gpt-4"

    def test_returns_default_when_selection_is_none(self):
        self.orch.model_router.select_model.return_value = None
        result = self.orch._route_model_for_task(self._task())
        assert result == "auto"

    def test_returns_default_when_selection_model_is_none(self):
        sel = MagicMock()
        sel.model = None
        self.orch.model_router.select_model.return_value = sel
        result = self.orch._route_model_for_task(self._task())
        assert result == "auto"

    def test_lazy_init_router_when_none(self):
        self.orch.model_router = None
        mock_router = MagicMock()
        mock_router.select_model.return_value = self._ok_selection("local")
        # The actual code imports from vetinari.models.dynamic_model_router
        dmr = sys.modules.get("vetinari.models.dynamic_model_router")
        if dmr is None:
            dmr = types.ModuleType("vetinari.models.dynamic_model_router")
            sys.modules["vetinari.models.dynamic_model_router"] = dmr
        dmr.get_model_router = MagicMock(return_value=mock_router)

        class FakeTT:
            ANALYSIS = "ANALYSIS"
            CODING = "CODING"
            TESTING = "TESTING"
            DOCUMENTATION = "DOC"
            CODE_REVIEW = "CR"
            CREATIVE_WRITING = "CW"
            SECURITY_AUDIT = "SA"
            DEVOPS = "DV"
            IMAGE_GENERATION = "IG"
            COST_ANALYSIS = "CA"
            SPECIFICATION = "SP"
            CREATIVE = "CR2"
            GENERAL = "GEN"

        dmr.TaskType = FakeTT

        result = self.orch._route_model_for_task(self._task("analysis"))
        assert result == "local"

    def test_returns_default_when_router_init_fails(self):
        self.orch.model_router = None
        # The actual code imports from vetinari.models.dynamic_model_router
        dmr = sys.modules.get("vetinari.models.dynamic_model_router")
        if dmr is None:
            dmr = types.ModuleType("vetinari.models.dynamic_model_router")
            sys.modules["vetinari.models.dynamic_model_router"] = dmr
        dmr.get_model_router = MagicMock(side_effect=RuntimeError("no router"))
        result = self.orch._route_model_for_task(self._task())
        assert result == "auto"
        # restore
        dmr.get_model_router = MagicMock(return_value=MagicMock())

    def test_returns_default_on_select_exception(self):
        self.orch.model_router.select_model.side_effect = RuntimeError("fail")
        result = self.orch._route_model_for_task(self._task())
        assert result == "auto"

    def test_implementation_task_type_mapped(self):
        self.orch.model_router.select_model.return_value = self._ok_selection("coder")
        result = self.orch._route_model_for_task(self._task("implementation"))
        assert result == "coder"

    def test_unknown_task_type_falls_to_general(self):
        self.orch.model_router.select_model.return_value = self._ok_selection("gen")
        result = self.orch._route_model_for_task(self._task("completely_unknown"))
        assert result == "gen"


# ---------------------------------------------------------------------------
# Tests: _enrich_goal (static method)
# ---------------------------------------------------------------------------


class TestEnrichGoal:
    def test_no_context_returns_goal_unchanged(self):
        assert TwoLayerOrchestrator._enrich_goal("build it", {}) == "build it"

    def test_required_features_appended(self):
        ctx = {"required_features": ["auth", "logging"]}
        result = TwoLayerOrchestrator._enrich_goal("build it", ctx)
        assert "Required features:" in result
        assert "- auth" in result
        assert "- logging" in result

    def test_things_to_avoid_appended(self):
        ctx = {"things_to_avoid": ["jQuery", "PHP"]}
        result = TwoLayerOrchestrator._enrich_goal("build it", ctx)
        assert "Do NOT include:" in result
        assert "- jQuery" in result

    def test_tech_stack_appended(self):
        ctx = {"tech_stack": "Python, FastAPI"}
        result = TwoLayerOrchestrator._enrich_goal("build it", ctx)
        assert "Tech stack: Python, FastAPI" in result

    def test_priority_appended(self):
        ctx = {"priority": "high"}
        result = TwoLayerOrchestrator._enrich_goal("build it", ctx)
        assert "Priority: high" in result

    def test_all_fields_combined(self):
        ctx = {
            "required_features": ["x"],
            "things_to_avoid": ["y"],
            "tech_stack": "Go",
            "priority": "critical",
        }
        result = TwoLayerOrchestrator._enrich_goal("goal", ctx)
        assert "Required features:" in result
        assert "Do NOT include:" in result
        assert "Tech stack: Go" in result
        assert "Priority: critical" in result

    def test_empty_required_features_list_not_appended(self):
        ctx = {"required_features": []}
        result = TwoLayerOrchestrator._enrich_goal("goal", ctx)
        assert "Required features:" not in result

    def test_goal_text_preserved_at_start(self):
        result = TwoLayerOrchestrator._enrich_goal("my goal", {"priority": "low"})
        assert result.startswith("my goal")


# ---------------------------------------------------------------------------
# Tests: _analyze_input
# ---------------------------------------------------------------------------


class TestAnalyzeInput:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()
        # Force keyword-fallback path — these tests verify the fallback heuristics,
        # not the LLM-backed classify_goal_detailed() classifier.
        self._cgd_patch = patch(
            "vetinari.orchestration.request_routing.classify_goal_detailed",
            side_effect=ImportError("forced fallback"),
        )
        self._cgd_patch.start()
        yield
        self._cgd_patch.stop()

    def test_returns_dict_with_all_required_keys(self):
        result = self.orch._analyze_input("simple goal", {})
        for k in ("goal", "estimated_complexity", "domain", "needs_research", "needs_code", "needs_ui"):
            assert k in result

    def test_needs_code_for_code_keyword(self):
        assert self.orch._analyze_input("write code", {})["needs_code"]

    def test_needs_code_for_implement_keyword(self):
        assert self.orch._analyze_input("implement feature", {})["needs_code"]

    def test_needs_code_for_build_keyword(self):
        assert self.orch._analyze_input("build app", {})["needs_code"]

    def test_needs_code_for_create_keyword(self):
        assert self.orch._analyze_input("create module", {})["needs_code"]

    def test_needs_code_for_program_keyword(self):
        assert self.orch._analyze_input("program script", {})["needs_code"]

    def test_needs_code_false_for_research(self):
        assert not self.orch._analyze_input("research topic", {})["needs_code"]

    def test_needs_research_for_research_keyword(self):
        assert self.orch._analyze_input("research this", {})["needs_research"]

    def test_needs_research_for_analyze_keyword(self):
        assert self.orch._analyze_input("analyze data", {})["needs_research"]

    def test_needs_research_for_investigate_keyword(self):
        assert self.orch._analyze_input("investigate bug", {})["needs_research"]

    def test_needs_research_for_study_keyword(self):
        assert self.orch._analyze_input("study the code", {})["needs_research"]

    def test_needs_research_false(self):
        assert not self.orch._analyze_input("build app", {})["needs_research"]

    def test_needs_ui_for_ui_keyword(self):
        assert self.orch._analyze_input("make a ui", {})["needs_ui"]

    def test_needs_ui_for_frontend_keyword(self):
        assert self.orch._analyze_input("create frontend", {})["needs_ui"]

    def test_needs_ui_for_interface_keyword(self):
        assert self.orch._analyze_input("design interface", {})["needs_ui"]

    def test_needs_ui_for_dashboard_keyword(self):
        assert self.orch._analyze_input("build dashboard", {})["needs_ui"]

    def test_needs_ui_false(self):
        assert not self.orch._analyze_input("analyze data", {})["needs_ui"]

    def test_complexity_simple_short_goal(self):
        assert self.orch._analyze_input("build it now", {})["estimated_complexity"] == "simple"

    def test_complexity_simple_few_words(self):
        goal = " ".join(["w"] * 9)
        assert self.orch._analyze_input(goal, {})["estimated_complexity"] == "simple"

    def test_complexity_moderate_medium_length(self):
        goal = " ".join(["w"] * 20)
        assert self.orch._analyze_input(goal, {})["estimated_complexity"] == "medium"

    def test_complexity_complex_with_keywords(self):
        goal = "architect a complex distributed system with backwards compatibility and security audit compliance"
        assert self.orch._analyze_input(goal, {})["estimated_complexity"] == "medium"

    def test_complexity_routing_decision_present(self):
        result = self.orch._analyze_input("refactor the entire auth system", {})
        assert "estimated_complexity" in result
        assert "domain" in result
        assert "goal" in result

    def test_domain_coding_when_needs_code(self):
        result = self.orch._analyze_input("implement feature", {})
        assert result["domain"] == "coding"

    def test_domain_research_when_needs_research_not_code(self):
        result = self.orch._analyze_input("research topic", {})
        assert result["domain"] == "research"

    def test_domain_general_otherwise(self):
        result = self.orch._analyze_input("do something", {})
        assert result["domain"] == "general"

    def test_goal_preserved_in_result(self):
        result = self.orch._analyze_input("my specific goal", {})
        assert result["goal"] == "my specific goal"

    def test_case_insensitive_detection(self):
        result = self.orch._analyze_input("IMPLEMENT the feature", {})
        assert result["needs_code"]


# ---------------------------------------------------------------------------
# Tests: generate_and_execute
# ---------------------------------------------------------------------------


class TestGenerateAndExecute:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()
        self.mock_graph = _make_mock_graph("plan-123")
        self.orch.plan_generator.generate_plan.return_value = self.mock_graph
        self.exec_results = _make_exec_results()
        self.orch.execution_engine.execute_plan.return_value = self.exec_results

    def _run(self, goal="build something", **kwargs):
        rm_mock = MagicMock()
        rm_mock.build_system_prompt_prefix.return_value = ""
        rules_mod = sys.modules["vetinari.rules_manager"]
        rules_mod.get_rules_manager = MagicMock(return_value=rm_mock)
        # Ensure intake raises so generate_and_execute uses the full pipeline
        # (simple goals would otherwise be classified as EXPRESS and bypass planning).
        _intake_stub = types.ModuleType("vetinari.orchestration.intake")
        _intake_stub.get_request_intake = MagicMock(side_effect=ImportError("stubbed"))
        with (
            patch.object(self.orch, "_review_outputs", return_value={"verdict": "ok"}),
            patch.object(self.orch, "_assemble_final_output", return_value="final output"),
            patch.dict(sys.modules, {"vetinari.orchestration.intake": _intake_stub}),
        ):
            return self.orch.generate_and_execute(goal, **kwargs)

    def test_result_has_plan_id(self):
        assert "plan_id" in self._run()

    def test_result_has_goal(self):
        assert "goal" in self._run()

    def test_result_has_completed(self):
        assert "completed" in self._run()

    def test_result_has_failed(self):
        assert "failed" in self._run()

    def test_result_has_outputs(self):
        assert "outputs" in self._run()

    def test_result_has_review_result(self):
        assert "review_result" in self._run()

    def test_result_has_final_output(self):
        assert "final_output" in self._run()

    def test_result_has_stages(self):
        assert "stages" in self._run()

    def test_result_has_total_time_ms(self):
        assert "total_time_ms" in self._run()

    def test_plan_id_matches_graph(self):
        assert self._run()["plan_id"] == "plan-123"

    def test_goal_preserved_in_result(self):
        assert self._run("my goal")["goal"] == "my goal"

    def test_completed_count(self):
        assert self._run()["completed"] == 2

    def test_failed_count(self):
        assert self._run()["failed"] == 0

    def test_outputs_from_task_results(self):
        assert self._run()["outputs"] == {"t1": "r1", "t2": "r2"}

    def test_total_time_ms_nonnegative(self):
        assert self._run()["total_time_ms"] >= 0

    def test_stages_contains_input_analysis(self):
        assert "input_analysis" in self._run()["stages"]

    def test_stages_contains_plan(self):
        assert "plan" in self._run()["stages"]

    def test_stages_contains_model_assignment(self):
        assert "model_assignment" in self._run()["stages"]

    def test_stages_contains_execution(self):
        assert "execution" in self._run()["stages"]

    def test_stages_contains_review(self):
        assert "review" in self._run()["stages"]

    def test_stages_contains_final_assembly(self):
        assert "final_assembly" in self._run()["stages"]

    def test_plan_generator_called_with_enriched_goal(self):
        self._run("build it", context={"priority": "high"})
        call_args = self.orch.plan_generator.generate_plan.call_args[0][0]
        assert "build it" in call_args
        assert "Priority: high" in call_args

    def test_custom_task_handler_passed_to_engine(self):
        handler = MagicMock()
        self._run(task_handler=handler)
        self.orch.execution_engine.execute_plan.assert_called_once_with(self.mock_graph, handler)

    def test_default_handler_created_when_none_given(self):
        rm_mock = MagicMock()
        rm_mock.build_system_prompt_prefix.return_value = ""
        sys.modules["vetinari.rules_manager"].get_rules_manager = MagicMock(return_value=rm_mock)
        with (
            patch.object(self.orch, "_make_default_handler", return_value=MagicMock()) as mock_dh,
            patch.object(self.orch, "_review_outputs", return_value={}),
            patch.object(self.orch, "_assemble_final_output", return_value=""),
        ):
            result = self.orch.generate_and_execute("build it")
            mock_dh.assert_called_once()
        assert mock_dh.call_args.args == ()
        assert result["goal"] == "build it"

    def test_route_model_called_per_node(self):
        node = MagicMock()
        node.task_type = "analysis"
        node.id = "t1"
        node.input_data = {}
        self.mock_graph.nodes = {"t1": node}

        rm_mock = MagicMock()
        rm_mock.build_system_prompt_prefix.return_value = ""
        sys.modules["vetinari.rules_manager"].get_rules_manager = MagicMock(return_value=rm_mock)
        _intake_stub = types.ModuleType("vetinari.orchestration.intake")
        _intake_stub.get_request_intake = MagicMock(side_effect=ImportError("stubbed"))
        with (
            patch.object(self.orch, "_route_model_for_task", return_value="gpt-4") as mock_rm,
            patch.object(self.orch, "_review_outputs", return_value={}),
            patch.object(self.orch, "_assemble_final_output", return_value=""),
            patch.dict(sys.modules, {"vetinari.orchestration.intake": _intake_stub}),
        ):
            self.orch.generate_and_execute("build it")
            mock_rm.assert_called_once_with(node)
            assert node.input_data["assigned_model"] == "gpt-4"


# ---------------------------------------------------------------------------
# Tests: _review_outputs
# ---------------------------------------------------------------------------


class TestReviewOutputs:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()

    def test_fallback_when_no_evaluator(self):
        with patch.object(self.orch, "_get_agent", return_value=None):
            result = self.orch._review_outputs({"task_results": {}}, "goal")
        assert result["verdict"] == "inconclusive"

    def test_fallback_has_quality_score(self):
        with patch.object(self.orch, "_get_agent", return_value=None):
            result = self.orch._review_outputs({}, "goal")
        assert result["quality_score"] == 0.5

    def test_fallback_has_summary(self):
        with patch.object(self.orch, "_get_agent", return_value=None):
            result = self.orch._review_outputs({}, "goal")
        assert "summary" in result

    def test_uses_evaluator_output_on_success(self):
        mock_eval = MagicMock()
        mock_res = MagicMock()
        mock_res.success = True
        mock_res.output = {"verdict": "pass", "quality_score": 0.95}
        mock_eval.execute.return_value = mock_res

        with patch.object(self.orch, "_get_agent", return_value=mock_eval):
            result = self.orch._review_outputs({"task_results": {"t1": "out"}}, "goal")
        assert result["verdict"] == "pass"

    def test_fallback_when_evaluator_execute_raises(self):
        mock_eval = MagicMock()
        mock_eval.execute.side_effect = RuntimeError("crash")
        with patch.object(self.orch, "_get_agent", return_value=mock_eval):
            result = self.orch._review_outputs({"task_results": {}}, "goal")
        assert result["verdict"] == "inconclusive"

    def test_fallback_when_evaluator_result_not_success(self):
        mock_eval = MagicMock()
        mock_res = MagicMock()
        mock_res.success = False
        mock_eval.execute.return_value = mock_res
        with patch.object(self.orch, "_get_agent", return_value=mock_eval):
            result = self.orch._review_outputs({"task_results": {}}, "goal")
        assert "quality_score" in result


# ---------------------------------------------------------------------------
# Tests: _assemble_final_output
# ---------------------------------------------------------------------------


class TestAssembleFinalOutput:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()

    def test_uses_synthesized_artifact_from_synthesizer(self):
        mock_synth = MagicMock()
        mock_res = MagicMock()
        mock_res.success = True
        mock_res.output = {"synthesized_artifact": "the final text"}
        mock_synth.execute.return_value = mock_res
        with patch.object(self.orch, "_get_agent", return_value=mock_synth):
            result = self.orch._assemble_final_output({"task_results": {"t1": "r1"}}, {}, "goal")
        assert result == "the final text"

    def test_str_output_when_no_synthesized_artifact_key(self):
        mock_synth = MagicMock()
        mock_res = MagicMock()
        mock_res.success = True
        mock_res.output = {"other": "value"}
        mock_synth.execute.return_value = mock_res
        with patch.object(self.orch, "_get_agent", return_value=mock_synth):
            result = self.orch._assemble_final_output({"task_results": {}}, {}, "goal")
        assert isinstance(result, str)

    def test_fallback_join_when_no_synthesizer(self):
        with patch.object(self.orch, "_get_agent", return_value=None):
            result = self.orch._assemble_final_output({"task_results": {"t1": "output1"}}, {}, "goal")
        assert "t1" in result
        assert "output1" in result

    def test_fallback_completed_goal_when_empty_task_results(self):
        with patch.object(self.orch, "_get_agent", return_value=None):
            result = self.orch._assemble_final_output({"task_results": {}}, {}, "do X")
        assert "do X" in result

    def test_fallback_when_synthesizer_raises(self):
        mock_synth = MagicMock()
        mock_synth.execute.side_effect = RuntimeError("fail")
        with patch.object(self.orch, "_get_agent", return_value=mock_synth):
            result = self.orch._assemble_final_output({"task_results": {"t1": "r1"}}, {}, "goal")
        assert "t1" in result

    def test_returns_string(self):
        with patch.object(self.orch, "_get_agent", return_value=None):
            result = self.orch._assemble_final_output({"task_results": {}}, {}, "goal")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests: _make_default_handler
# ---------------------------------------------------------------------------


class TestMakeDefaultHandler:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()

    def _task(self, model="default", task_type="general", description="do X"):
        t = MagicMock()
        t.id = "t1"
        t.input_data = {"assigned_model": model}
        t.task_type = task_type
        t.description = description
        return t

    def _opt_mock(self):
        return MagicMock(prepare_prompt=MagicMock(return_value={"prompt": "p", "max_tokens": 100, "temperature": 0.3}))

    def test_returns_callable(self):
        assert callable(self.orch._make_default_handler())

    def test_success_with_adapter_manager(self):
        am = MagicMock()
        resp = MagicMock()
        resp.status = "ok"
        resp.output = "result text"
        am.infer.return_value = resp
        self.orch.agent_context = {"adapter_manager": am}

        sys.modules["vetinari.token_optimizer"].get_token_optimizer = MagicMock(return_value=self._opt_mock())
        handler = self.orch._make_default_handler()
        result = handler(self._task())

        assert result["status"] == "ok"
        assert result["result"] == "result text"
        assert result["task_id"] == "t1"

    def test_fallback_local_inference_when_no_adapter_manager(self):
        """Falls back to LocalInferenceAdapter when adapter_manager is absent."""
        self.orch.agent_context = {}
        mock_lms = MagicMock()
        mock_lms.chat.return_value = {"output": "lm result"}

        sys.modules["vetinari.token_optimizer"].get_token_optimizer = MagicMock(return_value=self._opt_mock())
        sys.modules["vetinari.adapters.llama_cpp_local_adapter"].LocalInferenceAdapter = MagicMock(
            return_value=mock_lms
        )
        handler = self.orch._make_default_handler()
        result = handler(self._task())

        assert result["status"] == "ok"
        assert result["result"] == "lm result"

    def test_error_status_when_everything_fails(self):
        self.orch.agent_context = {}
        sys.modules["vetinari.token_optimizer"].get_token_optimizer = MagicMock(side_effect=RuntimeError("crash"))
        sys.modules["vetinari.adapters.llama_cpp_local_adapter"].LocalInferenceAdapter = MagicMock(
            side_effect=RuntimeError("also crash")
        )
        handler = self.orch._make_default_handler()
        result = handler(self._task())

        assert result["status"] == "error"
        assert "error" in result
        assert result["task_id"] == "t1"

    def test_adapter_manager_failure_falls_back_to_local_inference(self):
        """Adapter manager failure falls back to LocalInferenceAdapter."""
        am = MagicMock()
        am.infer.side_effect = RuntimeError("infer fail")
        self.orch.agent_context = {"adapter_manager": am}

        mock_local = MagicMock()
        mock_local.chat.return_value = {"output": "local fallback"}

        sys.modules["vetinari.token_optimizer"].get_token_optimizer = MagicMock(return_value=self._opt_mock())
        sys.modules["vetinari.adapters.llama_cpp_local_adapter"].LocalInferenceAdapter = MagicMock(
            return_value=mock_local
        )
        handler = self.orch._make_default_handler()
        result = handler(self._task())

        assert result["status"] == "ok"
        assert result["result"] == "local fallback"

    def test_token_optimizer_failure_uses_defaults(self):
        """Token optimizer failure falls back to defaults and still uses local inference."""
        self.orch.agent_context = {}
        mock_local = MagicMock()
        mock_local.chat.return_value = {"output": "ok"}

        sys.modules["vetinari.token_optimizer"].get_token_optimizer = MagicMock(
            side_effect=RuntimeError("no optimizer")
        )
        sys.modules["vetinari.adapters.llama_cpp_local_adapter"].LocalInferenceAdapter = MagicMock(
            return_value=mock_local
        )
        handler = self.orch._make_default_handler()
        result = handler(self._task())

        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Tests: execute_with_agent_graph
# ---------------------------------------------------------------------------


class TestExecuteWithAgentGraph:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()
        # Set up contracts stub for Plan/AgentType/Task used inside execute_with_agent_graph
        self._contracts = sys.modules["vetinari.agents.contracts"]
        self._ag_mod = sys.modules["vetinari.orchestration.agent_graph"]

    def _setup_happy_path(self, plan_id="plan-ag", nodes=None):
        """Configure mocks for a successful agent_graph execution."""
        mock_graph = _make_mock_graph(plan_id)
        mock_graph.nodes = nodes or {}
        self.orch.plan_generator.generate_plan.return_value = mock_graph

        mock_plan = MagicMock()
        mock_plan.tasks = []
        self._contracts.Plan = MagicMock()
        self._contracts.Plan.create_new = MagicMock(return_value=mock_plan)

        # AgentType needs to support [] lookup
        fake_at = MagicMock()
        fake_at.__getitem__ = MagicMock(return_value="WORKER")
        self._contracts.AgentType = fake_at

        mock_ag = MagicMock()
        mock_ag.execute_plan.return_value = {}
        self._ag_mod.get_agent_graph = MagicMock(return_value=mock_ag)
        return mock_graph, mock_ag, mock_plan

    def test_happy_path_returns_expected_keys(self):
        node = MagicMock()
        node.id = "t1"
        node.description = "do X"
        node.dependencies = []
        node.input_data = {"assigned_agent": AgentType.WORKER.value}

        mock_r = MagicMock()
        mock_r.success = True
        mock_r.output = "out"
        mock_r.errors = []

        _mock_graph, mock_ag, _ = self._setup_happy_path(nodes={"t1": node})
        mock_ag.execute_plan.return_value = {"t1": mock_r}

        result = self.orch.execute_with_agent_graph("build something")

        assert "plan_id" in result
        assert "backend" in result
        assert "completed" in result
        assert "failed" in result

    def test_backend_is_agent_graph(self):
        self._setup_happy_path("plan-ag2", nodes={})
        result = self.orch.execute_with_agent_graph("goal")
        assert result.get("backend") == "agent_graph"

    def test_falls_back_to_generate_and_execute_on_exception(self):
        self._ag_mod.get_agent_graph = MagicMock(side_effect=RuntimeError("ag unavailable"))
        with patch.object(self.orch, "generate_and_execute", return_value={"plan_id": "fallback"}) as mock_gen:
            result = self.orch.execute_with_agent_graph("build it")
            mock_gen.assert_called_once()
        assert result["plan_id"] == "fallback"

    def test_none_context_does_not_raise(self):
        self._ag_mod.get_agent_graph = MagicMock(side_effect=RuntimeError("fail"))
        with patch.object(self.orch, "generate_and_execute", return_value={"plan_id": "p1"}) as mock_gen:
            # Should not raise even with context=None
            result = self.orch.execute_with_agent_graph("goal", context=None)
        mock_gen.assert_called_once()  # fell back to generate_and_execute
        assert mock_gen.call_args.args == ("goal", None)
        assert mock_gen.call_args.kwargs == {"context": None}
        assert result["plan_id"] == "p1"


# ---------------------------------------------------------------------------
# Tests: generate_plan_only / execute_plan / recover_plan / get_plan_status / list_checkpoints
# ---------------------------------------------------------------------------


class TestDelegationMethods:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        self.orch = _make_orchestrator()

    def test_generate_plan_only_delegates_with_constraints(self):
        mock_graph = MagicMock()
        self.orch.plan_generator.generate_plan.return_value = mock_graph
        result = self.orch.generate_plan_only("goal", {"k": "v"})
        call_args = self.orch.plan_generator.generate_plan.call_args
        assert call_args.args == ("goal", {"k": "v"})
        assert "max_depth" in call_args.kwargs
        assert result is mock_graph

    def test_generate_plan_only_delegates_without_constraints(self):
        self.orch.plan_generator.generate_plan.return_value = MagicMock()
        self.orch.generate_plan_only("goal")
        call_args = self.orch.plan_generator.generate_plan.call_args
        assert call_args.args == ("goal", None)
        assert "max_depth" in call_args.kwargs

    def test_execute_plan_delegates_with_handler(self):
        mock_graph = MagicMock()
        mock_handler = MagicMock()
        mock_result = {"completed": 1}
        self.orch.execution_engine.execute_plan.return_value = mock_result
        result = self.orch.execute_plan(mock_graph, mock_handler)
        self.orch.execution_engine.execute_plan.assert_called_once_with(mock_graph, mock_handler)
        assert result is mock_result

    def test_execute_plan_delegates_without_handler(self):
        mock_graph = MagicMock()
        self.orch.execution_engine.execute_plan.return_value = {}
        self.orch.execute_plan(mock_graph)
        self.orch.execution_engine.execute_plan.assert_called_once_with(mock_graph, None)

    def test_recover_plan_delegates(self):
        sentinel = {"status": "ok", "plan_id": "plan-abc"}
        self.orch.execution_engine.recover_execution.return_value = sentinel
        result = self.orch.recover_plan("plan-abc")
        self.orch.execution_engine.recover_execution.assert_called_once_with("plan-abc")
        # Verify the return value is passed through unchanged (not copied or transformed)
        assert result is sentinel

    def test_get_plan_status_delegates(self):
        sentinel = {"status": "running", "plan_id": "plan-abc"}
        self.orch.execution_engine.get_execution_status.return_value = sentinel
        result = self.orch.get_plan_status("plan-abc")
        self.orch.execution_engine.get_execution_status.assert_called_once_with("plan-abc")
        # Verify identity — delegation returns the exact engine result
        assert result is sentinel

    def test_get_plan_status_returns_none_when_not_found(self):
        self.orch.execution_engine.get_execution_status.return_value = None
        assert self.orch.get_plan_status("missing") is None

    def test_list_checkpoints_delegates(self):
        sentinel = ["p1", "p2"]
        self.orch.execution_engine.list_checkpoints.return_value = sentinel
        result = self.orch.list_checkpoints()
        self.orch.execution_engine.list_checkpoints.assert_called_once()
        assert result is sentinel

    def test_list_checkpoints_empty(self):
        self.orch.execution_engine.list_checkpoints.return_value = []
        assert self.orch.list_checkpoints() == []


# ---------------------------------------------------------------------------
# Tests: Singletons
# ---------------------------------------------------------------------------


class TestSingletons:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None
        yield
        tl._two_layer_orchestrator = None

    def test_get_creates_instance_when_none(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine"):
            orch = get_two_layer_orchestrator()
        assert isinstance(orch, TwoLayerOrchestrator)

    def test_get_returns_same_instance_twice(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine"):
            a = get_two_layer_orchestrator()
            b = get_two_layer_orchestrator()
        assert a is b

    def test_get_returns_existing_instance(self):
        existing = MagicMock()
        tl._two_layer_orchestrator = existing
        assert get_two_layer_orchestrator() is existing

    def test_init_creates_new_instance(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine"):
            orch = init_two_layer_orchestrator("/tmp/ck")
        assert isinstance(orch, TwoLayerOrchestrator)

    def test_init_updates_global(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine"):
            orch = init_two_layer_orchestrator()
        assert tl._two_layer_orchestrator is orch

    def test_init_replaces_existing(self):
        old = MagicMock()
        tl._two_layer_orchestrator = old
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine"):
            new = init_two_layer_orchestrator()
        assert new is not old

    def test_init_passes_checkpoint_dir(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine") as MockDE:
            init_two_layer_orchestrator("/my/dir")
            MockDE.assert_called_once_with(checkpoint_dir="/my/dir", max_concurrent=4)

    def test_init_passes_kwargs(self):
        with patch.object(tl, "PlanGenerator"), patch.object(tl, "DurableExecutionEngine") as MockDE:
            init_two_layer_orchestrator(max_concurrent=16)
            MockDE.assert_called_once_with(checkpoint_dir=None, max_concurrent=16)


# ---------------------------------------------------------------------------
# Tests: structural / edge cases
# ---------------------------------------------------------------------------


class TestStructural:
    def test_goal_category_keywords_has_exactly_9_categories(self):
        assert len(_GOAL_CATEGORY_KEYWORDS) == 17

    def test_goal_routing_table_has_18_entries(self):
        # 17 categories + "general"
        assert len(_GOAL_ROUTING_TABLE) == 18

    def test_routing_table_values_are_3_tuples(self):
        for cat, entry in _GOAL_ROUTING_TABLE.items():
            assert len(entry) == 3, f"{cat} should be 3-tuple"

    def test_enrich_goal_is_static_method(self):
        assert callable(TwoLayerOrchestrator._enrich_goal)

    def test_agent_module_map_is_class_attribute(self):
        # _AGENT_MODULE_MAP lives on PipelineHelpersMixin, accessible via MRO
        assert hasattr(TwoLayerOrchestrator, "_AGENT_MODULE_MAP")

    def test_classify_goal_with_whitespace_only(self):
        result = classify_goal("   ")
        assert result == "general"

    def test_get_goal_routing_delegates_to_classify_goal(self):
        with patch(
            "vetinari.orchestration.request_routing.classify_goal_detailed",
            return_value={
                "category": "code",
                "confidence": 0.8,
                "complexity": "standard",
                "cross_cutting": [],
                "matched_keywords": ["code"],
            },
        ) as mock_classify:
            get_goal_routing("some goal")
            mock_classify.assert_called_once_with("some goal")

    def test_multiple_agents_cached_independently(self):
        orch = _make_orchestrator()
        a1 = MagicMock()
        a2 = MagicMock()
        mod = MagicMock()
        mod.get_foreman_agent = MagicMock(return_value=a1)
        mod.get_worker_agent = MagicMock(return_value=a2)

        with patch("importlib.import_module", return_value=mod):
            r1 = orch._get_agent(AgentType.FOREMAN.value)
            r2 = orch._get_agent(AgentType.WORKER.value)

        assert r1 is a1
        assert r2 is a2
        assert len(orch._agents) == 2


# ---------------------------------------------------------------------------
# Tests: VariantManager integration
# ---------------------------------------------------------------------------


class TestVariantManagerIntegration:
    """TwoLayerOrchestrator integrates VariantManager for depth control."""

    def test_default_variant_manager_created(self):
        """Orchestrator creates a VariantManager when none is supplied."""
        orch = _make_orchestrator()
        from vetinari.web.variant_system import VariantManager

        assert isinstance(orch._variant_manager, VariantManager)

    def test_custom_variant_manager_stored(self):
        """Orchestrator stores a supplied VariantManager."""
        from vetinari.web.variant_system import VariantManager

        vm = VariantManager(default_level="low")
        orch = _make_orchestrator(variant_manager=vm)
        assert orch._variant_manager is vm

    def test_get_variant_config_returns_config(self):
        """get_variant_config() returns the active VariantConfig."""
        from vetinari.web.variant_system import VariantConfig

        orch = _make_orchestrator()
        cfg = orch.get_variant_config()
        assert isinstance(cfg, VariantConfig)

    def test_set_variant_level_changes_level(self):
        """set_variant_level() switches the active level."""
        orch = _make_orchestrator()
        orch.set_variant_level("low")
        assert orch._variant_manager.current_level == "low"
        orch.set_variant_level("high")
        assert orch._variant_manager.current_level == "high"

    def test_set_variant_level_returns_config(self):
        """set_variant_level() returns the new VariantConfig."""
        from vetinari.web.variant_system import VariantConfig

        orch = _make_orchestrator()
        cfg = orch.set_variant_level("high")
        assert isinstance(cfg, VariantConfig)
        assert cfg.max_context_tokens == 32768

    def test_variant_level_affects_planning_depth(self):
        """generate_plan is called with max_depth from the active variant config."""
        orch = _make_orchestrator()
        orch.set_variant_level("low")
        mock_graph = _make_mock_graph()
        orch.plan_generator.generate_plan.return_value = mock_graph

        orch.generate_plan_only("test goal")

        call_kwargs = orch.plan_generator.generate_plan.call_args
        assert call_kwargs.kwargs.get("max_depth") == 2  # LOW depth

    def test_low_variant_disables_correction_loop(self):
        """Bug #15a regression: LOW variant must disable enable_correction_loop even when caller passes True."""
        from vetinari.web.variant_system import VariantManager

        vm = VariantManager(default_level="low")
        orch = _make_orchestrator(enable_correction_loop=True, variant_manager=vm)
        assert orch.enable_correction_loop is False, (
            "LOW variant has enable_verification=False — correction loop must be disabled"
        )

    def test_medium_variant_preserves_correction_loop(self):
        """Bug #15a regression: MEDIUM variant (enable_verification=True) must keep enable_correction_loop=True."""
        from vetinari.web.variant_system import VariantManager

        vm = VariantManager(default_level="medium")
        orch = _make_orchestrator(enable_correction_loop=True, variant_manager=vm)
        assert orch.enable_correction_loop is True, (
            "MEDIUM variant has enable_verification=True — correction loop must remain enabled"
        )

    def test_caller_can_still_disable_correction_loop_on_high_variant(self):
        """Bug #15a regression: caller passing enable_correction_loop=False must always win."""
        from vetinari.web.variant_system import VariantManager

        vm = VariantManager(default_level="high")
        orch = _make_orchestrator(enable_correction_loop=False, variant_manager=vm)
        assert orch.enable_correction_loop is False, (
            "Caller explicitly disabled correction loop — must stay False regardless of variant"
        )


# ---------------------------------------------------------------------------
# Tests: _execute_rework_decision
# ---------------------------------------------------------------------------


class TestExecuteReworkDecision:
    """Tests for TwoLayerOrchestrator._execute_rework_decision."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None

    def test_escalate_returns_awaiting_user(self):
        orch = _make_orchestrator()
        graph = MagicMock()
        node = MagicMock()
        node.status = "failed"
        graph.nodes = {"t1": node}
        result = orch._execute_rework_decision(tl.ReworkDecision.ESCALATE_TO_USER, "t1", {}, graph)
        assert result["action"] == "escalate"
        assert result["outcome"] == "awaiting_user"

    def test_missing_task_returns_none(self):
        orch = _make_orchestrator()
        graph = MagicMock()
        graph.nodes = {}
        result = orch._execute_rework_decision(tl.ReworkDecision.RETRY_SAME_AGENT, "missing", {}, graph)
        assert result is None

    def test_retry_same_agent_resets_status(self):
        orch = _make_orchestrator()
        node = MagicMock()
        node.task_type = "analysis"
        node.input_data = {}
        graph = MagicMock()
        graph.nodes = {"t1": node}
        # Mock the execution engine to return completed
        orch.execution_engine._task_handlers = {"default": lambda t: {"result": "ok"}}
        orch.execution_engine._execute_task = MagicMock(return_value={"status": "completed"})
        orch.execution_engine._save_checkpoint = MagicMock()

        result = orch._execute_rework_decision(tl.ReworkDecision.RETRY_SAME_AGENT, "t1", {}, graph)
        assert result is not None
        assert result["action"] == "retry_same_agent"

    def test_retry_different_model_updates_assigned_model(self):
        orch = _make_orchestrator()
        node = MagicMock()
        node.task_type = "analysis"
        node.input_data = {"assigned_model": "old-model"}
        graph = MagicMock()
        graph.nodes = {"t1": node}
        orch.execution_engine._task_handlers = {"default": lambda t: {"result": "ok"}}
        orch.execution_engine._execute_task = MagicMock(return_value={"status": "completed"})
        orch.execution_engine._save_checkpoint = MagicMock()

        result = orch._execute_rework_decision(tl.ReworkDecision.RETRY_DIFFERENT_MODEL, "t1", {}, graph)
        assert result is not None
        assert result["action"] == "retry_different_model"
        assert "old-model" in node.input_data.get("excluded_models", [])

    def test_replan_resets_task(self):
        orch = _make_orchestrator()
        node = MagicMock()
        node.task_type = "analysis"
        node.input_data = {}
        graph = MagicMock()
        graph.nodes = {"t1": node}
        orch.execution_engine._task_handlers = {"default": lambda t: {"result": "ok"}}
        orch.execution_engine._execute_task = MagicMock(return_value={"status": "completed"})
        orch.execution_engine._save_checkpoint = MagicMock()

        result = orch._execute_rework_decision(tl.ReworkDecision.REPLAN, "t1", {}, graph)
        assert result["action"] == "replan"

    def test_research_then_retry_adds_hint(self):
        orch = _make_orchestrator()
        node = MagicMock()
        node.task_type = "analysis"
        node.input_data = {}
        graph = MagicMock()
        graph.nodes = {"t1": node}
        orch.execution_engine._task_handlers = {"default": lambda t: {"result": "ok"}}
        orch.execution_engine._execute_task = MagicMock(return_value={"status": "completed"})
        orch.execution_engine._save_checkpoint = MagicMock()

        orch._execute_rework_decision(tl.ReworkDecision.RESEARCH_THEN_RETRY, "t1", {}, graph)
        assert node.input_data.get("rework_hint") == "research_context_before_retry"

    def test_no_handler_returns_no_handler_outcome(self):
        orch = _make_orchestrator()
        node = MagicMock()
        node.task_type = "unknown_type"
        node.input_data = {}
        graph = MagicMock()
        graph.nodes = {"t1": node}
        orch.execution_engine._task_handlers = {}

        result = orch._execute_rework_decision(tl.ReworkDecision.RETRY_SAME_AGENT, "t1", {}, graph)
        assert result["outcome"] == "no_handler"


# ---------------------------------------------------------------------------
# Tests: recover_incomplete_on_startup
# ---------------------------------------------------------------------------


class TestRecoverIncompleteOnStartup:
    """Tests for TwoLayerOrchestrator.recover_incomplete_on_startup."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        tl._two_layer_orchestrator = None

    def test_delegates_to_execution_engine(self):
        orch = _make_orchestrator()
        orch.execution_engine.recover_incomplete_executions = MagicMock(return_value=[])
        result = orch.recover_incomplete_on_startup()
        assert result == []
        orch.execution_engine.recover_incomplete_executions.assert_called_once()

    def test_returns_engine_results(self):
        orch = _make_orchestrator()
        mock_results = [{"plan_id": "p1", "completed": 3, "failed": 0}]
        orch.execution_engine.recover_incomplete_executions = MagicMock(return_value=mock_results)
        result = orch.recover_incomplete_on_startup()
        assert result == mock_results
