"""Helper script: generates the actual test file."""
import os

TARGET = os.path.join(os.path.dirname(__file__),
                      "test_multi_mode_planner_test_automation_agents.py")

CONTENT = """\
\"\"\"
Comprehensive tests for:
  - vetinari/agents/multi_mode_agent.py  (MultiModeAgent)
  - vetinari/agents/planner_agent.py     (PlannerAgent)
  - vetinari/agents/test_automation_agent.py (TestAutomationAgent)
\"\"\"
from __future__ import annotations
import sys, types, unittest
from unittest.mock import MagicMock


def _install_stubs():
    def _pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__package__ = name
        sys.modules[name] = m
        return m

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    from enum import Enum

    class AgentType(str, Enum):
        PLANNER = "PLANNER"
        EXPLORER = "EXPLORER"
        ORACLE = "ORACLE"
        LIBRARIAN = "LIBRARIAN"
        RESEARCHER = "RESEARCHER"
        EVALUATOR = "EVALUATOR"
        SYNTHESIZER = "SYNTHESIZER"
        BUILDER = "BUILDER"
        UI_PLANNER = "UI_PLANNER"
        SECURITY_AUDITOR = "SECURITY_AUDITOR"
        DATA_ENGINEER = "DATA_ENGINEER"
        DOCUMENTATION_AGENT = "DOCUMENTATION_AGENT"
        COST_PLANNER = "COST_PLANNER"
        TEST_AUTOMATION = "TEST_AUTOMATION"
        EXPERIMENTATION_MANAGER = "EXPERIMENTATION_MANAGER"
        IMPROVEMENT = "IMPROVEMENT"
        USER_INTERACTION = "USER_INTERACTION"
        DEVOPS = "DEVOPS"
        VERSION_CONTROL = "VERSION_CONTROL"
        ERROR_RECOVERY = "ERROR_RECOVERY"
        CONTEXT_MANAGER = "CONTEXT_MANAGER"
        IMAGE_GENERATOR = "IMAGE_GENERATOR"
        PONDER = "PONDER"
        ORCHESTRATOR = "ORCHESTRATOR"
        CONSOLIDATED_RESEARCHER = "CONSOLIDATED_RESEARCHER"
        CONSOLIDATED_ORACLE = "CONSOLIDATED_ORACLE"
        ARCHITECT = "ARCHITECT"
        QUALITY = "QUALITY"
        OPERATIONS = "OPERATIONS"

    class TaskStatus(str, Enum):
        PENDING = "pending"
        BLOCKED = "blocked"
        READY = "ready"
        ASSIGNED = "assigned"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        WAITING = "waiting"

    class ExecutionMode(str, Enum):
        PLANNING = "planning"
        EXECUTION = "execution"
        SANDBOX = "sandbox"

    _mod("vetinari.types", AgentType=AgentType,
         TaskStatus=TaskStatus, ExecutionMode=ExecutionMode)

    from dataclasses import dataclass, field
    from datetime import datetime
    from typing import Any, Dict, List
    import uuid

    @dataclass
    class AgentSpec:
        agent_type: Any
        name: str
        description: str = ""
        default_model: str = "test-model"
        thinking_variant: str = "medium"
        enabled: bool = True
        system_prompt: str = ""
        version: str = "1.0.0"
        def to_dict(self):
            return {"agent_type": self.agent_type.value, "name": self.name}

    @dataclass
    class AgentTask:
        task_id: str
        agent_type: Any
        description: str
        prompt: str
        status: Any = None
        result: Any = None
        error: str = ""
        started_at: str = ""
        completed_at: str = ""
        dependencies: List[str] = field(default_factory=list)
        context: Dict[str, Any] = field(default_factory=dict)
        def __post_init__(self):
            if self.status is None:
                self.status = TaskStatus.PENDING

    @dataclass
    class AgentResult:
        success: bool
        output: Any
        metadata: Dict[str, Any] = field(default_factory=dict)
        errors: List[str] = field(default_factory=list)
        provenance: List[Dict] = field(default_factory=list)

    @dataclass
    class VerificationResult:
        passed: bool
        issues: List[Any] = field(default_factory=list)
        suggestions: List[str] = field(default_factory=list)
        score: float = 0.0

    @dataclass
    class Task:
        id: str
        description: str
        inputs: List[str] = field(default_factory=list)
        outputs: List[str] = field(default_factory=list)
        dependencies: List[str] = field(default_factory=list)
        assigned_agent: Any = None
        model_override: str = ""
        depth: int = 0
        parent_id: str = ""
        status: Any = None
        def __post_init__(self):
            if self.status is None:
                self.status = TaskStatus.PENDING
            if self.assigned_agent is None:
                self.assigned_agent = AgentType.PLANNER
        def to_dict(self):
            return {
                "id": self.id, "description": self.description,
                "inputs": self.inputs, "outputs": self.outputs,
                "dependencies": self.dependencies,
                "assigned_agent": (self.assigned_agent.value
                                   if hasattr(self.assigned_agent, "value")
                                   else str(self.assigned_agent)),
                "depth": self.depth,
                "status": (self.status.value
                           if hasattr(self.status, "value") else self.status),
            }
        @classmethod
        def from_dict(cls, data):
            return cls(
                id=data["id"], description=data["description"],
                inputs=data.get("inputs", []), outputs=data.get("outputs", []),
                dependencies=data.get("dependencies", []),
                assigned_agent=AgentType(data.get("assigned_agent", "PLANNER")),
                depth=data.get("depth", 0),
            )

    @dataclass
    class Plan:
        plan_id: str
        version: str = "v0.1.0"
        goal: str = ""
        phase: int = 0
        tasks: List[Task] = field(default_factory=list)
        model_scores: List[Dict] = field(default_factory=list)
        notes: str = ""
        warnings: List[str] = field(default_factory=list)
        needs_context: bool = False
        follow_up_question: str = ""
        final_delivery_path: str = ""
        final_delivery_summary: str = ""
        created_at: str = field(
            default_factory=lambda: datetime.now().isoformat())
        def to_dict(self):
            return {
                "plan_id": self.plan_id, "version": self.version,
                "goal": self.goal,
                "tasks": [t.to_dict() for t in self.tasks],
                "warnings": self.warnings,
                "needs_context": self.needs_context,
                "follow_up_question": self.follow_up_question,
            }
        @classmethod
        def create_new(cls, goal, phase=0):
            return cls(plan_id=f"plan_{uuid.uuid4().hex[:8]}",
                       goal=goal, phase=phase)
        @classmethod
        def from_dict(cls, data):
            return cls(plan_id=data["plan_id"], goal=data.get("goal", ""),
                       tasks=[Task.from_dict(t)
                               for t in data.get("tasks", [])])

    _REG = {at: AgentSpec(at, at.value.replace("_", " ").title())
            for at in AgentType}
    _REG[AgentType.PLANNER] = AgentSpec(AgentType.PLANNER, "Planner")
    _REG[AgentType.TEST_AUTOMATION] = AgentSpec(
        AgentType.TEST_AUTOMATION, "Test Automation")

    def get_agent_spec(at): return _REG.get(at)
    def get_enabled_agents(): return [s for s in _REG.values() if s.enabled]

    _mod("vetinari.agents.contracts",
         AgentType=AgentType, TaskStatus=TaskStatus, AgentSpec=AgentSpec,
         AgentTask=AgentTask, AgentResult=AgentResult,
         VerificationResult=VerificationResult, Task=Task, Plan=Plan,
         AGENT_REGISTRY=_REG, get_agent_spec=get_agent_spec,
         get_enabled_agents=get_enabled_agents)

    class BaseAgent:
        def __init__(self, agent_type, config=None):
            self._agent_type = agent_type
            self._config = config or {}
            self._spec = get_agent_spec(agent_type)
            self._initialized = False
            self._context = {}
            self._adapter_manager = None
            self._web_search = None
            self._tool_registry = None

        @property
        def agent_type(self): return self._agent_type
        @property
        def name(self):
            return self._spec.name if self._spec else self._agent_type.value
        @property
        def description(self):
            return self._spec.description if self._spec else ""
        @property
        def default_model(self):
            return self._spec.default_model if self._spec else ""
        @property
        def thinking_variant(self):
            return self._spec.thinking_variant if self._spec else "medium"
        @property
        def is_initialized(self): return self._initialized

        def initialize(self, ctx):
            self._context = ctx
            self._adapter_manager = ctx.get("adapter_manager")
            self._initialized = True

        def _log(self, level, msg, **kw): pass

        def _infer(self, prompt, system_prompt=None, model_id=None,
                   max_tokens=4096, temperature=0.3, expect_json=False):
            return ""

        def _infer_json(self, prompt, system_prompt=None, model_id=None,
                        fallback=None, **kw):
            return fallback

        def validate_task(self, task):
            return task.agent_type == self._agent_type

        def prepare_task(self, task):
            from datetime import datetime
            task.started_at = datetime.now().isoformat()
            return task

        def complete_task(self, task, result):
            from datetime import datetime
            task.completed_at = datetime.now().isoformat()
            task.result = result.output
            if not result.success:
                task.error = "; ".join(result.errors)
            return task

        def get_capabilities(self): return []
        def get_system_prompt(self): return ""

        def get_metadata(self):
            return {"agent_type": self._agent_type.value, "name": self.name,
                    "capabilities": self.get_capabilities(),
                    "initialized": self._initialized}

        def verify(self, output):
            return VerificationResult(passed=True, score=1.0)

        def execute(self, task): raise NotImplementedError

        def __repr__(self):
            return f"<{self.__class__.__name__}(type={self._agent_type.value})>"

    _mod("vetinari.agents.base_agent", BaseAgent=BaseAgent)
    _mod("vetinari.adapters.base", LLMAdapter=MagicMock,
         InferenceRequest=MagicMock)
    _pkg("vetinari"); _pkg("vetinari.agents"); _pkg("vetinari.adapters")
    _pkg("vetinari.learning"); _pkg("vetinari.constraints")
    for s in [
        "vetinari.adapter_manager", "vetinari.lmstudio_adapter",
        "vetinari.token_optimizer", "vetinari.structured_logging",
        "vetinari.execution_context", "vetinari.learning.prompt_evolver",
        "vetinari.learning.quality_scorer", "vetinari.learning.feedback_loop",
        "vetinari.learning.model_selector", "vetinari.learning.training_data",
        "vetinari.learning.episode_memory", "vetinari.config",
        "vetinari.config.inference_config", "vetinari.constraints.registry",
        "vetinari.tools.web_search_tool",
    ]:
        _mod(s)

    return (AgentType, TaskStatus, AgentTask, AgentResult,
            VerificationResult, Task, Plan, BaseAgent)


(AgentType, TaskStatus, AgentTask, AgentResult,
 VerificationResult, Task, Plan, BaseAgent) = _install_stubs()

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.planner_agent import PlannerAgent, get_planner_agent
from vetinari.agents.test_automation_agent import (
    TestAutomationAgent, get_test_automation_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk(at, desc="Task", prompt=None, ctx=None, tid="t-1"):
    return AgentTask(task_id=tid, agent_type=at, description=desc,
                     prompt=prompt or desc, context=ctx or {})


def _pt(desc="Implement REST API", prompt=None, ctx=None):
    return _mk(AgentType.PLANNER, desc, prompt, ctx, "p-1")


def _tt(desc="Generate tests for auth", ctx=None):
    return _mk(AgentType.TEST_AUTOMATION, desc, ctx=ctx, tid="ta-1")


# ---------------------------------------------------------------------------
# MultiModeAgent concrete subclasses
# ---------------------------------------------------------------------------

class _C(MultiModeAgent):
    MODES = {"alpha": "_a", "beta": "_b", "gamma": "_g"}
    DEFAULT_MODE = "alpha"
    MODE_KEYWORDS = {
        "alpha": ["alpha", "first", "primary"],
        "beta":  ["beta", "secondary"],
        "gamma": ["gamma", "third", "special"],
    }
    LEGACY_TYPE_TO_MODE = {
        AgentType.EXPLORER.value: "alpha",
        AgentType.BUILDER.value:  "beta",
    }
    def get_system_prompt(self): return super().get_system_prompt()
    def get_capabilities(self): return super().get_capabilities()
    def verify(self, output): return super().verify(output)
    def _a(self, task): return AgentResult(success=True, output="alpha-out")
    def _b(self, task): return AgentResult(success=True, output="beta-out")
    def _g(self, task): return AgentResult(success=True, output="gamma-out")


class _R(_C):
    def _a(self, task): raise ValueError("boom in alpha")


class _E(MultiModeAgent):
    MODES = {}; DEFAULT_MODE = ""
    def get_system_prompt(self): return ""
    def verify(self, output): return VerificationResult(passed=True)


# ===========================================================================
# 1. MultiModeAgent Tests (30+ tests)
# ===========================================================================

class TestMMAInit(unittest.TestCase):
    def test_agent_type_stored(self):
        self.assertEqual(_C(AgentType.BUILDER).agent_type, AgentType.BUILDER)
    def test_current_mode_none(self):
        self.assertIsNone(_C(AgentType.BUILDER).current_mode)
    def test_available_modes_set(self):
        self.assertEqual(set(_C(AgentType.BUILDER).available_modes),
                         {"alpha", "beta", "gamma"})
    def test_config_empty_default(self):
        self.assertEqual(_C(AgentType.BUILDER)._config, {})
    def test_config_stored(self):
        self.assertEqual(_C(AgentType.BUILDER, {"x": 1})._config["x"], 1)
    def test_modes_dict_has_alpha(self):
        self.assertIn("alpha", _C(AgentType.BUILDER).MODES)
    def test_modes_dict_has_beta(self):
        self.assertIn("beta", _C(AgentType.BUILDER).MODES)
    def test_legacy_map_present(self):
        self.assertIn(AgentType.EXPLORER.value,
                      _C(AgentType.BUILDER).LEGACY_TYPE_TO_MODE)
    def test_default_mode_alpha(self):
        self.assertEqual(_C(AgentType.BUILDER).DEFAULT_MODE, "alpha")
    def test_repr_has_class_name(self):
        self.assertIn("_C", repr(_C(AgentType.BUILDER)))
    def test_available_modes_length(self):
        self.assertEqual(len(_C(AgentType.BUILDER).available_modes), 3)
    def test_available_modes_is_list(self):
        self.assertIsInstance(_C(AgentType.BUILDER).available_modes, list)


class TestMMAResolveMode(unittest.TestCase):
    def setUp(self): self.a = _C(AgentType.BUILDER)
    def _t(self, ctx=None, desc="x"):
        return AgentTask(task_id="r", agent_type=AgentType.BUILDER,
                         description=desc, prompt="p", context=ctx or {})
    def test_explicit_beta(self):
        self.assertEqual(self.a._resolve_mode(self._t(ctx={"mode": "beta"})), "beta")
    def test_explicit_gamma(self):
        self.assertEqual(self.a._resolve_mode(self._t(ctx={"mode": "gamma"})), "gamma")
    def test_explicit_alpha(self):
        self.assertEqual(self.a._resolve_mode(self._t(ctx={"mode": "alpha"})), "alpha")
    def test_explicit_unknown_falls_to_keyword(self):
        self.assertEqual(
            self.a._resolve_mode(self._t(ctx={"mode": "zzz"}, desc="alpha here")),
            "alpha")
    def test_legacy_explorer_to_alpha(self):
        self.assertEqual(
            self.a._resolve_mode(self._t(
                ctx={"legacy_agent_type": AgentType.EXPLORER.value})),
            "alpha")
    def test_legacy_builder_to_beta(self):
        self.assertEqual(
            self.a._resolve_mode(self._t(
                ctx={"legacy_agent_type": AgentType.BUILDER.value})),
            "beta")
    def test_keyword_alpha(self):
        self.assertEqual(
            self.a._resolve_mode(self._t(desc="primary alpha job")), "alpha")
    def test_keyword_beta(self):
        self.assertEqual(
            self.a._resolve_mode(self._t(desc="secondary beta process")), "beta")
    def test_keyword_gamma(self):
        self.assertEqual(
            self.a._resolve_mode(self._t(desc="gamma special operation")),
            "gamma")
    def test_no_match_returns_default(self):
        self.assertEqual(
            self.a._resolve_mode(self._t(desc="xyz abc")), "alpha")
    def test_empty_desc_returns_default(self):
        self.assertEqual(self.a._resolve_mode(self._t(desc="")), "alpha")
    def test_best_score_keyword(self):
        self.assertEqual(
            self.a._resolve_mode(self._t(desc="alpha primary first")), "alpha")
    def test_empty_modes_returns_empty(self):
        a = _E(AgentType.BUILDER)
        t = AgentTask(task_id="t", agent_type=AgentType.BUILDER,
                      description="d", prompt="p")
        self.assertEqual(a._resolve_mode(t), "")


class TestMMAExecute(unittest.TestCase):
    def setUp(self): self.a = _C(AgentType.BUILDER)
    def _t(self, ctx=None, desc="task"):
        return AgentTask(task_id="e", agent_type=AgentType.BUILDER,
                         description=desc, prompt="p", context=ctx or {})
    def test_alpha_out(self):
        self.assertEqual(self.a.execute(self._t(ctx={"mode": "alpha"})).output,
                         "alpha-out")
    def test_beta_out(self):
        self.assertEqual(self.a.execute(self._t(ctx={"mode": "beta"})).output,
                         "beta-out")
    def test_gamma_out(self):
        self.assertEqual(self.a.execute(self._t(ctx={"mode": "gamma"})).output,
                         "gamma-out")
    def test_success_true(self):
        self.assertTrue(self.a.execute(self._t(ctx={"mode": "alpha"})).success)
    def test_sets_current_mode(self):
        self.a.execute(self._t(ctx={"mode": "gamma"}))
        self.assertEqual(self.a.current_mode, "gamma")
    def test_missing_handler_fails(self):
        a = _C(AgentType.BUILDER); a.MODES = {"alpha": "_nosuch"}
        r = a.execute(self._t(ctx={"mode": "alpha"}))
        self.assertFalse(r.success)
        self.assertTrue(any("not implemented" in e for e in r.errors))
    def test_handler_exception_fails(self):
        a = _R(AgentType.BUILDER)
        r = a.execute(self._t(ctx={"mode": "alpha"}))
        self.assertFalse(r.success)
        self.assertTrue(any("boom in alpha" in e for e in r.errors))
    def test_returns_agent_result(self):
        self.assertIsInstance(
            self.a.execute(self._t(ctx={"mode": "beta"})), AgentResult)
    def test_keyword_dispatch(self):
        self.assertEqual(
            self.a.execute(self._t(desc="secondary beta")).output, "beta-out")
    def test_legacy_dispatch(self):
        t = self._t(ctx={"legacy_agent_type": AgentType.BUILDER.value})
        self.assertEqual(self.a.execute(t).output, "beta-out")
    def test_started_at_set(self):
        t = self._t(ctx={"mode": "alpha"})
        self.a.execute(t); self.assertNotEqual(t.started_at, "")
    def test_completed_at_set(self):
        t = self._t(ctx={"mode": "alpha"})
        self.a.execute(t); self.assertNotEqual(t.completed_at, "")
    def test_empty_modes_returns_failure(self):
        a = _E(AgentType.BUILDER)
        t = AgentTask(task_id="t", agent_type=AgentType.BUILDER,
                      description="d", prompt="p")
        self.assertFalse(a.execute(t).success)
    def test_default_fallback(self):
        self.assertEqual(
            self.a.execute(self._t(desc="no keyword match at all")).output,
            "alpha-out")


class TestMMACapabilitiesVerify(unittest.TestCase):
    def setUp(self): self.a = _C(AgentType.BUILDER)
    def test_capabilities_list(self):
        for m in ("alpha", "beta", "gamma"):
            self.assertIn(m, self.a.get_capabilities())
    def test_system_prompt_str(self):
        self.assertIsInstance(self.a.get_system_prompt(), str)
    def test_verify_none_fails(self):
        r = self.a.verify(None)
        self.assertFalse(r.passed); self.assertEqual(r.score, 0.0)
    def test_verify_non_none_passes(self):
        r = self.a.verify({"x": 1}); self.assertTrue(r.passed)
    def test_verify_returns_verification_result(self):
        self.assertIsInstance(self.a.verify("x"), VerificationResult)
    def test_empty_modes_available_empty(self):
        self.assertEqual(_E(AgentType.BUILDER).available_modes, [])


# ===========================================================================
# 2. PlannerAgent Tests (40+ tests)
# ===========================================================================

class TestPlannerInit(unittest.TestCase):
    def test_type(self):
        self.assertEqual(PlannerAgent().agent_type, AgentType.PLANNER)
    def test_max_depth(self): self.assertEqual(PlannerAgent()._max_depth, 14)
    def test_min_tasks(self): self.assertEqual(PlannerAgent()._min_tasks, 5)
    def test_max_tasks(self): self.assertEqual(PlannerAgent()._max_tasks, 15)
    def test_config_max_depth(self):
        self.assertEqual(PlannerAgent({"max_depth": 3})._max_depth, 3)
    def test_config_min_tasks(self):
        self.assertEqual(PlannerAgent({"min_tasks": 2})._min_tasks, 2)
    def test_config_max_tasks(self):
        self.assertEqual(PlannerAgent({"max_tasks": 8})._max_tasks, 8)
    def test_name(self): self.assertEqual(PlannerAgent().name, "Planner")
    def test_not_none(self): self.assertIsNotNone(PlannerAgent())
    def test_repr(self): self.assertIn("PLANNER", repr(PlannerAgent()))


class TestPlannerCapabilities(unittest.TestCase):
    def setUp(self): self.a = PlannerAgent()
    def test_prompt_str(self):
        self.assertIsInstance(self.a.get_system_prompt(), str)
    def test_prompt_nonempty(self):
        self.assertGreater(len(self.a.get_system_prompt()), 20)
    def test_caps_list(self):
        self.assertIsInstance(self.a.get_capabilities(), list)
    def test_cap_plan_gen(self):
        self.assertIn("plan_generation", self.a.get_capabilities())
    def test_cap_task_decomp(self):
        self.assertIn("task_decomposition", self.a.get_capabilities())
    def test_cap_dep_map(self):
        self.assertIn("dependency_mapping", self.a.get_capabilities())
    def test_cap_resource(self):
        self.assertIn("resource_estimation", self.a.get_capabilities())
    def test_cap_risk(self):
        self.assertIn("risk_assessment", self.a.get_capabilities())


class TestPlannerValidate(unittest.TestCase):
    def setUp(self): self.a = PlannerAgent()
    def test_valid(self): self.assertTrue(self.a.validate_task(_pt()))
    def test_wrong_type(self):
        t = AgentTask(task_id="t", agent_type=AgentType.BUILDER,
                      description="d", prompt="p")
        self.assertFalse(self.a.validate_task(t))


class TestPlannerInvalidTask(unittest.TestCase):
    def _bad(self):
        return AgentTask(task_id="t", agent_type=AgentType.BUILDER,
                         description="d", prompt="p")
    def test_fails(self):
        self.assertFalse(PlannerAgent().execute(self._bad()).success)
    def test_output_none(self):
        self.assertIsNone(PlannerAgent().execute(self._bad()).output)
    def test_has_errors(self):
        self.assertGreater(len(PlannerAgent().execute(self._bad()).errors), 0)


class TestPlannerVagueGoals(unittest.TestCase):
    def setUp(self): self.a = PlannerAgent()
    def _e(self, desc): return self.a.execute(_pt(desc=desc, prompt=desc))
    def test_one_word(self):
        r = self._e("hi")
        self.assertTrue(r.success)
        self.assertTrue(r.output.get("needs_context", False))
    def test_two_words(self):
        self.assertTrue(self._e("do stuff").output.get("needs_context", False))
    def test_fix_it(self):
        self.assertTrue(self._e("fix it").output.get("needs_context", False))
    def test_help_me(self):
        self.assertTrue(self._e("help me").output.get("needs_context", False))
    def test_vague_follow_up_question(self):
        r = self._e("stuff")
        self.assertIn("follow_up_question", r.output)
        self.assertGreater(len(r.output["follow_up_question"]), 0)
    def test_non_alnum(self):
        self.assertTrue(self._e("??? !!!").output.get("needs_context", False))
    def test_vague_no_tasks(self):
        self.assertEqual(len(self._e("go").output.get("tasks", [])), 0)
    def test_success_true_on_vague(self):
        self.assertTrue(self._e("help").success)


class TestPlannerKeywordFallback(unittest.TestCase):
    def setUp(self):
        self.a = PlannerAgent()
        self.a._infer_json = lambda *a, **kw: None

    def test_produces_tasks(self):
        self.assertGreater(len(
            self.a.execute(_pt("Implement a REST API server")).output["tasks"]), 0)
    def test_unique_ids(self):
        tasks = self.a.execute(_pt("Build a web application")).output["tasks"]
        ids = [t["id"] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
    def test_tasks_have_description(self):
        for t in self.a.execute(_pt("Create an app")).output["tasks"]:
            self.assertGreater(len(t["description"]), 0)
    def test_code_has_builder(self):
        agents = [t["assigned_agent"] for t in
                  self.a.execute(_pt("Build web app with code")).output["tasks"]]
        self.assertIn("BUILDER", agents)
    def test_ui_has_ui_planner(self):
        agents = [t["assigned_agent"] for t in
                  self.a.execute(_pt("Build frontend web dashboard ui app")).output["tasks"]]
        self.assertIn("UI_PLANNER", agents)
    def test_research_has_researcher(self):
        agents = [t["assigned_agent"] for t in
                  self.a.execute(_pt("Research and analyze AI trends")).output["tasks"]]
        self.assertIn("RESEARCHER", agents)
    def test_data_has_data_engineer(self):
        agents = [t["assigned_agent"] for t in
                  self.a.execute(_pt("Set up SQL database schema")).output["tasks"]]
        self.assertIn("DATA_ENGINEER", agents)
    def test_ends_with_security_auditor(self):
        agents = [t["assigned_agent"] for t in
                  self.a.execute(_pt("Build a Python app")).output["tasks"]]
        self.assertIn("SECURITY_AUDITOR", agents)
    def test_has_plan_id(self):
        self.assertIn("plan_id",
                      self.a.execute(_pt("Build something")).output)
    def test_has_goal(self):
        self.assertIn("goal", self.a.execute(_pt("Build a thing")).output)
    def test_metadata_task_count(self):
        self.assertGreater(
            self.a.execute(_pt("Implement feature")).metadata["task_count"], 0)
    def test_metadata_plan_id(self):
        self.assertIn("plan_id",
                      self.a.execute(_pt("Do coding work")).metadata)
    def test_plan_id_str(self):
        self.assertIsInstance(
            self.a.execute(_pt("Create a thing")).output["plan_id"], str)
    def test_dependencies_lists(self):
        for t in self.a.execute(_pt("Create Python app")).output["tasks"]:
            self.assertIsInstance(t["dependencies"], list)
    def test_first_task_no_deps(self):
        self.assertEqual(
            self.a.execute(_pt("Build a search engine")).output["tasks"][0]["dependencies"],
            [])
    def test_has_documentation_agent(self):
        agents = [t["assigned_agent"] for t in
                  self.a.execute(_pt("Build a large software system here")).output["tasks"]]
        self.assertIn("DOCUMENTATION_AGENT", agents)
    def test_success(self):
        self.assertTrue(self.a.execute(_pt("Implement a feature")).success)


class TestPlannerLLMPath(unittest.TestCase):
    def setUp(self): self.a = PlannerAgent()

    def _tasks(self, n=3):
        return [{"id": f"t{i}", "description": f"Step {i}",
                 "inputs": [], "outputs": [],
                 "dependencies": [f"t{i-1}"] if i > 1 else [],
                 "assigned_agent": "BUILDER"} for i in range(1, n + 1)]

    def test_llm_tasks_used(self):
        self.a._infer_json = lambda *a, **kw: self._tasks(3)
        r = self.a.execute(_pt("Build a complex system"))
        self.assertEqual(len(r.output["tasks"]), 3)
    def test_invalid_agent_falls_to_builder(self):
        self.a._infer_json = lambda *a, **kw: [
            {"id": "t1", "description": "Task", "inputs": [], "outputs": [],
             "dependencies": [], "assigned_agent": "NOT_REAL"}]
        r = self.a.execute(_pt("Build complex system here now"))
        self.assertEqual(r.output["tasks"][0]["assigned_agent"], "BUILDER")
    def test_empty_list_uses_keyword(self):
        self.a._infer_json = lambda *a, **kw: []
        self.assertGreater(
            len(self.a.execute(_pt("Implement REST API endpoint now")).output["tasks"]), 0)
    def test_non_list_uses_keyword(self):
        self.a._infer_json = lambda *a, **kw: {"bad": "response"}
        self.assertTrue(self.a.execute(_pt("Implement something useful now")).success)
    def test_depth_calculated(self):
        self.a._infer_json = lambda *a, **kw: self._tasks(2)
        depths = [t["depth"] for t in
                  self.a.execute(_pt("Build step by step system")).output["tasks"]]
        self.assertIn(0, depths); self.assertIn(1, depths)
    def test_too_many_tasks_warning(self):
        self.a._infer_json = lambda *a, **kw: self._tasks(20)
        self.a._max_tasks = 5
        r = self.a.execute(_pt("Build very complex system indeed now"))
        self.assertGreater(len(r.output.get("warnings", [])), 0)
    def test_skips_non_dict_items(self):
        self.a._infer_json = lambda *a, **kw: [
            "not a dict",
            {"id": "t1", "description": "Valid", "inputs": [], "outputs": [],
             "dependencies": [], "assigned_agent": "BUILDER"}]
        r = self.a.execute(_pt("Build something reasonable here"))
        self.assertEqual(len(r.output["tasks"]), 1)


class TestPlannerVerify(unittest.TestCase):
    def setUp(self): self.a = PlannerAgent()

    def _pd(self, tasks=None):
        if tasks is None:
            tasks = [{"id": f"t{i}", "description": f"T{i}",
                      "dependencies": ["t1"] if i > 1 else [],
                      "assigned_agent": "BUILDER"}
                     for i in range(1, 7)]
        return {"plan_id": "plan_abc", "goal": "Build thing", "tasks": tasks}

    def test_valid_passes(self):
        self.assertTrue(self.a.verify(self._pd()).passed)
    def test_non_dict_fails(self):
        self.assertFalse(self.a.verify("x").passed)
    def test_missing_plan_id_issue(self):
        p = self._pd(); del p["plan_id"]
        self.assertIn("plan_id", str(self.a.verify(p).issues))
    def test_missing_goal_issue(self):
        p = self._pd(); del p["goal"]
        self.assertIn("goal", str(self.a.verify(p).issues))
    def test_too_few_tasks_issue(self):
        p = self._pd(tasks=[{"id": "t1", "description": "Only",
                             "dependencies": []}])
        text = str(self.a.verify(p).issues).lower()
        self.assertTrue("too few" in text or "insufficient" in text)
    def test_no_deps_issue(self):
        p = self._pd(tasks=[{"id": f"t{i}", "description": f"T{i}",
                             "dependencies": []} for i in range(1, 7)])
        self.assertTrue(any("depend" in str(x).lower()
                            for x in self.a.verify(p).issues))
    def test_score_float(self):
        self.assertIsInstance(self.a.verify(self._pd()).score, float)
    def test_score_range(self):
        s = self.a.verify(self._pd()).score
        self.assertGreaterEqual(s, 0.0); self.assertLessEqual(s, 1.0)
    def test_empty_dict_fails(self):
        self.assertFalse(self.a.verify({}).passed)


class TestGetPlannerAgent(unittest.TestCase):
    def setUp(self):
        import vetinari.agents.planner_agent as m
        m._planner_agent = None

    def test_returns_instance(self):
        self.assertIsInstance(get_planner_agent(), PlannerAgent)
    def test_singleton(self):
        self.assertIs(get_planner_agent(), get_planner_agent())
    def test_config_applied(self):
        import vetinari.agents.planner_agent as m; m._planner_agent = None
        self.assertEqual(get_planner_agent(config={"max_tasks": 7})._max_tasks, 7)
    def test_subsequent_ignores_config(self):
        import vetinari.agents.planner_agent as m; m._planner_agent = None
        get_planner_agent(config={"max_tasks": 7})
        self.assertEqual(get_planner_agent(config={"max_tasks": 99})._max_tasks, 7)


class TestPlannerException(unittest.TestCase):
    def test_generate_plan_exception(self):
        a = PlannerAgent()
        def _raise(g, c): raise RuntimeError("kaboom")
        a._generate_plan = _raise
        r = a.execute(_pt("Build a real application"))
        self.assertFalse(r.success); self.assertIn("kaboom", r.errors[0])


# ===========================================================================
# 3. TestAutomationAgent Tests (30+ tests)
# ===========================================================================

class TestTAInit(unittest.TestCase):
    def test_type(self):
        self.assertEqual(TestAutomationAgent().agent_type, AgentType.TEST_AUTOMATION)
    def test_framework(self):
        self.assertEqual(TestAutomationAgent()._test_framework, "pytest")
    def test_language(self):
        self.assertEqual(TestAutomationAgent()._language, "python")
    def test_config_framework(self):
        self.assertEqual(
            TestAutomationAgent({"test_framework": "unittest"})._test_framework,
            "unittest")
    def test_config_language(self):
        self.assertEqual(
            TestAutomationAgent({"language": "js"})._language, "js")
    def test_name(self):
        self.assertEqual(TestAutomationAgent().name, "Test Automation")
    def test_not_none(self): self.assertIsNotNone(TestAutomationAgent())
    def test_repr(self):
        self.assertIn("TEST_AUTOMATION", repr(TestAutomationAgent()))


class TestTACapabilities(unittest.TestCase):
    def setUp(self): self.a = TestAutomationAgent()
    def test_prompt_str(self):
        self.assertIsInstance(self.a.get_system_prompt(), str)
    def test_prompt_has_pytest(self):
        self.assertIn("pytest", self.a.get_system_prompt())
    def test_prompt_has_python(self):
        self.assertIn("python", self.a.get_system_prompt())
    def test_caps_list(self):
        self.assertIsInstance(self.a.get_capabilities(), list)
    def test_cap_unit(self):
        self.assertIn("unit_test_generation", self.a.get_capabilities())
    def test_cap_coverage(self):
        self.assertIn("coverage_analysis", self.a.get_capabilities())
    def test_cap_execution(self):
        self.assertIn("test_execution", self.a.get_capabilities())
    def test_cap_ci(self):
        self.assertIn("ci_integration", self.a.get_capabilities())


class TestTAValidate(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(TestAutomationAgent().validate_task(_tt()))
    def test_wrong(self):
        t = AgentTask(task_id="t", agent_type=AgentType.PLANNER,
                      description="d", prompt="p")
        self.assertFalse(TestAutomationAgent().validate_task(t))


class TestTAInvalidExec(unittest.TestCase):
    def _bad(self):
        return AgentTask(task_id="t", agent_type=AgentType.PLANNER,
                         description="d", prompt="p")
    def test_fails(self):
        self.assertFalse(TestAutomationAgent().execute(self._bad()).success)
    def test_has_errors(self):
        self.assertGreater(
            len(TestAutomationAgent().execute(self._bad()).errors), 0)


class TestTAFallback(unittest.TestCase):
    def setUp(self):
        self.a = TestAutomationAgent()
        self.a._infer_json = lambda *a, **kw: None

    def _run(self, ctx=None):
        return self.a.execute(_tt(ctx=ctx or {"features": ["auth"]}))

    def test_succeeds(self): self.assertTrue(self._run().success)
    def test_has_test_files(self): self.assertIn("test_files", self._run().output)
    def test_has_test_scripts(self):
        self.assertIn("test_scripts", self._run().output)
    def test_has_coverage_report(self):
        self.assertIn("coverage_report", self._run().output)
    def test_has_test_data(self): self.assertIn("test_data", self._run().output)
    def test_has_summary(self): self.assertIn("summary", self._run().output)
    def test_has_test_results(self):
        self.assertIn("test_results", self._run().output)
    def test_metadata_features(self):
        r = self.a.execute(_tt(ctx={"features": ["a", "b"]}))
        self.assertEqual(r.metadata["features_tested"], 2)
    def test_metadata_coverage_target(self):
        r = self.a.execute(_tt(ctx={"features": ["auth"], "coverage_target": 0.9}))
        self.assertEqual(r.metadata["coverage_target"], 0.9)
    def test_default_coverage_08(self):
        self.assertEqual(self._run().metadata["coverage_target"], 0.8)
    def test_no_features_ok(self):
        self.assertTrue(
            self.a.execute(_tt(desc="Generate tests for payment")).success)
    def test_code_extracts_functions(self):
        code = "def login(u, p):\\n    pass\\ndef logout(u):\\n    pass\\n"
        r = self.a.execute(_tt(ctx={"features": ["auth"], "code": code}))
        self.assertGreater(r.metadata["functions_found"], 0)
    def test_returns_agent_result(self):
        self.assertIsInstance(self._run(), AgentResult)


class TestTAExtractFunctions(unittest.TestCase):
    def setUp(self): self.a = TestAutomationAgent()
    def test_def(self):
        self.assertIn("foo", self.a._extract_functions("def foo(x): pass"))
    def test_async_def(self):
        self.assertIn("bar", self.a._extract_functions("async def bar(x): pass"))
    def test_class(self):
        self.assertIn("MyClass", self.a._extract_functions("class MyClass:\\n    pass"))
    def test_empty(self): self.assertEqual([], self.a._extract_functions(""))
    def test_none(self): self.assertEqual([], self.a._extract_functions(None))
    def test_no_funcs(self):
        self.assertEqual([], self.a._extract_functions("x = 1\\ny = 2"))
    def test_dedup(self):
        self.assertEqual(
            1, self.a._extract_functions("def foo(): pass\\ndef foo(): pass").count("foo"))
    def test_multiple(self):
        self.assertEqual(
            3, len(self.a._extract_functions("def a(): pass\\ndef b(): pass\\ndef c(): pass")))


class TestTAFallbackTests(unittest.TestCase):
    def setUp(self): self.a = TestAutomationAgent()
    def test_is_dict(self):
        self.assertIsInstance(self.a._fallback_tests(["auth"], [], 0.8), dict)
    def test_has_files(self):
        self.assertGreater(
            len(self.a._fallback_tests(["auth"], [], 0.8)["test_files"]), 0)
    def test_has_scripts(self):
        self.assertIn("test_scripts", self.a._fallback_tests(["auth"], [], 0.8))
    def test_coverage_target(self):
        self.assertEqual(
            0.9, self.a._fallback_tests(["auth"], [], 0.9)["coverage_report"]["target"])
    def test_multi_features(self):
        self.assertEqual(
            3, len(self.a._fallback_tests(["a", "b", "c"], [], 0.8)["test_files"]))
    def test_funcs_in_content(self):
        r = self.a._fallback_tests(["auth"], ["login", "logout"], 0.8)
        self.assertIn("login", r["test_scripts"][0]["content"])
    def test_empty_features_default(self):
        self.assertEqual(
            1, len(self.a._fallback_tests([], [], 0.8)["test_files"]))
    def test_has_ci_config(self):
        self.assertIn("ci_config", self.a._fallback_tests(["auth"], [], 0.8))
    def test_has_summary(self):
        self.assertIn("summary", self.a._fallback_tests(["auth"], [], 0.8))
    def test_test_data_fixtures(self):
        self.assertIn(
            "fixtures", self.a._fallback_tests(["auth"], [], 0.8)["test_data"])


class TestTAVerify(unittest.TestCase):
    def setUp(self): self.a = TestAutomationAgent()
    def _g(self):
        return {
            "test_files": [{"name": "t.py", "test_count": 3}],
            "test_scripts": [{"name": "t.py",
                              "content": "def test_x(): assert 1 == 1"}],
            "test_data": {"fixtures": []},
            "coverage_report": {"target": 0.8},
        }
    def test_valid_passes(self): self.assertTrue(self.a.verify(self._g()).passed)
    def test_non_dict_fails(self):
        r = self.a.verify("bad"); self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.0)
    def test_missing_files_reduces(self):
        o = self._g(); del o["test_files"]
        self.assertLess(self.a.verify(o).score, 1.0)
    def test_missing_scripts_reduces(self):
        o = self._g(); del o["test_scripts"]
        self.assertLess(self.a.verify(o).score, 1.0)
    def test_missing_data_reduces(self):
        o = self._g(); del o["test_data"]
        self.assertLess(self.a.verify(o).score, 1.0)
    def test_stub_tests_flagged(self):
        o = self._g()
        o["test_scripts"] = [{"name": "bad.py",
                               "content": "assert True\\nassert True\\nassert True\\nassert True\\nassert x==1"}]
        self.assertTrue(any(i.get("type") == "stub_tests"
                            for i in self.a.verify(o).issues))
    def test_score_nonneg(self):
        self.assertGreaterEqual(self.a.verify({}).score, 0.0)
    def test_score_float(self):
        self.assertIsInstance(self.a.verify(self._g()).score, float)
    def test_empty_files_fails(self):
        o = self._g(); o["test_files"] = []
        self.assertFalse(self.a.verify(o).passed)


class TestTAExecuteTests(unittest.TestCase):
    def setUp(self): self.a = TestAutomationAgent()
    def test_empty_zero(self):
        r = self.a.execute_tests([])
        self.assertEqual(r["passed"], 0); self.assertEqual(r["returncode"], 0)
    def test_empty_content_error(self):
        self.assertIn("errors",
                      self.a.execute_tests([{"name": "t.py", "content": ""}]))
    def test_keys_present(self):
        for k in ("passed", "failed", "errors", "output", "returncode"):
            self.assertIn(k, self.a.execute_tests([]))
    def test_passing_test(self):
        r = self.a.execute_tests([
            {"name": "tp.py", "content": "def test_one():\\n    assert 1+1==2\\n"}])
        self.assertGreaterEqual(r["passed"], 1); self.assertEqual(r["failed"], 0)
    def test_failing_test(self):
        r = self.a.execute_tests([
            {"name": "tf.py", "content": "def test_fail():\\n    assert False\\n"}])
        self.assertGreater(r["failed"], 0)
    def test_output_str(self):
        r = self.a.execute_tests([
            {"name": "ts.py", "content": "def test_x():\\n    pass\\n"}])
        self.assertIsInstance(r["output"], str)
    def test_timeout(self):
        r = self.a.execute_tests([
            {"name": "th.py",
             "content": "import time\\ndef test_h():\\n    time.sleep(100)\\n"}],
            timeout=1)
        self.assertGreater(len(r["errors"]), 0)
        self.assertFalse(r.get("success", True))
    def test_errors_list(self):
        self.assertIsInstance(self.a.execute_tests([])["errors"], list)
    def test_multiple_scripts(self):
        r = self.a.execute_tests([
            {"name": "ta.py", "content": "def test_a():\\n    assert True\\n"},
            {"name": "tb.py", "content": "def test_b():\\n    assert True\\n"},
        ])
        self.assertGreaterEqual(r["passed"], 2)


class TestTAWriteToDisk(unittest.TestCase):
    def test_returns_list(self):
        import tempfile
        a = TestAutomationAgent()
        with tempfile.TemporaryDirectory() as d:
            r = a.write_tests_to_disk(
                [{"name": "t.py", "content": "def test_x(): pass\\n"}], d)
        self.assertIsInstance(r, list); self.assertEqual(len(r), 1)

    def test_file_created(self):
        import tempfile, os
        a = TestAutomationAgent()
        with tempfile.TemporaryDirectory() as d:
            paths = a.write_tests_to_disk(
                [{"name": "t.py", "content": "def test_x(): pass\\n"}], d)
            self.assertTrue(os.path.exists(paths[0]))

    def test_empty_content_skipped(self):
        import tempfile
        a = TestAutomationAgent()
        with tempfile.TemporaryDirectory() as d:
            r = a.write_tests_to_disk([{"name": "t.py", "content": ""}], d)
        self.assertEqual(len(r), 0)

    def test_creates_dir(self):
        import tempfile, os
        a = TestAutomationAgent()
        with tempfile.TemporaryDirectory() as d:
            new_dir = os.path.join(d, "sub", "tests")
            a.write_tests_to_disk(
                [{"name": "t.py", "content": "def test_x(): pass\\n"}], new_dir)
            self.assertTrue(os.path.isdir(new_dir))


class TestGetTAAgent(unittest.TestCase):
    def setUp(self):
        import vetinari.agents.test_automation_agent as m
        m._test_automation_agent = None

    def test_returns_instance(self):
        self.assertIsInstance(get_test_automation_agent(), TestAutomationAgent)
    def test_singleton(self):
        self.assertIs(get_test_automation_agent(), get_test_automation_agent())
    def test_config(self):
        import vetinari.agents.test_automation_agent as m
        m._test_automation_agent = None
        self.assertEqual(
            get_test_automation_agent(
                config={"test_framework": "unittest"})._test_framework,
            "unittest")
    def test_subsequent_ignores(self):
        import vetinari.agents.test_automation_agent as m
        m._test_automation_agent = None
        get_test_automation_agent(config={"test_framework": "unittest"})
        self.assertEqual(
            get_test_automation_agent(
                config={"test_framework": "nose"})._test_framework,
            "unittest")


if __name__ == "__main__":
    unittest.main(verbosity=2)
"""

with open(TARGET, "w", encoding="utf-8") as fh:
    fh.write(CONTENT)

print(f"Written {len(CONTENT)} chars to {TARGET}")
