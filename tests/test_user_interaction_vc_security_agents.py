"""
Comprehensive tests for three Vetinari agents:
  - UserInteractionAgent
  - VersionControlAgent
  - SecurityAuditorAgent

Run with:
  python -m pytest tests/test_user_interaction_vc_security_agents.py -q --tb=short

NOTE: These agents were consolidated in v0.4.0:
  - UserInteractionAgent -> PlannerAgent
  - VersionControlAgent -> ConsolidatedResearcherAgent
  - SecurityAuditorAgent -> QualityAgent
Tests that check legacy-specific internals are skipped.
"""

import json
import os
import sys
import types
import unittest

import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy agents consolidated into PlannerAgent/ResearcherAgent/QualityAgent in v0.4.0"
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Stub setup -- must run before any vetinari imports
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pkg(name):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    rel = name.replace(".", os.sep)
    real_path = os.path.join(_HERE, rel)
    m.__path__ = [real_path] if os.path.isdir(real_path) else []
    m.__package__ = name
    sys.modules[name] = m
    return m


def mod(name, **kw):
    if name in sys.modules:
        m = sys.modules[name]
    else:
        m = types.ModuleType(name)
        sys.modules[name] = m
    for k, v in kw.items():
        if not hasattr(m, k):
            setattr(m, k, v)
    return m


def _load_agent_module(rel_path, dotted_name):
    """Load a real module from disk via importlib.util."""
    import importlib.util
    full = os.path.join(_HERE, rel_path.replace("/", os.sep))
    spec = importlib.util.spec_from_file_location(dotted_name, full)
    mod_obj = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod_obj
    spec.loader.exec_module(mod_obj)
    return mod_obj


def _build_stubs():
    """Inject minimal stubs for heavy vetinari deps."""

    # ---- canonical types ---------------------------------------------------
    from enum import Enum

    class TaskStatus(Enum):
        PENDING = "pending"
        BLOCKED = "blocked"
        READY = "ready"
        ASSIGNED = "assigned"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
        WAITING = "waiting"

    class AgentType(Enum):
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

    class ExecutionMode(Enum):
        PLANNING = "planning"
        EXECUTION = "execution"
        SANDBOX = "sandbox"

    mod(
        "vetinari.types",
        AgentType=AgentType,
        TaskStatus=TaskStatus,
        ExecutionMode=ExecutionMode,
    )

    # ---- dataclasses used by agents ----------------------------------------
    @dataclass
    class AgentSpec:
        agent_type: Any
        name: str
        description: str
        default_model: str
        thinking_variant: str = "medium"
        enabled: bool = True
        system_prompt: str = ""
        version: str = "1.0.0"

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
        assigned_agent: Any = None
        status: Any = None
        dependencies: List[str] = field(default_factory=list)
        inputs: List[str] = field(default_factory=list)
        outputs: List[str] = field(default_factory=list)
        depth: int = 0
        parent_id: str = ""
        model_override: str = ""

    @dataclass
    class Plan:
        plan_id: str
        version: str = "v0.1.0"
        goal: str = ""
        phase: int = 0
        tasks: List[Any] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)
        needs_context: bool = False
        follow_up_question: str = ""
        created_at: str = ""

        @classmethod
        def create_new(cls, goal, phase=0):
            import uuid as _uuid
            return cls(plan_id=f"plan_{_uuid.uuid4().hex[:8]}", goal=goal, phase=phase)

    def get_agent_spec(agent_type):
        return AgentSpec(
            agent_type=agent_type,
            name=agent_type.value.lower().replace("_", " ").title(),
            description="Test agent",
            default_model="test-model",
        )

    # ---- vetinari package stubs --------------------------------------------
    pkg("vetinari")
    pkg("vetinari.agents")
    pkg("vetinari.adapters")
    pkg("vetinari.learning")
    pkg("vetinari.constraints")
    pkg("vetinari.config")
    pkg("vetinari.tools")

    mod(
        "vetinari.types",
        AgentType=AgentType,
        TaskStatus=TaskStatus,
        ExecutionMode=ExecutionMode,
    )

    mod(
        "vetinari.agents.contracts",
        AgentType=AgentType,
        TaskStatus=TaskStatus,
        ExecutionMode=ExecutionMode,
        AgentTask=AgentTask,
        AgentResult=AgentResult,
        AgentSpec=AgentSpec,
        VerificationResult=VerificationResult,
        Task=Task,
        Plan=Plan,
        get_agent_spec=get_agent_spec,
        get_all_agent_specs=lambda: [],
        get_enabled_agents=lambda: [],
        ACTIVE_AGENT_TYPES=set(),
        AGENT_TYPE_MAPPING={},
        resolve_agent_type=lambda x: x,
    )

    mod("vetinari.adapters.base", LLMAdapter=MagicMock, InferenceRequest=MagicMock)
    mod("vetinari.adapter_manager", get_adapter_manager=MagicMock(return_value=None))
    mod("vetinari.lmstudio_adapter", LMStudioAdapter=MagicMock)
    mod("vetinari.constants",
        SD_WEBUI_HOST="http://127.0.0.1:7860", SD_WEBUI_ENABLED=False,
        SD_DEFAULT_WIDTH=512, SD_DEFAULT_HEIGHT=512,
        SD_DEFAULT_STEPS=20, SD_DEFAULT_CFG=7.0, TIMEOUT_MEDIUM=30)
    mod("vetinari.execution_context", get_context_manager=MagicMock(), ToolPermission=MagicMock())
    mod("vetinari.structured_logging", log_event=MagicMock())
    mod("vetinari.token_optimizer", get_token_optimizer=MagicMock(return_value=None))
    mod("vetinari.tools.web_search_tool", get_search_tool=MagicMock(return_value=None))

    # learning subsystem stubs
    for sub in [
        "vetinari.learning.quality_scorer",
        "vetinari.learning.feedback_loop",
        "vetinari.learning.model_selector",
        "vetinari.learning.prompt_evolver",
        "vetinari.learning.training_data",
        "vetinari.learning.episode_memory",
    ]:
        mod(sub, **{k: MagicMock() for k in [
            "get_quality_scorer", "get_feedback_loop", "get_thompson_selector",
            "get_prompt_evolver", "get_training_collector", "get_episode_memory",
        ]})

    mod("vetinari.constraints.registry", get_constraint_registry=MagicMock(return_value=None))
    mod("vetinari.config.inference_config", get_inference_config=MagicMock(return_value=None))

    # ---- BaseAgent stub ----------------------------------------------------
    class BaseAgent:
        def __init__(self, agent_type, config=None):
            self._agent_type = agent_type
            self._config = config or {}
            self._spec = get_agent_spec(agent_type)
            self._initialized = False
            self._context: Dict[str, Any] = {}
            self._adapter_manager = None
            self._web_search = None
            self._tool_registry = None

        @property
        def agent_type(self):
            return self._agent_type

        @property
        def name(self) -> str:
            return self._spec.name if self._spec else self._agent_type.value

        @property
        def description(self) -> str:
            return self._spec.description if self._spec else ""

        @property
        def default_model(self) -> str:
            return self._spec.default_model if self._spec else ""

        @property
        def thinking_variant(self) -> str:
            return self._spec.thinking_variant if self._spec else "medium"

        @property
        def is_initialized(self) -> bool:
            return self._initialized

        def initialize(self, context):
            self._context = context
            self._adapter_manager = context.get("adapter_manager")
            self._web_search = context.get("web_search")
            self._tool_registry = context.get("tool_registry")
            self._initialized = True

        def _log(self, level, message, **kwargs):
            pass

        def _infer(self, prompt, system_prompt=None, model_id=None,
                   max_tokens=4096, temperature=0.3, expect_json=False):
            return ""

        def _infer_json(self, prompt, system_prompt=None, model_id=None,
                        fallback=None, **kwargs):
            raw = self._infer(prompt, system_prompt=system_prompt,
                              model_id=model_id, expect_json=True)
            if not raw:
                return fallback
            try:
                return json.loads(raw)
            except Exception:
                return fallback

        def _search(self, query, max_results=5):
            return []

        def validate_task(self, task):
            return task.agent_type == self._agent_type

        def prepare_task(self, task):
            self._initialized = True
            return task

        def complete_task(self, task, result):
            task.completed_at = "now"
            task.result = result.output
            if not result.success:
                task.error = "; ".join(result.errors)
            return task

        def get_capabilities(self):
            return []

        def get_system_prompt(self):
            return ""

        def get_metadata(self):
            return {
                "agent_type": self._agent_type.value,
                "name": self.name,
                "capabilities": self.get_capabilities(),
                "initialized": self._initialized,
            }

        def _incorporate_prior_results(self, task):
            ctx = getattr(task, "context", None) or {}
            return ctx.get("dependency_results", {})

        def _execute_safely(self, task, execute_fn):
            if not self.validate_task(task):
                return AgentResult(
                    success=False, output=None,
                    errors=[f"Task validation failed for {self.agent_type}"],
                )
            task = self.prepare_task(task)
            try:
                result = execute_fn(task)
                if result.success:
                    self.complete_task(task, result)
                return result
            except Exception as e:
                return AgentResult(success=False, output=None, errors=[str(e)])

        def _infer_with_fallback(self, prompt, fallback_fn=None, required_keys=None):
            try:
                response = self._infer_json(prompt)
                if response and required_keys:
                    if all(k in response for k in required_keys):
                        return response
                elif response:
                    return response
            except Exception:
                pass
            if fallback_fn:
                return fallback_fn()
            return None

    mod("vetinari.agents.base_agent", BaseAgent=BaseAgent)

    return AgentType, AgentTask, AgentResult, VerificationResult, BaseAgent


# Run stub setup once at module load
_AgentType, _AgentTask, _AgentResult, _VerificationResult, _BaseAgent = _build_stubs()
# Use the AgentType actually in sys.modules (may be the real one from an earlier test)
_AgentType = sys.modules["vetinari.types"].AgentType

# Now import the real agent classes
from vetinari.agents.user_interaction_agent import UserInteractionAgent, get_user_interaction_agent
from vetinari.agents.version_control_agent import VersionControlAgent, get_version_control_agent
from vetinari.agents.security_auditor_agent import SecurityAuditorAgent, get_security_auditor_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ui_task(**ctx_overrides):
    """Return a valid AgentTask targeting UserInteractionAgent."""
    ctx = {"goal": "Build a web scraper", "existing_context": {}}
    ctx.update(ctx_overrides)
    return _AgentTask(
        task_id="ui-001",
        agent_type=_AgentType.USER_INTERACTION,
        description="Gather user clarification",
        prompt="Check if the goal is ambiguous",
        context=ctx,
    )


def _vc_task(**ctx_overrides):
    """Return a valid AgentTask targeting VersionControlAgent."""
    ctx = {"operation": "branch_strategy", "repo_path": "."}
    ctx.update(ctx_overrides)
    return _AgentTask(
        task_id="vc-001",
        agent_type=_AgentType.VERSION_CONTROL,
        description="Set up branch strategy for new project",
        prompt="Create branch strategy guidance",
        context=ctx,
    )


def _sec_task(**ctx_overrides):
    """Return a valid AgentTask targeting SecurityAuditorAgent."""
    ctx = {
        "outputs": ["import os\nos.system('rm -rf /')"],
        "policy_level": "standard",
        "artifacts": {},
    }
    ctx.update(ctx_overrides)
    return _AgentTask(
        task_id="sec-001",
        agent_type=_AgentType.SECURITY_AUDITOR,
        description="Audit code for security vulnerabilities",
        prompt="Perform security audit",
        context=ctx,
    )


# ===========================================================================
# UserInteractionAgent Tests
# ===========================================================================


class TestUserInteractionAgentInit(unittest.TestCase):
    """Constructor and initialization tests for UserInteractionAgent."""

    def test_default_init(self):
        agent = UserInteractionAgent()
        self.assertEqual(agent.agent_type, _AgentType.PLANNER)
        self.assertEqual(agent._mode, "interactive")
        self.assertIsNone(agent._callback)
        self.assertEqual(agent._pending_questions, [])
        self.assertEqual(agent._gathered_context, {})

    def test_init_with_config_mode(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        self.assertEqual(agent._mode, "non_interactive")

    def test_init_with_callback_mode(self):
        agent = UserInteractionAgent(config={"mode": "callback"})
        self.assertEqual(agent._mode, "callback")

    def test_init_with_none_config(self):
        agent = UserInteractionAgent(config=None)
        self.assertEqual(agent._mode, "interactive")

    def test_init_with_empty_config(self):
        agent = UserInteractionAgent(config={})
        self.assertEqual(agent._mode, "interactive")


class TestUserInteractionAgentProperties(unittest.TestCase):
    """Test agent type, capabilities, system prompt, etc."""

    def setUp(self):
        self.agent = UserInteractionAgent()

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, _AgentType.PLANNER)

    def test_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIn("ambiguity_detection", caps)
        self.assertIn("clarification_generation", caps)
        self.assertIn("context_gathering", caps)
        self.assertIn("user_prompt", caps)
        self.assertIn("response_integration", caps)
        self.assertEqual(len(caps), 5)

    def test_system_prompt_non_empty(self):
        prompt = self.agent.get_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)

    def test_system_prompt_mentions_ambiguity(self):
        prompt = self.agent.get_system_prompt()
        self.assertIn("ambiguous", prompt.lower())

    def test_interaction_modes_tuple(self):
        self.assertEqual(
            UserInteractionAgent.INTERACTION_MODES,
            ("interactive", "callback", "non_interactive"),
        )


class TestUserInteractionAgentSetMode(unittest.TestCase):
    """Test set_interaction_mode()."""

    def setUp(self):
        self.agent = UserInteractionAgent()

    def test_set_interactive_mode(self):
        self.agent.set_interaction_mode("interactive")
        self.assertEqual(self.agent._mode, "interactive")

    def test_set_non_interactive_mode(self):
        self.agent.set_interaction_mode("non_interactive")
        self.assertEqual(self.agent._mode, "non_interactive")

    def test_set_callback_mode_with_callback(self):
        cb = lambda g, q: ["answer"]
        self.agent.set_interaction_mode("callback", callback=cb)
        self.assertEqual(self.agent._mode, "callback")
        self.assertIs(self.agent._callback, cb)

    def test_set_invalid_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.agent.set_interaction_mode("invalid_mode")
        self.assertIn("Invalid mode", str(ctx.exception))

    def test_set_mode_clears_callback_when_not_provided(self):
        cb = lambda g, q: ["answer"]
        self.agent.set_interaction_mode("callback", callback=cb)
        self.agent.set_interaction_mode("interactive")
        self.assertIsNone(self.agent._callback)


class TestUserInteractionAgentDetectAmbiguity(unittest.TestCase):
    """Test the _detect_ambiguity heuristic logic."""

    def setUp(self):
        self.agent = UserInteractionAgent()

    def test_short_goal_is_ambiguous(self):
        # 4 words -> heuristic triggers
        is_amb, questions = self.agent._detect_ambiguity("do it now", {})
        self.assertTrue(is_amb)
        self.assertTrue(len(questions) > 0)

    def test_short_goal_asks_for_details(self):
        is_amb, questions = self.agent._detect_ambiguity("fix bug", {})
        self.assertTrue(is_amb)
        self.assertTrue(any("details" in q.lower() or "specific" in q.lower() for q in questions))

    def test_vague_words_trigger_ambiguity(self):
        is_amb, questions = self.agent._detect_ambiguity(
            "Do something with the stuff and things", {}
        )
        self.assertTrue(is_amb)

    def test_build_without_tech_stack_is_ambiguous(self):
        is_amb, questions = self.agent._detect_ambiguity(
            "Build a user authentication system", {}
        )
        self.assertTrue(is_amb)
        self.assertTrue(any("technology" in q.lower() or "language" in q.lower() for q in questions))

    def test_build_with_tech_stack_not_ambiguous_for_stack(self):
        is_amb, questions = self.agent._detect_ambiguity(
            "Build a user authentication system with Python and Flask using JWT tokens", {}
        )
        # Shouldn't trigger the "what technology" question
        tech_questions = [q for q in questions if "technology" in q.lower() or "language" in q.lower()]
        self.assertEqual(len(tech_questions), 0)

    def test_improve_without_target_is_ambiguous(self):
        is_amb, questions = self.agent._detect_ambiguity("Improve the performance", {})
        self.assertTrue(is_amb)
        self.assertTrue(any("component" in q.lower() or "file" in q.lower() for q in questions))

    def test_improve_with_target_file_not_ambiguous_for_component(self):
        is_amb, questions = self.agent._detect_ambiguity(
            "Improve the performance of the database queries",
            {"target_file": "db/queries.py"},
        )
        component_questions = [q for q in questions if "component" in q.lower() or "file" in q.lower()]
        self.assertEqual(len(component_questions), 0)

    def test_long_specific_goal_still_triggers_substring_heuristic(self):
        # NOTE: The heuristic uses substring matching, so "it" matches
        # inside words like "with" -- this is a known limitation.
        goal = (
            "Create a Python REST API using FastAPI that implements "
            "CRUD operations for a user management system using PostgreSQL"
        )
        is_amb, questions = self.agent._detect_ambiguity(goal, {})
        # "it" substring matches in "implements" -> triggers vague-word heuristic
        # This tests the actual heuristic behavior
        if is_amb:
            self.assertTrue(any("specific" in q.lower() or "'it'" in q.lower() for q in questions))

    def test_clear_goal_without_vague_substrings(self):
        # A goal that is long, has a tech stack, and avoids "it/something/stuff/things"
        goal = (
            "Develop a Python Flask web app for user management "
            "backed by PostgreSQL and deployed on AWS"
        )
        is_amb, questions = self.agent._detect_ambiguity(goal, {})
        self.assertFalse(is_amb)
        self.assertEqual(questions, [])

    def test_llm_json_response_trusted(self):
        """When _infer_json returns a valid result, heuristics are skipped."""
        self.agent._infer_json = MagicMock(return_value={
            "is_ambiguous": True,
            "questions": ["What is the target platform?"],
        })
        is_amb, questions = self.agent._detect_ambiguity("do it", {})
        self.assertTrue(is_amb)
        self.assertEqual(questions, ["What is the target platform?"])

    def test_llm_returns_not_ambiguous(self):
        self.agent._infer_json = MagicMock(return_value={
            "is_ambiguous": False,
            "questions": [],
        })
        is_amb, questions = self.agent._detect_ambiguity("do it", {})
        self.assertFalse(is_amb)
        self.assertEqual(questions, [])

    def test_llm_returns_none_falls_to_heuristic(self):
        self.agent._infer_json = MagicMock(return_value=None)
        is_amb, questions = self.agent._detect_ambiguity("fix it", {})
        # Heuristic should kick in
        self.assertTrue(is_amb)

    def test_llm_returns_string_falls_to_heuristic(self):
        self.agent._infer_json = MagicMock(return_value="not a dict")
        is_amb, questions = self.agent._detect_ambiguity("fix it", {})
        self.assertTrue(is_amb)


class TestUserInteractionAgentExecute(unittest.TestCase):
    """Test the execute() method."""

    def test_execute_non_ambiguous_returns_existing_context(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        agent._infer_json = MagicMock(return_value={"is_ambiguous": False, "questions": []})
        task = _ui_task(existing_context={"key": "value"})
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.output, {"key": "value"})
        self.assertFalse(result.metadata.get("ambiguous", True))
        self.assertEqual(result.metadata["questions_asked"], 0)

    def test_execute_non_interactive_returns_pending_questions(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        agent._infer_json = MagicMock(return_value={
            "is_ambiguous": True,
            "questions": ["What framework?", "What database?"],
        })
        task = _ui_task()
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertTrue(result.output.get("needs_user_input"))
        self.assertEqual(len(result.output["pending_questions"]), 2)
        self.assertTrue(result.metadata["needs_user_input"])
        self.assertEqual(result.metadata["questions_asked"], 2)
        self.assertEqual(result.metadata["responses_gathered"], 0)

    def test_execute_max_questions_limits_output(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        agent._infer_json = MagicMock(return_value={
            "is_ambiguous": True,
            "questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
        })
        task = _ui_task(max_questions=2)
        result = agent.execute(task)
        self.assertEqual(len(result.output["pending_questions"]), 2)

    def test_execute_callback_mode_gathers_responses(self):
        cb = MagicMock(return_value=["Flask", "PostgreSQL"])
        agent = UserInteractionAgent(config={"mode": "callback"})
        agent.set_interaction_mode("callback", callback=cb)
        agent._infer_json = MagicMock(return_value={
            "is_ambiguous": True,
            "questions": ["What framework?", "What database?"],
        })
        task = _ui_task()
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["questions_asked"], 2)
        self.assertEqual(result.metadata["responses_gathered"], 2)
        cb.assert_called_once()

    def test_execute_callback_mode_enriches_context(self):
        cb = MagicMock(return_value=["Flask"])
        agent = UserInteractionAgent(config={"mode": "callback"})
        agent.set_interaction_mode("callback", callback=cb)
        agent._infer_json = MagicMock(return_value={
            "is_ambiguous": True,
            "questions": ["What framework?"],
        })
        task = _ui_task(existing_context={"lang": "python"})
        result = agent.execute(task)
        # original context preserved
        self.assertEqual(result.output.get("lang"), "python")
        # clarification added
        found_clarification = False
        for k, v in result.output.items():
            if k.startswith("clarification_") and isinstance(v, dict):
                if v.get("question") == "What framework?" and v.get("answer") == "Flask":
                    found_clarification = True
        self.assertTrue(found_clarification)

    def test_execute_interactive_mode_with_mock(self):
        agent = UserInteractionAgent(config={"mode": "interactive"})
        agent._infer_json = MagicMock(return_value={
            "is_ambiguous": True,
            "questions": ["What language?"],
        })
        # Mock _interactive_prompt to avoid actual stdin
        agent._interactive_prompt = MagicMock(return_value=["Python"])
        task = _ui_task()
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["responses_gathered"], 1)

    def test_execute_uses_goal_from_context(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        agent._infer_json = MagicMock(return_value={"is_ambiguous": False, "questions": []})
        task = _ui_task(goal="Specific goal from context")
        agent.execute(task)
        # Verify _infer_json was called with a prompt containing the goal
        call_args = agent._infer_json.call_args[0][0]
        self.assertIn("Specific goal from context", call_args)

    def test_execute_falls_back_to_task_description_for_goal(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        agent._infer_json = MagicMock(return_value={"is_ambiguous": False, "questions": []})
        task = _ui_task()
        task.context.pop("goal", None)
        agent.execute(task)
        # Should use task.description as goal fallback
        call_args = agent._infer_json.call_args[0][0]
        self.assertIn(task.description, call_args)

    def test_execute_empty_questions_not_ambiguous(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        agent._infer_json = MagicMock(return_value={
            "is_ambiguous": True,
            "questions": [],
        })
        task = _ui_task()
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertFalse(result.metadata.get("ambiguous", True))


class TestUserInteractionAgentVerify(unittest.TestCase):
    """Test the verify() method."""

    def setUp(self):
        self.agent = UserInteractionAgent()

    def test_verify_valid_dict(self):
        result = self.agent.verify({"key": "value"})
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.issues, [])

    def test_verify_empty_dict_passes(self):
        result = self.agent.verify({})
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)

    def test_verify_none_fails(self):
        result = self.agent.verify(None)
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_verify_string_fails(self):
        result = self.agent.verify("not a dict")
        self.assertFalse(result.passed)

    def test_verify_list_fails(self):
        result = self.agent.verify([1, 2, 3])
        self.assertFalse(result.passed)

    def test_verify_number_fails(self):
        result = self.agent.verify(42)
        self.assertFalse(result.passed)


class TestUserInteractionAgentCallbackPrompt(unittest.TestCase):
    """Test _callback_prompt()."""

    def setUp(self):
        self.agent = UserInteractionAgent()

    def test_callback_returns_list(self):
        self.agent._callback = MagicMock(return_value=["ans1", "ans2"])
        responses = self.agent._callback_prompt("goal", ["q1", "q2"])
        self.assertEqual(responses, ["ans1", "ans2"])

    def test_callback_returns_string_repeated(self):
        self.agent._callback = MagicMock(return_value="single answer")
        responses = self.agent._callback_prompt("goal", ["q1", "q2", "q3"])
        self.assertEqual(responses, ["single answer", "single answer", "single answer"])

    def test_callback_none_returns_no_callback(self):
        self.agent._callback = None
        responses = self.agent._callback_prompt("goal", ["q1", "q2"])
        self.assertEqual(responses, ["(no callback)", "(no callback)"])

    def test_callback_exception_returns_error(self):
        self.agent._callback = MagicMock(side_effect=RuntimeError("boom"))
        responses = self.agent._callback_prompt("goal", ["q1"])
        self.assertEqual(responses, ["(callback error)"])

    def test_callback_returns_unexpected_type(self):
        self.agent._callback = MagicMock(return_value=42)
        responses = self.agent._callback_prompt("goal", ["q1"])
        self.assertEqual(responses, ["(callback error)"])


class TestUserInteractionAgentInteractivePrompt(unittest.TestCase):
    """Test _interactive_prompt() with mocked stdin."""

    def test_interactive_prompt_collects_responses(self):
        agent = UserInteractionAgent()
        with patch("builtins.input", side_effect=["answer1", "answer2"]):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                responses = agent._interactive_prompt(["Q1?", "Q2?"])
        self.assertEqual(responses, ["answer1", "answer2"])

    def test_interactive_prompt_empty_response(self):
        agent = UserInteractionAgent()
        with patch("builtins.input", return_value=""):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                responses = agent._interactive_prompt(["Q1?"])
        self.assertEqual(responses, ["(no response)"])

    def test_interactive_prompt_eof_error(self):
        agent = UserInteractionAgent()
        with patch("builtins.input", side_effect=EOFError):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                responses = agent._interactive_prompt(["Q1?"])
        self.assertEqual(responses, ["(skipped)"])

    def test_interactive_prompt_keyboard_interrupt(self):
        agent = UserInteractionAgent()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                responses = agent._interactive_prompt(["Q1?"])
        self.assertEqual(responses, ["(skipped)"])

    def test_interactive_prompt_non_tty_uses_readline(self):
        agent = UserInteractionAgent()
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        mock_stdin.readline.return_value = "piped answer\n"
        with patch("sys.stdin", mock_stdin):
            # Use the real module's sys reference
            import vetinari.agents.user_interaction_agent as uia_mod
            orig_sys = uia_mod.sys
            uia_mod.sys = MagicMock()
            uia_mod.sys.stdin = mock_stdin
            try:
                responses = agent._interactive_prompt(["Q1?"])
            finally:
                uia_mod.sys = orig_sys
        self.assertEqual(responses, ["piped answer"])


class TestUserInteractionAgentAskForMore(unittest.TestCase):
    """Test ask_for_more_context()."""

    def test_non_interactive_returns_empty(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        result = agent.ask_for_more_context("goal", "need info")
        self.assertEqual(result, "")

    def test_callback_mode_uses_callback(self):
        cb = MagicMock(return_value=["the answer"])
        agent = UserInteractionAgent(config={"mode": "callback"})
        agent.set_interaction_mode("callback", callback=cb)
        result = agent.ask_for_more_context("goal", "need info")
        self.assertEqual(result, "the answer")

    def test_interactive_mode_with_mock_input(self):
        agent = UserInteractionAgent(config={"mode": "interactive"})
        with patch("builtins.input", return_value="user typed this"):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                result = agent.ask_for_more_context("goal", "need info")
        self.assertEqual(result, "user typed this")

    def test_interactive_mode_eof_returns_empty(self):
        agent = UserInteractionAgent(config={"mode": "interactive"})
        with patch("builtins.input", side_effect=EOFError):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.isatty.return_value = True
                result = agent.ask_for_more_context("goal", "need info")
        self.assertEqual(result, "")

    def test_long_goal_is_truncated_in_question(self):
        agent = UserInteractionAgent(config={"mode": "non_interactive"})
        long_goal = "A" * 200
        result = agent.ask_for_more_context(long_goal, "need info")
        # Just check it doesn't crash -- non_interactive returns ""
        self.assertEqual(result, "")


class TestUserInteractionAgentPendingQuestions(unittest.TestCase):
    """Test get_pending_questions() and answer_question()."""

    def setUp(self):
        self.agent = UserInteractionAgent()

    def test_no_pending_initially(self):
        self.assertEqual(self.agent.get_pending_questions(), [])

    def test_pending_after_non_interactive_execute(self):
        self.agent._mode = "non_interactive"
        self.agent._infer_json = MagicMock(return_value={
            "is_ambiguous": True,
            "questions": ["Q1?", "Q2?"],
        })
        task = _ui_task()
        self.agent.execute(task)
        pending = self.agent.get_pending_questions()
        self.assertEqual(len(pending), 2)

    def test_answer_question_marks_answered(self):
        self.agent._pending_questions = [
            {"question": "Q1?", "answered": False},
            {"question": "Q2?", "answered": False},
        ]
        self.agent.answer_question("Q1?", "Answer1")
        pending = self.agent.get_pending_questions()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["question"], "Q2?")

    def test_answer_question_updates_gathered_context(self):
        self.agent._pending_questions = [{"question": "What lang?", "answered": False}]
        self.agent.answer_question("What lang?", "Python")
        ctx = self.agent.get_gathered_context()
        found = False
        for v in ctx.values():
            if isinstance(v, dict) and v.get("answer") == "Python":
                found = True
        self.assertTrue(found)

    def test_answer_nonexistent_question_no_effect(self):
        self.agent._pending_questions = [{"question": "Q1?", "answered": False}]
        self.agent.answer_question("Q_nonexistent?", "Answer")
        pending = self.agent.get_pending_questions()
        self.assertEqual(len(pending), 1)

    def test_get_gathered_context_returns_copy(self):
        self.agent._gathered_context = {"key": "value"}
        ctx = self.agent.get_gathered_context()
        ctx["new_key"] = "modified"
        self.assertNotIn("new_key", self.agent._gathered_context)


class TestUserInteractionAgentSingleton(unittest.TestCase):
    """Test get_user_interaction_agent() singleton."""

    def test_singleton_returns_instance(self):
        import vetinari.agents.user_interaction_agent as uia_mod
        uia_mod._user_interaction_agent = None
        agent = get_user_interaction_agent()
        self.assertIsInstance(agent, UserInteractionAgent)

    def test_singleton_returns_same_instance(self):
        import vetinari.agents.user_interaction_agent as uia_mod
        uia_mod._user_interaction_agent = None
        a1 = get_user_interaction_agent()
        a2 = get_user_interaction_agent()
        self.assertIs(a1, a2)

    def test_singleton_with_config(self):
        import vetinari.agents.user_interaction_agent as uia_mod
        uia_mod._user_interaction_agent = None
        agent = get_user_interaction_agent(config={"mode": "callback"})
        self.assertEqual(agent._mode, "callback")
        uia_mod._user_interaction_agent = None  # cleanup


# ===========================================================================
# VersionControlAgent Tests
# ===========================================================================


class TestVersionControlAgentInit(unittest.TestCase):
    """Constructor and initialization tests."""

    def test_default_init(self):
        agent = VersionControlAgent()
        self.assertEqual(agent.agent_type, _AgentType.VERSION_CONTROL)
        self.assertEqual(agent._safe_git_timeout, 10)

    def test_init_with_git_timeout_config(self):
        agent = VersionControlAgent(config={"git_timeout": "30"})
        self.assertEqual(agent._safe_git_timeout, 30)

    def test_init_with_env_timeout(self):
        with patch.dict(os.environ, {"VETINARI_GIT_TIMEOUT": "20"}):
            agent = VersionControlAgent()
            self.assertEqual(agent._safe_git_timeout, 20)

    def test_config_timeout_takes_precedence_over_env(self):
        with patch.dict(os.environ, {"VETINARI_GIT_TIMEOUT": "20"}):
            agent = VersionControlAgent(config={"git_timeout": "45"})
            self.assertEqual(agent._safe_git_timeout, 45)

    def test_init_with_none_config(self):
        agent = VersionControlAgent(config=None)
        self.assertEqual(agent._safe_git_timeout, 10)


class TestVersionControlAgentProperties(unittest.TestCase):
    """Test agent type, capabilities, system prompt."""

    def setUp(self):
        self.agent = VersionControlAgent()

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, _AgentType.VERSION_CONTROL)

    def test_capabilities_list(self):
        caps = self.agent.get_capabilities()
        expected = [
            "branch_strategy",
            "commit_message_generation",
            "pr_description_creation",
            "merge_conflict_analysis",
            "code_review_coordination",
            "changelog_generation",
            "release_tagging",
            "git_workflow_guidance",
            "versioning_strategy",
        ]
        self.assertEqual(caps, expected)

    def test_system_prompt_non_empty(self):
        prompt = self.agent.get_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)

    def test_system_prompt_mentions_git(self):
        prompt = self.agent.get_system_prompt()
        self.assertIn("Git", prompt)

    def test_system_prompt_mentions_conventional_commits(self):
        prompt = self.agent.get_system_prompt()
        self.assertIn("Conventional Commits", prompt)

    def test_system_prompt_mentions_json(self):
        prompt = self.agent.get_system_prompt()
        self.assertIn("JSON", prompt)


class TestVersionControlAgentVerify(unittest.TestCase):
    """Test the verify() method."""

    def setUp(self):
        self.agent = VersionControlAgent()

    def test_verify_non_dict_fails(self):
        result = self.agent.verify("string")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_verify_none_fails(self):
        result = self.agent.verify(None)
        self.assertFalse(result.passed)

    def test_verify_empty_dict_has_warning(self):
        result = self.agent.verify({})
        # No actionable output -> score drops 0.5
        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.score, 0.5)

    def test_verify_with_recommendations_passes(self):
        result = self.agent.verify({"recommendations": ["Use semantic versioning"]})
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)

    def test_verify_with_branch_strategy_passes(self):
        result = self.agent.verify({"branch_strategy": {"workflow": "gitflow"}})
        self.assertTrue(result.passed)

    def test_verify_with_commit_messages_passes(self):
        result = self.agent.verify({"commit_messages": ["feat: add login"]})
        self.assertTrue(result.passed)

    def test_verify_with_pr_description_passes(self):
        result = self.agent.verify({"pr_description": {"title": "PR title"}})
        self.assertTrue(result.passed)

    def test_verify_with_changelog_passes(self):
        result = self.agent.verify({"changelog": "## v1.0\n### Added\n- Feature"})
        self.assertTrue(result.passed)

    def test_verify_list_fails(self):
        result = self.agent.verify([1, 2])
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)


class TestVersionControlAgentFallback(unittest.TestCase):
    """Test _fallback_vc_guidance()."""

    def setUp(self):
        self.agent = VersionControlAgent()

    def test_fallback_returns_dict(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIsInstance(fb, dict)

    def test_fallback_has_branch_strategy(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("branch_strategy", fb)
        self.assertIn("recommended_branches", fb["branch_strategy"])

    def test_fallback_has_commit_messages(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("commit_messages", fb)
        self.assertTrue(len(fb["commit_messages"]) > 0)

    def test_fallback_has_pr_description(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("pr_description", fb)
        self.assertIn("title", fb["pr_description"])

    def test_fallback_has_recommendations(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("recommendations", fb)
        self.assertTrue(len(fb["recommendations"]) > 0)

    def test_fallback_pr_title_uses_description(self):
        task = _vc_task()
        task.description = "Add user authentication"
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("Add user authentication", fb["pr_description"]["title"])

    def test_fallback_pr_title_empty_description(self):
        task = _vc_task()
        task.description = ""
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("changes", fb["pr_description"]["title"])

    def test_fallback_operation_from_context(self):
        task = _vc_task(operation="changelog_generation")
        fb = self.agent._fallback_vc_guidance(task)
        self.assertEqual(fb["operation"], "changelog_generation")

    def test_fallback_has_changelog(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("changelog", fb)

    def test_fallback_has_merge_strategy(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("merge_strategy", fb)

    def test_fallback_has_risks(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertIn("risks", fb)

    def test_fallback_conflicts_empty(self):
        task = _vc_task()
        fb = self.agent._fallback_vc_guidance(task)
        self.assertEqual(fb["conflicts"], [])


class TestVersionControlAgentGatherGitContext(unittest.TestCase):
    """Test _gather_git_context()."""

    def setUp(self):
        self.agent = VersionControlAgent()

    def test_no_git_repo(self):
        result = self.agent._gather_git_context("/nonexistent/path/xyz")
        self.assertIn("No git repository", result)

    @patch("subprocess.run")
    def test_git_context_with_branch(self, mock_run):
        """When git commands succeed, context includes branch."""
        # Create a temp dir with .git
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, ".git"))
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "main"
            mock_run.return_value = mock_result
            result = self.agent._gather_git_context(td)
            self.assertIn("main", result)

    @patch("subprocess.run")
    def test_git_context_handles_subprocess_failure(self, mock_run):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, ".git"))
            mock_run.side_effect = Exception("git not found")
            result = self.agent._gather_git_context(td)
            # Should handle gracefully
            self.assertIsInstance(result, str)

    def test_git_context_empty_path_defaults_dot(self):
        result = self.agent._gather_git_context("")
        self.assertIsInstance(result, str)


class TestVersionControlAgentExecute(unittest.TestCase):
    """Test execute() method."""

    def test_execute_uses_fallback_when_llm_fails(self):
        agent = VersionControlAgent()
        agent._infer_json = MagicMock(return_value=None)
        agent._search = MagicMock(return_value=[])
        agent._gather_git_context = MagicMock(return_value="No git repository.")
        task = _vc_task()
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("branch_strategy", result.output)

    def test_execute_uses_llm_response(self):
        agent = VersionControlAgent()
        llm_response = {
            "operation": "branch_strategy",
            "branch_strategy": {"workflow": "trunk-based"},
            "recommendations": ["Use short-lived branches"],
        }
        agent._infer_json = MagicMock(return_value=llm_response)
        agent._search = MagicMock(return_value=[])
        agent._gather_git_context = MagicMock(return_value="Current branch: main")
        task = _vc_task()
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.output["branch_strategy"]["workflow"], "trunk-based")

    def test_execute_wrong_agent_type_still_runs(self):
        # VersionControlAgent.execute() calls validate_task() but ignores
        # its return value, so a mismatched type still proceeds.
        agent = VersionControlAgent()
        agent._infer_json = MagicMock(return_value=None)
        agent._search = MagicMock(return_value=[])
        agent._gather_git_context = MagicMock(return_value="No git.")
        task = _AgentTask(
            task_id="wrong-001",
            agent_type=_AgentType.PLANNER,
            description="Wrong type",
            prompt="test",
            context={"operation": "general"},
        )
        result = agent.execute(task)
        # Doesn't fail -- validate_task return value is not checked
        self.assertTrue(result.success)

    def test_execute_metadata_includes_operation(self):
        agent = VersionControlAgent()
        agent._infer_json = MagicMock(return_value=None)
        agent._search = MagicMock(return_value=[])
        agent._gather_git_context = MagicMock(return_value="No git.")
        task = _vc_task(operation="changelog_generation")
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["operation"], "changelog_generation")

    def test_execute_metadata_includes_task_id(self):
        agent = VersionControlAgent()
        agent._infer_json = MagicMock(return_value=None)
        agent._search = MagicMock(return_value=[])
        agent._gather_git_context = MagicMock(return_value="No git.")
        task = _vc_task()
        result = agent.execute(task)
        self.assertEqual(result.metadata["task_id"], "vc-001")

    def test_execute_search_failure_gracefully_handled(self):
        agent = VersionControlAgent()
        agent._infer_json = MagicMock(return_value=None)
        agent._search = MagicMock(side_effect=Exception("search down"))
        agent._gather_git_context = MagicMock(return_value="No git.")
        task = _vc_task()
        result = agent.execute(task)
        # Should still succeed using fallback
        self.assertTrue(result.success)

    def test_execute_with_empty_context(self):
        agent = VersionControlAgent()
        agent._infer_json = MagicMock(return_value=None)
        agent._search = MagicMock(return_value=[])
        agent._gather_git_context = MagicMock(return_value="No git.")
        task = _vc_task()
        task.context = {}
        result = agent.execute(task)
        self.assertTrue(result.success)

    def test_execute_exception_in_perform_returns_failure(self):
        agent = VersionControlAgent()
        agent._perform_vc_operation = MagicMock(side_effect=RuntimeError("boom"))
        task = _vc_task()
        result = agent.execute(task)
        self.assertFalse(result.success)
        self.assertIn("boom", result.errors[0])


class TestVersionControlAgentSingleton(unittest.TestCase):
    """Test get_version_control_agent()."""

    def test_singleton_returns_instance(self):
        import vetinari.agents.version_control_agent as vc_mod
        vc_mod._version_control_agent = None
        agent = get_version_control_agent()
        self.assertIsInstance(agent, VersionControlAgent)

    def test_singleton_returns_same_instance(self):
        import vetinari.agents.version_control_agent as vc_mod
        vc_mod._version_control_agent = None
        a1 = get_version_control_agent()
        a2 = get_version_control_agent()
        self.assertIs(a1, a2)
        vc_mod._version_control_agent = None


# ===========================================================================
# SecurityAuditorAgent Tests
# ===========================================================================


class TestSecurityAuditorAgentInit(unittest.TestCase):
    """Constructor and initialization tests."""

    def test_default_init(self):
        agent = SecurityAuditorAgent()
        self.assertEqual(agent.agent_type, _AgentType.SECURITY_AUDITOR)

    def test_init_with_config(self):
        agent = SecurityAuditorAgent(config={"key": "value"})
        self.assertEqual(agent._config, {"key": "value"})

    def test_init_with_none_config(self):
        agent = SecurityAuditorAgent(config=None)
        self.assertEqual(agent._config, {})


class TestSecurityAuditorAgentProperties(unittest.TestCase):
    """Test agent type, capabilities, system prompt."""

    def setUp(self):
        self.agent = SecurityAuditorAgent()

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, _AgentType.SECURITY_AUDITOR)

    def test_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIn("vulnerability_scanning", caps)
        self.assertIn("policy_compliance_check", caps)
        self.assertIn("access_control_review", caps)
        self.assertIn("data_protection_analysis", caps)
        self.assertIn("encryption_verification", caps)
        self.assertIn("compliance_reporting", caps)
        self.assertEqual(len(caps), 6)

    def test_system_prompt_non_empty(self):
        prompt = self.agent.get_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)

    def test_system_prompt_mentions_security(self):
        prompt = self.agent.get_system_prompt()
        self.assertIn("security", prompt.lower())

    def test_system_prompt_mentions_cwe(self):
        prompt = self.agent.get_system_prompt()
        self.assertIn("CWE", prompt)

    def test_system_prompt_mentions_verdict(self):
        prompt = self.agent.get_system_prompt()
        self.assertIn("verdict", prompt)


class TestSecurityAuditorHeuristicChecks(unittest.TestCase):
    """Test _run_heuristic_checks() with various code patterns."""

    def setUp(self):
        self.agent = SecurityAuditorAgent()

    def test_eval_detected(self):
        code = "result = eval(user_input)"
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("eval()" in t for t in titles))

    def test_exec_detected(self):
        code = "exec(untrusted_code)"
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("exec()" in t for t in titles))

    def test_pickle_detected(self):
        code = "data = pickle.loads(payload)"
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("pickle" in t.lower() for t in titles))

    def test_os_system_detected(self):
        code = "os.system('rm -rf /')"
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("command" in t.lower() or "os" in t.lower() for t in titles))

    def test_hardcoded_password_detected(self):
        code = 'password = "mysecretpassword123"'
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("password" in t.lower() for t in titles))

    def test_hardcoded_api_key_detected(self):
        code = 'api_key = "sk-1234567890abcdef"'
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("api key" in t.lower() for t in titles))

    def test_hardcoded_secret_detected(self):
        code = 'secret = "abcdefghij"'
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("secret" in t.lower() for t in titles))

    def test_sql_injection_string_format(self):
        code = 'sql = "SELECT * FROM users WHERE id = %s" % user_id'
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("sql" in t.lower() for t in titles))

    def test_subprocess_shell_true(self):
        code = 'subprocess.call("ls", shell=True)'
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("shell" in t.lower() or "subprocess" in t.lower() for t in titles))

    def test_debug_mode_detected(self):
        code = "DEBUG = True"
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("debug" in t.lower() for t in titles))

    def test_ssl_verify_false_detected(self):
        code = 'requests.get(url, verify=False)'
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("ssl" in t.lower() or "verify" in t.lower() for t in titles))

    def test_md5_detected(self):
        code = "hash_val = hashlib.md5(data)"
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("md5" in t.lower() for t in titles))

    def test_insecure_random_detected(self):
        code = "token = random.random()"
        issues = self.agent._run_heuristic_checks(code)
        titles = [i["title"] for i in issues]
        self.assertTrue(any("random" in t.lower() for t in titles))

    def test_clean_code_no_issues(self):
        code = """
import secrets
import hashlib

def get_token():
    return secrets.token_hex(32)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()
"""
        issues = self.agent._run_heuristic_checks(code)
        # Should have no heuristic issues (AST may flag import of hashlib)
        heuristic_only = [i for i in issues if i["id"].startswith("HEUR-")]
        self.assertEqual(len(heuristic_only), 0)

    def test_multiple_issues_in_code(self):
        code = """
password = "hardcoded123"
eval(user_input)
os.system(cmd)
DEBUG = True
"""
        issues = self.agent._run_heuristic_checks(code)
        self.assertTrue(len(issues) >= 3)

    def test_max_three_matches_per_pattern(self):
        code = "\n".join([f"eval(x{i})" for i in range(10)])
        issues = self.agent._run_heuristic_checks(code)
        heur_eval = [i for i in issues if i["id"].startswith("HEUR-") and "eval" in i["title"]]
        self.assertLessEqual(len(heur_eval), 3)

    def test_issue_structure(self):
        code = "eval(x)"
        issues = self.agent._run_heuristic_checks(code)
        self.assertTrue(len(issues) > 0)
        issue = issues[0]
        self.assertIn("id", issue)
        self.assertIn("severity", issue)
        self.assertIn("category", issue)
        self.assertIn("title", issue)
        self.assertIn("description", issue)
        self.assertIn("location", issue)
        self.assertIn("evidence", issue)
        self.assertIn("remediation", issue)


class TestSecurityAuditorASTScan(unittest.TestCase):
    """Test _ast_security_scan()."""

    def setUp(self):
        self.agent = SecurityAuditorAgent()

    def test_ast_detects_eval_call(self):
        code = "eval(user_input)"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("eval_call", patterns)

    def test_ast_detects_exec_call(self):
        code = "exec(code_str)"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("exec_call", patterns)

    def test_ast_detects_dynamic_import(self):
        code = "__import__('os')"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("dynamic_import", patterns)

    def test_ast_detects_subprocess_call(self):
        code = "subprocess.call(['ls'])"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("subprocess_call", patterns)

    def test_ast_detects_subprocess_popen(self):
        code = "subprocess.Popen(['ls'])"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("subprocess_call", patterns)

    def test_ast_detects_subprocess_run(self):
        code = "subprocess.run(['ls'])"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("subprocess_call", patterns)

    def test_ast_detects_os_system(self):
        code = "os.system('cmd')"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("os_system_call", patterns)

    def test_ast_detects_os_popen(self):
        code = "os.popen('cmd')"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("os_system_call", patterns)

    def test_ast_detects_pickle_loads(self):
        code = "pickle.loads(data)"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("pickle_deserialise", patterns)

    def test_ast_detects_pickle_load(self):
        code = "pickle.load(f)"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("pickle_deserialise", patterns)

    def test_ast_detects_dangerous_import_os(self):
        code = "import os"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("os" in f["detail"] for f in findings))

    def test_ast_detects_dangerous_import_subprocess(self):
        code = "import subprocess"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("subprocess" in f["detail"] for f in findings))

    def test_ast_detects_dangerous_from_import(self):
        code = "from os.path import join"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("os" in f["detail"] for f in findings))

    def test_ast_detects_dunder_class(self):
        code = "x.__class__"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("__class__" in f["detail"] for f in findings))

    def test_ast_detects_dunder_bases(self):
        code = "x.__bases__"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("__bases__" in f["detail"] for f in findings))

    def test_ast_detects_dunder_subclasses(self):
        code = "x.__subclasses__()"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("__subclasses__" in f["detail"] for f in findings))

    def test_ast_detects_dunder_globals(self):
        code = "x.__globals__"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("__globals__" in f["detail"] for f in findings))

    def test_ast_detects_dunder_builtins(self):
        code = "x.__builtins__"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("__builtins__" in f["detail"] for f in findings))

    def test_ast_safe_code_no_findings(self):
        code = """
x = 1 + 2
y = [a for a in range(10)]
print(x + y[0])
"""
        findings = self.agent._ast_security_scan(code)
        self.assertEqual(len(findings), 0)

    def test_ast_syntax_error_returns_empty(self):
        code = "def broken("
        findings = self.agent._ast_security_scan(code)
        self.assertEqual(findings, [])

    def test_ast_non_python_returns_empty(self):
        code = "<html><body>Hello</body></html>"
        findings = self.agent._ast_security_scan(code)
        self.assertEqual(findings, [])

    def test_ast_finding_structure(self):
        code = "eval(x)"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(len(findings) > 0)
        f = findings[0]
        self.assertIn("pattern", f)
        self.assertIn("severity", f)
        self.assertIn("line", f)
        self.assertIn("detail", f)
        self.assertIn("evidence", f)

    def test_ast_import_shutil(self):
        code = "import shutil"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("shutil" in f["detail"] for f in findings))

    def test_ast_import_ctypes(self):
        code = "import ctypes"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("ctypes" in f["detail"] for f in findings))

    def test_ast_import_importlib(self):
        code = "import importlib"
        findings = self.agent._ast_security_scan(code)
        self.assertTrue(any("importlib" in f["detail"] for f in findings))

    def test_ast_subprocess_check_output(self):
        code = "subprocess.check_output(['ls'])"
        findings = self.agent._ast_security_scan(code)
        patterns = [f["pattern"] for f in findings]
        self.assertIn("subprocess_call", patterns)


class TestSecurityAuditorComputeVerdict(unittest.TestCase):
    """Test _compute_verdict()."""

    def setUp(self):
        self.agent = SecurityAuditorAgent()

    def test_no_issues_pass(self):
        verdict = self.agent._compute_verdict([])
        self.assertEqual(verdict, "pass")

    def test_critical_issue_fail(self):
        verdict = self.agent._compute_verdict([{"severity": "critical"}])
        self.assertEqual(verdict, "fail")

    def test_high_issue_conditional(self):
        verdict = self.agent._compute_verdict([{"severity": "high"}])
        self.assertEqual(verdict, "conditional")

    def test_many_low_issues_conditional(self):
        issues = [{"severity": "low"} for _ in range(6)]
        verdict = self.agent._compute_verdict(issues)
        self.assertEqual(verdict, "conditional")

    def test_few_low_issues_pass(self):
        issues = [{"severity": "low"} for _ in range(3)]
        verdict = self.agent._compute_verdict(issues)
        self.assertEqual(verdict, "pass")

    def test_medium_issues_pass(self):
        issues = [{"severity": "medium"}]
        verdict = self.agent._compute_verdict(issues)
        self.assertEqual(verdict, "pass")

    def test_info_issues_pass(self):
        issues = [{"severity": "info"} for _ in range(10)]
        verdict = self.agent._compute_verdict(issues)
        # More than 5 issues -> conditional
        self.assertEqual(verdict, "conditional")

    def test_mixed_severities_worst_wins(self):
        issues = [
            {"severity": "low"},
            {"severity": "medium"},
            {"severity": "critical"},
        ]
        verdict = self.agent._compute_verdict(issues)
        self.assertEqual(verdict, "fail")


class TestSecurityAuditorComputeScore(unittest.TestCase):
    """Test _compute_score()."""

    def setUp(self):
        self.agent = SecurityAuditorAgent()

    def test_no_issues_score_100(self):
        score = self.agent._compute_score([])
        self.assertEqual(score, 100)

    def test_critical_issue_reduces_30(self):
        score = self.agent._compute_score([{"severity": "critical"}])
        self.assertEqual(score, 70)

    def test_high_issue_reduces_15(self):
        score = self.agent._compute_score([{"severity": "high"}])
        self.assertEqual(score, 85)

    def test_medium_issue_reduces_8(self):
        score = self.agent._compute_score([{"severity": "medium"}])
        self.assertEqual(score, 92)

    def test_low_issue_reduces_3(self):
        score = self.agent._compute_score([{"severity": "low"}])
        self.assertEqual(score, 97)

    def test_info_issue_no_reduction(self):
        score = self.agent._compute_score([{"severity": "info"}])
        self.assertEqual(score, 100)

    def test_multiple_issues_cumulative(self):
        issues = [
            {"severity": "critical"},  # -30
            {"severity": "high"},      # -15
            {"severity": "medium"},    # -8
        ]
        score = self.agent._compute_score(issues)
        self.assertEqual(score, 47)

    def test_score_never_below_zero(self):
        issues = [{"severity": "critical"} for _ in range(10)]
        score = self.agent._compute_score(issues)
        self.assertEqual(score, 0)

    def test_unknown_severity_no_reduction(self):
        score = self.agent._compute_score([{"severity": "unknown"}])
        self.assertEqual(score, 100)

    def test_missing_severity_no_reduction(self):
        score = self.agent._compute_score([{}])
        self.assertEqual(score, 100)


class TestSecurityAuditorCollectCodeContent(unittest.TestCase):
    """Test _collect_code_content()."""

    def setUp(self):
        self.agent = SecurityAuditorAgent()

    def test_empty_inputs(self):
        result = self.agent._collect_code_content([], {}, "")
        self.assertEqual(result, "")

    def test_description_included(self):
        result = self.agent._collect_code_content([], {}, "Audit this code")
        self.assertIn("Audit this code", result)

    def test_string_outputs_included(self):
        result = self.agent._collect_code_content(["code line 1", "code line 2"], {}, "")
        self.assertIn("code line 1", result)
        self.assertIn("code line 2", result)

    def test_dict_outputs_with_long_strings(self):
        outputs = [{"source": "x" * 50}]
        result = self.agent._collect_code_content(outputs, {}, "")
        self.assertIn("source", result)

    def test_dict_outputs_short_strings_excluded(self):
        outputs = [{"key": "short"}]
        result = self.agent._collect_code_content(outputs, {}, "")
        self.assertNotIn("short", result)

    def test_artifacts_included(self):
        artifacts = {"main.py": "x" * 50}
        result = self.agent._collect_code_content([], artifacts, "")
        self.assertIn("main.py", result)

    def test_artifacts_short_strings_excluded(self):
        artifacts = {"key": "short"}
        result = self.agent._collect_code_content([], artifacts, "")
        self.assertNotIn("short", result)

    def test_artifacts_truncated_at_1000(self):
        artifacts = {"big.py": "x" * 5000}
        result = self.agent._collect_code_content([], artifacts, "")
        # The artifact content should be truncated
        self.assertLessEqual(len(result), 1100)  # some overhead for header


class TestSecurityAuditorFallbackAudit(unittest.TestCase):
    """Test _fallback_audit()."""

    def setUp(self):
        self.agent = SecurityAuditorAgent()

    def test_fallback_structure(self):
        fb = self.agent._fallback_audit([], "standard")
        self.assertIn("verdict", fb)
        self.assertIn("security_score", fb)
        self.assertIn("security_issues", fb)
        self.assertIn("recommendations", fb)
        self.assertIn("remediation_steps", fb)
        self.assertIn("compliance_notes", fb)
        self.assertIn("positive_findings", fb)
        self.assertIn("summary", fb)

    def test_fallback_no_issues_pass(self):
        fb = self.agent._fallback_audit([], "standard")
        self.assertEqual(fb["verdict"], "pass")
        self.assertEqual(fb["security_score"], 100)

    def test_fallback_with_critical_issue(self):
        issues = [{"severity": "critical", "title": "eval detected"}]
        fb = self.agent._fallback_audit(issues, "strict")
        self.assertEqual(fb["verdict"], "fail")

    def test_fallback_recommendations_present(self):
        fb = self.agent._fallback_audit([], "standard")
        self.assertTrue(len(fb["recommendations"]) > 0)

    def test_fallback_remediation_steps_present(self):
        fb = self.agent._fallback_audit([], "standard")
        self.assertTrue(len(fb["remediation_steps"]) > 0)

    def test_fallback_compliance_notes_present(self):
        fb = self.agent._fallback_audit([], "standard")
        self.assertIn("owasp_top10", fb["compliance_notes"])

    def test_fallback_summary_contains_verdict(self):
        fb = self.agent._fallback_audit([], "standard")
        self.assertIn("PASS", fb["summary"])


class TestSecurityAuditorVerify(unittest.TestCase):
    """Test the verify() method."""

    def setUp(self):
        self.agent = SecurityAuditorAgent()

    def test_verify_non_dict_fails(self):
        result = self.agent.verify("string")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_verify_none_fails(self):
        result = self.agent.verify(None)
        self.assertFalse(result.passed)

    def test_verify_complete_output_passes(self):
        output = {
            "verdict": "pass",
            "security_issues": [],
            "recommendations": ["Use HTTPS"],
            "remediation_steps": ["Enable TLS"],
        }
        result = self.agent.verify(output)
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)

    def test_verify_invalid_verdict(self):
        output = {
            "verdict": "maybe",
            "security_issues": [],
            "recommendations": ["a"],
            "remediation_steps": ["b"],
        }
        result = self.agent.verify(output)
        # Score drops by 0.3
        self.assertAlmostEqual(result.score, 0.7)

    def test_verify_missing_security_issues(self):
        output = {
            "verdict": "pass",
            "recommendations": ["a"],
            "remediation_steps": ["b"],
        }
        result = self.agent.verify(output)
        # Score drops by 0.2
        self.assertAlmostEqual(result.score, 0.8)

    def test_verify_missing_recommendations(self):
        output = {
            "verdict": "pass",
            "security_issues": [],
            "recommendations": [],
            "remediation_steps": ["b"],
        }
        result = self.agent.verify(output)
        # Score drops by 0.15
        self.assertAlmostEqual(result.score, 0.85)

    def test_verify_missing_remediation(self):
        output = {
            "verdict": "pass",
            "security_issues": [],
            "recommendations": ["a"],
            "remediation_steps": [],
        }
        result = self.agent.verify(output)
        self.assertAlmostEqual(result.score, 0.85)

    def test_verify_all_missing_still_passes_if_score_above_half(self):
        output = {
            "verdict": "pass",
            "security_issues": [],
            "recommendations": [],
            "remediation_steps": [],
        }
        result = self.agent.verify(output)
        # 1.0 - 0.15 - 0.15 = 0.7 -> passes
        self.assertTrue(result.passed)

    def test_verify_everything_missing_fails(self):
        output = {
            "verdict": "unknown",
            # missing security_issues, recommendations, remediation_steps
        }
        result = self.agent.verify(output)
        # 1.0 - 0.3 - 0.2 - 0.15 - 0.15 = 0.2 -> fails
        self.assertFalse(result.passed)

    def test_verify_conditional_verdict_passes(self):
        output = {
            "verdict": "conditional",
            "security_issues": [{"severity": "medium"}],
            "recommendations": ["Fix it"],
            "remediation_steps": ["Step 1"],
        }
        result = self.agent.verify(output)
        self.assertTrue(result.passed)

    def test_verify_fail_verdict_passes_verification(self):
        # verify() checks structure, not semantic pass/fail
        output = {
            "verdict": "fail",
            "security_issues": [{"severity": "critical"}],
            "recommendations": ["Fix it"],
            "remediation_steps": ["Step 1"],
        }
        result = self.agent.verify(output)
        self.assertTrue(result.passed)


class TestSecurityAuditorExecute(unittest.TestCase):
    """Test execute() method."""

    def test_execute_wrong_agent_type_fails(self):
        agent = SecurityAuditorAgent()
        task = _AgentTask(
            task_id="wrong-001",
            agent_type=_AgentType.PLANNER,
            description="Wrong type",
            prompt="test",
            context={"outputs": [], "artifacts": {}},
        )
        result = agent.execute(task)
        self.assertFalse(result.success)

    def test_execute_with_clean_code(self):
        agent = SecurityAuditorAgent()
        # side_effect returns fallback kwarg so fallback audit is used
        agent._infer_json = MagicMock(
            side_effect=lambda prompt, fallback=None, **kw: fallback
        )
        agent._search = MagicMock(return_value=[])
        task = _sec_task(outputs=["x = 1 + 2\nprint(x)"], artifacts={})
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("verdict", result.output)

    def test_execute_with_vulnerable_code(self):
        agent = SecurityAuditorAgent()
        agent._infer_json = MagicMock(
            side_effect=lambda prompt, fallback=None, **kw: fallback
        )
        agent._search = MagicMock(return_value=[])
        # Test code patterns that the security auditor should detect
        task = _sec_task(
            outputs=["result = eval(user_input)\nos.system(cmd)"],
            artifacts={},
        )
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertTrue(len(result.output.get("security_issues", [])) > 0)

    def test_execute_uses_llm_result(self):
        agent = SecurityAuditorAgent()
        llm_result = {
            "verdict": "conditional",
            "security_score": 65,
            "security_issues": [
                {"severity": "high", "title": "SQL Injection found"}
            ],
            "recommendations": ["Use parameterized queries"],
            "remediation_steps": ["Fix SQL"],
            "compliance_notes": {},
            "positive_findings": ["Uses HTTPS"],
            "summary": "Moderate risk",
        }
        agent._infer_json = MagicMock(return_value=llm_result)
        agent._search = MagicMock(return_value=[])
        task = _sec_task(outputs=["SELECT * FROM users WHERE id = " + "user_id"])
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.output["verdict"], "conditional")

    def test_execute_merges_heuristic_with_llm(self):
        agent = SecurityAuditorAgent()
        llm_result = {
            "verdict": "conditional",
            "security_score": 80,
            "security_issues": [
                {"title": "LLM Found Issue", "severity": "medium"}
            ],
        }
        agent._infer_json = MagicMock(return_value=llm_result)
        agent._search = MagicMock(return_value=[])
        task = _sec_task(outputs=["eval(x)"])
        result = agent.execute(task)
        self.assertTrue(result.success)
        titles = [i.get("title", "") for i in result.output.get("security_issues", [])]
        # Should contain both LLM issue and heuristic eval issue
        self.assertTrue(any("LLM Found" in t for t in titles))

    def test_execute_metadata_fields(self):
        agent = SecurityAuditorAgent()
        agent._infer_json = MagicMock(
            side_effect=lambda prompt, fallback=None, **kw: fallback
        )
        agent._search = MagicMock(return_value=[])
        task = _sec_task(outputs=["safe = True"])
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("policy_level", result.metadata)
        self.assertIn("verdict", result.metadata)
        self.assertIn("issues_found", result.metadata)
        self.assertIn("security_score", result.metadata)
        self.assertIn("artifacts_audited", result.metadata)

    def test_execute_exception_returns_failure(self):
        agent = SecurityAuditorAgent()
        agent._collect_code_content = MagicMock(side_effect=RuntimeError("parse error"))
        task = _sec_task()
        result = agent.execute(task)
        self.assertFalse(result.success)
        self.assertIn("parse error", result.errors[0])

    def test_execute_ensures_required_keys(self):
        agent = SecurityAuditorAgent()
        # LLM returns partial result missing some keys
        agent._infer_json = MagicMock(return_value={"security_issues": []})
        agent._search = MagicMock(return_value=[])
        task = _sec_task(outputs=["safe code"])
        result = agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("verdict", result.output)
        self.assertIn("security_score", result.output)
        self.assertIn("recommendations", result.output)
        self.assertIn("remediation_steps", result.output)
        self.assertIn("compliance_notes", result.output)
        self.assertIn("positive_findings", result.output)
        self.assertIn("summary", result.output)

    def test_execute_with_empty_outputs(self):
        agent = SecurityAuditorAgent()
        agent._infer_json = MagicMock(
            side_effect=lambda prompt, fallback=None, **kw: fallback
        )
        agent._search = MagicMock(return_value=[])
        task = _sec_task(outputs=[])
        result = agent.execute(task)
        self.assertTrue(result.success)

    def test_execute_search_failure_handled(self):
        agent = SecurityAuditorAgent()
        agent._infer_json = MagicMock(
            side_effect=lambda prompt, fallback=None, **kw: fallback
        )
        agent._search = MagicMock(side_effect=Exception("search down"))
        task = _sec_task(outputs=["x = 1"])
        result = agent.execute(task)
        self.assertTrue(result.success)

    def test_execute_policy_level_in_metadata(self):
        agent = SecurityAuditorAgent()
        agent._infer_json = MagicMock(
            side_effect=lambda prompt, fallback=None, **kw: fallback
        )
        agent._search = MagicMock(return_value=[])
        task = _sec_task(policy_level="strict")
        result = agent.execute(task)
        self.assertEqual(result.metadata["policy_level"], "strict")

    def test_execute_code_truncation_long_content(self):
        agent = SecurityAuditorAgent()
        # Use side_effect that returns the fallback argument
        agent._infer_json = MagicMock(side_effect=lambda prompt, fallback=None, **kw: fallback)
        agent._search = MagicMock(return_value=[])
        long_code = "x = 1\n" * 5000
        task = _sec_task(outputs=[long_code])
        result = agent.execute(task)
        self.assertTrue(result.success)


class TestSecurityAuditorSingleton(unittest.TestCase):
    """Test get_security_auditor_agent()."""

    def test_singleton_returns_instance(self):
        import vetinari.agents.security_auditor_agent as sec_mod
        sec_mod._security_auditor_agent = None
        agent = get_security_auditor_agent()
        self.assertIsInstance(agent, SecurityAuditorAgent)

    def test_singleton_returns_same_instance(self):
        import vetinari.agents.security_auditor_agent as sec_mod
        sec_mod._security_auditor_agent = None
        a1 = get_security_auditor_agent()
        a2 = get_security_auditor_agent()
        self.assertIs(a1, a2)
        sec_mod._security_auditor_agent = None


# ===========================================================================
# Cross-cutting tests
# ===========================================================================


class TestAllAgentsMetadata(unittest.TestCase):
    """Ensure all three agents expose consistent metadata."""

    def test_user_interaction_metadata(self):
        agent = UserInteractionAgent()
        meta = agent.get_metadata()
        self.assertEqual(meta["agent_type"], "USER_INTERACTION")
        self.assertTrue(len(meta["capabilities"]) > 0)

    def test_version_control_metadata(self):
        agent = VersionControlAgent()
        meta = agent.get_metadata()
        self.assertEqual(meta["agent_type"], "VERSION_CONTROL")
        self.assertTrue(len(meta["capabilities"]) > 0)

    def test_security_auditor_metadata(self):
        agent = SecurityAuditorAgent()
        meta = agent.get_metadata()
        self.assertEqual(meta["agent_type"], "SECURITY_AUDITOR")
        self.assertTrue(len(meta["capabilities"]) > 0)


class TestAllAgentsSystemPrompt(unittest.TestCase):
    """Ensure all three agents have substantive system prompts."""

    def test_prompts_are_strings(self):
        for cls in [UserInteractionAgent, VersionControlAgent, SecurityAuditorAgent]:
            agent = cls()
            prompt = agent.get_system_prompt()
            self.assertIsInstance(prompt, str, f"{cls.__name__} prompt not a string")

    def test_prompts_are_non_trivial(self):
        for cls in [UserInteractionAgent, VersionControlAgent, SecurityAuditorAgent]:
            agent = cls()
            prompt = agent.get_system_prompt()
            self.assertGreater(len(prompt), 50, f"{cls.__name__} prompt too short")


class TestAllAgentsVerifyEdgeCases(unittest.TestCase):
    """Ensure verify() handles weird inputs consistently."""

    def _agents(self):
        return [UserInteractionAgent(), VersionControlAgent(), SecurityAuditorAgent()]

    def test_verify_none(self):
        for agent in self._agents():
            result = agent.verify(None)
            self.assertFalse(result.passed, f"{agent.__class__.__name__} should fail on None")

    def test_verify_number(self):
        for agent in self._agents():
            result = agent.verify(42)
            self.assertFalse(result.passed, f"{agent.__class__.__name__} should fail on int")

    def test_verify_bool(self):
        for agent in self._agents():
            result = agent.verify(True)
            self.assertFalse(result.passed, f"{agent.__class__.__name__} should fail on bool")

    def test_verify_list(self):
        for agent in self._agents():
            result = agent.verify([1, 2, 3])
            self.assertFalse(result.passed, f"{agent.__class__.__name__} should fail on list")


if __name__ == "__main__":
    unittest.main()
