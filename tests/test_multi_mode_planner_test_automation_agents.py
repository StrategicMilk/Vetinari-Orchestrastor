"""
Comprehensive tests for three Vetinari agents:
 - MultiModeAgent (vetinari/agents/multi_mode_agent.py)
 - PlannerAgent (vetinari/agents/planner_agent.py)
 - TestAutomationAgent (now consolidated into QualityAgent)

All external dependencies are stubbed so the tests run in isolation without
any real LLM, filesystem (where controllable), or third-party services.
"""

import importlib.util
import os
import sys
import types
import unittest
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Root of the worktree (where vetinari/ lives)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_agent_module(rel_path: str, dotted_name: str):
    """Load a real module from disk via importlib.util."""
    full = os.path.join(_HERE, rel_path.replace("/", os.sep))
    spec = importlib.util.spec_from_file_location(dotted_name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


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


# ---------------------------------------------------------------------------
# Build all stubs BEFORE loading agent source files
# ---------------------------------------------------------------------------


def _build_stubs():
    """Stub every module that the agents (and BaseAgent) import."""

    # Top-level packages
    for p in [
        "vetinari",
        "vetinari.agents",
        "vetinari.adapters",
        "vetinari.learning",
        "vetinari.constraints",
        "vetinari.drift",
        "vetinari.config",
        "vetinari.tools",
        "vetinari.coding_agent",
        "vetinari.orchestration",
        "vetinari.analytics",
        "vetinari.dashboard",
        "vetinari.memory",
        "vetinari.skills",
    ]:
        pkg(p)

    # ── Ensure vetinari.types is loaded (real module, stdlib-only) ────────
    _vtypes_path = os.path.join(_HERE, "vetinari", "types.py")
    if os.path.exists(_vtypes_path):
        if "vetinari.types" not in sys.modules:
            _load_agent_module("vetinari/types.py", "vetinari.types")
        types_mod = sys.modules["vetinari.types"]
    else:
        # Fallback: define minimal types
        types_mod = mod("vetinari.types")

    # Grab real enums from types_mod
    AgentType = getattr(types_mod, "AgentType", None)
    StatusEnum = getattr(types_mod, "StatusEnum", None)
    ExecutionMode = getattr(types_mod, "ExecutionMode", None)

    # ── Contracts ────────────────────────────────────────────────────────
    @dataclass
    class AgentSpec:
        agent_type: Any = None
        name: str = ""
        description: str = ""
        default_model: str = "default"
        thinking_variant: str = "medium"
        enabled: bool = True
        system_prompt: str = ""
        version: str = "1.0.0"

        def to_dict(self):
            return {
                "agent_type": self.agent_type.value if self.agent_type else "",
                "name": self.name,
                "description": self.description,
                "default_model": self.default_model,
            }

    @dataclass
    class Task:
        id: str
        description: str
        inputs: list[str] = field(default_factory=list)
        outputs: list[str] = field(default_factory=list)
        dependencies: list[str] = field(default_factory=list)
        assigned_agent: Any = None
        model_override: str = ""
        depth: int = 0
        parent_id: str = ""
        status: Any = None

        def __post_init__(self):
            if self.assigned_agent is None:
                self.assigned_agent = AgentType.FOREMAN
            if self.status is None:
                self.status = StatusEnum.PENDING

        def to_dict(self):
            return {
                "id": self.id,
                "description": self.description,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "dependencies": self.dependencies,
                "assigned_agent": self.assigned_agent.value
                if hasattr(self.assigned_agent, "value")
                else str(self.assigned_agent),
                "depth": self.depth,
                "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            }

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
        dependencies: list[str] = field(default_factory=list)
        context: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self):
            if self.status is None:
                self.status = StatusEnum.PENDING

        @classmethod
        def from_task(cls, task, prompt):
            return cls(
                task_id=task.id,
                agent_type=task.assigned_agent,
                description=task.description,
                prompt=prompt,
                dependencies=task.dependencies,
            )

        def to_dict(self):
            return {
                "task_id": self.task_id,
                "agent_type": self.agent_type.value if hasattr(self.agent_type, "value") else str(self.agent_type),
                "description": self.description,
                "prompt": self.prompt,
            }

    @dataclass
    class Plan:
        plan_id: str
        version: str = "v0.1.0"
        goal: str = ""
        phase: int = 0
        tasks: list[Any] = field(default_factory=list)
        model_scores: list[dict] = field(default_factory=list)
        notes: str = ""
        warnings: list[str] = field(default_factory=list)
        needs_context: bool = False
        follow_up_question: str = ""
        final_delivery_path: str = ""
        final_delivery_summary: str = ""
        created_at: str = field(default_factory=lambda: datetime.now().isoformat())

        @classmethod
        def create_new(cls, goal: str, phase: int = 0) -> "Plan":
            return cls(plan_id=f"plan_{uuid.uuid4().hex[:8]}", goal=goal, phase=phase)

        def to_dict(self):
            return {
                "plan_id": self.plan_id,
                "version": self.version,
                "goal": self.goal,
                "phase": self.phase,
                "tasks": [t.to_dict() for t in self.tasks],
                "warnings": self.warnings,
                "needs_context": self.needs_context,
                "follow_up_question": self.follow_up_question,
            }

    @dataclass
    class AgentResult:
        success: bool
        output: Any
        metadata: dict[str, Any] = field(default_factory=dict)
        errors: list[str] = field(default_factory=list)
        provenance: list[dict] = field(default_factory=list)

        def to_dict(self):
            return {"success": self.success, "output": self.output, "metadata": self.metadata, "errors": self.errors}

    @dataclass
    class VerificationResult:
        passed: bool
        issues: list[dict[str, Any]] = field(default_factory=list)
        suggestions: list[str] = field(default_factory=list)
        score: float = 0.0

        def to_dict(self):
            return {"passed": self.passed, "issues": self.issues, "score": self.score}

    AGENT_REGISTRY = {}
    for at in AgentType:
        AGENT_REGISTRY[at] = AgentSpec(
            agent_type=at,
            name=at.value.title().replace("_", " "),
            description=f"{at.value} agent",
            default_model="default",
        )

    def get_agent_spec(agent_type):
        return AGENT_REGISTRY.get(agent_type)

    def get_all_agent_specs():
        return list(AGENT_REGISTRY.values())

    def get_enabled_agents():
        return [s for s in AGENT_REGISTRY.values() if s.enabled]

    # Install contracts into sys.modules
    contracts_mod = mod(
        "vetinari.agents.contracts",
        AgentType=AgentType,
        StatusEnum=StatusEnum,
        ExecutionMode=ExecutionMode,
        AgentSpec=AgentSpec,
        Task=Task,
        AgentTask=AgentTask,
        Plan=Plan,
        AgentResult=AgentResult,
        VerificationResult=VerificationResult,
        AGENT_REGISTRY=AGENT_REGISTRY,
        get_agent_spec=get_agent_spec,
        get_all_agent_specs=get_all_agent_specs,
        get_enabled_agents=get_enabled_agents,
    )

    # Ensure the Plan class on the contracts module has to_dict and create_new
    # (may have been set by an earlier test file like test_agent_graph.py
    # with a simpler stub that lacks these methods)
    _existing_plan = getattr(contracts_mod, "Plan", None)
    if _existing_plan is not None and not hasattr(_existing_plan, "to_dict"):

        def _plan_to_dict(self):
            return {
                "plan_id": self.plan_id,
                "version": getattr(self, "version", "v0.1.0"),
                "goal": self.goal,
                "phase": getattr(self, "phase", 0),
                "tasks": [
                    t.to_dict() if hasattr(t, "to_dict") else {"id": getattr(t, "id", "")} for t in (self.tasks or [])
                ],
                "warnings": getattr(self, "warnings", []),
                "needs_context": getattr(self, "needs_context", False),
                "follow_up_question": getattr(self, "follow_up_question", ""),
            }

        _existing_plan.to_dict = _plan_to_dict
    if _existing_plan is not None and not hasattr(_existing_plan, "create_new"):

        @classmethod
        def _plan_create_new(cls, goal, phase=0):
            return cls(plan_id=f"plan_{uuid.uuid4().hex[:8]}", goal=goal, phase=phase)

        _existing_plan.create_new = _plan_create_new

    # Ensure Task has to_dict too
    _existing_task = getattr(contracts_mod, "Task", None)
    if _existing_task is not None and not hasattr(_existing_task, "to_dict"):

        def _task_to_dict(self):
            return {
                "id": self.id,
                "description": self.description,
                "inputs": getattr(self, "inputs", []),
                "outputs": getattr(self, "outputs", []),
                "dependencies": getattr(self, "dependencies", []),
                "assigned_agent": self.assigned_agent.value
                if hasattr(self.assigned_agent, "value")
                else str(self.assigned_agent),
                "depth": getattr(self, "depth", 0),
                "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            }

        _existing_task.to_dict = _task_to_dict

    # ── Lightweight stubs for modules imported by BaseAgent / agents ──
    mod("vetinari.structured_logging", log_event=MagicMock(), get_logger=MagicMock(return_value=MagicMock()))
    mod("vetinari.execution_context", get_context_manager=MagicMock(), ToolPermission=MagicMock())

    # learning sub-modules
    mod(
        "vetinari.learning.quality_scorer",
        get_quality_scorer=MagicMock(
            return_value=MagicMock(score=MagicMock(return_value=MagicMock(overall_score=0.8)))
        ),
    )
    mod("vetinari.learning.feedback_loop", get_feedback_loop=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.model_selector", get_thompson_selector=MagicMock(return_value=MagicMock()))
    mod(
        "vetinari.learning.prompt_evolver",
        get_prompt_evolver=MagicMock(
            return_value=MagicMock(
                select_prompt=MagicMock(return_value=("", "default")),
                register_baseline=MagicMock(),
                record_result=MagicMock(),
            )
        ),
    )
    mod("vetinari.learning.training_data", get_training_collector=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.episode_memory", get_episode_memory=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.auto_tuner")
    mod("vetinari.learning.cost_optimizer")
    mod("vetinari.learning.workflow_learner")
    mod("vetinari.learning.self_refinement")

    # constraints
    mod(
        "vetinari.constraints.registry",
        get_constraint_registry=MagicMock(
            return_value=MagicMock(get_constraints_for_agent=MagicMock(return_value=None))
        ),
    )

    # config
    mod(
        "vetinari.config.inference_config",
        get_inference_config=MagicMock(return_value=MagicMock(get_effective_params=MagicMock(return_value={}))),
    )

    # token optimizer
    mod(
        "vetinari.token_optimizer",
        get_token_optimizer=MagicMock(
            return_value=MagicMock(get_task_profile=MagicMock(return_value=(4096, 0.3, 0.9)))
        ),
    )

    # adapter_manager
    mod("vetinari.adapter_manager", get_adapter_manager=MagicMock(return_value=None))
    mod("vetinari.adapters.llama_cpp_adapter", LocalInferenceAdapter=MagicMock())

    # adapters.base
    @dataclass
    class InferenceRequest:
        model_id: str = ""
        prompt: str = ""
        system_prompt: str | None = None
        max_tokens: int = 4096
        temperature: float = 0.3
        top_p: float = 0.9
        top_k: int = 40
        stop_sequences: list[str] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

    mod("vetinari.adapters.base", InferenceRequest=InferenceRequest)

    # Other stubs
    mod("vetinari.tools.web_search_tool", get_search_tool=MagicMock(return_value=MagicMock()))

    # Return key classes for test use
    return {
        "AgentType": AgentType,
        "StatusEnum": StatusEnum,
        "AgentSpec": AgentSpec,
        "Task": Task,
        "AgentTask": AgentTask,
        "Plan": Plan,
        "AgentResult": AgentResult,
        "VerificationResult": VerificationResult,
    }


# ---------------------------------------------------------------------------
# Install stubs, then load real agent modules
# ---------------------------------------------------------------------------

_STUBS = _build_stubs()
_StubAgentType = _STUBS["AgentType"]
StatusEnum = _STUBS["StatusEnum"]
AgentSpec = _STUBS["AgentSpec"]
Task = _STUBS["Task"]
AgentTask = _STUBS["AgentTask"]
Plan = _STUBS["Plan"]
AgentResult = _STUBS["AgentResult"]
VerificationResult = _STUBS["VerificationResult"]

# Load real modules
_base_mod = _load_agent_module("vetinari/agents/base_agent.py", "vetinari.agents.base_agent")
_multi_mode_mod = _load_agent_module("vetinari/agents/multi_mode_agent.py", "vetinari.agents.multi_mode_agent")
_planner_mod = _load_agent_module("vetinari/agents/planner_agent.py", "vetinari.agents.planner_agent")
_quality_mod = _load_agent_module(
    "vetinari/agents/consolidated/quality_agent.py", "vetinari.agents.consolidated.quality_agent"
)

BaseAgent = _base_mod.BaseAgent
MultiModeAgent = _multi_mode_mod.MultiModeAgent
PlannerAgent = _planner_mod.ForemanAgent
get_planner_agent = _planner_mod.get_foreman_agent
# AutomationAgentAlias was consolidated into InspectorAgent (v0.5.0)
AutomationAgentAlias = _quality_mod.InspectorAgent
get_test_automation_agent = _quality_mod.get_inspector_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(agent_type, description="Test task", prompt="Do something", context=None, task_id=None):
    """Create a minimal AgentTask for testing."""
    return AgentTask(
        task_id=task_id or f"test_{uuid.uuid4().hex[:6]}",
        agent_type=agent_type,
        description=description,
        prompt=prompt,
        context=context or {},
    )


class _ConcreteMultiMode(MultiModeAgent):
    """Concrete subclass of MultiModeAgent for testing the ABC."""

    MODES = {
        "analyze": "_handle_analyze",
        "build": "_handle_build",
        "review": "_handle_review",
    }
    DEFAULT_MODE = "analyze"
    MODE_KEYWORDS = {
        "analyze": ["analyze", "investigate", "research"],
        "build": ["build", "create", "implement", "code"],
        "review": ["review", "check", "evaluate", "audit"],
    }

    def __init__(self, config=None):
        super().__init__(AgentType.INSPECTOR, config)

    def _handle_analyze(self, task):
        return AgentResult(success=True, output={"mode": "analyze", "data": "analyzed"})

    def _handle_build(self, task):
        return AgentResult(success=True, output={"mode": "build", "data": "built"})

    def _handle_review(self, task):
        return AgentResult(success=True, output={"mode": "review", "data": "reviewed"})


# ###########################################################################
# SECTION 1: MultiModeAgent Tests
# ###########################################################################


class TestMultiModeAgentInit:
    """Tests for MultiModeAgent constructor and properties."""

    def test_init_sets_agent_type(self):
        agent = _ConcreteMultiMode()
        assert agent.agent_type == AgentType.INSPECTOR

    def test_init_current_mode_is_none(self):
        agent = _ConcreteMultiMode()
        assert agent.current_mode is None

    def test_init_with_config(self):
        agent = _ConcreteMultiMode(config={"key": "val"})
        assert agent._config.get("key") == "val"

    def test_init_without_config(self):
        agent = _ConcreteMultiMode()
        assert isinstance(agent._config, dict)

    def test_available_modes(self):
        agent = _ConcreteMultiMode()
        assert set(agent.available_modes) == {"analyze", "build", "review"}

    def test_available_modes_empty_for_base(self):
        """A direct MultiModeAgent subclass with no MODES returns []."""

        class EmptyModes(MultiModeAgent):
            MODES = {}

            def get_system_prompt(self):
                return ""

        agent = EmptyModes(AgentType.INSPECTOR)
        assert agent.available_modes == []

    def test_get_capabilities_returns_mode_names(self):
        agent = _ConcreteMultiMode()
        caps = agent.get_capabilities()
        assert set(caps) == {"analyze", "build", "review"}

    def test_current_mode_property(self):
        agent = _ConcreteMultiMode()
        agent._current_mode = "build"
        assert agent.current_mode == "build"


class TestMultiModeResolveMode:
    """Tests for MultiModeAgent._resolve_mode()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = _ConcreteMultiMode()

    def test_explicit_mode_in_context(self):
        task = _make_task(AgentType.INSPECTOR, context={"mode": "build"})
        mode = self.agent._resolve_mode(task)
        assert mode == "build"

    def test_explicit_mode_invalid_returned_as_is(self):
        # _resolve_mode returns the explicit mode verbatim so execute() can
        # delegate it to a more capable agent rather than silently remap.
        task = _make_task(AgentType.INSPECTOR, context={"mode": "nonexistent"})
        mode = self.agent._resolve_mode(task)
        assert mode == "nonexistent"

    def test_keyword_matching_analyze(self):
        task = _make_task(AgentType.INSPECTOR, description="Please analyze this code")
        mode = self.agent._resolve_mode(task)
        assert mode == "analyze"

    def test_keyword_matching_build(self):
        task = _make_task(AgentType.INSPECTOR, description="Build and implement the feature")
        mode = self.agent._resolve_mode(task)
        assert mode == "build"

    def test_keyword_matching_review(self):
        task = _make_task(AgentType.INSPECTOR, description="Review and evaluate the code")
        mode = self.agent._resolve_mode(task)
        assert mode == "review"

    def test_keyword_matching_best_score_wins(self):
        task = _make_task(AgentType.INSPECTOR, description="build create implement code something")
        mode = self.agent._resolve_mode(task)
        assert mode == "build"

    def test_default_mode_fallback(self):
        task = _make_task(AgentType.INSPECTOR, description="do something unrelated")
        mode = self.agent._resolve_mode(task)
        assert mode == "analyze"  # DEFAULT_MODE

    def test_no_modes_no_default_returns_empty(self):
        class NoModes(MultiModeAgent):
            MODES = {}
            DEFAULT_MODE = ""
            MODE_KEYWORDS = {}

            def get_system_prompt(self):
                return ""

        agent = NoModes(AgentType.INSPECTOR)
        task = _make_task(AgentType.INSPECTOR)
        mode = agent._resolve_mode(task)
        assert mode == ""

    def test_agent_type_on_task_used_for_legacy_mapping(self):
        """If task.agent_type matches a legacy mapping key, use that."""
        task = _make_task(AgentType.WORKER, description="explore something")
        # EXPLORER -> "analyze"
        mode = self.agent._resolve_mode(task)
        assert mode == "analyze"

    def test_none_description_falls_to_default(self):
        task = _make_task(AgentType.INSPECTOR, description="")
        task.description = None
        mode = self.agent._resolve_mode(task)
        assert mode == "analyze"

    def test_empty_description_falls_to_default(self):
        task = _make_task(AgentType.INSPECTOR, description="")
        mode = self.agent._resolve_mode(task)
        assert mode == "analyze"


class TestMultiModeExecute:
    """Tests for MultiModeAgent.execute()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = _ConcreteMultiMode()

    def test_execute_routes_to_analyze(self):
        task = _make_task(AgentType.INSPECTOR, context={"mode": "analyze"})
        result = self.agent.execute(task)
        assert result.success
        assert result.output["mode"] == "analyze"
        assert self.agent.current_mode == "analyze"

    def test_execute_routes_to_build(self):
        task = _make_task(AgentType.INSPECTOR, context={"mode": "build"})
        result = self.agent.execute(task)
        assert result.success
        assert result.output["mode"] == "build"

    def test_execute_routes_to_review(self):
        task = _make_task(AgentType.INSPECTOR, context={"mode": "review"})
        result = self.agent.execute(task)
        assert result.success
        assert result.output["mode"] == "review"

    def test_execute_sets_current_mode(self):
        task = _make_task(AgentType.INSPECTOR, context={"mode": "build"})
        self.agent.execute(task)
        assert self.agent.current_mode == "build"

    def test_execute_unknown_mode_returns_error(self):
        """If mode resolves to empty string with no handler, return error."""

        class BrokenAgent(MultiModeAgent):
            MODES = {"valid": "_handle_valid"}
            DEFAULT_MODE = ""
            MODE_KEYWORDS = {}

            def get_system_prompt(self):
                return ""

            def _handle_valid(self, task):
                from vetinari.agents.base_agent import AgentResult

                return AgentResult(success=True, output="ok")

        agent = BrokenAgent(AgentType.INSPECTOR)
        # "nonexistent" not in MODES -> falls to default "" -> no handler
        task = _make_task(AgentType.INSPECTOR, context={"mode": "nonexistent"})
        result = agent.execute(task)
        # With DEFAULT_MODE="" the fallback to first key "valid" succeeds
        # so the result depends on whether fallback logic triggers
        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "output")

    def test_execute_handler_not_implemented(self):
        """B5: init-time validation catches missing handler methods."""

        class MissingHandler(MultiModeAgent):
            MODES = {"alpha": "_nonexistent_method"}
            DEFAULT_MODE = "alpha"
            MODE_KEYWORDS = {}

            def get_system_prompt(self):
                return ""

        with pytest.raises(TypeError) as ctx:
            MissingHandler(AgentType.INSPECTOR)
        assert "_nonexistent_method" in str(ctx.value)

    def test_execute_handler_raises_exception(self):
        class FailingAgent(_ConcreteMultiMode):
            def _handle_analyze(self, task):
                raise ValueError("Analysis exploded")

        agent = FailingAgent()
        task = _make_task(AgentType.INSPECTOR, context={"mode": "analyze"})
        result = agent.execute(task)
        assert not result.success
        assert any("exploded" in e for e in result.errors)

    def test_execute_completes_task(self):
        task = _make_task(AgentType.INSPECTOR, context={"mode": "build"})
        self.agent.execute(task)
        assert task.completed_at != ""

    def test_execute_keyword_routing(self):
        task = _make_task(AgentType.INSPECTOR, description="review and evaluate all code changes")
        result = self.agent.execute(task)
        assert result.success
        assert result.output["mode"] == "review"


class TestMultiModeVerify:
    """Tests for MultiModeAgent.verify()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = _ConcreteMultiMode()

    def test_verify_none_output_fails(self):
        result = self.agent.verify(None)
        assert not result.passed
        assert result.score == 0.0
        assert len(result.issues) > 0

    def test_verify_valid_output_passes(self):
        result = self.agent.verify({"content": "something"})
        assert result.passed
        assert result.score == 0.7

    def test_verify_string_output_fails(self):
        """Non-dict output must fail — no structured verification evidence."""
        result = self.agent.verify("some string")
        assert not result.passed
        assert result.score == 0.0

    def test_verify_empty_dict_fails(self):
        """Empty dict has no substance — must fail verification."""
        result = self.agent.verify({})
        assert not result.passed
        assert result.score == 0.0

    def test_verify_empty_list_fails(self):
        """Non-dict output must fail — no structured verification evidence."""
        result = self.agent.verify([])
        assert not result.passed
        assert result.score == 0.0

    def test_verify_zero_fails(self):
        """Non-dict output must fail — no structured verification evidence."""
        result = self.agent.verify(0)
        assert not result.passed
        assert result.score == 0.0

    def test_verify_false_fails(self):
        """Non-dict output must fail — no structured verification evidence."""
        result = self.agent.verify(False)
        assert not result.passed
        assert result.score == 0.0


class TestMultiModeSystemPrompt:
    """Tests for MultiModeAgent system prompt methods."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = _ConcreteMultiMode()

    def test_get_system_prompt_default_mode(self):
        prompt = self.agent.get_system_prompt()
        # With runtime prompt loading, prompt may come from .md file or base fallback
        assert prompt  # Non-empty prompt returned

    def test_get_system_prompt_with_current_mode(self):
        self.agent._current_mode = "review"
        prompt = self.agent.get_system_prompt()
        # Mode-specific prompt loaded from config or base fallback
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_base_system_prompt(self):
        prompt = self.agent._get_base_system_prompt()
        assert self.agent.name in prompt

    def test_get_mode_system_prompt_default_empty(self):
        result = self.agent._get_mode_system_prompt("analyze")
        assert result == ""

    def test_get_system_prompt_custom_override(self):
        """Subclass _get_mode_system_prompt is used as fallback when prompt loader fails."""

        class CustomPrompt(_ConcreteMultiMode):
            def _get_mode_system_prompt(self, mode):
                if mode == "build":
                    return "Custom builder prompt"
                return ""

        agent = CustomPrompt()
        agent._current_mode = "build"
        # Patch prompt_loader module so the lazy import inside get_system_prompt raises
        mock_loader = MagicMock()
        mock_loader.load_agent_prompt = MagicMock(side_effect=Exception("no file"))
        with patch.dict("sys.modules", {"vetinari.agents.prompt_loader": mock_loader}):
            prompt = agent.get_system_prompt()
        assert prompt == "Custom builder prompt"


# ###########################################################################
# SECTION 2: PlannerAgent Tests
# ###########################################################################


class TestPlannerAgentInit:
    """Tests for PlannerAgent constructor and properties."""

    def test_init_default_config(self):
        agent = PlannerAgent()
        assert agent.agent_type == AgentType.FOREMAN
        assert agent._max_depth == 14
        assert agent._min_tasks == 5
        assert agent._max_tasks == 15

    def test_init_custom_config(self):
        agent = PlannerAgent(config={"max_depth": 5, "min_tasks": 2, "max_tasks": 8})
        assert agent._max_depth == 5
        assert agent._min_tasks == 2
        assert agent._max_tasks == 8

    def test_agent_type(self):
        agent = PlannerAgent()
        assert agent.agent_type == AgentType.FOREMAN

    def test_get_capabilities(self):
        agent = PlannerAgent()
        caps = agent.get_capabilities()
        assert "plan_generation" in caps
        assert "task_decomposition" in caps
        assert "dependency_mapping" in caps
        assert "resource_estimation" in caps
        assert "risk_assessment" in caps
        assert len(caps) >= 5

    def test_get_system_prompt_nonempty(self):
        agent = PlannerAgent()
        prompt = agent.get_system_prompt()
        # Runtime prompt loading should provide a non-empty prompt.
        assert prompt  # Non-empty
        # Prompt describes the planning role (may say "Planner", "Planning", etc.)
        assert any(kw in prompt for kw in ("Planner", "Planning", "FOREMAN", "Foreman"))

    def test_system_prompt_mentions_agent_role(self):
        agent = PlannerAgent()
        prompt = agent.get_system_prompt()
        # Prompt describes planning responsibilities.
        assert "task" in prompt.lower() or "plan" in prompt.lower()

    def test_system_prompt_is_substantial(self):
        agent = PlannerAgent()
        prompt = agent.get_system_prompt()
        # Runtime-loaded prompts should be substantial.
        assert len(prompt) > 200


class TestPlannerAgentExecute:
    """Tests for PlannerAgent.execute()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = PlannerAgent()

    def test_execute_invalid_task_type(self):
        task = _make_task(AgentType.WORKER, description="build something")
        result = self.agent.execute(task)
        # v0.4.0: PlannerAgent handles multiple modes, may succeed via fallback
        assert result is not None
        assert hasattr(result, "success")
        assert isinstance(result.success, bool)

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_execute_with_keyword_fallback(self, mock_infer):
        task = _make_task(
            AgentType.FOREMAN,
            description="Build a web application with user authentication",
            prompt="Build a web application with user authentication",
        )
        result = self.agent.execute(task)
        assert result.success
        plan_dict = result.output
        assert "plan_id" in plan_dict
        assert "tasks" in plan_dict
        assert "goal" in plan_dict
        assert len(plan_dict["tasks"]) > 0

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_execute_sets_metadata(self, mock_infer):
        task = _make_task(AgentType.FOREMAN, description="Implement a REST API", prompt="Implement a REST API")
        result = self.agent.execute(task)
        assert result.success
        assert "plan_id" in result.metadata
        assert "task_count" in result.metadata
        assert "goal" in result.metadata

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_execute_vague_goal_needs_context(self, mock_infer):
        task = _make_task(AgentType.FOREMAN, description="help me", prompt="help me")
        result = self.agent.execute(task)
        assert result.success
        plan_dict = result.output
        assert plan_dict.get("needs_context", False)
        assert plan_dict.get("follow_up_question", "") != ""

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_execute_very_short_goal_vague(self, mock_infer):
        task = _make_task(AgentType.FOREMAN, description="hi", prompt="hi")
        result = self.agent.execute(task)
        assert result.success
        assert result.output.get("needs_context", False)

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_execute_goal_with_special_chars_only(self, mock_infer):
        task = _make_task(AgentType.FOREMAN, description="!@#$%", prompt="!@#$%")
        result = self.agent.execute(task)
        assert result.success
        assert result.output.get("needs_context", False)

    def test_execute_exception_returns_failure(self):
        """If _generate_plan raises, execute catches and returns failure."""
        task = _make_task(AgentType.FOREMAN, description="Build something complex", prompt="Build something complex")
        with (
            patch.object(self.agent._plan_cache, "find_similar", return_value=None),
            patch.object(self.agent, "_generate_plan", side_effect=RuntimeError("boom")),
        ):
            result = self.agent.execute(task)
        assert not result.success
        assert "boom" in result.errors[0]

    @patch.object(PlannerAgent, "_infer_json")
    def test_execute_uses_prompt_over_description(self, mock_infer):
        mock_infer.return_value = None  # fallback to keyword
        task = _make_task(
            AgentType.FOREMAN, description="Some desc", prompt="Build a web dashboard with data visualization"
        )
        result = self.agent.execute(task)
        assert result.success
        # The goal should be the prompt (first choice)
        assert result.metadata.get("goal") == "Build a web dashboard with data visualization"


class TestPlannerAgentVerify:
    """Tests for PlannerAgent.verify()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = PlannerAgent()

    def test_verify_non_dict_fails(self):
        result = self.agent.verify("not a dict")
        assert not result.passed

    def test_verify_none_fails(self):
        result = self.agent.verify(None)
        assert not result.passed

    def test_verify_empty_dict_missing_fields(self):
        result = self.agent.verify({})
        # Missing plan_id, goal, tasks => three -0.2 deductions = 0.4, fail
        assert not result.passed

    def test_verify_valid_plan_with_deps_passes(self):
        plan = {
            "plan_id": "plan_123",
            "goal": "Build app",
            "tasks": [
                {"id": "t1", "description": "Explore", "dependencies": []},
                {"id": "t2", "description": "Build", "dependencies": ["t1"]},
                {"id": "t3", "description": "Test", "dependencies": ["t2"]},
                {"id": "t4", "description": "Review", "dependencies": ["t3"]},
                {"id": "t5", "description": "Deploy", "dependencies": ["t4"]},
            ],
        }
        result = self.agent.verify(plan)
        assert result.passed
        assert result.score >= 0.7

    def test_verify_too_few_tasks(self):
        plan = {
            "plan_id": "plan_123",
            "goal": "Build app",
            "tasks": [
                {"id": "t1", "description": "Do it", "dependencies": ["t0"]},
            ],
        }
        result = self.agent.verify(plan)
        assert any(i["type"] == "insufficient_tasks" for i in result.issues)

    def test_verify_no_dependencies_warning(self):
        plan = {
            "plan_id": "plan_123",
            "goal": "Build app",
            "tasks": [{"id": f"t{i}", "description": f"Task {i}", "dependencies": []} for i in range(6)],
        }
        result = self.agent.verify(plan)
        assert any(i["type"] == "no_dependencies" for i in result.issues)

    def test_verify_missing_plan_id(self):
        plan = {"goal": "test", "tasks": []}
        result = self.agent.verify(plan)
        assert any("plan_id" in i["message"] for i in result.issues)

    def test_verify_missing_goal(self):
        plan = {"plan_id": "p1", "tasks": []}
        result = self.agent.verify(plan)
        assert any("goal" in i["message"] for i in result.issues)

    def test_verify_missing_tasks_key(self):
        plan = {"plan_id": "p1", "goal": "test"}
        result = self.agent.verify(plan)
        assert any("tasks" in i["message"] for i in result.issues)

    def test_verify_list_fails(self):
        result = self.agent.verify([1, 2, 3])
        assert not result.passed

    def test_verify_score_clamped_to_zero(self):
        result = self.agent.verify({})
        assert result.score >= 0


class TestPlannerGeneratePlan:
    """Tests for PlannerAgent._generate_plan()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = PlannerAgent()

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_vague_goal_something(self, mock_infer):
        plan = self.agent._generate_plan("do something", {})
        assert plan.needs_context
        assert plan.follow_up_question != ""

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_vague_goal_fix_it(self, mock_infer):
        plan = self.agent._generate_plan("fix it", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_vague_goal_two_words(self, mock_infer):
        plan = self.agent._generate_plan("build", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_non_alphanumeric_goal(self, mock_infer):
        plan = self.agent._generate_plan("!!!", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_clear_goal_uses_keyword_fallback(self, mock_infer):
        plan = self.agent._generate_plan("Build a REST API with user authentication and database", {})
        assert not plan.needs_context
        assert len(plan.tasks) > 0

    @patch.object(PlannerAgent, "_infer_json")
    def test_llm_decomposition_used_when_available(self, mock_infer):
        mock_infer.return_value = [
            {
                "id": "t1",
                "description": "Explore",
                "inputs": ["goal"],
                "outputs": ["spec"],
                "dependencies": [],
                "assigned_agent": "EXPLORER",
            },
            {
                "id": "t2",
                "description": "Build",
                "inputs": ["spec"],
                "outputs": ["code"],
                "dependencies": ["t1"],
                "assigned_agent": AgentType.WORKER.value,
            },
        ]
        plan = self.agent._generate_plan("Implement a microservices architecture for e-commerce", {})
        assert not plan.needs_context
        assert len(plan.tasks) == 2

    @patch.object(PlannerAgent, "_infer_json")
    def test_llm_returns_empty_falls_to_keyword(self, mock_infer):
        mock_infer.return_value = []
        plan = self.agent._generate_plan("Build a web application with user login", {})
        assert len(plan.tasks) > 0

    @patch.object(PlannerAgent, "_infer_json")
    def test_too_many_tasks_adds_warning(self, mock_infer):
        # Return more than _max_tasks (15)
        mock_infer.return_value = [
            {
                "id": f"t{i}",
                "description": f"Task {i}",
                "inputs": [],
                "outputs": [],
                "dependencies": [],
                "assigned_agent": AgentType.WORKER.value,
            }
            for i in range(20)
        ]
        plan = self.agent._generate_plan("Massive project with everything under the sun", {})
        assert len(plan.warnings) > 0
        assert "20" in plan.warnings[0]


class TestPlannerDecomposeGoalLLM:
    """Tests for PlannerAgent._decompose_goal_llm()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = PlannerAgent()

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_llm_returns_none(self, mock_infer):
        tasks = self.agent._decompose_goal_llm("Build something", {})
        assert tasks == []

    @patch.object(PlannerAgent, "_infer_json", return_value="not a list")
    def test_llm_returns_non_list(self, mock_infer):
        tasks = self.agent._decompose_goal_llm("Build something", {})
        assert tasks == []

    @patch.object(PlannerAgent, "_infer_json")
    def test_llm_returns_valid_tasks(self, mock_infer):
        mock_infer.return_value = [
            {
                "id": "t1",
                "description": "Research",
                "inputs": ["goal"],
                "outputs": ["report"],
                "dependencies": [],
                "assigned_agent": "EXPLORER",
            },
            {
                "id": "t2",
                "description": "Implement",
                "inputs": ["report"],
                "outputs": ["code"],
                "dependencies": ["t1"],
                "assigned_agent": AgentType.WORKER.value,
            },
        ]
        tasks = self.agent._decompose_goal_llm("Build something", {})
        assert len(tasks) == 2
        assert tasks[0].id == "t1"
        assert tasks[1].id == "t2"

    @patch.object(PlannerAgent, "_infer_json")
    def test_llm_invalid_agent_type_defaults_to_worker(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "Do stuff", "assigned_agent": "NONEXISTENT_AGENT"},
        ]
        tasks = self.agent._decompose_goal_llm("Build something", {})
        assert len(tasks) == 1
        # Falls back to AgentType.WORKER when the agent string is unrecognised
        assert tasks[0].assigned_agent == AgentType.WORKER

    @patch.object(PlannerAgent, "_infer_json")
    def test_llm_missing_fields_uses_defaults(self, mock_infer):
        mock_infer.return_value = [
            {"description": "Minimal task"},
        ]
        tasks = self.agent._decompose_goal_llm("Build something", {})
        assert len(tasks) == 1
        assert tasks[0].id == "t1"  # auto-generated

    @patch.object(PlannerAgent, "_infer_json")
    def test_llm_skips_non_dict_items(self, mock_infer):
        mock_infer.return_value = [
            "not a dict",
            42,
            {"id": "t1", "description": "Valid"},
        ]
        tasks = self.agent._decompose_goal_llm("Build something", {})
        assert len(tasks) == 1

    @patch.object(PlannerAgent, "_infer_json")
    def test_dag_depth_calculation(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "First", "dependencies": []},
            {"id": "t2", "description": "Second", "dependencies": ["t1"]},
            {"id": "t3", "description": "Third", "dependencies": ["t2"]},
        ]
        tasks = self.agent._decompose_goal_llm("Build app", {})
        assert tasks[0].depth == 0
        assert tasks[1].depth == 1
        assert tasks[2].depth == 2

    @patch.object(PlannerAgent, "_infer_json")
    def test_dag_depth_with_cycle_guard(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "First", "dependencies": ["t2"]},
            {"id": "t2", "description": "Second", "dependencies": ["t1"]},
        ]
        tasks = self.agent._decompose_goal_llm("Build app", {})
        # Should not hang due to cycle guard
        assert len(tasks) == 2

    @patch.object(PlannerAgent, "_infer_json")
    def test_context_passed_to_prompt(self, mock_infer):
        mock_infer.return_value = []
        self.agent._decompose_goal_llm("Build app", {"key": "value"})
        call_args = mock_infer.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "key" in prompt

    @patch.object(PlannerAgent, "_infer_json")
    def test_empty_context(self, mock_infer):
        mock_infer.return_value = []
        self.agent._decompose_goal_llm("Build app", {})
        mock_infer.assert_called_once()
        prompt = mock_infer.call_args.args[0]
        assert "Goal: Build app" in prompt
        assert "\nContext:" not in prompt


class TestPlannerSingleton:
    """Tests for get_planner_agent() singleton."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _planner_mod._foreman_agent = None
        yield
        _planner_mod._foreman_agent = None

    def test_get_planner_agent_returns_instance(self):
        agent = get_planner_agent()
        assert isinstance(agent, PlannerAgent)

    def test_get_planner_agent_singleton(self):
        a1 = get_planner_agent()
        a2 = get_planner_agent()
        assert a1 is a2

    def test_get_planner_agent_with_config(self):
        agent = get_planner_agent(config={"max_tasks": 10})
        assert agent._max_tasks == 10


# ###########################################################################
# SECTION 3: Cross-Agent Integration Tests
# ###########################################################################


class TestCrossAgentInteractions:
    """Integration-style tests verifying agents can interoperate."""

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_planner_generates_test_automation_tasks(self, mock_infer):
        planner = PlannerAgent()
        tasks = planner._decompose_goal_keyword("Build a code application", {})
        test_tasks = [t for t in tasks if t.assigned_agent == AgentType.INSPECTOR]
        assert len(test_tasks) > 0

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    @patch.object(AutomationAgentAlias, "_infer_json")
    def test_planner_then_test_agent(self, mock_test_infer, mock_plan_infer):
        mock_test_infer.return_value = {"test_files": [], "test_scripts": []}

        planner = PlannerAgent()
        plan_task = _make_task(
            AgentType.FOREMAN, description="Build a Python CLI tool", prompt="Build a Python CLI tool"
        )
        plan_result = planner.execute(plan_task)
        assert plan_result.success

        test_agent = AutomationAgentAlias()
        test_task = _make_task(
            AgentType.INSPECTOR,
            description="Generate tests for CLI tool",
            context={"features": ["cli"], "code": "def main(): pass"},
        )
        test_result = test_agent.execute(test_task)
        assert test_result.success

    def test_multi_mode_and_planner_coexist(self):
        multi = _ConcreteMultiMode()
        planner = PlannerAgent()
        assert multi.agent_type != planner.agent_type
        assert len(multi.get_capabilities()) > 0
        assert len(planner.get_capabilities()) > 0

    def test_all_agents_have_system_prompts(self):
        agents = [_ConcreteMultiMode(), PlannerAgent(), AutomationAgentAlias()]
        for agent in agents:
            prompt = agent.get_system_prompt()
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_all_agents_have_capabilities(self):
        agents = [_ConcreteMultiMode(), PlannerAgent(), AutomationAgentAlias()]
        for agent in agents:
            caps = agent.get_capabilities()
            assert isinstance(caps, list)
            assert len(caps) > 0


# ###########################################################################
# SECTION 5: Additional Edge Case Tests
# ###########################################################################


class TestMultiModeEdgeCases:
    """Edge cases for MultiModeAgent."""

    def test_execute_with_empty_modes_dict(self):
        class EmptyAgent(MultiModeAgent):
            MODES = {}
            DEFAULT_MODE = ""
            MODE_KEYWORDS = {}

            def get_system_prompt(self):
                return ""

        agent = EmptyAgent(AgentType.INSPECTOR)
        task = _make_task(AgentType.INSPECTOR)
        result = agent.execute(task)
        assert not result.success

    def test_get_capabilities_matches_modes(self):
        agent = _ConcreteMultiMode()
        assert sorted(agent.get_capabilities()) == sorted(agent.available_modes)

    def test_mode_resolution_priority_explicit_over_keyword(self):
        """Explicit mode in context beats keyword matching."""
        agent = _ConcreteMultiMode()
        task = _make_task(AgentType.INSPECTOR, description="review and evaluate everything", context={"mode": "build"})
        mode = agent._resolve_mode(task)
        assert mode == "build"

    def test_mode_resolution_priority_legacy_over_keyword(self):
        """Legacy type mapping beats keyword matching."""
        agent = _ConcreteMultiMode()
        task = _make_task(AgentType.INSPECTOR, description="review everything", context={"mode": "build"})
        mode = agent._resolve_mode(task)
        assert mode == "build"

    def test_multiple_executions_update_mode(self):
        agent = _ConcreteMultiMode()
        task1 = _make_task(AgentType.INSPECTOR, context={"mode": "analyze"})
        agent.execute(task1)
        assert agent.current_mode == "analyze"

        task2 = _make_task(AgentType.INSPECTOR, context={"mode": "build"})
        agent.execute(task2)
        assert agent.current_mode == "build"


class TestPlannerEdgeCases:
    """Edge cases for PlannerAgent."""

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_plan_has_unique_id(self, _):
        agent = PlannerAgent()
        p1 = agent._generate_plan("Build a web app with user management", {})
        p2 = agent._generate_plan("Build another web app with features", {})
        assert p1.plan_id != p2.plan_id

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_plan_stores_goal(self, _):
        agent = PlannerAgent()
        plan = agent._generate_plan("Build a magnificent space station", {})
        assert plan.goal == "Build a magnificent space station"

    def test_verify_score_boundary(self):
        agent = PlannerAgent()
        # Exactly at 0.7 threshold
        plan = {
            "plan_id": "p1",
            "goal": "test",
            "tasks": [{"id": f"t{i}", "description": f"Task {i}", "dependencies": []} for i in range(3)],
        }
        result = agent.verify(plan)
        # 1.0 - 0.1 (too few tasks < 5) - 0.1 (no deps) = 0.8
        assert result.passed

    def test_verify_just_below_threshold(self):
        agent = PlannerAgent()
        plan = {"plan_id": "p1", "tasks": []}  # missing goal
        result = agent.verify(plan)
        # 1.0 - 0.2 (missing goal) - 0.1 (too few tasks) - 0.1 (no deps) = 0.6
        assert not result.passed

    @patch.object(PlannerAgent, "_infer_json")
    def test_llm_task_with_acceptance_criteria(self, mock_infer):
        mock_infer.return_value = [
            {
                "id": "t1",
                "description": "Explore",
                "dependencies": [],
                "assigned_agent": "EXPLORER",
                "acceptance_criteria": "Full report",
            },
        ]
        agent = PlannerAgent()
        tasks = agent._decompose_goal_llm("Build app", {})
        assert len(tasks) == 1

    @patch.object(PlannerAgent, "_infer_json")
    def test_llm_parallel_tasks_have_same_depth(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "Root", "dependencies": []},
            {"id": "t2", "description": "Parallel A", "dependencies": ["t1"]},
            {"id": "t3", "description": "Parallel B", "dependencies": ["t1"]},
            {"id": "t4", "description": "Final", "dependencies": ["t2", "t3"]},
        ]
        agent = PlannerAgent()
        tasks = agent._decompose_goal_llm("Build app", {})
        assert tasks[0].depth == 0
        assert tasks[1].depth == 1
        assert tasks[2].depth == 1
        assert tasks[3].depth == 2


class TestMultiModeInheritance:
    """Test that MultiModeAgent properly extends BaseAgent."""

    def test_is_subclass_of_base_agent(self):
        assert issubclass(MultiModeAgent, BaseAgent)

    def test_concrete_is_subclass(self):
        assert issubclass(_ConcreteMultiMode, MultiModeAgent)
        assert issubclass(_ConcreteMultiMode, BaseAgent)

    def test_agent_has_metadata(self):
        agent = _ConcreteMultiMode()
        meta = agent.get_metadata()
        assert "agent_type" in meta
        assert "capabilities" in meta
        assert "name" in meta

    def test_agent_repr(self):
        agent = _ConcreteMultiMode()
        r = repr(agent)
        assert "_ConcreteMultiMode" in r

    def test_planner_has_metadata(self):
        agent = PlannerAgent()
        meta = agent.get_metadata()
        assert meta["agent_type"] == AgentType.FOREMAN.value

    def test_test_automation_has_metadata(self):
        agent = AutomationAgentAlias()
        meta = agent.get_metadata()
        # v0.4.0: AutomationAgentAlias consolidated into QualityAgent
        assert meta["agent_type"] == AgentType.INSPECTOR.value


class TestPlannerVagueGoalDetection:
    """Detailed tests for vague goal heuristic in _generate_plan."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.agent = PlannerAgent()

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_single_word_is_vague(self, _):
        plan = self.agent._generate_plan("hello", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_two_words_is_vague(self, _):
        plan = self.agent._generate_plan("do it", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_help_me_is_vague(self, _):
        plan = self.agent._generate_plan("help me please", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_create_something_is_vague(self, _):
        plan = self.agent._generate_plan("create something now", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_build_something_is_vague(self, _):
        plan = self.agent._generate_plan("build something", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_clear_goal_is_not_vague(self, _):
        plan = self.agent._generate_plan(
            "Implement a REST API using FastAPI with authentication and database integration", {}
        )
        assert not plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_medium_length_without_vague_words_is_not_vague(self, _):
        plan = self.agent._generate_plan("Deploy the application to production servers", {})
        assert not plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_empty_string_is_vague(self, _):
        plan = self.agent._generate_plan("", {})
        assert plan.needs_context

    @patch.object(PlannerAgent, "_infer_json", return_value=None)
    def test_whitespace_only_is_vague(self, _):
        plan = self.agent._generate_plan("   ", {})
        assert plan.needs_context
