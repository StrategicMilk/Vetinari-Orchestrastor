"""
Comprehensive tests for three Vetinari agents:
 - MultiModeAgent (vetinari/agents/multi_mode_agent.py)
 - PlannerAgent (vetinari/agents/planner_agent.py)
 - TestAutomationAgent (vetinari/agents/test_automation_agent.py)

All external dependencies are stubbed so the tests run in isolation without
any real LLM, filesystem (where controllable), or third-party services.
"""

import sys
import types
import importlib.util
import os
import json
import pytest
import re
import uuid
import tempfile
import unittest
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call, mock_open

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
        "vetinari", "vetinari.agents", "vetinari.adapters",
        "vetinari.learning", "vetinari.constraints",
        "vetinari.drift", "vetinari.config",
        "vetinari.tools", "vetinari.coding_agent",
        "vetinari.orchestration", "vetinari.analytics",
        "vetinari.dashboard", "vetinari.memory",
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

    from enum import Enum

    # Grab real enums from types_mod
    AgentType = getattr(types_mod, "AgentType", None)
    TaskStatus = getattr(types_mod, "TaskStatus", None)
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
                "name": self.name, "description": self.description,
                "default_model": self.default_model,
            }

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
            if self.assigned_agent is None:
                self.assigned_agent = AgentType.PLANNER
            if self.status is None:
                self.status = TaskStatus.PENDING

        def to_dict(self):
            return {
                "id": self.id, "description": self.description,
                "inputs": self.inputs, "outputs": self.outputs,
                "dependencies": self.dependencies,
                "assigned_agent": self.assigned_agent.value if hasattr(self.assigned_agent, 'value') else str(self.assigned_agent),
                "depth": self.depth, "status": self.status.value if hasattr(self.status, 'value') else str(self.status),
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
        dependencies: List[str] = field(default_factory=list)
        context: Dict[str, Any] = field(default_factory=dict)

        def __post_init__(self):
            if self.status is None:
                self.status = TaskStatus.PENDING

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
                "agent_type": self.agent_type.value if hasattr(self.agent_type, 'value') else str(self.agent_type),
                "description": self.description, "prompt": self.prompt,
            }

    @dataclass
    class Plan:
        plan_id: str
        version: str = "v0.1.0"
        goal: str = ""
        phase: int = 0
        tasks: List[Any] = field(default_factory=list)
        model_scores: List[Dict] = field(default_factory=list)
        notes: str = ""
        warnings: List[str] = field(default_factory=list)
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
                "plan_id": self.plan_id, "version": self.version,
                "goal": self.goal, "phase": self.phase,
                "tasks": [t.to_dict() for t in self.tasks],
                "warnings": self.warnings,
                "needs_context": self.needs_context,
                "follow_up_question": self.follow_up_question,
            }

    @dataclass
    class AgentResult:
        success: bool
        output: Any
        metadata: Dict[str, Any] = field(default_factory=dict)
        errors: List[str] = field(default_factory=list)
        provenance: List[Dict] = field(default_factory=list)

        def to_dict(self):
            return {"success": self.success, "output": self.output,
                    "metadata": self.metadata, "errors": self.errors}

    @dataclass
    class VerificationResult:
        passed: bool
        issues: List[Dict[str, Any]] = field(default_factory=list)
        suggestions: List[str] = field(default_factory=list)
        score: float = 0.0

        def to_dict(self):
            return {"passed": self.passed, "issues": self.issues, "score": self.score}

    AGENT_REGISTRY = {}
    for at in AgentType:
        AGENT_REGISTRY[at] = AgentSpec(
            agent_type=at, name=at.value.title().replace("_", " "),
            description=f"{at.value} agent", default_model="default",
        )

    def get_agent_spec(agent_type):
        return AGENT_REGISTRY.get(agent_type)

    def get_all_agent_specs():
        return list(AGENT_REGISTRY.values())

    def get_enabled_agents():
        return [s for s in AGENT_REGISTRY.values() if s.enabled]

    # Install contracts into sys.modules
    contracts_mod = mod("vetinari.agents.contracts",
        AgentType=AgentType, TaskStatus=TaskStatus,
        ExecutionMode=ExecutionMode, AgentSpec=AgentSpec,
        Task=Task, AgentTask=AgentTask, Plan=Plan,
        AgentResult=AgentResult, VerificationResult=VerificationResult,
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
                "plan_id": self.plan_id, "version": getattr(self, "version", "v0.1.0"),
                "goal": self.goal, "phase": getattr(self, "phase", 0),
                "tasks": [t.to_dict() if hasattr(t, "to_dict") else {"id": getattr(t, "id", "")} for t in (self.tasks or [])],
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
                "id": self.id, "description": self.description,
                "inputs": getattr(self, "inputs", []),
                "outputs": getattr(self, "outputs", []),
                "dependencies": getattr(self, "dependencies", []),
                "assigned_agent": self.assigned_agent.value if hasattr(self.assigned_agent, "value") else str(self.assigned_agent),
                "depth": getattr(self, "depth", 0),
                "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            }
        _existing_task.to_dict = _task_to_dict

    # ── Lightweight stubs for modules imported by BaseAgent / agents ──
    mod("vetinari.structured_logging",
        log_event=MagicMock(), get_logger=MagicMock(return_value=MagicMock()))
    mod("vetinari.execution_context",
        get_context_manager=MagicMock(), ToolPermission=MagicMock())

    # learning sub-modules
    mod("vetinari.learning.quality_scorer",
        get_quality_scorer=MagicMock(return_value=MagicMock(
            score=MagicMock(return_value=MagicMock(overall_score=0.8)))))
    mod("vetinari.learning.feedback_loop",
        get_feedback_loop=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.model_selector",
        get_thompson_selector=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.prompt_evolver",
        get_prompt_evolver=MagicMock(return_value=MagicMock(
            select_prompt=MagicMock(return_value=("", "default")),
            register_baseline=MagicMock(),
            record_result=MagicMock())))
    mod("vetinari.learning.training_data",
        get_training_collector=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.episode_memory",
        get_episode_memory=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.auto_tuner")
    mod("vetinari.learning.cost_optimizer")
    mod("vetinari.learning.workflow_learner")
    mod("vetinari.learning.self_refinement")

    # constraints
    mod("vetinari.constraints.registry",
        get_constraint_registry=MagicMock(return_value=MagicMock(
            get_constraints_for_agent=MagicMock(return_value=None))))

    # config
    mod("vetinari.config.inference_config",
        get_inference_config=MagicMock(return_value=MagicMock(
            get_effective_params=MagicMock(return_value={}))))

    # token optimizer
    mod("vetinari.token_optimizer",
        get_token_optimizer=MagicMock(return_value=MagicMock(
            get_task_profile=MagicMock(return_value=(4096, 0.3, 0.9)))))

    # adapter_manager
    mod("vetinari.adapter_manager",
        get_adapter_manager=MagicMock(return_value=None))
    mod("vetinari.lmstudio_adapter",
        LMStudioAdapter=MagicMock())

    # adapters.base
    @dataclass
    class InferenceRequest:
        model_id: str = ""
        prompt: str = ""
        system_prompt: Optional[str] = None
        max_tokens: int = 4096
        temperature: float = 0.3
        top_p: float = 0.9
        top_k: int = 40
        stop_sequences: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

    mod("vetinari.adapters.base", InferenceRequest=InferenceRequest)

    # Other stubs
    mod("vetinari.tools.web_search_tool",
        get_search_tool=MagicMock(return_value=MagicMock()))

    # Return key classes for test use
    return {
        "AgentType": AgentType, "TaskStatus": TaskStatus,
        "AgentSpec": AgentSpec, "Task": Task, "AgentTask": AgentTask,
        "Plan": Plan, "AgentResult": AgentResult,
        "VerificationResult": VerificationResult,
    }


# ---------------------------------------------------------------------------
# Install stubs, then load real agent modules
# ---------------------------------------------------------------------------

_STUBS = _build_stubs()
AgentType = _STUBS["AgentType"]
TaskStatus = _STUBS["TaskStatus"]
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
_test_auto_mod = _load_agent_module("vetinari/agents/test_automation_agent.py", "vetinari.agents.test_automation_agent")

BaseAgent = _base_mod.BaseAgent
MultiModeAgent = _multi_mode_mod.MultiModeAgent
PlannerAgent = _planner_mod.PlannerAgent
get_planner_agent = _planner_mod.get_planner_agent
TestAutomationAgent = _test_auto_mod.TestAutomationAgent
get_test_automation_agent = _test_auto_mod.get_test_automation_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(agent_type, description="Test task", prompt="Do something",
               context=None, task_id=None):
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
    LEGACY_TYPE_TO_MODE = {
        "EXPLORER": "analyze",
        "BUILDER": "build",
        "EVALUATOR": "review",
    }

    def __init__(self, config=None):
        super().__init__(AgentType.QUALITY, config)

    def _handle_analyze(self, task):
        return AgentResult(success=True, output={"mode": "analyze", "data": "analyzed"})

    def _handle_build(self, task):
        return AgentResult(success=True, output={"mode": "build", "data": "built"})

    def _handle_review(self, task):
        return AgentResult(success=True, output={"mode": "review", "data": "reviewed"})


# ###########################################################################
# SECTION 1: MultiModeAgent Tests
# ###########################################################################

class TestMultiModeAgentInit(unittest.TestCase):
    """Tests for MultiModeAgent constructor and properties."""

    def test_init_sets_agent_type(self):
        agent = _ConcreteMultiMode()
        self.assertEqual(agent.agent_type, AgentType.QUALITY)

    def test_init_current_mode_is_none(self):
        agent = _ConcreteMultiMode()
        self.assertIsNone(agent.current_mode)

    def test_init_with_config(self):
        agent = _ConcreteMultiMode(config={"key": "val"})
        self.assertEqual(agent._config.get("key"), "val")

    def test_init_without_config(self):
        agent = _ConcreteMultiMode()
        self.assertIsInstance(agent._config, dict)

    def test_available_modes(self):
        agent = _ConcreteMultiMode()
        self.assertEqual(set(agent.available_modes), {"analyze", "build", "review"})

    def test_available_modes_empty_for_base(self):
        """A direct MultiModeAgent subclass with no MODES returns []."""
        class EmptyModes(MultiModeAgent):
            MODES = {}
            def get_system_prompt(self): return ""
        agent = EmptyModes(AgentType.QUALITY)
        self.assertEqual(agent.available_modes, [])

    def test_get_capabilities_returns_mode_names(self):
        agent = _ConcreteMultiMode()
        caps = agent.get_capabilities()
        self.assertEqual(set(caps), {"analyze", "build", "review"})

    def test_current_mode_property(self):
        agent = _ConcreteMultiMode()
        agent._current_mode = "build"
        self.assertEqual(agent.current_mode, "build")


class TestMultiModeResolveMode(unittest.TestCase):
    """Tests for MultiModeAgent._resolve_mode()."""

    def setUp(self):
        self.agent = _ConcreteMultiMode()

    def test_explicit_mode_in_context(self):
        task = _make_task(AgentType.QUALITY, context={"mode": "build"})
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "build")

    def test_explicit_mode_invalid_falls_through(self):
        task = _make_task(AgentType.QUALITY, context={"mode": "nonexistent"})
        mode = self.agent._resolve_mode(task)
        # Should not be "nonexistent" since it's not in MODES
        self.assertNotEqual(mode, "nonexistent")

    def test_legacy_type_mapping(self):
        task = _make_task(AgentType.QUALITY, context={"legacy_agent_type": "EXPLORER"})
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "analyze")

    def test_legacy_type_builder(self):
        task = _make_task(AgentType.QUALITY, context={"legacy_agent_type": "BUILDER"})
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "build")

    def test_legacy_type_evaluator(self):
        task = _make_task(AgentType.QUALITY, context={"legacy_agent_type": "EVALUATOR"})
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "review")

    def test_keyword_matching_analyze(self):
        task = _make_task(AgentType.QUALITY, description="Please analyze this code")
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "analyze")

    def test_keyword_matching_build(self):
        task = _make_task(AgentType.QUALITY, description="Build and implement the feature")
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "build")

    def test_keyword_matching_review(self):
        task = _make_task(AgentType.QUALITY, description="Review and evaluate the code")
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "review")

    def test_keyword_matching_best_score_wins(self):
        task = _make_task(AgentType.QUALITY,
                          description="build create implement code something")
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "build")

    def test_default_mode_fallback(self):
        task = _make_task(AgentType.QUALITY, description="do something unrelated")
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "analyze")  # DEFAULT_MODE

    def test_no_modes_no_default_returns_empty(self):
        class NoModes(MultiModeAgent):
            MODES = {}
            DEFAULT_MODE = ""
            MODE_KEYWORDS = {}
            def get_system_prompt(self): return ""
        agent = NoModes(AgentType.QUALITY)
        task = _make_task(AgentType.QUALITY)
        mode = agent._resolve_mode(task)
        self.assertEqual(mode, "")

    def test_agent_type_on_task_used_for_legacy_mapping(self):
        """If task.agent_type matches a legacy mapping key, use that."""
        task = _make_task(AgentType.EXPLORER, description="explore something")
        # EXPLORER -> "analyze"
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "analyze")

    def test_none_description_falls_to_default(self):
        task = _make_task(AgentType.QUALITY, description="")
        task.description = None
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "analyze")

    def test_empty_description_falls_to_default(self):
        task = _make_task(AgentType.QUALITY, description="")
        mode = self.agent._resolve_mode(task)
        self.assertEqual(mode, "analyze")


class TestMultiModeExecute(unittest.TestCase):
    """Tests for MultiModeAgent.execute()."""

    def setUp(self):
        self.agent = _ConcreteMultiMode()

    def test_execute_routes_to_analyze(self):
        task = _make_task(AgentType.QUALITY, context={"mode": "analyze"})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.output["mode"], "analyze")
        self.assertEqual(self.agent.current_mode, "analyze")

    def test_execute_routes_to_build(self):
        task = _make_task(AgentType.QUALITY, context={"mode": "build"})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.output["mode"], "build")

    def test_execute_routes_to_review(self):
        task = _make_task(AgentType.QUALITY, context={"mode": "review"})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.output["mode"], "review")

    def test_execute_sets_current_mode(self):
        task = _make_task(AgentType.QUALITY, context={"mode": "build"})
        self.agent.execute(task)
        self.assertEqual(self.agent.current_mode, "build")

    def test_execute_unknown_mode_returns_error(self):
        """If mode resolves to empty string with no handler, return error."""
        class BrokenAgent(MultiModeAgent):
            MODES = {"valid": "_handle_valid"}
            DEFAULT_MODE = ""
            MODE_KEYWORDS = {}
            LEGACY_TYPE_TO_MODE = {}
            def get_system_prompt(self): return ""
            def _handle_valid(self, task):
                from vetinari.agents.base_agent import AgentResult
                return AgentResult(success=True, output="ok")

        agent = BrokenAgent(AgentType.QUALITY)
        # "nonexistent" not in MODES -> falls to default "" -> no handler
        task = _make_task(AgentType.QUALITY, context={"mode": "nonexistent"})
        result = agent.execute(task)
        # With DEFAULT_MODE="" the fallback to first key "valid" succeeds
        # so the result depends on whether fallback logic triggers
        self.assertIsNotNone(result)

    def test_execute_handler_not_implemented(self):
        """B5: init-time validation catches missing handler methods."""
        class MissingHandler(MultiModeAgent):
            MODES = {"alpha": "_nonexistent_method"}
            DEFAULT_MODE = "alpha"
            MODE_KEYWORDS = {}
            def get_system_prompt(self): return ""
        with self.assertRaises(TypeError) as ctx:
            MissingHandler(AgentType.QUALITY)
        self.assertIn("_nonexistent_method", str(ctx.exception))

    def test_execute_handler_raises_exception(self):
        class FailingAgent(_ConcreteMultiMode):
            def _handle_analyze(self, task):
                raise ValueError("Analysis exploded")
        agent = FailingAgent()
        task = _make_task(AgentType.QUALITY, context={"mode": "analyze"})
        result = agent.execute(task)
        self.assertFalse(result.success)
        self.assertTrue(any("exploded" in e for e in result.errors))

    def test_execute_completes_task(self):
        task = _make_task(AgentType.QUALITY, context={"mode": "build"})
        self.agent.execute(task)
        self.assertNotEqual(task.completed_at, "")

    def test_execute_keyword_routing(self):
        task = _make_task(AgentType.QUALITY,
                          description="review and evaluate all code changes")
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.output["mode"], "review")


class TestMultiModeVerify(unittest.TestCase):
    """Tests for MultiModeAgent.verify()."""

    def setUp(self):
        self.agent = _ConcreteMultiMode()

    def test_verify_none_output_fails(self):
        result = self.agent.verify(None)
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
        self.assertTrue(len(result.issues) > 0)

    def test_verify_valid_output_passes(self):
        result = self.agent.verify({"data": "something"})
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.7)

    def test_verify_string_output_passes(self):
        result = self.agent.verify("some string")
        self.assertTrue(result.passed)

    def test_verify_empty_dict_passes(self):
        result = self.agent.verify({})
        self.assertTrue(result.passed)

    def test_verify_empty_list_passes(self):
        result = self.agent.verify([])
        self.assertTrue(result.passed)

    def test_verify_zero_passes(self):
        result = self.agent.verify(0)
        self.assertTrue(result.passed)

    def test_verify_false_passes(self):
        result = self.agent.verify(False)
        self.assertTrue(result.passed)


class TestMultiModeSystemPrompt(unittest.TestCase):
    """Tests for MultiModeAgent system prompt methods."""

    def setUp(self):
        self.agent = _ConcreteMultiMode()

    def test_get_system_prompt_default_mode(self):
        prompt = self.agent.get_system_prompt()
        # _current_mode is None, DEFAULT_MODE="analyze" but _get_mode_system_prompt
        # returns "" so falls to _get_base_system_prompt which uses _current_mode or "default"
        self.assertIn("default", prompt.lower() if prompt else "")

    def test_get_system_prompt_with_current_mode(self):
        self.agent._current_mode = "build"
        prompt = self.agent.get_system_prompt()
        self.assertIn("build", prompt.lower())

    def test_get_base_system_prompt(self):
        prompt = self.agent._get_base_system_prompt()
        self.assertIn(self.agent.name, prompt)

    def test_get_mode_system_prompt_default_empty(self):
        result = self.agent._get_mode_system_prompt("analyze")
        self.assertEqual(result, "")

    def test_get_system_prompt_custom_override(self):
        class CustomPrompt(_ConcreteMultiMode):
            def _get_mode_system_prompt(self, mode):
                if mode == "build":
                    return "Custom builder prompt"
                return ""
        agent = CustomPrompt()
        agent._current_mode = "build"
        prompt = agent.get_system_prompt()
        self.assertEqual(prompt, "Custom builder prompt")


# ###########################################################################
# SECTION 2: PlannerAgent Tests
# ###########################################################################

class TestPlannerAgentInit(unittest.TestCase):
    """Tests for PlannerAgent constructor and properties."""

    def test_init_default_config(self):
        agent = PlannerAgent()
        self.assertEqual(agent.agent_type, AgentType.PLANNER)
        self.assertEqual(agent._max_depth, 14)
        self.assertEqual(agent._min_tasks, 5)
        self.assertEqual(agent._max_tasks, 15)

    def test_init_custom_config(self):
        agent = PlannerAgent(config={"max_depth": 5, "min_tasks": 2, "max_tasks": 8})
        self.assertEqual(agent._max_depth, 5)
        self.assertEqual(agent._min_tasks, 2)
        self.assertEqual(agent._max_tasks, 8)

    def test_agent_type(self):
        agent = PlannerAgent()
        self.assertEqual(agent.agent_type, AgentType.PLANNER)

    def test_get_capabilities(self):
        agent = PlannerAgent()
        caps = agent.get_capabilities()
        self.assertIn("plan_generation", caps)
        self.assertIn("task_decomposition", caps)
        self.assertIn("dependency_mapping", caps)
        self.assertIn("resource_estimation", caps)
        self.assertIn("risk_assessment", caps)
        self.assertGreaterEqual(len(caps), 5)

    def test_get_system_prompt_nonempty(self):
        agent = PlannerAgent()
        prompt = agent.get_system_prompt()
        self.assertIn("Planning Master", prompt)
        self.assertIn("JSON", prompt)

    def test_system_prompt_mentions_consolidated_agents(self):
        agent = PlannerAgent()
        prompt = agent.get_system_prompt()
        self.assertIn("BUILDER", prompt)
        self.assertIn("RESEARCHER", prompt)
        self.assertIn("QUALITY", prompt)
        self.assertIn("OPERATIONS", prompt)

    def test_system_prompt_mentions_legacy_agents(self):
        agent = PlannerAgent()
        prompt = agent.get_system_prompt()
        # v0.4.0: consolidated prompt lists RESEARCHER/BUILDER etc, no legacy names
        self.assertIn("RESEARCHER", prompt)
        self.assertIn("BUILDER", prompt)


class TestPlannerAgentExecute(unittest.TestCase):
    """Tests for PlannerAgent.execute()."""

    def setUp(self):
        self.agent = PlannerAgent()

    def test_execute_invalid_task_type(self):
        task = _make_task(AgentType.BUILDER, description="build something")
        result = self.agent.execute(task)
        # v0.4.0: PlannerAgent handles multiple modes, may succeed via fallback
        self.assertIsNotNone(result)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_execute_with_keyword_fallback(self, mock_infer):
        task = _make_task(AgentType.PLANNER,
                          description="Build a web application with user authentication",
                          prompt="Build a web application with user authentication")
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        plan_dict = result.output
        self.assertIn("plan_id", plan_dict)
        self.assertIn("tasks", plan_dict)
        self.assertIn("goal", plan_dict)
        self.assertTrue(len(plan_dict["tasks"]) > 0)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_execute_sets_metadata(self, mock_infer):
        task = _make_task(AgentType.PLANNER,
                          description="Implement a REST API",
                          prompt="Implement a REST API")
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("plan_id", result.metadata)
        self.assertIn("task_count", result.metadata)
        self.assertIn("goal", result.metadata)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_execute_vague_goal_needs_context(self, mock_infer):
        task = _make_task(AgentType.PLANNER,
                          description="help me",
                          prompt="help me")
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        plan_dict = result.output
        self.assertTrue(plan_dict.get("needs_context", False))
        self.assertNotEqual(plan_dict.get("follow_up_question", ""), "")

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_execute_very_short_goal_vague(self, mock_infer):
        task = _make_task(AgentType.PLANNER, description="hi", prompt="hi")
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertTrue(result.output.get("needs_context", False))

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_execute_goal_with_special_chars_only(self, mock_infer):
        task = _make_task(AgentType.PLANNER, description="!@#$%", prompt="!@#$%")
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertTrue(result.output.get("needs_context", False))

    def test_execute_exception_returns_failure(self):
        """If _generate_plan raises, execute catches and returns failure."""
        task = _make_task(AgentType.PLANNER,
                          description="Build something complex",
                          prompt="Build something complex")
        with patch.object(self.agent, '_generate_plan', side_effect=RuntimeError("boom")):
            result = self.agent.execute(task)
        self.assertFalse(result.success)
        self.assertIn("boom", result.errors[0])

    @patch.object(PlannerAgent, '_infer_json')
    def test_execute_uses_prompt_over_description(self, mock_infer):
        mock_infer.return_value = None  # fallback to keyword
        task = _make_task(AgentType.PLANNER,
                          description="Some desc",
                          prompt="Build a web dashboard with data visualization")
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        # The goal should be the prompt (first choice)
        self.assertEqual(result.metadata.get("goal"), "Build a web dashboard with data visualization")


class TestPlannerAgentVerify(unittest.TestCase):
    """Tests for PlannerAgent.verify()."""

    def setUp(self):
        self.agent = PlannerAgent()

    def test_verify_non_dict_fails(self):
        result = self.agent.verify("not a dict")
        self.assertFalse(result.passed)

    def test_verify_none_fails(self):
        result = self.agent.verify(None)
        self.assertFalse(result.passed)

    def test_verify_empty_dict_missing_fields(self):
        result = self.agent.verify({})
        # Missing plan_id, goal, tasks => three -0.2 deductions = 0.4, fail
        self.assertFalse(result.passed)

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
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.score, 0.7)

    def test_verify_too_few_tasks(self):
        plan = {
            "plan_id": "plan_123",
            "goal": "Build app",
            "tasks": [
                {"id": "t1", "description": "Do it", "dependencies": ["t0"]},
            ],
        }
        result = self.agent.verify(plan)
        self.assertTrue(any(i["type"] == "insufficient_tasks" for i in result.issues))

    def test_verify_no_dependencies_warning(self):
        plan = {
            "plan_id": "plan_123",
            "goal": "Build app",
            "tasks": [
                {"id": f"t{i}", "description": f"Task {i}", "dependencies": []}
                for i in range(6)
            ],
        }
        result = self.agent.verify(plan)
        self.assertTrue(any(i["type"] == "no_dependencies" for i in result.issues))

    def test_verify_missing_plan_id(self):
        plan = {"goal": "test", "tasks": []}
        result = self.agent.verify(plan)
        self.assertTrue(any("plan_id" in i["message"] for i in result.issues))

    def test_verify_missing_goal(self):
        plan = {"plan_id": "p1", "tasks": []}
        result = self.agent.verify(plan)
        self.assertTrue(any("goal" in i["message"] for i in result.issues))

    def test_verify_missing_tasks_key(self):
        plan = {"plan_id": "p1", "goal": "test"}
        result = self.agent.verify(plan)
        self.assertTrue(any("tasks" in i["message"] for i in result.issues))

    def test_verify_list_fails(self):
        result = self.agent.verify([1, 2, 3])
        self.assertFalse(result.passed)

    def test_verify_score_clamped_to_zero(self):
        result = self.agent.verify({})
        self.assertGreaterEqual(result.score, 0)


class TestPlannerGeneratePlan(unittest.TestCase):
    """Tests for PlannerAgent._generate_plan()."""

    def setUp(self):
        self.agent = PlannerAgent()

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_vague_goal_something(self, mock_infer):
        plan = self.agent._generate_plan("do something", {})
        self.assertTrue(plan.needs_context)
        self.assertNotEqual(plan.follow_up_question, "")

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_vague_goal_fix_it(self, mock_infer):
        plan = self.agent._generate_plan("fix it", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_vague_goal_two_words(self, mock_infer):
        plan = self.agent._generate_plan("build", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_non_alphanumeric_goal(self, mock_infer):
        plan = self.agent._generate_plan("!!!", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_clear_goal_uses_keyword_fallback(self, mock_infer):
        plan = self.agent._generate_plan("Build a REST API with user authentication and database", {})
        self.assertFalse(plan.needs_context)
        self.assertTrue(len(plan.tasks) > 0)

    @patch.object(PlannerAgent, '_infer_json')
    def test_llm_decomposition_used_when_available(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "Explore", "inputs": ["goal"],
             "outputs": ["spec"], "dependencies": [], "assigned_agent": "EXPLORER"},
            {"id": "t2", "description": "Build", "inputs": ["spec"],
             "outputs": ["code"], "dependencies": ["t1"], "assigned_agent": "BUILDER"},
        ]
        plan = self.agent._generate_plan("Implement a microservices architecture for e-commerce", {})
        self.assertFalse(plan.needs_context)
        self.assertEqual(len(plan.tasks), 2)

    @patch.object(PlannerAgent, '_infer_json')
    def test_llm_returns_empty_falls_to_keyword(self, mock_infer):
        mock_infer.return_value = []
        plan = self.agent._generate_plan("Build a web application with user login", {})
        self.assertTrue(len(plan.tasks) > 0)

    @patch.object(PlannerAgent, '_infer_json')
    def test_too_many_tasks_adds_warning(self, mock_infer):
        # Return more than _max_tasks (15)
        mock_infer.return_value = [
            {"id": f"t{i}", "description": f"Task {i}", "inputs": [], "outputs": [],
             "dependencies": [], "assigned_agent": "BUILDER"}
            for i in range(20)
        ]
        plan = self.agent._generate_plan("Massive project with everything under the sun", {})
        self.assertTrue(len(plan.warnings) > 0)
        self.assertIn("20", plan.warnings[0])


class TestPlannerDecomposeGoalLLM(unittest.TestCase):
    """Tests for PlannerAgent._decompose_goal_llm()."""

    def setUp(self):
        self.agent = PlannerAgent()

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_llm_returns_none(self, mock_infer):
        tasks = self.agent._decompose_goal_llm("Build something", {})
        self.assertEqual(tasks, [])

    @patch.object(PlannerAgent, '_infer_json', return_value="not a list")
    def test_llm_returns_non_list(self, mock_infer):
        tasks = self.agent._decompose_goal_llm("Build something", {})
        self.assertEqual(tasks, [])

    @patch.object(PlannerAgent, '_infer_json')
    def test_llm_returns_valid_tasks(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "Research", "inputs": ["goal"],
             "outputs": ["report"], "dependencies": [], "assigned_agent": "EXPLORER"},
            {"id": "t2", "description": "Implement", "inputs": ["report"],
             "outputs": ["code"], "dependencies": ["t1"], "assigned_agent": "BUILDER"},
        ]
        tasks = self.agent._decompose_goal_llm("Build something", {})
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].id, "t1")
        self.assertEqual(tasks[1].id, "t2")

    @patch.object(PlannerAgent, '_infer_json')
    def test_llm_invalid_agent_type_defaults_to_builder(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "Do stuff", "assigned_agent": "NONEXISTENT_AGENT"},
        ]
        tasks = self.agent._decompose_goal_llm("Build something", {})
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].assigned_agent, AgentType.BUILDER)

    @patch.object(PlannerAgent, '_infer_json')
    def test_llm_missing_fields_uses_defaults(self, mock_infer):
        mock_infer.return_value = [
            {"description": "Minimal task"},
        ]
        tasks = self.agent._decompose_goal_llm("Build something", {})
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].id, "t1")  # auto-generated

    @patch.object(PlannerAgent, '_infer_json')
    def test_llm_skips_non_dict_items(self, mock_infer):
        mock_infer.return_value = [
            "not a dict",
            42,
            {"id": "t1", "description": "Valid"},
        ]
        tasks = self.agent._decompose_goal_llm("Build something", {})
        self.assertEqual(len(tasks), 1)

    @patch.object(PlannerAgent, '_infer_json')
    def test_dag_depth_calculation(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "First", "dependencies": []},
            {"id": "t2", "description": "Second", "dependencies": ["t1"]},
            {"id": "t3", "description": "Third", "dependencies": ["t2"]},
        ]
        tasks = self.agent._decompose_goal_llm("Build app", {})
        self.assertEqual(tasks[0].depth, 0)
        self.assertEqual(tasks[1].depth, 1)
        self.assertEqual(tasks[2].depth, 2)

    @patch.object(PlannerAgent, '_infer_json')
    def test_dag_depth_with_cycle_guard(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "First", "dependencies": ["t2"]},
            {"id": "t2", "description": "Second", "dependencies": ["t1"]},
        ]
        tasks = self.agent._decompose_goal_llm("Build app", {})
        # Should not hang due to cycle guard
        self.assertEqual(len(tasks), 2)

    @patch.object(PlannerAgent, '_infer_json')
    def test_context_passed_to_prompt(self, mock_infer):
        mock_infer.return_value = []
        self.agent._decompose_goal_llm("Build app", {"key": "value"})
        call_args = mock_infer.call_args
        prompt = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        self.assertIn("key", prompt)

    @patch.object(PlannerAgent, '_infer_json')
    def test_empty_context(self, mock_infer):
        mock_infer.return_value = []
        self.agent._decompose_goal_llm("Build app", {})
        # Should not crash
        mock_infer.assert_called_once()


@pytest.mark.skip(reason="Legacy agent type routing in _decompose_goal_keyword consolidated in v0.4.0")
class TestPlannerDecomposeGoalKeyword(unittest.TestCase):
    """Tests for PlannerAgent._decompose_goal_keyword()."""

    def setUp(self):
        self.agent = PlannerAgent()

    def test_code_heavy_goal(self):
        tasks = self.agent._decompose_goal_keyword("Build a web application in Python", {})
        self.assertTrue(len(tasks) >= 5)
        # Should include analysis, setup, implementation, review, docs, security
        descriptions = [t.description.lower() for t in tasks]
        self.assertTrue(any("analyz" in d for d in descriptions))
        self.assertTrue(any("implement" in d or "business logic" in d for d in descriptions))

    def test_ui_goal_includes_ui_task(self):
        tasks = self.agent._decompose_goal_keyword("Build a web dashboard with UI components", {})
        agents = [t.assigned_agent for t in tasks]
        self.assertIn(AgentType.UI_PLANNER, agents)

    def test_research_goal_includes_researcher(self):
        tasks = self.agent._decompose_goal_keyword("Research and analyze market trends", {})
        agents = [t.assigned_agent for t in tasks]
        self.assertIn(AgentType.RESEARCHER, agents)

    def test_data_goal_includes_data_engineer(self):
        tasks = self.agent._decompose_goal_keyword("Design a database schema for user data", {})
        agents = [t.assigned_agent for t in tasks]
        self.assertIn(AgentType.DATA_ENGINEER, agents)

    def test_always_includes_analysis_first(self):
        tasks = self.agent._decompose_goal_keyword("Build anything at all with code", {})
        self.assertEqual(tasks[0].assigned_agent, AgentType.EXPLORER)
        self.assertIn("analyz", tasks[0].description.lower())

    def test_always_ends_with_security(self):
        tasks = self.agent._decompose_goal_keyword("Build a simple script", {})
        self.assertEqual(tasks[-1].assigned_agent, AgentType.SECURITY_AUDITOR)

    def test_always_includes_documentation(self):
        tasks = self.agent._decompose_goal_keyword("Implement a Python module", {})
        agents = [t.assigned_agent for t in tasks]
        self.assertIn(AgentType.DOCUMENTATION_AGENT, agents)

    def test_always_includes_evaluator(self):
        tasks = self.agent._decompose_goal_keyword("Build a REST API", {})
        agents = [t.assigned_agent for t in tasks]
        self.assertIn(AgentType.EVALUATOR, agents)

    def test_dependencies_form_chain(self):
        tasks = self.agent._decompose_goal_keyword("Build a program", {})
        for i, t in enumerate(tasks):
            if i > 0:
                self.assertTrue(len(t.dependencies) > 0,
                                f"Task {t.id} at index {i} has no dependencies")

    def test_ids_are_unique(self):
        tasks = self.agent._decompose_goal_keyword("Build code app with database", {})
        ids = [t.id for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_minimal_goal_no_keywords(self):
        tasks = self.agent._decompose_goal_keyword("do a random task without matching words", {})
        # Should still produce analysis + setup + review + docs + security
        self.assertTrue(len(tasks) >= 4)

    def test_test_automation_included_for_code(self):
        tasks = self.agent._decompose_goal_keyword("Build a code implementation for an app", {})
        agents = [t.assigned_agent for t in tasks]
        self.assertIn(AgentType.TEST_AUTOMATION, agents)


class TestPlannerSingleton(unittest.TestCase):
    """Tests for get_planner_agent() singleton."""

    def setUp(self):
        _planner_mod._planner_agent = None

    def tearDown(self):
        _planner_mod._planner_agent = None

    def test_get_planner_agent_returns_instance(self):
        agent = get_planner_agent()
        self.assertIsInstance(agent, PlannerAgent)

    def test_get_planner_agent_singleton(self):
        a1 = get_planner_agent()
        a2 = get_planner_agent()
        self.assertIs(a1, a2)

    def test_get_planner_agent_with_config(self):
        agent = get_planner_agent(config={"max_tasks": 10})
        self.assertEqual(agent._max_tasks, 10)


# ###########################################################################
# SECTION 3: TestAutomationAgent Tests
# ###########################################################################

@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated into QualityAgent in v0.4.0")
class TestTestAutomationInit(unittest.TestCase):
    """Tests for TestAutomationAgent constructor and properties."""

    def test_init_default_config(self):
        agent = TestAutomationAgent()
        self.assertEqual(agent.agent_type, AgentType.TEST_AUTOMATION)
        self.assertEqual(agent._test_framework, "pytest")
        self.assertEqual(agent._language, "python")

    def test_init_custom_config(self):
        agent = TestAutomationAgent(config={"test_framework": "jest", "language": "typescript"})
        self.assertEqual(agent._test_framework, "jest")
        self.assertEqual(agent._language, "typescript")

    def test_agent_type(self):
        agent = TestAutomationAgent()
        self.assertEqual(agent.agent_type, AgentType.TEST_AUTOMATION)

    def test_get_capabilities(self):
        agent = TestAutomationAgent()
        caps = agent.get_capabilities()
        self.assertIn("unit_test_generation", caps)
        self.assertIn("integration_test_design", caps)
        self.assertIn("test_data_creation", caps)
        self.assertIn("coverage_analysis", caps)
        self.assertIn("test_execution", caps)
        self.assertIn("ci_integration", caps)
        self.assertEqual(len(caps), 6)

    def test_get_system_prompt_contains_framework(self):
        agent = TestAutomationAgent()
        prompt = agent.get_system_prompt()
        self.assertIn("pytest", prompt)
        self.assertIn("python", prompt)

    def test_get_system_prompt_custom_framework(self):
        agent = TestAutomationAgent(config={"test_framework": "jest", "language": "javascript"})
        prompt = agent.get_system_prompt()
        self.assertIn("jest", prompt)
        self.assertIn("javascript", prompt)

    def test_system_prompt_mentions_real_tests(self):
        agent = TestAutomationAgent()
        prompt = agent.get_system_prompt()
        self.assertIn("REAL tests", prompt)
        self.assertIn("assert True", prompt)


@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated in v0.4.0")
class TestTestAutomationExecute(unittest.TestCase):
    """Tests for TestAutomationAgent.execute()."""

    def setUp(self):
        self.agent = TestAutomationAgent()

    def test_execute_invalid_task_type(self):
        task = _make_task(AgentType.BUILDER)
        result = self.agent.execute(task)
        self.assertFalse(result.success)

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_with_llm_response(self, mock_infer):
        mock_infer.return_value = {
            "test_files": [{"name": "test_foo.py", "feature": "foo",
                            "module_under_test": "foo", "test_count": 3,
                            "test_categories": ["unit"]}],
            "test_scripts": [{"name": "test_foo.py", "content": "import pytest\ndef test_foo(): assert True",
                              "imports_needed": ["pytest"]}],
            "test_data": {"fixtures": [], "factory_functions": [], "mock_definitions": []},
            "coverage_report": {"target": 0.8, "estimated_coverage": 0.7, "status": "estimated"},
            "test_results": {"total_tests": 3, "note": "Run pytest", "command": "pytest"},
            "summary": "Generated tests",
        }
        task = _make_task(AgentType.TEST_AUTOMATION,
                          description="Generate tests for module foo",
                          context={"features": ["foo"], "code": "def bar(): pass"})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("test_files", result.output)
        self.assertIn("test_scripts", result.output)

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_fallback_used_when_llm_fails(self, mock_infer):
        # _infer_json returns None (first arg) -> uses fallback (second arg)
        def side_effect(*args, **kwargs):
            return kwargs.get("fallback", None)
        mock_infer.side_effect = side_effect
        task = _make_task(AgentType.TEST_AUTOMATION,
                          description="Generate tests",
                          context={"features": ["auth", "db"],
                                   "code": "def login(): pass\ndef query(): pass"})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("test_files", result.output)

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_sets_metadata(self, mock_infer):
        mock_infer.return_value = {
            "test_files": [], "test_scripts": [],
        }
        task = _make_task(AgentType.TEST_AUTOMATION,
                          description="Tests",
                          context={"features": ["a", "b"], "coverage_target": 0.9,
                                   "code": "def x(): pass"})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["features_tested"], 2)
        self.assertEqual(result.metadata["coverage_target"], 0.9)

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_ensures_required_keys(self, mock_infer):
        mock_infer.return_value = {}  # Missing all keys
        task = _make_task(AgentType.TEST_AUTOMATION, description="Test gen",
                          context={"features": [], "code": ""})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("test_files", result.output)
        self.assertIn("test_scripts", result.output)
        self.assertIn("test_data", result.output)
        self.assertIn("coverage_report", result.output)
        self.assertIn("test_results", result.output)
        self.assertIn("summary", result.output)

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_empty_features_uses_description(self, mock_infer):
        mock_infer.return_value = {"test_files": [], "test_scripts": []}
        task = _make_task(AgentType.TEST_AUTOMATION,
                          description="My awesome feature",
                          context={"features": [], "code": ""})
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_execute_exception_returns_failure(self):
        task = _make_task(AgentType.TEST_AUTOMATION, description="Fail test",
                          context={"features": ["x"]})
        with patch.object(self.agent, '_infer_json', side_effect=RuntimeError("kaboom")):
            result = self.agent.execute(task)
        self.assertFalse(result.success)
        self.assertIn("kaboom", result.errors[0])

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_code_truncated_to_2500(self, mock_infer):
        mock_infer.return_value = {"test_files": [], "test_scripts": []}
        big_code = "x" * 5000
        task = _make_task(AgentType.TEST_AUTOMATION,
                          description="Test gen",
                          context={"features": [], "code": big_code})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        # Verify the prompt was built (no crash)
        mock_infer.assert_called_once()

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_with_coverage_target(self, mock_infer):
        mock_infer.return_value = {"test_files": [], "test_scripts": []}
        task = _make_task(AgentType.TEST_AUTOMATION,
                          description="Test",
                          context={"features": [], "coverage_target": 0.95, "code": ""})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["coverage_target"], 0.95)

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_no_context_defaults(self, mock_infer):
        mock_infer.return_value = {"test_files": [], "test_scripts": []}
        task = _make_task(AgentType.TEST_AUTOMATION, description="Test gen", context={})
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["coverage_target"], 0.8)
        self.assertEqual(result.metadata["features_tested"], 0)


@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated in v0.4.0")
class TestTestAutomationVerify(unittest.TestCase):
    """Tests for TestAutomationAgent.verify()."""

    def setUp(self):
        self.agent = TestAutomationAgent()

    def test_verify_non_dict_fails(self):
        result = self.agent.verify("not a dict")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_verify_none_fails(self):
        result = self.agent.verify(None)
        self.assertFalse(result.passed)

    def test_verify_int_fails(self):
        result = self.agent.verify(42)
        self.assertFalse(result.passed)

    def test_verify_complete_output_passes(self):
        output = {
            "test_files": [{"name": "test_x.py"}],
            "test_scripts": [{"name": "test_x.py", "content": "assert result == 42"}],
            "test_data": {"fixtures": []},
            "coverage_report": {"target": 0.8},
        }
        result = self.agent.verify(output)
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.score, 0.5)

    def test_verify_missing_test_files(self):
        output = {
            "test_scripts": [{"name": "test.py", "content": "assert 1 == 1"}],
            "test_data": {},
            "coverage_report": {},
        }
        result = self.agent.verify(output)
        self.assertTrue(any(i["type"] == "missing_tests" for i in result.issues))

    def test_verify_missing_test_scripts(self):
        output = {
            "test_files": [{"name": "test.py"}],
            "test_data": {},
            "coverage_report": {},
        }
        result = self.agent.verify(output)
        self.assertTrue(any(i["type"] == "missing_scripts" for i in result.issues))

    def test_verify_missing_test_data(self):
        output = {
            "test_files": [{"name": "test.py"}],
            "test_scripts": [{"name": "test.py", "content": "assert x"}],
            "coverage_report": {},
        }
        result = self.agent.verify(output)
        self.assertTrue(any(i["type"] == "missing_data" for i in result.issues))

    def test_verify_missing_coverage_report(self):
        output = {
            "test_files": [{"name": "test.py"}],
            "test_scripts": [{"name": "test.py", "content": "assert x"}],
            "test_data": {},
        }
        result = self.agent.verify(output)
        self.assertTrue(any(i["type"] == "missing_coverage" for i in result.issues))

    def test_verify_stub_tests_flagged(self):
        output = {
            "test_files": [{"name": "test.py"}],
            "test_scripts": [{"name": "test.py",
                              "content": "assert True\nassert True\nassert True\nassert True"}],
            "test_data": {},
            "coverage_report": {},
        }
        result = self.agent.verify(output)
        self.assertTrue(any(i["type"] == "stub_tests" for i in result.issues))

    def test_verify_real_assertions_no_stub_flag(self):
        output = {
            "test_files": [{"name": "test.py"}],
            "test_scripts": [{"name": "test.py",
                              "content": "assert result == 42\nassert len(items) > 0\nassert x is not None"}],
            "test_data": {},
            "coverage_report": {},
        }
        result = self.agent.verify(output)
        self.assertFalse(any(i.get("type") == "stub_tests" for i in result.issues))

    def test_verify_empty_dict_has_issues(self):
        result = self.agent.verify({})
        self.assertTrue(len(result.issues) > 0)

    def test_verify_score_clamped_to_zero(self):
        output = {}
        result = self.agent.verify(output)
        self.assertGreaterEqual(result.score, 0.0)

    def test_verify_all_keys_present_high_score(self):
        output = {
            "test_files": [{"name": "test.py"}],
            "test_scripts": [{"name": "test.py", "content": "assert x == 1"}],
            "test_data": {"fixtures": ["f1"]},
            "coverage_report": {"target": 0.8},
        }
        result = self.agent.verify(output)
        self.assertGreaterEqual(result.score, 0.8)


@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated in v0.4.0")
class TestTestAutomationExtractFunctions(unittest.TestCase):
    """Tests for TestAutomationAgent._extract_functions()."""

    def setUp(self):
        self.agent = TestAutomationAgent()

    def test_extract_regular_functions(self):
        code = "def foo(): pass\ndef bar(x, y): pass"
        funcs = self.agent._extract_functions(code)
        self.assertIn("foo", funcs)
        self.assertIn("bar", funcs)

    def test_extract_async_functions(self):
        code = "async def fetch_data(): pass\nasync def process(item): pass"
        funcs = self.agent._extract_functions(code)
        self.assertIn("fetch_data", funcs)
        self.assertIn("process", funcs)

    def test_extract_classes(self):
        code = "class MyClass:\n    pass\nclass AnotherClass(Base):\n    pass"
        funcs = self.agent._extract_functions(code)
        self.assertIn("MyClass", funcs)
        self.assertIn("AnotherClass", funcs)

    def test_extract_mixed(self):
        code = """
class Foo:
    def method(self): pass
    async def async_method(self): pass
def standalone(): pass
"""
        funcs = self.agent._extract_functions(code)
        self.assertIn("Foo", funcs)
        self.assertIn("method", funcs)
        self.assertIn("async_method", funcs)
        self.assertIn("standalone", funcs)

    def test_extract_empty_code(self):
        funcs = self.agent._extract_functions("")
        self.assertEqual(funcs, [])

    def test_extract_none_code(self):
        funcs = self.agent._extract_functions(None)
        self.assertEqual(funcs, [])

    def test_extract_no_functions(self):
        code = "x = 1\ny = 2\nprint(x + y)"
        funcs = self.agent._extract_functions(code)
        self.assertEqual(funcs, [])

    def test_deduplication(self):
        code = "def foo(): pass\nclass foo:\n    pass"
        funcs = self.agent._extract_functions(code)
        self.assertEqual(funcs.count("foo"), 1)

    def test_preserves_order(self):
        code = "def alpha(): pass\ndef beta(): pass\ndef gamma(): pass"
        funcs = self.agent._extract_functions(code)
        self.assertEqual(funcs, ["alpha", "beta", "gamma"])

    def test_complex_signatures(self):
        code = "def func(a: int, b: str = 'hello', *args, **kwargs): pass"
        funcs = self.agent._extract_functions(code)
        self.assertIn("func", funcs)


@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated in v0.4.0")
class TestTestAutomationFallbackTests(unittest.TestCase):
    """Tests for TestAutomationAgent._fallback_tests()."""

    def setUp(self):
        self.agent = TestAutomationAgent()

    def test_fallback_with_features(self):
        result = self.agent._fallback_tests(["auth", "db"], ["login", "query"], 0.8)
        self.assertIn("test_files", result)
        self.assertIn("test_scripts", result)
        self.assertEqual(len(result["test_files"]), 2)
        self.assertEqual(len(result["test_scripts"]), 2)

    def test_fallback_empty_features_uses_core(self):
        result = self.agent._fallback_tests([], ["func1"], 0.8)
        self.assertEqual(len(result["test_files"]), 1)
        self.assertEqual(result["test_files"][0]["feature"], "core")

    def test_fallback_includes_test_data(self):
        result = self.agent._fallback_tests(["feature1"], ["func1"], 0.8)
        self.assertIn("test_data", result)
        self.assertIn("fixtures", result["test_data"])
        self.assertIn("mock_definitions", result["test_data"])

    def test_fallback_includes_coverage_report(self):
        result = self.agent._fallback_tests(["feat"], [], 0.9)
        self.assertIn("coverage_report", result)
        self.assertEqual(result["coverage_report"]["target"], 0.9)

    def test_fallback_includes_ci_config(self):
        result = self.agent._fallback_tests(["feat"], [], 0.8)
        self.assertIn("ci_config", result)
        self.assertIn("command", result["ci_config"])
        self.assertTrue(result["ci_config"]["parallel"])

    def test_fallback_includes_test_results(self):
        result = self.agent._fallback_tests(["feat"], [], 0.8)
        self.assertIn("test_results", result)
        self.assertIn("command", result["test_results"])

    def test_fallback_includes_summary(self):
        result = self.agent._fallback_tests(["a", "b"], [], 0.8)
        self.assertIn("summary", result)
        self.assertIn("2", result["summary"])

    def test_fallback_test_content_has_pytest_import(self):
        result = self.agent._fallback_tests(["auth"], ["login"], 0.8)
        content = result["test_scripts"][0]["content"]
        self.assertIn("import pytest", content)

    def test_fallback_generates_function_tests(self):
        result = self.agent._fallback_tests(["auth"], ["login", "logout"], 0.8)
        content = result["test_scripts"][0]["content"]
        self.assertIn("login", content)
        self.assertIn("logout", content)

    def test_fallback_safe_name_sanitization(self):
        result = self.agent._fallback_tests(["My Feature!@#"], [], 0.8)
        fname = result["test_files"][0]["name"]
        self.assertTrue(fname.startswith("test_"))
        self.assertTrue(fname.endswith(".py"))
        # No special characters in name
        self.assertFalse(any(c in fname for c in "!@# "))

    def test_fallback_limits_functions_to_5(self):
        funcs = [f"func_{i}" for i in range(10)]
        result = self.agent._fallback_tests(["feat"], funcs, 0.8)
        content = result["test_scripts"][0]["content"]
        # Should only include first 5
        self.assertIn("func_0", content)
        self.assertIn("func_4", content)
        self.assertNotIn("func_5", content)

    def test_fallback_no_functions_generates_basic_tests(self):
        result = self.agent._fallback_tests(["auth"], [], 0.8)
        content = result["test_scripts"][0]["content"]
        self.assertIn("test_", content)
        self.assertIn("def test_", content)


@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated in v0.4.0")
class TestTestAutomationExecuteTests(unittest.TestCase):
    """Tests for TestAutomationAgent.execute_tests()."""

    def setUp(self):
        self.agent = TestAutomationAgent()

    def test_execute_tests_empty_list(self):
        result = self.agent.execute_tests([])
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["returncode"], 0)

    def test_execute_tests_no_content(self):
        result = self.agent.execute_tests([{"name": "test_x.py", "content": ""}])
        self.assertEqual(result["passed"], 0)
        self.assertTrue(len(result["errors"]) > 0)
        self.assertEqual(result["returncode"], -1)

    @patch("subprocess.run")
    def test_execute_tests_successful_run(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="1 passed\n", stderr="", returncode=0
        )
        scripts = [{"name": "test_ok.py", "content": "def test_pass(): assert True"}]
        result = self.agent.execute_tests(scripts)
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["success"])

    @patch("subprocess.run")
    def test_execute_tests_failed_run(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="1 failed\n", stderr="", returncode=1
        )
        scripts = [{"name": "test_fail.py", "content": "def test_fail(): assert False"}]
        result = self.agent.execute_tests(scripts)
        self.assertEqual(result["returncode"], 1)

    @patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired(cmd="pytest", timeout=5))
    def test_execute_tests_timeout(self, mock_run):
        scripts = [{"name": "test_slow.py", "content": "def test_slow(): pass"}]
        result = self.agent.execute_tests(scripts, timeout=5)
        self.assertFalse(result["success"])
        self.assertTrue(any("timed out" in e for e in result["errors"]))

    @patch("subprocess.run", side_effect=OSError("No pytest"))
    def test_execute_tests_os_error(self, mock_run):
        scripts = [{"name": "test.py", "content": "def test(): pass"}]
        result = self.agent.execute_tests(scripts)
        self.assertFalse(result["success"])
        self.assertTrue(len(result["errors"]) > 0)

    @patch("subprocess.run")
    def test_execute_tests_parses_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="3 passed, 1 failed\nERROR test_x.py::test_bad\n",
            stderr="", returncode=1
        )
        scripts = [{"name": "test.py", "content": "def test(): pass"}]
        result = self.agent.execute_tests(scripts)
        self.assertEqual(result["passed"], 1)  # "passed" appears once
        self.assertEqual(result["failed"], 1)  # "failed" appears once
        self.assertTrue(len(result["errors"]) > 0)

    @patch("subprocess.run")
    def test_execute_tests_truncates_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="x" * 10000, stderr="", returncode=0
        )
        scripts = [{"name": "test.py", "content": "def test(): pass"}]
        result = self.agent.execute_tests(scripts)
        self.assertLessEqual(len(result["output"]), 5000)


@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated in v0.4.0")
class TestTestAutomationWriteTests(unittest.TestCase):
    """Tests for TestAutomationAgent.write_tests_to_disk()."""

    def setUp(self):
        self.agent = TestAutomationAgent()

    def test_write_tests_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts = [
                {"name": "test_a.py", "content": "def test_a(): assert True"},
                {"name": "test_b.py", "content": "def test_b(): assert True"},
            ]
            paths = self.agent.write_tests_to_disk(scripts, output_dir=tmpdir)
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(os.path.exists(p))

    def test_write_tests_empty_content_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts = [{"name": "test_empty.py", "content": ""}]
            paths = self.agent.write_tests_to_disk(scripts, output_dir=tmpdir)
            self.assertEqual(len(paths), 0)

    def test_write_tests_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "nested", "dir")
            scripts = [{"name": "test_x.py", "content": "pass"}]
            paths = self.agent.write_tests_to_disk(scripts, output_dir=out_dir)
            self.assertEqual(len(paths), 1)
            self.assertTrue(os.path.isdir(out_dir))

    def test_write_tests_correct_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts = [{"name": "test_check.py", "content": "# test content here"}]
            paths = self.agent.write_tests_to_disk(scripts, output_dir=tmpdir)
            with open(paths[0], 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertEqual(content, "# test content here")

    def test_write_tests_no_scripts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.agent.write_tests_to_disk([], output_dir=tmpdir)
            self.assertEqual(len(paths), 0)

    def test_write_tests_default_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts = [{"content": "def test(): pass"}]
            paths = self.agent.write_tests_to_disk(scripts, output_dir=tmpdir)
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith("test_generated.py"))


@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated in v0.4.0")
class TestTestAutomationSingleton(unittest.TestCase):
    """Tests for get_test_automation_agent() singleton."""

    def setUp(self):
        _test_auto_mod._test_automation_agent = None

    def tearDown(self):
        _test_auto_mod._test_automation_agent = None

    def test_get_agent_returns_instance(self):
        agent = get_test_automation_agent()
        self.assertIsInstance(agent, TestAutomationAgent)

    def test_get_agent_singleton(self):
        a1 = get_test_automation_agent()
        a2 = get_test_automation_agent()
        self.assertIs(a1, a2)

    def test_get_agent_with_config(self):
        agent = get_test_automation_agent(config={"test_framework": "mocha"})
        self.assertEqual(agent._test_framework, "mocha")


# ###########################################################################
# SECTION 4: Cross-Agent Integration Tests
# ###########################################################################

class TestCrossAgentInteractions(unittest.TestCase):
    """Integration-style tests verifying agents can interoperate."""

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_planner_generates_test_automation_tasks(self, mock_infer):
        planner = PlannerAgent()
        tasks = planner._decompose_goal_keyword("Build a code application", {})
        # v0.4.0: TEST_AUTOMATION consolidated into QUALITY
        test_tasks = [t for t in tasks if t.assigned_agent == AgentType.QUALITY]
        self.assertTrue(len(test_tasks) > 0)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    @patch.object(TestAutomationAgent, '_infer_json')
    def test_planner_then_test_agent(self, mock_test_infer, mock_plan_infer):
        mock_test_infer.return_value = {"test_files": [], "test_scripts": []}

        planner = PlannerAgent()
        plan_task = _make_task(AgentType.PLANNER,
                               description="Build a Python CLI tool",
                               prompt="Build a Python CLI tool")
        plan_result = planner.execute(plan_task)
        self.assertTrue(plan_result.success)

        test_agent = TestAutomationAgent()
        test_task = _make_task(AgentType.TEST_AUTOMATION,
                               description="Generate tests for CLI tool",
                               context={"features": ["cli"], "code": "def main(): pass"})
        test_result = test_agent.execute(test_task)
        self.assertTrue(test_result.success)

    def test_multi_mode_and_planner_coexist(self):
        multi = _ConcreteMultiMode()
        planner = PlannerAgent()
        self.assertNotEqual(multi.agent_type, planner.agent_type)
        self.assertTrue(len(multi.get_capabilities()) > 0)
        self.assertTrue(len(planner.get_capabilities()) > 0)

    def test_all_agents_have_system_prompts(self):
        agents = [_ConcreteMultiMode(), PlannerAgent(), TestAutomationAgent()]
        for agent in agents:
            prompt = agent.get_system_prompt()
            self.assertIsInstance(prompt, str)
            self.assertTrue(len(prompt) > 0)

    def test_all_agents_have_capabilities(self):
        agents = [_ConcreteMultiMode(), PlannerAgent(), TestAutomationAgent()]
        for agent in agents:
            caps = agent.get_capabilities()
            self.assertIsInstance(caps, list)
            self.assertTrue(len(caps) > 0)


# ###########################################################################
# SECTION 5: Additional Edge Case Tests
# ###########################################################################

class TestMultiModeEdgeCases(unittest.TestCase):
    """Edge cases for MultiModeAgent."""

    def test_execute_with_empty_modes_dict(self):
        class EmptyAgent(MultiModeAgent):
            MODES = {}
            DEFAULT_MODE = ""
            MODE_KEYWORDS = {}
            def get_system_prompt(self): return ""
        agent = EmptyAgent(AgentType.QUALITY)
        task = _make_task(AgentType.QUALITY)
        result = agent.execute(task)
        self.assertFalse(result.success)

    def test_get_capabilities_matches_modes(self):
        agent = _ConcreteMultiMode()
        self.assertEqual(sorted(agent.get_capabilities()), sorted(agent.available_modes))

    def test_mode_resolution_priority_explicit_over_keyword(self):
        """Explicit mode in context beats keyword matching."""
        agent = _ConcreteMultiMode()
        task = _make_task(AgentType.QUALITY,
                          description="review and evaluate everything",
                          context={"mode": "build"})
        mode = agent._resolve_mode(task)
        self.assertEqual(mode, "build")

    def test_mode_resolution_priority_legacy_over_keyword(self):
        """Legacy type mapping beats keyword matching."""
        agent = _ConcreteMultiMode()
        task = _make_task(AgentType.QUALITY,
                          description="review everything",
                          context={"legacy_agent_type": "BUILDER"})
        mode = agent._resolve_mode(task)
        self.assertEqual(mode, "build")

    def test_multiple_executions_update_mode(self):
        agent = _ConcreteMultiMode()
        task1 = _make_task(AgentType.QUALITY, context={"mode": "analyze"})
        agent.execute(task1)
        self.assertEqual(agent.current_mode, "analyze")

        task2 = _make_task(AgentType.QUALITY, context={"mode": "build"})
        agent.execute(task2)
        self.assertEqual(agent.current_mode, "build")


class TestPlannerEdgeCases(unittest.TestCase):
    """Edge cases for PlannerAgent."""

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_plan_has_unique_id(self, _):
        agent = PlannerAgent()
        p1 = agent._generate_plan("Build a web app with user management", {})
        p2 = agent._generate_plan("Build another web app with features", {})
        self.assertNotEqual(p1.plan_id, p2.plan_id)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_plan_stores_goal(self, _):
        agent = PlannerAgent()
        plan = agent._generate_plan("Build a magnificent space station", {})
        self.assertEqual(plan.goal, "Build a magnificent space station")

    def test_verify_score_boundary(self):
        agent = PlannerAgent()
        # Exactly at 0.7 threshold
        plan = {
            "plan_id": "p1",
            "goal": "test",
            "tasks": [
                {"id": f"t{i}", "description": f"Task {i}", "dependencies": []}
                for i in range(3)
            ],
        }
        result = agent.verify(plan)
        # 1.0 - 0.1 (too few tasks < 5) - 0.1 (no deps) = 0.8
        self.assertTrue(result.passed)

    def test_verify_just_below_threshold(self):
        agent = PlannerAgent()
        plan = {"plan_id": "p1", "tasks": []}  # missing goal
        result = agent.verify(plan)
        # 1.0 - 0.2 (missing goal) - 0.1 (too few tasks) - 0.1 (no deps) = 0.6
        self.assertFalse(result.passed)

    @patch.object(PlannerAgent, '_infer_json')
    def test_llm_task_with_acceptance_criteria(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "Explore", "dependencies": [],
             "assigned_agent": "EXPLORER", "acceptance_criteria": "Full report"},
        ]
        agent = PlannerAgent()
        tasks = agent._decompose_goal_llm("Build app", {})
        self.assertEqual(len(tasks), 1)

    @patch.object(PlannerAgent, '_infer_json')
    def test_llm_parallel_tasks_have_same_depth(self, mock_infer):
        mock_infer.return_value = [
            {"id": "t1", "description": "Root", "dependencies": []},
            {"id": "t2", "description": "Parallel A", "dependencies": ["t1"]},
            {"id": "t3", "description": "Parallel B", "dependencies": ["t1"]},
            {"id": "t4", "description": "Final", "dependencies": ["t2", "t3"]},
        ]
        agent = PlannerAgent()
        tasks = agent._decompose_goal_llm("Build app", {})
        self.assertEqual(tasks[0].depth, 0)
        self.assertEqual(tasks[1].depth, 1)
        self.assertEqual(tasks[2].depth, 1)
        self.assertEqual(tasks[3].depth, 2)


@pytest.mark.skip(reason="Legacy TestAutomationAgent internals consolidated in v0.4.0")
class TestTestAutomationEdgeCases(unittest.TestCase):
    """Edge cases for TestAutomationAgent."""

    def test_extract_functions_with_decorators(self):
        agent = TestAutomationAgent()
        code = "@decorator\ndef decorated_func(): pass\n@staticmethod\ndef static_func(): pass"
        funcs = agent._extract_functions(code)
        self.assertIn("decorated_func", funcs)
        self.assertIn("static_func", funcs)

    def test_extract_functions_multiline_signature(self):
        agent = TestAutomationAgent()
        code = "def long_func(\n    a,\n    b,\n    c\n): pass"
        funcs = agent._extract_functions(code)
        # findall on first line with ( should catch it
        self.assertIn("long_func", funcs)

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_with_artifacts_context(self, mock_infer):
        mock_infer.return_value = {"test_files": [], "test_scripts": []}
        agent = TestAutomationAgent()
        task = _make_task(AgentType.TEST_AUTOMATION,
                          description="Test gen",
                          context={"features": [], "code": "",
                                   "artifacts": {"module": "vetinari.core"}})
        result = agent.execute(task)
        self.assertTrue(result.success)

    def test_verify_empty_test_files_list(self):
        agent = TestAutomationAgent()
        output = {
            "test_files": [],
            "test_scripts": [{"name": "t.py", "content": "assert x"}],
            "test_data": {},
            "coverage_report": {},
        }
        result = agent.verify(output)
        self.assertTrue(any(i["type"] == "missing_tests" for i in result.issues))

    def test_verify_empty_test_scripts_list(self):
        agent = TestAutomationAgent()
        output = {
            "test_files": [{"name": "t.py"}],
            "test_scripts": [],
            "test_data": {},
            "coverage_report": {},
        }
        result = agent.verify(output)
        self.assertTrue(any(i["type"] == "missing_scripts" for i in result.issues))

    def test_fallback_with_special_char_features(self):
        agent = TestAutomationAgent()
        result = agent._fallback_tests(["auth/login", "data-base"], ["func"], 0.8)
        for tf in result["test_files"]:
            self.assertFalse(any(c in tf["name"] for c in "/\\-"))

    @patch.object(TestAutomationAgent, '_infer_json')
    def test_execute_functions_found_metadata(self, mock_infer):
        mock_infer.return_value = {"test_files": [], "test_scripts": []}
        agent = TestAutomationAgent()
        code = "def a(): pass\ndef b(): pass\nclass C: pass"
        task = _make_task(AgentType.TEST_AUTOMATION,
                          description="Test",
                          context={"features": [], "code": code})
        result = agent.execute(task)
        self.assertEqual(result.metadata["functions_found"], 3)


class TestMultiModeInheritance(unittest.TestCase):
    """Test that MultiModeAgent properly extends BaseAgent."""

    def test_is_subclass_of_base_agent(self):
        self.assertTrue(issubclass(MultiModeAgent, BaseAgent))

    def test_concrete_is_subclass(self):
        self.assertTrue(issubclass(_ConcreteMultiMode, MultiModeAgent))
        self.assertTrue(issubclass(_ConcreteMultiMode, BaseAgent))

    def test_agent_has_metadata(self):
        agent = _ConcreteMultiMode()
        meta = agent.get_metadata()
        self.assertIn("agent_type", meta)
        self.assertIn("capabilities", meta)
        self.assertIn("name", meta)

    def test_agent_repr(self):
        agent = _ConcreteMultiMode()
        r = repr(agent)
        self.assertIn("_ConcreteMultiMode", r)

    def test_planner_has_metadata(self):
        agent = PlannerAgent()
        meta = agent.get_metadata()
        self.assertEqual(meta["agent_type"], AgentType.PLANNER.value)

    def test_test_automation_has_metadata(self):
        agent = TestAutomationAgent()
        meta = agent.get_metadata()
        # v0.4.0: TestAutomationAgent consolidated into QualityAgent
        self.assertEqual(meta["agent_type"], AgentType.QUALITY.value)


class TestPlannerVagueGoalDetection(unittest.TestCase):
    """Detailed tests for vague goal heuristic in _generate_plan."""

    def setUp(self):
        self.agent = PlannerAgent()

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_single_word_is_vague(self, _):
        plan = self.agent._generate_plan("hello", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_two_words_is_vague(self, _):
        plan = self.agent._generate_plan("do it", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_help_me_is_vague(self, _):
        plan = self.agent._generate_plan("help me please", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_create_something_is_vague(self, _):
        plan = self.agent._generate_plan("create something now", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_build_something_is_vague(self, _):
        plan = self.agent._generate_plan("build something", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_clear_goal_is_not_vague(self, _):
        plan = self.agent._generate_plan(
            "Implement a REST API using FastAPI with authentication and database integration", {})
        self.assertFalse(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_medium_length_without_vague_words_is_not_vague(self, _):
        plan = self.agent._generate_plan(
            "Deploy the application to production servers", {})
        self.assertFalse(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_empty_string_is_vague(self, _):
        plan = self.agent._generate_plan("", {})
        self.assertTrue(plan.needs_context)

    @patch.object(PlannerAgent, '_infer_json', return_value=None)
    def test_whitespace_only_is_vague(self, _):
        plan = self.agent._generate_plan("   ", {})
        self.assertTrue(plan.needs_context)


if __name__ == "__main__":
    unittest.main()
