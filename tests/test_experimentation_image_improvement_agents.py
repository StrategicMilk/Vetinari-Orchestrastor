"""
Comprehensive tests for three Vetinari agents:
 - ExperimentationManagerAgent
 - ImageGeneratorAgent
 - ImprovementAgent

All external dependencies are stubbed so the tests run in isolation without
any real LLM, filesystem (where controllable), or third-party services.

NOTE: These agents were consolidated in v0.4.0:
  - ExperimentationManagerAgent -> OperationsAgent
  - ImageGeneratorAgent -> BuilderAgent
  - ImprovementAgent -> OperationsAgent
Tests that check legacy-specific internals are skipped.
"""

import sys
import types
import importlib.util
import os
import json

import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy agents consolidated into OperationsAgent/BuilderAgent in v0.4.0"
)
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


# ---------------------------------------------------------------------------
# Root of the worktree (where vetinari/ lives)
# ---------------------------------------------------------------------------
_WORTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_agent_module(rel_path: str, full_name: str):
    """Load a real source file and register it under full_name in sys.modules."""
    abs_path = os.path.join(_WORTREE_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(full_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Build all stubs BEFORE loading agent source files
# ---------------------------------------------------------------------------

def _build_stubs():
    """Stub every module that the agents (and BaseAgent) import."""

    def pkg(name):
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        rel = name.replace(".", os.sep)
        real_path = os.path.join(_WORTREE_ROOT, rel)
        m.__path__ = [real_path] if os.path.isdir(real_path) else []
        m.__package__ = name
        sys.modules[name] = m
        return m

    def mod(name, **kw):
        # Preserve existing real modules (e.g. vetinari.types loaded by earlier tests)
        if name in sys.modules:
            m = sys.modules[name]
        else:
            m = types.ModuleType(name)
            sys.modules[name] = m
        for k, v in kw.items():
            if not hasattr(m, k):
                setattr(m, k, v)
        return m

    # Top-level packages
    for p in [
        "vetinari", "vetinari.agents", "vetinari.adapters",
        "vetinari.learning", "vetinari.constraints",
        "vetinari.drift", "vetinari.config",
    ]:
        pkg(p)

    # ── Contracts ────────────────────────────────────────────────────────────
    from enum import Enum

    class AgentType(str, Enum):
        EXPERIMENTATION_MANAGER = "experimentation_manager"
        IMAGE_GENERATOR = "image_generator"
        IMPROVEMENT = "improvement"
        BUILDER = "builder"
        EVALUATOR = "evaluator"
        EXPLORER = "explorer"
        LIBRARIAN = "librarian"
        ORACLE = "oracle"
        RESEARCHER = "researcher"
        SYNTHESIZER = "synthesizer"
        UI_PLANNER = "ui_planner"

    class TaskStatus(str, Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"

    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional

    @dataclass
    class AgentSpec:
        name: str = ""
        description: str = ""
        default_model: str = "default"
        thinking_variant: str = "medium"

    @dataclass
    class AgentTask:
        task_id: str
        agent_type: Any
        description: str
        prompt: str
        status: Any = None
        result: Any = None
        error: str = ""
        dependencies: List[str] = field(default_factory=list)
        context: Dict[str, Any] = field(default_factory=dict)
        started_at: Optional[str] = None
        completed_at: Optional[str] = None

        def __post_init__(self):
            if self.status is None:
                self.status = TaskStatus.PENDING

    @dataclass
    class AgentResult:
        success: bool
        output: Any
        metadata: Dict[str, Any] = field(default_factory=dict)
        errors: List[str] = field(default_factory=list)

    @dataclass
    class VerificationResult:
        passed: bool
        issues: List[Any] = field(default_factory=list)
        suggestions: List[str] = field(default_factory=list)
        score: float = 0.0

    def get_agent_spec(agent_type):
        return AgentSpec(
            name=agent_type.value,
            description=f"Agent: {agent_type.value}",
            default_model="default",
            thinking_variant="medium",
        )

    contracts_mod = mod(
        "vetinari.agents.contracts",
        AgentType=AgentType,
        TaskStatus=TaskStatus,
        AgentTask=AgentTask,
        AgentResult=AgentResult,
        VerificationResult=VerificationResult,
        AgentSpec=AgentSpec,
        get_agent_spec=get_agent_spec,
    )

    # ── BaseAgent stub ───────────────────────────────────────────────────────
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
        def is_initialized(self):
            return self._initialized

        def get_metadata(self):
            return {
                "agent_type": self._agent_type.value,
                "name": self.name,
                "description": self.description,
                "default_model": self.default_model,
                "thinking_variant": self.thinking_variant,
                "capabilities": self.get_capabilities(),
                "initialized": self._initialized,
            }

        def initialize(self, context):
            self._context = context
            self._adapter_manager = context.get("adapter_manager")
            self._web_search = context.get("web_search")
            self._tool_registry = context.get("tool_registry")
            self._initialized = True

        def validate_task(self, task):
            return task.agent_type == self._agent_type

        def prepare_task(self, task):
            if not self._initialized:
                self.initialize({})
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

        def _log(self, level, message, **kwargs):
            pass  # suppress during tests

        def _infer(self, prompt, system_prompt=None, model_id=None,
                   max_tokens=4096, temperature=0.3, expect_json=False):
            return ""

        def _infer_json(self, prompt, system_prompt=None, model_id=None,
                        fallback=None, **kwargs):
            return fallback

        def _infer_with_fallback(self, prompt, fallback_fn=None, required_keys=None):
            if fallback_fn:
                return fallback_fn()
            return None

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

        def _incorporate_prior_results(self, task):
            ctx = getattr(task, "context", None) or {}
            return ctx.get("dependency_results", {})

        def get_system_prompt(self):
            return ""

        def get_capabilities(self):
            return []

        def verify(self, output):
            return VerificationResult(passed=True)

    mod("vetinari.agents.base_agent", BaseAgent=BaseAgent)

    # ── Adapters ─────────────────────────────────────────────────────────────
    mod("vetinari.adapters.base", LLMAdapter=MagicMock, InferenceRequest=MagicMock)

    # ── Constants ─────────────────────────────────────────────────────────────
    mod(
        "vetinari.constants",
        SD_WEBUI_HOST="http://localhost:7860",
        SD_WEBUI_ENABLED=False,
        SD_DEFAULT_WIDTH=512,
        SD_DEFAULT_HEIGHT=512,
        SD_DEFAULT_STEPS=20,
        SD_DEFAULT_CFG=7.0,
        TIMEOUT_MEDIUM=30,
        TIMEOUT_LONG=60,
        DEFAULT_MAX_TOKENS=4096,
    )

    # ── Learning subsystems ──────────────────────────────────────────────────
    _mock_selector = MagicMock()
    _mock_selector._arms = {}
    mod("vetinari.learning.model_selector",
        get_thompson_selector=lambda: _mock_selector)

    _mock_scorer = MagicMock()
    _mock_scorer.get_history.return_value = []
    mod("vetinari.learning.quality_scorer",
        get_quality_scorer=lambda: _mock_scorer)

    _mock_learner = MagicMock()
    _mock_learner.get_all_patterns.return_value = []
    mod("vetinari.learning.workflow_learner",
        get_workflow_learner=lambda: _mock_learner)

    _mock_tuner = MagicMock()
    _mock_tuner.run_cycle.return_value = []
    mod("vetinari.learning.auto_tuner",
        get_auto_tuner=lambda: _mock_tuner)

    _mock_evolver = MagicMock()
    _mock_evolver.select_prompt.return_value = ("", "default")
    mod("vetinari.learning.prompt_evolver",
        get_prompt_evolver=lambda: _mock_evolver)

    _mock_feedback = MagicMock()
    mod("vetinari.learning.feedback_loop",
        get_feedback_loop=lambda: _mock_feedback)

    _mock_training = MagicMock()
    mod("vetinari.learning.training_data",
        get_training_collector=lambda: _mock_training)

    _mock_ep_mem = MagicMock()
    mod("vetinari.learning.episode_memory",
        get_episode_memory=lambda: _mock_ep_mem)

    # ── Telemetry ─────────────────────────────────────────────────────────────
    # Load the REAL vetinari/telemetry.py (pure stdlib, no vetinari deps) so
    # later test files (e.g. test_dashboard_api.py) can import TelemetryCollector
    # etc. without hitting a hollow mock module.
    _load_agent_module("vetinari/telemetry.py", "vetinari.telemetry")

    # ── Misc optional imports ────────────────────────────────────────────────
    mod("vetinari.structured_logging", log_event=MagicMock())
    mod("vetinari.execution_context",
        get_context_manager=MagicMock(),
        ToolPermission=MagicMock())
    mod("vetinari.constraints.registry",
        get_constraint_registry=MagicMock())
    mod("vetinari.dynamic_model_router",
        get_model_router=MagicMock())
    mod("vetinari.token_optimizer",
        get_token_optimizer=MagicMock())
    mod("vetinari.config.inference_config",
        get_inference_config=MagicMock())
    mod("vetinari.lmstudio_adapter", LMStudioAdapter=MagicMock())
    mod("vetinari.adapter_manager", get_adapter_manager=MagicMock())

    return (
        AgentType, TaskStatus, AgentTask, AgentResult,
        VerificationResult, BaseAgent, get_agent_spec,
    )


(AgentType, TaskStatus, AgentTask, AgentResult,
 VerificationResult, BaseAgent, get_agent_spec) = _build_stubs()

# ---------------------------------------------------------------------------
# Load the real agent modules via filesystem (bypassing package resolution)
# ---------------------------------------------------------------------------
_em_mod = _load_agent_module(
    "vetinari/agents/experimentation_manager_agent.py",
    "vetinari.agents.experimentation_manager_agent",
)
_ig_mod = _load_agent_module(
    "vetinari/agents/image_generator_agent.py",
    "vetinari.agents.image_generator_agent",
)
_ia_mod = _load_agent_module(
    "vetinari/agents/improvement_agent.py",
    "vetinari.agents.improvement_agent",
)

ExperimentationManagerAgent = _em_mod.ExperimentationManagerAgent
get_experimentation_manager_agent = _em_mod.get_experimentation_manager_agent

ImageGeneratorAgent = _ig_mod.ImageGeneratorAgent
get_image_generator_agent = _ig_mod.get_image_generator_agent

ImprovementAgent = _ia_mod.ImprovementAgent
get_improvement_agent = _ia_mod.get_improvement_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(agent_type, task_id="t1", description="test task", prompt="do something",
               context=None):
    return AgentTask(
        task_id=task_id,
        agent_type=agent_type,
        description=description,
        prompt=prompt,
        context=context or {},
    )


def _patch_infer_json(agent, return_value):
    """Patch _infer_json to return a fixed value (ignoring the fallback kwarg)."""
    agent._infer_json = MagicMock(return_value=return_value)


def _patch_infer_json_passthrough(agent):
    """Patch _infer_json so it returns whatever the caller passes as fallback=.
    Use this when the production code does .setdefault() on the result."""
    def _impl(prompt, system_prompt=None, model_id=None, fallback=None, **kw):
        return fallback
    agent._infer_json = _impl


def _patch_infer(agent, return_value=""):
    agent._infer = MagicMock(return_value=return_value)


# ===========================================================================
#  SECTION 1 — ExperimentationManagerAgent (30+ tests)
# ===========================================================================

class TestExperimentationManagerAgentInit(unittest.TestCase):
    def setUp(self):
        self.agent = ExperimentationManagerAgent()

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, AgentType.EXPERIMENTATION_MANAGER)

    def test_default_config(self):
        self.assertEqual(self.agent._config, {})

    def test_custom_config_stored(self):
        a = ExperimentationManagerAgent(config={"foo": "bar"})
        self.assertEqual(a._config["foo"], "bar")

    def test_name_from_spec(self):
        self.assertEqual(self.agent.name, "experimentation_manager")

    def test_not_initialized_by_default(self):
        self.assertFalse(self.agent.is_initialized)

    def test_singleton_factory(self):
        _em_mod._experimentation_manager_agent = None
        a = get_experimentation_manager_agent()
        b = get_experimentation_manager_agent()
        self.assertIs(a, b)

    def test_singleton_factory_with_config(self):
        _em_mod._experimentation_manager_agent = None
        a = get_experimentation_manager_agent(config={"x": 1})
        self.assertEqual(a._config["x"], 1)


class TestExperimentationManagerAgentSystemPrompt(unittest.TestCase):
    def setUp(self):
        self.agent = ExperimentationManagerAgent()

    def test_system_prompt_is_string(self):
        sp = self.agent.get_system_prompt()
        self.assertIsInstance(sp, str)
        self.assertGreater(len(sp), 0)

    def test_system_prompt_contains_experimentation(self):
        sp = self.agent.get_system_prompt()
        self.assertIn("Experimentation", sp)

    def test_system_prompt_mentions_json(self):
        sp = self.agent.get_system_prompt()
        self.assertIn("JSON", sp)

    def test_system_prompt_mentions_experiment_log(self):
        sp = self.agent.get_system_prompt()
        self.assertIn("experiment_log", sp)

    def test_system_prompt_mentions_analysis(self):
        sp = self.agent.get_system_prompt()
        self.assertIn("analysis", sp)

    def test_system_prompt_mentions_source_field(self):
        sp = self.agent.get_system_prompt()
        self.assertIn("source", sp)


class TestExperimentationManagerAgentCapabilities(unittest.TestCase):
    def setUp(self):
        self.agent = ExperimentationManagerAgent()

    def test_capabilities_is_list(self):
        self.assertIsInstance(self.agent.get_capabilities(), list)

    def test_capabilities_not_empty(self):
        self.assertGreater(len(self.agent.get_capabilities()), 0)

    def test_experiment_planning_capability(self):
        self.assertIn("experiment_planning", self.agent.get_capabilities())

    def test_hypothesis_testing_capability(self):
        self.assertIn("hypothesis_testing", self.agent.get_capabilities())

    def test_result_recording_capability(self):
        self.assertIn("result_recording", self.agent.get_capabilities())

    def test_reproducibility_capability(self):
        self.assertIn("reproducibility_documentation", self.agent.get_capabilities())

    def test_configuration_tracking_capability(self):
        self.assertIn("configuration_tracking", self.agent.get_capabilities())

    def test_experiment_analysis_capability(self):
        self.assertIn("experiment_analysis", self.agent.get_capabilities())


class TestExperimentationManagerAgentExecute(unittest.TestCase):
    def setUp(self):
        self.agent = ExperimentationManagerAgent()

    def _good_task(self, context=None):
        return _make_task(AgentType.EXPERIMENTATION_MANAGER, context=context)

    def _wrong_task(self):
        return _make_task(AgentType.IMAGE_GENERATOR)

    def test_wrong_agent_type_returns_failure(self):
        result = self.agent.execute(self._wrong_task())
        self.assertFalse(result.success)
        self.assertIsNone(result.output)
        self.assertGreater(len(result.errors), 0)

    def test_success_with_llm_response(self):
        llm_response = {
            "experiment_log": [{"id": "exp_001", "name": "Test A"}],
            "configuration": {"model_variants": {"a": 1}},
            "results": {"metrics": [{"name": "quality", "value": 0.8}]},
            "reproducibility_plan": {"instructions": ["step 1"]},
            "analysis": {"insights": ["insight A"]},
            "summary": "Done",
        }
        _patch_infer_json(self.agent, llm_response)
        with patch.object(self.agent, "_collect_system_data", return_value={}):
            result = self.agent.execute(self._good_task())
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output)

    def test_output_has_required_keys_on_fallback(self):
        # _infer_json returns its fallback kwarg so management is a real dict
        _patch_infer_json_passthrough(self.agent)
        with patch.object(self.agent, "_collect_system_data", return_value={}):
            result = self.agent.execute(self._good_task())
        self.assertTrue(result.success)
        for key in ("experiment_log", "configuration", "results",
                    "reproducibility_plan", "analysis", "summary"):
            self.assertIn(key, result.output)

    def test_metadata_experiments_count(self):
        _patch_infer_json_passthrough(self.agent)
        ctx = {"experiments": [{"name": "A", "hypothesis": "H"}, {"name": "B"}]}
        with patch.object(self.agent, "_collect_system_data", return_value={}):
            result = self.agent.execute(self._good_task(context=ctx))
        self.assertEqual(result.metadata.get("experiments_count"), 2)

    def test_empty_experiments_context_succeeds(self):
        _patch_infer_json_passthrough(self.agent)
        with patch.object(self.agent, "_collect_system_data", return_value={}):
            result = self.agent.execute(self._good_task(context={}))
        self.assertTrue(result.success)

    def test_exception_in_collect_data_returns_failure(self):
        _patch_infer_json_passthrough(self.agent)
        with patch.object(self.agent, "_collect_system_data",
                          side_effect=RuntimeError("boom")):
            result = self.agent.execute(self._good_task())
        self.assertFalse(result.success)
        self.assertIn("boom", result.errors[0])

    def test_exception_in_infer_json_returns_failure(self):
        with patch.object(self.agent, "_collect_system_data", return_value={}):
            with patch.object(self.agent, "_infer_json",
                              side_effect=ValueError("bad json")):
                result = self.agent.execute(self._good_task())
        self.assertFalse(result.success)

    def test_metadata_has_real_data_sources(self):
        _patch_infer_json_passthrough(self.agent)
        fake_data = {"thompson_sampling": {"model_scores": {}}}
        with patch.object(self.agent, "_collect_system_data", return_value=fake_data):
            result = self.agent.execute(self._good_task())
        self.assertIn("real_data_sources", result.metadata)

    def test_output_summary_defaults(self):
        _patch_infer_json_passthrough(self.agent)
        with patch.object(self.agent, "_collect_system_data", return_value={}):
            result = self.agent.execute(self._good_task())
        self.assertIsInstance(result.output.get("summary"), str)


class TestExperimentationManagerAgentVerify(unittest.TestCase):
    def setUp(self):
        self.agent = ExperimentationManagerAgent()

    def test_verify_non_dict_fails(self):
        r = self.agent.verify("not a dict")
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.0)

    def test_verify_none_fails(self):
        r = self.agent.verify(None)
        self.assertFalse(r.passed)

    def test_verify_empty_dict_fails(self):
        r = self.agent.verify({})
        self.assertFalse(r.passed)

    def test_verify_full_valid_output_passes(self):
        output = {
            "experiment_log": [{"id": "e1"}],
            "configuration": {"seed": 42},
            "results": {"metrics": [{"name": "q", "value": 0.9}]},
            "reproducibility_plan": {"instructions": ["a"]},
            "analysis": {"insights": ["x"]},
            "summary": "ok",
        }
        r = self.agent.verify(output)
        self.assertTrue(r.passed)
        self.assertGreaterEqual(r.score, 0.5)

    def test_verify_missing_results_penalizes(self):
        output = {
            "experiment_log": [{"id": "e1"}],
            "analysis": {"insights": ["x"]},
        }
        r = self.agent.verify(output)
        self.assertLess(r.score, 1.0)

    def test_verify_missing_reproducibility_penalizes(self):
        output = {
            "experiment_log": [{"id": "e1"}],
            "results": {"metrics": []},
            "analysis": {"insights": []},
        }
        r = self.agent.verify(output)
        self.assertLess(r.score, 1.0)

    def test_verify_missing_analysis_penalizes(self):
        output = {
            "experiment_log": [{"id": "e1"}],
            "results": {"metrics": []},
            "reproducibility_plan": {"instructions": []},
        }
        r = self.agent.verify(output)
        self.assertLess(r.score, 1.0)

    def test_verify_returns_verification_result_type(self):
        r = self.agent.verify({})
        self.assertIsInstance(r, VerificationResult)

    def test_verify_score_non_negative(self):
        r = self.agent.verify({})
        self.assertGreaterEqual(r.score, 0.0)

    def test_verify_score_at_most_one(self):
        output = {
            "experiment_log": [{"id": "e1"}],
            "configuration": {},
            "results": {"metrics": []},
            "reproducibility_plan": {"instructions": []},
            "analysis": {"insights": []},
            "summary": "ok",
        }
        r = self.agent.verify(output)
        self.assertLessEqual(r.score, 1.0)


class TestExperimentationManagerAgentHelpers(unittest.TestCase):
    def setUp(self):
        self.agent = ExperimentationManagerAgent()

    def test_format_system_data_empty(self):
        result = self.agent._format_system_data({})
        self.assertIn("No system telemetry", result)

    def test_format_system_data_with_content(self):
        data = {"thompson_sampling": {"model_scores": {"gpt": {"mean": 0.8}}}}
        result = self.agent._format_system_data(data)
        self.assertIn("thompson_sampling", result)

    def test_fallback_management_no_experiments(self):
        fb = self.agent._fallback_management([], {})
        self.assertIn("experiment_log", fb)
        self.assertEqual(fb["experiment_log"], [])
        self.assertIn("analysis", fb)

    def test_fallback_management_with_experiments(self):
        exps = [{"name": "Exp A", "hypothesis": "H1", "tags": ["t1"]}]
        fb = self.agent._fallback_management(exps, {})
        self.assertEqual(len(fb["experiment_log"]), 1)
        self.assertEqual(fb["experiment_log"][0]["name"], "Exp A")

    def test_fallback_management_with_thompson_data(self):
        exps = [{"name": "E1"}]
        real = {"thompson_sampling": {"model_scores": {"m1": {"mean": 0.75}}}}
        fb = self.agent._fallback_management(exps, real)
        sources = [m.get("source") for m in fb["results"]["metrics"]]
        self.assertIn("measured", sources)

    def test_fallback_management_string_experiments(self):
        fb = self.agent._fallback_management(["plain string experiment"], {})
        self.assertEqual(len(fb["experiment_log"]), 1)

    def test_fallback_management_has_reproducibility(self):
        fb = self.agent._fallback_management([], {})
        self.assertIn("reproducibility_plan", fb)
        self.assertIn("instructions", fb["reproducibility_plan"])

    def test_fallback_management_has_summary(self):
        fb = self.agent._fallback_management([], {})
        self.assertIsInstance(fb.get("summary"), str)

    def test_collect_system_data_returns_dict(self):
        data = self.agent._collect_system_data()
        self.assertIsInstance(data, dict)


# ===========================================================================
#  SECTION 2 — ImageGeneratorAgent (30+ tests)
# ===========================================================================

class TestImageGeneratorAgentInit(unittest.TestCase):
    def _agent(self, **kwargs):
        return ImageGeneratorAgent(config={"output_dir": "/tmp/test_images", **kwargs})

    def test_agent_type(self):
        self.assertEqual(self._agent().agent_type, AgentType.BUILDER)

    def test_sd_host_default(self):
        self.assertEqual(self._agent()._sd_host, "http://localhost:7860")

    def test_sd_host_custom(self):
        a = self._agent(sd_host="http://myhost:1111")
        self.assertEqual(a._sd_host, "http://myhost:1111")

    def test_enabled_default_false(self):
        self.assertFalse(self._agent()._enabled)

    def test_enabled_override(self):
        self.assertTrue(self._agent(sd_enabled=True)._enabled)

    def test_default_width(self):
        self.assertEqual(self._agent()._default_width, 512)

    def test_default_height(self):
        self.assertEqual(self._agent()._default_height, 512)

    def test_custom_width_height(self):
        a = self._agent(width=1024, height=768)
        self.assertEqual(a._default_width, 1024)
        self.assertEqual(a._default_height, 768)

    def test_steps_default(self):
        self.assertEqual(self._agent()._steps, 20)

    def test_cfg_scale_default(self):
        self.assertAlmostEqual(self._agent()._cfg_scale, 7.0)

    def test_custom_steps(self):
        self.assertEqual(self._agent(steps=30)._steps, 30)

    def test_output_dir_is_path(self):
        self.assertIsInstance(self._agent()._output_dir, Path)

    def test_singleton_factory(self):
        _ig_mod._image_generator_agent = None
        a = get_image_generator_agent(config={"output_dir": "/tmp/test_images"})
        b = get_image_generator_agent()
        self.assertIs(a, b)


class TestImageGeneratorAgentSystemPromptAndCapabilities(unittest.TestCase):
    def setUp(self):
        self.agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})

    def test_system_prompt_is_string(self):
        sp = self.agent.get_system_prompt()
        self.assertIsInstance(sp, str)
        self.assertGreater(len(sp), 0)

    def test_system_prompt_mentions_image(self):
        self.assertIn("Image", self.agent.get_system_prompt())

    def test_system_prompt_mentions_sd_prompt(self):
        self.assertIn("sd_prompt", self.agent.get_system_prompt())

    def test_system_prompt_mentions_svg(self):
        self.assertIn("svg", self.agent.get_system_prompt().lower())

    def test_system_prompt_mentions_negative_prompt(self):
        self.assertIn("negative_prompt", self.agent.get_system_prompt())

    def test_capabilities_not_empty(self):
        self.assertGreater(len(self.agent.get_capabilities()), 0)

    def test_image_generation_capability(self):
        self.assertIn("image_generation", self.agent.get_capabilities())

    def test_svg_generation_capability(self):
        self.assertIn("svg_generation", self.agent.get_capabilities())

    def test_logo_design_capability(self):
        self.assertIn("logo_design", self.agent.get_capabilities())

    def test_ui_mockup_capability(self):
        self.assertIn("ui_mockup", self.agent.get_capabilities())

    def test_diagram_generation_capability(self):
        self.assertIn("diagram_generation", self.agent.get_capabilities())


class TestImageGeneratorAgentStyleDetection(unittest.TestCase):
    def setUp(self):
        self.agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})

    def test_detect_logo(self):
        self.assertEqual(self.agent._detect_style("company logo design"), "logo")

    def test_detect_brand(self):
        self.assertEqual(self.agent._detect_style("brand identity package"), "logo")

    def test_detect_icon(self):
        self.assertEqual(self.agent._detect_style("create an icon for the app"), "icon")

    def test_detect_ui_mockup_wireframe(self):
        self.assertEqual(self.agent._detect_style("wireframe for login screen"), "ui_mockup")

    def test_detect_ui_mockup_layout(self):
        self.assertEqual(self.agent._detect_style("dashboard layout design"), "ui_mockup")

    def test_detect_diagram(self):
        self.assertEqual(self.agent._detect_style("system architecture diagram"), "diagram")

    def test_detect_flowchart(self):
        self.assertEqual(self.agent._detect_style("create a flowchart"), "diagram")

    def test_detect_banner(self):
        self.assertEqual(self.agent._detect_style("marketing banner image"), "banner")

    def test_detect_background(self):
        self.assertEqual(self.agent._detect_style("dark background texture"), "background")

    def test_detect_wallpaper(self):
        self.assertEqual(self.agent._detect_style("desktop wallpaper pattern"), "background")

    def test_detect_default_logo(self):
        self.assertEqual(self.agent._detect_style("something completely random"), "logo")


class TestImageGeneratorAgentDefaultSizes(unittest.TestCase):
    def setUp(self):
        self.agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})

    def test_logo_size(self):
        self.assertEqual(self.agent._get_default_size("logo"), (512, 512))

    def test_icon_size(self):
        self.assertEqual(self.agent._get_default_size("icon"), (256, 256))

    def test_ui_mockup_size(self):
        self.assertEqual(self.agent._get_default_size("ui_mockup"), (1280, 720))

    def test_diagram_size(self):
        self.assertEqual(self.agent._get_default_size("diagram"), (1024, 768))

    def test_banner_size(self):
        self.assertEqual(self.agent._get_default_size("banner"), (1200, 400))

    def test_background_size(self):
        self.assertEqual(self.agent._get_default_size("background"), (1920, 1080))

    def test_unknown_style_defaults_to_512x512(self):
        self.assertEqual(self.agent._get_default_size("unknown"), (512, 512))

    def test_returns_tuple(self):
        size = self.agent._get_default_size("logo")
        self.assertIsInstance(size, tuple)
        self.assertEqual(len(size), 2)


class TestImageGeneratorAgentSvgFallback(unittest.TestCase):
    def setUp(self):
        self.agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})

    def test_minimal_svg_is_string(self):
        svg = self.agent._minimal_svg_placeholder("test logo", (512, 512))
        self.assertIsInstance(svg, str)

    def test_minimal_svg_starts_with_svg_tag(self):
        svg = self.agent._minimal_svg_placeholder("test", (512, 512))
        self.assertTrue(svg.strip().startswith("<svg"))

    def test_minimal_svg_ends_with_close_tag(self):
        svg = self.agent._minimal_svg_placeholder("test", (512, 512))
        self.assertTrue(svg.strip().endswith("</svg>"))

    def test_minimal_svg_contains_viewbox(self):
        svg = self.agent._minimal_svg_placeholder("test", (512, 512))
        self.assertIn("viewBox", svg)

    def test_minimal_svg_contains_width(self):
        svg = self.agent._minimal_svg_placeholder("test", (800, 600))
        self.assertIn("800", svg)

    def test_minimal_svg_contains_height(self):
        svg = self.agent._minimal_svg_placeholder("test", (800, 600))
        self.assertIn("600", svg)

    def test_minimal_svg_label_truncated_to_40(self):
        desc = "A" * 100
        svg = self.agent._minimal_svg_placeholder(desc, (512, 512))
        self.assertIn("A" * 40, svg)

    def test_generate_svg_fallback_extracts_svg_from_response(self):
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>'
        _patch_infer(self.agent, svg_content)
        result = self.agent._generate_svg_fallback("a logo", {"style_preset": "logo"})
        self.assertIn("<svg", result)

    def test_generate_svg_fallback_no_infer_output_uses_placeholder(self):
        _patch_infer(self.agent, "")
        result = self.agent._generate_svg_fallback("a logo", {"style_preset": "logo"})
        self.assertIn("<svg", result)

    def test_generate_svg_fallback_non_svg_response_uses_placeholder(self):
        _patch_infer(self.agent, "Sorry, cannot generate SVG.")
        result = self.agent._generate_svg_fallback("an icon", {"style_preset": "icon"})
        self.assertIn("<svg", result)

    def test_generate_svg_fallback_whole_response_is_svg(self):
        _patch_infer(self.agent, '<svg xmlns="http://www.w3.org/2000/svg"><g/></svg>')
        result = self.agent._generate_svg_fallback("test", {"style_preset": "logo"})
        self.assertTrue(result.strip().startswith("<svg"))


class TestImageGeneratorAgentExecute(unittest.TestCase):
    def setUp(self):
        self.agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})

    def _good_task(self, context=None):
        return _make_task(AgentType.IMAGE_GENERATOR,
                          description="a blue logo", prompt="create a blue logo",
                          context=context or {})

    def _wrong_task(self):
        return _make_task(AgentType.IMPROVEMENT)

    def test_wrong_agent_type_fails(self):
        result = self.agent.execute(self._wrong_task())
        self.assertFalse(result.success)

    def test_svg_fallback_when_sd_disabled(self):
        spec = {"style_preset": "logo", "description": "blue logo"}
        svg_code = '<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        with patch.object(self.agent, "_build_image_spec", return_value=spec):
            with patch.object(self.agent, "_generate_svg_fallback", return_value=svg_code):
                with patch.object(self.agent, "_save_svg",
                                  return_value=Path("/tmp/test_images/img_abc.svg")):
                    result = self.agent.execute(self._good_task())
        self.assertTrue(result.success)
        self.assertEqual(result.output["images"][0]["type"], "svg")

    def test_result_has_spec_key(self):
        spec = {"style_preset": "logo", "description": "x",
                "svg_fallback": '<svg xmlns="http://www.w3.org/2000/svg"></svg>'}
        with patch.object(self.agent, "_build_image_spec", return_value=spec):
            with patch.object(self.agent, "_save_svg", return_value=Path("/tmp/out.svg")):
                result = self.agent.execute(self._good_task())
        self.assertIn("spec", result.output)

    def test_result_has_sd_available_key(self):
        spec = {"style_preset": "logo", "description": "x",
                "svg_fallback": '<svg xmlns="http://www.w3.org/2000/svg"></svg>'}
        with patch.object(self.agent, "_build_image_spec", return_value=spec):
            with patch.object(self.agent, "_save_svg", return_value=Path("/tmp/out.svg")):
                result = self.agent.execute(self._good_task())
        self.assertIn("sd_available", result.output)

    def test_result_has_count_key(self):
        spec = {"style_preset": "logo", "description": "x"}
        svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        with patch.object(self.agent, "_build_image_spec", return_value=spec):
            with patch.object(self.agent, "_generate_svg_fallback", return_value=svg):
                with patch.object(self.agent, "_save_svg", return_value=Path("/tmp/out.svg")):
                    result = self.agent.execute(self._good_task())
        self.assertIn("count", result.output)

    def test_sd_enabled_failure_falls_back_to_svg(self):
        self.agent._enabled = True
        spec = {"style_preset": "logo", "description": "logo"}
        svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        with patch.object(self.agent, "_build_image_spec", return_value=spec):
            with patch.object(self.agent, "_generate_via_sd",
                              side_effect=RuntimeError("SD down")):
                with patch.object(self.agent, "_generate_svg_fallback", return_value=svg):
                    with patch.object(self.agent, "_save_svg",
                                      return_value=Path("/tmp/out.svg")):
                        result = self.agent.execute(self._good_task())
        self.assertTrue(result.success)
        self.assertIsNotNone(result.output.get("sd_error"))

    def test_no_images_returns_failure(self):
        spec = {"style_preset": "logo", "description": "x", "svg_fallback": None}
        with patch.object(self.agent, "_build_image_spec", return_value=spec):
            with patch.object(self.agent, "_generate_svg_fallback", return_value=""):
                result = self.agent.execute(self._good_task())
        self.assertFalse(result.success)

    def test_metadata_backend_svg_fallback(self):
        spec = {"style_preset": "logo", "description": "logo"}
        svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        with patch.object(self.agent, "_build_image_spec", return_value=spec):
            with patch.object(self.agent, "_generate_svg_fallback", return_value=svg):
                with patch.object(self.agent, "_save_svg", return_value=Path("/tmp/out.svg")):
                    result = self.agent.execute(self._good_task())
        self.assertEqual(result.metadata.get("backend"), "svg_fallback")

    def test_exception_in_build_spec_returns_failure(self):
        with patch.object(self.agent, "_build_image_spec",
                          side_effect=RuntimeError("spec error")):
            result = self.agent.execute(self._good_task())
        self.assertFalse(result.success)
        self.assertIn("spec error", result.errors[0])

    def test_spec_svg_fallback_used_directly(self):
        inline_svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>hi</text></svg>'
        spec = {"style_preset": "logo", "description": "x", "svg_fallback": inline_svg}
        with patch.object(self.agent, "_build_image_spec", return_value=spec):
            with patch.object(self.agent, "_save_svg", return_value=Path("/tmp/out.svg")):
                result = self.agent.execute(self._good_task())
        self.assertTrue(result.success)


class TestImageGeneratorAgentVerify(unittest.TestCase):
    def setUp(self):
        self.agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})

    def test_verify_non_dict_fails(self):
        r = self.agent.verify("bad")
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.0)

    def test_verify_no_images_fails(self):
        r = self.agent.verify({"images": []})
        self.assertFalse(r.passed)

    def test_verify_svg_with_code_passes(self):
        output = {"images": [{"type": "svg", "code": "<svg></svg>", "description": "x"}]}
        r = self.agent.verify(output)
        self.assertTrue(r.passed)

    def test_verify_svg_without_code_penalizes(self):
        output = {"images": [{"type": "svg", "code": "", "description": "x"}]}
        r = self.agent.verify(output)
        self.assertLess(r.score, 1.0)

    def test_verify_png_without_existing_file_penalizes(self):
        output = {"images": [{"type": "png", "path": "/nonexistent/path.png"}]}
        r = self.agent.verify(output)
        self.assertLess(r.score, 1.0)

    def test_verify_returns_verification_result(self):
        r = self.agent.verify({})
        self.assertIsInstance(r, VerificationResult)

    def test_verify_score_non_negative(self):
        output = {"images": [{"type": "svg", "code": ""}]}
        r = self.agent.verify(output)
        self.assertGreaterEqual(r.score, 0.0)

    def test_verify_no_images_key_fails(self):
        r = self.agent.verify({"spec": {}})
        self.assertFalse(r.passed)


class TestImageGeneratorAgentBuildSpec(unittest.TestCase):
    def setUp(self):
        self.agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})

    def test_build_spec_returns_dict(self):
        _patch_infer_json(self.agent, {})
        spec = self.agent._build_image_spec("a logo", {})
        self.assertIsInstance(spec, dict)

    def test_build_spec_sets_style_preset(self):
        _patch_infer_json(self.agent, {})
        spec = self.agent._build_image_spec("create a logo", {})
        self.assertIn("style_preset", spec)

    def test_build_spec_sets_sd_prompt(self):
        _patch_infer_json(self.agent, {})
        spec = self.agent._build_image_spec("a nice icon", {})
        self.assertIn("sd_prompt", spec)

    def test_build_spec_sets_negative_prompt(self):
        _patch_infer_json(self.agent, {})
        spec = self.agent._build_image_spec("a logo", {})
        self.assertIn("negative_prompt", spec)

    def test_build_spec_sets_dimensions(self):
        _patch_infer_json(self.agent, {})
        spec = self.agent._build_image_spec("a logo", {})
        self.assertIn("width", spec)
        self.assertIn("height", spec)

    def test_build_spec_sets_steps(self):
        _patch_infer_json(self.agent, {})
        spec = self.agent._build_image_spec("a logo", {})
        self.assertIn("steps", spec)

    def test_build_spec_sets_description(self):
        _patch_infer_json(self.agent, {})
        spec = self.agent._build_image_spec("my logo", {})
        self.assertEqual(spec["description"], "my logo")

    def test_build_spec_llm_override_preserved(self):
        llm_spec = {
            "style_preset": "diagram",
            "sd_prompt": "technical diagram",
            "negative_prompt": "bad",
            "width": 1024,
            "height": 768,
            "steps": 30,
            "description": "arch diagram",
        }
        _patch_infer_json(self.agent, llm_spec)
        spec = self.agent._build_image_spec("architecture diagram", {})
        self.assertEqual(spec["style_preset"], "diagram")
        self.assertEqual(spec["sd_prompt"], "technical diagram")


class TestImageGeneratorAgentStylePresets(unittest.TestCase):
    def setUp(self):
        self.agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})

    def test_all_style_presets_present(self):
        expected = {"logo", "icon", "ui_mockup", "diagram", "banner", "background"}
        self.assertEqual(set(self.agent.STYLE_PRESETS.keys()), expected)

    def test_negative_prompt_is_non_empty_string(self):
        self.assertIsInstance(self.agent.NEGATIVE_PROMPT, str)
        self.assertGreater(len(self.agent.NEGATIVE_PROMPT), 0)

    def test_logo_preset_has_vector_keywords(self):
        self.assertIn("vector", self.agent.STYLE_PRESETS["logo"].lower())

    def test_icon_preset_has_transparent_reference(self):
        self.assertIn("transparent", self.agent.STYLE_PRESETS["icon"].lower())


class TestImageGeneratorSvgSaving(unittest.TestCase):
    def test_save_svg_writes_file_and_returns_path(self):
        agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images_save"})
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle/></svg>'
        m = mock_open()
        with patch("builtins.open", m):
            path = agent._save_svg(svg, "test logo")
        self.assertIsInstance(path, Path)
        m.assert_called_once()

    def test_save_svg_filename_has_svg_extension(self):
        agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images_save"})
        svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        with patch("builtins.open", mock_open()):
            path = agent._save_svg(svg, "logo")
        self.assertTrue(str(path).endswith(".svg"))


# ===========================================================================
#  SECTION 3 — ImprovementAgent (30+ tests)
# ===========================================================================

class TestImprovementAgentInit(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, AgentType.IMPROVEMENT)

    def test_review_interval_default(self):
        self.assertEqual(self.agent._review_interval_hours, 1)

    def test_auto_apply_threshold_default(self):
        self.assertAlmostEqual(self.agent._auto_apply_threshold, 0.7)

    def test_custom_review_interval(self):
        a = ImprovementAgent(config={"review_interval_hours": 6})
        self.assertEqual(a._review_interval_hours, 6)

    def test_custom_auto_apply_threshold(self):
        a = ImprovementAgent(config={"auto_apply_threshold": 0.9})
        self.assertAlmostEqual(a._auto_apply_threshold, 0.9)

    def test_singleton_factory(self):
        _ia_mod._improvement_agent = None
        a = get_improvement_agent()
        b = get_improvement_agent()
        self.assertIs(a, b)

    def test_singleton_factory_with_config(self):
        _ia_mod._improvement_agent = None
        a = get_improvement_agent(config={"review_interval_hours": 3})
        self.assertEqual(a._review_interval_hours, 3)


class TestImprovementAgentSystemPrompt(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def test_system_prompt_is_string(self):
        sp = self.agent.get_system_prompt()
        self.assertIsInstance(sp, str)
        self.assertGreater(len(sp), 0)

    def test_system_prompt_mentions_improvement(self):
        self.assertIn("Improvement", self.agent.get_system_prompt())

    def test_system_prompt_mentions_performance(self):
        self.assertIn("performance", self.agent.get_system_prompt().lower())

    def test_system_prompt_mentions_recommendations(self):
        self.assertIn("recommendation", self.agent.get_system_prompt().lower())

    def test_system_prompt_mentions_risk(self):
        self.assertIn("risk", self.agent.get_system_prompt().lower())


class TestImprovementAgentCapabilities(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def test_capabilities_is_list(self):
        self.assertIsInstance(self.agent.get_capabilities(), list)

    def test_performance_analysis_capability(self):
        self.assertIn("performance_analysis", self.agent.get_capabilities())

    def test_model_recommendation_capability(self):
        self.assertIn("model_recommendation", self.agent.get_capabilities())

    def test_prompt_optimization_capability(self):
        self.assertIn("prompt_optimization", self.agent.get_capabilities())

    def test_cost_analysis_capability(self):
        self.assertIn("cost_analysis", self.agent.get_capabilities())

    def test_sla_monitoring_capability(self):
        self.assertIn("sla_monitoring", self.agent.get_capabilities())

    def test_anomaly_analysis_capability(self):
        self.assertIn("anomaly_analysis", self.agent.get_capabilities())

    def test_workflow_optimization_capability(self):
        self.assertIn("workflow_optimization", self.agent.get_capabilities())


class TestImprovementAgentExecute(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def _good_task(self, context=None):
        return _make_task(AgentType.IMPROVEMENT,
                          description="review system", prompt="run review",
                          context=context or {})

    def _wrong_task(self):
        return _make_task(AgentType.EXPERIMENTATION_MANAGER)

    def test_wrong_agent_type_fails(self):
        result = self.agent.execute(self._wrong_task())
        self.assertFalse(result.success)

    def test_successful_execution(self):
        with patch.object(self.agent, "_collect_metrics", return_value={}):
            with patch.object(self.agent, "_generate_recommendations", return_value=[]):
                with patch.object(self.agent, "_apply_safe_changes", return_value=[]):
                    result = self.agent.execute(self._good_task())
        self.assertTrue(result.success)

    def test_output_has_review_type_default_full(self):
        with patch.object(self.agent, "_collect_metrics", return_value={}):
            with patch.object(self.agent, "_generate_recommendations", return_value=[]):
                with patch.object(self.agent, "_apply_safe_changes", return_value=[]):
                    result = self.agent.execute(self._good_task())
        self.assertEqual(result.output["review_type"], "full")

    def test_output_uses_context_review_type(self):
        with patch.object(self.agent, "_collect_metrics", return_value={}):
            with patch.object(self.agent, "_generate_recommendations", return_value=[]):
                with patch.object(self.agent, "_apply_safe_changes", return_value=[]):
                    result = self.agent.execute(
                        self._good_task(context={"review_type": "partial"}))
        self.assertEqual(result.output["review_type"], "partial")

    def test_output_has_timestamp(self):
        with patch.object(self.agent, "_collect_metrics", return_value={}):
            with patch.object(self.agent, "_generate_recommendations", return_value=[]):
                with patch.object(self.agent, "_apply_safe_changes", return_value=[]):
                    result = self.agent.execute(self._good_task())
        self.assertIn("timestamp", result.output)

    def test_output_has_recommendations_key(self):
        recs = [{"type": "model_routing"}]
        with patch.object(self.agent, "_collect_metrics", return_value={}):
            with patch.object(self.agent, "_generate_recommendations", return_value=recs):
                with patch.object(self.agent, "_apply_safe_changes", return_value=[]):
                    result = self.agent.execute(self._good_task())
        self.assertEqual(len(result.output["recommendations"]), 1)

    def test_output_pending_approval_filters_high_risk(self):
        recs = [
            {"type": "t1", "risk": "high"},
            {"type": "t2", "risk": "low"},
        ]
        with patch.object(self.agent, "_collect_metrics", return_value={}):
            with patch.object(self.agent, "_generate_recommendations", return_value=recs):
                with patch.object(self.agent, "_apply_safe_changes", return_value=[]):
                    result = self.agent.execute(self._good_task())
        pending = result.output.get("pending_approval", [])
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["risk"], "high")

    def test_metadata_recommendations_count(self):
        recs = [{"type": "t"}, {"type": "u"}]
        with patch.object(self.agent, "_collect_metrics", return_value={}):
            with patch.object(self.agent, "_generate_recommendations", return_value=recs):
                with patch.object(self.agent, "_apply_safe_changes", return_value=[recs[0]]):
                    result = self.agent.execute(self._good_task())
        self.assertEqual(result.metadata["recommendations_count"], 2)
        self.assertEqual(result.metadata["auto_applied_count"], 1)

    def test_exception_in_collect_metrics_returns_failure(self):
        with patch.object(self.agent, "_collect_metrics",
                          side_effect=RuntimeError("db error")):
            result = self.agent.execute(self._good_task())
        self.assertFalse(result.success)
        self.assertIn("db error", result.errors[0])

    def test_output_has_metrics_summary(self):
        with patch.object(self.agent, "_collect_metrics", return_value={}):
            with patch.object(self.agent, "_generate_recommendations", return_value=[]):
                with patch.object(self.agent, "_apply_safe_changes", return_value=[]):
                    result = self.agent.execute(self._good_task())
        self.assertIn("metrics_summary", result.output)


class TestImprovementAgentVerify(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def test_verify_non_dict_fails(self):
        r = self.agent.verify("bad")
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 0.0)

    def test_verify_with_recommendations_passes_score_09(self):
        output = {"recommendations": [{"type": "model_routing"}]}
        r = self.agent.verify(output)
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.score, 0.9)

    def test_verify_no_recommendations_passes_score_05(self):
        output = {"recommendations": []}
        r = self.agent.verify(output)
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.score, 0.5)

    def test_verify_none_fails(self):
        r = self.agent.verify(None)
        self.assertFalse(r.passed)

    def test_verify_returns_verification_result(self):
        r = self.agent.verify({})
        self.assertIsInstance(r, VerificationResult)


class TestImprovementAgentRecommendations(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def test_generate_recommendations_empty_metrics(self):
        recs = self.agent._generate_recommendations({})
        self.assertIsInstance(recs, list)

    def test_underperforming_model_generates_routing_rec(self):
        metrics = {
            "model_arms": {"gpt:coding": {"mean": 0.3, "pulls": 15}},
            "avg_quality": 0.7,
        }
        recs = self.agent._generate_recommendations(metrics)
        self.assertIn("model_routing", [r.get("type") for r in recs])

    def test_low_quality_generates_prompt_evolution_rec(self):
        metrics = {"model_arms": {}, "avg_quality": 0.4}
        recs = self.agent._generate_recommendations(metrics)
        self.assertIn("prompt_evolution", [r.get("type") for r in recs])

    def test_low_success_rate_pattern_generates_workflow_rec(self):
        metrics = {
            "model_arms": {},
            "avg_quality": 0.7,
            "workflow_patterns": [
                {"domain": "coding", "sample_count": 10, "success_rate": 0.3}
            ],
        }
        recs = self.agent._generate_recommendations(metrics)
        self.assertIn("workflow_strategy", [r.get("type") for r in recs])

    def test_high_quality_no_routing_rec(self):
        metrics = {
            "model_arms": {"gpt:coding": {"mean": 0.9, "pulls": 20}},
            "avg_quality": 0.85,
        }
        recs = self.agent._generate_recommendations(metrics)
        routing = [r for r in recs if r.get("type") == "model_routing"]
        self.assertEqual(len(routing), 0)

    def test_insufficient_pulls_no_routing_rec(self):
        metrics = {
            "model_arms": {"gpt:coding": {"mean": 0.2, "pulls": 5}},  # < 10
            "avg_quality": 0.7,
        }
        recs = self.agent._generate_recommendations(metrics)
        routing = [r for r in recs if r.get("type") == "model_routing"]
        self.assertEqual(len(routing), 0)

    def test_routing_rec_has_required_fields(self):
        metrics = {
            "model_arms": {"gpt:coding": {"mean": 0.3, "pulls": 15}},
            "avg_quality": 0.7,
        }
        recs = self.agent._generate_recommendations(metrics)
        routing = [r for r in recs if r.get("type") == "model_routing"]
        self.assertEqual(len(routing), 1)
        rec = routing[0]
        for field in ("type", "priority", "risk", "action", "rationale", "auto_apply"):
            self.assertIn(field, rec)

    def test_apply_safe_changes_skips_high_risk(self):
        recs = [{"type": "model_routing", "risk": "high", "auto_apply": True,
                 "parameters": {"model_id": "gpt", "task_type": "coding"}}]
        applied = self.agent._apply_safe_changes(recs)
        self.assertEqual(len(applied), 0)

    def test_apply_safe_changes_skips_auto_apply_false(self):
        recs = [{"type": "model_routing", "risk": "low", "auto_apply": False,
                 "parameters": {}}]
        applied = self.agent._apply_safe_changes(recs)
        self.assertEqual(len(applied), 0)

    def test_apply_safe_changes_applies_low_risk(self):
        recs = [{"type": "model_routing", "risk": "low", "auto_apply": True,
                 "parameters": {"model_id": "gpt", "task_type": "coding",
                                "action": "reduce_weight"}}]
        with patch.object(self.agent, "_apply_model_routing_change"):
            applied = self.agent._apply_safe_changes(recs)
        self.assertEqual(len(applied), 1)

    def test_apply_safe_changes_ignores_unknown_type(self):
        recs = [{"type": "unknown_type", "risk": "low", "auto_apply": True,
                 "parameters": {}}]
        applied = self.agent._apply_safe_changes(recs)
        # unknown type does nothing — nothing applied
        self.assertEqual(len(applied), 0)


class TestImprovementAgentSummarizeMetrics(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def test_summarize_empty_metrics(self):
        summary = self.agent._summarize_metrics({})
        self.assertIsInstance(summary, dict)
        self.assertIn("model_count", summary)
        self.assertIn("total_model_pulls", summary)

    def test_summarize_with_arms(self):
        metrics = {
            "model_arms": {
                "gpt:coding": {"mean": 0.8, "pulls": 10},
                "gpt:general": {"mean": 0.7, "pulls": 5},
            }
        }
        summary = self.agent._summarize_metrics(metrics)
        self.assertEqual(summary["model_count"], 1)  # same model "gpt"
        self.assertEqual(summary["total_model_pulls"], 15)

    def test_summarize_avg_quality_default(self):
        summary = self.agent._summarize_metrics({})
        self.assertAlmostEqual(summary["avg_model_quality"], 0.7)

    def test_summarize_workflow_domains(self):
        metrics = {"workflow_patterns": [{"domain": "d1"}, {"domain": "d2"}]}
        summary = self.agent._summarize_metrics(metrics)
        self.assertEqual(summary["workflow_domains"], 2)

    def test_summarize_quality_samples(self):
        summary = self.agent._summarize_metrics({"quality_samples": 42})
        self.assertEqual(summary["quality_samples"], 42)

    def test_summarize_quality_samples_zero_by_default(self):
        summary = self.agent._summarize_metrics({})
        self.assertEqual(summary["quality_samples"], 0)


class TestImprovementAgentCollectMetrics(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def test_collect_metrics_returns_dict(self):
        self.assertIsInstance(self.agent._collect_metrics(), dict)

    def test_collect_metrics_has_model_arms(self):
        self.assertIn("model_arms", self.agent._collect_metrics())

    def test_collect_metrics_has_workflow_patterns(self):
        self.assertIn("workflow_patterns", self.agent._collect_metrics())


class TestImprovementAgentLLMRecommendations(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent()

    def test_llm_recommendations_empty_when_infer_returns_none(self):
        _patch_infer_json(self.agent, None)
        self.assertEqual(self.agent._generate_llm_recommendations({}, []), [])

    def test_llm_recommendations_from_valid_list(self):
        llm_recs = [
            {"type": "tooling", "priority": "low", "risk": "low",
             "action": "Add caching", "rationale": "Speed",
             "auto_apply": False, "parameters": {}}
        ]
        _patch_infer_json(self.agent, llm_recs)
        recs = self.agent._generate_llm_recommendations({}, [])
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["type"], "tooling")

    def test_llm_recommendations_caps_at_3(self):
        llm_recs = [{"type": "t"} for _ in range(10)]
        _patch_infer_json(self.agent, llm_recs)
        recs = self.agent._generate_llm_recommendations({}, [])
        self.assertLessEqual(len(recs), 3)

    def test_llm_recommendations_ignores_non_list(self):
        _patch_infer_json(self.agent, {"type": "t"})
        recs = self.agent._generate_llm_recommendations({}, [])
        self.assertEqual(recs, [])


# ===========================================================================
#  SECTION 4 — Cross-agent / integration tests
# ===========================================================================

class TestAgentValidation(unittest.TestCase):
    def test_experimentation_rejects_image_task(self):
        agent = ExperimentationManagerAgent()
        result = agent.execute(_make_task(AgentType.IMAGE_GENERATOR))
        self.assertFalse(result.success)

    def test_image_rejects_improvement_task(self):
        agent = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})
        result = agent.execute(_make_task(AgentType.IMPROVEMENT))
        self.assertFalse(result.success)

    def test_improvement_rejects_experimentation_task(self):
        agent = ImprovementAgent()
        result = agent.execute(_make_task(AgentType.EXPERIMENTATION_MANAGER))
        self.assertFalse(result.success)

    def test_each_agent_accepts_own_type(self):
        agents_and_types = [
            (ExperimentationManagerAgent(), AgentType.EXPERIMENTATION_MANAGER),
            (ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"}),
             AgentType.IMAGE_GENERATOR),
            (ImprovementAgent(), AgentType.IMPROVEMENT),
        ]
        for agent, atype in agents_and_types:
            task = _make_task(atype)
            self.assertTrue(agent.validate_task(task),
                            f"{agent.__class__.__name__} should accept its own type")


class TestAgentRepresentations(unittest.TestCase):
    def test_image_generator_has_style_presets(self):
        a = ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"})
        self.assertIsInstance(a.STYLE_PRESETS, dict)
        self.assertGreater(len(a.STYLE_PRESETS), 0)

    def test_all_agents_have_non_empty_capabilities(self):
        agents = [
            ExperimentationManagerAgent(),
            ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"}),
            ImprovementAgent(),
        ]
        for agent in agents:
            self.assertGreater(len(agent.get_capabilities()), 0,
                               f"{agent.__class__.__name__} capabilities empty")

    def test_all_agents_have_non_empty_system_prompts(self):
        agents = [
            ExperimentationManagerAgent(),
            ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"}),
            ImprovementAgent(),
        ]
        for agent in agents:
            sp = agent.get_system_prompt()
            self.assertIsInstance(sp, str)
            self.assertGreater(len(sp), 10,
                               f"{agent.__class__.__name__} system prompt too short")

    def test_all_agents_are_not_initialized_on_creation(self):
        agents = [
            ExperimentationManagerAgent(),
            ImageGeneratorAgent(config={"output_dir": "/tmp/test_images"}),
            ImprovementAgent(),
        ]
        for agent in agents:
            self.assertFalse(agent.is_initialized,
                             f"{agent.__class__.__name__} should not be initialized yet")


if __name__ == "__main__":
    unittest.main()
