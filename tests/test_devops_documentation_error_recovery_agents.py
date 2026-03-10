"""
Comprehensive tests for DevOpsAgent, DocumentationAgent, and ErrorRecoveryAgent.

Uses a minimal stub layer to avoid pulling in the full Vetinari dependency tree
while still importing the real agent classes.
"""

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy DevOpsAgent/DocumentationAgent/ErrorRecoveryAgent consolidated into ResearcherAgent/OperationsAgent in v0.4.0"
)


# ---------------------------------------------------------------------------
# Stub setup — must run before any vetinari imports
# ---------------------------------------------------------------------------

def _build_stubs():
    """Inject minimal stubs for all heavy vetinari dependencies."""

    def _pkg(name):
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        m.__path__ = []
        m.__package__ = name
        sys.modules[name] = m
        return m

    def _mod(name, **attrs):
        # Preserve existing real modules (e.g. vetinari.types loaded by earlier tests)
        if name in sys.modules:
            m = sys.modules[name]
        else:
            m = types.ModuleType(name)
            sys.modules[name] = m
        for k, v in attrs.items():
            if not hasattr(m, k):
                setattr(m, k, v)
        return m

    # ---- canonical types -----------------------------------------------
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

    _mod(
        "vetinari.types",
        AgentType=AgentType,
        TaskStatus=TaskStatus,
        ExecutionMode=ExecutionMode,
    )

    # ---- dataclasses used by agents ------------------------------------
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional

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

    def get_agent_spec(agent_type):
        return AgentSpec(
            agent_type=agent_type,
            name=agent_type.value.lower().replace("_", " ").title(),
            description="Test agent",
            default_model="test-model",
        )

    # ---- vetinari package stubs ----------------------------------------
    # We must register packages before mocking submodules, but we also need
    # the real on-disk path so that agent submodules (devops_agent.py, etc.)
    # can be found by the import system.
    import os as _os
    _worktree = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    def _pkg_real(name, real_subdir):
        m = types.ModuleType(name)
        real_path = _os.path.join(_worktree, *real_subdir.split("/"))
        m.__path__ = [real_path]
        m.__package__ = name
        sys.modules[name] = m
        return m

    _pkg_real("vetinari", "vetinari")
    _pkg_real("vetinari.agents", "vetinari/agents")
    _pkg_real("vetinari.adapters", "vetinari/adapters")
    _pkg_real("vetinari.learning", "vetinari/learning")
    _pkg_real("vetinari.constraints", "vetinari/constraints")
    _pkg_real("vetinari.config", "vetinari/config")

    _mod(
        "vetinari.types",
        AgentType=AgentType,
        TaskStatus=TaskStatus,
        ExecutionMode=ExecutionMode,
    )

    _mod(
        "vetinari.agents.contracts",
        AgentType=AgentType,
        TaskStatus=TaskStatus,
        AgentTask=AgentTask,
        AgentResult=AgentResult,
        AgentSpec=AgentSpec,
        VerificationResult=VerificationResult,
        get_agent_spec=get_agent_spec,
        get_all_agent_specs=lambda: [],
        get_enabled_agents=lambda: [],
    )

    _mod("vetinari.adapters.base", LLMAdapter=MagicMock, InferenceRequest=MagicMock)
    _mod("vetinari.adapter_manager", get_adapter_manager=MagicMock(return_value=None))
    _mod("vetinari.lmstudio_adapter", LMStudioAdapter=MagicMock)
    _mod("vetinari.execution_context", get_context_manager=MagicMock(), ToolPermission=MagicMock())
    _mod("vetinari.structured_logging", log_event=MagicMock())
    _mod("vetinari.token_optimizer", get_token_optimizer=MagicMock(return_value=None))
    _mod("vetinari.tools.web_search_tool", get_search_tool=MagicMock(return_value=None))

    # learning subsystem — all no-ops
    for sub in [
        "vetinari.learning.quality_scorer",
        "vetinari.learning.feedback_loop",
        "vetinari.learning.model_selector",
        "vetinari.learning.prompt_evolver",
        "vetinari.learning.training_data",
        "vetinari.learning.episode_memory",
    ]:
        _mod(sub, **{k: MagicMock() for k in [
            "get_quality_scorer", "get_feedback_loop", "get_thompson_selector",
            "get_prompt_evolver", "get_training_collector", "get_episode_memory",
        ]})

    _mod("vetinari.constraints.registry", get_constraint_registry=MagicMock(return_value=None))
    _mod("vetinari.config.inference_config", get_inference_config=MagicMock(return_value=None))

    # ---- BaseAgent stub -------------------------------------------------
    # We must stub base_agent BEFORE importing the real agents so they
    # pick up our lightweight version (which skips the heavy initialize path).

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

        # Minimal property mirrors
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
            pass  # silence logging in tests

        def _infer(self, prompt, system_prompt=None, model_id=None,
                   max_tokens=4096, temperature=0.3, expect_json=False):
            """Default: return empty string (simulates no adapter)."""
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

    _mod("vetinari.agents.base_agent", BaseAgent=BaseAgent)

    return AgentType, AgentTask, AgentResult, VerificationResult, BaseAgent


# Run stub setup once at module load
_AgentType, _AgentTask, _AgentResult, _VerificationResult, _BaseAgent = _build_stubs()
# Use the AgentType actually in sys.modules (may be the real one from an earlier test)
_AgentType = sys.modules["vetinari.types"].AgentType

# Now import the real agent classes
from vetinari.agents.devops_agent import DevOpsAgent
from vetinari.agents.documentation_agent import DocumentationAgent
from vetinari.agents.error_recovery_agent import ErrorRecoveryAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _devops_task(**ctx_overrides):
    """Return a valid AgentTask targeting DevOpsAgent."""
    return _AgentTask(
        task_id="devops-001",
        agent_type=_AgentType.DEVOPS,
        description="Set up CI/CD for a Python microservice",
        prompt="Deploy this service.",
        context=ctx_overrides,
    )


def _doc_task(**ctx_overrides):
    """Return a valid AgentTask targeting DocumentationAgent."""
    return _AgentTask(
        task_id="doc-001",
        agent_type=_AgentType.DOCUMENTATION_AGENT,
        description="Generate API docs for Vetinari",
        prompt="Write docs.",
        context=ctx_overrides,
    )


def _err_task(**ctx_overrides):
    """Return a valid AgentTask targeting ErrorRecoveryAgent."""
    return _AgentTask(
        task_id="err-001",
        agent_type=_AgentType.ERROR_RECOVERY,
        description="ConnectionRefusedError connecting to Redis",
        prompt="Recover from this error.",
        context=ctx_overrides,
    )


GOOD_DEVOPS_JSON = json.dumps({
    "ci_pipeline": {"platform": "github-actions", "stages": []},
    "containerisation": {"dockerfile": "FROM python:3.11"},
    "deployment_strategy": {"type": "blue-green"},
    "infrastructure": {"iac_tool": "terraform"},
    "monitoring": {"metrics_to_track": []},
    "runbooks": [],
    "summary": "Done",
})

GOOD_DOC_JSON = json.dumps({
    "docs_manifest": {"title": "Test Docs", "version": "1.0", "generated_at": "", "sections": [], "doc_type": "comprehensive"},
    "pages": [{"title": "Overview", "path": "overview.md", "section": "intro", "content": "# Overview", "toc": []}],
    "api_docs": {"title": "API", "endpoints": [{"path": "/status", "method": "GET"}]},
    "user_guides": [{"title": "Guide", "audience": "developer", "content": "# Guide", "prerequisites": [], "steps": []}],
    "references": {},
    "change_log": [{"version": "1.0", "date": "2025-01-01", "type": "feature", "changes": []}],
    "summary": "Complete",
})

GOOD_ERR_JSON = json.dumps({
    "root_cause": "Redis service not running",
    "errors_identified": [{"type": "connection_refused", "category": "network", "severity": "high", "location": "redis_client.py"}],
    "immediate_actions": [{"priority": 1, "action": "Restart Redis", "command": "systemctl restart redis", "expected_result": "Redis running"}],
    "recovery_strategies": [{"name": "Retry", "type": "retry", "description": "Retry with backoff", "implementation": "tenacity", "code_snippet": ""}],
    "retry_policy": {"max_attempts": 3, "initial_delay_ms": 1000, "backoff_multiplier": 2.0, "max_delay_ms": 30000, "jitter": True},
    "circuit_breaker": {"enabled": True, "failure_threshold": 5, "timeout_seconds": 60, "half_open_requests": 1},
    "fallback_plan": "Use in-memory cache",
    "preventive_measures": ["Add health checks"],
    "post_mortem_summary": "Redis was unreachable",
    "monitoring_recommendations": ["Alert on redis down"],
})


# ===========================================================================
# DevOpsAgent Tests
# ===========================================================================

class TestDevOpsAgentInit(unittest.TestCase):

    def test_default_platform_is_docker(self):
        agent = DevOpsAgent()
        self.assertEqual(agent._platform, "docker")

    def test_custom_platform_kubernetes(self):
        agent = DevOpsAgent(config={"platform": "kubernetes"})
        self.assertEqual(agent._platform, "kubernetes")

    def test_custom_platform_fargate(self):
        agent = DevOpsAgent(config={"platform": "fargate"})
        self.assertEqual(agent._platform, "fargate")

    def test_agent_type_is_devops(self):
        agent = DevOpsAgent()
        self.assertEqual(agent.agent_type, _AgentType.DEVOPS)

    def test_config_stored(self):
        agent = DevOpsAgent(config={"platform": "nomad", "extra": 42})
        self.assertEqual(agent._config["extra"], 42)

    def test_no_config_creates_empty_dict(self):
        agent = DevOpsAgent()
        self.assertIsInstance(agent._config, dict)


class TestDevOpsAgentSystemPrompt(unittest.TestCase):

    def test_system_prompt_non_empty(self):
        agent = DevOpsAgent()
        prompt = agent.get_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 10)

    def test_system_prompt_contains_platform(self):
        agent = DevOpsAgent(config={"platform": "kubernetes"})
        prompt = agent.get_system_prompt()
        self.assertIn("kubernetes", prompt)

    def test_system_prompt_contains_docker_default(self):
        agent = DevOpsAgent()
        prompt = agent.get_system_prompt()
        self.assertIn("docker", prompt)

    def test_system_prompt_contains_ci_pipeline(self):
        agent = DevOpsAgent()
        self.assertIn("ci_pipeline", agent.get_system_prompt())

    def test_system_prompt_contains_deployment_strategy(self):
        agent = DevOpsAgent()
        self.assertIn("deployment_strategy", agent.get_system_prompt())

    def test_system_prompt_contains_monitoring(self):
        agent = DevOpsAgent()
        self.assertIn("monitoring", agent.get_system_prompt())


class TestDevOpsAgentCapabilities(unittest.TestCase):

    def test_capabilities_non_empty(self):
        agent = DevOpsAgent()
        caps = agent.get_capabilities()
        self.assertIsInstance(caps, list)
        self.assertGreater(len(caps), 0)

    def test_capabilities_contains_ci_cd(self):
        caps = DevOpsAgent().get_capabilities()
        self.assertIn("ci_cd_pipeline_design", caps)

    def test_capabilities_contains_containerisation(self):
        self.assertIn("containerisation", DevOpsAgent().get_capabilities())

    def test_capabilities_contains_iac(self):
        self.assertIn("infrastructure_as_code", DevOpsAgent().get_capabilities())

    def test_capabilities_contains_deployment_strategy(self):
        self.assertIn("deployment_strategy", DevOpsAgent().get_capabilities())

    def test_capabilities_contains_monitoring(self):
        self.assertIn("monitoring_setup", DevOpsAgent().get_capabilities())

    def test_capabilities_contains_runbook(self):
        self.assertIn("runbook_creation", DevOpsAgent().get_capabilities())

    def test_capabilities_contains_security(self):
        self.assertIn("security_hardening", DevOpsAgent().get_capabilities())

    def test_capabilities_all_strings(self):
        for cap in DevOpsAgent().get_capabilities():
            self.assertIsInstance(cap, str)


class TestDevOpsAgentValidation(unittest.TestCase):

    def test_wrong_agent_type_returns_failure(self):
        agent = DevOpsAgent()
        task = _AgentTask(
            task_id="x", agent_type=_AgentType.EXPLORER,
            description="wrong", prompt="wrong",
        )
        result = agent.execute(task)
        self.assertFalse(result.success)
        self.assertIsNone(result.output)

    def test_wrong_agent_type_error_message(self):
        agent = DevOpsAgent()
        task = _AgentTask(
            task_id="x", agent_type=_AgentType.RESEARCHER,
            description="wrong", prompt="wrong",
        )
        result = agent.execute(task)
        self.assertTrue(len(result.errors) > 0)


class TestDevOpsAgentExecuteSuccess(unittest.TestCase):

    def _agent_with_good_infer(self):
        agent = DevOpsAgent()
        agent._infer = MagicMock(return_value=GOOD_DEVOPS_JSON)
        return agent

    def test_execute_returns_agent_result(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertIsInstance(result, _AgentResult)

    def test_execute_success_true(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertTrue(result.success)

    def test_execute_output_is_dict(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertIsInstance(result.output, dict)

    def test_execute_output_has_ci_pipeline(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertIn("ci_pipeline", result.output)

    def test_execute_output_has_containerisation(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertIn("containerisation", result.output)

    def test_execute_output_has_deployment_strategy(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertIn("deployment_strategy", result.output)

    def test_execute_output_has_monitoring(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertIn("monitoring", result.output)

    def test_execute_output_has_summary(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertIn("summary", result.output)

    def test_execute_metadata_has_platform(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_devops_task())
        self.assertEqual(result.metadata.get("platform"), "docker")

    def test_execute_metadata_platform_kubernetes(self):
        agent = DevOpsAgent(config={"platform": "kubernetes"})
        agent._infer = MagicMock(return_value=GOOD_DEVOPS_JSON)
        result = agent.execute(_devops_task())
        self.assertEqual(result.metadata.get("platform"), "kubernetes")

    def test_execute_uses_project_description_from_context(self):
        agent = DevOpsAgent()
        captured = []
        original_infer_json = agent._infer_json

        def capturing_infer_json(prompt, **kwargs):
            captured.append(prompt)
            return kwargs.get("fallback", {})

        agent._infer_json = capturing_infer_json
        agent.execute(_devops_task(project_description="My special project"))
        self.assertTrue(any("My special project" in p for p in captured))

    def test_execute_uses_tech_stack_from_context(self):
        agent = DevOpsAgent()
        captured = []

        def capturing_infer_json(prompt, **kwargs):
            captured.append(prompt)
            return kwargs.get("fallback", {})

        agent._infer_json = capturing_infer_json
        agent.execute(_devops_task(tech_stack="Rust"))
        self.assertTrue(any("Rust" in p for p in captured))

    def test_execute_defaults_tech_stack_to_python(self):
        agent = DevOpsAgent()
        captured = []

        def capturing_infer_json(prompt, **kwargs):
            captured.append(prompt)
            return kwargs.get("fallback", {})

        agent._infer_json = capturing_infer_json
        agent.execute(_devops_task())
        self.assertTrue(any("Python" in p for p in captured))


class TestDevOpsAgentExecuteFallback(unittest.TestCase):
    """When _infer returns empty string, fallback dict is used."""

    def test_fallback_path_success(self):
        agent = DevOpsAgent()
        # _infer returns "" → _infer_json returns fallback
        result = agent.execute(_devops_task())
        self.assertTrue(result.success)

    def test_fallback_output_has_required_keys(self):
        agent = DevOpsAgent()
        result = agent.execute(_devops_task())
        for key in ("ci_pipeline", "containerisation", "deployment_strategy",
                    "infrastructure", "monitoring", "runbooks", "summary"):
            self.assertIn(key, result.output)

    def test_fallback_ci_pipeline_has_stages(self):
        agent = DevOpsAgent()
        result = agent.execute(_devops_task())
        self.assertIn("stages", result.output["ci_pipeline"])

    def test_fallback_containerisation_has_dockerfile(self):
        agent = DevOpsAgent()
        result = agent.execute(_devops_task())
        self.assertIn("dockerfile", result.output["containerisation"])

    def test_fallback_deployment_strategy_type(self):
        agent = DevOpsAgent()
        result = agent.execute(_devops_task())
        self.assertIn("type", result.output["deployment_strategy"])


class TestDevOpsAgentExecuteFailure(unittest.TestCase):

    def test_execute_exception_returns_failure(self):
        agent = DevOpsAgent()
        agent._infer_json = MagicMock(side_effect=RuntimeError("LLM exploded"))
        result = agent.execute(_devops_task())
        self.assertFalse(result.success)

    def test_execute_exception_error_message(self):
        agent = DevOpsAgent()
        agent._infer_json = MagicMock(side_effect=RuntimeError("LLM exploded"))
        result = agent.execute(_devops_task())
        self.assertIn("LLM exploded", result.errors[0])

    def test_execute_exception_output_is_none(self):
        agent = DevOpsAgent()
        agent._infer_json = MagicMock(side_effect=ValueError("bad"))
        result = agent.execute(_devops_task())
        self.assertIsNone(result.output)


class TestDevOpsAgentVerify(unittest.TestCase):

    def test_verify_valid_output_passes(self):
        agent = DevOpsAgent()
        output = {
            "ci_pipeline": {"platform": "github-actions"},
            "containerisation": {"dockerfile": "FROM python:3.11"},
            "deployment_strategy": {"type": "blue-green"},
        }
        result = agent.verify(output)
        self.assertTrue(result.passed)

    def test_verify_non_dict_fails(self):
        agent = DevOpsAgent()
        result = agent.verify("not a dict")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_verify_missing_ci_pipeline_reduces_score(self):
        agent = DevOpsAgent()
        output = {
            "containerisation": {"dockerfile": "FROM python"},
            "deployment_strategy": {"type": "rolling"},
        }
        result = agent.verify(output)
        self.assertLess(result.score, 1.0)

    def test_verify_missing_containerisation_reduces_score(self):
        agent = DevOpsAgent()
        output = {
            "ci_pipeline": {"platform": "github-actions"},
            "deployment_strategy": {"type": "rolling"},
        }
        result = agent.verify(output)
        self.assertLess(result.score, 1.0)

    def test_verify_empty_dict_fails(self):
        agent = DevOpsAgent()
        result = agent.verify({})
        self.assertFalse(result.passed)

    def test_verify_score_is_float(self):
        agent = DevOpsAgent()
        result = agent.verify({"ci_pipeline": {}, "containerisation": {}, "deployment_strategy": {}})
        self.assertIsInstance(result.score, float)


class TestDevOpsAgentFallbackDevops(unittest.TestCase):

    def test_fallback_devops_is_dict(self):
        agent = DevOpsAgent()
        fb = agent._fallback_devops("My project", "Python")
        self.assertIsInstance(fb, dict)

    def test_fallback_devops_ci_pipeline_platform_github(self):
        agent = DevOpsAgent()
        fb = agent._fallback_devops("proj", "Node.js")
        self.assertEqual(fb["ci_pipeline"]["platform"], "github-actions")

    def test_fallback_devops_summary_contains_platform(self):
        agent = DevOpsAgent(config={"platform": "kubernetes"})
        fb = agent._fallback_devops("proj", "Go")
        self.assertIn("kubernetes", fb["summary"])

    def test_fallback_devops_security_hardening_list(self):
        agent = DevOpsAgent()
        fb = agent._fallback_devops("proj", "Python")
        self.assertIsInstance(fb["security_hardening"], list)
        self.assertGreater(len(fb["security_hardening"]), 0)

    def test_fallback_devops_runbooks_list(self):
        agent = DevOpsAgent()
        fb = agent._fallback_devops("proj", "Python")
        self.assertIsInstance(fb["runbooks"], list)
        self.assertGreater(len(fb["runbooks"]), 0)


class TestGetDevOpsAgent(unittest.TestCase):

    def test_get_devops_agent_returns_instance(self):
        from vetinari.agents.devops_agent import get_devops_agent
        agent = get_devops_agent()
        self.assertIsInstance(agent, DevOpsAgent)

    def test_get_devops_agent_singleton(self):
        from vetinari.agents.devops_agent import get_devops_agent
        import vetinari.agents.devops_agent as _mod
        _mod._devops_agent = None  # reset singleton
        a1 = get_devops_agent()
        a2 = get_devops_agent()
        self.assertIs(a1, a2)


# ===========================================================================
# DocumentationAgent Tests
# ===========================================================================

class TestDocumentationAgentInit(unittest.TestCase):

    def test_default_doc_format_markdown(self):
        agent = DocumentationAgent()
        self.assertEqual(agent._doc_format, "markdown")

    def test_custom_doc_format(self):
        agent = DocumentationAgent(config={"doc_format": "rst"})
        self.assertEqual(agent._doc_format, "rst")

    def test_agent_type_is_documentation(self):
        agent = DocumentationAgent()
        self.assertEqual(agent.agent_type, _AgentType.DOCUMENTATION_AGENT)

    def test_no_config_creates_empty_dict(self):
        agent = DocumentationAgent()
        self.assertIsInstance(agent._config, dict)


class TestDocumentationAgentSystemPrompt(unittest.TestCase):

    def test_system_prompt_non_empty(self):
        agent = DocumentationAgent()
        self.assertGreater(len(agent.get_system_prompt()), 10)

    def test_system_prompt_contains_doc_format(self):
        agent = DocumentationAgent(config={"doc_format": "rst"})
        self.assertIn("rst", agent.get_system_prompt())

    def test_system_prompt_contains_markdown_default(self):
        agent = DocumentationAgent()
        self.assertIn("markdown", agent.get_system_prompt())

    def test_system_prompt_contains_pages(self):
        self.assertIn("pages", DocumentationAgent().get_system_prompt())

    def test_system_prompt_contains_api_docs(self):
        self.assertIn("api_docs", DocumentationAgent().get_system_prompt())

    def test_system_prompt_contains_user_guides(self):
        self.assertIn("user_guides", DocumentationAgent().get_system_prompt())

    def test_system_prompt_contains_change_log(self):
        self.assertIn("change_log", DocumentationAgent().get_system_prompt())


class TestDocumentationAgentCapabilities(unittest.TestCase):

    def test_capabilities_non_empty(self):
        caps = DocumentationAgent().get_capabilities()
        self.assertIsInstance(caps, list)
        self.assertGreater(len(caps), 0)

    def test_capabilities_contains_api_documentation(self):
        self.assertIn("api_documentation", DocumentationAgent().get_capabilities())

    def test_capabilities_contains_user_guide(self):
        self.assertIn("user_guide_generation", DocumentationAgent().get_capabilities())

    def test_capabilities_contains_changelog(self):
        self.assertIn("changelog_creation", DocumentationAgent().get_capabilities())

    def test_capabilities_contains_markdown(self):
        self.assertIn("markdown_generation", DocumentationAgent().get_capabilities())

    def test_capabilities_all_strings(self):
        for cap in DocumentationAgent().get_capabilities():
            self.assertIsInstance(cap, str)


class TestDocumentationAgentValidation(unittest.TestCase):

    def test_wrong_agent_type_returns_failure(self):
        agent = DocumentationAgent()
        task = _AgentTask(
            task_id="x", agent_type=_AgentType.BUILDER,
            description="wrong", prompt="wrong",
        )
        result = agent.execute(task)
        self.assertFalse(result.success)

    def test_wrong_agent_type_no_output(self):
        agent = DocumentationAgent()
        task = _AgentTask(
            task_id="x", agent_type=_AgentType.PLANNER,
            description="wrong", prompt="wrong",
        )
        result = agent.execute(task)
        self.assertIsNone(result.output)


class TestDocumentationAgentExecuteSuccess(unittest.TestCase):

    def _agent_with_good_infer(self):
        agent = DocumentationAgent()
        agent._infer = MagicMock(return_value=GOOD_DOC_JSON)
        return agent

    def test_execute_returns_agent_result(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertIsInstance(result, _AgentResult)

    def test_execute_success_true(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertTrue(result.success)

    def test_execute_output_is_dict(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertIsInstance(result.output, dict)

    def test_execute_output_has_docs_manifest(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertIn("docs_manifest", result.output)

    def test_execute_output_has_pages(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertIn("pages", result.output)

    def test_execute_output_has_api_docs(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertIn("api_docs", result.output)

    def test_execute_output_has_user_guides(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertIn("user_guides", result.output)

    def test_execute_output_has_change_log(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertIn("change_log", result.output)

    def test_execute_output_has_summary(self):
        result = self._agent_with_good_infer().execute(_doc_task())
        self.assertIn("summary", result.output)

    def test_execute_metadata_artifacts_count(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_doc_task(artifacts=["a", "b"]))
        self.assertEqual(result.metadata.get("artifacts_documented"), 2)

    def test_execute_metadata_doc_type(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_doc_task(doc_type="api"))
        self.assertEqual(result.metadata.get("doc_type"), "api")

    def test_execute_metadata_pages_count(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_doc_task())
        self.assertIn("pages_count", result.metadata)

    def test_execute_manifest_generated_at_updated(self):
        agent = self._agent_with_good_infer()
        result = agent.execute(_doc_task())
        # generated_at should be a non-empty ISO timestamp string
        generated_at = result.output["docs_manifest"]["generated_at"]
        self.assertIsInstance(generated_at, str)
        self.assertGreater(len(generated_at), 0)


class TestDocumentationAgentExecuteFallback(unittest.TestCase):
    """_infer returns "" so _infer_json falls back to fallback dict."""

    def test_fallback_path_success(self):
        agent = DocumentationAgent()
        result = agent.execute(_doc_task())
        self.assertTrue(result.success)

    def test_fallback_output_has_required_keys(self):
        agent = DocumentationAgent()
        result = agent.execute(_doc_task())
        for key in ("docs_manifest", "pages", "api_docs", "user_guides",
                    "references", "change_log", "summary"):
            self.assertIn(key, result.output)

    def test_fallback_docs_manifest_has_title(self):
        agent = DocumentationAgent()
        result = agent.execute(_doc_task(project_name="MyApp"))
        self.assertIn("title", result.output["docs_manifest"])

    def test_fallback_pages_is_list(self):
        agent = DocumentationAgent()
        result = agent.execute(_doc_task())
        self.assertIsInstance(result.output["pages"], list)

    def test_fallback_uses_project_name(self):
        agent = DocumentationAgent()
        result = agent.execute(_doc_task(project_name="SuperProject"))
        title = result.output["docs_manifest"]["title"]
        self.assertIn("SuperProject", title)

    def test_fallback_version_in_manifest(self):
        agent = DocumentationAgent()
        result = agent.execute(_doc_task(version="2.5.0"))
        self.assertEqual(result.output["docs_manifest"]["version"], "2.5.0")


class TestDocumentationAgentExecuteFailure(unittest.TestCase):

    def test_execute_exception_returns_failure(self):
        agent = DocumentationAgent()
        agent._infer_json = MagicMock(side_effect=RuntimeError("crash"))
        result = agent.execute(_doc_task())
        self.assertFalse(result.success)

    def test_execute_exception_error_list(self):
        agent = DocumentationAgent()
        agent._infer_json = MagicMock(side_effect=ValueError("bad input"))
        result = agent.execute(_doc_task())
        self.assertIn("bad input", result.errors[0])

    def test_execute_exception_output_none(self):
        agent = DocumentationAgent()
        agent._infer_json = MagicMock(side_effect=TypeError("type error"))
        result = agent.execute(_doc_task())
        self.assertIsNone(result.output)


class TestDocumentationAgentVerify(unittest.TestCase):

    def _full_output(self):
        return {
            "docs_manifest": {"title": "Docs"},
            "pages": [{"title": "p1"}],
            "api_docs": {"endpoints": [{"path": "/x"}]},
            "user_guides": [{"title": "guide"}],
            "change_log": [{"version": "1.0"}],
        }

    def test_verify_full_output_passes(self):
        agent = DocumentationAgent()
        result = agent.verify(self._full_output())
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)

    def test_verify_non_dict_fails(self):
        agent = DocumentationAgent()
        result = agent.verify(["not", "a", "dict"])
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_verify_missing_manifest_reduces_score(self):
        agent = DocumentationAgent()
        output = self._full_output()
        del output["docs_manifest"]
        result = agent.verify(output)
        self.assertLess(result.score, 1.0)

    def test_verify_missing_pages_reduces_score(self):
        agent = DocumentationAgent()
        output = self._full_output()
        del output["pages"]
        result = agent.verify(output)
        # pages deducts 0.3; score = 0.7, still passes but score is reduced
        self.assertLess(result.score, 1.0)
        self.assertGreater(len(result.issues), 0)

    def test_verify_missing_api_docs_reduces_score(self):
        agent = DocumentationAgent()
        output = self._full_output()
        del output["api_docs"]
        result = agent.verify(output)
        self.assertLess(result.score, 1.0)

    def test_verify_empty_dict_fails(self):
        agent = DocumentationAgent()
        result = agent.verify({})
        self.assertFalse(result.passed)

    def test_verify_score_is_float(self):
        agent = DocumentationAgent()
        result = agent.verify(self._full_output())
        self.assertIsInstance(result.score, float)


class TestDocumentationAgentSummariseArtifacts(unittest.TestCase):

    def test_summarise_no_artifacts_returns_fallback(self):
        agent = DocumentationAgent()
        summary = agent._summarise_artifacts([], "", "")
        self.assertEqual(summary, "No specific artifacts provided.")

    def test_summarise_with_description(self):
        agent = DocumentationAgent()
        summary = agent._summarise_artifacts([], "", "Build a REST API")
        self.assertIn("Build a REST API", summary)

    def test_summarise_with_code(self):
        agent = DocumentationAgent()
        summary = agent._summarise_artifacts([], "def hello(): pass", "desc")
        self.assertIn("def hello", summary)

    def test_summarise_with_string_artifacts(self):
        agent = DocumentationAgent()
        summary = agent._summarise_artifacts(["Artifact content here"], "", "desc")
        self.assertIn("Artifact content here", summary)

    def test_summarise_with_dict_artifacts(self):
        agent = DocumentationAgent()
        summary = agent._summarise_artifacts([{"name": "MyClass", "description": "Does things"}], "", "desc")
        self.assertIn("MyClass", summary)

    def test_summarise_limits_artifacts_to_five(self):
        agent = DocumentationAgent()
        artifacts = [f"artifact {i}" for i in range(10)]
        summary = agent._summarise_artifacts(artifacts, "", "desc")
        # Only first 5 artifacts should appear
        self.assertIn("artifact 0", summary)
        self.assertNotIn("artifact 5", summary)

    def test_summarise_code_truncated_at_2000(self):
        agent = DocumentationAgent()
        long_code = "x = 1\n" * 1000  # > 2000 chars
        summary = agent._summarise_artifacts([], long_code, "desc")
        self.assertLess(len(summary), len(long_code) + 200)


class TestGetDocumentationAgent(unittest.TestCase):

    def test_get_documentation_agent_returns_instance(self):
        from vetinari.agents.documentation_agent import get_documentation_agent
        import vetinari.agents.documentation_agent as _dm
        _dm._documentation_agent = None
        agent = get_documentation_agent()
        self.assertIsInstance(agent, DocumentationAgent)

    def test_get_documentation_agent_singleton(self):
        from vetinari.agents.documentation_agent import get_documentation_agent
        import vetinari.agents.documentation_agent as _dm
        _dm._documentation_agent = None
        a1 = get_documentation_agent()
        a2 = get_documentation_agent()
        self.assertIs(a1, a2)


# ===========================================================================
# ErrorRecoveryAgent Tests
# ===========================================================================

class TestErrorRecoveryAgentInit(unittest.TestCase):

    def test_agent_type_is_error_recovery(self):
        agent = ErrorRecoveryAgent()
        self.assertEqual(agent.agent_type, _AgentType.ERROR_RECOVERY)

    def test_no_config_creates_empty_dict(self):
        agent = ErrorRecoveryAgent()
        self.assertIsInstance(agent._config, dict)

    def test_custom_config_stored(self):
        agent = ErrorRecoveryAgent(config={"max_retries": 5})
        self.assertEqual(agent._config["max_retries"], 5)


class TestErrorRecoveryAgentSystemPrompt(unittest.TestCase):

    def test_system_prompt_non_empty(self):
        agent = ErrorRecoveryAgent()
        self.assertGreater(len(agent.get_system_prompt()), 10)

    def test_system_prompt_contains_root_cause(self):
        self.assertIn("root cause", ErrorRecoveryAgent().get_system_prompt().lower())

    def test_system_prompt_contains_retry(self):
        self.assertIn("retry", ErrorRecoveryAgent().get_system_prompt().lower())

    def test_system_prompt_contains_circuit(self):
        self.assertIn("circuit", ErrorRecoveryAgent().get_system_prompt().lower())

    def test_system_prompt_contains_fallback(self):
        self.assertIn("fallback", ErrorRecoveryAgent().get_system_prompt().lower())


class TestErrorRecoveryAgentCapabilities(unittest.TestCase):

    def test_capabilities_non_empty(self):
        caps = ErrorRecoveryAgent().get_capabilities()
        self.assertIsInstance(caps, list)
        self.assertGreater(len(caps), 0)

    def test_capabilities_contains_root_cause_analysis(self):
        self.assertIn("root_cause_analysis", ErrorRecoveryAgent().get_capabilities())

    def test_capabilities_contains_error_classification(self):
        self.assertIn("error_classification", ErrorRecoveryAgent().get_capabilities())

    def test_capabilities_contains_retry_strategy(self):
        self.assertIn("retry_strategy_generation", ErrorRecoveryAgent().get_capabilities())

    def test_capabilities_contains_circuit_breaker(self):
        self.assertIn("circuit_breaker_design", ErrorRecoveryAgent().get_capabilities())

    def test_capabilities_contains_fallback_planning(self):
        self.assertIn("fallback_planning", ErrorRecoveryAgent().get_capabilities())

    def test_capabilities_contains_post_mortem(self):
        self.assertIn("post_mortem_analysis", ErrorRecoveryAgent().get_capabilities())

    def test_capabilities_contains_resilience(self):
        self.assertIn("resilience_improvement", ErrorRecoveryAgent().get_capabilities())

    def test_capabilities_all_strings(self):
        for cap in ErrorRecoveryAgent().get_capabilities():
            self.assertIsInstance(cap, str)


class TestErrorRecoveryAgentExecuteSuccess(unittest.TestCase):

    def _agent_with_good_infer(self):
        agent = ErrorRecoveryAgent()
        agent._infer = MagicMock(return_value=GOOD_ERR_JSON)
        return agent

    def test_execute_returns_agent_result(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIsInstance(result, _AgentResult)

    def test_execute_success_true(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertTrue(result.success)

    def test_execute_output_is_dict(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIsInstance(result.output, dict)

    def test_execute_output_has_root_cause(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIn("root_cause", result.output)

    def test_execute_output_has_recovery_strategies(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIn("recovery_strategies", result.output)

    def test_execute_output_has_immediate_actions(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIn("immediate_actions", result.output)

    def test_execute_output_has_retry_policy(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIn("retry_policy", result.output)

    def test_execute_output_has_circuit_breaker(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIn("circuit_breaker", result.output)

    def test_execute_output_has_fallback_plan(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIn("fallback_plan", result.output)

    def test_execute_metadata_has_task_id(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertEqual(result.metadata.get("task_id"), "err-001")

    def test_execute_metadata_has_agent_type(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertEqual(result.metadata.get("agent_type"), "ERROR_RECOVERY")

    def test_execute_metadata_has_error_count(self):
        result = self._agent_with_good_infer().execute(_err_task())
        self.assertIn("error_count", result.metadata)


class TestErrorRecoveryAgentExecuteFallback(unittest.TestCase):
    """_infer returns "" → fallback recovery is used."""

    def test_fallback_success(self):
        agent = ErrorRecoveryAgent()
        result = agent.execute(_err_task())
        self.assertTrue(result.success)

    def test_fallback_has_root_cause(self):
        agent = ErrorRecoveryAgent()
        result = agent.execute(_err_task())
        self.assertIn("root_cause", result.output)

    def test_fallback_has_recovery_strategies(self):
        agent = ErrorRecoveryAgent()
        result = agent.execute(_err_task())
        self.assertIsInstance(result.output["recovery_strategies"], list)
        self.assertGreater(len(result.output["recovery_strategies"]), 0)

    def test_fallback_has_retry_policy(self):
        agent = ErrorRecoveryAgent()
        result = agent.execute(_err_task())
        rp = result.output["retry_policy"]
        self.assertIn("max_attempts", rp)
        self.assertIn("backoff_multiplier", rp)

    def test_fallback_has_circuit_breaker(self):
        agent = ErrorRecoveryAgent()
        result = agent.execute(_err_task())
        cb = result.output["circuit_breaker"]
        self.assertIn("enabled", cb)
        self.assertTrue(cb["enabled"])

    def test_fallback_has_preventive_measures(self):
        agent = ErrorRecoveryAgent()
        result = agent.execute(_err_task())
        self.assertIsInstance(result.output["preventive_measures"], list)

    def test_fallback_has_monitoring_recommendations(self):
        agent = ErrorRecoveryAgent()
        result = agent.execute(_err_task())
        self.assertIsInstance(result.output["monitoring_recommendations"], list)


class TestErrorRecoveryAgentExecuteFailure(unittest.TestCase):

    def test_execute_exception_returns_failure(self):
        agent = ErrorRecoveryAgent()
        agent._infer_json = MagicMock(side_effect=RuntimeError("crash"))
        result = agent.execute(_err_task())
        self.assertFalse(result.success)

    def test_execute_exception_errors_list(self):
        agent = ErrorRecoveryAgent()
        agent._infer_json = MagicMock(side_effect=RuntimeError("boom"))
        result = agent.execute(_err_task())
        self.assertIn("boom", result.errors[0])

    def test_execute_exception_output_is_empty_dict(self):
        agent = ErrorRecoveryAgent()
        agent._infer_json = MagicMock(side_effect=ValueError("bad"))
        result = agent.execute(_err_task())
        # ErrorRecoveryAgent returns {} not None on exception
        self.assertEqual(result.output, {})


class TestErrorRecoveryHeuristicClassification(unittest.TestCase):

    def _agent(self):
        return ErrorRecoveryAgent()

    def test_connection_refused_detected(self):
        findings = self._agent()._classify_errors_heuristically("ConnectionRefusedError: [Errno 111]")
        types = [f["type"] for f in findings]
        self.assertIn("connection_refused", types)

    def test_timeout_detected(self):
        findings = self._agent()._classify_errors_heuristically("ReadTimeout: request timed out")
        types = [f["type"] for f in findings]
        self.assertIn("timeout", types)

    def test_rate_limit_detected(self):
        findings = self._agent()._classify_errors_heuristically("429 Too Many Requests")
        types = [f["type"] for f in findings]
        self.assertIn("rate_limit", types)

    def test_oom_detected(self):
        findings = self._agent()._classify_errors_heuristically("MemoryError: out of memory")
        types = [f["type"] for f in findings]
        self.assertIn("out_of_memory", types)

    def test_import_error_detected(self):
        findings = self._agent()._classify_errors_heuristically("ModuleNotFoundError: No module named 'foo'")
        types = [f["type"] for f in findings]
        self.assertIn("import_error", types)

    def test_attribute_error_detected(self):
        findings = self._agent()._classify_errors_heuristically("AttributeError: 'NoneType' has no attribute 'x'")
        types = [f["type"] for f in findings]
        self.assertIn("attribute_error", types)

    def test_key_error_detected(self):
        findings = self._agent()._classify_errors_heuristically("KeyError: 'missing_key'")
        types = [f["type"] for f in findings]
        self.assertIn("key_error", types)

    def test_type_error_detected(self):
        findings = self._agent()._classify_errors_heuristically("TypeError: unsupported operand type")
        types = [f["type"] for f in findings]
        self.assertIn("type_error", types)

    def test_permission_denied_detected(self):
        findings = self._agent()._classify_errors_heuristically("PermissionError: [Errno 13] Permission denied")
        types = [f["type"] for f in findings]
        self.assertIn("permission_denied", types)

    def test_not_found_detected(self):
        findings = self._agent()._classify_errors_heuristically("FileNotFoundError: /path/not/found")
        types = [f["type"] for f in findings]
        self.assertIn("not_found", types)

    def test_json_decode_detected(self):
        findings = self._agent()._classify_errors_heuristically("JSONDecodeError: Expecting value at line 1")
        types = [f["type"] for f in findings]
        self.assertIn("json_decode", types)

    def test_no_match_returns_empty_list(self):
        findings = self._agent()._classify_errors_heuristically("Everything is fine")
        self.assertEqual(findings, [])

    def test_finding_has_required_keys(self):
        findings = self._agent()._classify_errors_heuristically("ConnectionRefusedError")
        self.assertIn("type", findings[0])
        self.assertIn("category", findings[0])
        self.assertIn("severity", findings[0])
        self.assertIn("quick_fix", findings[0])

    def test_connection_refused_category_network(self):
        findings = self._agent()._classify_errors_heuristically("connection refused")
        net_findings = [f for f in findings if f["type"] == "connection_refused"]
        self.assertEqual(net_findings[0]["category"], "network")


class TestErrorRecoveryAgentVerify(unittest.TestCase):

    def _full_output(self):
        return {
            "root_cause": "Redis down",
            "errors_identified": [{"type": "connection_refused"}],
            "immediate_actions": [{"priority": 1, "action": "restart"}],
            "recovery_strategies": [{"name": "retry", "type": "retry"}],
        }

    def test_verify_full_output_passes(self):
        agent = ErrorRecoveryAgent()
        result = agent.verify(self._full_output())
        self.assertTrue(result.passed)

    def test_verify_non_dict_fails(self):
        agent = ErrorRecoveryAgent()
        result = agent.verify("not a dict")
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)

    def test_verify_missing_root_cause_reduces_score(self):
        agent = ErrorRecoveryAgent()
        output = self._full_output()
        del output["root_cause"]
        del output["errors_identified"]
        result = agent.verify(output)
        self.assertLess(result.score, 1.0)

    def test_verify_missing_recovery_strategies_reduces_score(self):
        agent = ErrorRecoveryAgent()
        output = self._full_output()
        del output["recovery_strategies"]
        result = agent.verify(output)
        self.assertLess(result.score, 1.0)

    def test_verify_missing_immediate_actions_reduces_score(self):
        agent = ErrorRecoveryAgent()
        output = self._full_output()
        del output["immediate_actions"]
        result = agent.verify(output)
        self.assertLess(result.score, 1.0)

    def test_verify_empty_dict_fails(self):
        agent = ErrorRecoveryAgent()
        result = agent.verify({})
        self.assertFalse(result.passed)

    def test_verify_score_is_float(self):
        agent = ErrorRecoveryAgent()
        result = agent.verify(self._full_output())
        self.assertIsInstance(result.score, float)

    def test_verify_score_rounded(self):
        agent = ErrorRecoveryAgent()
        result = agent.verify({})
        # score is rounded to 2dp
        self.assertEqual(result.score, round(result.score, 2))


class TestErrorRecoveryFallbackRecovery(unittest.TestCase):

    def test_fallback_recovery_is_dict(self):
        agent = ErrorRecoveryAgent()
        fb = agent._fallback_recovery(_err_task(), [])
        self.assertIsInstance(fb, dict)

    def test_fallback_has_default_actions_when_no_findings(self):
        agent = ErrorRecoveryAgent()
        fb = agent._fallback_recovery(_err_task(), [])
        self.assertGreater(len(fb["immediate_actions"]), 0)

    def test_fallback_uses_findings_for_actions(self):
        agent = ErrorRecoveryAgent()
        findings = [{"type": "timeout", "category": "network", "severity": "medium", "quick_fix": "Increase timeout"}]
        fb = agent._fallback_recovery(_err_task(), findings)
        actions = fb["immediate_actions"]
        self.assertTrue(any("Increase timeout" in a["action"] for a in actions))

    def test_fallback_uses_heuristic_findings_in_errors_identified(self):
        agent = ErrorRecoveryAgent()
        findings = [{"type": "rate_limit", "category": "api", "severity": "medium", "quick_fix": "Backoff"}]
        fb = agent._fallback_recovery(_err_task(), findings)
        self.assertEqual(fb["errors_identified"], findings)

    def test_fallback_default_errors_identified_unknown(self):
        agent = ErrorRecoveryAgent()
        fb = agent._fallback_recovery(_err_task(), [])
        self.assertEqual(fb["errors_identified"][0]["type"], "unknown")

    def test_fallback_circuit_breaker_enabled(self):
        agent = ErrorRecoveryAgent()
        fb = agent._fallback_recovery(_err_task(), [])
        self.assertTrue(fb["circuit_breaker"]["enabled"])

    def test_fallback_retry_policy_max_attempts(self):
        agent = ErrorRecoveryAgent()
        fb = agent._fallback_recovery(_err_task(), [])
        self.assertEqual(fb["retry_policy"]["max_attempts"], 3)

    def test_fallback_post_mortem_contains_description(self):
        agent = ErrorRecoveryAgent()
        task = _err_task()
        fb = agent._fallback_recovery(task, [])
        self.assertIn(task.description[:50], fb["post_mortem_summary"])


class TestErrorRecoveryAgentContextParsing(unittest.TestCase):

    def test_error_key_in_context_used(self):
        agent = ErrorRecoveryAgent()
        agent._infer = MagicMock(return_value=GOOD_ERR_JSON)
        result = agent.execute(_err_task(error="ConnectionRefusedError"))
        self.assertTrue(result.success)

    def test_error_message_key_in_context_used(self):
        agent = ErrorRecoveryAgent()
        agent._infer = MagicMock(return_value=GOOD_ERR_JSON)
        result = agent.execute(_err_task(error_message="TimeoutError"))
        self.assertTrue(result.success)

    def test_stack_trace_in_context(self):
        agent = ErrorRecoveryAgent()
        agent._infer = MagicMock(return_value=GOOD_ERR_JSON)
        result = agent.execute(_err_task(stack_trace="File main.py line 42"))
        self.assertTrue(result.success)

    def test_component_key_in_context(self):
        agent = ErrorRecoveryAgent()
        agent._infer = MagicMock(return_value=GOOD_ERR_JSON)
        result = agent.execute(_err_task(component="redis_client"))
        self.assertTrue(result.success)

    def test_empty_context_uses_description(self):
        agent = ErrorRecoveryAgent()
        # Empty context → falls back to task.description for error text
        result = agent.execute(_err_task())
        self.assertTrue(result.success)

    def test_heuristic_runs_on_connection_refused_in_description(self):
        agent = ErrorRecoveryAgent()
        task = _err_task()
        task.description = "ConnectionRefusedError when connecting to Redis"
        result = agent.execute(task)
        # Heuristic should classify and fallback provides at least one action from it
        self.assertTrue(result.success)
        actions = result.output.get("immediate_actions", [])
        self.assertGreater(len(actions), 0)


class TestGetErrorRecoveryAgent(unittest.TestCase):

    def test_get_error_recovery_agent_returns_instance(self):
        from vetinari.agents.error_recovery_agent import get_error_recovery_agent
        import vetinari.agents.error_recovery_agent as _em
        _em._error_recovery_agent = None
        agent = get_error_recovery_agent()
        self.assertIsInstance(agent, ErrorRecoveryAgent)

    def test_get_error_recovery_agent_singleton(self):
        from vetinari.agents.error_recovery_agent import get_error_recovery_agent
        import vetinari.agents.error_recovery_agent as _em
        _em._error_recovery_agent = None
        a1 = get_error_recovery_agent()
        a2 = get_error_recovery_agent()
        self.assertIs(a1, a2)


if __name__ == "__main__":
    unittest.main()
