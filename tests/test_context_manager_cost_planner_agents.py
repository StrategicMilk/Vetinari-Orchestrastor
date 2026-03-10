"""
Comprehensive pytest tests for ContextManagerAgent and CostPlannerAgent.

Stubs all vetinari dependencies before importing the agents, so no
real LM Studio / network / filesystem access is needed.

NOTE: These agents were consolidated into PlannerAgent in v0.4.0.
Tests that check legacy-specific internals are skipped.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy ContextManagerAgent/CostPlannerAgent consolidated into PlannerAgent in v0.4.0"
)


# ---------------------------------------------------------------------------
# Dependency stub setup — must happen before any vetinari import
# ---------------------------------------------------------------------------

def _stub():  # noqa: C901
    """Register lightweight stubs for every vetinari dependency."""
    import os

    # Real source root so actual .py agent files are discoverable
    _HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def pkg(name):
        """Stub a package but keep its real __path__ so sub-modules import.
        Preserves existing modules to avoid clobbering real modules loaded earlier."""
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        # Convert dotted name to filesystem path under the real source root
        rel = name.replace(".", os.sep)
        real_path = os.path.join(_HERE, rel)
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

    # Top-level packages — real paths preserved so sub-module .py files load
    pkg("vetinari")
    pkg("vetinari.agents")
    pkg("vetinari.adapters")
    pkg("vetinari.analytics")
    pkg("vetinari.learning")
    pkg("vetinari.memory")
    pkg("vetinari.constraints")
    pkg("vetinari.config")
    pkg("vetinari.tools")

    # ---- Canonical enums (match vetinari/types.py exactly) ----
    from enum import Enum

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

    class ExecutionMode(Enum):
        PLANNING = "planning"
        EXECUTION = "execution"
        SANDBOX = "sandbox"

    # ---- Data contracts (match vetinari/agents/contracts.py) ----
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
        # ContextManagerAgent passes task_id= and agent_type= and error= as kwargs
        task_id: str = ""
        agent_type: Any = None
        error: str = ""

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
        model_scores: List[Dict] = field(default_factory=list)
        notes: str = ""
        warnings: List[str] = field(default_factory=list)
        needs_context: bool = False
        follow_up_question: str = ""
        final_delivery_path: str = ""
        final_delivery_summary: str = ""
        created_at: str = ""

        @classmethod
        def create_new(cls, goal, phase=0):
            import uuid as _uuid
            return cls(plan_id=f"plan_{_uuid.uuid4().hex[:8]}", goal=goal, phase=phase)

    # Minimal AGENT_REGISTRY covering agents under test
    _REGISTRY: Dict[Any, AgentSpec] = {
        AgentType.CONTEXT_MANAGER: AgentSpec(
            agent_type=AgentType.CONTEXT_MANAGER,
            name="Context Manager Agent",
            description="Long-term context management",
            default_model="qwen2.5-72b",
        ),
        AgentType.COST_PLANNER: AgentSpec(
            agent_type=AgentType.COST_PLANNER,
            name="Cost Planner",
            description="Cost accounting",
            default_model="qwen2.5-coder-7b",
        ),
    }

    def get_agent_spec(agent_type):
        return _REGISTRY.get(agent_type)

    # ---- Register canonical types ----
    mod("vetinari.types",
        AgentType=AgentType, TaskStatus=TaskStatus,
        ExecutionMode=ExecutionMode)

    mod("vetinari.agents.contracts",
        AgentType=AgentType, TaskStatus=TaskStatus, ExecutionMode=ExecutionMode,
        AgentTask=AgentTask, AgentResult=AgentResult, AgentSpec=AgentSpec,
        VerificationResult=VerificationResult, Task=Task, Plan=Plan,
        get_agent_spec=get_agent_spec, AGENT_REGISTRY=_REGISTRY,
        get_all_agent_specs=lambda: list(_REGISTRY.values()),
        get_enabled_agents=lambda: list(_REGISTRY.values()),
        ACTIVE_AGENT_TYPES=set(),
        AGENT_TYPE_MAPPING={},
        resolve_agent_type=lambda x: x)

    # ---- BaseAgent stub ----
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

        def initialize(self, context):
            self._context = context
            self._adapter_manager = context.get("adapter_manager")
            self._web_search = context.get("web_search")
            self._tool_registry = context.get("tool_registry")
            self._initialized = True

        def _log(self, level, message, **kwargs):
            pass  # suppress output in tests

        def _infer(self, prompt, system_prompt=None, model_id=None,
                   max_tokens=4096, temperature=0.3, expect_json=False):
            return ""

        def _infer_json(self, prompt, system_prompt=None, model_id=None,
                        fallback=None, **kwargs):
            return fallback

        def _search(self, query, max_results=5):
            return []

        def validate_task(self, task):
            if task.agent_type != self._agent_type:
                return False
            return True

        def prepare_task(self, task):
            if not self._initialized:
                self.initialize({})
            task.started_at = "2026-01-01T00:00:00"
            return task

        def complete_task(self, task, result):
            task.completed_at = "2026-01-01T00:00:01"
            task.result = result.output
            if not result.success:
                task.error = "; ".join(result.errors) if result.errors else ""
            return task

        def get_capabilities(self):
            return []

        def get_system_prompt(self):
            return ""

        def get_metadata(self):
            return {
                "agent_type": self._agent_type.value,
                "name": self.name,
            }

        def _execute_safely(self, task, execute_fn):
            if not self.validate_task(task):
                return AgentResult(
                    success=False, output=None,
                    errors=[f"Task validation failed for {self.agent_type}"]
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
            if fallback_fn:
                return fallback_fn()
            return None

        def _incorporate_prior_results(self, task):
            return {}

    mod("vetinari.agents.base_agent", BaseAgent=BaseAgent)
    mod("vetinari.adapters.base", LLMAdapter=MagicMock, InferenceRequest=MagicMock)

    # ---- Analytics / learning / memory stubs ----
    mod("vetinari.analytics.cost",
        get_cost_tracker=MagicMock(return_value=MagicMock()))
    mod("vetinari.analytics.sla",
        get_sla_tracker=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.model_selector",
        get_thompson_selector=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.prompt_evolver",
        get_prompt_evolver=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.quality_scorer",
        get_quality_scorer=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.feedback_loop",
        get_feedback_loop=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.training_data",
        get_training_collector=MagicMock(return_value=MagicMock()))
    mod("vetinari.learning.episode_memory",
        get_episode_memory=MagicMock(return_value=MagicMock()))
    mod("vetinari.memory.dual_memory",
        get_dual_memory_store=MagicMock(return_value=MagicMock()))
    mod("vetinari.shared_memory", shared_memory=MagicMock())
    mod("vetinari.model_pool",
        get_model_pool=MagicMock(return_value=MagicMock()))
    # Load the REAL vetinari/telemetry.py (pure stdlib, no vetinari deps) so
    # later test files (e.g. test_dashboard_api.py) can import TelemetryCollector
    # etc. without hitting a hollow mock module.
    import importlib.util as _tel_ilu
    _tel_path = os.path.join(_HERE, "vetinari", "telemetry.py")
    _tel_spec = _tel_ilu.spec_from_file_location("vetinari.telemetry", _tel_path)
    _tel_mod = _tel_ilu.module_from_spec(_tel_spec)
    sys.modules["vetinari.telemetry"] = _tel_mod   # register BEFORE exec
    _tel_spec.loader.exec_module(_tel_mod)
    mod("vetinari.structured_logging", log_event=MagicMock())
    mod("vetinari.adapter_manager",
        get_adapter_manager=MagicMock(return_value=MagicMock()))
    mod("vetinari.lmstudio_adapter", LMStudioAdapter=MagicMock)
    mod("vetinari.token_optimizer",
        get_token_optimizer=MagicMock(return_value=MagicMock()))
    mod("vetinari.execution_context",
        get_context_manager=MagicMock(return_value=MagicMock()),
        ToolPermission=MagicMock())
    mod("vetinari.constraints.registry",
        get_constraint_registry=MagicMock(return_value=MagicMock()))
    mod("vetinari.config.inference_config",
        get_inference_config=MagicMock(return_value=MagicMock()))
    mod("vetinari.tools.web_search_tool",
        get_search_tool=MagicMock(return_value=MagicMock()))

    # constants stub (needed by builder_agent.py via compat chain)
    mod("vetinari.constants",
        SD_WEBUI_HOST="http://127.0.0.1:7860",
        SD_WEBUI_ENABLED=False,
        SD_DEFAULT_WIDTH=512,
        SD_DEFAULT_HEIGHT=512,
        SD_DEFAULT_STEPS=20,
        SD_DEFAULT_CFG=7.0,
        TIMEOUT_MEDIUM=30)

    return AgentType, AgentTask, AgentResult, VerificationResult, BaseAgent


_TYPES = _stub()
AgentType, AgentTask, AgentResult, VerificationResult, _BaseAgent = _TYPES

# Now import the real agents (dependency-free thanks to stubs above)
from vetinari.agents.context_manager_agent import (  # noqa: E402
    ContextManagerAgent,
    get_context_manager_agent,
)
from vetinari.agents.cost_planner_agent import (  # noqa: E402
    CostPlannerAgent,
    MODEL_PRICING,
    get_cost_planner_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cm_task(operation="consolidate", **ctx_extras) -> AgentTask:
    """Build a minimal AgentTask for ContextManagerAgent."""
    ctx = {"operation": operation}
    ctx.update(ctx_extras)
    return AgentTask(
        task_id="test-cm-001",
        agent_type=AgentType.CONTEXT_MANAGER,
        description="Test context management task",
        prompt="Consolidate context",
        context=ctx,
    )


def _cp_task(**ctx_extras) -> AgentTask:
    """Build a minimal AgentTask for CostPlannerAgent."""
    ctx = {"plan_outputs": "Build a REST API", "usage_stats": {}}
    ctx.update(ctx_extras)
    return AgentTask(
        task_id="test-cp-001",
        agent_type=AgentType.COST_PLANNER,
        description="Analyse costs for the project",
        prompt="Provide cost analysis",
        context=ctx,
    )


# ---------------------------------------------------------------------------
# ContextManagerAgent — 55 tests
# ---------------------------------------------------------------------------

class TestContextManagerAgentInit(unittest.TestCase):
    """Initialisation and basic attribute tests."""

    def setUp(self):
        self.agent = ContextManagerAgent()

    def test_agent_type_is_context_manager(self):
        self.assertEqual(self.agent.agent_type, AgentType.PLANNER)

    def test_default_max_context_tokens(self):
        self.assertEqual(self.agent._max_context_tokens, 4096)

    def test_custom_max_context_tokens(self):
        agent = ContextManagerAgent(config={"max_context_tokens": 8192})
        self.assertEqual(agent._max_context_tokens, 8192)

    def test_env_var_max_context_tokens(self):
        import os
        with patch.dict(os.environ, {"VETINARI_MAX_CONTEXT_TOKENS": "2048"}):
            agent = ContextManagerAgent()
        self.assertEqual(agent._max_context_tokens, 2048)

    def test_config_overrides_env_var(self):
        import os
        with patch.dict(os.environ, {"VETINARI_MAX_CONTEXT_TOKENS": "2048"}):
            agent = ContextManagerAgent(config={"max_context_tokens": 512})
        self.assertEqual(agent._max_context_tokens, 512)

    def test_none_config_accepted(self):
        agent = ContextManagerAgent(config=None)
        self.assertIsNotNone(agent)

    def test_empty_config_accepted(self):
        agent = ContextManagerAgent(config={})
        self.assertIsNotNone(agent)

    def test_agent_not_initialized_at_start(self):
        self.assertFalse(self.agent.is_initialized)


class TestContextManagerAgentSystemPrompt(unittest.TestCase):
    def setUp(self):
        self.agent = ContextManagerAgent()

    def test_system_prompt_is_string(self):
        self.assertIsInstance(self.agent.get_system_prompt(), str)

    def test_system_prompt_non_empty(self):
        self.assertTrue(len(self.agent.get_system_prompt()) > 0)

    def test_system_prompt_mentions_context(self):
        prompt = self.agent.get_system_prompt().lower()
        self.assertIn("context", prompt)

    def test_system_prompt_mentions_memory(self):
        prompt = self.agent.get_system_prompt().lower()
        self.assertIn("memory", prompt)

    def test_system_prompt_mentions_json(self):
        self.assertIn("JSON", self.agent.get_system_prompt())

    def test_system_prompt_mentions_summarise(self):
        prompt = self.agent.get_system_prompt().lower()
        self.assertIn("summar", prompt)


class TestContextManagerAgentCapabilities(unittest.TestCase):
    def setUp(self):
        self.agent = ContextManagerAgent()
        self.caps = self.agent.get_capabilities()

    def test_capabilities_is_list(self):
        self.assertIsInstance(self.caps, list)

    def test_at_least_5_capabilities(self):
        self.assertGreaterEqual(len(self.caps), 5)

    def test_memory_consolidation(self):
        self.assertIn("memory_consolidation", self.caps)

    def test_session_summarisation(self):
        self.assertIn("session_summarisation", self.caps)

    def test_context_window_optimisation(self):
        self.assertIn("context_window_optimisation", self.caps)

    def test_relevance_scoring(self):
        self.assertIn("relevance_scoring", self.caps)

    def test_knowledge_extraction(self):
        self.assertIn("knowledge_extraction", self.caps)

    def test_context_pruning(self):
        self.assertIn("context_pruning", self.caps)

    def test_retrieval_strategy_design(self):
        self.assertIn("retrieval_strategy_design", self.caps)

    def test_contradiction_detection(self):
        self.assertIn("contradiction_detection", self.caps)

    def test_cross_session_continuity(self):
        self.assertIn("cross_session_continuity", self.caps)

    def test_no_duplicate_capabilities(self):
        self.assertEqual(len(self.caps), len(set(self.caps)))


class TestContextManagerAgentExecuteConsolidate(unittest.TestCase):
    """Tests for the consolidate (default) operation path."""

    def setUp(self):
        self.agent = ContextManagerAgent()
        # Stub _infer_json to return a valid consolidation result
        self._valid_result = {
            "consolidated_summary": "Test summary",
            "session_summary": "Session done",
            "key_knowledge": [{"fact": "x", "confidence": 0.9, "source": "s", "relevance": "high"}],
            "entities_discovered": [],
            "patterns_identified": ["p1"],
            "contradictions": [],
            "stale_entries": [],
            "entries_to_retain": [],
            "retrieval_recommendations": [],
            "context_budget_analysis": {"total_entries": 0},
            "next_session_context": "Continue",
            "entries_processed": 0,
        }
        self.agent._infer_json = MagicMock(return_value=self._valid_result)

    def test_execute_returns_agent_result(self):
        task = _cm_task()
        result = self.agent.execute(task)
        self.assertIsInstance(result, AgentResult)

    def test_execute_success_on_valid_llm_response(self):
        task = _cm_task()
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_execute_output_is_dict(self):
        task = _cm_task()
        result = self.agent.execute(task)
        self.assertIsInstance(result.output, dict)

    def test_execute_metadata_contains_operation(self):
        task = _cm_task(operation="consolidate")
        result = self.agent.execute(task)
        self.assertIn("operation", result.metadata)

    def test_execute_metadata_operation_value(self):
        task = _cm_task(operation="consolidate")
        result = self.agent.execute(task)
        self.assertEqual(result.metadata["operation"], "consolidate")

    def test_execute_metadata_entries_processed(self):
        task = _cm_task()
        result = self.agent.execute(task)
        self.assertIn("entries_processed", result.metadata)

    def test_execute_default_operation_is_consolidate(self):
        # task with no "operation" key falls back to consolidate
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.CONTEXT_MANAGER,
            description="desc",
            prompt="p",
            context={},
        )
        result = self.agent.execute(task)
        self.assertIn("operation", result.metadata)

    def test_execute_none_context_falls_back(self):
        task = AgentTask(
            task_id="t2",
            agent_type=AgentType.CONTEXT_MANAGER,
            description="desc",
            prompt="p",
            context=None,
        )
        result = self.agent.execute(task)
        # Should not raise; success depends on fallback path
        self.assertIsNotNone(result)


class TestContextManagerAgentExecuteSummarise(unittest.TestCase):
    def setUp(self):
        self.agent = ContextManagerAgent()
        self._valid_summary = {
            "session_summary": "Summary text",
            "consolidated_summary": "Executive summary",
            "goals_achieved": ["goal1"],
            "goals_pending": [],
            "key_knowledge": [],
            "decisions_made": [],
            "artifacts_produced": [],
            "next_steps": [],
            "retrieval_recommendations": [],
            "entries_processed": 2,
        }
        self.agent._infer_json = MagicMock(return_value=self._valid_summary)

    def test_summarise_operation_success(self):
        task = _cm_task(operation="summarise", history=[{"msg": "hello"}])
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_summarize_us_spelling_success(self):
        task = _cm_task(operation="summarize", history=[{"msg": "hello"}])
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_summarise_with_messages_key(self):
        task = _cm_task(operation="summarise", messages=[{"msg": "a"}, {"msg": "b"}])
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_summarise_no_history_falls_back_to_memory(self):
        # No history in context — loads from memory (returns empty list via stub)
        task = _cm_task(operation="summarise")
        result = self.agent.execute(task)
        self.assertIsNotNone(result)

    def test_summarise_output_has_session_summary(self):
        task = _cm_task(operation="summarise", history=[{"msg": "x"}])
        result = self.agent.execute(task)
        self.assertIn("session_summary", result.output)


class TestContextManagerAgentExecutePrune(unittest.TestCase):
    def setUp(self):
        self.agent = ContextManagerAgent()
        self._valid_prune = {
            "consolidated_summary": "Retained content",
            "session_summary": "Pruning done",
            "entries_to_retain": ["e1"],
            "stale_entries": ["e2"],
            "pruning_rationale": "e2 is stale",
            "estimated_tokens_retained": 100,
            "key_knowledge": [],
            "retrieval_recommendations": [],
            "pruned_count": 1,
            "entries_processed": 2,
        }
        self.agent._infer_json = MagicMock(return_value=self._valid_prune)

    def test_prune_operation_success(self):
        task = _cm_task(operation="prune",
                        entries=[{"id": "e1", "content": "keep"},
                                 {"id": "e2", "content": "old"}])
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_prune_empty_entries_returns_fallback_no_llm(self):
        # No entries — should return fixed fallback dict without calling LLM
        self.agent._infer_json = MagicMock()
        task = _cm_task(operation="prune", entries=[])
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.agent._infer_json.assert_not_called()

    def test_prune_no_entries_key_returns_fallback(self):
        self.agent._infer_json = MagicMock()
        task = _cm_task(operation="prune")  # no "entries" key
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.agent._infer_json.assert_not_called()

    def test_prune_respects_max_tokens_from_context(self):
        task = _cm_task(operation="prune",
                        entries=[{"id": "e1"}],
                        max_tokens=512)
        result = self.agent.execute(task)
        self.assertIsNotNone(result)

    def test_prune_uses_agent_max_tokens_when_not_in_context(self):
        task = _cm_task(operation="prune", entries=[{"id": "e1"}])
        result = self.agent.execute(task)
        self.assertIsNotNone(result)


class TestContextManagerAgentExecuteExtract(unittest.TestCase):
    def setUp(self):
        self.agent = ContextManagerAgent()
        self._valid_extract = {
            "consolidated_summary": "Knowledge extracted",
            "session_summary": "Extraction done",
            "key_knowledge": [{"fact": "f", "confidence": 0.8, "source": "t", "relevance": "high"}],
            "entities_discovered": [{"name": "E", "type": "concept", "attributes": {}}],
            "relationships": [{"subject": "E", "predicate": "is", "object": "concept"}],
            "retrieval_recommendations": [],
            "entries_processed": 1,
        }
        self.agent._infer_json = MagicMock(return_value=self._valid_extract)

    def test_extract_operation_success(self):
        task = _cm_task(operation="extract", text="Python is a programming language.")
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_extract_uses_task_description_when_no_text(self):
        task = AgentTask(
            task_id="t3",
            agent_type=AgentType.CONTEXT_MANAGER,
            description="Extract knowledge from: Python is great",
            prompt="p",
            context={"operation": "extract"},
        )
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_extract_output_has_key_knowledge(self):
        task = _cm_task(operation="extract", text="Important fact here.")
        result = self.agent.execute(task)
        self.assertIn("key_knowledge", result.output)


class TestContextManagerAgentFallback(unittest.TestCase):
    """Verify the fallback path when LLM returns None/empty."""

    def setUp(self):
        self.agent = ContextManagerAgent()
        self.agent._infer_json = MagicMock(return_value=None)

    def test_fallback_returns_dict_on_consolidate(self):
        task = _cm_task(operation="consolidate")
        result = self.agent.execute(task)
        self.assertIsInstance(result.output, dict)

    def test_fallback_has_consolidated_summary(self):
        task = _cm_task(operation="consolidate")
        result = self.agent.execute(task)
        self.assertIn("consolidated_summary", result.output)

    def test_fallback_has_session_summary(self):
        task = _cm_task(operation="consolidate")
        result = self.agent.execute(task)
        self.assertIn("session_summary", result.output)

    def test_fallback_has_retrieval_recommendations(self):
        task = _cm_task(operation="consolidate")
        result = self.agent.execute(task)
        self.assertIn("retrieval_recommendations", result.output)

    def test_fallback_entries_processed_is_int(self):
        task = _cm_task(operation="consolidate")
        result = self.agent.execute(task)
        self.assertIsInstance(result.output.get("entries_processed"), int)

    def test_fallback_on_summarise(self):
        task = _cm_task(operation="summarise", history=[{"m": "x"}, {"m": "y"}])
        result = self.agent.execute(task)
        self.assertTrue(result.success)
        self.assertIn("consolidated_summary", result.output)

    def test_fallback_on_extract(self):
        task = _cm_task(operation="extract", text="some text")
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_fallback_on_unknown_operation_routes_to_consolidate(self):
        task = _cm_task(operation="unknown_op")
        result = self.agent.execute(task)
        self.assertTrue(result.success)


class TestContextManagerAgentExecuteFailure(unittest.TestCase):
    """Error handling when underlying methods raise."""

    def setUp(self):
        self.agent = ContextManagerAgent()

    def test_execute_catches_exception_returns_failure(self):
        self.agent._infer_json = MagicMock(side_effect=RuntimeError("boom"))
        # Also patch _load_memory_entries to raise so _consolidate_memory raises
        with patch.object(self.agent, "_load_memory_entries", side_effect=RuntimeError("boom")):
            task = _cm_task()
            result = self.agent.execute(task)
        self.assertFalse(result.success)

    def test_execute_failure_result_output_is_dict(self):
        with patch.object(self.agent, "_manage_context", side_effect=ValueError("err")):
            task = _cm_task()
            result = self.agent.execute(task)
        self.assertFalse(result.success)
        self.assertEqual(result.output, {})

    def test_execute_failure_has_error_message(self):
        with patch.object(self.agent, "_manage_context", side_effect=ValueError("specific error")):
            task = _cm_task()
            result = self.agent.execute(task)
        self.assertIn("specific error", result.error)


class TestContextManagerAgentVerify(unittest.TestCase):
    def setUp(self):
        self.agent = ContextManagerAgent()

    def _full_output(self):
        return {
            "consolidated_summary": "summary",
            "session_summary": "session",
            "key_knowledge": [{"fact": "f"}],
            "retrieval_recommendations": [{"strategy": "semantic"}],
        }

    def test_verify_non_dict_fails(self):
        result = self.agent.verify("not a dict")
        self.assertFalse(result.passed)

    def test_verify_non_dict_score_zero(self):
        result = self.agent.verify(42)
        self.assertEqual(result.score, 0.0)

    def test_verify_full_output_passes(self):
        result = self.agent.verify(self._full_output())
        self.assertTrue(result.passed)

    def test_verify_missing_both_summaries_reduces_score(self):
        out = {"key_knowledge": [{"f": "v"}], "retrieval_recommendations": [{"s": "x"}]}
        result = self.agent.verify(out)
        self.assertLess(result.score, 1.0)

    def test_verify_missing_key_knowledge_reduces_score(self):
        out = {"consolidated_summary": "s", "retrieval_recommendations": []}
        result = self.agent.verify(out)
        self.assertLess(result.score, 1.0)

    def test_verify_returns_verification_result_instance(self):
        result = self.agent.verify(self._full_output())
        self.assertIsInstance(result, VerificationResult)

    def test_verify_score_gte_zero(self):
        result = self.agent.verify({})
        self.assertGreaterEqual(result.score, 0.0)

    def test_verify_score_lte_one(self):
        result = self.agent.verify(self._full_output())
        self.assertLessEqual(result.score, 1.0)


class TestContextManagerAgentHelpers(unittest.TestCase):
    def setUp(self):
        self.agent = ContextManagerAgent()

    def test_fallback_consolidation_structure(self):
        task = _cm_task()
        fb = self.agent._fallback_consolidation(task, [])
        required = [
            "consolidated_summary", "session_summary", "key_knowledge",
            "entries_to_retain", "retrieval_recommendations",
            "context_budget_analysis", "next_session_context", "entries_processed",
        ]
        for key in required:
            self.assertIn(key, fb, f"Missing key: {key}")

    def test_fallback_entries_to_retain_limited_to_20(self):
        entries = [{"id": str(i)} for i in range(30)]
        task = _cm_task()
        fb = self.agent._fallback_consolidation(task, entries)
        self.assertLessEqual(len(fb["entries_to_retain"]), 20)

    def test_fallback_context_budget_analysis_total_entries(self):
        entries = [{"id": str(i)} for i in range(10)]
        task = _cm_task()
        fb = self.agent._fallback_consolidation(task, entries)
        self.assertEqual(fb["context_budget_analysis"]["total_entries"], 10)

    def test_fallback_recommended_prune_count_non_negative(self):
        entries = [{"id": str(i)} for i in range(5)]
        task = _cm_task()
        fb = self.agent._fallback_consolidation(task, entries)
        self.assertGreaterEqual(fb["context_budget_analysis"]["recommended_prune_count"], 0)

    def test_load_memory_entries_returns_list_when_imports_fail(self):
        # All memory imports are MagicMock'd; ensure it returns a list
        entries = self.agent._load_memory_entries("sid", "pid")
        self.assertIsInstance(entries, list)

    def test_persist_consolidation_does_not_raise(self):
        # Should silently fail if memory import fails
        self.agent._persist_consolidation({"consolidated_summary": "x"}, "s", "p")

    def test_max_entries_for_consolidation_constant(self):
        self.assertEqual(ContextManagerAgent._MAX_ENTRIES_FOR_CONSOLIDATION, 50)


class TestGetContextManagerAgentSingleton(unittest.TestCase):
    def tearDown(self):
        # Reset singleton after each test
        import vetinari.agents.context_manager_agent as cm_mod
        cm_mod._context_manager_agent = None

    def test_returns_context_manager_agent_instance(self):
        agent = get_context_manager_agent()
        self.assertIsInstance(agent, ContextManagerAgent)

    def test_singleton_returns_same_instance(self):
        a1 = get_context_manager_agent()
        a2 = get_context_manager_agent()
        self.assertIs(a1, a2)

    def test_config_passed_to_singleton(self):
        import vetinari.agents.context_manager_agent as cm_mod
        cm_mod._context_manager_agent = None
        agent = get_context_manager_agent(config={"max_context_tokens": 999})
        self.assertEqual(agent._max_context_tokens, 999)


# ---------------------------------------------------------------------------
# CostPlannerAgent — 55 tests
# ---------------------------------------------------------------------------

class TestModelPricing(unittest.TestCase):
    """Tests for the module-level MODEL_PRICING dict."""

    def test_model_pricing_is_dict(self):
        self.assertIsInstance(MODEL_PRICING, dict)

    def test_model_pricing_non_empty(self):
        self.assertGreater(len(MODEL_PRICING), 0)

    def test_all_values_are_float(self):
        for model, price in MODEL_PRICING.items():
            self.assertIsInstance(price, float, f"{model} price is not float")

    def test_all_prices_non_negative(self):
        for model, price in MODEL_PRICING.items():
            self.assertGreaterEqual(price, 0.0, f"{model} has negative price")

    def test_local_models_are_free(self):
        free_models = ["qwen2.5-coder-7b", "qwen2.5-72b", "llama-3.3-70b",
                       "qwen3-30b-a3b", "qwen2.5-vl-32b"]
        for m in free_models:
            self.assertIn(m, MODEL_PRICING)
            self.assertEqual(MODEL_PRICING[m], 0.0, f"{m} should be free")

    def test_claude_sonnet_has_positive_price(self):
        self.assertIn("claude-sonnet-4", MODEL_PRICING)
        self.assertGreater(MODEL_PRICING["claude-sonnet-4"], 0.0)

    def test_claude_opus_has_positive_price(self):
        self.assertIn("claude-opus-4", MODEL_PRICING)
        self.assertGreater(MODEL_PRICING["claude-opus-4"], 0.0)

    def test_claude_haiku_has_positive_price(self):
        self.assertIn("claude-haiku-3", MODEL_PRICING)
        self.assertGreater(MODEL_PRICING["claude-haiku-3"], 0.0)

    def test_gpt4o_has_positive_price(self):
        self.assertIn("gpt-4o", MODEL_PRICING)
        self.assertGreater(MODEL_PRICING["gpt-4o"], 0.0)

    def test_gpt4o_mini_has_positive_price(self):
        self.assertIn("gpt-4o-mini", MODEL_PRICING)
        self.assertGreater(MODEL_PRICING["gpt-4o-mini"], 0.0)

    def test_gemini_flash_is_free(self):
        self.assertIn("gemini-2.0-flash", MODEL_PRICING)
        self.assertEqual(MODEL_PRICING["gemini-2.0-flash"], 0.0)

    def test_gemini_pro_has_price(self):
        self.assertIn("gemini-1.5-pro", MODEL_PRICING)
        self.assertGreater(MODEL_PRICING["gemini-1.5-pro"], 0.0)

    def test_command_r_plus_has_price(self):
        self.assertIn("command-r-plus", MODEL_PRICING)
        self.assertGreater(MODEL_PRICING["command-r-plus"], 0.0)

    def test_opus_more_expensive_than_haiku(self):
        self.assertGreater(MODEL_PRICING["claude-opus-4"],
                           MODEL_PRICING["claude-haiku-3"])

    def test_sonnet_between_haiku_and_opus(self):
        self.assertGreater(MODEL_PRICING["claude-sonnet-4"],
                           MODEL_PRICING["claude-haiku-3"])
        self.assertLess(MODEL_PRICING["claude-sonnet-4"],
                        MODEL_PRICING["claude-opus-4"])

    def test_all_model_names_are_strings(self):
        for k in MODEL_PRICING:
            self.assertIsInstance(k, str)

    def test_at_least_10_models_listed(self):
        self.assertGreaterEqual(len(MODEL_PRICING), 10)


class TestCostPlannerAgentInit(unittest.TestCase):
    def setUp(self):
        self.agent = CostPlannerAgent()

    def test_agent_type_is_cost_planner(self):
        self.assertEqual(self.agent.agent_type, AgentType.PLANNER)

    def test_none_config_accepted(self):
        agent = CostPlannerAgent(config=None)
        self.assertIsNotNone(agent)

    def test_empty_config_accepted(self):
        agent = CostPlannerAgent(config={})
        self.assertIsNotNone(agent)

    def test_agent_not_initialized_at_start(self):
        self.assertFalse(self.agent.is_initialized)


class TestCostPlannerAgentSystemPrompt(unittest.TestCase):
    def setUp(self):
        self.agent = CostPlannerAgent()

    def test_system_prompt_is_string(self):
        self.assertIsInstance(self.agent.get_system_prompt(), str)

    def test_system_prompt_non_empty(self):
        self.assertTrue(len(self.agent.get_system_prompt()) > 0)

    def test_system_prompt_mentions_cost(self):
        self.assertIn("cost", self.agent.get_system_prompt().lower())

    def test_system_prompt_mentions_json(self):
        self.assertIn("JSON", self.agent.get_system_prompt())

    def test_system_prompt_mentions_model(self):
        self.assertIn("model", self.agent.get_system_prompt().lower())

    def test_system_prompt_mentions_budget(self):
        self.assertIn("budget", self.agent.get_system_prompt().lower())

    def test_system_prompt_mentions_optimis(self):
        self.assertIn("optim", self.agent.get_system_prompt().lower())

    def test_system_prompt_contains_cost_report_key(self):
        self.assertIn("cost_report", self.agent.get_system_prompt())

    def test_system_prompt_contains_model_recommendations_key(self):
        self.assertIn("model_recommendations", self.agent.get_system_prompt())


class TestCostPlannerAgentCapabilities(unittest.TestCase):
    def setUp(self):
        self.agent = CostPlannerAgent()
        self.caps = self.agent.get_capabilities()

    def test_capabilities_is_list(self):
        self.assertIsInstance(self.caps, list)

    def test_cost_calculation(self):
        self.assertIn("cost_calculation", self.caps)

    def test_budget_planning(self):
        self.assertIn("budget_planning", self.caps)

    def test_model_selection(self):
        self.assertIn("model_selection", self.caps)

    def test_usage_tracking(self):
        self.assertIn("usage_tracking", self.caps)

    def test_cost_reporting(self):
        self.assertIn("cost_reporting", self.caps)

    def test_optimization_analysis(self):
        self.assertIn("optimization_analysis", self.caps)

    def test_no_duplicate_capabilities(self):
        self.assertEqual(len(self.caps), len(set(self.caps)))


class TestCostPlannerAgentExecuteSuccess(unittest.TestCase):
    """Tests for the happy path."""

    def setUp(self):
        self.agent = CostPlannerAgent()
        self._valid_analysis = {
            "cost_report": {
                "total_tokens_used": 1000,
                "total_cost_usd": 0.05,
                "cost_by_model": {"claude-sonnet-4": 0.05},
                "currency": "USD",
                "trend": "stable",
            },
            "model_recommendations": [
                {
                    "task_type": "planning",
                    "current_model": "claude-sonnet-4",
                    "recommended_model": "qwen2.5-72b",
                    "estimated_savings_percent": 100,
                }
            ],
            "budget_constraints": {"local_only_mode": True},
            "optimizations": [{"technique": "caching", "estimated_savings_percent": 85}],
            "token_efficiency_analysis": {"avg_tokens_per_task": 500},
            "summary": "Analysis complete",
        }
        self.agent._infer_json = MagicMock(return_value=self._valid_analysis)
        self.agent._search = MagicMock(return_value=[])

    def test_execute_returns_agent_result(self):
        result = self.agent.execute(_cp_task())
        self.assertIsInstance(result, AgentResult)

    def test_execute_success_true(self):
        result = self.agent.execute(_cp_task())
        self.assertTrue(result.success)

    def test_execute_output_is_dict(self):
        result = self.agent.execute(_cp_task())
        self.assertIsInstance(result.output, dict)

    def test_execute_output_has_cost_report(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("cost_report", result.output)

    def test_execute_output_has_model_recommendations(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("model_recommendations", result.output)

    def test_execute_output_has_budget_constraints(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("budget_constraints", result.output)

    def test_execute_output_has_optimizations(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("optimizations", result.output)

    def test_execute_output_has_summary(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("summary", result.output)

    def test_execute_metadata_contains_total_cost(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("total_cost", result.metadata)

    def test_execute_metadata_contains_recommendations_count(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("recommendations_count", result.metadata)

    def test_execute_metadata_contains_optimizations_count(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("optimizations_count", result.metadata)

    def test_execute_metadata_recommendations_count_correct(self):
        result = self.agent.execute(_cp_task())
        self.assertEqual(result.metadata["recommendations_count"], 1)

    def test_execute_plan_outputs_from_context(self):
        task = _cp_task(plan_outputs="Build a microservice")
        result = self.agent.execute(task)
        self.assertTrue(result.success)

    def test_execute_usage_stats_from_context(self):
        task = _cp_task(usage_stats={"total_tokens": 5000})
        result = self.agent.execute(task)
        self.assertTrue(result.success)


class TestCostPlannerAgentExecuteValidation(unittest.TestCase):
    """Validation and edge cases."""

    def setUp(self):
        self.agent = CostPlannerAgent()
        self.agent._search = MagicMock(return_value=[])
        # _infer_json must honour the fallback= kwarg so the fallback analysis
        # is actually returned (mirrors real BaseAgent._infer_json behaviour
        # when the LLM is unavailable).
        def _infer_json_returning_fallback(prompt, fallback=None, **kwargs):
            return fallback
        self.agent._infer_json = MagicMock(side_effect=_infer_json_returning_fallback)

    def test_wrong_agent_type_returns_failure(self):
        task = AgentTask(
            task_id="t-bad",
            agent_type=AgentType.PLANNER,  # wrong type
            description="desc",
            prompt="p",
            context={},
        )
        result = self.agent.execute(task)
        self.assertFalse(result.success)

    def test_wrong_agent_type_errors_populated(self):
        task = AgentTask(
            task_id="t-bad",
            agent_type=AgentType.PLANNER,
            description="desc",
            prompt="p",
            context={},
        )
        result = self.agent.execute(task)
        self.assertTrue(len(result.errors) > 0)

    def test_empty_usage_stats_accepted(self):
        task = _cp_task(usage_stats={})
        result = self.agent.execute(task)
        # _infer_json returns fallback dict — should succeed
        self.assertTrue(result.success)

    def test_no_usage_stats_key_uses_description(self):
        task = AgentTask(
            task_id="t5",
            agent_type=AgentType.COST_PLANNER,
            description="Build a web app",
            prompt="p",
            context={},
        )
        result = self.agent.execute(task)
        self.assertTrue(result.success)


class TestCostPlannerAgentFallback(unittest.TestCase):
    """Verify the fallback analysis structure when LLM returns None."""

    def setUp(self):
        self.agent = CostPlannerAgent()
        self.agent._search = MagicMock(return_value=[])
        # Honour the fallback= kwarg so _fallback_analysis is actually returned
        def _infer_json_returning_fallback(prompt, fallback=None, **kwargs):
            return fallback
        self.agent._infer_json = MagicMock(side_effect=_infer_json_returning_fallback)

    def test_fallback_result_is_success(self):
        result = self.agent.execute(_cp_task())
        self.assertTrue(result.success)

    def test_fallback_has_cost_report(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("cost_report", result.output)

    def test_fallback_cost_report_has_currency(self):
        result = self.agent.execute(_cp_task())
        self.assertEqual(result.output["cost_report"]["currency"], "USD")

    def test_fallback_has_model_recommendations(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("model_recommendations", result.output)

    def test_fallback_model_recommendations_is_list(self):
        result = self.agent.execute(_cp_task())
        self.assertIsInstance(result.output["model_recommendations"], list)

    def test_fallback_has_budget_constraints(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("budget_constraints", result.output)

    def test_fallback_local_only_mode_true(self):
        result = self.agent.execute(_cp_task())
        self.assertTrue(result.output["budget_constraints"]["local_only_mode"])

    def test_fallback_has_optimizations(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("optimizations", result.output)

    def test_fallback_optimizations_is_list(self):
        result = self.agent.execute(_cp_task())
        self.assertIsInstance(result.output["optimizations"], list)

    def test_fallback_has_token_efficiency_analysis(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("token_efficiency_analysis", result.output)

    def test_fallback_has_summary(self):
        result = self.agent.execute(_cp_task())
        self.assertIn("summary", result.output)

    def test_fallback_summary_is_string(self):
        result = self.agent.execute(_cp_task())
        self.assertIsInstance(result.output["summary"], str)


class TestCostPlannerAgentExecuteFailure(unittest.TestCase):
    def setUp(self):
        self.agent = CostPlannerAgent()

    def test_exception_in_collect_returns_failure(self):
        self.agent._search = MagicMock(return_value=[])
        with patch.object(self.agent, "_collect_real_usage_data",
                          side_effect=RuntimeError("crash")):
            result = self.agent.execute(_cp_task())
        self.assertFalse(result.success)

    def test_failure_result_has_errors_list(self):
        self.agent._search = MagicMock(return_value=[])
        with patch.object(self.agent, "_collect_real_usage_data",
                          side_effect=RuntimeError("crash")):
            result = self.agent.execute(_cp_task())
        self.assertIsInstance(result.errors, list)
        self.assertTrue(len(result.errors) > 0)

    def test_failure_result_output_is_none(self):
        self.agent._search = MagicMock(return_value=[])
        with patch.object(self.agent, "_collect_real_usage_data",
                          side_effect=RuntimeError("crash")):
            result = self.agent.execute(_cp_task())
        self.assertIsNone(result.output)


class TestCostPlannerAgentVerify(unittest.TestCase):
    def setUp(self):
        self.agent = CostPlannerAgent()

    def _full_output(self):
        return {
            "cost_report": {"total_cost_usd": 0.0, "currency": "USD"},
            "model_recommendations": [{"task_type": "x"}],
            "budget_constraints": {"local_only_mode": True},
            "optimizations": [{"technique": "caching"}],
            "token_efficiency_analysis": {},
            "summary": "done",
        }

    def test_verify_full_output_passes(self):
        result = self.agent.verify(self._full_output())
        self.assertTrue(result.passed)

    def test_verify_non_dict_fails(self):
        result = self.agent.verify("not a dict")
        self.assertFalse(result.passed)

    def test_verify_non_dict_score_zero(self):
        result = self.agent.verify(None)
        self.assertEqual(result.score, 0.0)

    def test_verify_missing_cost_report_reduces_score(self):
        out = self._full_output()
        del out["cost_report"]
        result = self.agent.verify(out)
        self.assertLess(result.score, 1.0)

    def test_verify_missing_recommendations_reduces_score(self):
        out = self._full_output()
        out["model_recommendations"] = []
        result = self.agent.verify(out)
        self.assertLess(result.score, 1.0)

    def test_verify_missing_budget_constraints_reduces_score(self):
        out = self._full_output()
        out["budget_constraints"] = {}
        result = self.agent.verify(out)
        self.assertLess(result.score, 1.0)

    def test_verify_missing_optimizations_reduces_score(self):
        out = self._full_output()
        out["optimizations"] = []
        result = self.agent.verify(out)
        self.assertLess(result.score, 1.0)

    def test_verify_returns_verification_result(self):
        result = self.agent.verify(self._full_output())
        self.assertIsInstance(result, VerificationResult)

    def test_verify_issues_is_list(self):
        result = self.agent.verify({})
        self.assertIsInstance(result.issues, list)

    def test_verify_score_non_negative(self):
        result = self.agent.verify({})
        self.assertGreaterEqual(result.score, 0.0)


class TestCostPlannerAgentHelpers(unittest.TestCase):
    def setUp(self):
        self.agent = CostPlannerAgent()

    def test_format_usage_empty_returns_no_usage_string(self):
        result = self.agent._format_usage({})
        self.assertIn("No usage data", result)

    def test_format_usage_with_data_returns_string(self):
        result = self.agent._format_usage({"total_tokens": 1000})
        self.assertIn("total_tokens", result)

    def test_format_pricing_contains_local_label(self):
        result = self.agent._format_pricing()
        self.assertIn("LOCAL", result)

    def test_format_pricing_contains_dollar_sign(self):
        result = self.agent._format_pricing()
        self.assertIn("$", result)

    def test_format_pricing_is_string(self):
        self.assertIsInstance(self.agent._format_pricing(), str)

    def test_fallback_analysis_structure(self):
        fb = self.agent._fallback_analysis({}, "Test project")
        required = ["cost_report", "model_recommendations", "budget_constraints",
                    "optimizations", "token_efficiency_analysis", "summary"]
        for key in required:
            self.assertIn(key, fb, f"Missing key: {key}")

    def test_fallback_analysis_cost_report_currency_usd(self):
        fb = self.agent._fallback_analysis({}, "Test")
        self.assertEqual(fb["cost_report"]["currency"], "USD")

    def test_fallback_analysis_total_cost_zero_local(self):
        fb = self.agent._fallback_analysis({}, "Test")
        self.assertEqual(fb["cost_report"]["total_cost_usd"], 0.0)

    def test_collect_real_usage_data_returns_dict(self):
        result = self.agent._collect_real_usage_data({"base": "data"})
        self.assertIsInstance(result, dict)

    def test_collect_real_usage_data_includes_provided_stats(self):
        result = self.agent._collect_real_usage_data({"my_key": "my_val"})
        self.assertIn("my_key", result)

    def test_collect_real_usage_does_not_raise(self):
        # Even if analytics imports fail gracefully
        result = self.agent._collect_real_usage_data({})
        self.assertIsInstance(result, dict)


class TestGetCostPlannerAgentSingleton(unittest.TestCase):
    def tearDown(self):
        import vetinari.agents.cost_planner_agent as cp_mod
        cp_mod._cost_planner_agent = None

    def test_returns_cost_planner_agent_instance(self):
        agent = get_cost_planner_agent()
        self.assertIsInstance(agent, CostPlannerAgent)

    def test_singleton_returns_same_instance(self):
        a1 = get_cost_planner_agent()
        a2 = get_cost_planner_agent()
        self.assertIs(a1, a2)

    def test_config_passed_to_singleton(self):
        import vetinari.agents.cost_planner_agent as cp_mod
        cp_mod._cost_planner_agent = None
        agent = get_cost_planner_agent(config={"custom": "val"})
        self.assertEqual(agent._config.get("custom"), "val")


if __name__ == "__main__":
    unittest.main()
