"""
Comprehensive pytest tests for vetinari/orchestration/agent_graph.py

Tests cover:
- ExecutionStrategy enum values
- TaskNode dataclass
- ExecutionPlan dataclass
- AgentGraph init, initialize, create_execution_plan
- Topological sort (simple chain, parallel, cycle detection)
- execute_plan (sequential, parallel, adaptive)
- _build_execution_layers
- _execute_task_node (success, failure, self-correction, maker-checker, error recovery)
- _apply_maker_checker
- _validate_output_schema
- inject_task
- get_agent / get_registered_agents / get_agent_by_capability / get_skill_spec / get_agents_for_task_type
- get_agent_graph singleton
"""

from __future__ import annotations

import sys
import types
import asyncio
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Stub heavy external modules BEFORE importing anything from vetinari
# ---------------------------------------------------------------------------

def _make_mock_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    return mod


def _make_package_module(name: str) -> types.ModuleType:
    """Create a stub module that also acts as a package (has __path__)."""
    mod = types.ModuleType(name)
    mod.__path__ = []          # marks it as a package
    mod.__package__ = name
    return mod


def _install_stubs() -> None:
    """Install all required sys.modules stubs so vetinari imports don't fail."""

    # vetinari must be a package (has __path__) so sub-packages resolve.
    # Use the real on-disk path so Python can find submodules inside it.
    import os as _os
    import importlib.util as _ilu
    from enum import Enum
    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    # ---- vetinari.types ----
    # Load the REAL vetinari/types.py — it only imports stdlib (enum), so it is
    # safe to import directly without side-effects.  This ensures that ALL enums
    # (AgentType, TaskStatus, PlanStatus, SubtaskStatus, …) are available to any
    # transitive import (e.g. plan_types.py) that runs in the same pytest session.
    _vtypes_path = _os.path.join(_ROOT, "vetinari", "types.py")
    _vtypes_spec = _ilu.spec_from_file_location("vetinari.types", _vtypes_path)
    types_mod = _ilu.module_from_spec(_vtypes_spec)
    _vtypes_spec.loader.exec_module(types_mod)

    AgentType = types_mod.AgentType
    TaskStatus = types_mod.TaskStatus
    ExecutionMode = types_mod.ExecutionMode

    vetinari_pkg = _make_package_module("vetinari")
    vetinari_pkg.__path__ = [_os.path.join(_ROOT, "vetinari")]
    vetinari_pkg.types = types_mod
    sys.modules["vetinari"] = vetinari_pkg
    sys.modules["vetinari.types"] = types_mod

    # vetinari.orchestration must point to the real directory so that
    # `import vetinari.orchestration.agent_graph` can find the real file.
    orch_pkg = _make_package_module("vetinari.orchestration")
    orch_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "orchestration")]
    sys.modules["vetinari.orchestration"] = orch_pkg

    # ---- vetinari.agents.contracts ----
    contracts_mod = _make_mock_module("vetinari.agents.contracts")

    import uuid
    from dataclasses import dataclass as dc, field as f_field
    from datetime import datetime

    @dc
    class AgentSpec:
        agent_type: Any
        name: str
        description: str
        default_model: str
        thinking_variant: str = "medium"
        enabled: bool = True
        system_prompt: str = ""
        version: str = "1.0.0"

    @dc
    class Task:
        id: str
        description: str
        inputs: List[str] = f_field(default_factory=list)
        outputs: List[str] = f_field(default_factory=list)
        dependencies: List[str] = f_field(default_factory=list)
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

    @dc
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
        dependencies: List[str] = f_field(default_factory=list)
        context: Dict[str, Any] = f_field(default_factory=dict)

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

    @dc
    class Plan:
        plan_id: str
        version: str = "v0.1.0"
        goal: str = ""
        phase: int = 0
        tasks: List[Any] = f_field(default_factory=list)
        model_scores: List[Dict] = f_field(default_factory=list)
        notes: str = ""
        warnings: List[str] = f_field(default_factory=list)
        needs_context: bool = False
        follow_up_question: str = ""
        final_delivery_path: str = ""
        final_delivery_summary: str = ""
        created_at: str = f_field(default_factory=lambda: datetime.now().isoformat())

        @classmethod
        def create_new(cls, goal: str, phase: int = 0) -> "Plan":
            return cls(plan_id=f"plan_{uuid.uuid4().hex[:8]}", goal=goal, phase=phase)

    @dc
    class AgentResult:
        success: bool
        output: Any
        metadata: Dict[str, Any] = f_field(default_factory=dict)
        errors: List[str] = f_field(default_factory=list)
        provenance: List[Dict] = f_field(default_factory=list)

    @dc
    class VerificationResult:
        passed: bool
        issues: List[Dict[str, Any]] = f_field(default_factory=list)
        suggestions: List[str] = f_field(default_factory=list)
        score: float = 0.0

    AGENT_REGISTRY = {}

    def get_agent_spec(agent_type):
        return AGENT_REGISTRY.get(agent_type)

    contracts_mod.AgentType = AgentType
    contracts_mod.TaskStatus = TaskStatus
    contracts_mod.ExecutionMode = ExecutionMode
    contracts_mod.AgentSpec = AgentSpec
    contracts_mod.Task = Task
    contracts_mod.AgentTask = AgentTask
    contracts_mod.Plan = Plan
    contracts_mod.AgentResult = AgentResult
    contracts_mod.VerificationResult = VerificationResult
    contracts_mod.AGENT_REGISTRY = AGENT_REGISTRY
    contracts_mod.get_agent_spec = get_agent_spec
    agents_pkg = _make_package_module("vetinari.agents")
    # Set real __path__ so later test files can import vetinari.agents.* submodules
    agents_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "agents")]
    sys.modules["vetinari.agents"] = agents_pkg
    sys.modules["vetinari.agents.contracts"] = contracts_mod

    # ---- vetinari.constraints ----
    constraints_mod = _make_package_module("vetinari.constraints")
    # Set real __path__ so later test files can import vetinari.constraints.* submodules
    constraints_mod.__path__ = [_os.path.join(_ROOT, "vetinari", "constraints")]
    constraints_registry_mod = _make_mock_module("vetinari.constraints.registry")
    constraints_registry_mod.get_constraint_registry = MagicMock(return_value=None)
    sys.modules["vetinari.constraints"] = constraints_mod
    sys.modules["vetinari.constraints.registry"] = constraints_registry_mod

    # ---- vetinari.memory ----
    # Load the REAL vetinari.memory package (stdlib + sqlite only) so that
    # DualMemoryStore, MemoryStore, etc. are available for test_enhanced_memory.py
    # and other test files that need the real memory classes.
    _mem_init = _os.path.join(_ROOT, "vetinari", "memory", "__init__.py")
    _mem_dir = _os.path.join(_ROOT, "vetinari", "memory")
    _mem_spec = _ilu.spec_from_file_location(
        "vetinari.memory", _mem_init,
        submodule_search_locations=[_mem_dir],
    )
    memory_mod = _ilu.module_from_spec(_mem_spec)
    sys.modules["vetinari.memory"] = memory_mod
    _mem_spec.loader.exec_module(memory_mod)

    # ---- vetinari.blackboard ----
    # Load the REAL blackboard module (stdlib + sqlite only) so that
    # test_blackboard.py can still import Blackboard/BlackboardEntry in the
    # same pytest session.  Override only get_blackboard with a MagicMock so
    # agent_graph tests can control delegation behaviour.
    _bb_path = _os.path.join(_ROOT, "vetinari", "blackboard.py")
    _bb_spec = _ilu.spec_from_file_location("vetinari.blackboard", _bb_path)
    blackboard_mod = _ilu.module_from_spec(_bb_spec)
    # Register BEFORE exec_module so dataclass annotation resolution
    # (sys.modules.get(cls.__module__).__dict__) finds the module, not None.
    sys.modules["vetinari.blackboard"] = blackboard_mod
    _bb_spec.loader.exec_module(blackboard_mod)
    # Re-fetch: the shim replaces sys.modules["vetinari.blackboard"] with the canonical module
    blackboard_mod = sys.modules["vetinari.blackboard"]
    blackboard_mod.get_blackboard = MagicMock(return_value=MagicMock(delegate=MagicMock(return_value=None)))

    # ---- vetinari.execution_context ----
    # Load the REAL execution_context.py — it only imports from vetinari.types
    # (already in sys.modules) and stdlib, so it is safe to load directly.
    # This preserves ExecutionMode, ToolPermission, ExecutionContext etc. for
    # later test files (e.g. test_architect_skill.py) that need the real classes.
    _ec_path = _os.path.join(_ROOT, "vetinari", "execution_context.py")
    _ec_spec = _ilu.spec_from_file_location("vetinari.execution_context", _ec_path)
    exec_ctx_mod = _ilu.module_from_spec(_ec_spec)
    sys.modules["vetinari.execution_context"] = exec_ctx_mod  # register BEFORE exec
    _ec_spec.loader.exec_module(exec_ctx_mod)
    # Add MagicMock for get_context_manager so agent_graph.py tests work
    exec_ctx_mod.get_context_manager = MagicMock(return_value=MagicMock(enforce_permission=MagicMock()))

    # ---- vetinari.skills ----
    skills_mod = _make_package_module("vetinari.skills")
    # Set real __path__ so later test files can import vetinari.skills.* submodules
    # (mirrors how vetinari_pkg.__path__ is set above)
    skills_mod.__path__ = [_os.path.join(_ROOT, "vetinari", "skills")]
    skill_registry_mod = _make_mock_module("vetinari.skills.skill_registry")
    skill_registry_mod.get_skills_by_capability = MagicMock(return_value=[])
    skill_registry_mod.get_skill_for_agent_type = MagicMock(return_value=None)
    skill_registry_mod.get_all_skills = MagicMock(return_value={})
    sys.modules["vetinari.skills"] = skills_mod
    sys.modules["vetinari.skills.skill_registry"] = skill_registry_mod

    # Make the AgentType / TaskStatus available at module level for tests
    _install_stubs.AgentType = AgentType
    _install_stubs.TaskStatus = TaskStatus
    _install_stubs.Task = Task
    _install_stubs.Plan = Plan
    _install_stubs.AgentTask = AgentTask
    _install_stubs.AgentResult = AgentResult
    _install_stubs.VerificationResult = VerificationResult


# Run stubs immediately
_install_stubs()

# Re-export stubs for easy import in tests
_s = _install_stubs
AgentType = _s.AgentType
TaskStatus = _s.TaskStatus
Task = _s.Task
Plan = _s.Plan
AgentTask = _s.AgentTask
AgentResult = _s.AgentResult
VerificationResult = _s.VerificationResult

# Now import the module under test
import vetinari.orchestration.agent_graph as ag_module
from vetinari.orchestration.agent_graph import (
    ExecutionStrategy,
    TaskNode,
    ExecutionPlan,
    AgentGraph,
    get_agent_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent(agent_type, success=True, verification_passed=True, output="ok"):
    """Create a lightweight mock agent."""
    agent = MagicMock()
    agent.agent_type = agent_type
    result = AgentResult(success=success, output=output)
    verification = VerificationResult(passed=verification_passed)
    agent.execute.return_value = result
    agent.verify.return_value = verification
    agent.initialize.return_value = None
    return agent


def _make_plan(goal="test goal", tasks=None):
    plan = Plan.create_new(goal)
    if tasks:
        plan.tasks = tasks
    return plan


def _make_task(tid, agent_type=None, deps=None, description="task"):
    if agent_type is None:
        agent_type = AgentType.EXPLORER
    return Task(
        id=tid,
        description=description,
        assigned_agent=agent_type,
        dependencies=deps or [],
    )


def _make_graph_with_agents(strategy=ExecutionStrategy.SEQUENTIAL, agent_types=None):
    """Return an AgentGraph with mock agents pre-registered (no real initialize call)."""
    graph = AgentGraph(strategy=strategy)
    if agent_types:
        for at in agent_types:
            graph._agents[at] = _make_mock_agent(at)
    graph._initialized = True
    return graph


# ---------------------------------------------------------------------------
# 1. ExecutionStrategy enum
# ---------------------------------------------------------------------------

class TestExecutionStrategyEnum(unittest.TestCase):

    def test_sequential_value(self):
        self.assertEqual(ExecutionStrategy.SEQUENTIAL.value, "sequential")

    def test_parallel_value(self):
        self.assertEqual(ExecutionStrategy.PARALLEL.value, "parallel")

    def test_adaptive_value(self):
        self.assertEqual(ExecutionStrategy.ADAPTIVE.value, "adaptive")

    def test_enum_members_count(self):
        self.assertEqual(len(ExecutionStrategy), 3)

    def test_enum_identity_sequential(self):
        self.assertIs(ExecutionStrategy.SEQUENTIAL, ExecutionStrategy["SEQUENTIAL"])

    def test_enum_identity_parallel(self):
        self.assertIs(ExecutionStrategy.PARALLEL, ExecutionStrategy["PARALLEL"])

    def test_enum_identity_adaptive(self):
        self.assertIs(ExecutionStrategy.ADAPTIVE, ExecutionStrategy["ADAPTIVE"])


# ---------------------------------------------------------------------------
# 2. TaskNode dataclass
# ---------------------------------------------------------------------------

class TestTaskNode(unittest.TestCase):

    def _task(self, tid="t1"):
        return _make_task(tid)

    def test_default_status_is_pending(self):
        node = TaskNode(task=self._task())
        self.assertEqual(node.status, TaskStatus.PENDING)

    def test_default_result_is_none(self):
        node = TaskNode(task=self._task())
        self.assertIsNone(node.result)

    def test_default_dependencies_empty(self):
        node = TaskNode(task=self._task())
        self.assertIsInstance(node.dependencies, set)
        self.assertEqual(len(node.dependencies), 0)

    def test_default_dependents_empty(self):
        node = TaskNode(task=self._task())
        self.assertIsInstance(node.dependents, set)
        self.assertEqual(len(node.dependents), 0)

    def test_default_retries_zero(self):
        node = TaskNode(task=self._task())
        self.assertEqual(node.retries, 0)

    def test_default_max_retries_three(self):
        node = TaskNode(task=self._task())
        self.assertEqual(node.max_retries, 3)

    def test_custom_dependencies(self):
        node = TaskNode(task=self._task(), dependencies={"t0", "t-1"})
        self.assertIn("t0", node.dependencies)
        self.assertIn("t-1", node.dependencies)

    def test_custom_max_retries(self):
        node = TaskNode(task=self._task(), max_retries=5)
        self.assertEqual(node.max_retries, 5)

    def test_task_stored(self):
        t = self._task("myid")
        node = TaskNode(task=t)
        self.assertIs(node.task, t)

    def test_status_assignment(self):
        node = TaskNode(task=self._task(), status=TaskStatus.RUNNING)
        self.assertEqual(node.status, TaskStatus.RUNNING)

    def test_retries_mutable(self):
        node = TaskNode(task=self._task())
        node.retries += 1
        self.assertEqual(node.retries, 1)

    def test_result_assignment(self):
        node = TaskNode(task=self._task())
        r = AgentResult(success=True, output="x")
        node.result = r
        self.assertIs(node.result, r)

    def test_dependencies_are_independent_per_instance(self):
        n1 = TaskNode(task=self._task("a"))
        n2 = TaskNode(task=self._task("b"))
        n1.dependencies.add("x")
        self.assertNotIn("x", n2.dependencies)


# ---------------------------------------------------------------------------
# 3. ExecutionPlan dataclass
# ---------------------------------------------------------------------------

class TestExecutionPlan(unittest.TestCase):

    def _plan(self):
        return Plan.create_new("goal")

    def test_fields_set_on_creation(self):
        p = self._plan()
        ep = ExecutionPlan(plan_id="pid", original_plan=p)
        self.assertEqual(ep.plan_id, "pid")
        self.assertIs(ep.original_plan, p)

    def test_default_nodes_empty(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        self.assertIsInstance(ep.nodes, dict)
        self.assertEqual(len(ep.nodes), 0)

    def test_default_execution_order_empty(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        self.assertIsInstance(ep.execution_order, list)
        self.assertEqual(len(ep.execution_order), 0)

    def test_default_status_pending(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        self.assertEqual(ep.status, TaskStatus.PENDING)

    def test_default_started_at_none(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        self.assertIsNone(ep.started_at)

    def test_default_completed_at_none(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        self.assertIsNone(ep.completed_at)

    def test_nodes_dict_is_independent(self):
        ep1 = ExecutionPlan(plan_id="p1", original_plan=self._plan())
        ep2 = ExecutionPlan(plan_id="p2", original_plan=self._plan())
        ep1.nodes["x"] = MagicMock()
        self.assertNotIn("x", ep2.nodes)

    def test_status_mutation(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        ep.status = TaskStatus.RUNNING
        self.assertEqual(ep.status, TaskStatus.RUNNING)


# ---------------------------------------------------------------------------
# 4. AgentGraph.__init__
# ---------------------------------------------------------------------------

class TestAgentGraphInit(unittest.TestCase):

    def test_default_strategy_is_adaptive(self):
        g = AgentGraph()
        self.assertEqual(g._strategy, ExecutionStrategy.ADAPTIVE)

    def test_custom_strategy(self):
        g = AgentGraph(strategy=ExecutionStrategy.SEQUENTIAL)
        self.assertEqual(g._strategy, ExecutionStrategy.SEQUENTIAL)

    def test_default_max_workers(self):
        g = AgentGraph()
        self.assertEqual(g._max_workers, 5)

    def test_custom_max_workers(self):
        g = AgentGraph(max_workers=8)
        self.assertEqual(g._max_workers, 8)

    def test_agents_dict_empty(self):
        g = AgentGraph()
        self.assertIsInstance(g._agents, dict)
        self.assertEqual(len(g._agents), 0)

    def test_execution_plans_empty(self):
        g = AgentGraph()
        self.assertIsInstance(g._execution_plans, dict)
        self.assertEqual(len(g._execution_plans), 0)

    def test_not_initialized(self):
        g = AgentGraph()
        self.assertFalse(g._initialized)

    def test_parallel_strategy_graph(self):
        g = AgentGraph(strategy=ExecutionStrategy.PARALLEL, max_workers=10)
        self.assertEqual(g._strategy, ExecutionStrategy.PARALLEL)
        self.assertEqual(g._max_workers, 10)


# ---------------------------------------------------------------------------
# 5. AgentGraph.initialize()
# ---------------------------------------------------------------------------

class TestAgentGraphInitialize(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None

    def tearDown(self):
        ag_module._agent_graph = None

    def _mock_getter(self, agent_type):
        agent = _make_mock_agent(agent_type)
        return MagicMock(return_value=agent)

    def test_initialize_sets_initialized_flag(self):
        g = AgentGraph()
        all_agent_types = [
            AgentType.PLANNER, AgentType.EXPLORER, AgentType.ORACLE,
            AgentType.LIBRARIAN, AgentType.RESEARCHER, AgentType.EVALUATOR,
            AgentType.SYNTHESIZER, AgentType.BUILDER, AgentType.UI_PLANNER,
            AgentType.SECURITY_AUDITOR, AgentType.DATA_ENGINEER,
            AgentType.DOCUMENTATION_AGENT, AgentType.COST_PLANNER,
            AgentType.TEST_AUTOMATION, AgentType.EXPERIMENTATION_MANAGER,
            AgentType.IMPROVEMENT, AgentType.USER_INTERACTION,
            AgentType.DEVOPS, AgentType.VERSION_CONTROL,
            AgentType.ERROR_RECOVERY, AgentType.CONTEXT_MANAGER,
            AgentType.ORCHESTRATOR, AgentType.CONSOLIDATED_RESEARCHER,
            AgentType.CONSOLIDATED_ORACLE, AgentType.ARCHITECT,
            AgentType.QUALITY, AgentType.OPERATIONS,
        ]
        getters = {at: _make_mock_agent(at) for at in all_agent_types}

        def make_getter(at):
            return lambda: getters[at]

        import_patch = {
            "get_planner_agent": make_getter(AgentType.PLANNER),
            "get_explorer_agent": make_getter(AgentType.EXPLORER),
            "get_oracle_agent": make_getter(AgentType.ORACLE),
            "get_librarian_agent": make_getter(AgentType.LIBRARIAN),
            "get_researcher_agent": make_getter(AgentType.RESEARCHER),
            "get_evaluator_agent": make_getter(AgentType.EVALUATOR),
            "get_synthesizer_agent": make_getter(AgentType.SYNTHESIZER),
            "get_builder_agent": make_getter(AgentType.BUILDER),
            "get_ui_planner_agent": make_getter(AgentType.UI_PLANNER),
            "get_security_auditor_agent": make_getter(AgentType.SECURITY_AUDITOR),
            "get_data_engineer_agent": make_getter(AgentType.DATA_ENGINEER),
            "get_documentation_agent": make_getter(AgentType.DOCUMENTATION_AGENT),
            "get_cost_planner_agent": make_getter(AgentType.COST_PLANNER),
            "get_test_automation_agent": make_getter(AgentType.TEST_AUTOMATION),
            "get_experimentation_manager_agent": make_getter(AgentType.EXPERIMENTATION_MANAGER),
            "get_improvement_agent": make_getter(AgentType.IMPROVEMENT),
            "get_user_interaction_agent": make_getter(AgentType.USER_INTERACTION),
            "get_devops_agent": make_getter(AgentType.DEVOPS),
            "get_version_control_agent": make_getter(AgentType.VERSION_CONTROL),
            "get_error_recovery_agent": make_getter(AgentType.ERROR_RECOVERY),
            "get_context_manager_agent": make_getter(AgentType.CONTEXT_MANAGER),
            "get_orchestrator_agent": make_getter(AgentType.ORCHESTRATOR),
            "get_consolidated_researcher_agent": make_getter(AgentType.CONSOLIDATED_RESEARCHER),
            "get_consolidated_oracle_agent": make_getter(AgentType.CONSOLIDATED_ORACLE),
            "get_architect_agent": make_getter(AgentType.ARCHITECT),
            "get_quality_agent": make_getter(AgentType.QUALITY),
            "get_operations_agent": make_getter(AgentType.OPERATIONS),
        }
        agents_mod = sys.modules["vetinari.agents"]
        for name, fn in import_patch.items():
            setattr(agents_mod, name, fn)

        g.initialize()
        self.assertTrue(g._initialized)

    def test_initialize_idempotent(self):
        g = AgentGraph()
        g._initialized = True
        # Should return early without calling any getter
        agents_mod = sys.modules["vetinari.agents"]
        sentinel = MagicMock()
        agents_mod.get_planner_agent = sentinel
        g.initialize()
        sentinel.assert_not_called()

    def test_initialize_handles_getter_exception(self):
        """Agents that raise during initialize() are skipped gracefully."""
        g = AgentGraph()
        agents_mod = sys.modules["vetinari.agents"]

        def bad_getter():
            raise RuntimeError("fail")

        def make_getter(at):
            a = _make_mock_agent(at)
            return lambda: a

        for name in [
            "get_planner_agent", "get_explorer_agent", "get_oracle_agent",
            "get_librarian_agent", "get_researcher_agent", "get_evaluator_agent",
            "get_synthesizer_agent", "get_builder_agent", "get_ui_planner_agent",
            "get_security_auditor_agent", "get_data_engineer_agent",
            "get_documentation_agent", "get_cost_planner_agent",
            "get_test_automation_agent", "get_experimentation_manager_agent",
            "get_improvement_agent", "get_user_interaction_agent",
            "get_devops_agent", "get_version_control_agent",
            "get_error_recovery_agent", "get_context_manager_agent",
            "get_orchestrator_agent", "get_consolidated_researcher_agent",
            "get_consolidated_oracle_agent", "get_architect_agent",
            "get_quality_agent", "get_operations_agent",
        ]:
            setattr(agents_mod, name, bad_getter)

        # Should not raise
        g.initialize()
        self.assertTrue(g._initialized)
        self.assertEqual(len(g._agents), 0)

    def test_initialize_skips_none_agent(self):
        """Getter returning None skips registration."""
        g = AgentGraph()
        agents_mod = sys.modules["vetinari.agents"]

        def make_getter(at):
            a = _make_mock_agent(at)
            return lambda: a

        agents_mod.get_planner_agent = lambda: None
        # Rest return valid mocks
        for name, at in [
            ("get_explorer_agent", AgentType.EXPLORER),
        ]:
            setattr(agents_mod, name, make_getter(at))
        for name in [
            "get_oracle_agent", "get_librarian_agent", "get_researcher_agent",
            "get_evaluator_agent", "get_synthesizer_agent", "get_builder_agent",
            "get_ui_planner_agent", "get_security_auditor_agent",
            "get_data_engineer_agent", "get_documentation_agent",
            "get_cost_planner_agent", "get_test_automation_agent",
            "get_experimentation_manager_agent", "get_improvement_agent",
            "get_user_interaction_agent", "get_devops_agent",
            "get_version_control_agent", "get_error_recovery_agent",
            "get_context_manager_agent", "get_orchestrator_agent",
            "get_consolidated_researcher_agent", "get_consolidated_oracle_agent",
            "get_architect_agent", "get_quality_agent", "get_operations_agent",
        ]:
            at_name = name.replace("get_", "").replace("_agent", "").upper()
            for at in AgentType:
                if at.value.upper() == at_name.upper():
                    setattr(agents_mod, name, make_getter(at))
                    break
            else:
                setattr(agents_mod, name, lambda: None)

        g.initialize()
        self.assertNotIn(AgentType.PLANNER, g._agents)


# ---------------------------------------------------------------------------
# 6. AgentGraph.create_execution_plan
# ---------------------------------------------------------------------------

class TestCreateExecutionPlan(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents()

    def tearDown(self):
        ag_module._agent_graph = None

    def test_returns_execution_plan(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        self.assertIsInstance(ep, ExecutionPlan)

    def test_plan_id_matches(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        self.assertEqual(ep.plan_id, plan.plan_id)

    def test_nodes_created_for_each_task(self):
        tasks = [_make_task("t1"), _make_task("t2"), _make_task("t3")]
        plan = _make_plan(tasks=tasks)
        ep = self.graph.create_execution_plan(plan)
        self.assertEqual(len(ep.nodes), 3)
        self.assertIn("t1", ep.nodes)
        self.assertIn("t2", ep.nodes)
        self.assertIn("t3", ep.nodes)

    def test_execution_order_non_empty(self):
        plan = _make_plan(tasks=[_make_task("t1"), _make_task("t2")])
        ep = self.graph.create_execution_plan(plan)
        self.assertEqual(len(ep.execution_order), 2)

    def test_plan_stored_in_graph(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        retrieved = self.graph.get_execution_plan(plan.plan_id)
        self.assertIs(retrieved, ep)

    def test_dependencies_propagated_to_nodes(self):
        tasks = [
            _make_task("t1"),
            _make_task("t2", deps=["t1"]),
        ]
        plan = _make_plan(tasks=tasks)
        ep = self.graph.create_execution_plan(plan)
        self.assertIn("t1", ep.nodes["t2"].dependencies)

    def test_dependents_populated(self):
        tasks = [
            _make_task("t1"),
            _make_task("t2", deps=["t1"]),
        ]
        plan = _make_plan(tasks=tasks)
        ep = self.graph.create_execution_plan(plan)
        self.assertIn("t2", ep.nodes["t1"].dependents)

    def test_empty_plan_allowed(self):
        plan = _make_plan(tasks=[])
        ep = self.graph.create_execution_plan(plan)
        self.assertEqual(len(ep.nodes), 0)
        self.assertEqual(ep.execution_order, [])

    def test_constraint_exception_does_not_crash(self):
        """create_execution_plan must succeed even when constraint registry raises."""
        import sys
        reg_mod = sys.modules["vetinari.constraints.registry"]
        reg_mod.get_constraint_registry.side_effect = RuntimeError("oops")
        try:
            plan = _make_plan(tasks=[_make_task("t1")])
            ep = self.graph.create_execution_plan(plan)
            self.assertIsNotNone(ep)
        finally:
            reg_mod.get_constraint_registry.side_effect = None

    def test_original_plan_referenced(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        self.assertIs(ep.original_plan, plan)


# ---------------------------------------------------------------------------
# 7. Topological sort
# ---------------------------------------------------------------------------

class TestTopologicalSort(unittest.TestCase):

    def setUp(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def _node(self, tid, deps=None, dependents=None):
        return TaskNode(
            task=_make_task(tid),
            dependencies=set(deps or []),
            dependents=set(dependents or []),
        )

    def test_single_node(self):
        nodes = {"t1": self._node("t1")}
        result = self.graph._topological_sort(nodes)
        self.assertEqual(result, ["t1"])

    def test_simple_chain_t1_t2_t3(self):
        nodes = {
            "t1": self._node("t1", dependents=["t2"]),
            "t2": self._node("t2", deps=["t1"], dependents=["t3"]),
            "t3": self._node("t3", deps=["t2"]),
        }
        result = self.graph._topological_sort(nodes)
        self.assertEqual(result, ["t1", "t2", "t3"])

    def test_parallel_tasks_both_before_dependent(self):
        nodes = {
            "t1": self._node("t1", dependents=["t3"]),
            "t2": self._node("t2", dependents=["t3"]),
            "t3": self._node("t3", deps=["t1", "t2"]),
        }
        result = self.graph._topological_sort(nodes)
        self.assertEqual(len(result), 3)
        self.assertLess(result.index("t1"), result.index("t3"))
        self.assertLess(result.index("t2"), result.index("t3"))

    def test_cycle_detection_raises_value_error(self):
        nodes = {
            "t1": self._node("t1", deps=["t2"], dependents=["t2"]),
            "t2": self._node("t2", deps=["t1"], dependents=["t1"]),
        }
        with self.assertRaises(ValueError) as ctx:
            self.graph._topological_sort(nodes)
        self.assertIn("Circular dependency", str(ctx.exception))

    def test_diamond_dependency(self):
        # t1 -> t2, t1 -> t3, t2 -> t4, t3 -> t4
        nodes = {
            "t1": self._node("t1", dependents=["t2", "t3"]),
            "t2": self._node("t2", deps=["t1"], dependents=["t4"]),
            "t3": self._node("t3", deps=["t1"], dependents=["t4"]),
            "t4": self._node("t4", deps=["t2", "t3"]),
        }
        result = self.graph._topological_sort(nodes)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "t1")
        self.assertEqual(result[-1], "t4")

    def test_all_independent(self):
        nodes = {
            "t1": self._node("t1"),
            "t2": self._node("t2"),
            "t3": self._node("t3"),
        }
        result = self.graph._topological_sort(nodes)
        self.assertEqual(sorted(result), ["t1", "t2", "t3"])

    def test_empty_nodes(self):
        result = self.graph._topological_sort({})
        self.assertEqual(result, [])

    def test_three_cycle(self):
        nodes = {
            "a": self._node("a", deps=["c"], dependents=["b"]),
            "b": self._node("b", deps=["a"], dependents=["c"]),
            "c": self._node("c", deps=["b"], dependents=["a"]),
        }
        with self.assertRaises(ValueError):
            self.graph._topological_sort(nodes)


# ---------------------------------------------------------------------------
# 8. _build_execution_layers
# ---------------------------------------------------------------------------

class TestBuildExecutionLayers(unittest.TestCase):

    def setUp(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def _make_exec_plan(self, tasks):
        plan = _make_plan(tasks=tasks)
        return self.graph.create_execution_plan(plan)

    def test_independent_tasks_single_layer(self):
        tasks = [_make_task("t1"), _make_task("t2"), _make_task("t3")]
        ep = self._make_exec_plan(tasks)
        layers = self.graph._build_execution_layers(ep)
        self.assertEqual(len(layers), 1)
        self.assertEqual(sorted(layers[0]), ["t1", "t2", "t3"])

    def test_chain_tasks_separate_layers(self):
        tasks = [
            _make_task("t1"),
            _make_task("t2", deps=["t1"]),
            _make_task("t3", deps=["t2"]),
        ]
        ep = self._make_exec_plan(tasks)
        layers = self.graph._build_execution_layers(ep)
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0], ["t1"])
        self.assertEqual(layers[1], ["t2"])
        self.assertEqual(layers[2], ["t3"])

    def test_diamond_layers(self):
        # t1 -> [t2, t3] -> t4
        tasks = [
            _make_task("t1"),
            _make_task("t2", deps=["t1"]),
            _make_task("t3", deps=["t1"]),
            _make_task("t4", deps=["t2", "t3"]),
        ]
        ep = self._make_exec_plan(tasks)
        layers = self.graph._build_execution_layers(ep)
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0], ["t1"])
        self.assertIn("t2", layers[1])
        self.assertIn("t3", layers[1])
        self.assertEqual(layers[2], ["t4"])

    def test_single_task_one_layer(self):
        ep = self._make_exec_plan([_make_task("t1")])
        layers = self.graph._build_execution_layers(ep)
        self.assertEqual(layers, [["t1"]])

    def test_empty_plan_empty_layers(self):
        ep = self._make_exec_plan([])
        layers = self.graph._build_execution_layers(ep)
        self.assertEqual(layers, [])


# ---------------------------------------------------------------------------
# 9. execute_plan — sequential
# ---------------------------------------------------------------------------

class TestExecutePlanSequential(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER],
        )

    def tearDown(self):
        ag_module._agent_graph = None

    def test_sequential_executes_all_tasks(self):
        tasks = [
            _make_task("t1", AgentType.EXPLORER),
            _make_task("t2", AgentType.BUILDER, deps=["t1"]),
        ]
        plan = _make_plan(tasks=tasks)
        results = self.graph.execute_plan(plan)
        self.assertIn("t1", results)
        self.assertIn("t2", results)

    def test_sequential_returns_success_results(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        results = self.graph.execute_plan(plan)
        self.assertTrue(results["t1"].success)

    def test_sequential_sets_plan_completed(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertEqual(ep.status, TaskStatus.COMPLETED)

    def test_sequential_sets_started_at(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertIsNotNone(ep.started_at)

    def test_sequential_sets_completed_at(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertIsNotNone(ep.completed_at)

    def test_sequential_node_status_completed_on_success(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertEqual(ep.nodes["t1"].status, TaskStatus.COMPLETED)

    def test_sequential_node_status_failed_on_failure(self):
        fail_agent = _make_mock_agent(AgentType.EXPLORER, success=False, verification_passed=False)
        self.graph._agents[AgentType.EXPLORER] = fail_agent
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        results = self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertEqual(ep.nodes["t1"].status, TaskStatus.FAILED)

    def test_sequential_exception_sets_plan_failed(self):
        # The implementation catches exceptions internally, logs them, and sets
        # the task/plan status to FAILED without re-raising. Verify that the
        # call completes without propagating and that status is FAILED.
        bad_agent = _make_mock_agent(AgentType.EXPLORER)
        bad_agent.execute.side_effect = Exception("boom")
        self.graph._agents[AgentType.EXPLORER] = bad_agent
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        results = self.graph.execute_plan(plan)  # must not raise
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertEqual(ep.nodes["t1"].status, TaskStatus.FAILED)


# ---------------------------------------------------------------------------
# 10. execute_plan — parallel / adaptive
# ---------------------------------------------------------------------------

class TestExecutePlanParallel(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.PARALLEL,
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER],
        )

    def tearDown(self):
        ag_module._agent_graph = None

    def test_parallel_all_results_present(self):
        tasks = [_make_task("t1", AgentType.EXPLORER), _make_task("t2", AgentType.BUILDER)]
        plan = _make_plan(tasks=tasks)
        results = self.graph.execute_plan(plan)
        self.assertIn("t1", results)
        self.assertIn("t2", results)

    def test_parallel_plan_completed(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertEqual(ep.status, TaskStatus.COMPLETED)

    def test_adaptive_strategy_uses_layers(self):
        g = _make_graph_with_agents(
            strategy=ExecutionStrategy.ADAPTIVE,
            agent_types=[AgentType.EXPLORER],
        )
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        results = g.execute_plan(plan)
        self.assertIn("t1", results)

    def test_parallel_single_task_no_thread_overhead(self):
        """Single task in a layer should execute without ThreadPoolExecutor."""
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        results = self.graph.execute_plan(plan)
        self.assertIn("t1", results)
        self.assertTrue(results["t1"].success)


# ---------------------------------------------------------------------------
# 11. _execute_layer_parallel
# ---------------------------------------------------------------------------

class TestExecuteLayerParallel(unittest.TestCase):

    def setUp(self):
        self.graph = _make_graph_with_agents(
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER],
        )

    def test_single_task_returns_result(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        ep = self.graph.create_execution_plan(plan)
        result = self.graph._execute_layer_parallel(["t1"], ep, {})
        self.assertIn("t1", result)

    def test_multi_task_returns_all_results(self):
        tasks = [_make_task("t1", AgentType.EXPLORER), _make_task("t2", AgentType.BUILDER)]
        plan = _make_plan(tasks=tasks)
        ep = self.graph.create_execution_plan(plan)
        result = self.graph._execute_layer_parallel(["t1", "t2"], ep, {})
        self.assertIn("t1", result)
        self.assertIn("t2", result)

    def test_exception_in_task_returns_failure_result(self):
        bad_agent = _make_mock_agent(AgentType.EXPLORER)
        bad_agent.execute.side_effect = Exception("thread fail")
        self.graph._agents[AgentType.EXPLORER] = bad_agent
        plan = _make_plan(tasks=[
            _make_task("t1", AgentType.EXPLORER),
            _make_task("t2", AgentType.BUILDER),
        ])
        ep = self.graph.create_execution_plan(plan)
        result = self.graph._execute_layer_parallel(["t1", "t2"], ep, {})
        self.assertIn("t1", result)
        self.assertFalse(result["t1"].success)

    def test_single_task_sets_completed_status(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        ep = self.graph.create_execution_plan(plan)
        self.graph._execute_layer_parallel(["t1"], ep, {})
        self.assertEqual(ep.nodes["t1"].status, TaskStatus.COMPLETED)


# ---------------------------------------------------------------------------
# 12. _execute_task_node
# ---------------------------------------------------------------------------

class TestExecuteTaskNode(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER, AgentType.QUALITY,
                         AgentType.ERROR_RECOVERY],
        )

    def tearDown(self):
        ag_module._agent_graph = None

    def test_success_returns_success_result(self):
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=0)
        result = self.graph._execute_task_node(node, {})
        self.assertTrue(result.success)

    def test_unregistered_agent_delegates_to_blackboard(self):
        node = TaskNode(task=_make_task("t1", AgentType.PLANNER), max_retries=0)
        # PLANNER not in agents
        self.assertNotIn(AgentType.PLANNER, self.graph._agents)
        result = self.graph._execute_task_node(node, {})
        # Blackboard mock returns None, so AgentResult.success should be False
        self.assertFalse(result.success)

    def test_permission_denied_returns_failure(self):
        ctx_mod = sys.modules["vetinari.execution_context"]
        ctx_mgr = MagicMock()
        ctx_mgr.enforce_permission.side_effect = PermissionError("no")
        ctx_mod.get_context_manager.return_value = ctx_mgr
        try:
            node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=0)
            result = self.graph._execute_task_node(node, {})
            self.assertFalse(result.success)
            self.assertTrue(any("Permission" in e for e in result.errors))
        finally:
            ctx_mod.get_context_manager.return_value = MagicMock(enforce_permission=MagicMock())

    def test_exception_in_agent_returns_failure(self):
        bad_agent = _make_mock_agent(AgentType.EXPLORER)
        bad_agent.execute.side_effect = Exception("bang")
        self.graph._agents[AgentType.EXPLORER] = bad_agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=0)
        result = self.graph._execute_task_node(node, {})
        self.assertFalse(result.success)
        self.assertIn("bang", result.errors[0])

    def test_verification_failure_triggers_self_correction(self):
        agent = _make_mock_agent(AgentType.EXPLORER)
        # First attempt fails verification, second succeeds
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": "bad output"}]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="result")
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=1)
        result = self.graph._execute_task_node(node, {})
        self.assertTrue(result.success)
        self.assertEqual(agent.execute.call_count, 2)

    def test_self_correction_injects_feedback_in_description(self):
        agent = _make_mock_agent(AgentType.EXPLORER)
        issue_text = "something wrong"
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": issue_text}]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="result")
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=1)
        self.graph._execute_task_node(node, {})
        # Second call description should contain SELF-CORRECTION
        second_call_arg = agent.execute.call_args_list[1][0][0]
        self.assertIn("SELF-CORRECTION", second_call_arg.description)

    def test_builder_triggers_maker_checker(self):
        builder = _make_mock_agent(AgentType.BUILDER)
        quality = _make_mock_agent(AgentType.QUALITY, success=True, verification_passed=True)
        self.graph._agents[AgentType.BUILDER] = builder
        self.graph._agents[AgentType.QUALITY] = quality
        node = TaskNode(task=_make_task("t1", AgentType.BUILDER), max_retries=0)
        result = self.graph._execute_task_node(node, {})
        # Quality agent should be called during maker-checker
        quality.execute.assert_called()

    def test_non_builder_skips_maker_checker(self):
        explorer = _make_mock_agent(AgentType.EXPLORER)
        quality = _make_mock_agent(AgentType.QUALITY)
        self.graph._agents[AgentType.EXPLORER] = explorer
        self.graph._agents[AgentType.QUALITY] = quality
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=0)
        self.graph._execute_task_node(node, {})
        quality.execute.assert_not_called()

    def test_error_recovery_called_on_last_attempt_failure(self):
        fail_agent = _make_mock_agent(AgentType.EXPLORER, success=True, verification_passed=False)
        recovery = _make_mock_agent(AgentType.ERROR_RECOVERY, success=True)
        self.graph._agents[AgentType.EXPLORER] = fail_agent
        self.graph._agents[AgentType.ERROR_RECOVERY] = recovery
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=0)
        result = self.graph._execute_task_node(node, {})
        recovery.execute.assert_called()

    def test_prior_results_injected_into_context(self):
        agent = _make_mock_agent(AgentType.BUILDER)
        self.graph._agents[AgentType.BUILDER] = agent
        prior = {"t0": AgentResult(success=True, output="prior output")}
        task = _make_task("t1", AgentType.BUILDER, deps=["t0"])
        node = TaskNode(task=task, max_retries=0)
        self.graph._execute_task_node(node, prior)
        agent_task_called = agent.execute.call_args[0][0]
        self.assertIn("dependency_results", agent_task_called.context)

    def test_incorporate_prior_results_called_if_present(self):
        agent = _make_mock_agent(AgentType.EXPLORER)
        agent._incorporate_prior_results = MagicMock()
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=0)
        self.graph._execute_task_node(node, {})
        agent._incorporate_prior_results.assert_called_once()

    def test_constraint_registry_caps_retries(self):
        reg = MagicMock()
        ac = MagicMock()
        ac.resources.max_retries = 0
        reg.get_constraints_for_agent.return_value = ac
        reg_mod = sys.modules["vetinari.constraints.registry"]
        reg_mod.get_constraint_registry.return_value = reg
        reg_mod.get_constraint_registry.side_effect = None
        try:
            node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=5)
            self.graph._execute_task_node(node, {})
            self.assertEqual(node.max_retries, 0)
        finally:
            reg_mod.get_constraint_registry.return_value = None
            reg_mod.get_constraint_registry.side_effect = None

    def test_all_retries_exhausted_returns_failure(self):
        agent = _make_mock_agent(AgentType.EXPLORER, success=True, verification_passed=False)
        # Remove error recovery so it falls through
        self.graph._agents.pop(AgentType.ERROR_RECOVERY, None)
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=2)
        result = self.graph._execute_task_node(node, {})
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# 13. _apply_maker_checker
# ---------------------------------------------------------------------------

class TestApplyMakerChecker(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None
        self.graph = AgentGraph()
        self.graph._initialized = True

    def tearDown(self):
        ag_module._agent_graph = None

    def test_returns_result_if_no_quality_agent(self):
        result = AgentResult(success=True, output="x")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        self.assertIs(returned, result)

    def test_returns_result_if_no_builder_agent(self):
        self.graph._agents[AgentType.QUALITY] = _make_mock_agent(AgentType.QUALITY)
        result = AgentResult(success=True, output="x")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        self.assertIs(returned, result)

    def test_quality_approves_first_iteration(self):
        self.graph._agents[AgentType.QUALITY] = _make_mock_agent(AgentType.QUALITY)
        self.graph._agents[AgentType.BUILDER] = _make_mock_agent(AgentType.BUILDER)
        result = AgentResult(success=True, output="good output")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        self.assertTrue(returned.metadata.get("maker_checker", {}).get("approved"))

    def test_metadata_has_iterations_on_approval(self):
        self.graph._agents[AgentType.QUALITY] = _make_mock_agent(AgentType.QUALITY)
        self.graph._agents[AgentType.BUILDER] = _make_mock_agent(AgentType.BUILDER)
        result = AgentResult(success=True, output="good")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        self.assertIn("iterations", returned.metadata["maker_checker"])

    def test_quality_rejects_then_approves(self):
        quality = _make_mock_agent(AgentType.QUALITY)
        quality.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": "fix this"}]),
            VerificationResult(passed=True),
        ]
        quality.execute.return_value = AgentResult(success=True, output="review")
        builder = _make_mock_agent(AgentType.BUILDER)
        self.graph._agents[AgentType.QUALITY] = quality
        self.graph._agents[AgentType.BUILDER] = builder
        result = AgentResult(success=True, output="initial")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        self.assertTrue(returned.metadata.get("maker_checker", {}).get("approved"))

    def test_max_iterations_exhausted_sets_approved_false(self):
        quality = _make_mock_agent(AgentType.QUALITY)
        quality.verify.return_value = VerificationResult(passed=False, issues=[{"message": "bad"}])
        quality.execute.return_value = AgentResult(success=True, output="review")
        builder = _make_mock_agent(AgentType.BUILDER)
        builder.execute.return_value = AgentResult(success=True, output="rebuilt")
        self.graph._agents[AgentType.QUALITY] = quality
        self.graph._agents[AgentType.BUILDER] = builder
        result = AgentResult(success=True, output="initial")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        self.assertFalse(returned.metadata.get("maker_checker", {}).get("approved"))

    def test_maker_checker_exception_breaks_loop(self):
        quality = _make_mock_agent(AgentType.QUALITY)
        quality.execute.side_effect = Exception("crash")
        builder = _make_mock_agent(AgentType.BUILDER)
        self.graph._agents[AgentType.QUALITY] = quality
        self.graph._agents[AgentType.BUILDER] = builder
        result = AgentResult(success=True, output="initial")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        # Should not raise, and metadata should be set
        self.assertIsNotNone(returned.metadata)

    def test_builder_failure_in_fix_breaks_loop(self):
        quality = _make_mock_agent(AgentType.QUALITY)
        quality.verify.return_value = VerificationResult(passed=False, issues=[{"message": "bad"}])
        quality.execute.return_value = AgentResult(success=True, output="review")
        builder = _make_mock_agent(AgentType.BUILDER)
        builder.execute.return_value = AgentResult(success=False, output=None)
        self.graph._agents[AgentType.QUALITY] = quality
        self.graph._agents[AgentType.BUILDER] = builder
        result = AgentResult(success=True, output="initial")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        self.assertFalse(returned.metadata.get("maker_checker", {}).get("approved"))


# ---------------------------------------------------------------------------
# 14. _run_error_recovery
# ---------------------------------------------------------------------------

class TestRunErrorRecovery(unittest.TestCase):

    def setUp(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def test_error_recovery_called_with_recovery_task(self):
        recovery = _make_mock_agent(AgentType.ERROR_RECOVERY, success=True)
        self.graph._agents[AgentType.ERROR_RECOVERY] = recovery
        task = _make_task("t1", AgentType.EXPLORER)
        failed = AgentResult(success=False, output="bad", errors=["fail"])
        verification = VerificationResult(passed=False, issues=[{"message": "issue1"}])
        result = self.graph._run_error_recovery(task, failed, verification)
        recovery.execute.assert_called_once()
        self.assertTrue(result.success)

    def test_error_recovery_exception_returns_failure(self):
        recovery = _make_mock_agent(AgentType.ERROR_RECOVERY)
        recovery.execute.side_effect = Exception("recovery failed")
        self.graph._agents[AgentType.ERROR_RECOVERY] = recovery
        task = _make_task("t1")
        failed = AgentResult(success=False, output="x", errors=["e"])
        verification = VerificationResult(passed=False, issues=[])
        result = self.graph._run_error_recovery(task, failed, verification)
        self.assertFalse(result.success)
        self.assertTrue(any("Recovery failed" in e for e in result.errors))

    def test_error_recovery_passes_original_output_in_context(self):
        recovery = _make_mock_agent(AgentType.ERROR_RECOVERY, success=True)
        self.graph._agents[AgentType.ERROR_RECOVERY] = recovery
        task = _make_task("t1")
        failed = AgentResult(success=False, output="original_out", errors=[])
        verification = VerificationResult(passed=False, issues=[{"message": "v issue"}])
        self.graph._run_error_recovery(task, failed, verification)
        called_task = recovery.execute.call_args[0][0]
        self.assertEqual(called_task.context.get("original_output"), "original_out")


# ---------------------------------------------------------------------------
# 15. _validate_output_schema
# ---------------------------------------------------------------------------

class TestValidateOutputSchema(unittest.TestCase):

    def setUp(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def test_no_spec_returns_empty(self):
        issues = self.graph._validate_output_schema(AgentType.EXPLORER, {"key": "val"})
        self.assertEqual(issues, [])

    def test_none_output_returns_empty_when_no_spec(self):
        issues = self.graph._validate_output_schema(AgentType.EXPLORER, None)
        self.assertEqual(issues, [])

    def test_missing_required_field(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": ["result"],
            "properties": {"result": {"type": "string"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, {})
        self.assertTrue(any("result" in i for i in issues))

    def test_wrong_type_detected(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"count": {"type": "integer"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, {"count": "not_int"})
        self.assertTrue(any("count" in i for i in issues))

    def test_valid_output_returns_empty(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": ["result"],
            "properties": {"result": {"type": "string"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, {"result": "ok"})
        self.assertEqual(issues, [])

    def test_non_dict_output_skips_validation(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": ["result"],
            "properties": {"result": {"type": "string"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, "a string")
        self.assertEqual(issues, [])

    def test_null_output_skips_validation(self):
        spec = MagicMock()
        spec.output_schema = {"required": ["x"], "properties": {}}
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, None)
        self.assertEqual(issues, [])

    def test_spec_with_no_output_schema_returns_empty(self):
        spec = MagicMock()
        spec.output_schema = None
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, {"x": 1})
        self.assertEqual(issues, [])

    def test_number_type_accepts_int_and_float(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"val": {"type": "number"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues_int = self.graph._validate_output_schema(AgentType.EXPLORER, {"val": 42})
            issues_float = self.graph._validate_output_schema(AgentType.EXPLORER, {"val": 3.14})
        self.assertEqual(issues_int, [])
        self.assertEqual(issues_float, [])

    def test_array_type_validated(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"items": {"type": "array"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, {"items": "not_a_list"})
        self.assertTrue(any("items" in i for i in issues))

    def test_boolean_type_validated(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"flag": {"type": "boolean"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, {"flag": "yes"})
        self.assertTrue(any("flag" in i for i in issues))

    def test_object_type_validated(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"data": {"type": "object"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.EXPLORER, {"data": [1, 2]})
        self.assertTrue(any("data" in i for i in issues))


# ---------------------------------------------------------------------------
# 16. inject_task
# ---------------------------------------------------------------------------

class TestInjectTask(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(agent_types=[AgentType.EXPLORER])

    def tearDown(self):
        ag_module._agent_graph = None

    def _setup_plan_with_tasks(self, task_ids):
        tasks = [_make_task(tid, AgentType.EXPLORER) for tid in task_ids]
        plan = _make_plan(tasks=tasks)
        self.graph.create_execution_plan(plan)
        return plan

    def test_inject_task_nonexistent_plan_returns_false(self):
        new_task = _make_task("new", AgentType.EXPLORER)
        result = self.graph.inject_task("nonexistent_plan", new_task, "t1")
        self.assertFalse(result)

    def test_inject_task_nonexistent_after_task_returns_false(self):
        plan = self._setup_plan_with_tasks(["t1"])
        new_task = _make_task("new", AgentType.EXPLORER)
        result = self.graph.inject_task(plan.plan_id, new_task, "t_missing")
        self.assertFalse(result)

    def test_inject_task_duplicate_task_id_returns_false(self):
        plan = self._setup_plan_with_tasks(["t1", "t2"])
        # t2 already exists
        dup_task = _make_task("t2", AgentType.EXPLORER)
        result = self.graph.inject_task(plan.plan_id, dup_task, "t1")
        self.assertFalse(result)

    def test_inject_task_success_returns_true(self):
        plan = self._setup_plan_with_tasks(["t1"])
        new_task = _make_task("t_new", AgentType.EXPLORER)
        result = self.graph.inject_task(plan.plan_id, new_task, "t1")
        self.assertTrue(result)

    def test_inject_task_adds_node_to_plan(self):
        plan = self._setup_plan_with_tasks(["t1"])
        new_task = _make_task("t_new", AgentType.EXPLORER)
        self.graph.inject_task(plan.plan_id, new_task, "t1")
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertIn("t_new", ep.nodes)

    def test_inject_task_new_node_depends_on_after_task(self):
        plan = self._setup_plan_with_tasks(["t1"])
        new_task = _make_task("t_new", AgentType.EXPLORER)
        self.graph.inject_task(plan.plan_id, new_task, "t1")
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertIn("t1", ep.nodes["t_new"].dependencies)

    def test_inject_task_rewires_execution_order(self):
        plan = self._setup_plan_with_tasks(["t1", "t2"])
        # make t2 depend on t1
        ep = self.graph.get_execution_plan(plan.plan_id)
        ep.nodes["t2"].dependencies = {"t1"}
        ep.nodes["t1"].dependents = {"t2"}
        new_task = _make_task("t_mid", AgentType.EXPLORER)
        self.graph.inject_task(plan.plan_id, new_task, "t1")
        ep = self.graph.get_execution_plan(plan.plan_id)
        # t_mid should appear before t2 in execution order
        order = ep.execution_order
        self.assertLess(order.index("t_mid"), order.index("t2"))

    def test_inject_task_into_chain(self):
        plan = self._setup_plan_with_tasks(["t1"])
        nt = _make_task("t2", AgentType.EXPLORER)
        r1 = self.graph.inject_task(plan.plan_id, nt, "t1")
        nt2 = _make_task("t3", AgentType.EXPLORER)
        r2 = self.graph.inject_task(plan.plan_id, nt2, "t2")
        self.assertTrue(r1)
        self.assertTrue(r2)
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertEqual(len(ep.nodes), 3)


# ---------------------------------------------------------------------------
# 17. get_agent, get_registered_agents
# ---------------------------------------------------------------------------

class TestGetAgent(unittest.TestCase):

    def setUp(self):
        self.graph = _make_graph_with_agents(
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER]
        )

    def test_get_agent_returns_agent(self):
        agent = self.graph.get_agent(AgentType.EXPLORER)
        self.assertIsNotNone(agent)

    def test_get_agent_returns_none_for_unregistered(self):
        agent = self.graph.get_agent(AgentType.PLANNER)
        self.assertIsNone(agent)

    def test_get_registered_agents_returns_list(self):
        agents = self.graph.get_registered_agents()
        self.assertIsInstance(agents, list)

    def test_get_registered_agents_contains_registered_types(self):
        agents = self.graph.get_registered_agents()
        self.assertIn(AgentType.EXPLORER, agents)
        self.assertIn(AgentType.BUILDER, agents)

    def test_get_registered_agents_count(self):
        agents = self.graph.get_registered_agents()
        self.assertEqual(len(agents), 2)

    def test_get_agent_correct_instance(self):
        mock_agent = _make_mock_agent(AgentType.EXPLORER)
        self.graph._agents[AgentType.EXPLORER] = mock_agent
        returned = self.graph.get_agent(AgentType.EXPLORER)
        self.assertIs(returned, mock_agent)


# ---------------------------------------------------------------------------
# 18. get_agent_by_capability
# ---------------------------------------------------------------------------

class TestGetAgentByCapability(unittest.TestCase):

    def setUp(self):
        self.graph = _make_graph_with_agents(
            agent_types=[AgentType.EXPLORER, AgentType.ORCHESTRATOR]
        )

    def test_returns_none_when_no_matching_specs(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.return_value = []
        result = self.graph.get_agent_by_capability("code_search")
        self.assertIsNone(result)

    def test_returns_agent_when_spec_matches(self):
        spec = MagicMock()
        spec.agent_type = "EXPLORER"
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.return_value = [spec]
        result = self.graph.get_agent_by_capability("exploration")
        self.assertIsNotNone(result)

    def test_returns_none_when_agent_not_registered_for_matching_spec(self):
        spec = MagicMock()
        spec.agent_type = "PLANNER"  # PLANNER not in graph
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.return_value = [spec]
        result = self.graph.get_agent_by_capability("planning")
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.side_effect = Exception("skill error")
        result = self.graph.get_agent_by_capability("anything")
        self.assertIsNone(result)
        skill_reg.get_skills_by_capability.side_effect = None

    def test_prefers_consolidated_agents(self):
        spec_orchestrator = MagicMock()
        spec_orchestrator.agent_type = "ORCHESTRATOR"
        spec_explorer = MagicMock()
        spec_explorer.agent_type = "EXPLORER"
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        # Return explorer first, but orchestrator should win due to priority
        skill_reg.get_skills_by_capability.return_value = [spec_explorer, spec_orchestrator]
        result = self.graph.get_agent_by_capability("something")
        # ORCHESTRATOR has priority 10 vs EXPLORER not in priority map (0)
        expected = self.graph._agents.get(AgentType.ORCHESTRATOR)
        self.assertIs(result, expected)

    def test_invalid_agent_type_in_spec_skipped(self):
        spec = MagicMock()
        spec.agent_type = "INVALID_TYPE_XYZ"
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.return_value = [spec]
        result = self.graph.get_agent_by_capability("something")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 19. get_skill_spec
# ---------------------------------------------------------------------------

class TestGetSkillSpec(unittest.TestCase):

    def setUp(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def test_returns_spec_from_registry(self):
        mock_spec = MagicMock()
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skill_for_agent_type.return_value = mock_spec
        result = self.graph.get_skill_spec(AgentType.EXPLORER)
        self.assertIs(result, mock_spec)

    def test_returns_none_on_exception(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skill_for_agent_type.side_effect = Exception("not found")
        result = self.graph.get_skill_spec(AgentType.EXPLORER)
        self.assertIsNone(result)
        skill_reg.get_skill_for_agent_type.side_effect = None

    def test_calls_with_agent_type_value(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skill_for_agent_type.return_value = None
        self.graph.get_skill_spec(AgentType.BUILDER)
        skill_reg.get_skill_for_agent_type.assert_called_with(AgentType.BUILDER.value)


# ---------------------------------------------------------------------------
# 20. get_agents_for_task_type
# ---------------------------------------------------------------------------

class TestGetAgentsForTaskType(unittest.TestCase):

    def setUp(self):
        self.graph = _make_graph_with_agents(agent_types=[AgentType.EXPLORER])

    def test_returns_empty_list_when_no_matching_skills(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {}
        result = self.graph.get_agents_for_task_type("some_task")
        self.assertEqual(result, [])

    def test_returns_agents_matching_task_type(self):
        spec = MagicMock()
        spec.agent_type = "EXPLORER"
        spec.modes = ["some_task"]
        spec.capabilities = []
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {"explorer": spec}
        result = self.graph.get_agents_for_task_type("some_task")
        self.assertIn(AgentType.EXPLORER, result)

    def test_returns_agents_matching_capability(self):
        spec = MagicMock()
        spec.agent_type = "EXPLORER"
        spec.modes = []
        spec.capabilities = ["some_task"]
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {"explorer": spec}
        result = self.graph.get_agents_for_task_type("some_task")
        self.assertIn(AgentType.EXPLORER, result)

    def test_skips_agents_not_registered(self):
        spec = MagicMock()
        spec.agent_type = "PLANNER"  # not registered in graph
        spec.modes = ["some_task"]
        spec.capabilities = []
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {"planner": spec}
        result = self.graph.get_agents_for_task_type("some_task")
        self.assertEqual(result, [])

    def test_returns_empty_on_exception(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.side_effect = Exception("fail")
        result = self.graph.get_agents_for_task_type("x")
        self.assertEqual(result, [])
        skill_reg.get_all_skills.side_effect = None

    def test_skips_unrecognized_agent_type_in_spec(self):
        spec = MagicMock()
        spec.agent_type = "TOTALLY_UNKNOWN"
        spec.modes = ["some_task"]
        spec.capabilities = []
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {"unknown": spec}
        result = self.graph.get_agents_for_task_type("some_task")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# 21. get_agent_graph singleton
# ---------------------------------------------------------------------------

class TestGetAgentGraphSingleton(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None
        agents_mod = sys.modules["vetinari.agents"]
        all_names = [
            "get_planner_agent", "get_explorer_agent", "get_oracle_agent",
            "get_librarian_agent", "get_researcher_agent", "get_evaluator_agent",
            "get_synthesizer_agent", "get_builder_agent", "get_ui_planner_agent",
            "get_security_auditor_agent", "get_data_engineer_agent",
            "get_documentation_agent", "get_cost_planner_agent",
            "get_test_automation_agent", "get_experimentation_manager_agent",
            "get_improvement_agent", "get_user_interaction_agent",
            "get_devops_agent", "get_version_control_agent",
            "get_error_recovery_agent", "get_context_manager_agent",
            "get_orchestrator_agent", "get_consolidated_researcher_agent",
            "get_consolidated_oracle_agent", "get_architect_agent",
            "get_quality_agent", "get_operations_agent",
        ]
        for name in all_names:
            setattr(agents_mod, name, lambda: None)

    def tearDown(self):
        ag_module._agent_graph = None

    def test_returns_agent_graph_instance(self):
        g = get_agent_graph()
        self.assertIsInstance(g, AgentGraph)

    def test_same_instance_returned_twice(self):
        g1 = get_agent_graph()
        g2 = get_agent_graph()
        self.assertIs(g1, g2)

    def test_singleton_is_initialized(self):
        g = get_agent_graph()
        self.assertTrue(g._initialized)

    def test_singleton_uses_provided_strategy(self):
        g = get_agent_graph(strategy=ExecutionStrategy.SEQUENTIAL)
        self.assertEqual(g._strategy, ExecutionStrategy.SEQUENTIAL)

    def test_singleton_subsequent_call_ignores_strategy(self):
        """Once created, the singleton does not change strategy on re-calls."""
        g1 = get_agent_graph(strategy=ExecutionStrategy.SEQUENTIAL)
        g2 = get_agent_graph(strategy=ExecutionStrategy.PARALLEL)
        self.assertIs(g1, g2)
        self.assertEqual(g2._strategy, ExecutionStrategy.SEQUENTIAL)

    def test_singleton_reset_allows_new_instance(self):
        g1 = get_agent_graph()
        ag_module._agent_graph = None
        g2 = get_agent_graph()
        self.assertIsNot(g1, g2)


# ---------------------------------------------------------------------------
# 22. __repr__
# ---------------------------------------------------------------------------

class TestAgentGraphRepr(unittest.TestCase):

    def test_repr_contains_strategy(self):
        g = AgentGraph(strategy=ExecutionStrategy.SEQUENTIAL)
        self.assertIn("sequential", repr(g))

    def test_repr_contains_agentgraph(self):
        g = AgentGraph()
        self.assertIn("AgentGraph", repr(g))

    def test_repr_contains_agent_count(self):
        g = AgentGraph()
        g._agents[AgentType.EXPLORER] = MagicMock()
        r = repr(g)
        self.assertIn("1", r)

    def test_repr_adaptive(self):
        g = AgentGraph(strategy=ExecutionStrategy.ADAPTIVE)
        self.assertIn("adaptive", repr(g))

    def test_repr_parallel(self):
        g = AgentGraph(strategy=ExecutionStrategy.PARALLEL)
        self.assertIn("parallel", repr(g))


# ---------------------------------------------------------------------------
# 23. execute_plan_async
# ---------------------------------------------------------------------------

class TestExecutePlanAsync(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER]
        )

    def tearDown(self):
        ag_module._agent_graph = None

    def test_async_returns_results(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        results = asyncio.run(
            self.graph.execute_plan_async(plan)
        )
        self.assertIn("t1", results)

    def test_async_multiple_tasks(self):
        tasks = [
            _make_task("t1", AgentType.EXPLORER),
            _make_task("t2", AgentType.BUILDER),
        ]
        plan = _make_plan(tasks=tasks)
        results = asyncio.run(
            self.graph.execute_plan_async(plan)
        )
        self.assertIn("t1", results)
        self.assertIn("t2", results)

    def test_async_plan_completed(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        asyncio.run(
            self.graph.execute_plan_async(plan)
        )
        ep = self.graph.get_execution_plan(plan.plan_id)
        self.assertEqual(ep.status, TaskStatus.COMPLETED)

    def test_async_exception_wraps_as_failure_result(self):
        bad_agent = _make_mock_agent(AgentType.EXPLORER)
        bad_agent.execute.side_effect = Exception("async boom")
        self.graph._agents[AgentType.EXPLORER] = bad_agent
        plan = _make_plan(tasks=[
            _make_task("t1", AgentType.EXPLORER),
            _make_task("t2", AgentType.BUILDER),
        ])
        results = asyncio.run(
            self.graph.execute_plan_async(plan)
        )
        self.assertFalse(results["t1"].success)
        self.assertTrue(results["t2"].success)


# ---------------------------------------------------------------------------
# 24. get_execution_plan
# ---------------------------------------------------------------------------

class TestGetExecutionPlan(unittest.TestCase):

    def setUp(self):
        self.graph = _make_graph_with_agents()

    def test_returns_none_for_unknown_plan_id(self):
        result = self.graph.get_execution_plan("unknown_id")
        self.assertIsNone(result)

    def test_returns_plan_after_create(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        retrieved = self.graph.get_execution_plan(plan.plan_id)
        self.assertIs(retrieved, ep)


# ---------------------------------------------------------------------------
# 25. Edge cases and integration-style tests
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        ag_module._agent_graph = None

    def tearDown(self):
        ag_module._agent_graph = None

    def test_execute_plan_empty_tasks(self):
        graph = _make_graph_with_agents()
        plan = _make_plan(tasks=[])
        results = graph.execute_plan(plan)
        self.assertEqual(results, {})

    def test_execute_plan_single_task_sequential(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.EXPLORER],
        )
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        results = graph.execute_plan(plan)
        self.assertIn("t1", results)
        self.assertTrue(results["t1"].success)

    def test_execute_plan_chain_sequential_order(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER],
        )
        call_order = []
        explorer = graph._agents[AgentType.EXPLORER]
        builder = graph._agents[AgentType.BUILDER]
        explorer.execute.side_effect = lambda t: (call_order.append("explorer"), AgentResult(success=True, output="e"))[1]
        builder.execute.side_effect = lambda t: (call_order.append("builder"), AgentResult(success=True, output="b"))[1]

        tasks = [
            _make_task("t1", AgentType.EXPLORER),
            _make_task("t2", AgentType.BUILDER, deps=["t1"]),
        ]
        plan = _make_plan(tasks=tasks)
        graph.execute_plan(plan)
        self.assertEqual(call_order, ["explorer", "builder"])

    def test_multiple_plans_independent(self):
        graph = _make_graph_with_agents(agent_types=[AgentType.EXPLORER])
        plan1 = _make_plan("goal1", tasks=[_make_task("t1", AgentType.EXPLORER)])
        plan2 = _make_plan("goal2", tasks=[_make_task("t2", AgentType.EXPLORER)])
        ep1 = graph.create_execution_plan(plan1)
        ep2 = graph.create_execution_plan(plan2)
        self.assertIsNot(ep1, ep2)
        self.assertNotEqual(ep1.plan_id, ep2.plan_id)

    def test_task_node_status_updated_after_parallel_execution(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.PARALLEL,
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER],
        )
        tasks = [
            _make_task("t1", AgentType.EXPLORER),
            _make_task("t2", AgentType.BUILDER),
        ]
        plan = _make_plan(tasks=tasks)
        graph.execute_plan(plan)
        ep = graph.get_execution_plan(plan.plan_id)
        self.assertEqual(ep.nodes["t1"].status, TaskStatus.COMPLETED)
        self.assertEqual(ep.nodes["t2"].status, TaskStatus.COMPLETED)

    def test_inject_then_execute(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.EXPLORER],
        )
        plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
        graph.create_execution_plan(plan)
        new_task = _make_task("t2", AgentType.EXPLORER)
        injected = graph.inject_task(plan.plan_id, new_task, "t1")
        self.assertTrue(injected)
        ep = graph.get_execution_plan(plan.plan_id)
        self.assertIn("t2", ep.nodes)

    def test_blackboard_delegate_called_for_unknown_agent(self):
        graph = _make_graph_with_agents()
        blackboard_mod = sys.modules["vetinari.blackboard"]
        mock_board = MagicMock()
        mock_board.delegate.return_value = AgentResult(success=True, output="delegated")
        blackboard_mod.get_blackboard.return_value = mock_board
        node = TaskNode(task=_make_task("t1", AgentType.PLANNER), max_retries=0)
        result = graph._execute_task_node(node, {})
        mock_board.delegate.assert_called_once()

    def test_self_correction_increments_retries(self):
        graph = _make_graph_with_agents(agent_types=[AgentType.EXPLORER])
        agent = _make_mock_agent(AgentType.EXPLORER)
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": "fix"}]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="x")
        graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=1)
        graph._execute_task_node(node, {})
        self.assertEqual(node.retries, 1)

    def test_agent_graph_max_workers_limits_parallel_pool(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.PARALLEL,
            agent_types=[AgentType.EXPLORER, AgentType.BUILDER],
        )
        tasks = [
            _make_task("t1", AgentType.EXPLORER),
            _make_task("t2", AgentType.BUILDER),
        ]
        plan = _make_plan(tasks=tasks)
        results = graph.execute_plan(plan)
        self.assertIn("t1", results)
        self.assertIn("t2", results)

    def test_create_execution_plan_with_multiple_deps(self):
        graph = _make_graph_with_agents()
        tasks = [
            _make_task("t1"),
            _make_task("t2"),
            _make_task("t3", deps=["t1", "t2"]),
        ]
        plan = _make_plan(tasks=tasks)
        ep = graph.create_execution_plan(plan)
        self.assertIn("t1", ep.nodes["t3"].dependencies)
        self.assertIn("t2", ep.nodes["t3"].dependencies)

    def test_topological_sort_preserves_all_ids(self):
        graph = AgentGraph()
        from vetinari.orchestration.agent_graph import TaskNode as TN
        ids = [f"t{i}" for i in range(10)]
        nodes = {tid: TN(task=_make_task(tid)) for tid in ids}
        result = graph._topological_sort(nodes)
        self.assertEqual(sorted(result), sorted(ids))

    def test_execute_plan_result_keys_match_task_ids(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.EXPLORER],
        )
        tasks = [_make_task(f"t{i}", AgentType.EXPLORER) for i in range(5)]
        plan = _make_plan(tasks=tasks)
        results = graph.execute_plan(plan)
        for task in tasks:
            self.assertIn(task.id, results)

    def test_schema_issues_logged_but_do_not_block_execution(self):
        """Schema issues should be non-blocking (logged only)."""
        spec = MagicMock()
        spec.output_schema = {
            "required": ["missing_key"],
            "properties": {},
        }
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.EXPLORER],
        )
        with patch.object(graph, "get_skill_spec", return_value=spec):
            plan = _make_plan(tasks=[_make_task("t1", AgentType.EXPLORER)])
            results = graph.execute_plan(plan)
        # Should complete successfully (schema issues are warnings, not errors)
        self.assertTrue(results["t1"].success)


# ---------------------------------------------------------------------------
# 26. Additional coverage for issue text formatting in self-correction
# ---------------------------------------------------------------------------

class TestIssueTextFormatting(unittest.TestCase):

    def setUp(self):
        self.graph = _make_graph_with_agents(agent_types=[AgentType.EXPLORER])

    def test_string_issues_in_verification(self):
        """Verification issues that are plain strings (not dicts) should work."""
        agent = _make_mock_agent(AgentType.EXPLORER)
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=["plain string issue"]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="x")
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=1)
        result = self.graph._execute_task_node(node, {})
        self.assertTrue(result.success)

    def test_dict_issues_in_verification(self):
        agent = _make_mock_agent(AgentType.EXPLORER)
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": "dict issue"}]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="x")
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=1)
        result = self.graph._execute_task_node(node, {})
        self.assertTrue(result.success)

    def test_empty_issues_list_in_verification(self):
        agent = _make_mock_agent(AgentType.EXPLORER, verification_passed=False)
        agent.verify.return_value = VerificationResult(passed=False, issues=[])
        agent.execute.return_value = AgentResult(success=True, output="x")
        self.graph._agents.pop(AgentType.ERROR_RECOVERY, None)
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=0)
        result = self.graph._execute_task_node(node, {})
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# 27. AgentTask.from_task usage in _execute_task_node
# ---------------------------------------------------------------------------

class TestAgentTaskCreation(unittest.TestCase):

    def setUp(self):
        self.graph = _make_graph_with_agents(agent_types=[AgentType.EXPLORER])

    def test_agent_task_created_from_task(self):
        agent = _make_mock_agent(AgentType.EXPLORER)
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER, description="do this"), max_retries=0)
        self.graph._execute_task_node(node, {})
        called_arg = agent.execute.call_args[0][0]
        self.assertEqual(called_arg.task_id, "t1")
        self.assertEqual(called_arg.description, "do this")

    def test_agent_task_has_empty_context_by_default(self):
        agent = _make_mock_agent(AgentType.EXPLORER)
        self.graph._agents[AgentType.EXPLORER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.EXPLORER), max_retries=0)
        self.graph._execute_task_node(node, {})
        called_arg = agent.execute.call_args[0][0]
        self.assertIsInstance(called_arg.context, dict)


if __name__ == "__main__":
    unittest.main()
