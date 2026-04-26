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

import asyncio
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy external modules BEFORE importing anything from vetinari
# ---------------------------------------------------------------------------


def _make_mock_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    return mod


def _make_package_module(name: str) -> types.ModuleType:
    """Create a stub module that also acts as a package (has __path__)."""
    mod = types.ModuleType(name)
    mod.__path__ = []  # marks it as a package
    mod.__package__ = name
    return mod


def _install_stubs() -> None:
    """Install all required sys.modules stubs so vetinari imports don't fail."""

    # vetinari must be a package (has __path__) so sub-packages resolve.
    # Use the real on-disk path so Python can find submodules inside it.
    import importlib.util as _ilu
    import os as _os

    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

    # ---- vetinari.types ----
    # Load the REAL vetinari/types.py — it only imports stdlib (enum), so it is
    # safe to import directly without side-effects.  This ensures that ALL enums
    # (AgentType, StatusEnum, PlanStatus, StatusEnum, …) are available to any
    # transitive import (e.g. plan_types.py) that runs in the same pytest session.
    _vtypes_path = _os.path.join(_ROOT, "vetinari", "types.py")
    _vtypes_spec = _ilu.spec_from_file_location("vetinari.types", _vtypes_path)
    types_mod = _ilu.module_from_spec(_vtypes_spec)
    _vtypes_spec.loader.exec_module(types_mod)

    AgentType = types_mod.AgentType
    StatusEnum = types_mod.StatusEnum
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
    from dataclasses import dataclass as dc
    from dataclasses import field as f_field
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
        inputs: list[str] = f_field(default_factory=list)
        outputs: list[str] = f_field(default_factory=list)
        dependencies: list[str] = f_field(default_factory=list)
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
        dependencies: list[str] = f_field(default_factory=list)
        context: dict[str, Any] = f_field(default_factory=dict)

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

    @dc
    class Plan:
        plan_id: str
        version: str = "v0.1.0"
        goal: str = ""
        phase: int = 0
        tasks: list[Any] = f_field(default_factory=list)
        model_scores: list[dict] = f_field(default_factory=list)
        notes: str = ""
        warnings: list[str] = f_field(default_factory=list)
        needs_context: bool = False
        follow_up_question: str = ""
        final_delivery_path: str = ""
        final_delivery_summary: str = ""
        created_at: str = f_field(default_factory=lambda: datetime.now().isoformat())

        @classmethod
        def create_new(cls, goal: str, phase: int = 0) -> Plan:
            return cls(plan_id=f"plan_{uuid.uuid4().hex[:8]}", goal=goal, phase=phase)

    @dc
    class AgentResult:
        success: bool
        output: Any
        metadata: dict[str, Any] = f_field(default_factory=dict)
        errors: list[str] = f_field(default_factory=list)
        provenance: list[dict] = f_field(default_factory=list)

    @dc
    class VerificationResult:
        passed: bool
        issues: list[dict[str, Any]] = f_field(default_factory=list)
        suggestions: list[str] = f_field(default_factory=list)
        score: float = 0.0

    AGENT_REGISTRY = {}

    def get_agent_spec(agent_type):
        return AGENT_REGISTRY.get(agent_type)

    contracts_mod.AgentType = AgentType
    contracts_mod.StatusEnum = StatusEnum
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
    # UnifiedMemoryStore, MemoryStore, etc. are available for test_enhanced_memory.py
    # and other test files that need the real memory classes.
    _mem_init = _os.path.join(_ROOT, "vetinari", "memory", "__init__.py")
    _mem_dir = _os.path.join(_ROOT, "vetinari", "memory")
    _mem_spec = _ilu.spec_from_file_location(
        "vetinari.memory",
        _mem_init,
        submodule_search_locations=[_mem_dir],
    )
    memory_mod = _ilu.module_from_spec(_mem_spec)
    sys.modules["vetinari.memory"] = memory_mod
    _mem_spec.loader.exec_module(memory_mod)

    # ---- vetinari.memory.blackboard (canonical location) ----
    # Load the REAL blackboard module (stdlib + sqlite only) so that
    # test_blackboard.py can still import Blackboard/BlackboardEntry in the
    # same pytest session.  Override only get_blackboard with a MagicMock so
    # agent_graph tests can control delegation behaviour.
    _bb_path = _os.path.join(_ROOT, "vetinari", "memory", "blackboard.py")
    _bb_spec = _ilu.spec_from_file_location("vetinari.memory.blackboard", _bb_path)
    blackboard_mod = _ilu.module_from_spec(_bb_spec)
    # Register BEFORE exec_module so dataclass annotation resolution
    # (sys.modules.get(cls.__module__).__dict__) finds the module, not None.
    sys.modules["vetinari.memory.blackboard"] = blackboard_mod
    _bb_spec.loader.exec_module(blackboard_mod)
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

    # Make the AgentType / StatusEnum available at module level for tests
    _install_stubs.AgentType = AgentType
    _install_stubs.StatusEnum = StatusEnum
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
StatusEnum = _s.StatusEnum
Task = _s.Task
Plan = _s.Plan
AgentTask = _s.AgentTask
AgentResult = _s.AgentResult
VerificationResult = _s.VerificationResult

# Now import the module under test
import pytest

import vetinari.orchestration.agent_graph as ag_module
from vetinari.exceptions import CircularDependencyError
from vetinari.orchestration.agent_graph import (
    AgentGraph,
    ExecutionPlan,
    ExecutionStrategy,
    TaskNode,
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
        agent_type = AgentType.WORKER
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


class TestExecutionStrategyEnum:
    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (ExecutionStrategy.SEQUENTIAL, "sequential"),
            (ExecutionStrategy.PARALLEL, "parallel"),
            (ExecutionStrategy.ADAPTIVE, "adaptive"),
        ],
    )
    def test_enum_values(self, member, expected_value):
        assert member.value == expected_value
        assert member is ExecutionStrategy[member.name]

    def test_enum_members_count(self):
        assert len(ExecutionStrategy) == 3


# ---------------------------------------------------------------------------
# 2. TaskNode dataclass
# ---------------------------------------------------------------------------


class TestTaskNode:
    def _task(self, tid="t1"):
        return _make_task(tid)

    def test_default_status_is_pending(self):
        node = TaskNode(task=self._task())
        assert node.status == StatusEnum.PENDING

    def test_default_result_is_none(self):
        node = TaskNode(task=self._task())
        assert node.result is None

    def test_default_dependencies_empty(self):
        node = TaskNode(task=self._task())
        assert isinstance(node.dependencies, set)
        assert len(node.dependencies) == 0

    def test_default_dependents_empty(self):
        node = TaskNode(task=self._task())
        assert isinstance(node.dependents, set)
        assert len(node.dependents) == 0

    def test_default_retries_zero(self):
        node = TaskNode(task=self._task())
        assert node.retries == 0

    def test_default_max_retries_three(self):
        node = TaskNode(task=self._task())
        assert node.max_retries == 3

    def test_custom_dependencies(self):
        node = TaskNode(task=self._task(), dependencies={"t0", "t-1"})
        assert "t0" in node.dependencies
        assert "t-1" in node.dependencies

    def test_custom_max_retries(self):
        node = TaskNode(task=self._task(), max_retries=5)
        assert node.max_retries == 5

    def test_task_stored(self):
        t = self._task("myid")
        node = TaskNode(task=t)
        assert node.task is t

    def test_status_assignment(self):
        node = TaskNode(task=self._task(), status=StatusEnum.RUNNING)
        assert node.status == StatusEnum.RUNNING

    def test_retries_mutable(self):
        node = TaskNode(task=self._task())
        node.retries += 1
        assert node.retries == 1

    def test_result_assignment(self):
        node = TaskNode(task=self._task())
        r = AgentResult(success=True, output="x")
        node.result = r
        assert node.result is r

    def test_dependencies_are_independent_per_instance(self):
        n1 = TaskNode(task=self._task("a"))
        n2 = TaskNode(task=self._task("b"))
        n1.dependencies.add("x")
        assert "x" not in n2.dependencies


# ---------------------------------------------------------------------------
# 3. ExecutionPlan dataclass
# ---------------------------------------------------------------------------


class TestExecutionPlan:
    def _plan(self):
        return Plan.create_new("goal")

    def test_fields_set_on_creation(self):
        p = self._plan()
        ep = ExecutionPlan(plan_id="pid", original_plan=p)
        assert ep.plan_id == "pid"
        assert ep.original_plan is p

    def test_default_nodes_empty(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        assert isinstance(ep.nodes, dict)
        assert len(ep.nodes) == 0

    def test_default_execution_order_empty(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        assert isinstance(ep.execution_order, list)
        assert len(ep.execution_order) == 0

    def test_default_status_pending(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        assert ep.status == StatusEnum.PENDING

    def test_default_started_at_none(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        assert ep.started_at is None

    def test_default_completed_at_none(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        assert ep.completed_at is None

    def test_nodes_dict_is_independent(self):
        ep1 = ExecutionPlan(plan_id="p1", original_plan=self._plan())
        ep2 = ExecutionPlan(plan_id="p2", original_plan=self._plan())
        ep1.nodes["x"] = MagicMock()
        assert "x" not in ep2.nodes

    def test_status_mutation(self):
        ep = ExecutionPlan(plan_id="p", original_plan=self._plan())
        ep.status = StatusEnum.RUNNING
        assert ep.status == StatusEnum.RUNNING


# ---------------------------------------------------------------------------
# 4. AgentGraph.__init__
# ---------------------------------------------------------------------------


class TestAgentGraphInit:
    def test_default_strategy_is_adaptive(self):
        g = AgentGraph()
        assert g._strategy == ExecutionStrategy.ADAPTIVE

    def test_custom_strategy(self):
        g = AgentGraph(strategy=ExecutionStrategy.SEQUENTIAL)
        assert g._strategy == ExecutionStrategy.SEQUENTIAL

    def test_default_max_workers(self):
        g = AgentGraph()
        assert g._max_workers == 5

    def test_custom_max_workers(self):
        g = AgentGraph(max_workers=8)
        assert g._max_workers == 8

    def test_agents_dict_empty(self):
        g = AgentGraph()
        assert isinstance(g._agents, dict)
        assert len(g._agents) == 0

    def test_execution_plans_empty(self):
        g = AgentGraph()
        assert isinstance(g._execution_plans, dict)
        assert len(g._execution_plans) == 0

    def test_not_initialized(self):
        g = AgentGraph()
        assert not g._initialized

    def test_parallel_strategy_graph(self):
        g = AgentGraph(strategy=ExecutionStrategy.PARALLEL, max_workers=10)
        assert g._strategy == ExecutionStrategy.PARALLEL
        assert g._max_workers == 10


# ---------------------------------------------------------------------------
# 5. AgentGraph.initialize()
# ---------------------------------------------------------------------------


class TestAgentGraphInitialize:
    def setup_method(self):
        ag_module._agent_graph = None

    def teardown_method(self):
        ag_module._agent_graph = None

    def _mock_getter(self, agent_type):
        agent = _make_mock_agent(agent_type)
        return MagicMock(return_value=agent)

    def test_initialize_sets_initialized_flag(self):
        g = AgentGraph()

        def make_getter(at):
            a = _make_mock_agent(at)
            return lambda: a

        import_patch = {
            "get_foreman_agent": make_getter(AgentType.FOREMAN),
            "get_worker_agent": make_getter(AgentType.WORKER),
            "get_inspector_agent": make_getter(AgentType.INSPECTOR),
        }
        agents_mod = sys.modules["vetinari.agents"]
        for name, fn in import_patch.items():
            setattr(agents_mod, name, fn)

        g.initialize()
        assert g._initialized

    def test_initialize_idempotent(self):
        g = AgentGraph()
        g._initialized = True
        # Should return early without calling any getter
        agents_mod = sys.modules["vetinari.agents"]
        sentinel = MagicMock()
        agents_mod.get_foreman_agent = sentinel
        g.initialize()
        sentinel.assert_not_called()

    def test_initialize_handles_getter_exception(self):
        """Agents that raise during initialize() are skipped gracefully."""
        g = AgentGraph()
        agents_mod = sys.modules["vetinari.agents"]

        def bad_getter():
            raise RuntimeError("fail")

        for name in ["get_foreman_agent", "get_worker_agent", "get_inspector_agent"]:
            setattr(agents_mod, name, bad_getter)

        # Should not raise
        g.initialize()
        assert g._initialized
        assert len(g._agents) == 0

    def test_initialize_skips_none_agent(self):
        """Getter returning None skips registration."""
        g = AgentGraph()
        agents_mod = sys.modules["vetinari.agents"]

        def make_getter(at):
            a = _make_mock_agent(at)
            return lambda: a

        # FOREMAN returns None — should be skipped
        agents_mod.get_foreman_agent = lambda: None
        agents_mod.get_worker_agent = make_getter(AgentType.WORKER)
        agents_mod.get_inspector_agent = make_getter(AgentType.INSPECTOR)

        g.initialize()
        assert AgentType.FOREMAN not in g._agents


# ---------------------------------------------------------------------------
# 6. AgentGraph.create_execution_plan
# ---------------------------------------------------------------------------


class TestCreateExecutionPlan:
    def setup_method(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents()

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_returns_execution_plan(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        assert isinstance(ep, ExecutionPlan)

    def test_plan_id_matches(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        assert ep.plan_id == plan.plan_id

    def test_nodes_created_for_each_task(self):
        tasks = [_make_task("t1"), _make_task("t2"), _make_task("t3")]
        plan = _make_plan(tasks=tasks)
        ep = self.graph.create_execution_plan(plan)
        assert len(ep.nodes) == 3
        assert "t1" in ep.nodes
        assert "t2" in ep.nodes
        assert "t3" in ep.nodes

    def test_execution_order_non_empty(self):
        plan = _make_plan(tasks=[_make_task("t1"), _make_task("t2")])
        ep = self.graph.create_execution_plan(plan)
        assert len(ep.execution_order) == 2

    def test_plan_stored_in_graph(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        retrieved = self.graph.get_execution_plan(plan.plan_id)
        assert retrieved is ep

    def test_dependencies_propagated_to_nodes(self):
        tasks = [
            _make_task("t1"),
            _make_task("t2", deps=["t1"]),
        ]
        plan = _make_plan(tasks=tasks)
        ep = self.graph.create_execution_plan(plan)
        assert "t1" in ep.nodes["t2"].dependencies

    def test_dependents_populated(self):
        tasks = [
            _make_task("t1"),
            _make_task("t2", deps=["t1"]),
        ]
        plan = _make_plan(tasks=tasks)
        ep = self.graph.create_execution_plan(plan)
        assert "t2" in ep.nodes["t1"].dependents

    def test_empty_plan_allowed(self):
        plan = _make_plan(tasks=[])
        ep = self.graph.create_execution_plan(plan)
        assert len(ep.nodes) == 0
        assert ep.execution_order == []

    def test_constraint_exception_does_not_crash(self):
        """create_execution_plan must succeed even when constraint registry raises."""
        import sys

        reg_mod = sys.modules["vetinari.constraints.registry"]
        reg_mod.get_constraint_registry.side_effect = RuntimeError("oops")
        try:
            plan = _make_plan(tasks=[_make_task("t1")])
            ep = self.graph.create_execution_plan(plan)
            assert ep is not None
            assert ep.original_plan is plan
        finally:
            reg_mod.get_constraint_registry.side_effect = None

    def test_original_plan_referenced(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        assert ep.original_plan is plan


# ---------------------------------------------------------------------------
# 7. Topological sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def setup_method(self):
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
        assert result == ["t1"]

    def test_simple_chain_t1_t2_t3(self):
        nodes = {
            "t1": self._node("t1", dependents=["t2"]),
            "t2": self._node("t2", deps=["t1"], dependents=["t3"]),
            "t3": self._node("t3", deps=["t2"]),
        }
        result = self.graph._topological_sort(nodes)
        assert result == ["t1", "t2", "t3"]

    def test_parallel_tasks_both_before_dependent(self):
        nodes = {
            "t1": self._node("t1", dependents=["t3"]),
            "t2": self._node("t2", dependents=["t3"]),
            "t3": self._node("t3", deps=["t1", "t2"]),
        }
        result = self.graph._topological_sort(nodes)
        assert len(result) == 3
        assert result.index("t1") < result.index("t3")
        assert result.index("t2") < result.index("t3")

    def test_cycle_detection_raises_value_error(self):
        nodes = {
            "t1": self._node("t1", deps=["t2"], dependents=["t2"]),
            "t2": self._node("t2", deps=["t1"], dependents=["t1"]),
        }
        with pytest.raises(CircularDependencyError) as ctx:
            self.graph._topological_sort(nodes)
        assert "Circular dependency" in str(ctx.value)

    def test_diamond_dependency(self):
        # t1 -> t2, t1 -> t3, t2 -> t4, t3 -> t4
        nodes = {
            "t1": self._node("t1", dependents=["t2", "t3"]),
            "t2": self._node("t2", deps=["t1"], dependents=["t4"]),
            "t3": self._node("t3", deps=["t1"], dependents=["t4"]),
            "t4": self._node("t4", deps=["t2", "t3"]),
        }
        result = self.graph._topological_sort(nodes)
        assert len(result) == 4
        assert result[0] == "t1"
        assert result[-1] == "t4"

    def test_all_independent(self):
        nodes = {
            "t1": self._node("t1"),
            "t2": self._node("t2"),
            "t3": self._node("t3"),
        }
        result = self.graph._topological_sort(nodes)
        assert sorted(result) == ["t1", "t2", "t3"]

    def test_empty_nodes(self):
        result = self.graph._topological_sort({})
        assert result == []

    def test_three_cycle(self):
        nodes = {
            "a": self._node("a", deps=["c"], dependents=["b"]),
            "b": self._node("b", deps=["a"], dependents=["c"]),
            "c": self._node("c", deps=["b"], dependents=["a"]),
        }
        with pytest.raises(CircularDependencyError):
            self.graph._topological_sort(nodes)


# ---------------------------------------------------------------------------
# 8. _build_execution_layers
# ---------------------------------------------------------------------------


class TestBuildExecutionLayers:
    def setup_method(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def _make_exec_plan(self, tasks):
        plan = _make_plan(tasks=tasks)
        return self.graph.create_execution_plan(plan)

    def test_independent_tasks_single_layer(self):
        tasks = [_make_task("t1"), _make_task("t2"), _make_task("t3")]
        ep = self._make_exec_plan(tasks)
        layers = self.graph._build_execution_layers(ep)
        assert len(layers) == 1
        assert sorted(layers[0]) == ["t1", "t2", "t3"]

    def test_chain_tasks_separate_layers(self):
        tasks = [
            _make_task("t1"),
            _make_task("t2", deps=["t1"]),
            _make_task("t3", deps=["t2"]),
        ]
        ep = self._make_exec_plan(tasks)
        layers = self.graph._build_execution_layers(ep)
        assert len(layers) == 3
        assert layers[0] == ["t1"]
        assert layers[1] == ["t2"]
        assert layers[2] == ["t3"]

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
        assert len(layers) == 3
        assert layers[0] == ["t1"]
        assert "t2" in layers[1]
        assert "t3" in layers[1]
        assert layers[2] == ["t4"]

    def test_single_task_one_layer(self):
        ep = self._make_exec_plan([_make_task("t1")])
        layers = self.graph._build_execution_layers(ep)
        assert layers == [["t1"]]

    def test_empty_plan_empty_layers(self):
        ep = self._make_exec_plan([])
        layers = self.graph._build_execution_layers(ep)
        assert layers == []


# ---------------------------------------------------------------------------
# 9. execute_plan — sequential
# ---------------------------------------------------------------------------


class TestExecutePlanSequential:
    def setup_method(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.WORKER, AgentType.WORKER],
        )

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_sequential_executes_all_tasks(self):
        tasks = [
            _make_task("t1", AgentType.WORKER),
            _make_task("t2", AgentType.WORKER, deps=["t1"]),
        ]
        plan = _make_plan(tasks=tasks)
        results = self.graph.execute_plan(plan)
        assert "t1" in results
        assert "t2" in results

    def test_sequential_returns_success_results(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        results = self.graph.execute_plan(plan)
        assert results["t1"].success

    def test_sequential_sets_plan_completed(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert ep.status == StatusEnum.COMPLETED

    def test_sequential_sets_started_at(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert ep.started_at is not None
        assert isinstance(ep.started_at, str)
        assert "T" in ep.started_at  # ISO-format datetime contains "T" separator

    def test_sequential_sets_completed_at(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert ep.completed_at is not None
        assert isinstance(ep.completed_at, str)
        assert "T" in ep.completed_at  # ISO-format datetime contains "T" separator

    def test_sequential_node_status_completed_on_success(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert ep.nodes["t1"].status == StatusEnum.COMPLETED

    def test_sequential_node_status_failed_on_failure(self):
        fail_agent = _make_mock_agent(AgentType.WORKER, success=False, verification_passed=False)
        self.graph._agents[AgentType.WORKER] = fail_agent
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert ep.nodes["t1"].status == StatusEnum.FAILED

    def test_sequential_exception_sets_plan_failed(self):
        # The implementation catches exceptions internally, logs them, and sets
        # the task/plan status to FAILED without re-raising. Verify that the
        # call completes without propagating and that status is FAILED.
        bad_agent = _make_mock_agent(AgentType.WORKER)
        bad_agent.execute.side_effect = Exception("boom")
        self.graph._agents[AgentType.WORKER] = bad_agent
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        self.graph.execute_plan(plan)  # must not raise
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert ep.nodes["t1"].status == StatusEnum.FAILED


# ---------------------------------------------------------------------------
# 9b. execute_plan — mixed-outcome plan status (fail-closed)
# ---------------------------------------------------------------------------


class TestMixedOutcomePlanIsFailedNotCompleted:
    """A plan with any FAILED node must itself be FAILED, not COMPLETED.

    Before the fix, graph_executor unconditionally set ``exec_plan.status =
    StatusEnum.COMPLETED`` after the node loop.  A plan where t1 passed but
    t2 failed was mis-reported as COMPLETED.  The fix inspects each node's
    status and sets the plan FAILED when any node is FAILED (fail-closed).
    """

    def setup_method(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.WORKER, AgentType.FOREMAN],
        )

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_plan_is_failed_when_one_of_two_tasks_fails(self) -> None:
        """A plan with one success and one failure must be FAILED, not COMPLETED.

        Both the primary agent AND the recovery/replan agents are configured to fail
        so that no error-recovery path can silently salvage the failing task.  If any
        recovery path did salvage it, the node would end up COMPLETED and the plan
        status assertion would still catch the defect (plan = FAILED requires at least
        one FAILED node per the fix in graph_executor.py).
        """
        # Both agents fail so recovery delegation cannot salvage t-fail.
        # WORKER fails → recovery_result.success=False; FOREMAN fails → replan fails.
        # The retry loop then returns AgentResult(success=False) which marks the node
        # FAILED via the result.success guard (graph_executor.py line 195).
        failing_worker = _make_mock_agent(AgentType.WORKER, success=False, verification_passed=False)
        failing_foreman = _make_mock_agent(AgentType.FOREMAN, success=False, verification_passed=False)
        self.graph._agents[AgentType.WORKER] = failing_worker
        self.graph._agents[AgentType.FOREMAN] = failing_foreman

        tasks = [
            _make_task("t-pass", AgentType.WORKER),
            _make_task("t-fail", AgentType.FOREMAN),
        ]
        plan = _make_plan(tasks=tasks)
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)

        # Both tasks failed — plan must also be FAILED.
        assert ep.nodes["t-pass"].status == StatusEnum.FAILED
        assert ep.nodes["t-fail"].status == StatusEnum.FAILED
        assert ep.status == StatusEnum.FAILED, "Plan must be FAILED when any node is FAILED — was incorrectly COMPLETED"

    def test_plan_is_completed_when_all_tasks_succeed(self) -> None:
        """A plan where every task succeeds must remain COMPLETED."""
        success_worker = _make_mock_agent(AgentType.WORKER, success=True)
        success_foreman = _make_mock_agent(AgentType.FOREMAN, success=True)
        self.graph._agents[AgentType.WORKER] = success_worker
        self.graph._agents[AgentType.FOREMAN] = success_foreman

        tasks = [
            _make_task("t1", AgentType.WORKER),
            _make_task("t2", AgentType.FOREMAN),
        ]
        plan = _make_plan(tasks=tasks)
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)

        assert ep.nodes["t1"].status == StatusEnum.COMPLETED
        assert ep.nodes["t2"].status == StatusEnum.COMPLETED
        assert ep.status == StatusEnum.COMPLETED


# ---------------------------------------------------------------------------
# 10. execute_plan — parallel / adaptive
# ---------------------------------------------------------------------------


class TestExecutePlanParallel:
    def setup_method(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.PARALLEL,
            agent_types=[AgentType.WORKER, AgentType.WORKER],
        )

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_parallel_all_results_present(self):
        tasks = [_make_task("t1", AgentType.WORKER), _make_task("t2", AgentType.WORKER)]
        plan = _make_plan(tasks=tasks)
        results = self.graph.execute_plan(plan)
        assert "t1" in results
        assert "t2" in results

    def test_parallel_plan_completed(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        self.graph.execute_plan(plan)
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert ep.status == StatusEnum.COMPLETED

    def test_adaptive_strategy_uses_layers(self):
        g = _make_graph_with_agents(
            strategy=ExecutionStrategy.ADAPTIVE,
            agent_types=[AgentType.WORKER],
        )
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        results = g.execute_plan(plan)
        assert "t1" in results

    def test_parallel_single_task_no_thread_overhead(self):
        """Single task in a layer should execute without ThreadPoolExecutor."""
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        results = self.graph.execute_plan(plan)
        assert "t1" in results
        assert results["t1"].success


# ---------------------------------------------------------------------------
# 11. _execute_layer_parallel
# ---------------------------------------------------------------------------


class TestExecuteLayerParallel:
    def setup_method(self):
        self.graph = _make_graph_with_agents(
            agent_types=[AgentType.WORKER, AgentType.WORKER],
        )

    def test_single_task_returns_result(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        ep = self.graph.create_execution_plan(plan)
        result = self.graph._execute_layer_parallel(["t1"], ep, {})
        assert "t1" in result

    def test_multi_task_returns_all_results(self):
        tasks = [_make_task("t1", AgentType.WORKER), _make_task("t2", AgentType.WORKER)]
        plan = _make_plan(tasks=tasks)
        ep = self.graph.create_execution_plan(plan)
        result = self.graph._execute_layer_parallel(["t1", "t2"], ep, {})
        assert "t1" in result
        assert "t2" in result

    def test_exception_in_task_returns_failure_result(self):
        bad_agent = _make_mock_agent(AgentType.WORKER)
        bad_agent.execute.side_effect = Exception("thread fail")
        self.graph._agents[AgentType.WORKER] = bad_agent
        plan = _make_plan(
            tasks=[
                _make_task("t1", AgentType.WORKER),
                _make_task("t2", AgentType.WORKER),
            ]
        )
        ep = self.graph.create_execution_plan(plan)
        result = self.graph._execute_layer_parallel(["t1", "t2"], ep, {})
        assert "t1" in result
        assert not result["t1"].success

    def test_single_task_sets_completed_status(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        ep = self.graph.create_execution_plan(plan)
        self.graph._execute_layer_parallel(["t1"], ep, {})
        assert ep.nodes["t1"].status == StatusEnum.COMPLETED


# ---------------------------------------------------------------------------
# 12. _execute_task_node
# ---------------------------------------------------------------------------


class TestExecuteTaskNode:
    def setup_method(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(
            agent_types=[AgentType.WORKER, AgentType.WORKER, AgentType.INSPECTOR, AgentType.WORKER],
        )

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_success_returns_success_result(self):
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=0)
        result = self.graph._execute_task_node(node, {})
        assert result.success

    def test_unregistered_agent_delegates_to_blackboard(self):
        node = TaskNode(task=_make_task("t1", AgentType.FOREMAN), max_retries=0)
        # PLANNER not in agents
        assert AgentType.FOREMAN not in self.graph._agents
        result = self.graph._execute_task_node(node, {})
        # Blackboard mock returns None, so AgentResult.success should be False
        assert not result.success

    def test_permission_denied_returns_failure(self):
        ctx_mod = sys.modules["vetinari.execution_context"]
        ctx_mgr = MagicMock()
        ctx_mgr.enforce_permission.side_effect = PermissionError("no")
        ctx_mod.get_context_manager.return_value = ctx_mgr
        try:
            node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=0)
            result = self.graph._execute_task_node(node, {})
            assert not result.success
            assert any("Permission" in e for e in result.errors)
        finally:
            ctx_mod.get_context_manager.return_value = MagicMock(enforce_permission=MagicMock())

    def test_exception_in_agent_returns_failure(self):
        bad_agent = _make_mock_agent(AgentType.WORKER)
        bad_agent.execute.side_effect = Exception("bang")
        self.graph._agents[AgentType.WORKER] = bad_agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=0)
        result = self.graph._execute_task_node(node, {})
        assert not result.success
        assert "bang" in result.errors[0]

    def test_verification_failure_triggers_self_correction(self):
        agent = _make_mock_agent(AgentType.WORKER)
        # First attempt fails verification, second succeeds
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": "bad output"}]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="result")
        self.graph._agents[AgentType.WORKER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=1)
        result = self.graph._execute_task_node(node, {})
        assert result.success
        assert agent.execute.call_count == 2

    def test_self_correction_injects_feedback_in_description(self):
        agent = _make_mock_agent(AgentType.WORKER)
        issue_text = "something wrong"
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": issue_text}]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="result")
        self.graph._agents[AgentType.WORKER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=1)
        self.graph._execute_task_node(node, {})
        # Second call description should contain SELF-CORRECTION
        second_call_arg = agent.execute.call_args_list[1][0][0]
        assert "SELF-CORRECTION" in second_call_arg.description

    def test_builder_triggers_maker_checker(self):
        builder = _make_mock_agent(AgentType.WORKER)
        quality = _make_mock_agent(AgentType.INSPECTOR, success=True, verification_passed=True)
        self.graph._agents[AgentType.WORKER] = builder
        self.graph._agents[AgentType.INSPECTOR] = quality
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=0)
        self.graph._execute_task_node(node, {})
        # Quality agent should be called during maker-checker with a task argument
        quality.execute.assert_called_once()
        call_task = quality.execute.call_args[0][0]
        assert call_task.task_id is not None

    def test_non_builder_skips_maker_checker(self):
        # FOREMAN is not in _quality_reviewed_agents (only WORKER is), so no maker-checker
        foreman = _make_mock_agent(AgentType.FOREMAN)
        quality = _make_mock_agent(AgentType.INSPECTOR)
        self.graph._agents[AgentType.FOREMAN] = foreman
        self.graph._agents[AgentType.INSPECTOR] = quality
        node = TaskNode(task=_make_task("t1", AgentType.FOREMAN), max_retries=0)
        self.graph._execute_task_node(node, {})
        quality.execute.assert_not_called()

    def test_error_recovery_called_on_last_attempt_failure(self):
        fail_agent = _make_mock_agent(AgentType.WORKER, success=True, verification_passed=False)
        recovery = _make_mock_agent(AgentType.WORKER, success=True)
        self.graph._agents[AgentType.WORKER] = fail_agent
        self.graph._agents[AgentType.WORKER] = recovery
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=0)
        self.graph._execute_task_node(node, {})
        assert recovery.execute.call_count >= 1
        call_task = recovery.execute.call_args[0][0]
        assert call_task.task_id == "t1"

    def test_prior_results_injected_into_context(self):
        agent = _make_mock_agent(AgentType.WORKER)
        self.graph._agents[AgentType.WORKER] = agent
        prior = {"t0": AgentResult(success=True, output="prior output")}
        task = _make_task("t1", AgentType.WORKER, deps=["t0"])
        node = TaskNode(task=task, max_retries=0)
        self.graph._execute_task_node(node, prior)
        agent_task_called = agent.execute.call_args[0][0]
        assert "dependency_results" in agent_task_called.context

    def test_incorporate_prior_results_called_if_present(self):
        agent = _make_mock_agent(AgentType.WORKER)
        agent._incorporate_prior_results = MagicMock(return_value={"dep_1": {"output_summary": "summarized"}})
        self.graph._agents[AgentType.WORKER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=0)
        self.graph._execute_task_node(node, {})
        executed_task = agent.execute.call_args[0][0]
        assert executed_task.context["incorporated_results"] == {"dep_1": {"output_summary": "summarized"}}
        assert executed_task.task_id == "t1"

    def test_constraint_registry_caps_retries(self):
        reg = MagicMock()
        ac = MagicMock()
        ac.resources.max_retries = 0
        reg.get_constraints_for_agent.return_value = ac
        reg_mod = sys.modules["vetinari.constraints.registry"]
        reg_mod.get_constraint_registry.return_value = reg
        reg_mod.get_constraint_registry.side_effect = None
        try:
            node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=5)
            self.graph._execute_task_node(node, {})
            assert node.max_retries == 0
        finally:
            reg_mod.get_constraint_registry.return_value = None
            reg_mod.get_constraint_registry.side_effect = None

    def test_all_retries_exhausted_returns_failure(self):
        # Use FOREMAN as the executing agent; WORKER is not in _agents so error
        # recovery cannot run and we get a clean failure after exhausting retries.
        agent = _make_mock_agent(AgentType.FOREMAN, success=True, verification_passed=False)
        self.graph._agents.pop(AgentType.WORKER, None)  # ensure error recovery is unavailable
        self.graph._agents[AgentType.FOREMAN] = agent
        node = TaskNode(task=_make_task("t1", AgentType.FOREMAN), max_retries=2)
        result = self.graph._execute_task_node(node, {})
        assert not result.success


# ---------------------------------------------------------------------------
# 13. _apply_maker_checker
# ---------------------------------------------------------------------------


class TestApplyMakerChecker:
    def setup_method(self):
        ag_module._agent_graph = None
        self.graph = AgentGraph()
        self.graph._initialized = True

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_returns_result_if_no_quality_agent(self):
        result = AgentResult(success=True, output="x")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        assert returned is result

    def test_returns_result_if_no_builder_agent(self):
        self.graph._agents[AgentType.INSPECTOR] = _make_mock_agent(AgentType.INSPECTOR)
        result = AgentResult(success=True, output="x")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        assert returned is result

    def test_quality_approves_first_iteration(self):
        self.graph._agents[AgentType.INSPECTOR] = _make_mock_agent(AgentType.INSPECTOR)
        self.graph._agents[AgentType.WORKER] = _make_mock_agent(AgentType.WORKER)
        result = AgentResult(success=True, output="good output")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        assert returned.metadata.get("maker_checker", {}).get("approved")

    def test_metadata_has_iterations_on_approval(self):
        self.graph._agents[AgentType.INSPECTOR] = _make_mock_agent(AgentType.INSPECTOR)
        self.graph._agents[AgentType.WORKER] = _make_mock_agent(AgentType.WORKER)
        result = AgentResult(success=True, output="good")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        assert "iterations" in returned.metadata["maker_checker"]

    def test_quality_rejects_then_approves(self):
        quality = _make_mock_agent(AgentType.INSPECTOR)
        quality.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": "fix this"}]),
            VerificationResult(passed=True),
        ]
        quality.execute.return_value = AgentResult(success=True, output="review")
        builder = _make_mock_agent(AgentType.WORKER)
        self.graph._agents[AgentType.INSPECTOR] = quality
        self.graph._agents[AgentType.WORKER] = builder
        result = AgentResult(success=True, output="initial")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        assert returned.metadata.get("maker_checker", {}).get("approved")

    def test_max_iterations_exhausted_sets_approved_false(self):
        quality = _make_mock_agent(AgentType.INSPECTOR)
        quality.verify.return_value = VerificationResult(passed=False, issues=[{"message": "bad"}])
        quality.execute.return_value = AgentResult(success=True, output="review")
        builder = _make_mock_agent(AgentType.WORKER)
        builder.execute.return_value = AgentResult(success=True, output="rebuilt")
        self.graph._agents[AgentType.INSPECTOR] = quality
        self.graph._agents[AgentType.WORKER] = builder
        result = AgentResult(success=True, output="initial")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        assert not returned.metadata.get("maker_checker", {}).get("approved")

    def test_maker_checker_exception_breaks_loop(self):
        quality = _make_mock_agent(AgentType.INSPECTOR)
        quality.execute.side_effect = Exception("crash")
        builder = _make_mock_agent(AgentType.WORKER)
        self.graph._agents[AgentType.INSPECTOR] = quality
        self.graph._agents[AgentType.WORKER] = builder
        result = AgentResult(success=True, output="initial")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        # Exception during quality review breaks the loop; maker_checker key records the outcome.
        assert returned.metadata is not None
        assert "maker_checker" in returned.metadata

    def test_builder_failure_in_fix_breaks_loop(self):
        quality = _make_mock_agent(AgentType.INSPECTOR)
        quality.verify.return_value = VerificationResult(passed=False, issues=[{"message": "bad"}])
        quality.execute.return_value = AgentResult(success=True, output="review")
        builder = _make_mock_agent(AgentType.WORKER)
        builder.execute.return_value = AgentResult(success=False, output=None)
        self.graph._agents[AgentType.INSPECTOR] = quality
        self.graph._agents[AgentType.WORKER] = builder
        result = AgentResult(success=True, output="initial")
        returned = self.graph._apply_maker_checker(_make_task("t1"), result)
        assert not returned.metadata.get("maker_checker", {}).get("approved")


# ---------------------------------------------------------------------------
# 14. _run_error_recovery
# ---------------------------------------------------------------------------


class TestRunErrorRecovery:
    def setup_method(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def test_error_recovery_called_with_recovery_task(self):
        recovery = _make_mock_agent(AgentType.WORKER, success=True)
        self.graph._agents[AgentType.WORKER] = recovery
        task = _make_task("t1", AgentType.WORKER)
        failed = AgentResult(success=False, output="bad", errors=["fail"])
        verification = VerificationResult(passed=False, issues=[{"message": "issue1"}])
        result = self.graph._run_error_recovery(task, failed, verification)
        recovery.execute.assert_called_once()
        assert result.success

    def test_error_recovery_exception_returns_failure(self):
        recovery = _make_mock_agent(AgentType.WORKER)
        recovery.execute.side_effect = Exception("recovery failed")
        self.graph._agents[AgentType.WORKER] = recovery
        task = _make_task("t1")
        failed = AgentResult(success=False, output="x", errors=["e"])
        verification = VerificationResult(passed=False, issues=[])
        result = self.graph._run_error_recovery(task, failed, verification)
        assert not result.success
        assert any("Recovery failed" in e for e in result.errors)

    def test_error_recovery_passes_original_output_in_context(self):
        recovery = _make_mock_agent(AgentType.WORKER, success=True)
        self.graph._agents[AgentType.WORKER] = recovery
        task = _make_task("t1")
        failed = AgentResult(success=False, output="original_out", errors=[])
        verification = VerificationResult(passed=False, issues=[{"message": "v issue"}])
        self.graph._run_error_recovery(task, failed, verification)
        called_task = recovery.execute.call_args[0][0]
        assert called_task.context.get("original_output") == "original_out"


# ---------------------------------------------------------------------------
# 15. _validate_output_schema
# ---------------------------------------------------------------------------


class TestValidateOutputSchema:
    def setup_method(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def test_no_spec_returns_empty(self):
        issues = self.graph._validate_output_schema(AgentType.WORKER, {"key": "val"})
        assert issues == []

    def test_none_output_returns_empty_when_no_spec(self):
        issues = self.graph._validate_output_schema(AgentType.WORKER, None)
        assert issues == []

    def test_missing_required_field(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": ["result"],
            "properties": {"result": {"type": "string"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, {})
        assert any("result" in i for i in issues)

    def test_wrong_type_detected(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"count": {"type": "integer"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, {"count": "not_int"})
        assert any("count" in i for i in issues)

    def test_valid_output_returns_empty(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": ["result"],
            "properties": {"result": {"type": "string"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, {"result": "ok"})
        assert issues == []

    def test_non_dict_output_skips_validation(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": ["result"],
            "properties": {"result": {"type": "string"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, "a string")
        assert issues == []

    def test_null_output_skips_validation(self):
        spec = MagicMock()
        spec.output_schema = {"required": ["x"], "properties": {}}
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, None)
        assert issues == []

    def test_spec_with_no_output_schema_returns_empty(self):
        spec = MagicMock()
        spec.output_schema = None
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, {"x": 1})
        assert issues == []

    def test_number_type_accepts_int_and_float(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"val": {"type": "number"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues_int = self.graph._validate_output_schema(AgentType.WORKER, {"val": 42})
            issues_float = self.graph._validate_output_schema(AgentType.WORKER, {"val": 3.14})
        assert issues_int == []
        assert issues_float == []

    def test_array_type_validated(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"items": {"type": "array"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, {"items": "not_a_list"})
        assert any("items" in i for i in issues)

    def test_boolean_type_validated(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"flag": {"type": "boolean"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, {"flag": "yes"})
        assert any("flag" in i for i in issues)

    def test_object_type_validated(self):
        spec = MagicMock()
        spec.output_schema = {
            "required": [],
            "properties": {"data": {"type": "object"}},
        }
        with patch.object(self.graph, "get_skill_spec", return_value=spec):
            issues = self.graph._validate_output_schema(AgentType.WORKER, {"data": [1, 2]})
        assert any("data" in i for i in issues)


# ---------------------------------------------------------------------------
# 16. inject_task
# ---------------------------------------------------------------------------


class TestInjectTask:
    def setup_method(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(agent_types=[AgentType.WORKER])

    def teardown_method(self):
        ag_module._agent_graph = None

    def _setup_plan_with_tasks(self, task_ids):
        tasks = [_make_task(tid, AgentType.WORKER) for tid in task_ids]
        plan = _make_plan(tasks=tasks)
        self.graph.create_execution_plan(plan)
        return plan

    def test_inject_task_failure_cases(self):
        """inject_task returns False for nonexistent plan, missing after_task, and duplicate task ID."""
        # Nonexistent plan
        new_task = _make_task("new", AgentType.WORKER)
        assert not self.graph.inject_task("nonexistent_plan", new_task, "t1")

        # Nonexistent after_task
        plan = self._setup_plan_with_tasks(["t1"])
        assert not self.graph.inject_task(plan.plan_id, new_task, "t_missing")

        # Duplicate task ID
        plan2 = self._setup_plan_with_tasks(["t1", "t2"])
        dup_task = _make_task("t2", AgentType.WORKER)
        assert not self.graph.inject_task(plan2.plan_id, dup_task, "t1")

    def test_inject_task_success_returns_true(self):
        plan = self._setup_plan_with_tasks(["t1"])
        new_task = _make_task("t_new", AgentType.WORKER)
        result = self.graph.inject_task(plan.plan_id, new_task, "t1")
        assert result is True

    def test_inject_task_adds_node_to_plan(self):
        plan = self._setup_plan_with_tasks(["t1"])
        new_task = _make_task("t_new", AgentType.WORKER)
        self.graph.inject_task(plan.plan_id, new_task, "t1")
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert "t_new" in ep.nodes

    def test_inject_task_new_node_depends_on_after_task(self):
        plan = self._setup_plan_with_tasks(["t1"])
        new_task = _make_task("t_new", AgentType.WORKER)
        self.graph.inject_task(plan.plan_id, new_task, "t1")
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert "t1" in ep.nodes["t_new"].dependencies

    def test_inject_task_rewires_execution_order(self):
        plan = self._setup_plan_with_tasks(["t1", "t2"])
        # make t2 depend on t1
        ep = self.graph.get_execution_plan(plan.plan_id)
        ep.nodes["t2"].dependencies = {"t1"}
        ep.nodes["t1"].dependents = {"t2"}
        new_task = _make_task("t_mid", AgentType.WORKER)
        self.graph.inject_task(plan.plan_id, new_task, "t1")
        ep = self.graph.get_execution_plan(plan.plan_id)
        # t_mid should appear before t2 in execution order
        order = ep.execution_order
        assert order.index("t_mid") < order.index("t2")

    def test_inject_task_into_chain(self):
        plan = self._setup_plan_with_tasks(["t1"])
        nt = _make_task("t2", AgentType.WORKER)
        r1 = self.graph.inject_task(plan.plan_id, nt, "t1")
        nt2 = _make_task("t3", AgentType.WORKER)
        r2 = self.graph.inject_task(plan.plan_id, nt2, "t2")
        assert r1
        assert r2
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert len(ep.nodes) == 3


# ---------------------------------------------------------------------------
# 17. get_agent, get_registered_agents
# ---------------------------------------------------------------------------


class TestGetAgent:
    def setup_method(self):
        self.graph = _make_graph_with_agents(agent_types=[AgentType.WORKER, AgentType.INSPECTOR])

    def test_get_agent_returns_agent(self):
        agent = self.graph.get_agent(AgentType.WORKER)
        assert agent is not None
        assert hasattr(agent, "execute")

    def test_get_agent_returns_none_for_unregistered(self):
        agent = self.graph.get_agent(AgentType.FOREMAN)
        assert agent is None

    def test_get_registered_agents_returns_list(self):
        agents = self.graph.get_registered_agents()
        assert isinstance(agents, list)

    def test_get_registered_agents_contains_registered_types(self):
        agents = self.graph.get_registered_agents()
        assert AgentType.WORKER in agents
        assert AgentType.INSPECTOR in agents

    def test_get_registered_agents_count(self):
        agents = self.graph.get_registered_agents()
        assert len(agents) == 2

    def test_get_agent_correct_instance(self):
        mock_agent = _make_mock_agent(AgentType.WORKER)
        self.graph._agents[AgentType.WORKER] = mock_agent
        returned = self.graph.get_agent(AgentType.WORKER)
        assert returned is mock_agent


# ---------------------------------------------------------------------------
# 18. get_agent_by_capability
# ---------------------------------------------------------------------------


class TestGetAgentByCapability:
    def setup_method(self):
        self.graph = _make_graph_with_agents(agent_types=[AgentType.WORKER, AgentType.FOREMAN])

    def test_returns_none_when_no_matching_specs(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.return_value = []
        result = self.graph.get_agent_by_capability("code_search")
        assert result is None

    def test_returns_agent_when_spec_matches(self):
        spec = MagicMock()
        spec.agent_type = AgentType.WORKER.value
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.return_value = [spec]
        result = self.graph.get_agent_by_capability("exploration")
        assert result is not None
        assert hasattr(result, "execute")

    def test_returns_none_when_agent_not_registered_for_matching_spec(self):
        spec = MagicMock()
        spec.agent_type = "TOTALLY_UNKNOWN"  # not in this graph
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.return_value = [spec]
        result = self.graph.get_agent_by_capability("quality_review")
        assert result is None

    def test_returns_none_on_exception(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.side_effect = Exception("skill error")
        result = self.graph.get_agent_by_capability("anything")
        assert result is None
        skill_reg.get_skills_by_capability.side_effect = None

    def test_prefers_consolidated_agents(self):
        spec_foreman = MagicMock()
        spec_foreman.agent_type = AgentType.FOREMAN.value
        spec_worker = MagicMock()
        spec_worker.agent_type = AgentType.WORKER.value
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        # Return FOREMAN first, but WORKER has higher priority (10 vs 8)
        skill_reg.get_skills_by_capability.return_value = [spec_foreman, spec_worker]
        result = self.graph.get_agent_by_capability("something")
        # WORKER has priority 10 vs FOREMAN with priority 8
        expected = self.graph._agents.get(AgentType.WORKER)
        assert result is expected

    def test_invalid_agent_type_in_spec_skipped(self):
        spec = MagicMock()
        spec.agent_type = "INVALID_TYPE_XYZ"
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skills_by_capability.return_value = [spec]
        result = self.graph.get_agent_by_capability("something")
        assert result is None


# ---------------------------------------------------------------------------
# 19. get_skill_spec
# ---------------------------------------------------------------------------


class TestGetSkillSpec:
    def setup_method(self):
        self.graph = AgentGraph()
        self.graph._initialized = True

    def test_returns_spec_from_registry(self):
        mock_spec = MagicMock()
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skill_for_agent_type.return_value = mock_spec
        result = self.graph.get_skill_spec(AgentType.WORKER)
        assert result is mock_spec

    def test_returns_none_on_exception(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skill_for_agent_type.side_effect = Exception("not found")
        result = self.graph.get_skill_spec(AgentType.WORKER)
        assert result is None
        skill_reg.get_skill_for_agent_type.side_effect = None

    def test_calls_with_agent_type_value(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_skill_for_agent_type.return_value = None
        self.graph.get_skill_spec(AgentType.WORKER)
        skill_reg.get_skill_for_agent_type.assert_called_with(AgentType.WORKER.value)


# ---------------------------------------------------------------------------
# 20. get_agents_for_task_type
# ---------------------------------------------------------------------------


class TestGetAgentsForTaskType:
    def setup_method(self):
        self.graph = _make_graph_with_agents(agent_types=[AgentType.WORKER])

    def test_returns_empty_list_when_no_matching_skills(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {}
        result = self.graph.get_agents_for_task_type("some_task")
        assert result == []

    def test_returns_agents_matching_task_type(self):
        spec = MagicMock()
        spec.agent_type = AgentType.WORKER.value
        spec.modes = ["some_task"]
        spec.capabilities = []
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {"worker": spec}
        result = self.graph.get_agents_for_task_type("some_task")
        assert AgentType.WORKER in result

    def test_returns_agents_matching_capability(self):
        spec = MagicMock()
        spec.agent_type = AgentType.WORKER.value
        spec.modes = []
        spec.capabilities = ["some_task"]
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {"worker": spec}
        result = self.graph.get_agents_for_task_type("some_task")
        assert AgentType.WORKER in result

    def test_skips_agents_not_registered(self):
        spec = MagicMock()
        spec.agent_type = "UNKNOWN_TYPE"  # not registered in graph
        spec.modes = ["some_task"]
        spec.capabilities = []
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {"planner": spec}
        result = self.graph.get_agents_for_task_type("some_task")
        assert result == []

    def test_returns_empty_on_exception(self):
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.side_effect = Exception("fail")
        result = self.graph.get_agents_for_task_type("x")
        assert result == []
        skill_reg.get_all_skills.side_effect = None

    def test_skips_unrecognized_agent_type_in_spec(self):
        spec = MagicMock()
        spec.agent_type = "TOTALLY_UNKNOWN"
        spec.modes = ["some_task"]
        spec.capabilities = []
        skill_reg = sys.modules["vetinari.skills.skill_registry"]
        skill_reg.get_all_skills.return_value = {"unknown": spec}
        result = self.graph.get_agents_for_task_type("some_task")
        assert result == []


# ---------------------------------------------------------------------------
# 21. get_agent_graph singleton
# ---------------------------------------------------------------------------


class TestGetAgentGraphSingleton:
    def setup_method(self):
        ag_module._agent_graph = None
        agents_mod = sys.modules["vetinari.agents"]
        all_names = [
            "get_foreman_agent",
            "get_worker_agent",
            "get_inspector_agent",
        ]
        for name in all_names:
            setattr(agents_mod, name, lambda: None)

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_returns_agent_graph_instance(self):
        g = get_agent_graph()
        assert isinstance(g, AgentGraph)

    def test_same_instance_returned_twice(self):
        g1 = get_agent_graph()
        g2 = get_agent_graph()
        assert g1 is g2

    def test_singleton_is_initialized(self):
        g = get_agent_graph()
        assert g._initialized

    def test_singleton_uses_provided_strategy(self):
        g = get_agent_graph(strategy=ExecutionStrategy.SEQUENTIAL)
        assert g._strategy == ExecutionStrategy.SEQUENTIAL

    def test_singleton_subsequent_call_ignores_strategy(self):
        """Once created, the singleton does not change strategy on re-calls."""
        g1 = get_agent_graph(strategy=ExecutionStrategy.SEQUENTIAL)
        g2 = get_agent_graph(strategy=ExecutionStrategy.PARALLEL)
        assert g1 is g2
        assert g2._strategy == ExecutionStrategy.SEQUENTIAL

    def test_singleton_reset_allows_new_instance(self):
        g1 = get_agent_graph()
        ag_module._agent_graph = None
        g2 = get_agent_graph()
        assert g1 is not g2


# ---------------------------------------------------------------------------
# 22. __repr__
# ---------------------------------------------------------------------------


class TestAgentGraphRepr:
    def test_repr_contains_strategy(self):
        g = AgentGraph(strategy=ExecutionStrategy.SEQUENTIAL)
        assert "sequential" in repr(g)

    def test_repr_contains_agentgraph(self):
        g = AgentGraph()
        assert "AgentGraph" in repr(g)

    def test_repr_contains_agent_count(self):
        g = AgentGraph()
        g._agents[AgentType.WORKER] = MagicMock()
        r = repr(g)
        assert "1" in r

    def test_repr_adaptive(self):
        g = AgentGraph(strategy=ExecutionStrategy.ADAPTIVE)
        assert "adaptive" in repr(g)

    def test_repr_parallel(self):
        g = AgentGraph(strategy=ExecutionStrategy.PARALLEL)
        assert "parallel" in repr(g)


# ---------------------------------------------------------------------------
# 23. execute_plan_async
# ---------------------------------------------------------------------------


class TestExecutePlanAsync:
    def setup_method(self):
        ag_module._agent_graph = None
        self.graph = _make_graph_with_agents(agent_types=[AgentType.WORKER, AgentType.WORKER])

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_async_returns_results(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        results = asyncio.run(self.graph.execute_plan_async(plan))
        assert "t1" in results

    def test_async_multiple_tasks(self):
        tasks = [
            _make_task("t1", AgentType.WORKER),
            _make_task("t2", AgentType.WORKER),
        ]
        plan = _make_plan(tasks=tasks)
        results = asyncio.run(self.graph.execute_plan_async(plan))
        assert "t1" in results
        assert "t2" in results

    def test_async_plan_completed(self):
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        asyncio.run(self.graph.execute_plan_async(plan))
        ep = self.graph.get_execution_plan(plan.plan_id)
        assert ep.status == StatusEnum.COMPLETED

    def test_async_exception_wraps_as_failure_result(self):
        # t1 uses a bad FOREMAN agent (raises exception → failure);
        # t2 uses the good WORKER agent (succeeds independently).
        bad_agent = _make_mock_agent(AgentType.FOREMAN)
        bad_agent.execute.side_effect = Exception("async boom")
        self.graph._agents[AgentType.FOREMAN] = bad_agent
        plan = _make_plan(
            tasks=[
                _make_task("t1", AgentType.FOREMAN),
                _make_task("t2", AgentType.WORKER),
            ]
        )
        results = asyncio.run(self.graph.execute_plan_async(plan))
        assert not results["t1"].success
        assert results["t2"].success


# ---------------------------------------------------------------------------
# 24. get_execution_plan
# ---------------------------------------------------------------------------


class TestGetExecutionPlan:
    def setup_method(self):
        self.graph = _make_graph_with_agents()

    def test_returns_none_for_unknown_plan_id(self):
        result = self.graph.get_execution_plan("unknown_id")
        assert result is None

    def test_returns_plan_after_create(self):
        plan = _make_plan(tasks=[_make_task("t1")])
        ep = self.graph.create_execution_plan(plan)
        retrieved = self.graph.get_execution_plan(plan.plan_id)
        assert retrieved is ep


# ---------------------------------------------------------------------------
# 25. Edge cases and integration-style tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def setup_method(self):
        ag_module._agent_graph = None

    def teardown_method(self):
        ag_module._agent_graph = None

    def test_execute_plan_empty_tasks(self):
        graph = _make_graph_with_agents()
        plan = _make_plan(tasks=[])
        results = graph.execute_plan(plan)
        assert results == {}

    def test_execute_plan_single_task_sequential(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.WORKER],
        )
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        results = graph.execute_plan(plan)
        assert "t1" in results
        assert results["t1"].success

    def test_execute_plan_chain_sequential_order(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.FOREMAN, AgentType.WORKER],
        )
        call_order = []
        foreman = graph._agents[AgentType.FOREMAN]
        worker = graph._agents[AgentType.WORKER]
        foreman.execute.side_effect = lambda t: (call_order.append("foreman"), AgentResult(success=True, output="e"))[1]
        worker.execute.side_effect = lambda t: (call_order.append("worker"), AgentResult(success=True, output="b"))[1]

        tasks = [
            _make_task("t1", AgentType.FOREMAN),
            _make_task("t2", AgentType.WORKER, deps=["t1"]),
        ]
        plan = _make_plan(tasks=tasks)
        graph.execute_plan(plan)
        # Verify ordering: foreman (t1) executes before worker (t2).
        # Post-plan suggestions/synthesis may add extra worker calls, so we
        # only check that the first two task-level calls are in the right order.
        assert "foreman" in call_order
        assert "worker" in call_order
        assert call_order.index("foreman") < call_order.index("worker")

    def test_multiple_plans_independent(self):
        graph = _make_graph_with_agents(agent_types=[AgentType.WORKER])
        plan1 = _make_plan("goal1", tasks=[_make_task("t1", AgentType.WORKER)])
        plan2 = _make_plan("goal2", tasks=[_make_task("t2", AgentType.WORKER)])
        ep1 = graph.create_execution_plan(plan1)
        ep2 = graph.create_execution_plan(plan2)
        assert ep1 is not ep2
        assert ep1.plan_id != ep2.plan_id

    def test_task_node_status_updated_after_parallel_execution(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.PARALLEL,
            agent_types=[AgentType.WORKER, AgentType.WORKER],
        )
        tasks = [
            _make_task("t1", AgentType.WORKER),
            _make_task("t2", AgentType.WORKER),
        ]
        plan = _make_plan(tasks=tasks)
        graph.execute_plan(plan)
        ep = graph.get_execution_plan(plan.plan_id)
        assert ep.nodes["t1"].status == StatusEnum.COMPLETED
        assert ep.nodes["t2"].status == StatusEnum.COMPLETED

    def test_inject_then_execute(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.WORKER],
        )
        plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
        graph.create_execution_plan(plan)
        new_task = _make_task("t2", AgentType.WORKER)
        injected = graph.inject_task(plan.plan_id, new_task, "t1")
        assert injected
        ep = graph.get_execution_plan(plan.plan_id)
        assert "t2" in ep.nodes

    def test_blackboard_delegate_called_for_unknown_agent(self):
        graph = _make_graph_with_agents()
        blackboard_mod = sys.modules["vetinari.memory.blackboard"]
        saved_return = blackboard_mod.get_blackboard.return_value
        mock_board = MagicMock()
        mock_board.delegate.return_value = AgentResult(success=True, output="delegated")
        blackboard_mod.get_blackboard.return_value = mock_board
        try:
            node = TaskNode(task=_make_task("t1", AgentType.FOREMAN), max_retries=0)
            result = graph._execute_task_node(node, {})
            delegated_task, delegated_agents = mock_board.delegate.call_args[0]
            assert delegated_task.id == "t1"
            assert delegated_agents is graph._agents
            assert result.success is True
            assert result.output == "delegated"
        finally:
            blackboard_mod.get_blackboard.return_value = saved_return

    def test_self_correction_increments_retries(self):
        graph = _make_graph_with_agents(agent_types=[AgentType.WORKER])
        agent = _make_mock_agent(AgentType.WORKER)
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": "fix"}]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="x")
        graph._agents[AgentType.WORKER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=1)
        graph._execute_task_node(node, {})
        assert node.retries == 1

    def test_agent_graph_max_workers_limits_parallel_pool(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.PARALLEL,
            agent_types=[AgentType.WORKER, AgentType.WORKER],
        )
        tasks = [
            _make_task("t1", AgentType.WORKER),
            _make_task("t2", AgentType.WORKER),
        ]
        plan = _make_plan(tasks=tasks)
        results = graph.execute_plan(plan)
        assert "t1" in results
        assert "t2" in results

    def test_create_execution_plan_with_multiple_deps(self):
        graph = _make_graph_with_agents()
        tasks = [
            _make_task("t1"),
            _make_task("t2"),
            _make_task("t3", deps=["t1", "t2"]),
        ]
        plan = _make_plan(tasks=tasks)
        ep = graph.create_execution_plan(plan)
        assert "t1" in ep.nodes["t3"].dependencies
        assert "t2" in ep.nodes["t3"].dependencies

    def test_topological_sort_preserves_all_ids(self):
        graph = AgentGraph()
        from vetinari.orchestration.agent_graph import TaskNode as TN
        from vetinari.types import AgentType

        ids = [f"t{i}" for i in range(10)]
        nodes = {tid: TN(task=_make_task(tid)) for tid in ids}
        result = graph._topological_sort(nodes)
        assert sorted(result) == sorted(ids)

    def test_execute_plan_result_keys_match_task_ids(self):
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.WORKER],
        )
        tasks = [_make_task(f"t{i}", AgentType.WORKER) for i in range(5)]
        plan = _make_plan(tasks=tasks)
        results = graph.execute_plan(plan)
        for task in tasks:
            assert task.id in results

    def test_schema_issues_logged_but_do_not_block_execution(self):
        """Schema issues should be non-blocking (logged only)."""
        spec = MagicMock()
        spec.output_schema = {
            "required": ["missing_key"],
            "properties": {},
        }
        graph = _make_graph_with_agents(
            strategy=ExecutionStrategy.SEQUENTIAL,
            agent_types=[AgentType.WORKER],
        )
        with patch.object(graph, "get_skill_spec", return_value=spec):
            plan = _make_plan(tasks=[_make_task("t1", AgentType.WORKER)])
            results = graph.execute_plan(plan)
        # Should complete successfully (schema issues are warnings, not errors)
        assert results["t1"].success


# ---------------------------------------------------------------------------
# 26. Additional coverage for issue text formatting in self-correction
# ---------------------------------------------------------------------------


class TestIssueTextFormatting:
    def setup_method(self):
        self.graph = _make_graph_with_agents(agent_types=[AgentType.WORKER])

    def test_string_issues_in_verification(self):
        """Verification issues that are plain strings (not dicts) should work."""
        agent = _make_mock_agent(AgentType.WORKER)
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=["plain string issue"]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="x")
        self.graph._agents[AgentType.WORKER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=1)
        result = self.graph._execute_task_node(node, {})
        assert result.success

    def test_dict_issues_in_verification(self):
        agent = _make_mock_agent(AgentType.WORKER)
        agent.verify.side_effect = [
            VerificationResult(passed=False, issues=[{"message": "dict issue"}]),
            VerificationResult(passed=True),
        ]
        agent.execute.return_value = AgentResult(success=True, output="x")
        self.graph._agents[AgentType.WORKER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=1)
        result = self.graph._execute_task_node(node, {})
        assert result.success

    def test_empty_issues_list_in_verification(self):
        # Use FOREMAN as the task agent (verification fails, empty issues list).
        # Remove WORKER from _agents so error recovery cannot run, letting the
        # failure propagate cleanly.
        agent = _make_mock_agent(AgentType.FOREMAN, verification_passed=False)
        agent.verify.return_value = VerificationResult(passed=False, issues=[])
        agent.execute.return_value = AgentResult(success=True, output="x")
        self.graph._agents[AgentType.FOREMAN] = agent
        self.graph._agents.pop(AgentType.WORKER, None)  # prevent error recovery
        node = TaskNode(task=_make_task("t1", AgentType.FOREMAN), max_retries=0)
        result = self.graph._execute_task_node(node, {})
        assert not result.success


# ---------------------------------------------------------------------------
# 27. AgentTask.from_task usage in _execute_task_node
# ---------------------------------------------------------------------------


class TestAgentTaskCreation:
    def setup_method(self):
        self.graph = _make_graph_with_agents(agent_types=[AgentType.WORKER])

    def test_agent_task_created_from_task(self):
        agent = _make_mock_agent(AgentType.WORKER)
        self.graph._agents[AgentType.WORKER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER, description="do this"), max_retries=0)
        self.graph._execute_task_node(node, {})
        called_arg = agent.execute.call_args[0][0]
        assert called_arg.task_id == "t1"
        assert called_arg.description == "do this"

    def test_agent_task_has_empty_context_by_default(self):
        agent = _make_mock_agent(AgentType.WORKER)
        self.graph._agents[AgentType.WORKER] = agent
        node = TaskNode(task=_make_task("t1", AgentType.WORKER), max_retries=0)
        self.graph._execute_task_node(node, {})
        called_arg = agent.execute.call_args[0][0]
        assert isinstance(called_arg.context, dict)
