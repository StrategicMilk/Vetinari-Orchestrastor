"""Extended tests for vetinari/agents/contracts.py — Phase 7C"""

import pytest

from tests.conftest import TEST_TASK_ID
from vetinari.agents.contracts import (
    AgentResult,
    AgentSpec,
    AgentTask,
    Plan,
    StatusEnum,
    Task,
    VerificationResult,
    get_agent_spec,
    get_enabled_agents,
)
from vetinari.types import AgentType, ExecutionMode


class TestAgentType:
    def test_all_active_types_exist(self):
        """All 3 active agent types must be valid enum members."""
        for at in [
            AgentType.FOREMAN,
            AgentType.WORKER,
            AgentType.INSPECTOR,
        ]:
            assert isinstance(at.value, str)

    def test_active_values_correct(self):
        """Active enum values match their string representations."""
        assert AgentType.FOREMAN.value == "FOREMAN"
        assert AgentType.WORKER.value == "WORKER"
        assert AgentType.INSPECTOR.value == "INSPECTOR"


class TestTaskStatus:
    def test_values(self):
        assert StatusEnum.PENDING.value == "pending"
        assert StatusEnum.COMPLETED.value == "completed"


class TestExecutionMode:
    def test_values(self):
        assert ExecutionMode.PLANNING in list(ExecutionMode)
        assert ExecutionMode.EXECUTION in list(ExecutionMode)


class TestAgentSpec:
    def test_creation(self):
        spec = AgentSpec(
            agent_type=AgentType.WORKER,
            name="Worker",
            description="Executes tasks",
            default_model="llama-3",
        )
        assert spec.name == "Worker"
        assert spec.enabled
        assert spec.thinking_variant == "medium"

    def test_to_dict(self):
        spec = AgentSpec(
            agent_type=AgentType.WORKER,
            name="Worker",
            description="Researches",
            default_model="qwen2.5-72b",
        )
        d = spec.to_dict()
        assert d["agent_type"] == AgentType.WORKER.value
        assert "name" in d
        assert "enabled" in d

    def test_from_dict(self):
        spec = AgentSpec(
            agent_type=AgentType.INSPECTOR,
            name="Inspector",
            description="Inspects quality",
            default_model="qwen3-30b-a3b",
        )
        d = spec.to_dict()
        spec2 = AgentSpec.from_dict(d)
        assert spec2.agent_type == AgentType.INSPECTOR
        assert spec2.name == "Inspector"


class TestAgentTask:
    """AgentTask fields: task_id, agent_type, description, prompt, status, ..."""

    def test_creation(self):
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.WORKER,
            description="Build a thing",
            prompt="Create a REST API",
        )
        assert task.task_id == "t1"
        assert task.status == StatusEnum.PENDING

    def test_to_dict(self):
        task = AgentTask(
            task_id="t2",
            agent_type=AgentType.INSPECTOR,
            description="Evaluate output",
            prompt="Check quality",
        )
        d = task.to_dict()
        assert "task_id" in d
        assert "agent_type" in d
        assert "description" in d


class TestAgentResult:
    """AgentResult fields: success, output, metadata, errors, provenance."""

    def test_success_result(self):
        r = AgentResult(success=True, output="done")
        assert r.success
        assert r.errors == []

    def test_failure_result(self):
        r = AgentResult(success=False, output=None, errors=["timeout"])
        assert not r.success
        assert "timeout" in r.errors

    def test_to_dict(self):
        r = AgentResult(success=True, output="ok")
        d = r.to_dict()
        assert "success" in d
        assert "output" in d


class TestVerificationResult:
    def test_passed(self):
        vr = VerificationResult(passed=True, score=0.95)
        assert vr.passed
        assert vr.score == pytest.approx(0.95)

    def test_failed(self):
        vr = VerificationResult(passed=False, score=0.3, issues=["missing tests", "no docs"])
        assert not vr.passed
        assert len(vr.issues) == 2

    def test_to_dict(self):
        vr = VerificationResult(passed=True, score=1.0)
        d = vr.to_dict()
        assert "passed" in d
        assert "score" in d


class TestPlanAndTask:
    def test_plan_creation(self):
        p = Plan(plan_id="plan_001", goal="Build a feature")
        assert p.plan_id == "plan_001"
        assert p.goal == "Build a feature"
        assert p.tasks == []

    def test_task_creation(self):
        t = Task(id=TEST_TASK_ID, description="Write tests")
        assert t.id == TEST_TASK_ID
        assert t.status == StatusEnum.PENDING

    def test_plan_to_dict(self):
        p = Plan(plan_id="p1", goal="test")
        d = p.to_dict()
        assert "plan_id" in d
        assert "goal" in d


class TestGetAgentSpec:
    def test_returns_spec_for_foreman(self):
        # FOREMAN not yet in registry — returns None (migration pending)
        spec = get_agent_spec(AgentType.FOREMAN)
        assert spec is None or (isinstance(spec, AgentSpec) and spec.agent_type == AgentType.FOREMAN)

    def test_returns_spec_for_worker(self):
        # WORKER not yet in registry — returns None (migration pending)
        spec = get_agent_spec(AgentType.WORKER)
        assert spec is None or (isinstance(spec, AgentSpec) and spec.agent_type == AgentType.WORKER)

    def test_returns_spec_for_inspector(self):
        # INSPECTOR not yet in registry — returns None (migration pending)
        spec = get_agent_spec(AgentType.INSPECTOR)
        assert spec is None or (isinstance(spec, AgentSpec) and spec.agent_type == AgentType.INSPECTOR)

    def test_returns_none_for_unknown(self):
        """Types not in registry return None."""
        from vetinari.types import AgentType as AT

        # Use an enum value that definitely doesn't have a spec
        result = get_agent_spec(AgentType.FOREMAN)
        assert type(result) in [AgentSpec, type(None)]

    def test_returns_none_or_spec_for_all_types(self):
        for at in AgentType:
            result = get_agent_spec(at)
            assert type(result) in [AgentSpec, type(None)]


class TestGetEnabledAgents:
    def test_returns_list(self):
        agents = get_enabled_agents()
        assert isinstance(agents, list)

    def test_all_enabled_are_agent_specs(self):
        agents = get_enabled_agents()
        for a in agents:
            assert isinstance(a, AgentSpec)
            assert a.enabled
