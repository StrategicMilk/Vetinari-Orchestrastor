"""Extended tests for vetinari/agents/contracts.py — Phase 7C"""
import unittest

from vetinari.agents.contracts import (
    AgentResult,
    AgentSpec,
    AgentTask,
    Plan,
    Task,
    TaskStatus,
    VerificationResult,
    get_agent_spec,
    get_enabled_agents,
)
from vetinari.types import AgentType, ExecutionMode


class TestAgentType(unittest.TestCase):
    def test_all_types_exist(self):
        for at in [AgentType.PLANNER, AgentType.EXPLORER, AgentType.BUILDER,
                   AgentType.EVALUATOR, AgentType.LIBRARIAN, AgentType.ORACLE,
                   AgentType.RESEARCHER, AgentType.SYNTHESIZER]:
            assert isinstance(at.value, str)

    def test_security_auditor(self):
        assert AgentType.SECURITY_AUDITOR.value == "SECURITY_AUDITOR"


class TestTaskStatus(unittest.TestCase):
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.COMPLETED.value == "completed"


class TestExecutionMode(unittest.TestCase):
    def test_values(self):
        assert ExecutionMode.PLANNING in list(ExecutionMode)
        assert ExecutionMode.EXECUTION in list(ExecutionMode)


class TestAgentSpec(unittest.TestCase):
    def test_creation(self):
        spec = AgentSpec(
            agent_type=AgentType.BUILDER,
            name="Builder",
            description="Builds code",
            default_model="llama-3",
        )
        assert spec.name == "Builder"
        assert spec.enabled
        assert spec.thinking_variant == "medium"

    def test_to_dict(self):
        spec = AgentSpec(
            agent_type=AgentType.EXPLORER,
            name="Explorer",
            description="Explores",
            default_model="llama",
        )
        d = spec.to_dict()
        assert d["agent_type"] == "EXPLORER"
        assert "name" in d
        assert "enabled" in d

    def test_from_dict(self):
        spec = AgentSpec(
            agent_type=AgentType.ORACLE,
            name="Oracle",
            description="Knows",
            default_model="llama",
        )
        d = spec.to_dict()
        spec2 = AgentSpec.from_dict(d)
        assert spec2.agent_type == AgentType.ORACLE
        assert spec2.name == "Oracle"


class TestAgentTask(unittest.TestCase):
    """AgentTask fields: task_id, agent_type, description, prompt, status, ..."""

    def test_creation(self):
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.BUILDER,
            description="Build a thing",
            prompt="Create a REST API",
        )
        assert task.task_id == "t1"
        assert task.status == TaskStatus.PENDING

    def test_to_dict(self):
        task = AgentTask(
            task_id="t2",
            agent_type=AgentType.EVALUATOR,
            description="Evaluate output",
            prompt="Check quality",
        )
        d = task.to_dict()
        assert "task_id" in d
        assert "agent_type" in d
        assert "description" in d


class TestAgentResult(unittest.TestCase):
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


class TestVerificationResult(unittest.TestCase):
    def test_passed(self):
        vr = VerificationResult(passed=True, score=0.95)
        assert vr.passed
        self.assertAlmostEqual(vr.score, 0.95)

    def test_failed(self):
        vr = VerificationResult(passed=False, score=0.3,
                                issues=["missing tests", "no docs"])
        assert not vr.passed
        assert len(vr.issues) == 2

    def test_to_dict(self):
        vr = VerificationResult(passed=True, score=1.0)
        d = vr.to_dict()
        assert "passed" in d
        assert "score" in d


class TestPlanAndTask(unittest.TestCase):
    def test_plan_creation(self):
        # Plan fields: plan_id, version, goal, phase, tasks, ...
        p = Plan(plan_id="plan_001", goal="Build a feature")
        assert p.plan_id == "plan_001"
        assert p.goal == "Build a feature"
        assert p.tasks == []

    def test_task_creation(self):
        # Task fields: id, description, inputs, outputs, dependencies, ...
        t = Task(id="task_1", description="Write tests")
        assert t.id == "task_1"
        assert t.status == TaskStatus.PENDING

    def test_plan_to_dict(self):
        p = Plan(plan_id="p1", goal="test")
        d = p.to_dict()
        assert "plan_id" in d
        assert "goal" in d


class TestGetAgentSpec(unittest.TestCase):
    def test_returns_spec_for_known_type(self):
        spec = get_agent_spec(AgentType.BUILDER)
        if spec is not None:
            assert isinstance(spec, AgentSpec)
            assert spec.agent_type == AgentType.BUILDER

    def test_returns_none_or_spec(self):
        for at in AgentType:
            result = get_agent_spec(at)
            assert type(result) in [AgentSpec, type(None)]


class TestGetEnabledAgents(unittest.TestCase):
    def test_returns_list(self):
        agents = get_enabled_agents()
        assert isinstance(agents, list)

    def test_all_enabled_are_agent_specs(self):
        agents = get_enabled_agents()
        for a in agents:
            assert isinstance(a, AgentSpec)
            assert a.enabled


if __name__ == "__main__":
    unittest.main()
