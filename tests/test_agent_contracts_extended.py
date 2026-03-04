"""Extended tests for vetinari/agents/contracts.py — Phase 7C"""
import unittest
from vetinari.agents.contracts import (
    AgentType, TaskStatus, ExecutionMode,
    AgentSpec, AgentTask, AgentResult, VerificationResult,
    Plan, Task,
    get_agent_spec, get_enabled_agents,
)


class TestAgentType(unittest.TestCase):
    def test_all_types_exist(self):
        for at in [AgentType.PLANNER, AgentType.EXPLORER, AgentType.BUILDER,
                   AgentType.EVALUATOR, AgentType.LIBRARIAN, AgentType.ORACLE,
                   AgentType.RESEARCHER, AgentType.SYNTHESIZER]:
            self.assertIsInstance(at.value, str)

    def test_security_auditor(self):
        self.assertEqual(AgentType.SECURITY_AUDITOR.value, "SECURITY_AUDITOR")


class TestTaskStatus(unittest.TestCase):
    def test_values(self):
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.COMPLETED.value, "completed")


class TestExecutionMode(unittest.TestCase):
    def test_values(self):
        self.assertIn(ExecutionMode.PLANNING, list(ExecutionMode))
        self.assertIn(ExecutionMode.EXECUTION, list(ExecutionMode))


class TestAgentSpec(unittest.TestCase):
    def test_creation(self):
        spec = AgentSpec(
            agent_type=AgentType.BUILDER,
            name="Builder",
            description="Builds code",
            default_model="llama-3",
        )
        self.assertEqual(spec.name, "Builder")
        self.assertTrue(spec.enabled)
        self.assertEqual(spec.thinking_variant, "medium")

    def test_to_dict(self):
        spec = AgentSpec(
            agent_type=AgentType.EXPLORER,
            name="Explorer",
            description="Explores",
            default_model="llama",
        )
        d = spec.to_dict()
        self.assertEqual(d["agent_type"], "EXPLORER")
        self.assertIn("name", d)
        self.assertIn("enabled", d)

    def test_from_dict(self):
        spec = AgentSpec(
            agent_type=AgentType.ORACLE,
            name="Oracle",
            description="Knows",
            default_model="llama",
        )
        d = spec.to_dict()
        spec2 = AgentSpec.from_dict(d)
        self.assertEqual(spec2.agent_type, AgentType.ORACLE)
        self.assertEqual(spec2.name, "Oracle")


class TestAgentTask(unittest.TestCase):
    """AgentTask fields: task_id, agent_type, description, prompt, status, ..."""

    def test_creation(self):
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.BUILDER,
            description="Build a thing",
            prompt="Create a REST API",
        )
        self.assertEqual(task.task_id, "t1")
        self.assertEqual(task.status, TaskStatus.PENDING)

    def test_to_dict(self):
        task = AgentTask(
            task_id="t2",
            agent_type=AgentType.EVALUATOR,
            description="Evaluate output",
            prompt="Check quality",
        )
        d = task.to_dict()
        self.assertIn("task_id", d)
        self.assertIn("agent_type", d)
        self.assertIn("description", d)


class TestAgentResult(unittest.TestCase):
    """AgentResult fields: success, output, metadata, errors, provenance."""

    def test_success_result(self):
        r = AgentResult(success=True, output="done")
        self.assertTrue(r.success)
        self.assertEqual(r.errors, [])

    def test_failure_result(self):
        r = AgentResult(success=False, output=None, errors=["timeout"])
        self.assertFalse(r.success)
        self.assertIn("timeout", r.errors)

    def test_to_dict(self):
        r = AgentResult(success=True, output="ok")
        d = r.to_dict()
        self.assertIn("success", d)
        self.assertIn("output", d)


class TestVerificationResult(unittest.TestCase):
    def test_passed(self):
        vr = VerificationResult(passed=True, score=0.95)
        self.assertTrue(vr.passed)
        self.assertAlmostEqual(vr.score, 0.95)

    def test_failed(self):
        vr = VerificationResult(passed=False, score=0.3,
                                issues=["missing tests", "no docs"])
        self.assertFalse(vr.passed)
        self.assertEqual(len(vr.issues), 2)

    def test_to_dict(self):
        vr = VerificationResult(passed=True, score=1.0)
        d = vr.to_dict()
        self.assertIn("passed", d)
        self.assertIn("score", d)


class TestPlanAndTask(unittest.TestCase):
    def test_plan_creation(self):
        # Plan fields: plan_id, version, goal, phase, tasks, ...
        p = Plan(plan_id="plan_001", goal="Build a feature")
        self.assertEqual(p.plan_id, "plan_001")
        self.assertEqual(p.goal, "Build a feature")
        self.assertEqual(p.tasks, [])

    def test_task_creation(self):
        # Task fields: id, description, inputs, outputs, dependencies, ...
        t = Task(id="task_1", description="Write tests")
        self.assertEqual(t.id, "task_1")
        self.assertEqual(t.status, TaskStatus.PENDING)

    def test_plan_to_dict(self):
        p = Plan(plan_id="p1", goal="test")
        d = p.to_dict()
        self.assertIn("plan_id", d)
        self.assertIn("goal", d)


class TestGetAgentSpec(unittest.TestCase):
    def test_returns_spec_for_known_type(self):
        spec = get_agent_spec(AgentType.BUILDER)
        if spec is not None:
            self.assertIsInstance(spec, AgentSpec)
            self.assertEqual(spec.agent_type, AgentType.BUILDER)

    def test_returns_none_or_spec(self):
        for at in AgentType:
            result = get_agent_spec(at)
            self.assertIn(type(result), [AgentSpec, type(None)])


class TestGetEnabledAgents(unittest.TestCase):
    def test_returns_list(self):
        agents = get_enabled_agents()
        self.assertIsInstance(agents, list)

    def test_all_enabled_are_agent_specs(self):
        agents = get_enabled_agents()
        for a in agents:
            self.assertIsInstance(a, AgentSpec)
            self.assertTrue(a.enabled)


if __name__ == "__main__":
    unittest.main()
