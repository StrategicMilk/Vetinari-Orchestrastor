"""Coverage tests for vetinari/agents/* agent classes — Phase 7C"""
import unittest
from unittest.mock import MagicMock


def _make_agent(agent_type_value="BUILDER"):
    from vetinari.agents.contracts import AgentType
    from vetinari.agents.base_agent import BaseAgent

    class MinimalAgent(BaseAgent):
        def execute(self, task, context=None):
            from vetinari.agents.contracts import AgentResult
            return AgentResult(success=True, output="done")

        def verify(self, result, criteria=None):
            from vetinari.agents.contracts import VerificationResult
            return VerificationResult(passed=True, score=1.0)

        def get_system_prompt(self):
            return "You are a test agent."

        def get_capabilities(self):
            return ["test"]

    return MinimalAgent(AgentType[agent_type_value])


# ─── BaseAgent ───────────────────────────────────────────────────────────────

class TestBaseAgentCoverage(unittest.TestCase):

    def test_properties(self):
        agent = _make_agent("BUILDER")
        self.assertIsNotNone(agent.agent_type)
        self.assertIsInstance(agent.name, str)
        self.assertIsInstance(agent.description, str)
        self.assertFalse(agent.is_initialized)

    def test_initialize(self):
        agent = _make_agent("EXPLORER")
        agent.initialize({"project": "test"})
        self.assertTrue(agent.is_initialized)

    def test_execute(self):
        from vetinari.agents.contracts import AgentTask, AgentType
        agent = _make_agent("EVALUATOR")
        task  = AgentTask(task_id="t1", agent_type=AgentType.EVALUATOR,
                          description="evaluate this", prompt="check it")
        result = agent.execute(task)
        self.assertTrue(result.success)

    def test_verify(self):
        from vetinari.agents.contracts import AgentResult
        agent = _make_agent("ORACLE")
        result = AgentResult(success=True, output="answer")
        vr = agent.verify(result)
        self.assertTrue(vr.passed)

    def test_get_system_prompt(self):
        agent = _make_agent("LIBRARIAN")
        prompt = agent.get_system_prompt()
        self.assertIsInstance(prompt, str)

    def test_get_capabilities(self):
        agent = _make_agent("RESEARCHER")
        caps = agent.get_capabilities()
        self.assertIsInstance(caps, list)


# ─── PlannerAgent ─────────────────────────────────────────────────────────────

class TestPlannerAgent(unittest.TestCase):

    def setUp(self):
        from vetinari.agents.planner_agent import PlannerAgent
        self.agent = PlannerAgent()

    def test_properties(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.PLANNER)

    def test_get_system_prompt(self):
        prompt = self.agent.get_system_prompt()
        self.assertIn("Plan", prompt)

    def test_get_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIn("plan_generation", caps)

    def test_initialize(self):
        self.agent.initialize({"project_root": "/tmp"})
        self.assertTrue(self.agent.is_initialized)

    def test_execute_returns_result(self):
        from vetinari.agents.contracts import AgentTask, AgentType
        task   = AgentTask(task_id="t1", agent_type=AgentType.PLANNER,
                           description="Plan task", prompt="Plan: build a REST API")
        result = self.agent.execute(task)
        self.assertIsNotNone(result)


# ─── BuilderAgent ─────────────────────────────────────────────────────────────

class TestBuilderAgent(unittest.TestCase):

    def setUp(self):
        from vetinari.agents.builder_agent import BuilderAgent
        self.agent = BuilderAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.BUILDER)

    def test_get_system_prompt(self):
        prompt = self.agent.get_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 10)

    def test_get_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIsInstance(caps, list)
        self.assertGreater(len(caps), 0)


# ─── ExplorerAgent (-> ConsolidatedResearcherAgent) ──────────────────────────

class TestExplorerAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.explorer_agent import ExplorerAgent
        self.agent = ExplorerAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.CONSOLIDATED_RESEARCHER)

    def test_get_capabilities(self):
        self.assertIsInstance(self.agent.get_capabilities(), list)


# ─── EvaluatorAgent (-> QualityAgent) ────────────────────────────────────────

class TestEvaluatorAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.evaluator_agent import EvaluatorAgent
        self.agent = EvaluatorAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.QUALITY)

    def test_get_capabilities(self):
        self.assertIsInstance(self.agent.get_capabilities(), list)


# ─── ResearcherAgent (-> ConsolidatedResearcherAgent) ────────────────────────

class TestResearcherAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.researcher_agent import ResearcherAgent
        self.agent = ResearcherAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.CONSOLIDATED_RESEARCHER)

    def test_get_capabilities(self):
        self.assertIsInstance(self.agent.get_capabilities(), list)


# ─── SynthesizerAgent (-> OperationsAgent) ───────────────────────────────────

class TestSynthesizerAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.synthesizer_agent import SynthesizerAgent
        self.agent = SynthesizerAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.OPERATIONS)

    def test_get_capabilities(self):
        self.assertIsInstance(self.agent.get_capabilities(), list)


# ─── LibrarianAgent (-> ConsolidatedResearcherAgent) ─────────────────────────

class TestLibrarianAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.librarian_agent import LibrarianAgent
        self.agent = LibrarianAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.CONSOLIDATED_RESEARCHER)


# ─── OracleAgent (-> ConsolidatedOracleAgent) ────────────────────────────────

class TestOracleAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.oracle_agent import OracleAgent
        self.agent = OracleAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.CONSOLIDATED_ORACLE)


# ─── SecurityAuditorAgent (-> QualityAgent) ──────────────────────────────────

class TestSecurityAuditorAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.security_auditor_agent import SecurityAuditorAgent
        self.agent = SecurityAuditorAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.QUALITY)

    def test_get_capabilities(self):
        self.assertIsInstance(self.agent.get_capabilities(), list)


# ─── DataEngineerAgent (-> ConsolidatedResearcherAgent) ──────────────────────

class TestDataEngineerAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.data_engineer_agent import DataEngineerAgent
        self.agent = DataEngineerAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.CONSOLIDATED_RESEARCHER)


# ─── DocumentationAgent (-> OperationsAgent) ─────────────────────────────────

class TestDocumentationAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.documentation_agent import DocumentationAgent
        self.agent = DocumentationAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.OPERATIONS)


# ─── CostPlannerAgent (-> PlannerAgent) ──────────────────────────────────────

class TestCostPlannerAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.cost_planner_agent import CostPlannerAgent
        self.agent = CostPlannerAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.PLANNER)


# ─── UIPlannerAgent (-> ConsolidatedResearcherAgent) ─────────────────────────

class TestUIPlannerAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.ui_planner_agent import UIPlannerAgent
        self.agent = UIPlannerAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.CONSOLIDATED_RESEARCHER)


# ─── TestAutomationAgent (-> QualityAgent) ───────────────────────────────────

class TestTestAutomationAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.test_automation_agent import TestAutomationAgent
        self.agent = TestAutomationAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.QUALITY)


# ─── ExperimentationManagerAgent (-> OperationsAgent) ────────────────────────

class TestExperimentationManagerAgent(unittest.TestCase):
    def setUp(self):
        from vetinari.agents.experimentation_manager_agent import ExperimentationManagerAgent
        self.agent = ExperimentationManagerAgent()

    def test_agent_type(self):
        from vetinari.agents.contracts import AgentType
        self.assertEqual(self.agent.agent_type, AgentType.OPERATIONS)


if __name__ == "__main__":
    unittest.main()
