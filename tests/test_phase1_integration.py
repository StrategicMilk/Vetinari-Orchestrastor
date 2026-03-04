"""
Phase 1 Tests - End-to-End Agent Orchestration

These tests demonstrate the completion of Phase 1 requirements:
- Librarian and Researcher integrated into orchestration
- End-to-end plans with 3+ agents executing successfully
- Dependency validation and result aggregation
"""

import unittest
from typing import Dict

from vetinari.agents.contracts import (
    AgentType,
    Plan,
    Task,
    TaskStatus,
    AgentResult
)
from vetinari.orchestration import (
    get_agent_graph,
    ExecutionStrategy
)


class Phase1EndToEndTests(unittest.TestCase):
    """End-to-end tests for Phase 1."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = get_agent_graph(strategy=ExecutionStrategy.ADAPTIVE)
        self.graph.initialize()
    
    def test_explorer_librarian_pipeline(self):
        """Test: Explorer -> Librarian pipeline (2 agents)."""
        plan = Plan.create_new("Explore web framework patterns and look up documentation")
        
        # Task 1: Explorer searches for patterns
        plan.tasks.append(Task(
            id="t1",
            description="Search for web framework patterns",
            inputs=["goal"],
            outputs=["patterns", "references"],
            dependencies=[],
            assigned_agent=AgentType.EXPLORER,
            depth=0
        ))
        
        # Task 2: Librarian looks up documentation based on findings
        plan.tasks.append(Task(
            id="t2",
            description="Look up documentation for found frameworks",
            inputs=["patterns"],
            outputs=["documentation", "fit_assessment"],
            dependencies=["t1"],
            assigned_agent=AgentType.LIBRARIAN,
            depth=1
        ))
        
        # Execute plan
        results = self.graph.execute_plan(plan)
        
        # Verify
        self.assertEqual(len(results), 2)
        self.assertIn("t1", results)
        self.assertIn("t2", results)
        
        # Check task order (t1 should complete before t2)
        exec_plan = self.graph.get_execution_plan(plan.plan_id)
        self.assertEqual(exec_plan.execution_order[0], "t1")
        self.assertEqual(exec_plan.execution_order[1], "t2")
    
    def test_researcher_analysis_pipeline(self):
        """Test: Explorer -> Researcher pipeline (2 agents)."""
        plan = Plan.create_new("Research technology feasibility for web application")
        
        # Task 1: Explorer finds patterns
        plan.tasks.append(Task(
            id="t1",
            description="Explore available web technologies",
            inputs=["goal"],
            outputs=["tech_options"],
            dependencies=[],
            assigned_agent=AgentType.EXPLORER,
            depth=0
        ))
        
        # Task 2: Researcher analyzes feasibility
        plan.tasks.append(Task(
            id="t2",
            description="Research feasibility of each technology option",
            inputs=["tech_options"],
            outputs=["feasibility_report", "recommendations"],
            dependencies=["t1"],
            assigned_agent=AgentType.RESEARCHER,
            depth=1
        ))
        
        # Execute plan
        results = self.graph.execute_plan(plan)
        
        # Verify
        self.assertEqual(len(results), 2)
        for task_id in ["t1", "t2"]:
            self.assertIn(task_id, results)
            self.assertTrue(results[task_id].success)
    
    def test_three_agent_pipeline(self):
        """Test: Explorer -> Librarian -> Researcher (3+ agents requirement)."""
        plan = Plan.create_new("Complete technology research pipeline")
        
        # Task 1: Explorer
        plan.tasks.append(Task(
            id="t1",
            description="Explore code patterns",
            inputs=["goal"],
            outputs=["patterns"],
            dependencies=[],
            assigned_agent=AgentType.EXPLORER,
            depth=0
        ))
        
        # Task 2: Librarian
        plan.tasks.append(Task(
            id="t2",
            description="Look up related documentation",
            inputs=["patterns"],
            outputs=["docs"],
            dependencies=["t1"],
            assigned_agent=AgentType.LIBRARIAN,
            depth=1
        ))
        
        # Task 3: Researcher
        plan.tasks.append(Task(
            id="t3",
            description="Research feasibility",
            inputs=["patterns", "docs"],
            outputs=["feasibility"],
            dependencies=["t1", "t2"],
            assigned_agent=AgentType.RESEARCHER,
            depth=2
        ))
        
        # Execute plan
        results = self.graph.execute_plan(plan)
        
        # Verify
        self.assertEqual(len(results), 3)
        
        # Verify all tasks completed
        for task_id in ["t1", "t2", "t3"]:
            self.assertIn(task_id, results)
            self.assertTrue(results[task_id].success, 
                          f"Task {task_id} failed: {results[task_id].errors}")
        
        # Verify execution order
        exec_plan = self.graph.get_execution_plan(plan.plan_id)
        t1_idx = exec_plan.execution_order.index("t1")
        t2_idx = exec_plan.execution_order.index("t2")
        t3_idx = exec_plan.execution_order.index("t3")
        
        self.assertLess(t1_idx, t2_idx)  # t1 before t2
        self.assertLess(t2_idx, t3_idx)  # t2 before t3
    
    def test_parallel_exploration_branches(self):
        """Test: Parallel independent exploration tasks."""
        plan = Plan.create_new("Parallel research on two independent topics")
        
        # Task 1: Explore topic A
        plan.tasks.append(Task(
            id="t1",
            description="Explore database technologies",
            inputs=["goal"],
            outputs=["db_options"],
            dependencies=[],
            assigned_agent=AgentType.EXPLORER,
            depth=0
        ))
        
        # Task 2: Explore topic B (independent of t1)
        plan.tasks.append(Task(
            id="t2",
            description="Explore frontend frameworks",
            inputs=["goal"],
            outputs=["frontend_options"],
            dependencies=[],
            assigned_agent=AgentType.EXPLORER,
            depth=0
        ))
        
        # Task 3: Research findings from both (depends on t1 and t2)
        plan.tasks.append(Task(
            id="t3",
            description="Research integration approach",
            inputs=["db_options", "frontend_options"],
            outputs=["integration_plan"],
            dependencies=["t1", "t2"],
            assigned_agent=AgentType.RESEARCHER,
            depth=1
        ))
        
        # Execute plan
        results = self.graph.execute_plan(plan)
        
        # Verify
        self.assertEqual(len(results), 3)
        
        # Verify all succeeded
        for task_id in results:
            self.assertTrue(results[task_id].success)
        
        # Verify execution order (t1 and t2 should both come before t3)
        exec_plan = self.graph.get_execution_plan(plan.plan_id)
        t1_idx = exec_plan.execution_order.index("t1")
        t2_idx = exec_plan.execution_order.index("t2")
        t3_idx = exec_plan.execution_order.index("t3")
        
        self.assertLess(t1_idx, t3_idx)
        self.assertLess(t2_idx, t3_idx)
    
    def test_librarian_researcher_combined(self):
        """Test: Both Librarian and Researcher in same plan."""
        plan = Plan.create_new("Research with documentation and domain analysis")
        
        # Task 1: Librarian looks up docs
        plan.tasks.append(Task(
            id="t1",
            description="Look up API documentation",
            inputs=["goal"],
            outputs=["api_docs"],
            dependencies=[],
            assigned_agent=AgentType.LIBRARIAN,
            depth=0
        ))
        
        # Task 2: Researcher analyzes market
        plan.tasks.append(Task(
            id="t2",
            description="Analyze market feasibility",
            inputs=["goal"],
            outputs=["market_analysis"],
            dependencies=[],
            assigned_agent=AgentType.RESEARCHER,
            depth=0
        ))
        
        # Task 3: Synthesize results (different agent)
        plan.tasks.append(Task(
            id="t3",
            description="Synthesize research findings",
            inputs=["api_docs", "market_analysis"],
            outputs=["synthesis"],
            dependencies=["t1", "t2"],
            assigned_agent=AgentType.ORACLE,
            depth=1
        ))
        
        # Execute plan
        results = self.graph.execute_plan(plan)
        
        # Verify all tasks completed successfully
        self.assertEqual(len(results), 3)
        for task_id in results:
            self.assertTrue(results[task_id].success)


class Phase1AgentCapabilityTests(unittest.TestCase):
    """Test agent capabilities for Phase 1."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = get_agent_graph()
        self.graph.initialize()
    
    def test_explorer_capabilities(self):
        """Test Explorer agent has correct capabilities."""
        explorer = self.graph.get_agent(AgentType.EXPLORER)
        capabilities = explorer.get_capabilities()
        
        self.assertIn("code_search", capabilities)
        self.assertIn("pattern_discovery", capabilities)
        self.assertGreater(len(capabilities), 0)
    
    def test_librarian_capabilities(self):
        """Test Librarian agent has correct capabilities."""
        librarian = self.graph.get_agent(AgentType.LIBRARIAN)
        capabilities = librarian.get_capabilities()
        
        self.assertIn("api_documentation_lookup", capabilities)
        self.assertIn("library_discovery", capabilities)
        self.assertGreater(len(capabilities), 0)
    
    def test_researcher_capabilities(self):
        """Test Researcher agent has correct capabilities."""
        researcher = self.graph.get_agent(AgentType.RESEARCHER)
        capabilities = researcher.get_capabilities()
        
        self.assertIn("domain_analysis", capabilities)
        self.assertIn("feasibility_assessment", capabilities)
        self.assertIn("competitor_analysis", capabilities)
        self.assertGreater(len(capabilities), 0)
    
    def test_all_agents_have_capabilities(self):
        """Test that all agents have at least one capability."""
        for agent_type in AgentType:
            agent = self.graph.get_agent(agent_type)
            if agent:
                capabilities = agent.get_capabilities()
                self.assertGreater(len(capabilities), 0, 
                                 f"{agent_type.value} has no capabilities")


class Phase1IntegrationTests(unittest.TestCase):
    """Integration tests for Phase 1 components."""
    
    def test_plan_creation_and_execution(self):
        """Test creating and executing a plan."""
        graph = get_agent_graph()
        graph.initialize()
        
        plan = Plan.create_new("Test plan")
        plan.tasks = [
            Task(id="t1", description="Task 1", assigned_agent=AgentType.EXPLORER),
        ]
        
        exec_plan = graph.create_execution_plan(plan)
        self.assertIsNotNone(exec_plan)
        self.assertEqual(exec_plan.plan_id, plan.plan_id)
    
    def test_agent_initialization_and_metadata(self):
        """Test agent initialization and metadata retrieval."""
        graph = get_agent_graph()
        graph.initialize()
        
        for agent_type in [AgentType.EXPLORER, AgentType.LIBRARIAN, AgentType.RESEARCHER]:
            agent = graph.get_agent(agent_type)
            self.assertIsNotNone(agent)
            self.assertTrue(agent.is_initialized)
            
            metadata = agent.get_metadata()
            self.assertIn("agent_type", metadata)
            self.assertIn("name", metadata)
            self.assertIn("capabilities", metadata)


class Phase1AcceptanceCriteria(unittest.TestCase):
    """Acceptance criteria tests for Phase 1."""
    
    def test_acceptance_criterion_3_plus_agents(self):
        """Acceptance Criterion: End-to-end plan with 3+ agents executes successfully."""
        graph = get_agent_graph(strategy=ExecutionStrategy.ADAPTIVE)
        graph.initialize()
        
        plan = Plan.create_new("Phase 1 acceptance test with 3+ agents")
        
        # Create a 3-agent plan with dependencies
        plan.tasks = [
            Task(id="t1", description="Explore", assigned_agent=AgentType.EXPLORER),
            Task(id="t2", description="Research", assigned_agent=AgentType.RESEARCHER, dependencies=["t1"]),
            Task(id="t3", description="Consult", assigned_agent=AgentType.ORACLE, dependencies=["t2"]),
        ]
        
        # Execute the plan
        results = graph.execute_plan(plan)
        
        # Verify acceptance criteria
        self.assertEqual(len(results), 3, "Plan must have 3 or more agents")
        
        # All tasks must execute successfully
        for task_id, result in results.items():
            self.assertTrue(result.success, 
                          f"All agents must execute successfully. {task_id} failed: {result.errors}")
        
        # Execution order must respect dependencies
        exec_plan = graph.get_execution_plan(plan.plan_id)
        t1_idx = exec_plan.execution_order.index("t1")
        t2_idx = exec_plan.execution_order.index("t2")
        t3_idx = exec_plan.execution_order.index("t3")
        
        self.assertLess(t1_idx, t2_idx)
        self.assertLess(t2_idx, t3_idx)


if __name__ == "__main__":
    unittest.main()
