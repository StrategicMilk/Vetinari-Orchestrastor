"""
Unit tests for Vetinari AgentGraph orchestration engine.
"""

import unittest
from datetime import datetime

from vetinari.agents.contracts import (
    AgentType,
    Plan,
    Task,
    TaskStatus
)
from vetinari.orchestration import (
    AgentGraph,
    ExecutionPlan,
    ExecutionStrategy,
    TaskNode,
    get_agent_graph
)


class TestExecutionPlan(unittest.TestCase):
    """Test the ExecutionPlan class."""
    
    def test_execution_plan_creation(self):
        """Test creating an ExecutionPlan."""
        plan = Plan.create_new("Test goal")
        
        exec_plan = ExecutionPlan(plan_id=plan.plan_id, original_plan=plan)
        
        self.assertEqual(exec_plan.plan_id, plan.plan_id)
        self.assertEqual(exec_plan.status, TaskStatus.PENDING)
        self.assertEqual(len(exec_plan.nodes), 0)
    
    def test_execution_plan_with_tasks(self):
        """Test ExecutionPlan with tasks."""
        plan = Plan.create_new("Test goal")
        plan.tasks.append(Task(
            id="t1",
            description="Task 1",
            assigned_agent=AgentType.EXPLORER
        ))
        
        exec_plan = ExecutionPlan(plan_id=plan.plan_id, original_plan=plan)
        
        # Add task node
        node = TaskNode(task=plan.tasks[0])
        exec_plan.nodes["t1"] = node
        
        self.assertEqual(len(exec_plan.nodes), 1)


class TestTaskNode(unittest.TestCase):
    """Test the TaskNode class."""
    
    def test_task_node_creation(self):
        """Test creating a TaskNode."""
        task = Task(
            id="t1",
            description="Test task",
            assigned_agent=AgentType.EXPLORER
        )
        
        node = TaskNode(task=task)
        
        self.assertEqual(node.task.id, "t1")
        self.assertEqual(node.status, TaskStatus.PENDING)
        self.assertEqual(len(node.dependencies), 0)
    
    def test_task_node_with_dependencies(self):
        """Test TaskNode with dependencies."""
        task = Task(
            id="t2",
            description="Task 2",
            assigned_agent=AgentType.BUILDER,
            dependencies=["t1"]
        )
        
        node = TaskNode(task=task, dependencies={"t1"})
        
        self.assertIn("t1", node.dependencies)
    
    def test_task_node_retries(self):
        """Test task node retry tracking."""
        task = Task(id="t1", description="Test", assigned_agent=AgentType.EXPLORER)
        node = TaskNode(task=task, max_retries=3)
        
        self.assertEqual(node.retries, 0)
        self.assertEqual(node.max_retries, 3)


class TestAgentGraph(unittest.TestCase):
    """Test the AgentGraph orchestration engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = AgentGraph(strategy=ExecutionStrategy.SEQUENTIAL)
    
    def test_graph_initialization(self):
        """Test AgentGraph initialization."""
        self.assertFalse(self.graph._initialized)
        self.assertEqual(len(self.graph._agents), 0)
    
    def test_graph_initialize(self):
        """Test initializing the graph with all agents."""
        self.graph.initialize()
        
        self.assertTrue(self.graph._initialized)
        self.assertEqual(len(self.graph._agents), 15)
    
    def test_graph_all_agent_types(self):
        """Test that graph has all 15 agent types."""
        self.graph.initialize()
        
        agent_types = [
            AgentType.PLANNER,
            AgentType.EXPLORER,
            AgentType.LIBRARIAN,
            AgentType.ORACLE,
            AgentType.RESEARCHER,
            AgentType.EVALUATOR,
            AgentType.SYNTHESIZER,
            AgentType.BUILDER,
            AgentType.UI_PLANNER,
            AgentType.SECURITY_AUDITOR,
            AgentType.DATA_ENGINEER,
            AgentType.DOCUMENTATION_AGENT,
            AgentType.COST_PLANNER,
            AgentType.TEST_AUTOMATION,
            AgentType.EXPERIMENTATION_MANAGER
        ]
        
        for agent_type in agent_types:
            self.assertIn(agent_type, self.graph._agents)
    
    def test_get_agent(self):
        """Test retrieving an agent from the graph."""
        self.graph.initialize()
        
        agent = self.graph.get_agent(AgentType.EXPLORER)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.agent_type, AgentType.EXPLORER)
    
    def test_topological_sort_simple(self):
        """Test topological sorting with simple dependencies."""
        self.graph.initialize()
        
        # Create simple dependency graph: t1 -> t2 -> t3
        nodes = {
            "t1": TaskNode(
                task=Task(id="t1", description="Task 1", assigned_agent=AgentType.EXPLORER),
                dependencies=set()
            ),
            "t2": TaskNode(
                task=Task(id="t2", description="Task 2", assigned_agent=AgentType.BUILDER),
                dependencies={"t1"}
            ),
            "t3": TaskNode(
                task=Task(id="t3", description="Task 3", assigned_agent=AgentType.EVALUATOR),
                dependencies={"t2"}
            )
        }
        
        # Update reverse dependencies
        nodes["t1"].dependents.add("t2")
        nodes["t2"].dependents.add("t3")
        
        order = self.graph._topological_sort(nodes)
        
        self.assertEqual(order, ["t1", "t2", "t3"])
    
    def test_topological_sort_parallel(self):
        """Test topological sorting with parallel independent tasks."""
        self.graph.initialize()
        
        # Create parallel tasks
        nodes = {
            "t1": TaskNode(
                task=Task(id="t1", description="Task 1", assigned_agent=AgentType.EXPLORER),
                dependencies=set()
            ),
            "t2": TaskNode(
                task=Task(id="t2", description="Task 2", assigned_agent=AgentType.LIBRARIAN),
                dependencies=set()
            ),
            "t3": TaskNode(
                task=Task(id="t3", description="Task 3", assigned_agent=AgentType.BUILDER),
                dependencies={"t1", "t2"}
            )
        }
        
        # Update reverse dependencies
        nodes["t1"].dependents.add("t3")
        nodes["t2"].dependents.add("t3")
        
        order = self.graph._topological_sort(nodes)
        
        self.assertEqual(len(order), 3)
        # t1 and t2 should come before t3
        self.assertLess(order.index("t1"), order.index("t3"))
        self.assertLess(order.index("t2"), order.index("t3"))
    
    def test_create_execution_plan(self):
        """Test creating an execution plan from a Plan."""
        self.graph.initialize()
        
        plan = Plan.create_new("Test goal")
        plan.tasks = [
            Task(id="t1", description="Task 1", assigned_agent=AgentType.EXPLORER),
            Task(id="t2", description="Task 2", assigned_agent=AgentType.BUILDER, dependencies=["t1"])
        ]
        
        exec_plan = self.graph.create_execution_plan(plan)
        
        self.assertEqual(exec_plan.plan_id, plan.plan_id)
        self.assertEqual(len(exec_plan.nodes), 2)
        self.assertEqual(len(exec_plan.execution_order), 2)
    
    def test_execution_plan_storage(self):
        """Test that execution plans are stored in the graph."""
        self.graph.initialize()
        
        plan = Plan.create_new("Test goal")
        exec_plan = self.graph.create_execution_plan(plan)
        
        retrieved = self.graph.get_execution_plan(plan.plan_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.plan_id, exec_plan.plan_id)
    
    def test_graph_singleton(self):
        """Test that get_agent_graph returns singleton."""
        graph1 = get_agent_graph()
        graph2 = get_agent_graph()
        
        self.assertIs(graph1, graph2)
    
    def test_graph_repr(self):
        """Test graph string representation."""
        repr_str = repr(self.graph)
        
        self.assertIn("AgentGraph", repr_str)
        self.assertIn("sequential", repr_str)


class TestExecutionStrategy(unittest.TestCase):
    """Test execution strategies."""
    
    def test_sequential_strategy(self):
        """Test sequential execution strategy."""
        graph = AgentGraph(strategy=ExecutionStrategy.SEQUENTIAL)
        self.assertEqual(graph._strategy, ExecutionStrategy.SEQUENTIAL)
    
    def test_parallel_strategy(self):
        """Test parallel execution strategy."""
        graph = AgentGraph(strategy=ExecutionStrategy.PARALLEL)
        self.assertEqual(graph._strategy, ExecutionStrategy.PARALLEL)
    
    def test_adaptive_strategy(self):
        """Test adaptive execution strategy."""
        graph = AgentGraph(strategy=ExecutionStrategy.ADAPTIVE)
        self.assertEqual(graph._strategy, ExecutionStrategy.ADAPTIVE)


class TestCircularDependencyDetection(unittest.TestCase):
    """Test circular dependency detection."""
    
    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        graph = AgentGraph()
        graph.initialize()
        
        # Create circular dependency: t1 -> t2 -> t1
        nodes = {
            "t1": TaskNode(
                task=Task(id="t1", description="Task 1", assigned_agent=AgentType.EXPLORER),
                dependencies={"t2"}
            ),
            "t2": TaskNode(
                task=Task(id="t2", description="Task 2", assigned_agent=AgentType.BUILDER),
                dependencies={"t1"}
            )
        }
        
        nodes["t1"].dependents.add("t2")
        nodes["t2"].dependents.add("t1")
        
        with self.assertRaises(ValueError):
            graph._topological_sort(nodes)


if __name__ == "__main__":
    unittest.main()
