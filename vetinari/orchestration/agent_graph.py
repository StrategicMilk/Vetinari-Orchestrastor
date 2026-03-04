"""
Vetinari AgentGraph - Orchestration Engine

This module provides the core orchestration engine that coordinates all 15 agents
in the Vetinari hierarchical multi-agent system. It manages plan execution,
task assignment, dependency resolution, and result aggregation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from vetinari.agents.contracts import (
    AgentType,
    AgentResult,
    AgentTask,
    Plan,
    Task,
    TaskStatus,
    get_agent_spec,
    AGENT_REGISTRY
)

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Strategy for task execution."""
    SEQUENTIAL = "sequential"  # Execute tasks one at a time
    PARALLEL = "parallel"      # Execute independent tasks in parallel
    ADAPTIVE = "adaptive"      # Adapt strategy based on dependencies


@dataclass
class TaskNode:
    """A node in the execution DAG."""
    task: Task
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[AgentResult] = None
    dependencies: Set[str] = field(default_factory=set)  # Task IDs this depends on
    dependents: Set[str] = field(default_factory=set)    # Tasks that depend on this
    retries: int = 0
    max_retries: int = 3


@dataclass
class ExecutionPlan:
    """An execution plan with task DAG and scheduling."""
    plan_id: str
    original_plan: Plan
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class AgentGraph:
    """
    Hierarchical multi-agent orchestration engine.
    
    Coordinates 15 specialized agents through a Plan DAG, managing:
    - Task decomposition and assignment
    - Dependency resolution
    - Parallel and sequential execution
    - Retry and failure handling
    - Result aggregation and synthesis
    """
    
    def __init__(self, strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE, max_workers: int = 5):
        """Initialize the AgentGraph.
        
        Args:
            strategy: Execution strategy (sequential, parallel, adaptive)
            max_workers: Maximum concurrent tasks for parallel execution
        """
        self._strategy = strategy
        self._max_workers = max_workers
        self._agents: Dict[AgentType, Any] = {}
        self._execution_plans: Dict[str, ExecutionPlan] = {}
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize all agents."""
        if self._initialized:
            return
        
        # Import agent getter functions
        from vetinari.agents import (
            get_planner_agent,
            get_explorer_agent,
            get_oracle_agent,
            get_librarian_agent,
            get_researcher_agent,
            get_evaluator_agent,
            get_synthesizer_agent,
            get_builder_agent,
            get_ui_planner_agent,
            get_security_auditor_agent,
            get_data_engineer_agent,
            get_documentation_agent,
            get_cost_planner_agent,
            get_test_automation_agent,
            get_experimentation_manager_agent
        )
        
        # Get singleton instances of all agents
        self._agents[AgentType.PLANNER] = get_planner_agent()
        self._agents[AgentType.EXPLORER] = get_explorer_agent()
        self._agents[AgentType.ORACLE] = get_oracle_agent()
        self._agents[AgentType.LIBRARIAN] = get_librarian_agent()
        self._agents[AgentType.RESEARCHER] = get_researcher_agent()
        self._agents[AgentType.EVALUATOR] = get_evaluator_agent()
        self._agents[AgentType.SYNTHESIZER] = get_synthesizer_agent()
        self._agents[AgentType.BUILDER] = get_builder_agent()
        self._agents[AgentType.UI_PLANNER] = get_ui_planner_agent()
        self._agents[AgentType.SECURITY_AUDITOR] = get_security_auditor_agent()
        self._agents[AgentType.DATA_ENGINEER] = get_data_engineer_agent()
        self._agents[AgentType.DOCUMENTATION_AGENT] = get_documentation_agent()
        self._agents[AgentType.COST_PLANNER] = get_cost_planner_agent()
        self._agents[AgentType.TEST_AUTOMATION] = get_test_automation_agent()
        self._agents[AgentType.EXPERIMENTATION_MANAGER] = get_experimentation_manager_agent()
        
        # Initialize each agent
        for agent_type, agent in self._agents.items():
            agent.initialize({})
            logger.info(f"Initialized {agent.name}")
        
        self._initialized = True
    
    def create_execution_plan(self, plan: Plan) -> ExecutionPlan:
        """Create an execution plan from a Plan DAG.
        
        Args:
            plan: The plan to create execution plan from
            
        Returns:
            ExecutionPlan with task nodes and execution order
        """
        exec_plan = ExecutionPlan(plan_id=plan.plan_id, original_plan=plan)
        
        # Create task nodes from plan tasks
        for task in plan.tasks:
            node = TaskNode(
                task=task,
                dependencies=set(task.dependencies),
                status=TaskStatus.PENDING
            )
            exec_plan.nodes[task.id] = node
        
        # Build dependents map
        for task_id, node in exec_plan.nodes.items():
            for dep_id in node.dependencies:
                if dep_id in exec_plan.nodes:
                    exec_plan.nodes[dep_id].dependents.add(task_id)
        
        # Determine execution order
        exec_plan.execution_order = self._topological_sort(exec_plan.nodes)
        
        self._execution_plans[plan.plan_id] = exec_plan
        return exec_plan
    
    def _topological_sort(self, nodes: Dict[str, TaskNode]) -> List[str]:
        """Perform topological sort on tasks to determine execution order.
        
        Args:
            nodes: Dictionary of task nodes
            
        Returns:
            List of task IDs in execution order
        """
        # Calculate in-degree for each node
        in_degree = {task_id: len(node.dependencies) for task_id, node in nodes.items()}
        
        # Find all nodes with no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Process nodes with no dependencies
            current = queue.pop(0)
            result.append(current)
            
            # Reduce in-degree for dependents
            for dependent_id in nodes[current].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        # Check for cycles
        if len(result) != len(nodes):
            raise ValueError("Circular dependency detected in task graph")
        
        return result
    
    def execute_plan(self, plan: Plan) -> Dict[str, AgentResult]:
        """Execute a complete plan using the agent graph.
        
        Args:
            plan: The plan to execute
            
        Returns:
            Dictionary mapping task IDs to AgentResults
        """
        # Create execution plan
        exec_plan = self.create_execution_plan(plan)
        exec_plan.status = TaskStatus.RUNNING
        exec_plan.started_at = datetime.now().isoformat()
        
        results = {}
        
        try:
            # Execute tasks in order determined by dependencies
            for task_id in exec_plan.execution_order:
                node = exec_plan.nodes[task_id]
                result = self._execute_task_node(node)
                results[task_id] = result
                
                # Update node status
                if result.success:
                    node.status = TaskStatus.COMPLETED
                else:
                    node.status = TaskStatus.FAILED
                    logger.error(f"Task {task_id} failed: {result.errors}")
            
            exec_plan.status = TaskStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Plan execution failed: {str(e)}")
            exec_plan.status = TaskStatus.FAILED
            raise
        
        finally:
            exec_plan.completed_at = datetime.now().isoformat()
        
        return results
    
    def _execute_task_node(self, node: TaskNode) -> AgentResult:
        """Execute a single task node.
        
        Args:
            node: The task node to execute
            
        Returns:
            AgentResult from task execution
        """
        task = node.task
        agent_type = task.assigned_agent
        
        # Get the agent for this task
        if agent_type not in self._agents:
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Unknown agent type: {agent_type}"]
            )
        
        agent = self._agents[agent_type]
        
        # Create AgentTask from Task
        agent_task = AgentTask.from_task(task, task.description)
        
        # Execute with retries
        for attempt in range(node.max_retries + 1):
            try:
                logger.info(f"Executing task {task.id} with {agent.name} (attempt {attempt + 1})")
                
                # Execute the task
                result = agent.execute(agent_task)
                
                # Verify the result
                verification = agent.verify(result.output)
                
                if result.success and verification.passed:
                    logger.info(f"Task {task.id} completed successfully")
                    return result
                
                # Verification failed, retry if we have retries left
                if attempt < node.max_retries:
                    logger.warning(f"Task {task.id} verification failed, retrying...")
                    node.retries += 1
                    continue
                else:
                    return AgentResult(
                        success=False,
                        output=result.output,
                        errors=[f"Verification failed: {verification.issues}"]
                    )
                
            except Exception as e:
                logger.error(f"Task {task.id} execution failed: {str(e)}")
                if attempt < node.max_retries:
                    continue
                else:
                    return AgentResult(
                        success=False,
                        output=None,
                        errors=[str(e)]
                    )
        
        return AgentResult(
            success=False,
            output=None,
            errors=["Task execution failed after all retries"]
        )
    
    async def execute_plan_async(self, plan: Plan) -> Dict[str, AgentResult]:
        """Execute a plan asynchronously.
        
        Args:
            plan: The plan to execute
            
        Returns:
            Dictionary mapping task IDs to AgentResults
        """
        # For now, fall back to synchronous execution
        # In production, this would use actual async agents
        return self.execute_plan(plan)
    
    def get_execution_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get an execution plan by ID.
        
        Args:
            plan_id: The plan ID
            
        Returns:
            ExecutionPlan if found, None otherwise
        """
        return self._execution_plans.get(plan_id)
    
    def get_agent(self, agent_type: AgentType) -> Any:
        """Get an agent by type.
        
        Args:
            agent_type: The agent type
            
        Returns:
            The agent instance
        """
        return self._agents.get(agent_type)
    
    def __repr__(self) -> str:
        return f"<AgentGraph(strategy={self._strategy.value}, agents={len(self._agents)})>"


# Singleton instance
_agent_graph: Optional[AgentGraph] = None


def get_agent_graph(strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE) -> AgentGraph:
    """Get the singleton AgentGraph instance.
    
    Args:
        strategy: Execution strategy
        
    Returns:
        The AgentGraph instance
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = AgentGraph(strategy=strategy)
        _agent_graph.initialize()
    return _agent_graph
