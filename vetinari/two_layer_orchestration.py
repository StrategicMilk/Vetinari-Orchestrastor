"""
Two-Layer Orchestration System for Vetinari

Layer 1: Graph-Based Planning (LangGraph-inspired)
- Plan decomposition into DAG of tasks
- Multi-candidate plan generation
- Plan selection and optimization

Layer 2: Durable Execution Engine (Temporal-inspired)  
- Stateful task execution with checkpointing
- Retry policies and error handling
- Event-driven progress tracking
- Crash recovery and replay

This provides production-grade orchestration with reliability.
"""

import os
import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


# ============================================================
# Layer 1: Graph-Based Planning
# ============================================================

class TaskStatus(Enum):
    """Status of a task in the execution graph."""
    PENDING = "pending"
    BLOCKED = "blocked"  # Waiting for dependencies
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlanStatus(Enum):
    """Status of a plan."""
    DRAFT = "draft"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskNode:
    """A single task node in the execution graph."""
    id: str
    description: str
    task_type: str = "general"
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    depended_by: List[str] = field(default_factory=list)
    
    # Execution
    status: TaskStatus = TaskStatus.PENDING
    assigned_model: str = ""
    
    # Results
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Checkpoint
    checkpoint_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type,
            "depends_on": self.depends_on,
            "depended_by": self.depended_by,
            "status": self.status.value,
            "assigned_model": self.assigned_model,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "checkpoint_id": self.checkpoint_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TaskNode':
        node = cls(
            id=data["id"],
            description=data["description"],
            task_type=data.get("task_type", "general"),
            depends_on=data.get("depends_on", []),
            depended_by=data.get("depended_by", []),
            status=TaskStatus(data.get("status", "pending")),
            assigned_model=data.get("assigned_model", ""),
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            error=data.get("error", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            checkpoint_id=data.get("checkpoint_id", ""),
        )
        return node


@dataclass
class ExecutionGraph:
    """
    Directed Acyclic Graph (DAG) of tasks for execution.
    
    Supports:
    - Parallel execution of independent tasks
    - Dependency resolution
    - Execution ordering
    - Checkpoint management
    """
    
    plan_id: str
    goal: str
    
    # Nodes in the graph
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    
    # Graph metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Execution state
    status: PlanStatus = PlanStatus.DRAFT
    current_layer: int = 0
    completed_count: int = 0
    failed_count: int = 0
    
    def add_task(self, 
                task_id: str, 
                description: str,
                task_type: str = "general",
                depends_on: List[str] = None,
                input_data: Dict[str, Any] = None) -> TaskNode:
        """Add a task to the graph."""
        # Create node
        node = TaskNode(
            id=task_id,
            description=description,
            task_type=task_type,
            depends_on=depends_on or [],
            input_data=input_data or {}
        )
        
        # Update dependency links
        for dep_id in node.depends_on:
            if dep_id in self.nodes:
                self.nodes[dep_id].depended_by.append(task_id)
        
        self.nodes[task_id] = node
        self.updated_at = datetime.now().isoformat()
        
        return node
    
    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready = []
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_met = all(
                self.nodes.get(dep_id, TaskNode(id=dep_id, description="")).status == TaskStatus.COMPLETED
                for dep_id in node.depends_on
            )
            
            if deps_met:
                ready.append(node)
        
        return ready
    
    def get_next_layer(self) -> List[List["TaskNode"]]:
        """
        Get the full execution schedule as layers of tasks that can run in parallel.

        Uses a simulation pass: tasks are "virtually completed" after being placed in
        a layer so that subsequent layers correctly detect their dependencies as satisfied.

        Returns list of layers (each layer is a list of tasks that can run in parallel).
        """
        layers = []
        # Work on copies of status so we don't mutate real node state
        simulated_completed: set = {
            nid for nid, n in self.nodes.items()
            if n.status == TaskStatus.COMPLETED
        }
        remaining = {
            nid: n for nid, n in self.nodes.items()
            if n.status in (TaskStatus.PENDING, TaskStatus.BLOCKED)
        }

        while remaining:
            current_layer = []
            to_remove = []

            for task_id, node in remaining.items():
                # A task is ready when ALL its dependencies are in the simulated-completed set
                deps_met = all(dep_id in simulated_completed for dep_id in node.depends_on)

                if deps_met:
                    current_layer.append(node)
                    to_remove.append(task_id)

            if not current_layer:
                # No progress possible -- circular dependency or all remaining tasks blocked
                break

            layers.append(current_layer)

            for task_id in to_remove:
                del remaining[task_id]
                simulated_completed.add(task_id)

        return layers
    
    def get_execution_order(self) -> List[List[TaskNode]]:
        """Get full execution order as layers."""
        return self.get_next_layer()
    
    def can_retry(self, task_id: str) -> bool:
        """Check if a failed task can be retried."""
        if task_id not in self.nodes:
            return False
        node = self.nodes[task_id]
        return node.retry_count < node.max_retries
    
    def get_blocked_tasks(self) -> List[TaskNode]:
        """Get tasks that are blocked waiting for dependencies."""
        blocked = []
        for node in self.nodes.values():
            if node.status == TaskStatus.BLOCKED:
                blocked.append(node)
        return blocked
    
    def get_failed_tasks(self) -> List[TaskNode]:
        """Get all failed tasks."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]
    
    def get_completed_tasks(self) -> List[TaskNode]:
        """Get all completed tasks."""
        return [n for n in self.nodes.values() if n.status == TaskStatus.COMPLETED]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "current_layer": self.current_layer,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
        }
    
    def to_dag_json(self) -> str:
        """Export graph in DAG format for visualization."""
        return json.dumps(self.to_dict(), indent=2)


class PlanGenerator:
    """
    Generates execution plans from goals.
    
    Features:
    - Multi-candidate plan generation
    - Plan scoring and selection
    - Constraint handling
    """
    
    def __init__(self, model_router=None):
        self.model_router = model_router
    
    def generate_plan(self, 
                     goal: str, 
                     constraints: Dict[str, Any] = None,
                     max_depth: int = 10) -> ExecutionGraph:
        """
        Generate an execution graph from a goal.
        
        Args:
            goal: The goal to achieve
            constraints: Any constraints (budget, time, etc.)
            max_depth: Maximum depth of task decomposition
            
        Returns:
            ExecutionGraph with decomposed tasks
        """
        constraints = constraints or {}
        plan_id = f"plan-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
        
        # Create graph
        graph = ExecutionGraph(plan_id=plan_id, goal=goal)
        
        # Decompose goal into tasks
        tasks = self._decompose_goal(goal, max_depth)
        
        # Add tasks to graph
        for task_spec in tasks:
            graph.add_task(
                task_id=task_spec["id"],
                description=task_spec["description"],
                task_type=task_spec.get("type", "general"),
                depends_on=task_spec.get("depends_on", []),
                input_data=task_spec.get("input", {})
            )
        
        # Validate graph
        if self._has_circular_dependency(graph):
            logger.warning(f"Plan {plan_id} has circular dependencies")
        
        graph.status = PlanStatus.DRAFT
        return graph
    
    def _decompose_goal(self, goal: str, max_depth: int) -> List[Dict]:
        """
        Decompose a goal into tasks using the assembly-line pattern.

        Assembly-line stages:
        1. INPUT ANALYSIS  -- classify request, assess complexity
        2. PLAN GENERATION -- high-level workflow
        3. TASK DECOMP     -- break plan into atomic tasks
        4. MODEL ASSIGNMENT-- assign model to each task
        5. PARALLEL EXEC   -- execute assigned tasks (DAG)
        6. OUTPUT REVIEW   -- verify outputs for consistency
        7. FINAL ASSEMBLY  -- combine outputs

        Uses the PlannerAgent (LLM-powered) when available, falls back to
        keyword-based heuristics.
        """
        # Try to use the PlannerAgent for intelligent decomposition
        try:
            from vetinari.agents.planner_agent import get_planner_agent
            from vetinari.agents.contracts import AgentTask, AgentType

            planner = get_planner_agent()
            task = AgentTask(
                task_id="decomp-0",
                agent_type=AgentType.PLANNER,
                description=goal,
                prompt=goal,
                context={"max_depth": max_depth},
            )
            result = planner.execute(task)
            if result.success and result.output and result.output.get("tasks"):
                # Convert Plan.tasks to internal format
                return [
                    {
                        "id": t.get("id", f"t{i+1}"),
                        "description": t.get("description", "Task"),
                        "type": t.get("assigned_agent", "general").lower() if isinstance(t.get("assigned_agent"), str) else "general",
                        "depends_on": t.get("dependencies", []),
                        "input": {"goal": goal, "inputs": t.get("inputs", [])},
                    }
                    for i, t in enumerate(result.output["tasks"])
                ]
        except Exception as e:
            logger.warning(f"PlannerAgent decomposition failed: {e}, using keyword fallback")

        # Keyword-based fallback decomposition
        tasks = []
        counter = [1]

        def next_id(p="t"):
            tid = f"{p}{counter[0]}"
            counter[0] += 1
            return tid

        goal_lower = goal.lower()
        is_code = any(k in goal_lower for k in ["code", "implement", "build", "create", "program", "app", "web", "software"])
        is_research = any(k in goal_lower for k in ["research", "analyze", "investigate", "study", "review"])
        is_docs = any(k in goal_lower for k in ["document", "readme", "explain", "write", "report"])

        # Stage 1: Analysis
        t1 = next_id()
        tasks.append({"id": t1, "description": "Analyze requirements and create specification",
                      "type": "analysis", "depends_on": [], "input": {"goal": goal}})

        # Stage 2: Implementation
        if is_code:
            t2 = next_id()
            tasks.append({"id": t2, "description": "Set up project structure",
                          "type": "implementation", "depends_on": [t1], "input": {}})
            t3 = next_id()
            tasks.append({"id": t3, "description": "Implement core functionality",
                          "type": "implementation", "depends_on": [t2], "input": {}})
            t4 = next_id()
            tasks.append({"id": t4, "description": "Write and run tests",
                          "type": "testing", "depends_on": [t3], "input": {}})
            t5 = next_id()
            tasks.append({"id": t5, "description": "Verify and validate output",
                          "type": "verification", "depends_on": [t4], "input": {}})
        elif is_research:
            t2 = next_id()
            tasks.append({"id": t2, "description": "Gather information and sources",
                          "type": "research", "depends_on": [t1], "input": {}})
            t3 = next_id()
            tasks.append({"id": t3, "description": "Analyze and synthesize findings",
                          "type": "analysis", "depends_on": [t2], "input": {}})
        else:
            t2 = next_id()
            tasks.append({"id": t2, "description": "Execute primary task",
                          "type": "implementation", "depends_on": [t1], "input": {}})

        # Stage 3: Review and Assembly
        prev = tasks[-1]["id"]
        trev = next_id()
        tasks.append({"id": trev, "description": "Review output quality and consistency",
                      "type": "verification", "depends_on": [prev], "input": {}})

        if is_docs or is_code:
            tdoc = next_id()
            tasks.append({"id": tdoc, "description": "Create documentation and final summary",
                          "type": "documentation", "depends_on": [trev], "input": {}})

        return tasks
    
    def _has_circular_dependency(self, graph: ExecutionGraph) -> bool:
        """Check for circular dependencies in the graph."""
        visited = set()
        rec_stack = set()
        
        def visit(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = graph.nodes.get(node_id)
            if node:
                for dep in node.depends_on:
                    if visit(dep):
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in graph.nodes:
            if visit(node_id):
                return True
        
        return False


# ============================================================
# Layer 2: Durable Execution Engine
# ============================================================

@dataclass
class ExecutionEvent:
    """An event in the execution history."""
    event_id: str
    event_type: str  # task_started, task_completed, task_failed, etc.
    task_id: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """A checkpoint for durable execution."""
    checkpoint_id: str
    plan_id: str
    created_at: str
    graph_state: Dict[str, Any]
    completed_tasks: List[str]
    running_tasks: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DurableExecutionEngine:
    """
    Durable execution engine inspired by Temporal.
    
    Features:
    - State persistence with checkpoints
    - Retry policies
    - Event sourcing
    - Crash recovery
    - Deterministic replay
    """
    
    def __init__(self, 
                 checkpoint_dir: str = None,
                 max_concurrent: int = 4,
                 default_timeout: float = 300.0):
        """
        Initialize the durable execution engine.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_concurrent: Maximum concurrent task executions
            default_timeout: Default task timeout in seconds
        """
        self.checkpoint_dir = Path(checkpoint_dir or "./vetinari_checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        
        # Active executions
        self._active_executions: Dict[str, ExecutionGraph] = {}
        self._execution_lock = threading.Lock()
        
        # Event history
        self._event_history: List[ExecutionEvent] = []
        
        # Task handlers
        self._task_handlers: Dict[str, Callable] = {}
        
        # Callbacks
        self._on_task_start: Optional[Callable] = None
        self._on_task_complete: Optional[Callable] = None
        self._on_task_fail: Optional[Callable] = None
        
        logger.info(f"DurableExecutionEngine initialized (checkpoint_dir={self.checkpoint_dir})")
    
    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type."""
        self._task_handlers[task_type] = handler
        logger.debug(f"Registered handler for task type: {task_type}")
    
    def set_callbacks(self, 
                     on_task_start: Callable = None,
                     on_task_complete: Callable = None,
                     on_task_fail: Callable = None):
        """Set execution callbacks."""
        self._on_task_start = on_task_start
        self._on_task_complete = on_task_complete
        self._on_task_fail = on_task_fail
    
    def create_execution(self, graph: ExecutionGraph) -> str:
        """Create a new execution from a graph."""
        plan_id = graph.plan_id
        
        with self._execution_lock:
            self._active_executions[plan_id] = graph
        
        # Save initial checkpoint
        self._save_checkpoint(plan_id, graph)
        
        logger.info(f"Created execution for plan: {plan_id}")
        return plan_id
    
    def execute_plan(self, 
                   graph: ExecutionGraph,
                   task_handler: Callable = None) -> Dict[str, Any]:
        """
        Execute a plan with durable semantics.
        
        Args:
            graph: The execution graph
            task_handler: Handler function for executing tasks
            
        Returns:
            Execution results
        """
        plan_id = graph.plan_id
        graph.status = PlanStatus.RUNNING
        
        # Register handler if provided
        if task_handler:
            self._task_handlers["default"] = task_handler
        
        # Create execution
        self.create_execution(graph)
        
        # Get execution order
        layers = graph.get_execution_order()
        
        results = {
            "plan_id": plan_id,
            "total_tasks": len(graph.nodes),
            "completed": 0,
            "failed": 0,
            "task_results": {}
        }
        
        # Execute layer by layer
        for layer_idx, layer in enumerate(layers):
            logger.info(f"Executing layer {layer_idx + 1}/{len(layers)} with {len(layer)} tasks")
            
            layer_results = self._execute_layer(graph, layer)
            
            # Update results
            for task_id, result in layer_results.items():
                results["task_results"][task_id] = result
                if result.get("status") == "completed":
                    results["completed"] += 1
                else:
                    results["failed"] += 1
            
            # Check for failures that block downstream
            failed_tasks = [t for t in layer if t.status == TaskStatus.FAILED]
            if failed_tasks:
                # Mark dependent tasks as blocked/cancelled
                self._handle_layer_failure(graph, failed_tasks)
        
        # Finalize
        graph.status = PlanStatus.COMPLETED if results["failed"] == 0 else PlanStatus.FAILED
        
        # Save final checkpoint
        self._save_checkpoint(plan_id, graph)
        
        return results
    
    def _execute_layer(self, graph: ExecutionGraph, layer: List[TaskNode]) -> Dict[str, Any]:
        """Execute a layer of tasks in parallel."""
        results = {}
        
        # Execute tasks in parallel (simplified - use ThreadPoolExecutor in production)
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(layer), self.max_concurrent)) as executor:
            future_to_task = {
                executor.submit(self._execute_task, graph, task): task
                for task in layer
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.id] = result
                except Exception as e:
                    logger.error(f"Task {task.id} failed with exception: {e}")
                    results[task.id] = {"status": "failed", "error": str(e)}
        
        return results
    
    def _execute_task(self, graph: ExecutionGraph, task: TaskNode) -> Dict[str, Any]:
        """Execute a single task."""
        task_id = task.id
        
        # Emit event
        self._emit_event("task_started", task_id, {"description": task.description})
        
        # Update status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        
        # Callback
        if self._on_task_start:
            try:
                self._on_task_start(task)
            except Exception as e:
                logger.warning(f"Task start callback failed: {e}")
        
        # Get handler
        handler = self._task_handlers.get(task.task_type) or self._task_handlers.get("default")
        
        if not handler:
            # No handler - mark as completed with warning
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.output_data = {"warning": "No handler registered"}
            self._emit_event("task_completed", task_id, {"status": "completed"})
            return {"status": "completed", "output": task.output_data}
        
        # Execute with timeout and retry
        max_attempts = task.max_retries + 1
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                # Execute handler
                output = handler(task)
                
                # Success
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                task.output_data = output if isinstance(output, dict) else {"output": output}
                
                # Emit event
                self._emit_event("task_completed", task_id, {
                    "status": "completed",
                    "attempts": attempt + 1
                })

                # Wire learning & analytics: record outcome after each completed task
                try:
                    output_str = (
                        output if isinstance(output, str)
                        else str(output)[:800]
                    )
                    model_id = task.input_data.get("assigned_model") or task.assigned_model or "default"
                    task_type_str = task.task_type.lower() if hasattr(task, "task_type") and task.task_type else "general"

                    from vetinari.learning.quality_scorer import get_quality_scorer
                    scorer = get_quality_scorer()
                    q_score = scorer.score(
                        task_id=task_id,
                        model_id=model_id,
                        task_type=task_type_str,
                        task_description=task.description or "",
                        output=output_str,
                        use_llm=False,
                    )

                    from vetinari.learning.feedback_loop import get_feedback_loop
                    get_feedback_loop().record_outcome(
                        task_id=task_id,
                        model_id=model_id,
                        task_type=task_type_str,
                        quality_score=q_score.overall_score,
                        success=True,
                    )

                    from vetinari.learning.model_selector import get_thompson_selector
                    get_thompson_selector().update(model_id, task_type_str, q_score.overall_score, True)
                except Exception as _learn_err:
                    logger.debug(f"Learning hook failed (non-fatal): {_learn_err}")

                # Callback
                if self._on_task_complete:
                    try:
                        self._on_task_complete(task)
                    except Exception as e:
                        logger.warning(f"Task complete callback failed: {e}")
                
                # Save checkpoint
                self._save_checkpoint(graph.plan_id, graph)
                
                return {"status": "completed", "output": task.output_data}
                
            except Exception as e:
                last_error = str(e)
                task.retry_count = attempt + 1
                logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {e}")
                
                # Wait before retry
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # All attempts failed
        task.status = TaskStatus.FAILED
        task.error = last_error
        task.completed_at = datetime.now().isoformat()
        
        # Emit event
        self._emit_event("task_failed", task_id, {
            "status": "failed",
            "error": last_error,
            "attempts": max_attempts
        })
        
        # Callback
        if self._on_task_fail:
            try:
                self._on_task_fail(task)
            except Exception as e:
                logger.warning(f"Task fail callback failed: {e}")
        
        # Save checkpoint
        self._save_checkpoint(graph.plan_id, graph)
        
        return {"status": "failed", "error": last_error}
    
    def _handle_layer_failure(self, graph: ExecutionGraph, failed_tasks: List[TaskNode]):
        """Handle failure in a layer - cancel dependent tasks transitively."""
        cancelled_ids: Set[str] = {t.id for t in failed_tasks}

        # Iteratively expand cancellation to all transitive dependents
        changed = True
        while changed:
            changed = False
            for node in graph.nodes.values():
                if node.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    continue
                if any(dep in cancelled_ids for dep in node.depends_on):
                    if node.id not in cancelled_ids:
                        node.status = TaskStatus.CANCELLED
                        cancelled_ids.add(node.id)
                        self._emit_event("task_cancelled", node.id, {
                            "reason": "dependency_failed",
                            "failed_dependencies": [
                                dep for dep in node.depends_on if dep in cancelled_ids
                            ]
                        })
                        changed = True
    
    def _emit_event(self, event_type: str, task_id: str, data: Dict[str, Any]):
        """Emit an execution event."""
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            data=data
        )
        self._event_history.append(event)
        
        # Also log
        logger.debug(f"Event: {event_type} - {task_id}")
    
    def _save_checkpoint(self, plan_id: str, graph: ExecutionGraph):
        """Save a checkpoint of the execution state."""
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            plan_id=plan_id,
            created_at=datetime.now().isoformat(),
            graph_state=graph.to_dict(),
            completed_tasks=[t.id for t in graph.get_completed_tasks()],
            running_tasks=[t.id for t in graph.nodes.values() if t.status == TaskStatus.RUNNING],
            metadata={"event_count": len(self._event_history)}
        )
        
        # Save to file
        checkpoint_file = self.checkpoint_dir / f"{plan_id}_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump({
                "checkpoint_id": checkpoint.checkpoint_id,
                "plan_id": checkpoint.plan_id,
                "created_at": checkpoint.created_at,
                "graph_state": checkpoint.graph_state,
                "completed_tasks": checkpoint.completed_tasks,
                "running_tasks": checkpoint.running_tasks,
                "metadata": checkpoint.metadata,
            }, f, indent=2)
    
    def load_checkpoint(self, plan_id: str) -> Optional[ExecutionGraph]:
        """Load a checkpoint to resume execution."""
        checkpoint_file = self.checkpoint_dir / f"{plan_id}_checkpoint.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"No checkpoint found for plan: {plan_id}")
            return None
        
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
        
        # Reconstruct graph
        graph_data = data["graph_state"]
        graph = ExecutionGraph(
            plan_id=graph_data["plan_id"],
            goal=graph_data["goal"],
            created_at=graph_data["created_at"],
            updated_at=graph_data["updated_at"],
            status=PlanStatus(graph_data["status"]),
            current_layer=graph_data.get("current_layer", 0),
            completed_count=graph_data.get("completed_count", 0),
            failed_count=graph_data.get("failed_count", 0),
        )
        
        # Reconstruct nodes
        for node_id, node_data in graph_data["nodes"].items():
            graph.nodes[node_id] = TaskNode.from_dict(node_data)
        
        # Restore to active executions
        with self._execution_lock:
            self._active_executions[plan_id] = graph
        
        logger.info(f"Loaded checkpoint for plan: {plan_id}")
        return graph
    
    def recover_execution(self, plan_id: str) -> Dict[str, Any]:
        """
        Recover and continue an execution from checkpoint.
        
        Returns:
            Recovery results
        """
        graph = self.load_checkpoint(plan_id)
        
        if not graph:
            return {"status": "error", "message": "No checkpoint found"}
        
        # Find incomplete tasks
        incomplete = [
            n for n in graph.nodes.values()
            if n.status in (TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.FAILED)
        ]
        
        # Reset failed tasks for retry
        for node in graph.nodes.values():
            if node.status == TaskStatus.FAILED:
                if node.retry_count < node.max_retries:
                    node.status = TaskStatus.PENDING
                    node.error = ""
        
        # Continue execution
        logger.info(f"Recovering {len(incomplete)} incomplete tasks for plan: {plan_id}")
        
        return self.execute_plan(graph)
    
    def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an execution."""
        with self._execution_lock:
            graph = self._active_executions.get(plan_id)
        
        if not graph:
            # Try to load from checkpoint
            graph = self.load_checkpoint(plan_id)
        
        if not graph:
            return None
        
        return {
            "plan_id": plan_id,
            "status": graph.status.value,
            "total_tasks": len(graph.nodes),
            "completed": len(graph.get_completed_tasks()),
            "failed": len(graph.get_failed_tasks()),
            "blocked": len(graph.get_blocked_tasks()),
            "progress": len(graph.get_completed_tasks()) / len(graph.nodes) if graph.nodes else 0,
        }
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return [f.stem.replace("_checkpoint", "") 
                for f in self.checkpoint_dir.glob("*_checkpoint.json")]


# ============================================================
# Two-Layer Orchestrator
# ============================================================

class TwoLayerOrchestrator:
    """
    Complete two-layer orchestration system implementing the assembly-line pattern.

    Assembly-line workflow:
      1. INPUT ANALYSIS   -- classify and assess complexity
      2. PLAN GENERATION  -- LLM-powered task decomposition
      3. TASK DECOMP      -- recursive breakdown to atomic tasks
      4. MODEL ASSIGNMENT -- intelligent model routing per task
      5. PARALLEL EXEC    -- DAG-scheduled execution
      6. OUTPUT REVIEW    -- consistency and quality check
      7. FINAL ASSEMBLY   -- synthesize final output

    Combines:
    - Layer 1: Graph-Based Planning (PlanGenerator)
    - Layer 2: Durable Execution (DurableExecutionEngine)
    """

    def __init__(self,
                 checkpoint_dir: str = None,
                 max_concurrent: int = 4,
                 model_router=None,
                 agent_context: Dict[str, Any] = None):
        self.plan_generator = PlanGenerator(model_router)
        self.execution_engine = DurableExecutionEngine(
            checkpoint_dir=checkpoint_dir,
            max_concurrent=max_concurrent
        )
        self.model_router = model_router
        self.agent_context: Dict[str, Any] = agent_context or {}
        self._agents: Dict[str, Any] = {}

        logger.info("TwoLayerOrchestrator initialized (assembly-line mode)")

    def set_task_handlers(self, handlers: Dict[str, Callable]):
        """Set task handlers for execution."""
        for task_type, handler in handlers.items():
            self.execution_engine.register_handler(task_type, handler)

    def set_agent_context(self, context: Dict[str, Any]) -> None:
        """Set the shared agent context (adapter_manager, web_search, etc.)."""
        self.agent_context = context
        # Re-initialize any cached agents
        self._agents.clear()

    # Complete mapping from agent type string to (module, getter_function)
    _AGENT_MODULE_MAP = {
        "PLANNER":                  ("vetinari.agents.planner_agent",               "get_planner_agent"),
        "EXPLORER":                 ("vetinari.agents.explorer_agent",              "get_explorer_agent"),
        "ORACLE":                   ("vetinari.agents.oracle_agent",                "get_oracle_agent"),
        "LIBRARIAN":                ("vetinari.agents.librarian_agent",             "get_librarian_agent"),
        "RESEARCHER":               ("vetinari.agents.researcher_agent",            "get_researcher_agent"),
        "EVALUATOR":                ("vetinari.agents.evaluator_agent",             "get_evaluator_agent"),
        "SYNTHESIZER":              ("vetinari.agents.synthesizer_agent",           "get_synthesizer_agent"),
        "BUILDER":                  ("vetinari.agents.builder_agent",               "get_builder_agent"),
        "UI_PLANNER":               ("vetinari.agents.ui_planner_agent",            "get_ui_planner_agent"),
        "SECURITY_AUDITOR":         ("vetinari.agents.security_auditor_agent",      "get_security_auditor_agent"),
        "DATA_ENGINEER":            ("vetinari.agents.data_engineer_agent",         "get_data_engineer_agent"),
        "DOCUMENTATION_AGENT":      ("vetinari.agents.documentation_agent",         "get_documentation_agent"),
        "COST_PLANNER":             ("vetinari.agents.cost_planner_agent",          "get_cost_planner_agent"),
        "TEST_AUTOMATION":          ("vetinari.agents.test_automation_agent",       "get_test_automation_agent"),
        "EXPERIMENTATION_MANAGER":  ("vetinari.agents.experimentation_manager_agent", "get_experimentation_manager_agent"),
        "IMPROVEMENT":              ("vetinari.agents.improvement_agent",           "get_improvement_agent"),
        "USER_INTERACTION":         ("vetinari.agents.user_interaction_agent",      "get_user_interaction_agent"),
        "DEVOPS":                   ("vetinari.agents.devops_agent",                "get_devops_agent"),
        "VERSION_CONTROL":          ("vetinari.agents.version_control_agent",       "get_version_control_agent"),
        "ERROR_RECOVERY":           ("vetinari.agents.error_recovery_agent",        "get_error_recovery_agent"),
        "CONTEXT_MANAGER":          ("vetinari.agents.context_manager_agent",       "get_context_manager_agent"),
        "IMAGE_GENERATOR":          ("vetinari.agents.image_generator_agent",       "get_image_generator_agent"),
    }

    def _get_agent(self, agent_type_str: str):
        """Get or create an agent by type string, initialized with context."""
        key = agent_type_str.upper()
        if key in self._agents:
            return self._agents[key]
        if key not in self._AGENT_MODULE_MAP:
            logger.debug(f"No agent module registered for type: {key}")
            return None
        try:
            import importlib
            mod_path, fn_name = self._AGENT_MODULE_MAP[key]
            mod = importlib.import_module(mod_path)
            getter = getattr(mod, fn_name, None)
            if getter is None:
                return None
            agent = getter()
            if self.agent_context:
                agent.initialize(self.agent_context)
            self._agents[key] = agent
            return agent
        except Exception as e:
            logger.warning(f"Could not get agent '{key}': {e}")
            return None

    def _route_model_for_task(self, task: "TaskNode") -> str:
        """Select the best model for a task using dynamic model routing."""
        if self.model_router is None:
            try:
                from vetinari.dynamic_model_router import get_model_router
                self.model_router = get_model_router()
            except Exception:
                return "default"
        try:
            from vetinari.dynamic_model_router import TaskType
            task_type_map = {
                "analysis": TaskType.ANALYSIS,
                "implementation": TaskType.CODING,
                "testing": TaskType.TESTING,
                "research": TaskType.ANALYSIS,
                "documentation": TaskType.DOCUMENTATION,
                "verification": TaskType.CODE_REVIEW,
            }
            t_type = task_type_map.get(task.task_type.lower(), TaskType.GENERAL)
            selection = self.model_router.select_model(t_type)
            if selection and selection.model:
                return selection.model.id
        except Exception as e:
            logger.debug(f"Model routing failed for task {task.id}: {e}")
        return "default"

    def generate_and_execute(self,
                             goal: str,
                             constraints: Dict[str, Any] = None,
                             task_handler: Callable = None,
                             context: Dict[str, Any] = None,
                             project_id: Optional[str] = None,
                             model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Full assembly-line pipeline: analyze → plan → decompose →
        assign models → execute → review → assemble.

        Args:
            goal: The goal to achieve
            constraints: Optional constraints (budget, time, etc.)
            task_handler: Optional custom task handler; defaults to
                          the registered agent-based handler
            context: Additional context from the project intake form
                     (required_features, things_to_avoid, tech_stack, etc.)
            project_id: Optional project ID for rules injection
            model_id: Optional model ID for rules injection

        Returns:
            Dict with keys: plan_id, completed, failed, outputs,
                            review_result, final_output, stages
        """
        stages: Dict[str, Any] = {}
        start_time = time.time()
        context = context or {}

        # Enrich goal with intake form context
        enriched_goal = goal
        if context.get("required_features"):
            enriched_goal += "\n\nRequired features:\n" + "\n".join(
                f"- {f}" for f in context["required_features"]
            )
        if context.get("things_to_avoid"):
            enriched_goal += "\n\nDo NOT include:\n" + "\n".join(
                f"- {a}" for a in context["things_to_avoid"]
            )
        if context.get("tech_stack"):
            enriched_goal += f"\n\nTech stack: {context['tech_stack']}"
        if context.get("priority"):
            enriched_goal += f"\n\nPriority: {context['priority']}"

        # Inject rules into context for agent prompts
        try:
            from vetinari.rules_manager import get_rules_manager
            rm = get_rules_manager()
            context["_rules_prefix"] = rm.build_system_prompt_prefix(
                project_id=project_id, model_id=model_id
            )
        except Exception:
            pass

        # ----------------------------------------------------------------
        # STAGE 1: Input Analysis
        # ----------------------------------------------------------------
        logger.info(f"[Pipeline] Stage 1: Input Analysis for goal: {goal[:80]}")
        analysis = self._analyze_input(enriched_goal, constraints or {})
        stages["input_analysis"] = analysis

        # ----------------------------------------------------------------
        # STAGE 2 & 3: Plan Generation + Task Decomposition
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 2-3: Plan Generation & Decomposition")
        graph = self.plan_generator.generate_plan(enriched_goal, constraints)
        stages["plan"] = {"plan_id": graph.plan_id, "tasks": len(graph.nodes)}

        # ----------------------------------------------------------------
        # STAGE 4: Model Assignment
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 4: Model Assignment")
        for node in graph.nodes.values():
            assigned_model = self._route_model_for_task(node)
            node.input_data["assigned_model"] = assigned_model
            logger.debug(f"  Task {node.id} ({node.task_type}) -> {assigned_model}")
        stages["model_assignment"] = {nid: n.input_data.get("assigned_model") for nid, n in graph.nodes.items()}

        # ----------------------------------------------------------------
        # STAGE 5: Parallel Execution
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 5: Parallel Execution")
        effective_handler = task_handler or self._make_default_handler()
        exec_results = self.execution_engine.execute_plan(graph, effective_handler)
        stages["execution"] = exec_results

        # ----------------------------------------------------------------
        # STAGE 6: Output Review
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 6: Output Review")
        review_result = self._review_outputs(exec_results, goal)
        stages["review"] = review_result

        # ----------------------------------------------------------------
        # STAGE 7: Final Assembly
        # ----------------------------------------------------------------
        logger.info("[Pipeline] Stage 7: Final Assembly")
        final_output = self._assemble_final_output(exec_results, review_result, goal)
        stages["final_assembly"] = {"output_length": len(str(final_output))}

        total_time = int((time.time() - start_time) * 1000)
        return {
            "plan_id": graph.plan_id,
            "goal": goal,
            "completed": exec_results.get("completed", 0),
            "failed": exec_results.get("failed", 0),
            "outputs": exec_results.get("task_results", {}),
            "review_result": review_result,
            "final_output": final_output,
            "stages": stages,
            "total_time_ms": total_time,
        }

    def _analyze_input(self, goal: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the input goal and estimate complexity."""
        result = {
            "goal": goal,
            "estimated_complexity": "medium",
            "domain": "general",
            "needs_research": False,
            "needs_code": False,
            "needs_ui": False,
        }
        g = goal.lower()
        result["needs_code"] = any(k in g for k in ["code", "implement", "build", "create", "program", "software"])
        result["needs_research"] = any(k in g for k in ["research", "analyze", "investigate", "study"])
        result["needs_ui"] = any(k in g for k in ["ui", "frontend", "interface", "web app", "dashboard"])
        result["domain"] = ("coding" if result["needs_code"] else
                            "research" if result["needs_research"] else "general")
        word_count = len(goal.split())
        result["estimated_complexity"] = "simple" if word_count < 10 else "complex" if word_count > 30 else "medium"
        return result

    def _make_default_handler(self) -> Callable:
        """Create a default task handler that uses agent inference with token optimisation."""
        def handle_task(task: "TaskNode") -> Dict[str, Any]:
            try:
                # Determine if target model is a cloud model
                assigned_model = task.input_data.get("assigned_model", "default")
                is_cloud = not any(
                    x in assigned_model.lower()
                    for x in ["qwen", "llama", "mistral", "gemma", "phi", "local", "lm_studio", "default"]
                )

                # Apply token optimisation
                task_context = " ".join(
                    str(v)[:500] for v in task.input_data.values() if v
                ) if task.input_data else ""

                try:
                    from vetinari.token_optimizer import get_token_optimizer
                    optimizer = get_token_optimizer()
                    opt_result = optimizer.prepare_prompt(
                        prompt=task.description,
                        context=task_context,
                        task_type=task.task_type or "general",
                        task_description=task.description,
                        is_cloud_model=is_cloud,
                        task_id=task.id,
                    )
                    optimised_prompt = opt_result["prompt"]
                    max_tokens = opt_result["max_tokens"]
                    temperature = opt_result["temperature"]
                except Exception:
                    optimised_prompt = task.description
                    max_tokens = 2048
                    temperature = 0.3

                adapter_manager = self.agent_context.get("adapter_manager")
                if adapter_manager:
                    try:
                        from vetinari.adapters.base import InferenceRequest
                        req = InferenceRequest(
                            model_id=assigned_model,
                            prompt=optimised_prompt,
                            system_prompt=f"Execute this {task.task_type or 'general'} task precisely and completely.",
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        resp = adapter_manager.infer(req)
                        if resp.status == "ok":
                            return {"result": resp.output, "status": "ok", "task_id": task.id}
                    except Exception as e:
                        logger.warning(f"Adapter inference failed for task {task.id}: {e}")

                # Fallback: use LM Studio adapter directly
                import os
                from vetinari.lmstudio_adapter import LMStudioAdapter
                host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
                adapter = LMStudioAdapter(host=host)
                result = adapter.chat(
                    model_id=assigned_model,
                    system_prompt=f"You are executing a {task.task_type or 'general'} task.",
                    input_text=optimised_prompt,
                )
                return {"result": result.get("output", ""), "status": "ok", "task_id": task.id}
            except Exception as e:
                logger.error(f"Task handler failed for {task.id}: {e}")
                return {"result": "", "status": "error", "error": str(e), "task_id": task.id}

        return handle_task

    def _review_outputs(self, exec_results: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Use EvaluatorAgent to review execution outputs for quality."""
        try:
            evaluator = self._get_agent("EVALUATOR")
            if evaluator:
                from vetinari.agents.contracts import AgentTask, AgentType
                task_results = exec_results.get("task_results", {})
                artifacts = [str(v) for v in task_results.values() if v]
                eval_task = AgentTask(
                    task_id="review-0",
                    agent_type=AgentType.EVALUATOR,
                    description=f"Review outputs for goal: {goal}",
                    context={"artifacts": artifacts[:5], "focus": "all"},
                )
                result = evaluator.execute(eval_task)
                if result.success:
                    return result.output
        except Exception as e:
            logger.warning(f"Output review failed: {e}")
        return {"verdict": "inconclusive", "quality_score": 0.5, "summary": "Review skipped (evaluator unavailable)"}

    def _assemble_final_output(self, exec_results: Dict[str, Any],
                               review_result: Dict[str, Any], goal: str) -> str:
        """Use SynthesizerAgent to assemble a final coherent output."""
        try:
            synthesizer = self._get_agent("SYNTHESIZER")
            if synthesizer:
                from vetinari.agents.contracts import AgentTask, AgentType
                task_results = exec_results.get("task_results", {})
                sources = [
                    {"agent": k, "artifact": str(v)[:500]}
                    for k, v in task_results.items() if v
                ]
                sources.append({"agent": "review", "artifact": str(review_result)[:200]})
                synth_task = AgentTask(
                    task_id="assemble-0",
                    agent_type=AgentType.SYNTHESIZER,
                    description=f"Final assembly for goal: {goal}",
                    context={"sources": sources, "type": "final_report"},
                )
                result = synthesizer.execute(synth_task)
                if result.success and result.output:
                    return result.output.get("synthesized_artifact", str(result.output))
        except Exception as e:
            logger.warning(f"Final assembly failed: {e}")

        # Fallback: join task_results
        task_results = exec_results.get("task_results", {})
        parts = [f"# Task {k}\n{v}" for k, v in task_results.items() if v]
        return "\n\n".join(parts) if parts else f"Completed: {goal}"

    def generate_plan_only(self,
                           goal: str,
                           constraints: Dict[str, Any] = None) -> "ExecutionGraph":
        """Generate a plan without executing."""
        return self.plan_generator.generate_plan(goal, constraints)

    def execute_plan(self,
                     graph: "ExecutionGraph",
                     task_handler: Callable = None) -> Dict[str, Any]:
        """Execute an existing plan."""
        return self.execution_engine.execute_plan(graph, task_handler)

    def recover_plan(self, plan_id: str) -> Dict[str, Any]:
        """Recover and continue a plan from checkpoint."""
        return self.execution_engine.recover_execution(plan_id)

    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a plan."""
        return self.execution_engine.get_execution_status(plan_id)

    def list_checkpoints(self) -> List[str]:
        """List all available plan checkpoints."""
        return self.execution_engine.list_checkpoints()


# Global orchestrator
_two_layer_orchestrator: Optional[TwoLayerOrchestrator] = None


def get_two_layer_orchestrator() -> TwoLayerOrchestrator:
    """Get or create the global two-layer orchestrator."""
    global _two_layer_orchestrator
    if _two_layer_orchestrator is None:
        _two_layer_orchestrator = TwoLayerOrchestrator()
    return _two_layer_orchestrator


def init_two_layer_orchestrator(checkpoint_dir: str = None, **kwargs) -> TwoLayerOrchestrator:
    """Initialize a new two-layer orchestrator."""
    global _two_layer_orchestrator
    _two_layer_orchestrator = TwoLayerOrchestrator(checkpoint_dir=checkpoint_dir, **kwargs)
    return _two_layer_orchestrator


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test the orchestrator
    orchestrator = TwoLayerOrchestrator()
    
    # Simple task handler
    def handle_task(task: TaskNode) -> Dict[str, Any]:
        logger.info(f"Executing task: {task.description}")
        time.sleep(0.5)  # Simulate work
        return {"result": f"Completed: {task.description}", "status": "ok"}
    
    orchestrator.set_task_handlers({
        "default": handle_task,
        "analysis": handle_task,
        "implementation": handle_task,
        "testing": handle_task,
        "verification": handle_task,
    })
    
    # Generate and execute
    logger.info("\n=== Generating and executing plan ===")
    results = orchestrator.generate_and_execute(
        "Build a Python web application with user authentication",
        task_handler=handle_task
    )

    logger.info(f"\nResults:")
    logger.info(f"  Completed: {results['completed']}")
    logger.info(f"  Failed: {results['failed']}")

    # Test recovery
    logger.info("\n=== Testing checkpoint recovery ===")
    recovery = orchestrator.recover_plan(results["plan_id"])
    logger.info(f"Recovery status: {recovery.get('status', 'unknown')}")
