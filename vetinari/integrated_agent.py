"""
Integrated Agent System for Vetinari

Combines all components into a unified orchestration system:
- Web search for information gathering
- Dynamic model routing
- Two-layer orchestration
- Enhanced memory
- Code execution sandbox

This is the main orchestration layer that handles:
- Goal analysis and planning
- Task decomposition and execution
- Model selection
- Information gathering
- Validation and verification
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    GATHERING_INFO = "gathering_info"
    VALIDATING = "validating"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentRequest:
    """Request to the agent system."""
    goal: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: str = ""
    session_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "constraints": self.constraints,
            "context": self.context,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }


@dataclass
class AgentResponse:
    """Response from the agent system."""
    success: bool
    state: AgentState
    goal: str
    plan_id: str = ""
    results: Dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "state": self.state.value,
            "goal": self.goal,
            "plan_id": self.plan_id,
            "results": self.results,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "citations": self.citations,
        }


class IntegratedAgent:
    """
    Integrated agent system combining all Vetinari components.
    
    Workflow:
    1. Analyze goal and constraints
    2. Gather information (web search if needed)
    3. Generate plan with two-layer orchestration
    4. Execute tasks with dynamic model routing
    5. Validate outputs
    6. Return results
    """
    
    def __init__(self,
                 model_pool=None,
                 model_router=None,
                 memory_manager=None,
                 search_tool=None,
                 two_layer_orchestrator=None,
                 code_executor=None,
                 config: Dict[str, Any] = None):
        """
        Initialize the integrated agent.
        
        All components are optional - they will be created if not provided.
        """
        self.config = config or {}
        
        # Components
        self.model_pool = model_pool
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.search_tool = search_tool
        self.two_layer_orchestrator = two_layer_orchestrator
        self.code_executor = code_executor
        
        # State
        self.state = AgentState.IDLE
        self.current_request: Optional[AgentRequest] = None
        self.current_plan_id: str = ""
        
        # Lazy initialization of components
        self._init_components()
        
        logger.info("IntegratedAgent initialized")
    
    def _init_components(self):
        """Initialize any missing components."""
        # Import here to avoid circular dependencies
        
        # Model pool
        if self.model_pool is None:
            try:
                from vetinari.model_pool import ModelPool
                config = self.config.get("model_pool", {})
                self.model_pool = ModelPool(config)
            except Exception as e:
                logger.warning(f"Could not initialize model pool: {e}")
        
        # Model router
        if self.model_router is None:
            try:
                from vetinari.dynamic_model_router import DynamicModelRouter, init_model_router
                prefer_local = self.config.get("prefer_local", True)
                self.model_router = init_model_router(prefer_local=prefer_local)
                
                # Register models from pool
                if self.model_pool:
                    self.model_router.register_models_from_pool(
                        self.model_pool.models
                    )
            except Exception as e:
                logger.warning(f"Could not initialize model router: {e}")
        
        # Memory manager
        if self.memory_manager is None:
            try:
                from vetinari.enhanced_memory import get_memory_manager, MemoryType
                self.memory_manager = get_memory_manager()
            except Exception as e:
                logger.warning(f"Could not initialize memory manager: {e}")
        
        # Search tool
        if self.search_tool is None:
            try:
                from vetinari.tools.web_search_tool import get_search_tool
                self.search_tool = get_search_tool()
            except Exception as e:
                logger.warning(f"Could not initialize search tool: {e}")
        
        # Two-layer orchestrator
        if self.two_layer_orchestrator is None:
            try:
                from vetinari.two_layer_orchestration import get_two_layer_orchestrator
                self.two_layer_orchestrator = get_two_layer_orchestrator()
            except Exception as e:
                logger.warning(f"Could not initialize two-layer orchestrator: {e}")
        
        # Code executor
        if self.code_executor is None:
            try:
                from vetinari.code_sandbox import get_code_executor
                self.code_executor = get_code_executor()
            except Exception as e:
                logger.warning(f"Could not initialize code executor: {e}")
    
    def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Execute a goal through the agent system.
        
        Args:
            request: The agent request with goal and constraints
            
        Returns:
            AgentResponse with results
        """
        self.current_request = request
        start_time = time.time()
        
        # Update state
        self.state = AgentState.PLANNING
        
        try:
            # Step 1: Analyze goal and gather context
            context = self._analyze_goal(request.goal, request.constraints)
            
            # Step 2: Gather information if needed
            if self._needs_information(request.goal):
                self.state = AgentState.GATHERING_INFO
                info = self._gather_information(request.goal)
                context["gathered_info"] = info
            
            # Step 3: Generate and execute plan
            self.state = AgentState.EXECUTING
            results = self._execute_plan(request.goal, context)
            
            # Step 4: Validate outputs
            self.state = AgentState.VALIDATING
            validation = self._validate_results(results)
            
            # Step 5: Remember the execution
            self._remember_execution(request, results)
            
            # Success
            self.state = AgentState.COMPLETED
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return AgentResponse(
                success=True,
                state=self.state,
                goal=request.goal,
                plan_id=self.current_plan_id,
                results=results,
                output=self._format_output(results),
                metadata={
                    "execution_time_ms": execution_time,
                    "context": context,
                    "validation": validation,
                },
                citations=context.get("citations", [])
            )
            
        except Exception as e:
            self.state = AgentState.FAILED
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            
            return AgentResponse(
                success=False,
                state=self.state,
                goal=request.goal,
                error=str(e),
                metadata={"exception": str(e)}
            )
    
    def _analyze_goal(self, goal: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the goal and determine requirements."""
        context = {
            "goal": goal,
            "constraints": constraints,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Infer task type
        if self.model_router:
            try:
                from vetinari.dynamic_model_router import infer_task_type, TaskType
                task_type = infer_task_type(goal)
                context["task_type"] = task_type.value
            except Exception:
                pass
        
        # Check for web search needs
        context["needs_web_search"] = self._needs_information(goal)
        
        return context
    
    def _needs_information(self, goal: str) -> bool:
        """Determine if the goal requires web information."""
        need_keywords = [
            "latest", "newest", "current", "recent",
            "how to", "tutorial", "example",
            "documentation", "api", "reference",
            "best", "compare", "vs"
        ]
        
        goal_lower = goal.lower()
        return any(kw in goal_lower for kw in need_keywords)
    
    def _gather_information(self, goal: str) -> Dict[str, Any]:
        """Gather information from web search."""
        if not self.search_tool:
            return {"error": "Search tool not available"}
        
        # Determine search queries
        queries = self._generate_search_queries(goal)
        
        results = []
        citations = []
        
        for query in queries:
            try:
                response = self.search_tool.search(query, max_results=3)
                for r in response.results:
                    results.append(r.to_dict())
                    citations.append(f"[{r.title}]({r.url})")
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
        
        # Also do a comprehensive research if needed
        research = None
        if len(queries) > 1:
            try:
                research = self.search_tool.research_topic(goal)
            except Exception as e:
                logger.warning(f"Research failed: {e}")
        
        return {
            "queries": queries,
            "results": results,
            "research": research,
            "citations": citations,
        }
    
    def _generate_search_queries(self, goal: str) -> List[str]:
        """Generate search queries from goal."""
        queries = [goal]  # Always include the full goal
        
        # Extract key terms
        goal_lower = goal.lower()
        
        # Add specific queries based on context
        if "python" in goal_lower:
            queries.append(f"{goal} Python tutorial")
        if "api" in goal_lower:
            queries.append(f"{goal} REST API example")
        if "error" in goal_lower or "bug" in goal_lower:
            queries.append(f"{goal} troubleshooting")
        
        return queries[:5]  # Limit to 5 queries
    
    def _execute_plan(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan using two-layer orchestration."""
        if not self.two_layer_orchestrator:
            return {"error": "Orchestrator not available"}
        
        # Create task handler
        def task_handler(task):
            return self._execute_task(task, context)
        
        # Generate and execute plan
        try:
            # Register handlers
            self.two_layer_orchestrator.set_task_handlers({
                "default": task_handler,
                "analysis": task_handler,
                "implementation": task_handler,
                "testing": task_handler,
                "verification": task_handler,
            })
            
            # Execute
            results = self.two_layer_orchestrator.generate_and_execute(
                goal=goal,
                constraints=context.get("constraints", {}),
                task_handler=task_handler
            )
            
            self.current_plan_id = results.get("plan_id", "")
            
            return results
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return {"error": str(e), "plan_execution": "failed"}
    
    def _execute_task(self, task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task using appropriate model."""
        task_id = task.id if hasattr(task, 'id') else task.get("id", "unknown")
        task_desc = task.description if hasattr(task, 'description') else task.get("description", "")
        
        logger.info(f"Executing task: {task_id} - {task_desc}")
        
        # Determine task type
        task_type = task.task_type if hasattr(task, 'task_type') else task.get("task_type", "general")
        
        # Select model
        model_info = None
        if self.model_router:
            try:
                from vetinari.dynamic_model_router import TaskType
                
                # Map task type
                type_map = {
                    "analysis": TaskType.ANALYSIS,
                    "implementation": TaskType.CODING,
                    "testing": TaskType.TESTING,
                    "verification": TaskType.CODE_REVIEW,
                    "documentation": TaskType.DOCUMENTATION,
                }
                
                router_task_type = type_map.get(task_type, TaskType.GENERAL)
                selection = self.model_router.select_model(router_task_type, task_desc)
                
                if selection:
                    model_info = selection.model
                    logger.info(f"Selected model: {model_info.id}")
                    
            except Exception as e:
                logger.warning(f"Model selection failed: {e}")
        
        # Execute with model
        output = self._run_with_model(task_desc, model_info)
        
        # Run code if applicable
        if task_type in ("implementation", "testing") and self.code_executor:
            try:
                if isinstance(output, dict) and "code" in output:
                    code_result = self.code_executor.run(output["code"])
                    output["code_result"] = code_result
            except Exception as e:
                logger.warning(f"Code execution failed: {e}")
        
        return {
            "task_id": task_id,
            "task_type": task_type,
            "model_used": model_info.id if model_info else "unknown",
            "output": output,
            "status": "completed"
        }
    
    def _run_with_model(self, prompt: str, model_info=None) -> Any:
        """Run a prompt with the selected model."""
        if not self.model_pool:
            return {"error": "Model pool not available", "prompt": prompt}
        
        try:
            # Use LM Studio adapter
            from vetinari.lmstudio_adapter import LMStudioAdapter
            
            _default_host = os.environ.get("LM_STUDIO_HOST", "http://100.78.30.7:1234")
            adapter = LMStudioAdapter(host=self.config.get("lmstudio_host", _default_host))
            
            model_id = model_info.id if model_info else "default"
            
            # Simple prompt execution
            response = adapter.chat(
                model_id=model_id,
                system_prompt="You are a helpful coding assistant.",
                input_text=prompt
            )
            
            return {
                "response": response.get("output", ""),
                "model": model_id,
                "latency_ms": response.get("latency_ms", 0)
            }
            
        except Exception as e:
            logger.warning(f"Model execution failed: {e}")
            return {"error": str(e), "prompt": prompt}
    
    def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate execution results."""
        validation = {
            "success": True,
            "checks": {},
        }
        
        # Check for errors
        if results.get("error"):
            validation["success"] = False
            validation["checks"]["no_errors"] = False
        else:
            validation["checks"]["no_errors"] = True
        
        # Check completion
        completed = results.get("completed", 0)
        total = results.get("total_tasks", 0)
        
        if total > 0:
            validation["checks"]["all_tasks_completed"] = completed == total
            validation["completion_rate"] = completed / total
        else:
            validation["checks"]["all_tasks_completed"] = False
        
        # Check failed tasks
        failed = results.get("failed", 0)
        validation["checks"]["no_failed_tasks"] = failed == 0
        validation["failed_count"] = failed
        
        # Update overall success
        validation["success"] = validation["success"] and all(
            v for k, v in validation["checks"].items() if k != "completion_rate"
        )
        
        return validation
    
    def _remember_execution(self, request: AgentRequest, results: Dict[str, Any]):
        """Remember the execution in memory."""
        if not self.memory_manager:
            return
        
        try:
            from vetinari.enhanced_memory import MemoryType
            
            # Remember the goal and outcome
            self.memory_manager.remember(
                content=f"Goal: {request.goal}\nSuccess: {results.get('completed', 0) > 0}",
                memory_type=MemoryType.RESULT,
                metadata={
                    "goal": request.goal,
                    "success": results.get("completed", 0) > 0,
                    "plan_id": self.current_plan_id,
                },
                tags=["execution", "result"]
            )
            
        except Exception as e:
            logger.warning(f"Failed to remember execution: {e}")
    
    def _format_output(self, results: Dict[str, Any]) -> str:
        """Format results for output."""
        lines = []
        
        # Summary
        completed = results.get("completed", 0)
        failed = results.get("failed", 0)
        total = results.get("total_tasks", 0)
        
        lines.append(f"Execution complete: {completed}/{total} tasks completed")
        if failed > 0:
            lines.append(f"Warning: {failed} tasks failed")
        
        # Task results
        task_results = results.get("task_results", {})
        if task_results:
            lines.append("\nTask Results:")
            for task_id, result in task_results.items():
                status = result.get("status", "unknown")
                lines.append(f"  - {task_id}: {status}")
        
        return "\n".join(lines)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "state": self.state.value,
            "current_goal": self.current_request.goal if self.current_request else None,
            "plan_id": self.current_plan_id,
            "components": {
                "model_router": self.model_router is not None,
                "memory_manager": self.memory_manager is not None,
                "search_tool": self.search_tool is not None,
                "orchestrator": self.two_layer_orchestrator is not None,
                "code_executor": self.code_executor is not None,
            }
        }


# Global agent instance
_agent_instance: Optional[IntegratedAgent] = None


def get_agent(config: Dict[str, Any] = None) -> IntegratedAgent:
    """Get or create the global agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = IntegratedAgent(config=config)
    return _agent_instance


def init_agent(**kwargs) -> IntegratedAgent:
    """Initialize a new agent instance."""
    global _agent_instance
    _agent_instance = IntegratedAgent(**kwargs)
    return _agent_instance


# Convenience function for CLI
def run_agent_goal(goal: str, **kwargs) -> AgentResponse:
    """Run a goal through the agent system."""
    agent = get_agent()
    request = AgentRequest(goal=goal, **kwargs)
    return agent.execute(request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the agent
    agent = IntegratedAgent()
    
    # Create request
    request = AgentRequest(
        goal="Create a simple Python function that calculates fibonacci numbers",
        constraints={"max_time": 60}
    )
    
    print("=== Running Agent ===")
    response = agent.execute(request)
    
    print(f"\nSuccess: {response.success}")
    print(f"State: {response.state.value}")
    print(f"Output: {response.output}")
    if response.error:
        print(f"Error: {response.error}")
    if response.citations:
        print(f"\nCitations:")
        for cite in response.citations:
            print(f"  {cite}")
