"""
Tool Registry Integration for Vetinari

.. deprecated::
    This module is deprecated. Use ``vetinari.tool_interface.get_tool_registry()``
    directly instead.

Registers all tools in the tool registry including:
- Web Search Tool
- Code Sandbox
- Memory Tools
- Model Router Tools
- Two-Layer Orchestration Tools
"""

import logging
import warnings
from typing import Dict, Any, List, Optional

from vetinari.tool_interface import (
    Tool,
    ToolMetadata,
    ToolParameter,
    ToolResult,
    ToolCategory,
    get_tool_registry,
)

from vetinari.execution_context import ToolPermission, ExecutionMode

warnings.warn(
    "vetinari.tools.tool_registry_integration is deprecated. "
    "Use vetinari.tool_interface.get_tool_registry() directly instead.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)


class WebSearchToolWrapper(Tool):
    """Wrapper for the web search tool."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="web_search",
            description="Search the web for information with citations and provenance tracking",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="query",
                    type=str,
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type=int,
                    description="Maximum number of results",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="backend",
                    type=str,
                    description="Search backend to use",
                    required=False,
                    default="duckduckgo",
                ),
            ],
            required_permissions=[ToolPermission.WEB_ACCESS],
            allowed_modes=[ExecutionMode.EXECUTION],
            tags=["search", "web", "information", "research"],
        )
        super().__init__(metadata)
        
        # Lazy import
        self._search_tool = None
    
    def _get_search_tool(self):
        if self._search_tool is None:
            from vetinari.tools.web_search_tool import get_search_tool
            self._search_tool = get_search_tool()
        return self._search_tool
    
    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        backend = kwargs.get("backend", "duckduckgo")
        
        try:
            search_tool = self._get_search_tool()
            response = search_tool.search(query, max_results=max_results)
            
            return ToolResult(
                success=True,
                output={
                    "results": [r.to_dict() for r in response.results],
                    "query": response.query,
                    "total_results": response.total_results,
                    "citations": response.get_citations(),
                },
                metadata={
                    "backend": backend,
                    "execution_time_ms": response.execution_time_ms,
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


class ResearchTopicToolWrapper(Tool):
    """Wrapper for comprehensive topic research."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="research_topic",
            description="Perform comprehensive research on a topic with multiple queries and sources",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="topic",
                    type=str,
                    description="Topic to research",
                    required=True,
                ),
                ToolParameter(
                    name="aspects",
                    type=List[str],
                    description="Specific aspects to investigate",
                    required=False,
                    default=None,
                ),
            ],
            required_permissions=[ToolPermission.WEB_ACCESS],
            allowed_modes=[ExecutionMode.EXECUTION],
            tags=["research", "web", "information"],
        )
        super().__init__(metadata)
        
        self._search_tool = None
    
    def _get_search_tool(self):
        if self._search_tool is None:
            from vetinari.tools.web_search_tool import get_search_tool
            self._search_tool = get_search_tool()
        return self._search_tool
    
    def execute(self, **kwargs) -> ToolResult:
        topic = kwargs.get("topic", "")
        aspects = kwargs.get("aspects")
        
        try:
            search_tool = self._get_search_tool()
            result = search_tool.research_topic(topic, aspects)
            
            return ToolResult(
                success=True,
                output=result,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


class CodeExecutionToolWrapper(Tool):
    """Wrapper for code execution sandbox."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="execute_code",
            description="Execute code in a sandboxed environment",
            category=ToolCategory.CODE_EXECUTION,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="code",
                    type=str,
                    description="Code to execute",
                    required=True,
                ),
                ToolParameter(
                    name="language",
                    type=str,
                    description="Programming language",
                    required=False,
                    default="python",
                ),
                ToolParameter(
                    name="timeout",
                    type=int,
                    description="Execution timeout in seconds",
                    required=False,
                    default=60,
                ),
            ],
            required_permissions=[ToolPermission.CODE_EXECUTION],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.SANDBOX],
            tags=["code", "execution", "sandbox"],
        )
        super().__init__(metadata)
        
        self._executor = None
    
    def _get_executor(self):
        if self._executor is None:
            from vetinari.sandbox import get_code_executor
            self._executor = get_code_executor()
        return self._executor
    
    def execute(self, **kwargs) -> ToolResult:
        code = kwargs.get("code", "")
        language = kwargs.get("language", "python")
        timeout = kwargs.get("timeout", 60)
        
        try:
            executor = self._get_executor()
            result = executor.run(code, language=language, timeout=timeout)
            
            return ToolResult(
                success=result.get("success", False),
                output=result,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


class MemoryRecallToolWrapper(Tool):
    """Wrapper for memory recall."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="recall_memory",
            description="Recall information from memory",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="query",
                    type=str,
                    description="Query to search memory",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="memory_type",
                    type=str,
                    description="Type of memory to search",
                    required=False,
                    default=None,
                ),
                ToolParameter(
                    name="limit",
                    type=int,
                    description="Maximum results",
                    required=False,
                    default=5,
                ),
            ],
            required_permissions=[],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["memory", "recall", "context"],
        )
        super().__init__(metadata)
        
        self._memory = None
    
    def _get_memory(self):
        if self._memory is None:
            from vetinari.enhanced_memory import get_memory_manager
            self._memory = get_memory_manager()
        return self._memory
    
    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        memory_type = kwargs.get("memory_type")
        limit = kwargs.get("limit", 5)
        
        try:
            memory = self._get_memory()
            results = memory.recall(query=query, limit=limit)
            
            return ToolResult(
                success=True,
                output={
                    "entries": [e.to_dict() for e in results],
                    "count": len(results),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


class MemoryRememberToolWrapper(Tool):
    """Wrapper for storing in memory."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="remember",
            description="Store information in memory for future recall",
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="content",
                    type=str,
                    description="Content to remember",
                    required=True,
                ),
                ToolParameter(
                    name="memory_type",
                    type=str,
                    description="Type of memory",
                    required=False,
                    default="context",
                ),
                ToolParameter(
                    name="tags",
                    type=List[str],
                    description="Tags for the memory",
                    required=False,
                    default=None,
                ),
            ],
            required_permissions=[],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["memory", "remember", "store"],
        )
        super().__init__(metadata)
        
        self._memory = None
    
    def _get_memory(self):
        if self._memory is None:
            from vetinari.enhanced_memory import get_memory_manager
            self._memory = get_memory_manager()
        return self._memory
    
    def execute(self, **kwargs) -> ToolResult:
        content = kwargs.get("content", "")
        memory_type = kwargs.get("memory_type", "context")
        tags = kwargs.get("tags", [])
        
        try:
            memory = self._get_memory()
            
            # Import MemoryType
            from vetinari.enhanced_memory import MemoryType
            mem_type = MemoryType(memory_type)
            
            entry_id = memory.remember(
                content=content,
                memory_type=mem_type,
                tags=tags,
            )
            
            return ToolResult(
                success=True,
                output={"entry_id": entry_id},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


class ModelSelectToolWrapper(Tool):
    """Wrapper for dynamic model selection."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="select_model",
            description="Select the best model for a task based on capabilities",
            category=ToolCategory.MODEL_INFERENCE,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="task_type",
                    type=str,
                    description="Type of task (coding, analysis, reasoning, etc.)",
                    required=True,
                ),
                ToolParameter(
                    name="task_description",
                    type=str,
                    description="Description of the task",
                    required=False,
                    default="",
                ),
            ],
            required_permissions=[ToolPermission.MODEL_INFERENCE],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["model", "selection", "routing"],
        )
        super().__init__(metadata)
        
        self._router = None
    
    def _get_router(self):
        if self._router is None:
            from vetinari.dynamic_model_router import get_model_router, TaskType
            self._router = get_model_router()
        return self._router
    
    def execute(self, **kwargs) -> ToolResult:
        task_type = kwargs.get("task_type", "general")
        task_description = kwargs.get("task_description", "")
        
        try:
            router = self._get_router()
            
            from vetinari.dynamic_model_router import TaskType
            task_enum = TaskType(task_type)
            
            selection = router.select_model(task_enum, task_description)
            
            if selection:
                return ToolResult(
                    success=True,
                    output={
                        "model_id": selection.model.id,
                        "model_name": selection.model.name,
                        "reasoning": selection.reasoning,
                        "confidence": selection.confidence,
                        "alternatives": [a.id for a in selection.alternatives],
                    },
                )
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error="No suitable model found",
                )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


class GeneratePlanToolWrapper(Tool):
    """Wrapper for plan generation."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="generate_plan",
            description="Generate an execution plan from a goal",
            category=ToolCategory.SYSTEM_OPERATIONS,
            version="1.0.0",
            parameters=[
                ToolParameter(
                    name="goal",
                    type=str,
                    description="Goal to achieve",
                    required=True,
                ),
                ToolParameter(
                    name="constraints",
                    type=Dict,
                    description="Constraints for the plan",
                    required=False,
                    default=None,
                ),
            ],
            required_permissions=[ToolPermission.PLANNING],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=["planning", "plan", "decomposition"],
        )
        super().__init__(metadata)
        
        self._orchestrator = None
    
    def _get_orchestrator(self):
        if self._orchestrator is None:
            from vetinari.orchestration.two_layer import get_two_layer_orchestrator
            self._orchestrator = get_two_layer_orchestrator()
        return self._orchestrator
    
    def execute(self, **kwargs) -> ToolResult:
        goal = kwargs.get("goal", "")
        constraints = kwargs.get("constraints", {})
        
        try:
            orchestrator = self._get_orchestrator()
            graph = orchestrator.generate_plan_only(goal, constraints)
            
            return ToolResult(
                success=True,
                output=graph.to_dict(),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


def register_all_tools():
    """Register all tools in the global registry."""
    registry = get_tool_registry()
    
    # List of tool instances to register
    tools = [
        WebSearchToolWrapper(),
        ResearchTopicToolWrapper(),
        CodeExecutionToolWrapper(),
        MemoryRecallToolWrapper(),
        MemoryRememberToolWrapper(),
        ModelSelectToolWrapper(),
        GeneratePlanToolWrapper(),
    ]
    
    # Register each tool
    for tool in tools:
        try:
            registry.register(tool)
            logger.info("Registered tool: %s", tool.metadata.name)
        except Exception as e:
            logger.error("Failed to register tool %s: %s", tool.metadata.name, e)
    
    return len(tools)


# Auto-register on import
def _auto_register():
    """Auto-register tools when module is imported."""
    try:
        count = register_all_tools()
        logger.info("Auto-registered %s tools", count)
    except Exception as e:
        logger.warning("Auto-registration failed: %s", e)


# Try to auto-register
try:
    _auto_register()
except Exception:
    pass  # Auto-registration skipped — tool registry imports not available


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Manual registration for testing
    count = register_all_tools()
    print(f"Registered {count} tools")
    
    # List registered tools
    registry = get_tool_registry()
    print("\nRegistered tools:")
    for tool in registry.list_tools():
        print(f"  - {tool.metadata.name}: {tool.metadata.description}")
