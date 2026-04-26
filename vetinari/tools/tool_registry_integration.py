"""Tool Registry Integration for Vetinari.

Provides convenience wrappers that register tools in the tool registry:
- Web Search Tool
- Code Sandbox
- Memory Tools
- Model Router Tools
- Orchestration Tools
"""

from __future__ import annotations

import logging
import threading

from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
    get_tool_registry,
)
from vetinari.types import ExecutionMode

logger = logging.getLogger(__name__)


def _redact_text(text: str) -> str:
    return f"<redacted:{len(text)} chars>"


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
        """Execute.

        Returns:
            The ToolResult result.
        """
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        # backend param accepted for API compatibility; actual backend is set at startup
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
                },
            )
        except Exception:
            logger.exception("Web search failed for query %s", _redact_text(str(query)))
            return ToolResult(
                success=False,
                output=None,
                error="Web search failed",
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
                    type=list[str],
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
        """Execute.

        Returns:
            The ToolResult result.
        """
        topic = kwargs.get("topic", "")
        aspects = kwargs.get("aspects")

        try:
            search_tool = self._get_search_tool()
            result = search_tool.research_topic(topic, aspects)

            return ToolResult(
                success=True,
                output=result,
            )
        except Exception:
            logger.exception("Topic research failed for topic %s", _redact_text(str(topic)))
            return ToolResult(
                success=False,
                output=None,
                error="Topic research failed",
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
            allowed_modes=[ExecutionMode.EXECUTION],
            tags=["code", "execution", "sandbox"],
        )
        super().__init__(metadata)

        self._executor = None

    def _get_executor(self):
        if self._executor is None:
            from vetinari.code_sandbox import get_subprocess_executor

            self._executor = get_subprocess_executor()
        return self._executor

    def execute(self, **kwargs) -> ToolResult:
        """Execute.

        Returns:
            The ToolResult result.
        """
        code = kwargs.get("code", "")
        language = kwargs.get("language", "python")
        timeout = kwargs.get("timeout", 60)

        try:
            if str(language).lower() != "python":
                return ToolResult(
                    success=False,
                    output=None,
                    error="Only python code execution is supported by the canonical sandbox manager",
                )

            from vetinari.sandbox_manager import get_sandbox_manager

            result = get_sandbox_manager().execute(
                code=code,
                sandbox_type="subprocess",
                timeout=timeout,
                context={},
                client_id="tool_registry.execute_code",
            )
            output = (
                result.to_dict()
                if hasattr(result, "to_dict")
                else {
                    "success": result.success,
                    "output": result.result,
                    "error": result.error,
                }
            )

            return ToolResult(
                success=output.get("success", False),
                output=output,
                error=None if output.get("success", False) else "Sandbox execution failed",
            )
        except Exception:
            logger.exception("Sandbox execution failed for language %r", language)
            return ToolResult(
                success=False,
                output=None,
                error="Sandbox execution failed",
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
            from vetinari.memory import get_unified_memory_store

            self._memory = get_unified_memory_store()
        return self._memory

    def execute(self, **kwargs) -> ToolResult:
        """Execute.

        Returns:
            The ToolResult result.
        """
        query = kwargs.get("query", "")
        memory_type = kwargs.get("memory_type")
        limit = kwargs.get("limit", 5)

        try:
            memory = self._get_memory()
            search_kwargs: dict = {"query": query, "limit": limit}
            if memory_type:
                search_kwargs["entry_type"] = memory_type
            results = memory.search(**search_kwargs)

            return ToolResult(
                success=True,
                output={
                    "entries": [e.to_dict() for e in results],
                    "count": len(results),
                },
            )
        except Exception:
            logger.exception("Memory operation failed for recall query %s", _redact_text(str(query)))
            return ToolResult(
                success=False,
                output=None,
                error="Memory operation failed",
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
                    type=list[str],
                    description="Tags for the memory",
                    required=False,
                    default=None,
                ),
            ],
            required_permissions=[ToolPermission.MEMORY_WRITE],
            allowed_modes=[ExecutionMode.EXECUTION],
            tags=["memory", "remember", "store"],
        )
        super().__init__(metadata)

        self._memory = None

    def _get_memory(self):
        if self._memory is None:
            from vetinari.memory import get_unified_memory_store

            self._memory = get_unified_memory_store()
        return self._memory

    def execute(self, **kwargs) -> ToolResult:
        """Execute.

        Returns:
            The ToolResult result.
        """
        content = kwargs.get("content", "")
        memory_type = kwargs.get("memory_type", "context")
        tags = kwargs.get("tags", [])

        try:
            memory = self._get_memory()

            from vetinari.memory import MemoryEntry
            from vetinari.types import MemoryType

            mem_type = MemoryType(memory_type)

            entry = MemoryEntry(
                content=content,
                entry_type=mem_type,
                metadata={"tags": tags} if tags else None,
            )
            entry_id = memory.remember(entry)

            return ToolResult(
                success=True,
                output={"entry_id": entry_id},
            )
        except Exception:
            logger.exception("Memory operation failed while storing content")
            return ToolResult(
                success=False,
                output=None,
                error="Memory operation failed",
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
            from vetinari.models.dynamic_model_router import get_model_router

            self._router = get_model_router()
        return self._router

    def execute(self, **kwargs) -> ToolResult:
        """Execute.

        Returns:
            The ToolResult result.
        """
        task_type = kwargs.get("task_type", "general")
        task_description = kwargs.get("task_description", "")

        try:
            router = self._get_router()

            from vetinari.models.dynamic_model_router import parse_task_type

            task_enum = parse_task_type(task_type)

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
            return ToolResult(
                success=False,
                output=None,
                error="No suitable model found",
            )
        except Exception:
            logger.exception("Model selection failed for task type %r", task_type)
            return ToolResult(
                success=False,
                output=None,
                error="Model selection failed",
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
                    type=dict,
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
        """Execute.

        Returns:
            The ToolResult result.
        """
        goal = kwargs.get("goal", "")
        constraints = kwargs.get("constraints", {})

        try:
            orchestrator = self._get_orchestrator()
            graph = orchestrator.generate_plan_only(goal, constraints)

            return ToolResult(
                success=True,
                output=graph.to_dict(),
            )
        except Exception:
            logger.exception("Plan generation failed for goal %r", goal)
            return ToolResult(
                success=False,
                output=None,
                error="Plan generation failed",
            )


def _make_file_tool() -> Tool | None:
    """Lazily create a FileOperationsTool instance rooted at the project root.

    The project root is resolved as two levels above the ``vetinari/`` package
    directory (i.e. the repository root).  This avoids using ``os.getcwd()``,
    which is process-dependent and unsafe for sandboxed file operations.
    """
    try:
        from pathlib import Path

        from vetinari.tools.file_tool import FileOperationsTool

        # vetinari/tools/tool_registry_integration.py → up 3 parents → repo root
        project_root = Path(__file__).resolve().parents[2]
        return FileOperationsTool(str(project_root))
    except Exception as e:
        logger.warning("FileOperationsTool unavailable — project root resolution failed: %s", e)
        return None


def _make_git_tool() -> Tool | None:
    """Lazily create a GitOperationsTool instance rooted at the project repository.

    The repo root is resolved as two levels above the ``vetinari/`` package
    directory (i.e. the repository root).  This avoids using ``os.getcwd()``,
    which is process-dependent and unsafe for sandboxed git operations.
    """
    try:
        from pathlib import Path

        from vetinari.tools.git_tool import GitOperationsTool

        # vetinari/tools/tool_registry_integration.py → up 3 parents → repo root
        repo_root = Path(__file__).resolve().parents[2]
        return GitOperationsTool(str(repo_root))
    except Exception as e:
        logger.warning("GitOperationsTool unavailable — repo root resolution failed: %s", e)
        return None


def _make_plan_tool() -> Tool | None:
    """Lazily create a GeneratePlanToolWrapper only when planning is available.

    Importing GeneratePlanToolWrapper unconditionally would register the tool
    even when the planning subsystem is absent, causing confusing
    "capability unavailable" errors at call time rather than at registration.
    """
    try:
        return GeneratePlanToolWrapper()
    except Exception as e:
        logger.warning("GeneratePlanToolWrapper unavailable — planning capability absent: %s", e)
        return None


def register_all_tools() -> None:
    """Register all tools in the global registry.

    Returns:
        The len result.
    """
    registry = get_tool_registry()

    # List of tool instances to register
    tools = [
        WebSearchToolWrapper(),
        ResearchTopicToolWrapper(),
        CodeExecutionToolWrapper(),
        MemoryRecallToolWrapper(),
        MemoryRememberToolWrapper(),
        ModelSelectToolWrapper(),
    ]

    # Concrete tools (file I/O, git, planning) — may fail gracefully
    for factory in (_make_file_tool, _make_git_tool, _make_plan_tool):
        tool = factory()
        if tool is not None:
            tools.append(tool)

    # Register each tool
    for tool in tools:
        try:
            registry.register(tool)
            logger.info("Registered tool: %s", tool.metadata.name)
        except Exception as e:
            logger.error("Failed to register tool %s: %s", tool.metadata.name, e)

    return len(tools)


def _auto_register() -> bool:
    """Register all tools with the tool registry.

    Returns:
        True if registration completed without error, False on failure.
        A False return allows the next call to ``ensure_tools_registered``
        to retry rather than permanently skipping registration.
    """
    try:
        count = register_all_tools()
        logger.info("Auto-registered %s tools", count)
        return True
    except Exception as exc:
        logger.warning(
            "Auto-registration failed — tools not registered, next call will retry: %s",
            exc,
        )
        return False


# Flag is only set True after a successful registration attempt so that
# a transient failure on the first call allows a retry on the next call.
_auto_registered: bool = False
_auto_register_lock: threading.Lock = threading.Lock()


def ensure_tools_registered() -> None:
    """Register tools at most once, retrying after transient failures.

    Uses double-checked locking for thread safety.  Unlike a strict
    "exactly-once" pattern, the flag is only committed on success so that
    a transient registration error on one call does not permanently prevent
    tools from being registered on a subsequent call.
    """
    global _auto_registered
    if not _auto_registered:
        with _auto_register_lock:
            if not _auto_registered:
                # Only mark as registered when the attempt actually succeeded.
                _auto_registered = _auto_register()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Manual registration for testing
    count = register_all_tools()
    logger.info("Registered %d tools", count)

    # List registered tools
    registry = get_tool_registry()
    logger.info("Registered tools:")
    for tool in registry.list_tools():
        logger.info("  - %s: %s", tool.metadata.name, tool.metadata.description)
