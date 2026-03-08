"""
Librarian Skill Tool Wrapper

Migrates the librarian skill to the Tool interface, providing library research,
documentation lookup, and example finding capabilities as a standardized Vetinari tool.

The librarian skill specializes in:
- Documentation lookup (official docs, APIs)
- GitHub examples search
- Package information retrieval
- Best practice guidance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from enum import Enum

from vetinari.tool_interface import (
    Tool,
    ToolMetadata,
    ToolResult,
    ToolParameter,
    ToolCategory,
)
from vetinari.execution_context import ToolPermission, ExecutionMode

# Note: Web fetching capabilities are provided by the system environment

logger = logging.getLogger(__name__)


class LibrarianCapability(str, Enum):
    """Capabilities of the librarian skill."""
    DOCS_LOOKUP = "docs_lookup"
    GITHUB_EXAMPLES = "github_examples"
    API_REFERENCE = "api_reference"
    PACKAGE_INFO = "package_info"
    BEST_PRACTICES = "best_practices"


class ThinkingMode(str, Enum):
    """Thinking modes for research depth."""
    LOW = "low"      # Quick doc lookup, return official docs link
    MEDIUM = "medium"  # Find official docs + key examples
    HIGH = "high"      # Comprehensive research with multiple sources
    XHIGH = "xhigh"    # Deep dive with real-world examples from GitHub


@dataclass
class ResearchRequest:
    """Request structure for librarian operations."""
    capability: LibrarianCapability
    query: str
    context: Optional[str] = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    focus_areas: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "capability": self.capability.value,
            "query": self.query,
            "context": self.context,
            "thinking_mode": self.thinking_mode.value,
            "focus_areas": self.focus_areas,
        }


@dataclass
class ResearchResult:
    """Result of a librarian operation."""
    success: bool
    summary: Optional[str] = None
    documentation_url: Optional[str] = None
    code_example: Optional[str] = None
    best_practices: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "summary": self.summary,
            "documentation_url": self.documentation_url,
            "code_example": self.code_example,
            "best_practices": self.best_practices,
            "citations": self.citations,
        }


class LibrarianSkillTool(Tool):
    """
    Tool wrapper for the librarian skill.
    
    Provides library research, documentation lookup, and example finding
    through a standardized Tool interface.
    
    Permissions:
    - WEB_FETCH: To retrieve external documentation and examples
    - MODEL_INFERENCE: To summarize and synthesize research
    - FILE_READ: To read local reference materials (e.g., doc_sources.md)
    
    Allowed Modes:
    - EXECUTION: Full research capabilities (including web fetching)
    - PLANNING: Analysis only, no external calls
    """
    
    def __init__(self):
        """Initialize the librarian skill tool."""
        metadata = ToolMetadata(
            name="librarian",
            description=(
                "Library research, documentation lookup, and example finding. "
                "Use when user asks about libraries, APIs, frameworks, or needs "
                "real-world code examples."
            ),
            category=ToolCategory.SEARCH_ANALYSIS, # Closest fit for research
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="capability",
                    type=str,
                    description="The librarian capability to use",
                    required=True,
                    allowed_values=[c.value for c in LibrarianCapability],
                ),
                ToolParameter(
                    name="query",
                    type=str,
                    description="The specific library, API, or concept to research",
                    required=True,
                ),
                ToolParameter(
                    name="context",
                    type=str,
                    description="Additional context for the research (e.g., framework version)",
                    required=False,
                ),
                ToolParameter(
                    name="thinking_mode",
                    type=str,
                    description="Research depth (low/medium/high/xhigh)",
                    required=False,
                    default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
                ToolParameter(
                    name="focus_areas",
                    type=list,
                    description="Specific aspects to focus on (e.g., security, performance)",
                    required=False,
                ),
            ],
            required_permissions=[
                ToolPermission.NETWORK_REQUEST,
                ToolPermission.MODEL_INFERENCE,
                ToolPermission.FILE_READ, # For accessing internal guides
            ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=[
                "research",
                "documentation",
                "libraries",
                "examples",
                "api",
            ],
        )
        super().__init__(metadata)
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute a librarian operation (research/lookup).
        
        Args:
            capability: LibrarianCapability to use
            query: What to research
            context: Additional context (optional)
            thinking_mode: Research depth (default: medium)
            focus_areas: Areas to focus on (optional)
            
        Returns:
            ToolResult with research findings
        """
        try:
            # Extract parameters
            capability_str = kwargs.get("capability")
            query = kwargs.get("query")
            context = kwargs.get("context")
            thinking_mode_str = kwargs.get("thinking_mode", "medium")
            focus_areas = kwargs.get("focus_areas", [])
            
            # Validate required parameters
            if not query:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Query parameter is required for librarian operations",
                )
            
            # Convert to enums
            try:
                capability = LibrarianCapability(capability_str)
            except ValueError:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid capability: {capability_str}",
                )
            
            try:
                thinking_mode = ThinkingMode(thinking_mode_str)
            except ValueError:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid thinking_mode: {thinking_mode_str}",
                )
            
            # Create request
            request = ResearchRequest(
                capability=capability,
                query=query,
                context=context,
                thinking_mode=thinking_mode,
                focus_areas=focus_areas,
            )
            
            # Get execution mode
            ctx = self._context_manager.current_context
            execution_mode = ctx.mode
            
            # Execute based on capability
            result = self._execute_capability(request, execution_mode)
            
            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "Librarian research failed",
                metadata={
                    "capability": capability.value,
                    "thinking_mode": thinking_mode.value,
                    "execution_mode": execution_mode.value,
                    "source_used": "webfetch" if execution_mode == ExecutionMode.EXECUTION else "none",
                },
            )
        
        except Exception as e:
            logger.error(f"Librarian tool execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                error=f"Librarian tool execution failed: {str(e)}",
            )
    
    def _execute_capability(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """
        Execute a specific librarian capability.
        """
        capability = request.capability
        
        if capability == LibrarianCapability.DOCS_LOOKUP:
            return self._lookup_documentation(request, execution_mode)
        elif capability == LibrarianCapability.GITHUB_EXAMPLES:
            return self._find_github_examples(request, execution_mode)
        elif capability == LibrarianCapability.API_REFERENCE:
            return self._retrieve_api_reference(request, execution_mode)
        elif capability == LibrarianCapability.PACKAGE_INFO:
            return self._retrieve_package_info(request, execution_mode)
        elif capability == LibrarianCapability.BEST_PRACTICES:
            return self._research_best_practices(request, execution_mode)
        else:
            return ResearchResult(
                success=False,
                summary=f"Unknown capability: {capability.value}",
            )
    
    def _lookup_documentation(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Look up official documentation using web search if available."""
        logger.info(f"Looking up documentation for: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would fetch documentation for '{request.query}' at {request.thinking_mode.value} depth.",
            )

        # Try real web search via the web_search tool
        try:
            from vetinari.tools.web_search_tool import WebSearchTool
            search_tool = WebSearchTool()
            search_result = search_tool.execute(
                query=f"{request.query} official documentation",
                max_results=3,
            )
            if search_result.success and search_result.output:
                results = search_result.output if isinstance(search_result.output, list) else search_result.output.get("results", [])
                urls = [r.get("url", r.get("link", "")) for r in results if isinstance(r, dict)]
                titles = [r.get("title", "") for r in results if isinstance(r, dict)]
                snippets = [r.get("snippet", r.get("description", "")) for r in results if isinstance(r, dict)]

                summary = f"Documentation search for '{request.query}': found {len(results)} results."
                if snippets:
                    summary += f" Top result: {snippets[0][:200]}"

                return ResearchResult(
                    success=True,
                    summary=summary,
                    documentation_url=urls[0] if urls else None,
                    best_practices=[],
                    citations=urls[:3],
                )
        except Exception as e:
            logger.warning(f"Web search unavailable for documentation lookup: {e}")

        # Fallback: search unavailable
        return ResearchResult(
            success=True,
            summary=f"Documentation lookup for '{request.query}': search_unavailable. No live search could be performed.",
            documentation_url=None,
            code_example=None,
            best_practices=[],
            citations=[],
        )
            
    def _find_github_examples(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Find real-world usage examples on GitHub."""
        logger.info(f"Searching GitHub for examples of: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would search GitHub for examples of '{request.query}' using {request.thinking_mode.value} depth.",
            )

        # Try real web search for GitHub examples
        try:
            from vetinari.tools.web_search_tool import WebSearchTool
            search_tool = WebSearchTool()
            search_result = search_tool.execute(
                query=f"{request.query} site:github.com example",
                max_results=3,
            )
            if search_result.success and search_result.output:
                results = search_result.output if isinstance(search_result.output, list) else search_result.output.get("results", [])
                urls = [r.get("url", r.get("link", "")) for r in results if isinstance(r, dict)]
                snippets = [r.get("snippet", r.get("description", "")) for r in results if isinstance(r, dict)]

                summary = f"GitHub examples search for '{request.query}': found {len(results)} results."
                if snippets:
                    summary += f" Top: {snippets[0][:200]}"

                return ResearchResult(
                    success=True,
                    summary=summary,
                    code_example=None,
                    citations=urls[:3],
                    best_practices=[],
                )
        except Exception as e:
            logger.warning(f"Web search unavailable for GitHub examples: {e}")

        # Fallback: search unavailable
        return ResearchResult(
            success=True,
            summary=f"GitHub examples for '{request.query}': search_unavailable. No live search could be performed.",
            code_example=None,
            citations=[],
            best_practices=[],
        )

    def _retrieve_api_reference(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Retrieve specific API reference details."""
        logger.info(f"Retrieving API reference for: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would fetch API reference for '{request.query}'.",
            )

        # Try real web search for API reference
        try:
            from vetinari.tools.web_search_tool import WebSearchTool
            search_tool = WebSearchTool()
            search_result = search_tool.execute(
                query=f"{request.query} API reference",
                max_results=3,
            )
            if search_result.success and search_result.output:
                results = search_result.output if isinstance(search_result.output, list) else search_result.output.get("results", [])
                urls = [r.get("url", r.get("link", "")) for r in results if isinstance(r, dict)]
                snippets = [r.get("snippet", r.get("description", "")) for r in results if isinstance(r, dict)]

                summary = f"API reference search for '{request.query}': found {len(results)} results."
                if snippets:
                    summary += f" Top: {snippets[0][:200]}"

                return ResearchResult(
                    success=True,
                    summary=summary,
                    documentation_url=urls[0] if urls else None,
                    citations=urls[:3],
                )
        except Exception as e:
            logger.warning(f"Web search unavailable for API reference: {e}")

        # Fallback: search unavailable
        return ResearchResult(
            success=True,
            summary=f"API reference for '{request.query}': search_unavailable. No live search could be performed.",
            documentation_url=None,
            citations=[],
        )

    def _retrieve_package_info(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Retrieve package information (e.g., version, dependencies)."""
        logger.info(f"Retrieving package info for: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would retrieve package info for '{request.query}'.",
            )

        package_name = request.query.split()[0].strip()

        # Try real web search for package info
        try:
            from vetinari.tools.web_search_tool import WebSearchTool
            search_tool = WebSearchTool()
            search_result = search_tool.execute(
                query=f"{package_name} package pypi npm",
                max_results=3,
            )
            if search_result.success and search_result.output:
                results = search_result.output if isinstance(search_result.output, list) else search_result.output.get("results", [])
                urls = [r.get("url", r.get("link", "")) for r in results if isinstance(r, dict)]
                snippets = [r.get("snippet", r.get("description", "")) for r in results if isinstance(r, dict)]

                summary = f"Package info search for '{package_name}': found {len(results)} results."
                if snippets:
                    summary += f" Top: {snippets[0][:200]}"

                return ResearchResult(
                    success=True,
                    summary=summary,
                    documentation_url=urls[0] if urls else None,
                    best_practices=[],
                    citations=urls[:3],
                )
        except Exception as e:
            logger.warning(f"Web search unavailable for package info: {e}")

        # Fallback: search unavailable
        return ResearchResult(
            success=True,
            summary=f"Package info for '{package_name}': search_unavailable. No live search could be performed.",
            documentation_url=None,
            best_practices=[],
            citations=[],
        )

    def _research_best_practices(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Research general best practices for a topic."""
        logger.info(f"Researching best practices for: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would research best practices for '{request.query}' using {request.thinking_mode.value} depth.",
            )

        # Try real web search for best practices
        try:
            from vetinari.tools.web_search_tool import WebSearchTool
            search_tool = WebSearchTool()
            search_result = search_tool.execute(
                query=f"{request.query} best practices",
                max_results=3,
            )
            if search_result.success and search_result.output:
                results = search_result.output if isinstance(search_result.output, list) else search_result.output.get("results", [])
                urls = [r.get("url", r.get("link", "")) for r in results if isinstance(r, dict)]
                snippets = [r.get("snippet", r.get("description", "")) for r in results if isinstance(r, dict)]

                summary = f"Best practices search for '{request.query}': found {len(results)} results."
                if snippets:
                    summary += f" Top: {snippets[0][:200]}"

                return ResearchResult(
                    success=True,
                    summary=summary,
                    best_practices=[],
                    citations=urls[:3],
                )
        except Exception as e:
            logger.warning(f"Web search unavailable for best practices: {e}")

        # Fallback: search unavailable
        return ResearchResult(
            success=True,
            summary=f"Best practices for '{request.query}': search_unavailable. No live search could be performed.",
            best_practices=[],
            citations=[],
        )
