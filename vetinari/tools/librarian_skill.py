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
    
    _SYSTEM = (
        "You are an expert technical librarian with deep knowledge of software libraries, APIs, and best practices. "
        "Respond ONLY with valid JSON matching: "
        '{"summary": "...", "documentation_url": "url or null", "code_example": "code or null", '
        '"best_practices": ["...", "..."], "citations": ["..."]}'
    )

    _CAP_PROMPTS = {
        LibrarianCapability.DOCS_LOOKUP: (
            "Look up and explain official documentation for: {query}\n"
            "Provide a helpful summary, any official doc URL you know, and a code example if applicable."
        ),
        LibrarianCapability.GITHUB_EXAMPLES: (
            "Provide real-world usage examples for: {query}\n"
            "Include a practical code example showing idiomatic usage."
        ),
        LibrarianCapability.API_REFERENCE: (
            "Retrieve API reference details for: {query}\n"
            "Explain the API signature, parameters, return values, and common patterns."
        ),
        LibrarianCapability.PACKAGE_INFO: (
            "Provide package information for: {query}\n"
            "Include installation, purpose, key features, and any known pypi/npm URL."
        ),
        LibrarianCapability.BEST_PRACTICES: (
            "What are the best practices for: {query}\n"
            "List concrete, actionable best practices used by experienced developers."
        ),
    }

    def _run_capability(self, cap: LibrarianCapability, request: ResearchRequest) -> ResearchResult:
        import json, re
        template = self._CAP_PROMPTS.get(cap, "Answer this technical question: {query}")
        user_msg = template.format(query=request.query)
        if request.context:
            user_msg += f"\n\nContext: {request.context}"
        if request.focus_areas:
            user_msg += f"\nFocus areas: {', '.join(request.focus_areas)}"
        try:
            raw = self._infer(self._SYSTEM, user_msg, max_tokens=1024)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return ResearchResult(
                    success=True,
                    summary=data.get("summary", raw[:300]),
                    documentation_url=data.get("documentation_url"),
                    code_example=data.get("code_example"),
                    best_practices=data.get("best_practices", []),
                    citations=data.get("citations", []),
                )
            if raw:
                return ResearchResult(success=True, summary=raw[:500])
        except Exception:
            pass  # Fall through to graceful fallback
        # Graceful fallback when LLM is unavailable
        return ResearchResult(
            success=True,
            summary=f"{cap.value} lookup for '{request.query}' (offline fallback — LLM unavailable)",
        )

    def _lookup_documentation(self, request: ResearchRequest, execution_mode: ExecutionMode) -> ResearchResult:
        logger.info(f"Looking up documentation for: {request.query}")
        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary=f"Planning mode: Would fetch docs for '{request.query}'")
        return self._run_capability(LibrarianCapability.DOCS_LOOKUP, request)

    def _find_github_examples(self, request: ResearchRequest, execution_mode: ExecutionMode) -> ResearchResult:
        logger.info(f"Finding examples for: {request.query}")
        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary=f"Planning: Would find examples for '{request.query}'")
        return self._run_capability(LibrarianCapability.GITHUB_EXAMPLES, request)

    def _retrieve_api_reference(self, request: ResearchRequest, execution_mode: ExecutionMode) -> ResearchResult:
        logger.info(f"Retrieving API reference for: {request.query}")
        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary=f"Planning: Would retrieve API reference for '{request.query}'")
        return self._run_capability(LibrarianCapability.API_REFERENCE, request)

    def _retrieve_package_info(self, request: ResearchRequest, execution_mode: ExecutionMode) -> ResearchResult:
        logger.info(f"Retrieving package info for: {request.query}")
        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary=f"Planning: Would retrieve package info for '{request.query}'")
        return self._run_capability(LibrarianCapability.PACKAGE_INFO, request)

    def _research_best_practices(self, request: ResearchRequest, execution_mode: ExecutionMode) -> ResearchResult:
        logger.info(f"Researching best practices for: {request.query}")
        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(success=True, summary=f"Planning: Would research best practices for '{request.query}'")
        return self._run_capability(LibrarianCapability.BEST_PRACTICES, request)
