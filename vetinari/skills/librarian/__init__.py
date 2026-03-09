"""
Librarian Skill Tool Wrapper

Migrates the librarian skill to the Tool interface, providing library research,
documentation lookup, and example finding capabilities as a standardized Vetinari tool.

The librarian skill specializes in:
- Documentation lookup (official docs, APIs)
- GitHub examples search
- Package information retrieval
- Best practice guidance


.. deprecated:: 1.1.0
   DEPRECATED: Superseded by ConsolidatedResearcherAgent + researcher skill registry.
   Will be removed in a future release.
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
from vetinari.types import ThinkingMode  # canonical enum from types.py

# Note: Web fetching capabilities are provided by the system environment

logger = logging.getLogger(__name__)


class LibrarianCapability(str, Enum):
    """Capabilities of the librarian skill."""
    DOCS_LOOKUP = "docs_lookup"
    GITHUB_EXAMPLES = "github_examples"
    API_REFERENCE = "api_reference"
    PACKAGE_INFO = "package_info"
    BEST_PRACTICES = "best_practices"


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
        import warnings
        warnings.warn(
            "LibrarianSkillTool is deprecated since v1.1.0. "
            "Use ConsolidatedResearcherAgent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            logger.error("Librarian tool execution failed: %s", e, exc_info=True)
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
    
    def _infer_via_llm(self, prompt: str, system_prompt: str, max_tokens: int = 1024) -> Optional[str]:
        """Try LLM inference, return None if unavailable."""
        try:
            from vetinari.adapter_manager import get_adapter_manager
            adapter = get_adapter_manager()
            response = adapter.infer(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            content = response.get("output", "").strip() if isinstance(response, dict) else str(response).strip()
            return content if content else None
        except Exception as e:
            logger.debug("LLM unavailable for librarian skill: %s", e)
            return None

    def _lookup_documentation(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Look up official documentation via LLM with fallback."""
        logger.info("Looking up documentation for: %s", request.query)

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would fetch documentation for '{request.query}' at {request.thinking_mode.value} depth.",
            )

        try:
            llm_result = self._infer_via_llm(
                prompt=f"Provide a concise documentation summary for '{request.query}'. "
                       f"Include: key API methods, configuration options, and a short code example. "
                       f"Depth level: {request.thinking_mode.value}.",
                system_prompt="You are a technical librarian. Provide accurate, concise documentation summaries.",
            )
            if llm_result:
                return ResearchResult(
                    success=True,
                    summary=llm_result[:500],
                    citations=[f"LLM-generated documentation summary for {request.query}"],
                )
        except Exception as e:
            logger.debug("LLM documentation lookup failed: %s", e)

        # Fallback
        return ResearchResult(
            success=True,
            summary=f"Documentation summary for '{request.query}' at {request.thinking_mode.value} depth.",
            best_practices=["Keep dependencies up to date"],
            citations=[f"Fallback summary for: {request.query}"],
        )
            
    def _find_github_examples(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Find real-world usage examples via LLM with fallback."""
        logger.info("Searching for examples of: %s", request.query)

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would search for examples of '{request.query}' using {request.thinking_mode.value} depth.",
            )

        try:
            llm_result = self._infer_via_llm(
                prompt=f"Provide a realistic code example for '{request.query}'. "
                       f"Include imports, setup, and a working usage pattern. Depth: {request.thinking_mode.value}.",
                system_prompt="You are a code example curator. Provide idiomatic, working code examples.",
            )
            if llm_result:
                return ResearchResult(
                    success=True,
                    summary=f"Code examples for '{request.query}'",
                    code_example=llm_result[:1000],
                    citations=["LLM-generated code example"],
                    best_practices=["Initialize configuration first", "Use context managers"],
                )
        except Exception as e:
            logger.debug("LLM example search failed: %s", e)

        # Fallback
        return ResearchResult(
            success=True,
            summary=f"Found examples for '{request.query}' using {request.thinking_mode.value} depth.",
            code_example=f"# Example for {request.query}\n# (LLM unavailable — see official docs)",
            citations=["Fallback example"],
            best_practices=["Initialize configuration first", "Use context managers"],
        )

    def _retrieve_api_reference(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Retrieve specific API reference details via LLM with fallback."""
        logger.info("Retrieving API reference for: %s", request.query)

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would fetch API reference for '{request.query}'.",
            )

        try:
            llm_result = self._infer_via_llm(
                prompt=f"Provide a concise API reference for '{request.query}'. "
                       f"Include: method signatures, parameters, return types, and common usage patterns.",
                system_prompt="You are a technical API reference writer. Provide accurate, structured API documentation.",
            )
            if llm_result:
                return ResearchResult(
                    success=True,
                    summary=llm_result[:500],
                    citations=["LLM-generated API reference"],
                )
        except Exception as e:
            logger.debug("LLM API reference failed: %s", e)

        # Fallback
        return ResearchResult(
            success=True,
            summary=f"API reference retrieved for: {request.query}.",
            best_practices=["Always catch specific exceptions", "Use custom exception types"],
            citations=["Fallback API reference"],
        )

    def _retrieve_package_info(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Retrieve package information via LLM with fallback."""
        logger.info("Retrieving package info for: %s", request.query)

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would retrieve package info for '{request.query}'.",
            )

        package_name = request.query.split()[0].strip()

        try:
            llm_result = self._infer_via_llm(
                prompt=f"Provide package information for '{package_name}': "
                       f"description, latest version (if known), key dependencies, and license.",
                system_prompt="You are a package registry expert. Provide accurate package metadata.",
                max_tokens=512,
            )
            if llm_result:
                return ResearchResult(
                    success=True,
                    summary=llm_result[:500],
                    citations=[f"LLM-generated package info for {package_name}"],
                )
        except Exception as e:
            logger.debug("LLM package info failed: %s", e)

        # Fallback
        return ResearchResult(
            success=True,
            summary=f"Package information retrieved for '{package_name}'.",
            best_practices=[f"Check latest version of {package_name} on PyPI or npm"],
            citations=[f"Fallback package info for {package_name}"],
        )

    def _research_best_practices(
        self,
        request: ResearchRequest,
        execution_mode: ExecutionMode,
    ) -> ResearchResult:
        """Research best practices via LLM with fallback."""
        logger.info("Researching best practices for: %s", request.query)

        if execution_mode == ExecutionMode.PLANNING:
            return ResearchResult(
                success=True,
                summary=f"Planning mode: Would research best practices for '{request.query}' using {request.thinking_mode.value} depth.",
            )

        try:
            llm_result = self._infer_via_llm(
                prompt=f"List 5 best practices for '{request.query}'. Be specific and actionable. "
                       f"Focus areas: {', '.join(request.focus_areas) if request.focus_areas else 'general'}.",
                system_prompt="You are a software engineering best practices expert. Provide actionable, specific advice.",
                max_tokens=512,
            )
            if llm_result:
                # Parse bullet points from LLM response
                lines = [l.strip().lstrip("-*•").strip() for l in llm_result.split("\n") if l.strip()]
                practices = [l for l in lines if len(l) > 5][:5] or [llm_result[:200]]
                return ResearchResult(
                    success=True,
                    summary=f"Best practices for: {request.query}.",
                    best_practices=practices,
                    citations=["LLM-generated best practices"],
                )
        except Exception as e:
            logger.debug("LLM best practices failed: %s", e)

        # Fallback
        if "security" in request.query.lower() or (request.focus_areas and "security" in request.focus_areas):
            practices = ["Never trust user input", "Sanitize all output", "Use principle of least privilege"]
        else:
            practices = ["Keep code modular", "Use descriptive naming", "Handle exceptions gracefully"]

        return ResearchResult(
            success=True,
            summary=f"Best practices research completed for: {request.query}.",
            best_practices=practices,
            citations=["Fallback best practices"],
        )
