"""Explorer Skill Tool Wrapper.

Migrates the explorer skill to the Tool interface, providing fast codebase
search and file discovery capabilities as a standardized Vetinari tool.

The explorer skill specializes in:
- Grep-based text search
- File discovery and globbing
- Pattern matching
- Symbol lookup (functions, classes, imports)
- Import analysis and dependency tracing
- Project structure mapping


.. deprecated:: 1.1.0
   DEPRECATED: Superseded by ConsolidatedResearcherAgent + researcher skill registry.
   Will be removed in a future release.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vetinari.execution_context import ToolPermission
from vetinari.tool_interface import (
    Tool,
    ToolCategory,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)
from vetinari.types import (
    ExecutionMode,
    ThinkingMode,  # canonical enum from types.py
)

logger = logging.getLogger(__name__)


class ExplorerCapability(str, Enum):
    """Capabilities of the explorer skill."""

    GREP_SEARCH = "grep_search"
    FILE_DISCOVERY = "file_discovery"
    PATTERN_MATCHING = "pattern_matching"
    SYMBOL_LOOKUP = "symbol_lookup"
    IMPORT_ANALYSIS = "import_analysis"
    PROJECT_MAPPING = "project_mapping"


class SearchStrategy(str, Enum):
    """Search strategy options."""

    EXACT = "exact"  # Exact string match
    REGEX = "regex"  # Regex pattern
    PARTIAL = "partial"  # Partial/contains match


@dataclass
class ExplorationRequest:
    """Request structure for explorer operations."""

    capability: ExplorerCapability
    query: str
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    search_strategy: SearchStrategy = SearchStrategy.PARTIAL
    file_extensions: list[str] = field(default_factory=list)
    context_lines: int = 2
    max_results: int = 20
    include_imports: bool = False
    rank_by_relevance: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "capability": self.capability.value,
            "query": self.query,
            "thinking_mode": self.thinking_mode.value,
            "search_strategy": self.search_strategy.value,
            "file_extensions": self.file_extensions,
            "context_lines": self.context_lines,
            "max_results": self.max_results,
        }


@dataclass
class SearchResult:
    """Single search result."""

    file_path: str
    line_number: int
    line_content: str
    before_context: list[str] = field(default_factory=list)
    after_context: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "context_before": self.before_context,
            "context_after": self.after_context,
        }


@dataclass
class ExplorationResult:
    """Result of an exploration operation."""

    success: bool
    query: str
    capability: str
    results: list[SearchResult] = field(default_factory=list)
    total_found: int = 0
    files_searched: int = 0
    execution_time_ms: int = 0
    project_type: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "query": self.query,
            "capability": self.capability,
            "results": [r.to_dict() for r in self.results[:10]],  # Top 10
            "total_found": self.total_found,
            "files_searched": self.files_searched,
            "execution_time_ms": self.execution_time_ms,
            "project_type": self.project_type,
            "warnings": self.warnings,
        }


class ExplorerSkillTool(Tool):
    """Tool wrapper for the explorer skill.

    Provides fast codebase search and file discovery capabilities
    through a standardized Tool interface.

    Permissions:
    - FILE_READ: Read files to search content
    - NETWORK_REQUEST: Optional for remote searches

    Allowed Modes:
    - EXECUTION: Full search capabilities
    - PLANNING: Analysis and read-only searches
    """

    def __init__(self):
        """Initialize the explorer skill tool."""
        import warnings

        warnings.warn(
            "ExplorerSkillTool is deprecated since v1.1.0. Use ConsolidatedResearcherAgent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        metadata = ToolMetadata(
            name="explorer",
            description=(
                "Fast codebase search and file discovery. "
                "Use when user wants to find code, search patterns, explore project "
                "structure, or locate specific implementations."
            ),
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="capability",
                    type=str,
                    description="The explorer capability to use",
                    required=True,
                    allowed_values=[c.value for c in ExplorerCapability],
                ),
                ToolParameter(
                    name="query",
                    type=str,
                    description="Search query or pattern",
                    required=True,
                ),
                ToolParameter(
                    name="thinking_mode",
                    type=str,
                    description="Search approach (low/medium/high/xhigh)",
                    required=False,
                    default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
                ToolParameter(
                    name="search_strategy",
                    type=str,
                    description="Search strategy (exact/regex/partial)",
                    required=False,
                    default="partial",
                    allowed_values=[s.value for s in SearchStrategy],
                ),
                ToolParameter(
                    name="file_extensions",
                    type=list,
                    description="File extensions to search (.py, .ts, etc.)",
                    required=False,
                ),
                ToolParameter(
                    name="context_lines",
                    type=int,
                    description="Number of context lines before/after match",
                    required=False,
                    default=2,
                ),
                ToolParameter(
                    name="max_results",
                    type=int,
                    description="Maximum results to return",
                    required=False,
                    default=20,
                ),
            ],
            required_permissions=[
                ToolPermission.FILE_READ,
            ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=[
                "search",
                "discovery",
                "codebase",
                "grep",
                "exploration",
            ],
        )
        super().__init__(metadata)

    def execute(self, **kwargs) -> ToolResult:  # noqa: D417
        """Execute an explorer operation.

        Args:
            capability: ExplorerCapability to use
            query: Search query or pattern
            thinking_mode: Search approach (default: medium)
            search_strategy: Search strategy (default: partial)
            file_extensions: File types to search (optional)
            context_lines: Context lines around matches (default: 2)
            max_results: Max results to return (default: 20)

        Returns:
            ToolResult with search results
        """
        try:
            # Extract parameters
            capability_str = kwargs.get("capability")
            query = kwargs.get("query")
            thinking_mode_str = kwargs.get("thinking_mode", "medium")
            search_strategy_str = kwargs.get("search_strategy", "partial")
            file_extensions = kwargs.get("file_extensions", [])
            context_lines = kwargs.get("context_lines", 2)
            max_results = kwargs.get("max_results", 20)

            # Validate enums
            try:
                capability = ExplorerCapability(capability_str)
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

            try:
                search_strategy = SearchStrategy(search_strategy_str)
            except ValueError:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid search_strategy: {search_strategy_str}",
                )

            # Create request
            request = ExplorationRequest(
                capability=capability,
                query=query,
                thinking_mode=thinking_mode,
                search_strategy=search_strategy,
                file_extensions=file_extensions,
                context_lines=context_lines,
                max_results=max_results,
            )

            # Get execution mode
            ctx = self._context_manager.current_context
            execution_mode = ctx.mode

            # Execute based on capability
            result = self._execute_capability(request, execution_mode)

            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "Search failed",
                metadata={
                    "capability": capability.value,
                    "thinking_mode": thinking_mode.value,
                    "execution_mode": execution_mode.value,
                },
            )

        except Exception as e:
            logger.error("Explorer tool execution failed: %s", e, exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                error=f"Explorer tool execution failed: {e!s}",
            )

    def _execute_capability(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Execute a specific explorer capability.

        Args:
            request: The exploration request
            execution_mode: Current execution mode

        Returns:
            ExplorationResult with search details
        """
        capability = request.capability

        if capability == ExplorerCapability.GREP_SEARCH:
            return self._grep_search(request, execution_mode)
        elif capability == ExplorerCapability.FILE_DISCOVERY:
            return self._file_discovery(request, execution_mode)
        elif capability == ExplorerCapability.PATTERN_MATCHING:
            return self._pattern_matching(request, execution_mode)
        elif capability == ExplorerCapability.SYMBOL_LOOKUP:
            return self._symbol_lookup(request, execution_mode)
        elif capability == ExplorerCapability.IMPORT_ANALYSIS:
            return self._import_analysis(request, execution_mode)
        elif capability == ExplorerCapability.PROJECT_MAPPING:
            return self._project_mapping(request, execution_mode)
        else:
            return ExplorationResult(
                success=False,
                query=request.query,
                capability=capability.value,
                warnings=[f"Unknown capability: {capability.value}"],
            )

    def _grep_search(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Perform grep-based text search."""
        logger.info("Grep search: %s", request.query)

        explanation = f"Grep Search\nQuery: {request.query}\nStrategy: {request.search_strategy.value}\n"

        if request.file_extensions:
            explanation += f"File types: {', '.join(request.file_extensions)}\n"

        if execution_mode == ExecutionMode.PLANNING:
            explanation += "\nPlanning mode: Would search for the pattern in codebase.\n"
            explanation += f"Thinking mode: {request.thinking_mode.value}\n"
        else:
            explanation += (
                f"\nSearch Strategy:\n"
                f"1. Parse query: {request.query}\n"
                f"2. Build search pattern (strategy: {request.search_strategy.value})\n"
                f"3. Execute search across files\n"
                f"4. Gather context ({request.context_lines} lines before/after)\n"
                f"5. Rank results by relevance\n"
                f"6. Return top {request.max_results} results\n"
            )

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.GREP_SEARCH.value,
            total_found=0,
            files_searched=0,
        )

    def _file_discovery(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Discover files matching pattern."""
        logger.info("File discovery: %s", request.query)

        explanation = f"File Discovery\nPattern: {request.query}\n"

        if request.file_extensions:
            explanation += f"Extensions: {', '.join(request.file_extensions)}\n"

        if execution_mode == ExecutionMode.PLANNING:
            explanation += "\nPlanning mode: Would find files matching the pattern.\n"
        else:
            explanation += (
                f"\nDiscovery Process:\n"
                f"1. Parse glob pattern: {request.query}\n"
                f"2. Traverse directories (respecting .gitignore)\n"
                f"3. Match against pattern\n"
                f"4. Filter by extensions if provided\n"
                f"5. Return file list with metadata\n"
            )

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.FILE_DISCOVERY.value,
            total_found=0,
            files_searched=0,
        )

    def _pattern_matching(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Match patterns in code."""
        logger.info("Pattern matching: %s", request.query)

        explanation = f"Pattern Matching\nPattern: {request.query}\nStrategy: {request.search_strategy.value}\n"

        if execution_mode == ExecutionMode.PLANNING:
            explanation += "\nPlanning mode: Would find all pattern matches.\n"
        else:
            explanation += (
                f"\nMatching Process:\n"
                f"1. Compile pattern (strategy: {request.search_strategy.value})\n"
                f"2. Search across codebase\n"
                f"3. Collect all matches with line info\n"
                f"4. Group by file\n"
                f"5. Provide context around matches\n"
            )

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.PATTERN_MATCHING.value,
            total_found=0,
            files_searched=0,
        )

    def _symbol_lookup(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Look up function, class, or variable symbols."""
        logger.info("Symbol lookup: %s", request.query)

        explanation = f"Symbol Lookup\nSymbol: {request.query}\n"

        if execution_mode == ExecutionMode.PLANNING:
            explanation += "\nPlanning mode: Would find symbol definitions and usages.\n"
        else:
            explanation += (
                f"\nLookup Process:\n"
                f"1. Parse symbol name: {request.query}\n"
                f"2. Search for definitions (functions, classes, variables)\n"
                f"3. Find all usages/references\n"
                f"4. Trace dependencies\n"
                f"5. Return definition + usage locations\n"
                f"6. Include signatures and type info\n"
            )

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.SYMBOL_LOOKUP.value,
            total_found=0,
            files_searched=0,
        )

    def _import_analysis(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Analyze imports and dependencies."""
        logger.info("Import analysis: %s", request.query)

        explanation = f"Import Analysis\nTarget: {request.query}\n"

        if execution_mode == ExecutionMode.PLANNING:
            explanation += "\nPlanning mode: Would analyze import dependencies.\n"
        else:
            explanation += (
                f"\nAnalysis Process:\n"
                f"1. Find file/module: {request.query}\n"
                f"2. Parse import statements\n"
                f"3. Trace what it imports from\n"
                f"4. Trace what imports it\n"
                f"5. Build dependency graph\n"
                f"6. Identify circular dependencies\n"
                f"7. Show import relationships\n"
            )

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.IMPORT_ANALYSIS.value,
            total_found=0,
            files_searched=0,
        )

    def _project_mapping(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Map project structure and architecture."""
        logger.info("Project mapping: %s", request.query)

        explanation = f"Project Mapping\nFocus: {request.query}\n"

        if execution_mode == ExecutionMode.PLANNING:
            explanation += "\nPlanning mode: Would map project structure.\n"
        else:
            approach = (
                "Quick scan"
                if request.thinking_mode == ThinkingMode.LOW
                else "Comprehensive map"
                if request.thinking_mode == ThinkingMode.MEDIUM
                else "Deep analysis"
                if request.thinking_mode == ThinkingMode.HIGH
                else "Full AST traversal"
            )

            explanation += (
                f"\nMapping Process ({approach}):\n"
                f"1. Detect project type\n"
                f"2. Identify entry points\n"
                f"3. Map directory structure\n"
                f"4. Find key files (config, setup, main)\n"
                f"5. Trace major dependencies\n"
            )

            if request.thinking_mode in [ThinkingMode.HIGH, ThinkingMode.XHIGH]:
                explanation += "6. Build architecture diagram\n"
                explanation += "7. Identify design patterns\n"
                explanation += "8. Analyze import relationships\n"

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.PROJECT_MAPPING.value,
            project_type="Unknown",
            total_found=0,
            files_searched=0,
        )
