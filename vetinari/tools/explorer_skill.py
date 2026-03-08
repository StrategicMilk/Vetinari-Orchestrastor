"""
Explorer Skill Tool Wrapper

Migrates the explorer skill to the Tool interface, providing fast codebase
search and file discovery capabilities as a standardized Vetinari tool.

The explorer skill specializes in:
- Grep-based text search
- File discovery and globbing
- Pattern matching
- Symbol lookup (functions, classes, imports)
- Import analysis and dependency tracing
- Project structure mapping
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
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
from vetinari.tools.output_validation import validate_output

logger = logging.getLogger(__name__)


class ExplorerCapability(str, Enum):
    """Capabilities of the explorer skill."""
    GREP_SEARCH = "grep_search"
    FILE_DISCOVERY = "file_discovery"
    PATTERN_MATCHING = "pattern_matching"
    SYMBOL_LOOKUP = "symbol_lookup"
    IMPORT_ANALYSIS = "import_analysis"
    PROJECT_MAPPING = "project_mapping"


class ThinkingMode(str, Enum):
    """Thinking modes for search approach."""
    LOW = "low"              # Quick grep search, return first results
    MEDIUM = "medium"        # Comprehensive search with context
    HIGH = "high"            # Full project mapping with dependencies
    XHIGH = "xhigh"          # Deep code analysis with AST traversal


class SearchStrategy(str, Enum):
    """Search strategy options."""
    EXACT = "exact"          # Exact string match
    REGEX = "regex"          # Regex pattern
    PARTIAL = "partial"      # Partial/contains match


@dataclass
class ExplorationRequest:
    """Request structure for explorer operations."""
    capability: ExplorerCapability
    query: str
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    search_strategy: SearchStrategy = SearchStrategy.PARTIAL
    file_extensions: List[str] = field(default_factory=list)
    context_lines: int = 2
    max_results: int = 20
    include_imports: bool = False
    rank_by_relevance: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
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
    before_context: List[str] = field(default_factory=list)
    after_context: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
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
    results: List[SearchResult] = field(default_factory=list)
    total_found: int = 0
    files_searched: int = 0
    execution_time_ms: int = 0
    project_type: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
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
    """
    Tool wrapper for the explorer skill.
    
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
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute an explorer operation.
        
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

            # Validate output before returning
            validation = validate_output(
                result, required_fields=["success"]
            )
            if not validation["valid"]:
                logger.warning("Explorer output validation failed: %s", validation["errors"])

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
            logger.error(f"Explorer tool execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                error=f"Explorer tool execution failed: {str(e)}",
            )
    
    def _execute_capability(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """
        Execute a specific explorer capability.
        
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
    
    def _try_grep_search(
        self,
        request: ExplorationRequest,
    ) -> ExplorationResult:
        """Attempt real grep search via GrepContext backend."""
        try:
            from vetinari.search.grep_context import get_grep_context
            import os
            import time

            grep = get_grep_context()
            root = os.getcwd()

            # Collect files to search
            target_extensions = set(request.file_extensions) if request.file_extensions else {
                '.py', '.js', '.ts', '.tsx', '.jsx'
            }
            file_paths = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('venv', 'node_modules', '__pycache__')]
                for fn in filenames:
                    from pathlib import Path as _Path
                    if _Path(fn).suffix.lower() in target_extensions:
                        file_paths.append(os.path.join(dirpath, fn))
                        if len(file_paths) >= 500:
                            break
                if len(file_paths) >= 500:
                    break

            start = time.time()
            matches = grep.extract_patterns(
                file_paths,
                [request.query],
                context_lines=request.context_lines,
                max_matches=request.max_results,
            )
            elapsed_ms = int((time.time() - start) * 1000)

            results = [
                SearchResult(
                    file_path=m.file_path,
                    line_number=m.line_number,
                    line_content=m.line_content,
                    before_context=m.context_before,
                    after_context=m.context_after,
                )
                for m in matches
            ]

            return ExplorationResult(
                success=True,
                query=request.query,
                capability=request.capability.value,
                results=results,
                total_found=len(results),
                files_searched=len(file_paths),
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.debug(f"Grep search backend failed: {e}")
            return None

    def _grep_search(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Perform grep-based text search."""
        logger.info(f"Grep search: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.GREP_SEARCH.value,
                warnings=["Planning mode: search not executed"],
            )

        result = self._try_grep_search(request)
        if result is not None:
            return result

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.GREP_SEARCH.value,
            total_found=0,
            files_searched=0,
            warnings=["Search backend unavailable; no results returned"],
        )

    def _file_discovery(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Discover files matching pattern."""
        logger.info(f"File discovery: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.FILE_DISCOVERY.value,
                warnings=["Planning mode: discovery not executed"],
            )

        try:
            import os
            import fnmatch
            import time

            root = os.getcwd()
            target_extensions = set(request.file_extensions) if request.file_extensions else None
            start = time.time()
            found_files = []

            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('venv', 'node_modules', '__pycache__')]
                for fn in filenames:
                    if target_extensions:
                        from pathlib import Path as _Path
                        if _Path(fn).suffix.lower() not in target_extensions:
                            continue
                    if fnmatch.fnmatch(fn, request.query) or request.query.lower() in fn.lower():
                        filepath = os.path.join(dirpath, fn)
                        found_files.append(SearchResult(
                            file_path=filepath,
                            line_number=0,
                            line_content=fn,
                        ))
                        if len(found_files) >= request.max_results:
                            break
                if len(found_files) >= request.max_results:
                    break

            elapsed_ms = int((time.time() - start) * 1000)
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.FILE_DISCOVERY.value,
                results=found_files,
                total_found=len(found_files),
                files_searched=0,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            logger.debug(f"File discovery failed: {e}")
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.FILE_DISCOVERY.value,
                total_found=0,
                files_searched=0,
                warnings=[f"File discovery failed: {e}"],
            )

    def _pattern_matching(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Match patterns in code."""
        logger.info(f"Pattern matching: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.PATTERN_MATCHING.value,
                warnings=["Planning mode: matching not executed"],
            )

        # Reuse grep search for pattern matching
        result = self._try_grep_search(request)
        if result is not None:
            result.capability = ExplorerCapability.PATTERN_MATCHING.value
            return result

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.PATTERN_MATCHING.value,
            total_found=0,
            files_searched=0,
            warnings=["Search backend unavailable; no results returned"],
        )

    def _symbol_lookup(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Look up function, class, or variable symbols."""
        logger.info(f"Symbol lookup: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.SYMBOL_LOOKUP.value,
                warnings=["Planning mode: lookup not executed"],
            )

        # Use grep to find definitions matching the symbol name
        import copy
        symbol_request = copy.copy(request)
        symbol_request.query = rf"(def|class|function|const|let|var)\s+{request.query}\b"
        symbol_request.search_strategy = SearchStrategy.REGEX

        result = self._try_grep_search(symbol_request)
        if result is not None:
            result.capability = ExplorerCapability.SYMBOL_LOOKUP.value
            result.query = request.query
            return result

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.SYMBOL_LOOKUP.value,
            total_found=0,
            files_searched=0,
            warnings=["Search backend unavailable; no results returned"],
        )

    def _import_analysis(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Analyze imports and dependencies."""
        logger.info(f"Import analysis: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.IMPORT_ANALYSIS.value,
                warnings=["Planning mode: analysis not executed"],
            )

        # Search for import statements referencing the query
        import copy
        import_request = copy.copy(request)
        import_request.query = rf"(import|from)\s+.*{request.query}"
        import_request.search_strategy = SearchStrategy.REGEX

        result = self._try_grep_search(import_request)
        if result is not None:
            result.capability = ExplorerCapability.IMPORT_ANALYSIS.value
            result.query = request.query
            return result

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.IMPORT_ANALYSIS.value,
            total_found=0,
            files_searched=0,
            warnings=["Search backend unavailable; no results returned"],
        )

    def _project_mapping(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Map project structure and architecture."""
        logger.info(f"Project mapping: {request.query}")

        if execution_mode == ExecutionMode.PLANNING:
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.PROJECT_MAPPING.value,
                warnings=["Planning mode: mapping not executed"],
            )

        try:
            import os
            import time

            root = os.getcwd()
            start = time.time()
            project_type = None

            # Detect project type by key files
            type_indicators = {
                "package.json": "Node.js",
                "setup.py": "Python",
                "pyproject.toml": "Python",
                "Cargo.toml": "Rust",
                "go.mod": "Go",
                "pom.xml": "Java/Maven",
                "build.gradle": "Java/Gradle",
            }
            for indicator, ptype in type_indicators.items():
                if os.path.exists(os.path.join(root, indicator)):
                    project_type = ptype
                    break

            # Collect key files
            key_files = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('venv', 'node_modules', '__pycache__')]
                for fn in filenames:
                    if fn in type_indicators or fn in ('README.md', 'Makefile', 'Dockerfile', '.gitignore'):
                        key_files.append(SearchResult(
                            file_path=os.path.join(dirpath, fn),
                            line_number=0,
                            line_content=fn,
                        ))
                if len(key_files) >= request.max_results:
                    break

            elapsed_ms = int((time.time() - start) * 1000)
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.PROJECT_MAPPING.value,
                results=key_files,
                total_found=len(key_files),
                files_searched=0,
                execution_time_ms=elapsed_ms,
                project_type=project_type,
            )
        except Exception as e:
            logger.debug(f"Project mapping failed: {e}")
            return ExplorationResult(
                success=True,
                query=request.query,
                capability=ExplorerCapability.PROJECT_MAPPING.value,
                project_type=None,
                total_found=0,
                files_searched=0,
                warnings=[f"Project mapping failed: {e}"],
            )
