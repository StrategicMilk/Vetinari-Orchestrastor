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

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    
    # ------------------------------------------------------------------ helpers

    def _get_root(self) -> Path:
        """Return project root for file searches."""
        try:
            from vetinari.config import get_project_root
            return get_project_root()
        except Exception:
            return Path(".").resolve()

    def _iter_files(self, root: Path, extensions: List[str]) -> List[Path]:
        """Recursively yield files, respecting .gitignore-style skip dirs."""
        skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                     ".mypy_cache", ".pytest_cache", "dist", "build", ".claude"}
        exts = set(e if e.startswith(".") else f".{e}" for e in extensions) if extensions else None
        results = []
        for path in root.rglob("*"):
            if path.is_file():
                if any(part in skip_dirs for part in path.parts):
                    continue
                if exts and path.suffix not in exts:
                    continue
                results.append(path)
        return results

    def _build_pattern(self, query: str, strategy: SearchStrategy) -> re.Pattern:
        """Build a compiled regex from the query and strategy."""
        if strategy == SearchStrategy.EXACT:
            return re.compile(re.escape(query))
        elif strategy == SearchStrategy.REGEX:
            return re.compile(query)
        else:  # PARTIAL
            return re.compile(re.escape(query), re.IGNORECASE)

    def _search_file(self, file_path: Path, pattern: re.Pattern,
                     context_lines: int, max_results: int,
                     found: List[SearchResult]) -> int:
        """Search a single file and append matches to found. Returns match count."""
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return 0
        count = 0
        for i, line in enumerate(lines):
            if len(found) >= max_results:
                break
            if pattern.search(line):
                before = lines[max(0, i - context_lines): i]
                after = lines[i + 1: i + 1 + context_lines]
                found.append(SearchResult(
                    file_path=str(file_path),
                    line_number=i + 1,
                    line_content=line,
                    before_context=before,
                    after_context=after,
                ))
                count += 1
        return count

    # ------------------------------------------------------------------ capabilities

    def _grep_search(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Perform grep-based text search."""
        logger.info(f"Grep search: {request.query}")
        t0 = time.time()
        root = self._get_root()
        extensions = request.file_extensions or [".py", ".ts", ".js", ".md", ".yaml", ".json"]
        files = self._iter_files(root, extensions)
        pattern = self._build_pattern(request.query, request.search_strategy)
        found: List[SearchResult] = []
        for fp in files:
            if len(found) >= request.max_results:
                break
            self._search_file(fp, pattern, request.context_lines, request.max_results, found)
        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.GREP_SEARCH.value,
            results=found,
            total_found=len(found),
            files_searched=len(files),
            execution_time_ms=int((time.time() - t0) * 1000),
        )

    def _file_discovery(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Discover files matching a glob pattern or name fragment."""
        logger.info(f"File discovery: {request.query}")
        t0 = time.time()
        root = self._get_root()
        query = request.query.strip()
        # Determine if it's a glob pattern or a plain name fragment
        is_glob = any(c in query for c in ("*", "?", "[", "]"))
        results: List[SearchResult] = []
        if is_glob:
            matches = list(root.rglob(query))
        else:
            # Treat as case-insensitive substring of file name
            matches = [p for p in root.rglob("*") if query.lower() in p.name.lower()]
        # Filter hidden / cache dirs
        skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                     ".mypy_cache", ".pytest_cache", "dist", "build", ".claude"}
        ext_filter = set(
            e if e.startswith(".") else f".{e}" for e in request.file_extensions
        ) if request.file_extensions else None
        for p in matches:
            if len(results) >= request.max_results:
                break
            if any(part in skip_dirs for part in p.parts):
                continue
            if ext_filter and p.suffix not in ext_filter:
                continue
            results.append(SearchResult(
                file_path=str(p),
                line_number=0,
                line_content=str(p.relative_to(root)),
            ))
        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.FILE_DISCOVERY.value,
            results=results,
            total_found=len(results),
            files_searched=len(matches),
            execution_time_ms=int((time.time() - t0) * 1000),
        )

    def _pattern_matching(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Match code patterns (delegates to grep with regex strategy)."""
        logger.info(f"Pattern matching: {request.query}")
        # Pattern matching is grep with regex forced
        adjusted = ExplorationRequest(
            capability=request.capability,
            query=request.query,
            thinking_mode=request.thinking_mode,
            search_strategy=SearchStrategy.REGEX,
            file_extensions=request.file_extensions or [".py", ".ts", ".js"],
            context_lines=request.context_lines,
            max_results=request.max_results,
        )
        result = self._grep_search(adjusted, execution_mode)
        result.capability = ExplorerCapability.PATTERN_MATCHING.value
        return result

    def _symbol_lookup(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Look up function, class, or variable definitions."""
        logger.info(f"Symbol lookup: {request.query}")
        t0 = time.time()
        root = self._get_root()
        symbol = request.query.strip()
        # Build pattern: match def/class/var assignment with the symbol name
        py_pattern = re.compile(
            rf"(?:^|\s)(?:def|class|async\s+def)\s+{re.escape(symbol)}\b"
            rf"|^{re.escape(symbol)}\s*=",
            re.MULTILINE,
        )
        usage_pattern = re.compile(rf"\b{re.escape(symbol)}\b")
        files = self._iter_files(root, [".py"])
        found: List[SearchResult] = []
        # First pass: definitions
        for fp in files:
            if len(found) >= request.max_results:
                break
            try:
                lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines):
                if py_pattern.search(line):
                    found.append(SearchResult(
                        file_path=str(fp),
                        line_number=i + 1,
                        line_content=line,
                        before_context=lines[max(0, i - 1): i],
                        after_context=lines[i + 1: i + 1 + request.context_lines],
                    ))
        # Second pass: usages (if medium+ thinking mode)
        if request.thinking_mode in (ThinkingMode.MEDIUM, ThinkingMode.HIGH, ThinkingMode.XHIGH):
            for fp in files:
                if len(found) >= request.max_results:
                    break
                self._search_file(fp, usage_pattern, 1, request.max_results, found)
        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.SYMBOL_LOOKUP.value,
            results=found,
            total_found=len(found),
            files_searched=len(files),
            execution_time_ms=int((time.time() - t0) * 1000),
        )

    def _import_analysis(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Analyze imports of a module or find who imports a given module."""
        logger.info(f"Import analysis: {request.query}")
        t0 = time.time()
        root = self._get_root()
        target = request.query.strip()
        files = self._iter_files(root, [".py"])
        found: List[SearchResult] = []
        # Pattern: import target / from target import ... / from X import target
        import_pattern = re.compile(
            rf"(?:^import\s+{re.escape(target)}"
            rf"|^from\s+{re.escape(target)}\s+import"
            rf"|^from\s+\S+\s+import\s+.*\b{re.escape(target)}\b)",
            re.MULTILINE,
        )
        for fp in files:
            if len(found) >= request.max_results:
                break
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
            except Exception:
                continue
            for i, line in enumerate(lines):
                if import_pattern.search(line):
                    found.append(SearchResult(
                        file_path=str(fp),
                        line_number=i + 1,
                        line_content=line,
                        after_context=lines[i + 1: i + 1 + request.context_lines],
                    ))
        # If target looks like a file path, also parse it with ast to show its own imports
        own_imports: List[str] = []
        if request.thinking_mode in (ThinkingMode.HIGH, ThinkingMode.XHIGH):
            # Try to find the target file
            for fp in files:
                if target.replace(".", "/") in str(fp) or fp.stem == target:
                    try:
                        tree = ast.parse(fp.read_text(encoding="utf-8", errors="replace"))
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.Import, ast.ImportFrom)):
                                if isinstance(node, ast.Import):
                                    own_imports += [alias.name for alias in node.names]
                                elif node.module:
                                    own_imports.append(node.module)
                    except Exception:
                        pass
                    break
        warnings = [f"Own imports: {own_imports}"] if own_imports else []
        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.IMPORT_ANALYSIS.value,
            results=found,
            total_found=len(found),
            files_searched=len(files),
            execution_time_ms=int((time.time() - t0) * 1000),
            warnings=warnings,
        )

    def _project_mapping(
        self,
        request: ExplorationRequest,
        execution_mode: ExecutionMode,
    ) -> ExplorationResult:
        """Map project structure and detect project type."""
        logger.info(f"Project mapping: {request.query}")
        t0 = time.time()
        root = self._get_root()
        skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules",
                     ".mypy_cache", ".pytest_cache", "dist", "build", ".claude"}

        # Detect project type from marker files
        markers: Dict[str, str] = {
            "pyproject.toml": "Python",
            "setup.py": "Python",
            "setup.cfg": "Python",
            "package.json": "Node.js",
            "cargo.toml": "Rust",
            "go.mod": "Go",
            "pom.xml": "Java/Maven",
            "build.gradle": "Java/Gradle",
            "Makefile": "C/C++",
        }
        project_type = "Unknown"
        for marker, ptype in markers.items():
            if (root / marker).exists() or (root / marker.lower()).exists():
                project_type = ptype
                break

        # Collect top-level structure
        results: List[SearchResult] = []
        files_checked = 0
        for item in sorted(root.iterdir()):
            if item.name in skip_dirs or item.name.startswith("."):
                continue
            results.append(SearchResult(
                file_path=str(item),
                line_number=0,
                line_content=f"{'[dir] ' if item.is_dir() else '[file]'} {item.name}",
            ))

        # For MEDIUM+ mode, include subdirectory listing
        if request.thinking_mode in (ThinkingMode.MEDIUM, ThinkingMode.HIGH, ThinkingMode.XHIGH):
            all_files = list(root.rglob("*"))
            files_checked = len(all_files)
            py_files = [f for f in all_files if f.suffix == ".py"
                        and not any(p in skip_dirs for p in f.parts)]
            if py_files and len(results) < request.max_results:
                results.append(SearchResult(
                    file_path=str(root),
                    line_number=0,
                    line_content=f"Python files: {len(py_files)}",
                ))

        return ExplorationResult(
            success=True,
            query=request.query,
            capability=ExplorerCapability.PROJECT_MAPPING.value,
            results=results[:request.max_results],
            total_found=len(results),
            files_searched=files_checked,
            execution_time_ms=int((time.time() - t0) * 1000),
            project_type=project_type,
        )
