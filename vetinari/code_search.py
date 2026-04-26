"""Code Search module.

Provides content-based code search via configurable backends (CocoIndex by
default with a grep fallback), plus structural analysis capabilities via
RepoMap (Aider-style codebase summaries) and ASTIndexer (symbol lookup).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from vetinari.constants import CODE_SEARCH_TIMEOUT, INDEX_BUILD_TIMEOUT, TIMEOUT_SHORT
from vetinari.exceptions import ConfigurationError
from vetinari.repo_map import ASTIndexer, RepoMap, SymbolInfo, get_ast_indexer, get_repo_map
from vetinari.utils.registry import BaseRegistry
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)
_COCOINDEX_PACKAGE_ENV = "VETINARI_COCOINDEX_CODE_PACKAGE"


def _get_pinned_cocoindex_package() -> str | None:
    """Return a pinned CocoIndex package spec or None when not configured."""
    package = os.environ.get(_COCOINDEX_PACKAGE_ENV, "").strip()
    if not package:
        return None
    if "@latest" in package or package.endswith("@"):
        raise ConfigurationError(f"{_COCOINDEX_PACKAGE_ENV} must not use a moving @latest package")
    if "==" not in package and "@" not in package:
        raise ConfigurationError(f"{_COCOINDEX_PACKAGE_ENV} must include an immutable version or digest")
    return package


class SearchBackendStatus(Enum):
    """Search backend status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INDEXING = "indexing"
    ERROR = "error"


@dataclass
class CodeSearchResult:
    """Code search result."""

    file_path: str
    language: str
    content: str
    line_start: int
    line_end: int
    score: float
    context_before: str = ""
    context_after: str = ""

    def __repr__(self) -> str:
        return f"CodeSearchResult(file_path={self.file_path!r}, line_start={self.line_start!r}, score={self.score!r})"

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


class CocoIndexAdapter:
    """Code search backend using CocoIndex with a Python re fallback."""

    name = "cocoindex"

    def __init__(self, root_path: str | None = None, embedding_model: str | None = None):
        self.root_path = root_path or os.getcwd()
        self.embedding_model = embedding_model
        self._status = None

    def _check_availability(self) -> bool:
        try:
            _get_pinned_cocoindex_package()
        except ConfigurationError as exc:
            logger.warning("CocoIndex disabled: %s", exc)
            return False
        if _get_pinned_cocoindex_package() is None:
            return False
        if not shutil.which("uvx"):
            return False
        try:
            result = subprocess.run(["uvx", "--version"], capture_output=True, timeout=TIMEOUT_SHORT)  # noqa: S607 - tool name is intentionally resolved by the runtime environment
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            logger.warning("uvx not available on PATH — code search tool check skipped, search will be unavailable")
            return False

    def search(self, query: str, limit: int = 10, filters: dict | None = None) -> list[CodeSearchResult]:
        """Search the indexed codebase via CocoIndex, falling back to Python re on failure.

        Args:
            query: Natural-language or keyword search query passed to CocoIndex.
            limit: Maximum number of results to return.
            filters: Reserved for future backend-specific filter criteria (currently unused).

        Returns:
            Up to ``limit`` CodeSearchResult objects ranked by relevance, or results from
            the fallback Python re search if CocoIndex is unavailable.
        """
        try:
            package = _get_pinned_cocoindex_package()
        except ConfigurationError as exc:
            logger.warning("CocoIndex indexing disabled: %s", exc)
            return False
        if package is None:
            self._last_backend = "regex_fallback"
            return self._fallback_search(query, limit)

        cmd = [
            "uvx",
            "--prerelease=explicit",
            package,
            "search",
            "--query",
            query,
            "--limit",
            str(limit),
        ]

        try:
            result = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
                cmd, capture_output=True, text=True, timeout=CODE_SEARCH_TIMEOUT, cwd=self.root_path
            )

            if result.returncode != 0:
                self._last_backend = "regex_fallback"
                return self._fallback_search(query, limit)

            self._last_backend = "cocoindex"
            return self._parse_results(result.stdout)

        except Exception as e:
            logger.error("CocoIndex search error: %s", e)
            self._last_backend = "regex_fallback"
            return self._fallback_search(query, limit)

    def _parse_results(self, output: str) -> list[CodeSearchResult]:
        results = []

        try:
            data = json.loads(output)
            if isinstance(data, list):
                for item in data:
                    content = item.get("content", "")
                    if content:
                        results.append(
                            CodeSearchResult(
                                file_path=item.get("file", "unknown"),
                                language=self._detect_language(item.get("file", "")),
                                content=content[:500],
                                line_start=item.get("line_number", 1),
                                line_end=item.get("line_number", 1),
                                score=item.get("score", 0.5),
                            ),
                        )
        except json.JSONDecodeError:
            results.extend(
                CodeSearchResult(
                    file_path="unknown",
                    language="text",
                    content=line[:500],
                    line_start=1,
                    line_end=1,
                    score=0.5,
                )
                for line in output.strip().split("\n")
                if line.strip()
            )

        return results[:10]

    def _detect_language(self, file_path: str) -> str:
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".sql": "sql",
            ".sh": "shell",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "unknown")

    def _fallback_search(self, query: str, limit: int) -> list[CodeSearchResult]:
        """Cross-platform fallback search using Python's os.walk + re."""
        results = []
        target_extensions = {".py", ".js", ".ts", ".tsx", ".jsx"}
        query_pattern = re.compile(re.escape(query), re.IGNORECASE)

        try:
            root = Path(self.root_path) if self.root_path else Path()
            for dirpath, dirnames, filenames in os.walk(root):
                # Skip hidden directories and venv
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "venv"]
                for filename in filenames:
                    if Path(filename).suffix.lower() not in target_extensions:
                        continue
                    filepath = Path(dirpath) / filename
                    try:
                        text = filepath.read_text(encoding="utf-8", errors="ignore")
                        for lineno, line in enumerate(text.splitlines(), 1):
                            if query_pattern.search(line):
                                results.append(
                                    CodeSearchResult(
                                        file_path=str(filepath),
                                        language=self._detect_language(str(filepath)),
                                        content=line.strip()[:200],
                                        line_start=lineno,
                                        line_end=lineno,
                                        score=0.5,
                                    ),
                                )
                                if len(results) >= limit:
                                    return results
                    except (OSError, UnicodeDecodeError):
                        logger.warning("Skipping file %s during fallback search", filepath, exc_info=True)
                        continue
        except (OSError, re.error) as e:
            logger.warning("Fallback search error: %s", e)

        return results[:limit]

    def index_project(self, project_path: str, force: bool = False) -> bool:
        """Build or update the CocoIndex search index for the given project directory.

        Args:
            project_path: Root directory of the project to index.
            force: When True, passes ``--refresh`` to force a full re-index.

        Returns:
            True if CocoIndex exited successfully, False if it was unavailable or errored.
        """
        try:
            package = _get_pinned_cocoindex_package()
        except ConfigurationError as exc:
            logger.warning("CocoIndex indexing disabled: %s", exc)
            return False
        if package is None:
            logger.warning("CocoIndex indexing unavailable: %s is not configured", _COCOINDEX_PACKAGE_ENV)
            return False

        cmd = ["uvx", "--prerelease=explicit", package, "index", "--path", project_path]

        if force:
            cmd.append("--refresh")

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=INDEX_BUILD_TIMEOUT, cwd=project_path)  # noqa: S603 - argv is controlled and shell interpolation is not used
            return result.returncode == 0
        except Exception as e:
            logger.error("CocoIndex index error: %s", e)
            return False

    def get_status(self) -> SearchBackendStatus:
        """Check whether the CocoIndex backend is available on this machine.

        Returns:
            SearchBackendStatus.AVAILABLE if ``uvx`` is present and exits cleanly,
            otherwise SearchBackendStatus.UNAVAILABLE. Result is cached after the
            first check.
        """
        if self._status:
            return self._status

        if self._check_availability():
            self._status = SearchBackendStatus.AVAILABLE
        else:
            self._status = SearchBackendStatus.UNAVAILABLE

        return self._status

    def get_indexed_projects(self) -> list[str]:
        """Find all project directories that contain a CocoIndex index.

        Returns:
            List of directory paths (under root_path) that contain a
            ``.cocoindex_code`` subdirectory.
        """
        indexed = []
        for root, dirs, _files in os.walk(self.root_path):
            if ".cocoindex_code" in dirs:
                indexed.append(root)
        return indexed


class CodeSearchRegistry(BaseRegistry[str, type]):
    """Registry for code search backends."""

    DEFAULT_BACKEND = "cocoindex"

    def __init__(self) -> None:
        super().__init__()
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("cocoindex", CocoIndexAdapter)

    def unregister(self, name: str) -> type | None:  # type: ignore[override]
        """Remove a registered search backend by name.

        The default backend cannot be unregistered.

        Args:
            name: The backend name to remove.

        Returns:
            The removed adapter class, or None if the name was not found or
            is the protected default backend.
        """
        if name == self.DEFAULT_BACKEND:
            return None
        return super().unregister(name)

    def get_adapter(self, name: str | None = None, **kwargs) -> CocoIndexAdapter:
        """Instantiate and return a registered search backend adapter.

        Args:
            name: Backend name to look up; defaults to the registry's DEFAULT_BACKEND.
            **kwargs: Keyword arguments forwarded to the adapter's constructor.

        Returns:
            A fresh instance of the requested adapter.

        Raises:
            ValueError: If ``name`` is not registered in the backend registry.
        """
        name = name or self.DEFAULT_BACKEND
        adapter_class = self.get(name)
        if adapter_class is None:
            raise ConfigurationError(f"Unknown backend: {name}")
        return adapter_class(**kwargs)

    def list_backends(self) -> list[str]:
        """Return the names of all registered search backends.

        Returns:
            List of backend name strings.
        """
        return self.list_keys()

    def get_backend_info(self, name: str) -> dict:
        """Return availability status and indexed projects for a named backend.

        Returns:
            Dictionary with keys ``name``, ``status``, and ``indexed_projects``.
            ``indexed_projects`` is an empty list when the backend is unavailable.
        """
        try:
            adapter = self.get_adapter(name)
            status = adapter.get_status()
            return {
                "name": name,
                "status": status.value,
                "indexed_projects": adapter.get_indexed_projects() if status == SearchBackendStatus.AVAILABLE else [],
            }
        except Exception:  # Broad catch intentional — any backend failure should degrade gracefully
            logger.warning("Backend %s status check failed", name, exc_info=True)
            return {"name": name, "status": "error"}


code_search_registry = CodeSearchRegistry()


# ── Structural analysis helpers ──────────────────────────────────────────────


def get_structural_map(
    root_path: str,
    max_tokens: int = 2000,
    task_description: str | None = None,
    include_private: bool = False,
) -> str:
    """Generate a structural summary of a codebase using RepoMap.

    Produces a concise, token-efficient representation of the codebase
    (module hierarchy, class names, function signatures) suitable for
    providing LLM context without sending entire file contents.

    When ``task_description`` is provided the map is ranked so that modules
    most relevant to the task appear first.

    Args:
        root_path: Root directory of the project to analyse.
        max_tokens: Approximate token budget (~4 chars per token). Controls
            how much of the structure is included before truncation.
        task_description: Optional natural-language description of the current
            task. When given, modules are ranked by relevance to the task.
        include_private: Whether to include private (``_``-prefixed) members
            in the output.

    Returns:
        A multi-line string representing the repository structure, or an
        error message prefixed with ``[RepoMap]`` when the path is invalid.
    """
    mapper: RepoMap = get_repo_map()
    if task_description:
        logger.debug(
            "Generating task-focused repo map for '%s' (max_tokens=%d)",
            task_description[:60],
            max_tokens,
        )
        return mapper.generate_for_task(root_path, task_description, max_tokens)

    logger.debug("Generating repo map for %s (max_tokens=%d)", root_path, max_tokens)
    return mapper.generate(root_path, max_tokens, include_private)


def find_symbol_definitions(
    name: str,
    root_path: str = ".",
) -> list[SymbolInfo]:
    """Find all definitions of a symbol by name using ASTIndexer.

    Indexes the project on first call (with disk caching) then performs an
    exact name lookup against the symbol table. Useful for jump-to-definition
    and impact analysis when the content-based search would return too many
    hits.

    Args:
        name: The symbol name to look up (class, function, or method name).
        root_path: Root directory of the project. Defaults to the current
            working directory.

    Returns:
        A list of :class:`~vetinari.repo_map.SymbolInfo` instances describing
        every location where ``name`` is defined. Returns an empty list when
        the symbol is not found.
    """
    indexer: ASTIndexer = get_ast_indexer(root_path)
    results = indexer.find_symbol(name)
    logger.debug("find_symbol_definitions(%r) -> %d result(s)", name, len(results))
    return results
