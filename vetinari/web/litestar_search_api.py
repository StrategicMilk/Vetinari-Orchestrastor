"""Code-search and search-index endpoints as native Litestar handlers.

Native Litestar equivalents of the routes previously registered by
``search_api``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

Wraps the ``vetinari.code_search`` registry so the web UI can submit
full-text and semantic queries against indexed projects and trigger
re-indexing without importing the registry directly.

Endpoints
---------
    GET  /api/code-search           — Search project code
    POST /api/search/index          — Trigger indexing of a project directory
    GET  /api/search/status         — Return status of all registered backends
    GET  /api/structural-map        — Token-efficient structural summary
    GET  /api/repo-map/usages       — Files that reference a given symbol
    GET  /api/repo-map/symbols      — Symbols extracted from a specific file
    GET  /api/repo-map/import-graph — Intra-project import dependency graph
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import get, post
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_search_handlers() -> list[Any]:
    """Create and return all search API route handlers.

    Returns an empty list when Litestar is not installed so the caller can
    safely extend its handler list without guarding the call.

    Returns:
        List of Litestar route handler objects for all search endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response, success_response

    # -- Code search ---------------------------------------------------------

    @get("/api/code-search")
    async def api_code_search(
        q: str = Parameter(query="q", default=""),
        limit: int = Parameter(query="limit", default=10, ge=1, le=100),
        backend: str = Parameter(query="backend", default="cocoindex"),
    ) -> Any:
        """Search within project code using CocoIndex/ripgrep backends.

        Args:
            q: Search query string (required).
            limit: Maximum number of results to return (default 10).
            backend: Backend identifier to route the query through
                (default ``"cocoindex"``).

        Returns:
            JSON object with ``query``, ``backend``, ``results`` list (each
            item serialized via ``SearchResult.to_dict()``), and ``total``
            count. Returns a 400 error when ``q`` is missing.
        """
        from vetinari.code_search import code_search_registry

        if not q:
            return litestar_error_response("Query required", code=400)

        try:
            adapter = code_search_registry.get_adapter(backend)
            results = adapter.search(q, limit=limit)
        except Exception:
            logger.exception(
                "Code search failed for query %r using backend %r — returning 500",
                q,
                backend,
            )
            return litestar_error_response("Search system unavailable", code=500)

        actual_backend = getattr(adapter, "_last_backend", None)  # noqa: VET112 - empty fallback preserves optional request metadata contract
        if not isinstance(actual_backend, str):
            actual_backend = backend

        return success_response({
            "query": q,
            "backend": actual_backend,
            "backend_requested": backend,
            "results": [r.to_dict() for r in results],
            "total": len(results),
        })

    # -- Search index --------------------------------------------------------

    @post("/api/search/index", guards=[admin_guard])
    async def api_search_index(data: dict[str, Any]) -> Any:
        """Trigger indexing of a project directory using the specified backend.

        Requires admin privileges.

        Args:
            data: JSON request body with the following fields:
                project_path: Absolute or relative path to the project
                    directory to index (required).
                backend: Backend identifier to use for indexing
                    (default ``"cocoindex"``).
                force: When ``True``, forces a full re-index even if an
                    existing index is present (default ``False``).

        Returns:
            JSON object with ``status`` (``"indexing"`` on success,
            ``"error"`` on adapter failure), ``project_path``, and
            ``backend``. Returns a 400 error when ``project_path`` is missing
            or invalid, or 403 when it is outside the projects directory.
        """
        from vetinari.code_search import code_search_registry
        from vetinari.constants import PROJECTS_DIR

        project_path = data.get("project_path")
        backend = data.get("backend", "cocoindex")
        force = data.get("force", False)

        if not project_path:
            return litestar_error_response("project_path required", code=400)
        if not isinstance(project_path, str):
            return litestar_error_response("project_path must be a string", code=400)
        if not isinstance(backend, str) or not backend:
            return litestar_error_response("backend must be a non-empty string", code=400)
        if not isinstance(force, bool):
            return litestar_error_response("force must be a boolean", code=400)

        resolved = Path(project_path).resolve()
        if not resolved.is_relative_to(PROJECTS_DIR.resolve()):
            return litestar_error_response("project_path must be within the projects directory", code=403)
        if not resolved.is_dir():
            return litestar_error_response("project_path must be an existing directory", code=400)

        try:
            adapter = code_search_registry.get_adapter(backend)
            success = adapter.index_project(str(resolved), force=force)
        except Exception as e:
            logger.warning(
                "Search indexing failed for project %r using backend %r — %s",
                project_path,
                backend,
                type(e).__name__,
                exc_info=True,
            )
            return litestar_error_response("Search service unavailable", 503)

        return {
            "status": "indexing" if success else "error",
            "project_path": project_path,
            "backend": backend,
        }

    # -- Search status -------------------------------------------------------

    @get("/api/search/status")
    async def api_search_status() -> Any:
        """Return the status of all registered search backends.

        Iterates over every backend known to ``code_search_registry`` and
        collects its info dict. Backends that raise an exception are reported
        with status ``"error"`` rather than propagating the exception.

        Returns:
            JSON object with a ``backends`` dict (name -> info) and
            ``default_backend`` string indicating which backend is used when
            none is specified by the caller.
        """
        try:
            from vetinari.code_search import code_search_registry
        except Exception:
            logger.exception("Search status check failed — could not import code_search_registry")
            return litestar_error_response("Search system unavailable", code=500)

        backends: dict[str, Any] = {}
        for name in code_search_registry.list_backends():
            try:
                backends[name] = code_search_registry.get_backend_info(name)
            except Exception:
                logger.warning(
                    "Could not retrieve info for search backend %r — reporting error status",
                    name,
                )
                backends[name] = {"name": name, "status": "error"}

        return {"backends": backends, "default_backend": code_search_registry.DEFAULT_BACKEND}

    # -- Structural map ------------------------------------------------------

    @get("/api/structural-map")
    async def api_structural_map(
        project_path: str = Parameter(query="project_path", default=""),
        max_tokens: int = Parameter(query="max_tokens", default=2000, ge=100, le=20000),
        task: str | None = Parameter(query="task", default=None),
        include_private: str = Parameter(query="include_private", default="0"),
    ) -> Any:
        """Return a token-efficient structural summary of a project codebase.

        Uses ``get_structural_map`` (RepoMap) to produce a concise
        module/class/function hierarchy suitable for supplying as LLM context.

        Args:
            project_path: Absolute or relative path to the project directory
                (required).
            max_tokens: Approximate token budget — controls how much structure
                is included (default 2000).
            task: Optional natural-language task description. When provided,
                modules are ranked by relevance to the task.
            include_private: Pass ``"1"`` to include ``_``-prefixed members
                (default ``"0"``).

        Returns:
            JSON object with ``project_path``, ``structural_map`` (the
            generated string), and ``max_tokens`` used.
            Returns a 400 error when ``project_path`` is missing or points
            outside the projects directory.
        """
        from vetinari.code_search import get_structural_map
        from vetinari.constants import PROJECTS_DIR

        if not project_path:
            return litestar_error_response("project_path required", code=400)

        resolved = Path(project_path).resolve()
        if not resolved.is_relative_to(PROJECTS_DIR.resolve()):
            return litestar_error_response("project_path must be within the projects directory", code=403)
        if not resolved.is_dir():
            return litestar_error_response("project_path must be an existing directory", code=400)

        try:
            structural_map = get_structural_map(
                str(resolved),
                max_tokens=max_tokens,
                task_description=task or None,
                include_private=include_private == "1",
            )
        except Exception:
            logger.exception(
                "Structural map generation failed for %r — returning 500",
                project_path,
            )
            return litestar_error_response("Search system unavailable", code=500)

        return success_response({
            "project_path": project_path,
            "structural_map": structural_map,
            "max_tokens": max_tokens,
        })

    # -- Repo-map: usages ----------------------------------------------------

    @get("/api/repo-map/usages")
    async def api_repo_map_find_usages(
        name: str = Parameter(query="name", default=""),
    ) -> Any:
        """Return files that import or reference a given symbol name.

        Uses ``RepoMap.find_usages`` to search the indexed codebase for
        references to a symbol across imports and docstrings.

        Args:
            name: Symbol name to search for (required).

        Returns:
            JSON with ``name`` and ``files`` (sorted list of relative paths).
            Returns 400 when ``name`` is absent.
        """
        symbol = name.strip()
        if not symbol:
            return litestar_error_response("name parameter required", code=400)

        try:
            from vetinari.repo_map import get_repo_map

            files = get_repo_map().find_usages(symbol)
        except Exception:
            logger.exception(
                "Repo-map usages lookup failed for symbol %r — returning 500",
                symbol,
            )
            return litestar_error_response("Search system unavailable", code=500)

        return success_response({"name": symbol, "files": files})

    # -- Repo-map: symbols ---------------------------------------------------

    @get("/api/repo-map/symbols")
    async def api_repo_map_file_symbols(
        file_path: str = Parameter(query="file_path", default=""),
    ) -> Any:
        """Return all symbols extracted from a specific indexed file.

        Uses ``RepoMap.get_file_symbols`` to look up classes, functions, and
        imports for a file that has been indexed.

        Args:
            file_path: Relative file path within the project index (required).

        Returns:
            JSON with ``file_path`` and ``symbols`` list. Each symbol has
            ``name``, ``kind``, ``line``, ``parent``, and ``docstring``
            fields. Returns 400 when ``file_path`` is absent.
        """
        import re

        fp = file_path.strip()
        if not fp:
            return litestar_error_response("file_path parameter required", code=400)

        # Reject absolute paths — file_path must be relative to the indexed project root.
        # This prevents callers from probing arbitrary on-disk paths via the symbol API.
        _WINDOWS_ABS = re.compile(r"^[A-Za-z][\\/]|^[A-Za-z]:[/\\]")
        if fp.startswith("/") or _WINDOWS_ABS.match(fp):
            return litestar_error_response(
                "file_path must be relative to the project root",
                code=400,
            )

        try:
            from vetinari.repo_map import get_repo_map

            symbols = get_repo_map().get_file_symbols(fp)
        except Exception:
            logger.exception(
                "Repo-map symbol lookup failed for file %r — returning 500",
                fp,
            )
            return litestar_error_response("Search system unavailable", code=500)

        return success_response({
            "file_path": fp,
            "symbols": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "line": s.line,
                    "parent": s.parent,
                    "docstring": s.docstring,
                }
                for s in symbols
            ],
        })

    # -- Repo-map: import graph ----------------------------------------------

    @get("/api/repo-map/import-graph")
    async def api_repo_map_import_graph() -> Any:
        """Return the intra-project import dependency graph.

        Uses ``RepoMap.get_import_graph`` to produce a mapping from each
        indexed file to the vetinari modules it imports. Useful for
        understanding module coupling and finding circular dependencies.

        Returns:
            JSON with ``graph`` — a dict mapping file paths to lists of
            imported module names.
        """
        try:
            from vetinari.repo_map import get_repo_map

            graph = get_repo_map().get_import_graph()
        except Exception:
            logger.exception(
                "Repo-map import graph generation failed — returning 500",
            )
            return litestar_error_response("Search system unavailable", code=500)

        return success_response({"graph": graph})

    return [
        api_code_search,
        api_search_index,
        api_search_status,
        api_structural_map,
        api_repo_map_find_usages,
        api_repo_map_file_symbols,
        api_repo_map_import_graph,
    ]
