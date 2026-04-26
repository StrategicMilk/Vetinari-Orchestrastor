"""Project Git API — native Litestar handlers for project-level git helpers.

Exposes git commit-message generation and merge-conflict detection from
:mod:`vetinari.project.git_integration` over HTTP.

Endpoints
---------
    POST /api/v1/project/git/commit-message      — generate Conventional Commit message
                                                   from staged changes (branch-aware)
    POST /api/v1/project/git/commit-message-path — generate commit message by inspecting
                                                   staged diff at a given repo path
    POST /api/v1/project/git/conflicts            — detect potential merge conflicts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, post

    from vetinari.web.litestar_guards import admin_guard
    from vetinari.web.responses import litestar_error_response, success_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_project_git_handlers() -> list[Any]:
    """Create all Litestar route handlers for the project git API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — project git API handlers not registered")
        return []

    from vetinari.web.shared import PROJECT_ROOT

    def _validate_project_path(raw_path: str) -> Path | None:
        """Resolve ``raw_path`` and confirm it sits inside PROJECT_ROOT.

        Args:
            raw_path: The raw path string supplied by the caller.

        Returns:
            The resolved ``Path`` when it is safely confined, or ``None`` when
            the path escapes PROJECT_ROOT (absolute paths to other trees,
            traversal sequences, etc.).
        """
        try:
            resolved = Path(raw_path).resolve()
        except (ValueError, OSError) as exc:
            logger.warning("Could not resolve path %s — invalid path (%s), skipping git operation", raw_path, type(exc).__name__)
            return None
        # Confine all git helper paths to PROJECT_ROOT so callers cannot point
        # git operations at arbitrary filesystem locations (e.g. "C:\\Windows").
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            logger.debug("Path %s escapes PROJECT_ROOT — rejecting for security, skipping git operation", resolved)
            return None
        return resolved

    # -- POST /api/v1/project/git/commit-message ------------------------------

    @post("/api/v1/project/git/commit-message", media_type=MediaType.JSON, guards=[admin_guard])
    async def project_git_commit_message(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate a Conventional Commit message for the currently staged changes.

        Accepts a JSON body with ``project_path`` (required), ``description``
        (optional), and ``scope`` (optional). Delegates to
        :func:`~vetinari.project.git_integration.generate_commit_message_for_branch`.

        Args:
            data: JSON request body with commit generation parameters.

        Returns:
            ADR-0072 success envelope with ``commit`` containing:
            ``message``, ``type``, ``scope``, ``breaking``, ``files_changed``.
            Returns a 400 response when ``project_path`` is missing or empty.
        """
        body = data if data is not None else {}
        raw_path = body.get("project_path", "").strip()
        if not raw_path:
            return litestar_error_response(  # type: ignore[return-value]
                "project_path is required", 400
            )

        safe_path = _validate_project_path(raw_path)
        if safe_path is None:
            return litestar_error_response(  # type: ignore[return-value]
                "Invalid project_path: path must be within the project root", 400
            )

        from vetinari.project.git_integration import generate_commit_message_for_branch

        description: str | None = body.get("description") or None
        scope: str | None = body.get("scope") or None

        info = generate_commit_message_for_branch(
            project_path=safe_path,
            description=description,
            scope=scope,
        )
        return success_response({
            "commit": {
                "message": info.format_message(),
                "type": info.type,
                "scope": info.scope,
                "breaking": info.breaking,
                "files_changed": info.files_changed,
            }
        })

    # -- POST /api/v1/project/git/commit-message-path -------------------------

    @post(
        "/api/v1/project/git/commit-message-path",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def project_git_commit_message_path(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Generate a commit message by inspecting the staged diff at a given path.

        Differs from ``/commit-message`` in that it uses
        :func:`~vetinari.project.git_integration.generate_commit_message_for_path`
        which creates a short-lived GitOperations instance directly from the
        repo root, rather than deriving context from the current branch.

        Accepts a JSON body with ``repo_path`` (required — absolute path to the
        repository root).

        Args:
            data: JSON request body with the repository path.

        Returns:
            ADR-0072 success envelope with ``commit`` containing:
            ``message``, ``type``, ``scope``, ``breaking``, ``files_changed``.
            Returns a 400 response when ``repo_path`` is missing or empty.
        """
        body = data if data is not None else {}
        raw_path = body.get("repo_path", "").strip()
        if not raw_path:
            return litestar_error_response(  # type: ignore[return-value]
                "repo_path is required", 400
            )

        safe_path = _validate_project_path(raw_path)
        if safe_path is None:
            return litestar_error_response(  # type: ignore[return-value]
                "Invalid repo_path: path must be within the project root", 400
            )

        from vetinari.project.git_integration import generate_commit_message_for_path

        info = generate_commit_message_for_path(safe_path)
        return success_response({
            "commit": {
                "message": info.format_message(),
                "type": info.type,
                "scope": info.scope,
                "breaking": info.breaking,
                "files_changed": info.files_changed,
            }
        })

    # -- POST /api/v1/project/git/conflicts -----------------------------------

    @post("/api/v1/project/git/conflicts", media_type=MediaType.JSON, guards=[admin_guard])
    async def project_git_conflicts(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Detect potential merge conflicts between the current branch and a target.

        Accepts a JSON body with ``project_path`` (required) and
        ``target_branch`` (optional, defaults to ``"main"``). Performs a
        dry-run merge via
        :func:`~vetinari.project.git_integration.detect_merge_conflicts`.

        Args:
            data: JSON request body with project path and optional target branch.

        Returns:
            ADR-0072 success envelope with ``conflicts`` containing:
            ``count`` and ``items`` (list of objects with ``file_path``,
            ``conflict_type``, and ``suggestion``).
            Returns a 400 response when ``project_path`` is missing or empty.
        """
        body = data if data is not None else {}
        raw_path = body.get("project_path", "").strip()
        if not raw_path:
            return litestar_error_response(  # type: ignore[return-value]
                "project_path is required", 400
            )

        safe_path = _validate_project_path(raw_path)
        if safe_path is None:
            return litestar_error_response(  # type: ignore[return-value]
                "Invalid project_path: path must be within the project root", 400
            )

        from vetinari.project.git_integration import detect_merge_conflicts

        target_branch: str = body.get("target_branch") or "main"

        conflict_list = detect_merge_conflicts(
            project_path=safe_path,
            target_branch=target_branch,
        )
        return success_response({
            "conflicts": {
                "count": len(conflict_list),
                "items": [
                    {
                        "file_path": c.file_path,
                        "conflict_type": c.conflict_type,
                        "suggestion": c.suggestion,
                    }
                    for c in conflict_list
                ],
            }
        })

    return [
        project_git_commit_message,
        project_git_commit_message_path,
        project_git_conflicts,
    ]
