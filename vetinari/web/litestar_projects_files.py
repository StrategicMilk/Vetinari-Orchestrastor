"""Project workspace file Litestar handlers.

Covers workspace read, write, list, and admin artifact routes extracted from
``litestar_projects_api.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _create_projects_file_handlers(
    *,
    workspace_read_max_bytes: int,
    workspace_list_max_files: int,
) -> list[Any]:
    """Create Litestar handlers for project workspace file routes.

    Args:
        workspace_read_max_bytes: Maximum bytes served by the workspace reader.
        workspace_list_max_files: Maximum files returned by workspace listing.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    # -- Files ---------------------------------------------------------------

    @post("/api/project/{project_id:str}/files/read", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_read_file(project_id: str, data: dict[str, Any]) -> Response:
        """Read a file from within a project's workspace sandbox.

        Validates that the resolved path stays inside the project's
        ``workspace/`` directory to prevent directory traversal.

        Args:
            project_id: The project whose workspace is being read.
            data: JSON body with ``path`` field (relative to workspace root).

        Returns:
            JSON with ``status``, ``path``, ``content``, and ``size``, or
            HTTP 403 when the path escapes the workspace sandbox.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        file_path = data.get("path", "")
        if not file_path:
            return litestar_error_response("path is required", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        workspace_dir = project_dir / "workspace"
        allowed_base = workspace_dir.resolve()
        target_path = (workspace_dir / file_path).resolve()

        try:
            target_path.relative_to(allowed_base)
        except ValueError:
            logger.warning(
                "Workspace path traversal attempt blocked for project %s: %s",
                project_id,
                file_path,
            )
            return litestar_error_response("Access denied: path outside workspace", 403)

        if not target_path.exists():
            return litestar_error_response("File not found", 404)
        if not target_path.is_file():
            return litestar_error_response("Not a file", 400)

        try:
            file_size = target_path.stat().st_size
            if file_size > workspace_read_max_bytes:
                return litestar_error_response(
                    "File too large to read in workspace viewer",
                    413,
                    details={"max_bytes": workspace_read_max_bytes, "actual_bytes": file_size},
                )
            content = target_path.read_text(encoding="utf-8")
        except PermissionError:
            logger.warning("Permission denied reading workspace file %s in project %s", file_path, project_id)
            return litestar_error_response("Permission denied reading file", 403)
        except UnicodeDecodeError:
            logger.warning("Non-UTF-8 workspace file read blocked for project %s: %s", project_id, file_path)
            return litestar_error_response("File is not valid UTF-8 text", 415)
        except OSError as exc:
            logger.warning("OS error reading workspace file %s in project %s: %s", file_path, project_id, exc)
            return litestar_error_response("File read failed — check server logs", 503)
        relative_path = target_path.relative_to(project_dir).as_posix()
        logger.info("IO Read: %s (project: %s)", relative_path, project_id)

        return Response(
            content={
                "status": "ok",
                "path": relative_path,
                "content": content,
                "size": file_size,
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/files/write", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_write_file(project_id: str, data: dict[str, Any]) -> Response:
        """Write a file into a project's workspace sandbox.

        Validates the path stays inside ``workspace/`` and creates parent
        directories automatically.

        Args:
            project_id: The project whose workspace is being written to.
            data: JSON body with ``path`` (relative to workspace) and
                ``content`` string.

        Returns:
            JSON with ``status``, ``path``, and ``size``, or HTTP 403 when
            the path escapes the workspace sandbox.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        # Validate before any filesystem operations to prevent traversal side-effects.
        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        file_path = data.get("path", "")
        content = data.get("content", "")

        if not file_path:
            return litestar_error_response("path is required", 400)
        if not isinstance(content, str):
            return litestar_error_response("content must be a string", 422)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        workspace_dir = project_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        allowed_base = workspace_dir.resolve()
        target_path = (workspace_dir / file_path).resolve()

        try:
            target_path.relative_to(allowed_base)
        except ValueError:
            logger.warning(
                "Workspace path traversal attempt blocked for project %s: %s",
                project_id,
                file_path,
            )
            return litestar_error_response("Access denied: path outside workspace", 403)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
        except PermissionError:
            logger.warning("Permission denied writing workspace file %s in project %s", file_path, project_id)
            return litestar_error_response("Permission denied writing file", 403)
        except OSError as exc:
            logger.warning("OS error writing workspace file %s in project %s: %s", file_path, project_id, exc)
            return litestar_error_response("File write failed — check server logs", 503)
        relative_path = target_path.relative_to(project_dir).as_posix()
        logger.info("IO Write: %s (project: %s, size: %d)", relative_path, project_id, len(content))

        return Response(
            content={
                "status": "ok",
                "path": relative_path,
                "size": len(content),
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/project/{project_id:str}/files/list", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_list_files(project_id: str) -> Response:
        """List all files in a project's workspace directory.

        Returns path, size, and mtime for each regular file found recursively.

        Args:
            project_id: The project whose workspace is being listed.

        Returns:
            JSON with a ``files`` list, or an empty list when the workspace
            does not yet exist.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        workspace_dir = project_dir / "workspace"
        if not workspace_dir.exists():
            return Response(
                content={"files": [], "truncated": False, "max_files": workspace_list_max_files},
                status_code=200,
                media_type=MediaType.JSON,
            )

        files: list[dict[str, Any]] = []
        truncated = False
        workspace_base = workspace_dir.resolve()
        try:
            for f in workspace_dir.rglob("*"):
                if len(files) >= workspace_list_max_files:
                    truncated = True
                    break
                if f.is_symlink() or not f.is_file():
                    continue
                resolved_file = f.resolve(strict=True)
                resolved_file.relative_to(workspace_base)
                stat = resolved_file.stat()
                files.append({
                    "path": resolved_file.relative_to(workspace_base).as_posix(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })
        except (OSError, ValueError) as exc:
            logger.warning("Workspace listing failed for project %s: %s", project_id, exc)
            return litestar_error_response("Workspace listing failed — check server logs", 503)
        return Response(
            content={"files": files, "truncated": truncated, "max_files": workspace_list_max_files},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/artifacts", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_artifacts() -> dict[str, Any]:
        """List build artifacts from the project build directory.

        Scans the ``build/artifacts`` directory under the project root and
        returns metadata (name, size, path) for every regular file found.
        The ``path`` field is the filename only — absolute host paths are
        never exposed in the response.

        Returns:
            JSON with an ``artifacts`` list of file metadata dicts.
        """
        from vetinari.web import shared

        build_dir = shared.PROJECT_ROOT / "build" / "artifacts"
        artifacts: list[dict[str, Any]] = []
        if build_dir.exists():
            try:
                # Use filename only — never expose absolute host filesystem paths
                artifacts.extend(
                    {"name": f.name, "size": f.stat().st_size, "path": f.name}
                    for f in build_dir.iterdir()
                    if f.is_file()
                )
            except OSError:
                logger.warning(
                    "Build artifacts directory %s could not be read — returning empty artifact list",
                    build_dir,
                )
                return {"artifacts": [], "error": "Build artifacts directory unreadable"}
        return {"artifacts": artifacts}

    return [
        api_read_file,
        api_write_file,
        api_list_files,
        api_artifacts,
    ]
