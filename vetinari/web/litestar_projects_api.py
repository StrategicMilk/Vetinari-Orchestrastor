"""Project management CRUD and operational Litestar handlers.

Native Litestar equivalents of the project lifecycle, task management, output
assembly, file I/O, and model-management routes previously registered via the
Flask Blueprint in ``projects_api.py`` (ADR-0066). URL paths are identical to
the Flask originals so existing clients require no changes.

Split boundary: this public factory preserves the project route registration
order while cohesive route groups live in sibling ``litestar_projects_*``
modules. SSE streaming, cancel, pause, and resume live in
``litestar_projects_execution.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from vetinari.web.litestar_projects_creation import _create_projects_creation_handlers
from vetinari.web.litestar_projects_files import _create_projects_file_handlers
from vetinari.web.litestar_projects_formatters import (
    DEFAULT_SYSTEM_PROMPT as _DEFAULT_SYSTEM_PROMPT,
)
from vetinari.web.litestar_projects_interactions import _create_projects_interaction_handlers
from vetinari.web.litestar_projects_lifecycle import _create_projects_lifecycle_handlers
from vetinari.web.litestar_projects_models import _create_projects_model_handlers
from vetinari.web.litestar_projects_output import _create_projects_output_handlers
from vetinari.web.litestar_projects_tasks import _create_projects_task_handlers

_PROJECT_EXPORT_MAX_FILES = 1000
_PROJECT_EXPORT_MAX_BYTES = 50 * 1024 * 1024
_WORKSPACE_READ_MAX_BYTES = 1024 * 1024
_WORKSPACE_LIST_MAX_FILES = 1000

logger = logging.getLogger(__name__)


def create_projects_api_handlers() -> list[Any]:
    """Create Litestar handlers for project CRUD and operational routes.

    Covers lifecycle (list, detail, rename, archive, delete, create), task
    management (add, update, delete, rerun, execute, message, verify-goal),
    output assembly (review, approve, task-output, assemble, merge, download),
    file I/O (read, write, list, artifacts), and model management (search,
    override, refresh).

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    return [
        *_create_projects_lifecycle_handlers(),
        *_create_projects_creation_handlers(default_system_prompt=_DEFAULT_SYSTEM_PROMPT),
        *_create_projects_task_handlers(),
        *_create_projects_interaction_handlers(default_system_prompt=_DEFAULT_SYSTEM_PROMPT),
        *_create_projects_output_handlers(
            project_export_max_files=_PROJECT_EXPORT_MAX_FILES,
            project_export_max_bytes=_PROJECT_EXPORT_MAX_BYTES,
        ),
        *_create_projects_file_handlers(
            workspace_read_max_bytes=_WORKSPACE_READ_MAX_BYTES,
            workspace_list_max_files=_WORKSPACE_LIST_MAX_FILES,
        ),
        *_create_projects_model_handlers(),
    ]
