"""Flask Blueprint for all project-related API routes.

Backward-compatibility shim: the routes are now split into two modules:
  - crud_routes.py        -- project/task CRUD, file I/O, artifacts
  - task_execution_routes.py -- streaming, new-project, execution, review, merge, etc.

This file re-exports the blueprints so existing callers that do
``from vetinari.web.projects_bp import projects_bp`` continue to work.
The ``projects_bp`` name is kept as an alias for crud_bp for any code that
registers it by that name; task_exec_bp must be registered separately.
"""

from vetinari.web.crud_routes import crud_bp as projects_bp  # noqa: F401
from vetinari.web.crud_routes import crud_bp  # noqa: F401
from vetinari.web.task_execution_routes import task_exec_bp  # noqa: F401

__all__ = ["projects_bp", "crud_bp", "task_exec_bp"]
