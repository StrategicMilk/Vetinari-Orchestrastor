"""VET142 BAD fixture — DELETE route that calls shutil.rmtree without a guard.

This module represents a vetinari/web/ route handler that performs a
full-tree directory deletion via shutil.rmtree without either
@protected_mutation on the function or a # VET142-excluded: comment in
the body.  VET142 must flag the shutil.rmtree call site.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def delete_project(project_id: str, project_dir: Path) -> dict:
    """Delete a project directory directly — MISSING guard.

    Args:
        project_id: The project to delete.
        project_dir: Filesystem path to the project directory.

    Returns:
        Status dict confirming deletion.
    """
    shutil.rmtree(project_dir)  # VET142 should flag this line
    return {"status": "deleted", "project_id": project_id}
