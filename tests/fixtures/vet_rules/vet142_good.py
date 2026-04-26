"""VET142 GOOD fixture — DELETE route guarded via lifecycle-fence comment.

This module represents a vetinari/web/ route handler that has replaced the
bare shutil.rmtree call with RecycleStore.retire and carries the canonical
# VET142-excluded: lifecycle-fenced comment so the checker accepts it.
VET142 must NOT flag this module.
"""

from __future__ import annotations

from pathlib import Path


def delete_project_guarded(project_id: str, project_dir: Path) -> dict:
    """Delete a project via the recycle-bin lifecycle fence.

    Args:
        project_id: The project to delete.
        project_dir: Filesystem path to the project directory.

    Returns:
        Status dict confirming the retirement record.
    """
    from vetinari.safety.recycle import RecycleStore

    # VET142-excluded: lifecycle-fenced deletion via RecycleStore.retire
    record = RecycleStore().retire(project_dir, reason=f"delete_project {project_id}")
    return {"status": "deleted", "project_id": project_id, "recycle_record_id": record.record_id}
