"""Project output and packaging Litestar handlers.

Covers review, approval, task-output, assembly, merge, and ZIP download routes
extracted from ``litestar_projects_api.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from vetinari.web.litestar_projects_formatters import build_final_report as _build_final_report

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _create_projects_output_handlers(
    *,
    project_export_max_files: int,
    project_export_max_bytes: int,
) -> list[Any]:
    """Create Litestar handlers for project output and packaging routes.

    Args:
        project_export_max_files: Maximum files allowed in a project ZIP export.
        project_export_max_bytes: Maximum bytes allowed in a project ZIP export.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_guards import admin_guard

    # -- Output / assembly ---------------------------------------------------

    @get("/api/project/{project_id:str}/review", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_project_review(project_id: str) -> Response:
        """Return review-ready task outputs for a project.

        Collects each planned task's output text and generated file list.
        Sets ``has_all_outputs`` to ``false`` when any task is still pending.

        Args:
            project_id: The project to review.

        Returns:
            JSON with ``project_id``, ``goal``, ``tasks``, ``model``, and
            ``has_all_outputs`` flag.
        """
        import pathlib

        import yaml

        from vetinari.constants import TRUNCATE_OUTPUT_PREVIEW
        from vetinari.security.redaction import redact_text
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_file = project_dir / "project.yaml"
        config: dict[str, Any] = {}
        if config_file.exists():
            with pathlib.Path(config_file).open(encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        outputs_dir = project_dir / "outputs"
        review_data: dict[str, Any] = {
            "project_id": project_id,
            "goal": config.get("high_level_goal", ""),
            "tasks": [],
            "model": config.get("model", ""),
            "has_all_outputs": True,
        }

        for task in config.get("tasks", []):
            tid = task["id"]
            task_output_dir = outputs_dir / tid
            output_content = ""
            status = "pending"
            files: list[dict[str, Any]] = []

            if task_output_dir.exists():
                output_file = task_output_dir / "output.txt"
                if output_file.exists():
                    output_content = output_file.read_text(encoding="utf-8")
                    status = "completed"
                generated_dir = task_output_dir / "generated"
                if generated_dir.exists():
                    files.extend(
                        {"name": f.name, "path": f.relative_to(project_dir).as_posix()}
                        for f in generated_dir.iterdir()
                        if f.is_file()
                    )

            if status == "pending":
                review_data["has_all_outputs"] = False

            review_data["tasks"].append({
                "id": tid,
                "description": task.get("description", ""),
                "model_override": task.get("model_override", ""),
                "status": status,
                "output": redact_text(output_content[:TRUNCATE_OUTPUT_PREVIEW]) if output_content else "",
                "output_length": len(output_content) if output_content else 0,
                "files": files,
            })

        return Response(content=review_data, status_code=200, media_type=MediaType.JSON)

    @post("/api/project/{project_id:str}/approve", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_approve_outputs(project_id: str) -> Response:
        """Mark a project's outputs as approved by writing an approval timestamp.

        Creates ``outputs_approved.txt`` in the project directory so downstream
        processes know a human reviewer has accepted the deliverables.

        Args:
            project_id: The project whose outputs are being approved.

        Returns:
            JSON with ``status`` and ``project_id``.
        """
        from datetime import datetime, timezone

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        approval_file = project_dir / "outputs_approved.txt"
        approval_file.write_text(f"Approved at: {datetime.now(timezone.utc).isoformat()}", encoding="utf-8")
        return Response(
            content={"status": "approved", "project_id": project_id},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get(
        "/api/project/{project_id:str}/task/{task_id:str}/output",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def api_task_output(project_id: str, task_id: str) -> Response:
        """Retrieve the full output and generated files for a completed task.

        Args:
            project_id: The project containing the task.
            task_id: The task whose output is requested.

        Returns:
            JSON with ``project_id``, ``task_id``, ``output`` text, and
            ``files`` list.
        """
        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id) or not validate_path_param(task_id):
            return litestar_error_response("Invalid parameters", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        output_path = project_dir / "outputs" / task_id / "output.txt"
        if not output_path.exists():
            return litestar_error_response(f"No output for task '{task_id}'", 404)
        output = output_path.read_text(encoding="utf-8")

        generated_dir = project_dir / "outputs" / task_id / "generated"
        files: list[dict[str, Any]] = []
        if generated_dir.exists():
            files.extend(
                {"name": f.name, "path": str(f.relative_to(project_dir))}
                for f in generated_dir.iterdir()
                if f.is_file()
            )

        return Response(
            content={"project_id": project_id, "task_id": task_id, "output": output, "files": files},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/assemble", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_project_assemble(project_id: str) -> Response:
        """Assemble all task outputs into a Markdown final-delivery report.

        Walks each planned task's output and generated files, writes
        ``final_delivery/final_report.md``, and records the path in
        ``project.yaml``.

        Args:
            project_id: The project to assemble.

        Returns:
            JSON with ``project_id``, ``final_report_path``, ``status``, and
            ``task_count``.
        """
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        import pathlib

        import yaml

        from vetinari.web import shared

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        config_file = project_dir / "project.yaml"
        if not config_file.exists():
            return litestar_error_response("Project config not found", 404)

        with pathlib.Path(config_file).open(encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f) or {}

        final_dir = project_dir / "final_delivery"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_report_path = final_dir / "final_report.md"

        task_entries: list[dict[str, Any]] = []
        for t in config.get("tasks", []):
            tid = t.get("id", "")
            output_path = project_dir / "outputs" / tid / "output.txt"
            output = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            gen_dir = project_dir / "outputs" / tid / "generated"
            generated: list[str] = []
            if gen_dir.exists():
                generated.extend(x.name for x in gen_dir.iterdir() if x.is_file())
            task_entries.append({
                "id": tid,
                "description": t.get("description", ""),
                "assigned_model": t.get("assigned_model_id", ""),
                "output": output,
                "generated": generated,
                "quality_score": t.get("quality_score"),
            })

        if not any(e.get("output") for e in task_entries):
            return litestar_error_response(
                "No task outputs available to assemble — run tasks before assembling",
                409,
            )

        final_content = _build_final_report(
            project_id=project_id,
            project_config=config,
            task_entries=task_entries,
        )
        final_report_path.write_text(final_content, encoding="utf-8")
        config["final_delivery_path"] = str(final_report_path)

        with pathlib.Path(config_file).open("w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)

        return Response(
            content={
                "project_id": project_id,
                "final_report_path": str(final_report_path),
                "status": "assembled",
                "task_count": len(task_entries),
            },
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/project/{project_id:str}/merge", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_merge_project(project_id: str) -> Response:
        """Merge a project's outputs into a self-contained final delivery package.

        Copies task outputs and generated artifacts into ``final_projects/<id>/``,
        writes a ``manifest.yaml``, ``README.md``, and ``build_report.json``.

        Args:
            project_id: The project to merge.

        Returns:
            JSON with ``status``, ``project_id``, and ``final_path``.
        """
        import json as _json
        import pathlib
        import shutil
        from datetime import datetime, timezone

        import yaml

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)

        final_root = shared.PROJECT_ROOT / "final_projects"
        final_root.mkdir(parents=True, exist_ok=True)

        final_dir = final_root / project_id
        if final_dir.exists():
            # VET142-excluded: lifecycle-fenced deletion via RecycleStore.retire
            from vetinari.safety.recycle import RecycleStore

            RecycleStore().retire(
                final_dir,
                reason=f"merge_project overwrote final_dir for {project_id}",
            )
        final_dir.mkdir(parents=True, exist_ok=True)

        outputs_dir = project_dir / "outputs"
        final_outputs = final_dir / "outputs"
        if outputs_dir.exists():
            shutil.copytree(outputs_dir, final_outputs)

        final_artifacts = final_dir / "artifacts"
        final_artifacts.mkdir(parents=True, exist_ok=True)
        if outputs_dir.exists():
            for task_subdir in outputs_dir.iterdir():
                if task_subdir.is_dir():
                    generated_dir = task_subdir / "generated"
                    if generated_dir.exists():
                        for f in generated_dir.iterdir():
                            if f.is_file():
                                dest = final_artifacts / f"{task_subdir.name}_{f.name}"
                                shutil.copy2(f, dest)

        config_file = project_dir / "project.yaml"
        manifest: dict[str, Any] = {
            "project_id": project_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": project_id,
        }
        if config_file.exists():
            with pathlib.Path(config_file).open(encoding="utf-8") as f:
                config: dict[str, Any] = yaml.safe_load(f) or {}
                manifest["goal"] = config.get("high_level_goal", "")
                manifest["tasks"] = config.get("tasks", [])
                manifest["model"] = config.get("model", "")

        with pathlib.Path(final_dir / "manifest.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(manifest, f, allow_unicode=True)

        readme_lines = [f"# Project: {project_id}", "", manifest.get("goal", "No description"), "", "## Tasks", ""]
        readme_lines.extend(
            f"- **{task.get('id')}**: {task.get('description', '')}" for task in manifest.get("tasks", [])
        )
        readme_lines += [
            "",
            "## Files",
            "",
            "Generated artifacts are in the `artifacts/` directory.",
            "",
            "## Usage",
            "",
            "Run the generated code from the artifacts folder.",
            "",
        ]
        (final_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

        build_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": project_id,
            "tasks": manifest.get("tasks", []),
            "model": manifest.get("model", ""),
        }
        (final_dir / "build_report.json").write_text(_json.dumps(build_report, indent=2), encoding="utf-8")

        return Response(
            content={"status": "merged", "project_id": project_id, "final_path": str(final_dir)},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/project/{project_id:str}/download", media_type=MediaType.JSON, guards=[admin_guard])
    async def api_project_download(project_id: str) -> Response:
        """Download a ZIP archive of the project workspace and outputs.

        Bundles ``outputs/``, ``final_delivery/``, and ``workspace/`` directories
        into a single ZIP file returned as ``application/zip``.

        Args:
            project_id: The project to download.

        Returns:
            ZIP file as an ``application/zip`` attachment, or HTTP 404 when
            the project directory does not exist.
        """
        import io
        import zipfile

        from vetinari.web import shared
        from vetinari.web.responses import litestar_error_response
        from vetinari.web.shared import validate_path_param

        if not validate_path_param(project_id):
            return litestar_error_response("Invalid project ID", 400)

        project_dir = shared.PROJECT_ROOT / "projects" / project_id
        if not project_dir.exists():
            return litestar_error_response(f"Project not found: {project_id}", 404)
        project_base = project_dir.resolve()

        include_dirs = ["outputs", "final_delivery", "workspace"]
        skip_names = {".paused", "project.yaml"}

        buffer = io.BytesIO()
        file_count = 0
        total_bytes = 0
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for subdir_name in include_dirs:
                subdir = project_dir / subdir_name
                if not subdir.exists():
                    continue
                allowed_base = subdir.resolve()
                for file_path in subdir.rglob("*"):
                    if file_count >= project_export_max_files:
                        return litestar_error_response(
                            "Project export file limit exceeded",
                            413,
                            details={"max_files": project_export_max_files},
                        )
                    if file_path.is_symlink() or not file_path.is_file():
                        continue
                    if file_path.name in skip_names:
                        continue
                    try:
                        resolved_file = file_path.resolve(strict=True)
                        resolved_file.relative_to(allowed_base)
                        arcname = resolved_file.relative_to(project_base)
                        file_size = resolved_file.stat().st_size
                    except (OSError, ValueError) as exc:
                        logger.warning("Skipped unsafe export path %s in project %s: %s", file_path, project_id, exc)
                        continue
                    if total_bytes + file_size > project_export_max_bytes:
                        return litestar_error_response(
                            "Project export byte limit exceeded",
                            413,
                            details={
                                "max_bytes": project_export_max_bytes,
                                "current_bytes": total_bytes,
                                "next_file_bytes": file_size,
                            },
                        )
                    zf.write(resolved_file, arcname.as_posix())
                    file_count += 1
                    total_bytes += file_size

        buffer.seek(0)
        logger.info("Project ZIP download: %s", project_id)

        return Response(
            content=buffer.getvalue(),
            status_code=200,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{project_id}.zip"'},
        )

    return [
        api_project_review,
        api_approve_outputs,
        api_task_output,
        api_project_assemble,
        api_merge_project,
        api_project_download,
    ]
