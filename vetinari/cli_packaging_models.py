"""Packaging CLI — ``vetinari models`` command and all model management helpers.

Provides the model management subcommand group for Vetinari's CLI:
- List local GGUF files with rich table output
- Download models from HuggingFace Hub with optional progress bar
- Remove models with confirmation prompt
- Show per-file metadata (size, quantization, family, GGUF header validity)
- Recommend models based on detected VRAM
- Scan common directories for GGUF/AWQ model files
- ``cmd_forget``, ``cmd_config_reload``, ``cmd_resume``, ``cmd_quick_action``

Imported by ``cli_packaging.py`` which re-exports these for the CLI dispatch table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import vetinari.cli_packaging_models_local as _local_models
from vetinari.cli_packaging_data import _RICH_AVAILABLE, _console, _detect_hardware
from vetinari.cli_packaging_models_local import (
    _find_models_dir,
    _find_native_models_dir,
    _guess_family,
    _guess_quantization,
    _iter_model_files,
    _local_file_matches_filters,
    _models_check,
    _models_info,
    _models_recommend,
    _models_remove,
    _models_scan,
)
from vetinari.cli_packaging_models_local import (
    _models_list as _local_models_list,
)
from vetinari.cli_packaging_models_remote import (
    _NATIVE_BACKENDS,
    _download_with_progress,
    _infer_cli_backend,
    _models_download,
    _models_files,
    _models_status,
    _verify_sha256,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_NATIVE_BACKENDS",
    "_RICH_AVAILABLE",
    "_console",
    "_download_with_progress",
    "_find_models_dir",
    "_find_native_models_dir",
    "_guess_family",
    "_guess_quantization",
    "_infer_cli_backend",
    "_iter_model_files",
    "_local_file_matches_filters",
    "_models_check",
    "_models_download",
    "_models_files",
    "_models_info",
    "_models_list",
    "_models_recommend",
    "_models_remove",
    "_models_scan",
    "_models_status",
    "_verify_sha256",
    "cmd_config_reload",
    "cmd_forget",
    "cmd_models",
    "cmd_quick_action",
    "cmd_resume",
]


def _models_list(
    models_dir: Path,
    *,
    objective: str | None = None,
    family: str | None = None,
    quantization: str | None = None,
    file_type: str | None = None,
    min_size_gb: float | None = None,
    max_size_gb: float | None = None,
) -> int:
    """Print local model files through the historical patchable facade."""
    _local_models._RICH_AVAILABLE = _RICH_AVAILABLE
    _local_models._console = _console
    return _local_models_list(
        models_dir,
        objective=objective,
        family=family,
        quantization=quantization,
        file_type=file_type,
        min_size_gb=min_size_gb,
        max_size_gb=max_size_gb,
    )


def cmd_models(args: Any) -> int:
    """Manage local and Hugging Face model artifacts.

    Supports model-management sub-actions selected via ``args.models_action``:

    * ``list``       — scan models directory and print a summary table.
    * ``download``   — download a model from HuggingFace Hub.
    * ``remove``     — delete a model file after confirmation.
    * ``info``       — print detailed metadata for a single model file.
    * ``recommend``  — suggest optimal models based on detected VRAM.
    * ``scan``       — discover .gguf/.awq files across common directories.
    * ``check``      — check for newer, better models via benchmarks and sentiment.

    Args:
        args: Parsed CLI arguments.  Recognises ``args.models_action``,
            ``args.repo``, ``args.filename``, and ``args.name``.

    Returns:
        0 on success, 1 on failure.
    """
    action = getattr(args, "models_action", "list")
    backend = getattr(args, "backend", "auto")
    model_format = getattr(args, "model_format", None)
    filename = getattr(args, "filename", None)
    revision = getattr(args, "revision", None)
    objective = getattr(args, "objective", None)
    family = getattr(args, "family", None)
    quantization = getattr(args, "quantization", None)
    file_type = getattr(args, "file_type", None)
    min_size_gb = getattr(args, "min_size_gb", None)
    max_size_gb = getattr(args, "max_size_gb", None)
    vram_gb = getattr(args, "vram_gb", 32)
    backend_normalized = _infer_cli_backend(
        backend,
        filename=filename,
        model_format=model_format,
        action=action,
    )
    models_dir = _find_native_models_dir() if backend_normalized in _NATIVE_BACKENDS else _find_models_dir()

    if action == "list":
        return _models_list(
            models_dir,
            objective=objective,
            family=family,
            quantization=quantization,
            file_type=file_type,
            min_size_gb=min_size_gb,
            max_size_gb=max_size_gb,
        )

    if action == "download":
        repo = getattr(args, "repo", None)
        return _models_download(
            repo,
            filename,
            models_dir,
            backend=backend_normalized,
            model_format=model_format,
            revision=revision,
        )

    if action == "files":
        repo = getattr(args, "repo", None)
        return _models_files(
            repo,
            backend=backend_normalized,
            model_format=model_format,
            revision=revision,
            vram_gb=vram_gb,
            objective=objective,
            family=family,
            quantization=quantization,
            file_type=file_type,
            min_size_gb=min_size_gb,
            max_size_gb=max_size_gb,
        )

    if action == "status":
        return _models_status(getattr(args, "download_id", None))

    if action == "remove":
        name = getattr(args, "name", None)
        return _models_remove(name, models_dir)

    if action == "info":
        name = getattr(args, "name", None)
        return _models_info(name, models_dir)

    if action == "recommend":
        hw = _detect_hardware()
        return _models_recommend(hw["vram_gb"])

    if action == "scan":
        return _models_scan()

    if action == "check":
        return _models_check()

    print(f"Unknown models action: {action}")
    print("Usage: vetinari models {{list|files|download|status|remove|info|recommend|scan|check}}")
    return 1


# ── Remaining packaging commands ───────────────────────────────────────────────


def cmd_forget(args: Any) -> int:
    """Purge all learned data for a specific project.

    Args:
        args: Parsed CLI arguments with ``project`` name.

    Returns:
        0 on success, 1 on error.
    """
    project = getattr(args, "project", None)
    if not project:
        print("Error: --project is required. Usage: vetinari forget --project <name>")
        return 1
    print(f"[Vetinari] Forgetting all learned data for project: {project}")
    try:
        from vetinari.database import get_connection

        with get_connection() as conn:
            # Delete subtasks for plans whose goal matches the project name, then
            # delete the plans themselves.  The unified schema (schema.sql) uses
            # PlanHistory + SubtaskMemory — there is no legacy training_records or
            # episodes table.
            conn.execute(
                """
                DELETE FROM SubtaskMemory
                WHERE plan_id IN (
                    SELECT plan_id FROM PlanHistory WHERE goal LIKE ?
                )
                """,
                (f"%{project}%",),
            )
            conn.execute("DELETE FROM PlanHistory WHERE goal LIKE ?", (f"%{project}%",))
            conn.commit()
        print(f"  Cleared plan history and subtask memory for project: {project}")
    except Exception as exc:
        logger.warning("Could not clear project data from database for %s — data not purged", project, exc_info=True)
        print(f"  Database cleanup failed: {exc}")
        return 1
    print(f"  Done. Project '{project}' data has been forgotten.")
    return 0


def cmd_config_reload(_args: Any) -> int:
    """Hot-reload the VetinariSettings singleton without restarting.

    Returns:
        0 always.
    """
    from vetinari.config.settings import reset_settings

    reset_settings()
    print("[Vetinari] Settings reloaded from environment and config files.")
    return 0


def cmd_resume(args: Any) -> int:
    """Resume a previously interrupted plan execution from checkpoint.

    Args:
        args: Parsed CLI arguments with ``plan_id``.

    Returns:
        0 on success, 1 on error.
    """
    plan_id = getattr(args, "plan_id", None)
    if not plan_id:
        print("Error: plan_id is required. Usage: vetinari resume <plan_id>")
        return 1
    print(f"[Vetinari] Resuming plan: {plan_id}")
    try:
        from vetinari.orchestration.durable_execution import DurableExecutionEngine

        engine = DurableExecutionEngine()
        # Verify the checkpoint exists before attempting recovery.
        checkpoint = engine.load_checkpoint(plan_id)
        if checkpoint is None:
            print(f"  No checkpoint found for plan {plan_id}")
            return 1
        print("  Checkpoint found — attempting recovery via DurableExecutionEngine")
        result = engine.recover_execution(plan_id)
        status = result.get("status", "unknown")
        completed_count = result.get("completed_tasks", 0)
        failed_count = result.get("failed_tasks", 0)
        print(f"  Recovery complete — status: {status}")
        print(f"  Completed: {completed_count}  Failed: {failed_count}")
        return 0 if status not in ("failed", "error") else 1
    except Exception as exc:
        logger.warning(
            "Could not resume plan %s — checkpoint may be corrupt or execution context missing",
            plan_id,
            exc_info=True,
        )
        print(f"  Resume failed: {exc}")
        return 1


def cmd_quick_action(args: Any) -> int:
    """Execute a quick action (explain/test/review/fix) on a file.

    Args:
        args: Parsed CLI arguments with ``quick_action`` and ``file``.

    Returns:
        0 on success, 1 on error.
    """
    action = getattr(args, "quick_action", "explain")
    file_path = getattr(args, "file", None)
    if not file_path:
        print(f"Error: file path required. Usage: vetinari {action} <file>")
        return 1
    goal_map = {
        "explain": f"Explain what the file {file_path} does, its role in the codebase, and key functions/classes.",
        "test": f"Generate comprehensive tests for the file {file_path}.",
        "review": f"Review the file {file_path} for bugs, security issues, and code quality problems.",
        "fix": f"Fix any issues found in the file {file_path}.",
    }
    goal = goal_map.get(action, f"{action} the file {file_path}")
    print(f"[Vetinari] {action.capitalize()}: {file_path}")
    try:
        from vetinari.orchestration.two_layer import get_two_layer_orchestrator

        orch = get_two_layer_orchestrator()
        results = orch.generate_and_execute(goal=goal)
        if results.get("final_output"):
            print("\n--- Output ---")
            print(str(results["final_output"])[:3000])
        return 0
    except Exception as exc:
        print(f"[Vetinari] Error: {exc}")
        logger.warning("Quick action '%s' failed for %s: %s", action, file_path, exc)
        return 1
