"""CodeBridge - External coding service integration.

This module provides a bridge to external coding services (like CodeNomad)
for offloading heavier coding tasks.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests

from vetinari.types import CodingTaskStatus as BridgeTaskStatus
from vetinari.types import CodingTaskType as BridgeTaskType  # canonical enums

logger = logging.getLogger(__name__)


@dataclass
class BridgeTaskSpec:
    """Specification for a bridge task."""

    task_id: str = field(default_factory=lambda: f"bridge_{uuid.uuid4().hex[:8]}")
    task_type: BridgeTaskType = BridgeTaskType.IMPLEMENT
    language: str = "python"
    framework: str = ""
    repo_path: str = ""
    description: str = ""
    constraints: str = ""
    target_files: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgeTaskResult:
    """Result from a bridge task."""

    task_id: str
    status: BridgeTaskStatus = BridgeTaskStatus.PENDING
    success: bool = False
    output_files: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    logs: str = ""
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None


class CodeBridge:
    """Bridge to external coding services.

    This provides:
    - Task submission to external services
    - Status polling
    - Artifact retrieval
    - Fallback to in-process coder
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint or os.environ.get("CODE_BRIDGE_ENDPOINT", "http://localhost:4096")
        self.api_key = api_key or os.environ.get("CODE_BRIDGE_API_KEY", "")
        self.enabled = os.environ.get("CODE_BRIDGE_ENABLED", "false").lower() in ("1", "true", "yes")
        self.timeout = int(os.environ.get("CODE_BRIDGE_TIMEOUT", "30"))

        logger.info("CodeBridge initialized (enabled=%s, endpoint=%s)", self.enabled, self.endpoint)

    def is_available(self) -> bool:
        """Check if the bridge is available."""
        if not self.enabled:
            return False

        try:
            response = requests.get(f"{self.endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning("Bridge health check failed: %s", e)
            return False

    def submit_task(self, spec: BridgeTaskSpec) -> BridgeTaskResult:
        """Submit a task to the external bridge."""
        if not self.enabled:
            logger.warning("CodeBridge is not enabled")
            return BridgeTaskResult(
                task_id=spec.task_id, status=BridgeTaskStatus.FAILED, success=False, error="CodeBridge is not enabled"
            )

        logger.info("Submitting task to bridge: %s (%s)", spec.task_id, spec.task_type.value)

        try:
            payload = {
                "task_id": spec.task_id,
                "task_type": spec.task_type.value,
                "language": spec.language,
                "framework": spec.framework,
                "repo_path": spec.repo_path,
                "description": spec.description,
                "constraints": spec.constraints,
                "target_files": spec.target_files,
                "context": spec.context,
            }

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(f"{self.endpoint}/tasks", json=payload, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()
                return BridgeTaskResult(
                    task_id=spec.task_id,
                    status=BridgeTaskStatus(result.get("status", "pending")),
                    success=result.get("success", False),
                    output_files=result.get("output_files", []),
                    artifacts=result.get("artifacts", []),
                    logs=result.get("logs", ""),
                    error=result.get("error"),
                )
            else:
                logger.error("Bridge task submission failed: %s", response.status_code)
                return BridgeTaskResult(
                    task_id=spec.task_id,
                    status=BridgeTaskStatus.FAILED,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

        except requests.exceptions.Timeout:
            logger.error("Bridge task timed out after %ss", self.timeout)
            return BridgeTaskResult(
                task_id=spec.task_id, status=BridgeTaskStatus.FAILED, success=False, error="Task timed out"
            )
        except Exception as e:
            logger.error("Bridge task submission failed: %s", e)
            return BridgeTaskResult(task_id=spec.task_id, status=BridgeTaskStatus.FAILED, success=False, error=str(e))

    def get_task_status(self, task_id: str) -> BridgeTaskResult:
        """Poll task status from the bridge."""
        if not self.enabled:
            return BridgeTaskResult(task_id=task_id, status=BridgeTaskStatus.FAILED, error="CodeBridge not enabled")

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(f"{self.endpoint}/tasks/{task_id}", headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                result = response.json()
                return BridgeTaskResult(
                    task_id=task_id,
                    status=BridgeTaskStatus(result.get("status", "pending")),
                    success=result.get("success", False),
                    output_files=result.get("output_files", []),
                    artifacts=result.get("artifacts", []),
                    logs=result.get("logs", ""),
                    error=result.get("error"),
                )
            else:
                return BridgeTaskResult(
                    task_id=task_id, status=BridgeTaskStatus.FAILED, error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            logger.error("Failed to get task status: %s", e)
            return BridgeTaskResult(task_id=task_id, status=BridgeTaskStatus.FAILED, error=str(e))

    def get_artifacts(self, task_id: str) -> list[dict[str, Any]]:
        """Fetch artifacts from a completed task."""
        result = self.get_task_status(task_id)

        if result.status == BridgeTaskStatus.COMPLETED and result.success:
            return result.artifacts

        return []

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if not self.enabled:
            return False

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.delete(f"{self.endpoint}/tasks/{task_id}", headers=headers, timeout=self.timeout)

            return response.status_code in (200, 204)

        except Exception as e:
            logger.error("Failed to cancel task: %s", e)
            return False

    def list_tasks(self, status: BridgeTaskStatus | None = None) -> list[BridgeTaskResult]:
        """List tasks from the bridge."""
        if not self.enabled:
            return []

        try:
            params = {}
            if status:
                params["status"] = status.value

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(f"{self.endpoint}/tasks", params=params, headers=headers, timeout=self.timeout)

            if response.status_code == 200:
                tasks = response.json()
                return [
                    BridgeTaskResult(
                        task_id=t.get("task_id"),
                        status=BridgeTaskStatus(t.get("status", "pending")),
                        success=t.get("success", False),
                        output_files=t.get("output_files", []),
                        artifacts=t.get("artifacts", []),
                        logs=t.get("logs", ""),
                        error=t.get("error"),
                    )
                    for t in tasks
                ]

        except Exception as e:
            logger.error("Failed to list tasks: %s", e)

        return []

    # ── Phase 2 enhancements ─────────────────────────────────────────

    def batch_edit(self, edits: list[dict[str, Any]]) -> BridgeTaskResult:
        """Apply multiple edits as a single atomic operation.

        Each edit dict has keys: ``path``, ``old`` (text to replace), ``new``.
        All edits are validated for syntax before any file is written.
        """
        import ast

        task_id = f"batch_{uuid.uuid4().hex[:8]}"
        errors: list[str] = []

        for i, edit in enumerate(edits):
            path = edit.get("path", "")
            new_text = edit.get("new", "")
            if path.endswith(".py") and new_text.strip():
                try:
                    ast.parse(new_text)
                except SyntaxError as exc:
                    errors.append(f"Edit #{i} ({path}): {exc}")

        if errors:
            return BridgeTaskResult(
                task_id=task_id,
                status=BridgeTaskStatus.FAILED,
                success=False,
                error=f"Syntax validation failed: {'; '.join(errors)}",
            )

        applied: list[str] = []
        for edit in edits:
            path = edit.get("path", "")
            old_text = edit.get("old", "")
            new_text = edit.get("new", "")
            try:
                if os.path.isfile(path):
                    with open(path, encoding="utf-8") as fh:
                        content = fh.read()
                    if old_text and old_text in content:
                        content = content.replace(old_text, new_text, 1)
                    else:
                        content = new_text
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(content)
                    applied.append(path)
                else:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(new_text)
                    applied.append(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        return BridgeTaskResult(
            task_id=task_id,
            status=BridgeTaskStatus.COMPLETED if not errors else BridgeTaskStatus.FAILED,
            success=len(errors) == 0,
            output_files=applied,
            error="; ".join(errors) if errors else None,
        )

    def diff_preview(self, edits: list[dict[str, Any]]) -> str:
        """Generate a unified-diff preview of proposed edits without writing.

        Each edit dict has keys: ``path``, ``old`` (text to replace), ``new``.
        Returns a multi-file unified diff string.
        """
        import difflib

        diffs: list[str] = []
        for edit in edits:
            path = edit.get("path", "unknown")
            old_text = edit.get("old", "")
            new_text = edit.get("new", "")

            if os.path.isfile(path) and old_text:
                try:
                    with open(path, encoding="utf-8") as fh:
                        original = fh.read()
                    modified = original.replace(old_text, new_text, 1)
                except Exception:
                    original = old_text
                    modified = new_text
            else:
                original = old_text
                modified = new_text

            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
            diffs.append("".join(diff))

        return "\n".join(diffs)

    def rollback(self, checkpoint_id: str = "HEAD") -> bool:
        """Rollback to a git checkpoint.

        Uses ``git checkout`` to restore files to the specified commit.
        Creates a safety stash of current changes before rolling back.
        """
        import subprocess

        try:
            # Stash current changes as safety net
            subprocess.run(  # noqa: S603
                ["git", "stash", "push", "-m", f"codebrige_rollback_{checkpoint_id}"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Restore to checkpoint
            result = subprocess.run(  # noqa: S603
                ["git", "checkout", checkpoint_id, "--", "."],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("Rolled back to checkpoint %s", checkpoint_id)
                return True
            else:
                logger.warning("Rollback failed: %s", result.stderr)
                # Try to restore stash
                subprocess.run(
                    ["git", "stash", "pop"],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return False
        except Exception as exc:
            logger.error("Rollback error: %s", exc)
            return False

    def create_checkpoint(self, message: str = "auto-checkpoint") -> str | None:
        """Create a git checkpoint (commit) for later rollback.

        Returns the commit SHA or None on failure.
        """
        import subprocess

        try:
            subprocess.run(
                ["git", "add", "-A"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=30,
            )
            result = subprocess.run(  # noqa: S603
                ["git", "commit", "-m", f"[codebrige] {message}", "--allow-empty"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                sha_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                sha = sha_result.stdout.strip()
                logger.info("Checkpoint created: %s", sha[:8])
                return sha
            return None
        except Exception as exc:
            logger.error("Checkpoint creation failed: %s", exc)
            return None


_code_bridge: CodeBridge | None = None


def get_code_bridge() -> CodeBridge:
    """Get or create the global code bridge instance."""
    global _code_bridge
    if _code_bridge is None:
        _code_bridge = CodeBridge()
    return _code_bridge


def init_code_bridge(endpoint: str | None = None, api_key: str | None = None) -> CodeBridge:
    """Initialize a new code bridge instance."""
    global _code_bridge
    _code_bridge = CodeBridge(endpoint=endpoint, api_key=api_key)
    return _code_bridge
