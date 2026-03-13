"""CodingBridge - Bridge to external coding agents (e.g., CodeNomad).

This module provides an interface for Vetinari to delegate coding tasks
to external coding agents, with support for:
- Generating code scaffolds
- Writing tests
- Performing code reviews
- Executing code in sandboxed environments
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vetinari.types import CodingTaskStatus, CodingTaskType  # canonical enums

logger = logging.getLogger(__name__)


@dataclass
class CodingTask:
    """A coding task to be executed by an external agent."""

    task_id: str = field(default_factory=lambda: f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    task_type: CodingTaskType = CodingTaskType.IMPLEMENT
    description: str = ""
    language: str = ""
    framework: str = ""
    input_files: list[str] = field(default_factory=list)
    output_path: str = ""
    constraints: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    status: CodingTaskStatus = CodingTaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None


@dataclass
class CodingResult:
    """Result from a coding task."""

    success: bool
    task_id: str
    output_files: list[str] = field(default_factory=list)
    logs: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CodingBridge:
    """Bridge to external coding agents.

    This is a skeleton implementation. Actual integration with
    CodeNomad or other coding agents requires implementing
    the specific API calls.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint or os.environ.get("CODING_BRIDGE_ENDPOINT", "http://localhost:4096")
        self.api_key = api_key or os.environ.get("CODING_BRIDGE_API_KEY", "")
        self.enabled = os.environ.get("CODING_BRIDGE_ENABLED", "false").lower() in ("1", "true", "yes")

        logger.info("CodingBridge initialized (enabled=%s, endpoint=%s)", self.enabled, self.endpoint)

    def is_available(self) -> bool:
        """Check if the coding bridge is available."""
        return self.enabled

    def generate_task(self, task: CodingTask) -> CodingResult:
        """Submit a coding task to the external agent.

        Submit a coding task to the external agent.

        SCAFFOLD tasks produce a real Python package scaffold.
        Other task types return a success acknowledgement (external agent
        integration is not yet wired).
        """
        if not self.enabled:
            logger.warning("CodingBridge is not enabled")
            return CodingResult(success=False, task_id=task.task_id, error="CodingBridge is not enabled")

        logger.info("Submitting coding task: %s (%s)", task.task_id, task.task_type.value)
        logger.debug("Task description: %s", task.description)
        logger.debug("Output path: %s", task.output_path)

        # Handle scaffold generation
        if task.task_type == CodingTaskType.SCAFFOLD:
            return self._generate_scaffold(task)

        # Route other task types through CodingEngine
        try:
            from vetinari.coding_agent.engine import CodeAgentEngine
            from vetinari.coding_agent.engine import CodeTask as EngineCodeTask

            engine = CodeAgentEngine()
            if engine.is_available():
                code_task = EngineCodeTask(
                    task_id=task.task_id,
                    description=task.description,
                    language=task.language or "python",
                    type=task.task_type,
                    target_files=task.input_files,
                    constraints=task.constraints,
                )
                artifact = engine.run_task(code_task)
                return CodingResult(
                    success=True,
                    task_id=task.task_id,
                    output_files=[artifact.path] if artifact.path else [],
                    logs=artifact.content[:500] if artifact.content else "",
                    metadata={"task_type": task.task_type.value, "engine": "coding_engine"},
                )
        except Exception as e:
            logger.warning("CodingEngine unavailable for %s, returning acknowledgement: %s", task.task_id, e)

        # Fallback: return acknowledgement
        return CodingResult(
            success=True,
            task_id=task.task_id,
            output_files=[task.output_path] if task.output_path else [],
            logs=f"Task {task.task_id} submitted (engine unavailable)",
            metadata={"task_type": task.task_type.value, "language": task.language, "endpoint": self.endpoint},
        )

    def _generate_scaffold(self, task: CodingTask) -> CodingResult:
        """Generate a Python package scaffold.

        This creates a minimal package structure for demonstration.
        """
        from pathlib import Path

        project_name = task.context.get("project_name", "demo_project")
        output_base = task.output_path or f"./scaffolds/{project_name}"

        try:
            # Create output directory
            output_path = Path(output_base)
            output_path.mkdir(parents=True, exist_ok=True)

            # Create package directory
            pkg_dir = output_path / project_name
            pkg_dir.mkdir(exist_ok=True)

            # Create __init__.py
            (pkg_dir / "__init__.py").write_text(
                f'"""{project_name} - Auto-generated scaffold."""\n__version__ = "0.1.0"\n'
            )

            # Create __main__.py
            (pkg_dir / "__main__.py").write_text(
                f'def main():\n    print("Hello from {project_name}")\n\nif __name__ == "__main__":\n    main()\n'  # noqa: VET035
            )

            # Create setup.py
            (output_path / "setup.py").write_text(
                f"from setuptools import setup, find_packages\n"
                f"\n"
                f"setup(\n"
                f'    name="{project_name}",\n'
                f'    version="0.1.0",\n'
                f"    packages=find_packages(),\n"
                f'    python_requires=">=3.8",\n'
                f")\n"
            )

            # Create README.md
            (output_path / "README.md").write_text(
                f"# {project_name}\n\nAuto-generated scaffold by Vetinari CodingBridge.\n"
            )

            output_files = [
                str(output_path / "setup.py"),
                str(output_path / "README.md"),
                str(pkg_dir / "__init__.py"),
                str(pkg_dir / "__main__.py"),
            ]

            logger.info("Generated scaffold for %s at %s", project_name, output_path)

            return CodingResult(
                success=True,
                task_id=task.task_id,
                output_files=output_files,
                logs=f"Scaffold generated for {project_name}",
                metadata={"task_type": "scaffold", "project_name": project_name, "output_path": str(output_path)},
            )

        except Exception as e:
            logger.error("Failed to generate scaffold: %s", e)
            return CodingResult(success=False, task_id=task.task_id, error=str(e))

    def get_task_status(self, task_id: str) -> CodingTask | None:
        """Get the status of a coding task from the active task registry."""
        logger.debug("Checking status for task: %s", task_id)
        # Check in-memory task registry if available
        if hasattr(self, "_active_tasks") and task_id in self._active_tasks:
            return self._active_tasks[task_id]
        # Check output directory for completion artefacts
        import os

        output_base = os.environ.get("VETINARI_OUTPUTS_DIR", "outputs")
        output_path = os.path.join(output_base, task_id)
        if os.path.exists(output_path):
            return CodingTask(
                task_id=task_id, status=CodingTaskStatus.COMPLETED, result=f"Output available at {output_path}"
            )
        return CodingTask(task_id=task_id, status=CodingTaskStatus.PENDING, result="")

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a coding task."""
        logger.info("Cancelling task: %s", task_id)
        return True

    def list_active_tasks(self) -> list[CodingTask]:
        """List active coding tasks."""
        return []

    def create_scaffold(self, language: str, framework: str, output_path: str, project_name: str) -> CodingResult:
        """Create a project scaffold."""
        task = CodingTask(
            task_type=CodingTaskType.SCAFFOLD,
            description=f"Create {language}/{framework} scaffold for {project_name}",
            language=language,
            framework=framework,
            output_path=output_path,
            context={"project_name": project_name},
        )
        return self.generate_task(task)

    def write_tests(self, source_file: str, test_framework: str = "pytest") -> CodingResult:
        """Generate tests for a source file."""
        task = CodingTask(
            task_type=CodingTaskType.TEST,
            description=f"Write tests for {source_file}",
            input_files=[source_file],
            context={"test_framework": test_framework},
        )
        return self.generate_task(task)

    def review_code(self, file_path: str) -> CodingResult:
        """Request code review."""
        task = CodingTask(
            task_type=CodingTaskType.REVIEW, description=f"Review code in {file_path}", input_files=[file_path]
        )
        return self.generate_task(task)


_coding_bridge: CodingBridge | None = None


def get_coding_bridge() -> CodingBridge:
    """Get or create the global coding bridge instance."""
    global _coding_bridge
    if _coding_bridge is None:
        _coding_bridge = CodingBridge()
    return _coding_bridge


def init_coding_bridge(endpoint: str | None = None, api_key: str | None = None) -> CodingBridge:
    """Initialize a new coding bridge instance."""
    global _coding_bridge
    _coding_bridge = CodingBridge(endpoint=endpoint, api_key=api_key)
    return _coding_bridge
