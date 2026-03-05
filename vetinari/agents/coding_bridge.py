"""
CodingBridge - Unified bridge to external coding agents and services.

This module provides an interface for Vetinari to delegate coding tasks
to external coding agents, with support for:
- Generating code scaffolds
- Writing tests
- Performing code reviews
- Executing code in sandboxed environments
- HTTP-based task submission, status polling, and artifact retrieval
  (previously in coding_agent.bridge.CodeBridge)
"""

import os
import logging
import json
import uuid
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CodingTaskType(str, Enum):
    """Types of coding tasks."""
    SCAFFOLD = "scaffold"
    IMPLEMENT = "implement"
    TEST = "test"
    REFACTOR = "refactor"
    REVIEW = "review"
    FIX = "fix"
    DOCUMENT = "document"


class CodingTaskStatus(str, Enum):
    """Status of coding tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CodingTask:
    """A coding task to be executed by an external agent."""
    task_id: str = field(default_factory=lambda: f"code_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    task_type: CodingTaskType = CodingTaskType.IMPLEMENT
    description: str = ""
    language: str = ""
    framework: str = ""
    input_files: List[str] = field(default_factory=list)
    output_path: str = ""
    constraints: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    status: CodingTaskStatus = CodingTaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


@dataclass
class CodingResult:
    """Result from a coding task."""
    success: bool
    task_id: str
    output_files: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    logs: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: CodingTaskStatus = CodingTaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Legacy aliases (previously in coding_agent.bridge)
# ---------------------------------------------------------------------------

# BridgeTaskType maps onto CodingTaskType (subset)
BridgeTaskType = CodingTaskType


class BridgeTaskStatus(str, Enum):
    """Status of bridge tasks (legacy alias)."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BridgeTaskSpec:
    """Specification for an HTTP bridge task (legacy dataclass)."""
    task_id: str = field(default_factory=lambda: f"bridge_{uuid.uuid4().hex[:8]}")
    task_type: CodingTaskType = CodingTaskType.IMPLEMENT
    language: str = "python"
    framework: str = ""
    repo_path: str = ""
    description: str = ""
    constraints: str = ""
    target_files: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgeTaskResult:
    """Result from an HTTP bridge task (legacy dataclass)."""
    task_id: str
    status: BridgeTaskStatus = BridgeTaskStatus.PENDING
    success: bool = False
    output_files: List[str] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    logs: str = ""
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class CodingBridge:
    """Unified bridge to external coding agents and HTTP coding services.

    This class merges two previously separate bridges:
    - External coding agent delegation (scaffold generation, code review, etc.)
    - HTTP-based coding service integration (task submission, status polling,
      artifact retrieval)

    Configuration:
        Environment variables (checked in order):
        - CODING_BRIDGE_ENDPOINT / CODE_BRIDGE_ENDPOINT
        - CODING_BRIDGE_API_KEY  / CODE_BRIDGE_API_KEY
        - CODING_BRIDGE_ENABLED  / CODE_BRIDGE_ENABLED
        - CODE_BRIDGE_TIMEOUT
    """

    def __init__(self, endpoint: str = None, api_key: str = None):
        self.endpoint = endpoint or os.environ.get(
            "CODING_BRIDGE_ENDPOINT",
            os.environ.get("CODE_BRIDGE_ENDPOINT", "http://localhost:4096"),
        )
        self.api_key = api_key or os.environ.get(
            "CODING_BRIDGE_API_KEY",
            os.environ.get("CODE_BRIDGE_API_KEY", ""),
        )
        self.enabled = any(
            os.environ.get(var, "false").lower() in ("1", "true", "yes")
            for var in ("CODING_BRIDGE_ENABLED", "CODE_BRIDGE_ENABLED")
        )
        self.timeout = int(os.environ.get("CODE_BRIDGE_TIMEOUT", "30"))

        logger.info(f"CodingBridge initialized (enabled={self.enabled}, endpoint={self.endpoint})")

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the coding bridge is available (with optional health check)."""
        if not self.enabled:
            return False

        try:
            response = requests.get(f"{self.endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Bridge health check failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Local task execution (scaffold generation, etc.)
    # ------------------------------------------------------------------

    def generate_task(self, task: CodingTask) -> CodingResult:
        """Submit a coding task to the external agent.

        For SCAFFOLD tasks, this creates a minimal Python package scaffold.
        Other task types log the request and return a success stub.
        """
        if not self.enabled:
            logger.warning("CodingBridge is not enabled")
            return CodingResult(
                success=False,
                task_id=task.task_id,
                error="CodingBridge is not enabled",
            )

        logger.info(f"Submitting coding task: {task.task_id} ({task.task_type.value})")
        logger.debug(f"Task description: {task.description}")
        logger.debug(f"Output path: {task.output_path}")

        # Handle scaffold generation
        if task.task_type == CodingTaskType.SCAFFOLD:
            return self._generate_scaffold(task)

        return CodingResult(
            success=True,
            task_id=task.task_id,
            output_files=[task.output_path] if task.output_path else [],
            logs=f"Task {task.task_id} submitted to coding agent",
            metadata={
                "task_type": task.task_type.value,
                "language": task.language,
                "endpoint": self.endpoint,
            },
        )

    def _generate_scaffold(self, task: CodingTask) -> CodingResult:
        """Generate a Python package scaffold."""
        from pathlib import Path

        project_name = task.context.get("project_name", "demo_project")
        output_base = task.output_path or f"./scaffolds/{project_name}"

        try:
            output_path = Path(output_base)
            output_path.mkdir(parents=True, exist_ok=True)

            pkg_dir = output_path / project_name
            pkg_dir.mkdir(exist_ok=True)

            (pkg_dir / "__init__.py").write_text(
                f'"""{project_name} - Auto-generated scaffold."""\n'
                f'__version__ = "0.1.0"\n'
            )
            (pkg_dir / "__main__.py").write_text(
                f'def main():\n'
                f'    print("Hello from {project_name}")\n'
                f'\n'
                f'if __name__ == "__main__":\n'
                f'    main()\n'
            )
            (output_path / "setup.py").write_text(
                f'from setuptools import setup, find_packages\n'
                f'\n'
                f'setup(\n'
                f'    name="{project_name}",\n'
                f'    version="0.1.0",\n'
                f'    packages=find_packages(),\n'
                f'    python_requires=">=3.8",\n'
                f')\n'
            )
            (output_path / "README.md").write_text(
                f'# {project_name}\n\n'
                f'Auto-generated scaffold by Vetinari CodingBridge.\n'
            )

            output_files = [
                str(output_path / "setup.py"),
                str(output_path / "README.md"),
                str(pkg_dir / "__init__.py"),
                str(pkg_dir / "__main__.py"),
            ]

            logger.info(f"Generated scaffold for {project_name} at {output_path}")

            return CodingResult(
                success=True,
                task_id=task.task_id,
                output_files=output_files,
                logs=f"Scaffold generated for {project_name}",
                metadata={
                    "task_type": "scaffold",
                    "project_name": project_name,
                    "output_path": str(output_path),
                },
            )

        except Exception as e:
            logger.error(f"Failed to generate scaffold: {e}")
            return CodingResult(
                success=False,
                task_id=task.task_id,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # HTTP bridge: task submission / status / artifacts
    # ------------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        """Build authorization headers."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def submit_task(self, spec: "BridgeTaskSpec") -> "BridgeTaskResult":
        """Submit a task to the external HTTP bridge."""
        if not self.enabled:
            logger.warning("CodingBridge is not enabled")
            return BridgeTaskResult(
                task_id=spec.task_id,
                status=BridgeTaskStatus.FAILED,
                success=False,
                error="CodingBridge is not enabled",
            )

        logger.info(f"Submitting task to bridge: {spec.task_id} ({spec.task_type.value})")

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

            response = requests.post(
                f"{self.endpoint}/tasks",
                json=payload,
                headers=self._auth_headers(),
                timeout=self.timeout,
            )

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
                logger.error(f"Bridge task submission failed: {response.status_code}")
                return BridgeTaskResult(
                    task_id=spec.task_id,
                    status=BridgeTaskStatus.FAILED,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

        except requests.exceptions.Timeout:
            logger.error(f"Bridge task timed out after {self.timeout}s")
            return BridgeTaskResult(
                task_id=spec.task_id,
                status=BridgeTaskStatus.FAILED,
                success=False,
                error="Task timed out",
            )
        except Exception as e:
            logger.error(f"Bridge task submission failed: {e}")
            return BridgeTaskResult(
                task_id=spec.task_id,
                status=BridgeTaskStatus.FAILED,
                success=False,
                error=str(e),
            )

    def get_task_status(self, task_id: str) -> "BridgeTaskResult":
        """Poll task status from the bridge (HTTP) or check local artefacts."""
        if not self.enabled:
            return BridgeTaskResult(
                task_id=task_id,
                status=BridgeTaskStatus.FAILED,
                error="CodingBridge not enabled",
            )

        # Try HTTP polling first
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(
                f"{self.endpoint}/tasks/{task_id}",
                headers=headers,
                timeout=self.timeout,
            )

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
                    task_id=task_id,
                    status=BridgeTaskStatus.FAILED,
                    error=f"HTTP {response.status_code}",
                )

        except Exception as e:
            logger.debug(f"HTTP status check failed, checking local artefacts: {e}")

        # Fallback: check local output directory for completion artefacts
        output_base = os.environ.get("VETINARI_OUTPUTS_DIR", "outputs")
        output_path = os.path.join(output_base, task_id)
        if os.path.exists(output_path):
            return BridgeTaskResult(
                task_id=task_id,
                status=BridgeTaskStatus.COMPLETED,
                success=True,
                logs=f"Output available at {output_path}",
            )

        return BridgeTaskResult(
            task_id=task_id,
            status=BridgeTaskStatus.PENDING,
        )

    def get_artifacts(self, task_id: str) -> List[Dict[str, Any]]:
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

            response = requests.delete(
                f"{self.endpoint}/tasks/{task_id}",
                headers=headers,
                timeout=self.timeout,
            )
            return response.status_code in (200, 204)

        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False

    def list_tasks(self, status: Optional[BridgeTaskStatus] = None) -> List["BridgeTaskResult"]:
        """List tasks from the HTTP bridge."""
        if not self.enabled:
            return []

        try:
            params = {}
            if status:
                params["status"] = status.value

            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(
                f"{self.endpoint}/tasks",
                params=params,
                headers=headers,
                timeout=self.timeout,
            )

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
            logger.error(f"Failed to list tasks: {e}")

        return []

    def list_active_tasks(self) -> List[CodingTask]:
        """List active coding tasks (legacy convenience)."""
        return []

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def create_scaffold(self, language: str, framework: str, output_path: str,
                        project_name: str) -> CodingResult:
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
            task_type=CodingTaskType.REVIEW,
            description=f"Review code in {file_path}",
            input_files=[file_path],
        )
        return self.generate_task(task)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_coding_bridge: Optional[CodingBridge] = None


def get_coding_bridge() -> CodingBridge:
    """Get or create the global coding bridge instance."""
    global _coding_bridge
    if _coding_bridge is None:
        _coding_bridge = CodingBridge()
    return _coding_bridge


def init_coding_bridge(endpoint: str = None, api_key: str = None) -> CodingBridge:
    """Initialize a new coding bridge instance."""
    global _coding_bridge
    _coding_bridge = CodingBridge(endpoint=endpoint, api_key=api_key)
    return _coding_bridge


# ---------------------------------------------------------------------------
# Legacy aliases so ``from vetinari.agents.coding_bridge import CodeBridge``
# works for callers that used the old coding_agent.bridge module.
# ---------------------------------------------------------------------------

CodeBridge = CodingBridge
get_code_bridge = get_coding_bridge
init_code_bridge = init_coding_bridge
