"""
CodingBridge - Bridge to external coding agents (e.g., CodeNomad).

This module provides an interface for Vetinari to delegate coding tasks
to external coding agents, with support for:
- Generating code scaffolds
- Writing tests
- Performing code reviews
- Executing code in sandboxed environments
"""

import os
import logging
import json
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
    logs: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodingBridge:
    """Bridge to external coding agents.
    
    This is a skeleton implementation. Actual integration with
    CodeNomad or other coding agents requires implementing
    the specific API calls.
    """
    
    def __init__(self, endpoint: str = None, api_key: str = None):
        self.endpoint = endpoint or os.environ.get("CODING_BRIDGE_ENDPOINT", "http://localhost:4096")
        self.api_key = api_key or os.environ.get("CODING_BRIDGE_API_KEY", "")
        self.enabled = os.environ.get("CODING_BRIDGE_ENABLED", "false").lower() in ("1", "true", "yes")
        
        logger.info(f"CodingBridge initialized (enabled={self.enabled}, endpoint={self.endpoint})")
    
    def is_available(self) -> bool:
        """Check if the coding bridge is available."""
        if not self.enabled:
            return False
        return True
    
    def generate_task(self, task: CodingTask) -> CodingResult:
        """Submit a coding task to the external agent.
        
        This is a placeholder that logs the task. Actual implementation
        would call the external coding agent API.
        
        For SCAFFOLD tasks, this creates a minimal Python package scaffold.
        """
        if not self.enabled:
            logger.warning("CodingBridge is not enabled")
            return CodingResult(
                success=False,
                task_id=task.task_id,
                error="CodingBridge is not enabled"
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
                "endpoint": self.endpoint
            }
        )
    
    def _generate_scaffold(self, task: CodingTask) -> CodingResult:
        """Generate a Python package scaffold.
        
        This creates a minimal package structure for demonstration.
        """
        import shutil
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
                f'"""{project_name} - Auto-generated scaffold."""\n'
                f'__version__ = "0.1.0"\n'
            )
            
            # Create __main__.py
            (pkg_dir / "__main__.py").write_text(
                f'def main():\n'
                f'    print("Hello from {project_name}")\n'
                f'\n'
                f'if __name__ == "__main__":\n'
                f'    main()\n'
            )
            
            # Create setup.py
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
            
            # Create README.md
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
                    "output_path": str(output_path)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to generate scaffold: {e}")
            return CodingResult(
                success=False,
                task_id=task.task_id,
                error=str(e)
            )
    
    def get_task_status(self, task_id: str) -> Optional[CodingTask]:
        """Get the status of a coding task from the active task registry."""
        logger.debug(f"Checking status for task: {task_id}")
        # Check in-memory task registry if available
        if hasattr(self, "_active_tasks") and task_id in self._active_tasks:
            return self._active_tasks[task_id]
        # Check output directory for completion artefacts
        import os
        output_base = os.environ.get("VETINARI_OUTPUTS_DIR", "outputs")
        output_path = os.path.join(output_base, task_id)
        if os.path.exists(output_path):
            return CodingTask(
                task_id=task_id,
                status=CodingTaskStatus.COMPLETED,
                result=f"Output available at {output_path}"
            )
        return CodingTask(
            task_id=task_id,
            status=CodingTaskStatus.PENDING,
            result=""
        )
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a coding task."""
        logger.info(f"Cancelling task: {task_id}")
        return True
    
    def list_active_tasks(self) -> List[CodingTask]:
        """List active coding tasks."""
        return []
    
    def create_scaffold(self, language: str, framework: str, output_path: str, 
                       project_name: str) -> CodingResult:
        """Create a project scaffold."""
        task = CodingTask(
            task_type=CodingTaskType.SCAFFOLD,
            description=f"Create {language}/{framework} scaffold for {project_name}",
            language=language,
            framework=framework,
            output_path=output_path,
            context={"project_name": project_name}
        )
        return self.generate_task(task)
    
    def write_tests(self, source_file: str, test_framework: str = "pytest") -> CodingResult:
        """Generate tests for a source file."""
        task = CodingTask(
            task_type=CodingTaskType.TEST,
            description=f"Write tests for {source_file}",
            input_files=[source_file],
            context={"test_framework": test_framework}
        )
        return self.generate_task(task)
    
    def review_code(self, file_path: str) -> CodingResult:
        """Request code review."""
        task = CodingTask(
            task_type=CodingTaskType.REVIEW,
            description=f"Review code in {file_path}",
            input_files=[file_path]
        )
        return self.generate_task(task)


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
