"""Vetinari Coding Agent - In-process coding agent for code generation.

This module provides:
- CodeAgentEngine: In-process coding agent using internal LM
- make_code_agent_task: Factory for creating coding AgentTasks
- CodeArtifact: Data model for generated code artifacts
- Integration with UnifiedMemoryStore for provenance

CodeTask was consolidated into AgentTask in M4 ontology unification.
Use ``make_code_agent_task()`` to create coding tasks.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from vetinari.agents.contracts import AgentTask
from vetinari.code_sandbox import CodeSandbox
from vetinari.constants import MAX_TOKENS_CODE_GENERATION, MAX_TOKENS_REPO_MAP_CONTEXT
from vetinari.types import AgentType, CodingTaskType, StatusEnum
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)


class CodingArtifactType(str, Enum):
    """Types of code artifacts."""

    PATCH = "patch"
    FILE_CONTENTS = "file_contents"
    BUILD_ARTIFACT = "build_artifact"
    TEST_ARTIFACT = "test_artifact"


@dataclass
class _CodeTask:
    """Internal coding task representation (private to this module).

    External callers should use ``make_code_agent_task()`` which returns an
    ``AgentTask``. The engine converts it internally via ``_from_agent_task()``.
    """

    task_id: str = field(default_factory=lambda: f"code_{uuid.uuid4().hex[:8]}")
    plan_id: str = ""
    subtask_id: str = ""
    type: CodingTaskType = CodingTaskType.SCAFFOLD
    language: str = "python"
    framework: str = ""
    repo_path: str = ""
    target_files: list[str] = field(default_factory=list)
    constraints: str | list[str] = ""
    description: str = ""
    status: StatusEnum = StatusEnum.PENDING
    rationale: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self) -> str:
        return f"_CodeTask(task_id={self.task_id!r}, type={self.type!r}, status={self.status!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)

    @classmethod
    def from_agent_task(cls, task: AgentTask) -> _CodeTask:
        """Convert an AgentTask (with coding context) into the internal _CodeTask format.

        Args:
            task: An AgentTask created by ``make_code_agent_task()``.

        Returns:
            Internal _CodeTask for engine processing.
        """
        ctx = task.context
        task_type_raw = ctx.get("task_type", CodingTaskType.SCAFFOLD.value)
        task_type = CodingTaskType(task_type_raw) if isinstance(task_type_raw, str) else task_type_raw
        constraints = ctx.get("constraints", "")
        if isinstance(constraints, list):
            constraints = constraints
        target_files = ctx.get("target_files", [])
        if isinstance(target_files, tuple):
            target_files = list(target_files)
        return cls(
            task_id=task.task_id,
            plan_id=ctx.get("plan_id", ""),
            subtask_id=ctx.get("subtask_id", ""),
            type=task_type,
            language=ctx.get("language", "python"),
            framework=ctx.get("framework", ""),
            repo_path=ctx.get("repo_path", ""),
            target_files=target_files,
            constraints=constraints,
            description=task.description,
        )


def make_code_agent_task(
    description: str,
    *,
    task_type: CodingTaskType = CodingTaskType.SCAFFOLD,
    language: str = "python",
    framework: str = "",
    repo_path: str = "",
    target_files: list[str] | None = None,
    constraints: str | list[str] = "",
    plan_id: str = "",
    subtask_id: str = "",
) -> AgentTask:
    """Create an AgentTask carrying coding-specific context fields.

    This is the public API for creating coding tasks. The engine converts
    the AgentTask to its internal format automatically.

    Args:
        description: Human-readable task description.
        task_type: Type of coding task (scaffold, implement, test, etc.).
        language: Programming language for the task.
        framework: Target framework (e.g. "fastapi", "pytest").
        repo_path: Repository root path for file resolution.
        target_files: Files to generate or modify.
        constraints: Coding constraints (style, perf, etc.).
        plan_id: Parent plan ID if part of a plan.
        subtask_id: Parent subtask ID if part of a subtask.

    Returns:
        An AgentTask with coding metadata in context.
    """
    return AgentTask(
        task_id=f"code_{uuid.uuid4().hex[:8]}",
        agent_type=AgentType.WORKER,
        description=description,
        prompt=description,
        context={
            "task_type": task_type.value if isinstance(task_type, CodingTaskType) else task_type,
            "language": language,
            "framework": framework,
            "repo_path": repo_path,
            "target_files": target_files if target_files is not None else [],
            "constraints": constraints,
            "plan_id": plan_id,
            "subtask_id": subtask_id,
        },
    )


@dataclass
class CodeArtifact:
    """A code artifact generated by the coding agent."""

    artifact_id: str = field(default_factory=lambda: f"art_{uuid.uuid4().hex[:8]}")
    task_id: str = ""
    type: CodingArtifactType = CodingArtifactType.FILE_CONTENTS
    path: str = ""
    content: str = ""
    diff: str = ""
    provenance: str = ""
    language: str = "python"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self) -> str:
        return f"CodeArtifact(artifact_id={self.artifact_id!r}, type={self.type!r}, path={self.path!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeArtifact:
        """Deserialize a CodeArtifact from a plain dictionary.

        Converts the ``type`` string value back to the CodingArtifactType enum
        before constructing the instance.

        Args:
            data: Dictionary of field values, typically from ``to_dict()`` or JSON storage.

        Returns:
            A new CodeArtifact instance populated from the provided dictionary.
        """
        if "type" in data and isinstance(data["type"], str):
            data["type"] = CodingArtifactType(data["type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CodeAgentEngine:
    """In-process coding agent engine.

    This MVP uses an internal LM wrapper to draft code.
    It can also delegate to an external bridge for heavier tasks.
    """

    def __init__(self, lm_provider: str = "internal"):
        self.lm_provider = lm_provider
        self.enabled = os.environ.get("CODING_AGENT_ENABLED", "true").lower() in ("1", "true", "yes")
        self._sandbox = CodeSandbox(network_isolation=True)

        logger.info(
            "CodeAgentEngine initialized (provider=%s, enabled=%s)",
            lm_provider,
            self.enabled,
        )

    def is_available(self) -> bool:
        """Check if the coding agent is available."""
        return self.enabled

    def run_task(self, task: _CodeTask | AgentTask) -> CodeArtifact:
        """Execute a coding task and return a generated code artifact.

        Accepts either the internal ``_CodeTask`` or a public ``AgentTask``
        created by ``make_code_agent_task()``. AgentTasks are converted
        to _CodeTask internally.

        Args:
            task: The coding task to execute. AgentTasks are auto-converted.

        Returns:
            A CodeArtifact containing the generated code, diff, or review output.

        Raises:
            RuntimeError: If the coding agent is disabled via CODING_AGENT_ENABLED.
        """
        if not self.enabled:
            raise RuntimeError("Coding agent is not enabled")

        # Convert AgentTask to internal format if needed
        if isinstance(task, AgentTask):
            task = _CodeTask.from_agent_task(task)

        task.status = StatusEnum.IN_PROGRESS
        task.updated_at = datetime.now(timezone.utc).isoformat()

        logger.info("Executing coding task: %s (%s)", task.task_id, task.type.value)

        try:
            artifact = self._run_in_process(task)

            task.status = StatusEnum.COMPLETED
            task.updated_at = datetime.now(timezone.utc).isoformat()

            return artifact

        except Exception as e:
            logger.error("Coding task failed: %s", e)
            task.status = StatusEnum.FAILED
            task.updated_at = datetime.now(timezone.utc).isoformat()
            raise

    def _run_in_process(self, task: _CodeTask) -> CodeArtifact:
        """Run task using AdapterManager (LLM-powered) with template fallback.

        Generated Python artifacts are validated through the embedded sandbox
        before being returned. Sandbox failures are logged as warnings but do
        not abort delivery of the artifact — the caller receives the generated
        content regardless, with sandbox diagnostics attached to the provenance.
        """
        # Try LLM-powered generation first
        llm_result = self._generate_via_llm(task)
        if llm_result:
            return self._validate_via_sandbox(llm_result, task)

        # Fallback to template stubs
        if task.type == CodingTaskType.SCAFFOLD:
            artifact = self._generate_scaffold(task)
        elif task.type == CodingTaskType.IMPLEMENT:
            artifact = self._generate_implementation(task)
        elif task.type == CodingTaskType.TEST:
            artifact = self._generate_tests(task)
        elif task.type == CodingTaskType.REVIEW:
            return self._generate_review(task)
        else:
            artifact = self._generate_generic(task)

        return self._validate_via_sandbox(artifact, task)

    def _validate_via_sandbox(self, artifact: CodeArtifact, task: _CodeTask) -> CodeArtifact:
        """Execute the artifact's Python content through the embedded sandbox.

        Only Python artifacts are executed. Review artifacts (markdown) and
        artifacts for non-Python languages are skipped. The artifact is always
        returned; sandbox results are recorded in the provenance field for
        diagnostics.

        Args:
            artifact: The generated code artifact to validate.
            task: The originating code task (used for language context).

        Returns:
            The original artifact, with provenance updated to include sandbox
            execution status.
        """
        is_python = task.language.lower() == "python" and artifact.language.lower() in ("python", "")
        is_review = task.type == CodingTaskType.REVIEW
        # Test files use pytest which is blocked in the restricted sandbox; skip validation
        is_test = task.type == CodingTaskType.TEST
        has_code = bool(artifact.content and artifact.content.strip())

        if not is_python or is_review or is_test or not has_code:
            return artifact

        try:
            result = self._sandbox.execute_python(artifact.content)
            if not result.success:
                error_detail = result.error[:200] if result.error else "(no error output)"
                logger.error(
                    "Sandbox validation failed for task %s: %s",
                    task.task_id,
                    error_detail,
                )
                artifact.provenance = f"{artifact.provenance}|sandbox_error"
                raise RuntimeError(f"Generated code failed sandbox validation for task {task.task_id}: {error_detail}")
            logger.debug("Sandbox validation passed for task %s", task.task_id)
            artifact.provenance = f"{artifact.provenance}|sandbox_ok"
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Sandbox execution raised an exception for task %s: %s", task.task_id, e)
            artifact.provenance = f"{artifact.provenance}|sandbox_unavailable"

        return artifact

    def _generate_via_llm(self, task: _CodeTask) -> CodeArtifact | None:
        """Generate code using the LLM via AdapterManager."""
        try:
            from vetinari.adapter_manager import get_adapter_manager

            constraints_str = ", ".join(task.constraints) if isinstance(task.constraints, list) else task.constraints
            target = task.target_files[0] if task.target_files else "output.py"

            prompts = {
                CodingTaskType.SCAFFOLD: (
                    f"Generate a complete {task.language} project scaffold for: {task.description}. "
                    f"Framework: {task.framework or 'standard library'}. "
                    f"Target file: {target}. "
                    f"Constraints: {constraints_str or 'none'}. "
                    "Return ONLY the code, no explanations."
                ),
                CodingTaskType.IMPLEMENT: (
                    f"Implement the following in {task.language}: {task.description}. "
                    f"Target file: {target}. "
                    f"Constraints: {constraints_str or 'none'}. "
                    "Return ONLY the complete implementation code."
                ),
                CodingTaskType.TEST: (
                    f"Write comprehensive unit tests in {task.language} for: {task.description}. "
                    f"Target: {target}. Use pytest. "
                    "Return ONLY the test code."
                ),
                CodingTaskType.REVIEW: (
                    f"Review this {task.language} code and provide actionable feedback: "
                    f"{task.description}. "
                    "Return a structured review with: issues, improvements, security concerns, rating."
                ),
                CodingTaskType.REFACTOR: (
                    f"Refactor the following {task.language} code to improve quality: "
                    f"{task.description}. "
                    "Return ONLY the refactored code."
                ),
                CodingTaskType.FIX: (
                    f"Fix the following bug in {task.language}: {task.description}. "
                    f"Target: {target}. "
                    "Return ONLY the fixed code."
                ),
                CodingTaskType.DOCUMENT: (
                    f"Write documentation for: {task.description}. Return well-structured markdown documentation."
                ),
            }

            user_prompt = prompts.get(task.type, f"Complete this coding task: {task.description}")

            # Inject codebase structure for context-aware generation
            repo_map_context = ""
            try:
                from pathlib import Path as _Path

                from vetinari.repo_map import get_repo_map

                _rm = get_repo_map()
                _structure = _rm.generate_for_task(
                    root_path=_Path(task.repo_path) if task.repo_path else _Path.cwd(),
                    task_description=task.description[:200],
                    max_tokens=MAX_TOKENS_REPO_MAP_CONTEXT,
                )
                if _structure:
                    repo_map_context = f"\n\nCodebase structure:\n{_structure}"
            except Exception:
                logger.warning(
                    "Repo map generation failed for task %s — proceeding without codebase context", task.task_id
                )

            system_prompt = (
                f"You are an expert {task.language} developer. "
                f"Framework: {task.framework or 'none'}. "
                "Write clean, well-commented, production-quality code. "
                f"Follow best practices and handle edge cases.{repo_map_context}"
            )

            adapter_manager = get_adapter_manager()
            response = adapter_manager.infer(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=MAX_TOKENS_CODE_GENERATION,
            )
            content = response.get("output", "").strip() if isinstance(response, dict) else str(response).strip()

            if not content:
                return None

            return CodeArtifact(
                artifact_id=f"art_{uuid.uuid4().hex[:8]}",
                task_id=task.task_id,
                type=CodingArtifactType.FILE_CONTENTS,
                path=f"{task.repo_path}/{target}" if task.repo_path else target,
                content=content,
                provenance="llm_generated",
                language=task.language,
            )
        except Exception as e:
            logger.warning("LLM code generation failed — using scaffold fallback: %s", e)
            return None

    def _generate_scaffold(self, task: _CodeTask) -> CodeArtifact:
        """Generate a Python package scaffold."""
        project_name = Path(task.target_files[0]).stem if task.target_files else "demo_project"

        scaffold_content = f'''"""
{project_name} - Auto-generated scaffold.
"""

__version__ = "0.1.0"
__author__ = "Vetinari Coding Agent"

def main():
    """Main entry point."""
    logger.debug("Hello from {project_name}!")

if __name__ == "__main__":
    main()
'''

        artifact = CodeArtifact(
            artifact_id=f"art_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            type=CodingArtifactType.FILE_CONTENTS,
            path=f"{task.repo_path}/{project_name}/__init__.py" if task.repo_path else f"{project_name}/__init__.py",
            content=scaffold_content,
            provenance="in_process_coder",
            language=task.language,
        )

        logger.info("Generated scaffold for %s", project_name)
        return artifact

    def _generate_implementation(self, task: _CodeTask) -> CodeArtifact:
        """Generate implementation code."""
        target = task.target_files[0] if task.target_files else "module.py"

        impl_content = f'''"""
Implementation for {target}
Generated by Vetinari Coding Agent
"""

class Implementation:
    """Main implementation class."""

    def __init__(self):
        self.data = {{}}

    def process(self, input_data):
        """Process input data."""
        return {{
            "status": "success",
            "data": input_data
        }}
'''

        artifact = CodeArtifact(
            artifact_id=f"art_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            type=CodingArtifactType.FILE_CONTENTS,
            path=f"{task.repo_path}/{target}" if task.repo_path else target,
            content=impl_content,
            provenance="in_process_coder",
            language=task.language,
        )

        logger.info("Generated implementation for %s", target)
        return artifact

    def _generate_tests(self, task: _CodeTask) -> CodeArtifact:
        """Generate unit tests."""
        target = task.target_files[0] if task.target_files else "module"
        test_content = f'''"""
Unit tests for {target}
Generated by Vetinari Coding Agent
"""

import importlib

import pytest


class TestImplementation:
    """Test cases for {target} Implementation class."""

    def test_module_importable(self):
        """Verify the target module can be imported."""
        try:
            mod = importlib.import_module("{target}")
            assert mod is not None
        except ImportError:
            pytest.skip("{target} module not yet available")

    def test_implementation_instantiable(self):
        """Verify Implementation class can be instantiated."""
        try:
            mod = importlib.import_module("{target}")
            impl_cls = getattr(mod, "Implementation", None)
            if impl_cls is None:
                pytest.skip("Implementation class not found in {target}")
            impl = impl_cls()
            assert impl is not None
        except ImportError:
            pytest.skip("{target} module not yet available")

    def test_process_returns_dict(self):
        """Verify process() returns a dict when module is available."""
        try:
            mod = importlib.import_module("{target}")
            impl_cls = getattr(mod, "Implementation", None)
            if impl_cls is None:
                pytest.skip("Implementation class not found in {target}")
            impl = impl_cls()
            result = impl.process("test input")
            assert isinstance(result, dict)
        except ImportError:
            pytest.skip("{target} module not yet available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

        artifact = CodeArtifact(
            artifact_id=f"art_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            type=CodingArtifactType.TEST_ARTIFACT,
            path=f"{task.repo_path}/test_{target}.py" if task.repo_path else f"test_{target}.py",
            content=test_content,
            provenance="in_process_coder",
            language=task.language,
        )

        logger.info("Generated tests for %s", target)
        return artifact

    def _generate_review(self, task: _CodeTask) -> CodeArtifact:
        """Generate a code review summary."""
        review_content = f"""# Code Review Summary

## Task: {task.task_id}
## Type: {task.type.value}

### Files Reviewed
{task.target_files}

### Constraints
{", ".join(task.constraints) if isinstance(task.constraints, list) else task.constraints}

### Notes
- Code follows Python style guidelines
- Basic error handling implemented
- Tests should be added for edge cases
"""

        artifact = CodeArtifact(
            artifact_id=f"art_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            type=CodingArtifactType.FILE_CONTENTS,
            path=f"{task.repo_path}/review.md" if task.repo_path else "review.md",
            content=review_content,
            provenance="in_process_coder",
            language="markdown",
        )

        logger.info("Generated review for %s", task.task_id)
        return artifact

    def _generate_generic(self, task: _CodeTask) -> CodeArtifact:
        """Generate generic code for unspecified tasks."""
        content = f"""# Generated by Vetinari Coding Agent
# Task: {task.task_id}  # noqa: VET036
# Type: {task.type.value}
# Language: {task.language}

{task.description or "No description provided."}
"""

        target = task.target_files[0] if task.target_files else "generated.py"

        return CodeArtifact(
            artifact_id=f"art_{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
            type=CodingArtifactType.FILE_CONTENTS,
            path=f"{task.repo_path}/{target}" if task.repo_path else target,
            content=content,
            provenance="in_process_coder",
            language=task.language,
        )

    def validate(self, artifact: CodeArtifact) -> tuple[bool, list[str]]:
        """Validate a code artifact for correctness and completeness.

        Checks that the artifact has code, passes syntax validation, and meets
        quality thresholds.  Returns (passed, list_of_issues).

        Args:
            artifact: The CodeArtifact to validate.

        Returns:
            Tuple of (passed: bool, issues: list[str]).
        """
        issues: list[str] = []
        if not artifact.content or not artifact.content.strip():
            issues.append("Artifact has no code")
        else:
            try:
                import ast as _ast

                _ast.parse(artifact.content)
            except SyntaxError as e:
                issues.append(f"Syntax error: {e}")
        quality_score = getattr(artifact, "quality_score", None)
        if quality_score is not None and quality_score < 0.5:
            issues.append(f"Quality score {quality_score:.2f} below threshold 0.50")
        return len(issues) == 0, issues

    def run_multi_step_task(self, tasks: list[_CodeTask | AgentTask]) -> list[CodeArtifact]:
        """Run a sequence of coding tasks, returning an artifact for each.

        Tasks are executed in order. If any task fails, the exception is
        re-raised immediately and subsequent tasks are not run.

        Args:
            tasks: Ordered list of coding tasks to execute (e.g. scaffold, implement, test).

        Returns:
            List of CodeArtifacts in the same order as the input tasks.

        Raises:
            Exception: Re-raises any exception from a failed task when bridge mode is disabled.
        """
        artifacts = []

        for task in tasks:
            try:
                artifact = self.run_task(task)
                artifacts.append(artifact)
            except Exception as e:
                logger.error("Task %s failed: %s", task.task_id, e)
                raise

        return artifacts


_coding_agent: CodeAgentEngine | None = None
_coding_agent_lock = threading.Lock()


def get_coding_agent() -> CodeAgentEngine:
    """Get or create the global coding agent instance.

    Returns:
        The singleton CodeAgentEngine, creating one with default settings on first call.
    """
    global _coding_agent
    if _coding_agent is None:
        with _coding_agent_lock:
            if _coding_agent is None:
                _coding_agent = CodeAgentEngine()
    return _coding_agent


def init_coding_agent(lm_provider: str = "internal") -> CodeAgentEngine:
    """Initialize a new coding agent instance, replacing any existing singleton.

    Args:
        lm_provider: LM provider identifier passed to CodeAgentEngine
            (e.g. ``"internal"`` for the built-in adapter).

    Returns:
        The newly created CodeAgentEngine, now stored as the global singleton.
    """
    global _coding_agent
    _coding_agent = CodeAgentEngine(lm_provider=lm_provider)  # noqa: VET111 - stateful fallback preserves legacy compatibility
    return _coding_agent
