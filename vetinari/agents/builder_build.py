"""Build-mode execution mixin for BuilderAgent.

Contains the ``BuilderBuildMixin`` class with methods for memory context
retrieval, scaffold generation, and build task execution. Extracted from
``builder_agent.py`` to keep it under the 550-line limit.

This module is step 1 of the Builder pipeline:
Intake → **Scaffold generation** → File I/O → Quality Gate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.config.inference_config import get_inference_config
from vetinari.constants import TEMPERATURE_LOW
from vetinari.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)


class BuilderBuildMixin:
    """Mixin providing build-mode execution for BuilderAgent.

    Requires the host class to provide:
    - ``self._infer(prompt, temperature)`` — raw LLM call
    - ``self._infer_json(prompt, temperature)`` — structured LLM call
    - ``self._log(level, msg, *args)`` — agent-aware logging
    - ``self._language`` — target language string
    - ``self._write_scaffold_to_disk(scaffold, output_dir)``
    - ``self._check_syntax(code)``
    """

    # -- Memory context retrieval --

    def _retrieve_memory_context(self, spec: str) -> str:
        """Retrieve read-only memory context relevant to the build task.

        Searches DECISION, PATTERN, and WARNING memory types to provide
        the Builder with awareness of prior decisions and known pitfalls.
        Builder reads memory but never writes to it (single-writer rule).

        Args:
            spec: The build specification to search against.

        Returns:
            A formatted string of relevant memory entries, or empty string
            if memory is unavailable or returns no results.
        """
        try:
            from vetinari.memory import get_unified_memory_store
            from vetinari.types import MemoryType

            store = get_unified_memory_store()
            query = spec[:200]  # truncate to reasonable search length
            results = store.search(query, limit=5)

            relevant_types = {
                MemoryType.DECISION,
                MemoryType.PATTERN,
                MemoryType.WARNING,
                MemoryType.SOLUTION,
            }

            lines: list[str] = []
            for entry in results:
                if entry.entry_type in relevant_types:
                    label = entry.entry_type.value.upper()
                    title = entry.title or ""
                    content_preview = (entry.content or "")[:300]
                    lines.append(f"[{label}] {title}: {content_preview}")

            if lines:
                return "\n".join(lines[:5])
        except Exception:
            logger.warning("Memory context retrieval unavailable; continuing without it", exc_info=True)
        return ""

    # -- Build task execution --

    def _execute_build(self, task: AgentTask) -> AgentResult:
        """Execute the build scaffolding task.

        Retrieves memory context, generates a scaffold via LLM, validates
        the output, and optionally writes files to disk.

        Args:
            task: The AgentTask describing what to scaffold.

        Returns:
            AgentResult with the scaffold output and metadata.
        """
        spec = task.context.get("spec", task.description)
        feature_name = task.context.get("feature_name", "feature")
        output_dir = task.context.get("output_dir", "")

        # Retrieve read-only memory context (decisions, patterns, warnings)
        memory_context = self._retrieve_memory_context(spec)
        if memory_context:
            spec = f"{spec}\n\nRelevant prior decisions and patterns:\n{memory_context}"
            logger.info("Builder enriched spec with %d memory entries", memory_context.count("["))

        scaffold = self._generate_scaffold(spec, feature_name)

        # Validate the generated scaffold code via CodeAgentEngine.validate()
        validation_issues: list[str] = []
        try:
            from vetinari.coding_agent.engine import CodeAgentEngine, CodeArtifact

            _engine = CodeAgentEngine()
            _artifact = CodeArtifact(content=scaffold.get("scaffold_code", ""), language=self._language)
            _valid, validation_issues = _engine.validate(_artifact)
            if not _valid:
                logger.warning("Code artifact validation failed: %s", "; ".join(validation_issues))
        except Exception as _exc:
            logger.warning("CodeAgentEngine validation unavailable: %s", _exc)

        written_files: list[str] = []
        if output_dir or task.context.get("write_files", False):
            written_files = self._write_scaffold_to_disk(scaffold, output_dir or ".")

        syntax_errors = self._check_syntax(scaffold.get("scaffold_code", ""))

        # Chain-of-verification: Worker generates verification questions
        # after code generation and answers them independently to catch
        # hallucinated imports before Inspector review.
        verification_issues: list[str] = []
        code_text = scaffold.get("scaffold_code", "")
        if code_text and not syntax_errors:
            try:
                from vetinari.config.inference_config import get_inference_config
                from vetinari.llm_helpers import quick_llm_call

                _vf_profile = get_inference_config().get_profile("verification")
                verification = quick_llm_call(
                    prompt=(
                        f"You just generated this code:\n```python\n{code_text[:2000]}\n```\n\n"
                        "Generate 3 verification questions about this code and answer each:\n"
                        "1. Does every import reference a real module?\n"
                        "2. Are all function signatures consistent with their call sites?\n"
                        "3. Are there any undefined variables or missing dependencies?\n"
                        "For each question, answer YES (ok) or NO (problem found). "
                        "If NO, describe the specific issue."
                    ),
                    system_prompt="You verify code correctness by answering self-check questions.",
                    max_tokens=_vf_profile.max_tokens,
                    temperature=_vf_profile.temperature,
                )
                if verification and "NO" in verification.upper():
                    verification_issues.append(verification)
                    logger.info("Chain-of-verification found potential issues in generated code")
            except Exception:
                logger.warning("Chain-of-verification unavailable for %s — skipping verification step", feature_name)

        return AgentResult(
            success=True,
            output=scaffold,
            metadata={
                "mode": "build",
                "feature_name": feature_name,
                "files_generated": len(scaffold.get("artifacts", [])),
                "test_count": len(scaffold.get("tests", [])),
                "written_files": written_files,
                "syntax_errors": syntax_errors,
                "validation_issues": validation_issues,
                "verification_issues": verification_issues,
            },
        )

    def _generate_scaffold(self, spec: str, feature_name: str) -> dict[str, Any]:
        """Generate code scaffold using LLM-powered code generation.

        Attempts structured JSON generation first, then falls back to plain
        text generation with a minimal skeleton.

        Args:
            spec: The feature specification to scaffold.
            feature_name: Human-readable name for the feature being generated.

        Returns:
            Scaffold dict with keys: ``scaffold_code``, ``tests``, ``artifacts``,
            ``implementation_notes``, and ``summary``.
        """
        # Inject codebase structure map for context-aware code generation
        repo_map_section = ""
        try:
            from vetinari.repo_map import get_repo_map

            repo_map = get_repo_map()
            cfg = get_inference_config().get_profile("coding")
            structure = repo_map.generate_for_task(
                root_path=Path.cwd(),
                task_description=f"{feature_name}: {spec[:200]}",
                max_tokens=cfg.max_tokens,
            )
            if structure:
                repo_map_section = f"\n## Codebase Structure\n{structure}\n"
        except Exception:
            logger.warning("Repo map unavailable for %s — generating without codebase context", feature_name)

        prompt = f"""You are a code generation expert. Generate a complete, production-ready code scaffold.

FEATURE NAME: {feature_name}
SPECIFICATION: {spec}{repo_map_section}

Produce a JSON response with this exact structure:
{{
  "scaffold_code": "complete Python module code as a string",
  "tests": [
    {{"filename": "test_{feature_name.lower().replace(" ", "_")}.py", "content": "complete test code"}}
  ],
  "artifacts": [
    {{"filename": "README.md", "content": "complete README"}},
    {{"filename": "config.yaml", "content": "config template"}},
    {{"filename": ".gitignore", "content": "gitignore content"}}
  ],
  "implementation_notes": ["note 1", "note 2"],
  "summary": "brief summary"
}}

Requirements:
- Generate real, functional code that implements the specification
- Include proper error handling, logging, and documentation
- Tests should cover happy path and edge cases
- Code must be syntactically valid Python"""

        result = self._infer_json(prompt, temperature=TEMPERATURE_LOW)

        if result and isinstance(result, dict) and result.get("scaffold_code"):
            return result

        # Fallback: minimal scaffold
        self._log("warning", "JSON scaffold failed, attempting plain text generation")
        safe_name = feature_name.lower().replace(" ", "_")
        class_name = feature_name.replace(" ", "").capitalize()

        code_prompt = (
            f"Write a complete Python class named {class_name} that implements: {spec}\n"
            "Include __init__, execute(), validate() methods with full docstrings and error handling."
        )
        try:
            generated_code = self._infer(code_prompt, temperature=TEMPERATURE_LOW)
        except ModelUnavailableError:
            logger.warning("Model unavailable — returning minimal scaffold for %s", feature_name)
            generated_code = ""

        return {
            "scaffold_code": generated_code
            or f'"""Auto-generated {feature_name} module."""\n\nclass {class_name}:\n    pass\n',
            "tests": [
                {
                    "filename": f"test_{safe_name}.py",
                    "content": (
                        f'"""Tests for {class_name}."""\n\n'
                        f"from {safe_name} import {class_name}\n\n\n"
                        f"class Test{class_name}:\n"
                        f'    """Test suite for {class_name}."""\n\n'
                        f"    def test_instantiation(self) -> None:\n"
                        f'        """Verify {class_name} can be instantiated."""\n'
                        f"        obj = {class_name}()\n"
                        f"        assert obj is not None\n"
                    ),
                },
            ],
            "artifacts": [
                {"filename": "README.md", "content": f"# {class_name}\n\n{spec}\n"},
                {"filename": "config.yaml", "content": f"feature:\n  name: {safe_name}\n  version: 1.0.0\n"},
                {"filename": ".gitignore", "content": "__pycache__/\n*.pyc\n.pytest_cache/\nvenv/\n"},
            ],
            "implementation_notes": ["Review and customize the generated scaffold", "Run tests with: pytest"],
            "summary": f"Scaffold generated for {feature_name}",
        }
