"""Vetinari Builder Agent (v0.4.0).

The Builder agent handles code scaffolding, boilerplate generation, test
scaffolding, writing generated files to disk, and image generation.

Absorbs: IMAGE_GENERATOR (image_generation mode)
Modes: build, image_generation
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    VerificationResult,
)
from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.constants import (
    SD_DEFAULT_CFG,
    SD_DEFAULT_HEIGHT,
    SD_DEFAULT_STEPS,
    SD_DEFAULT_WIDTH,
    SD_WEBUI_ENABLED,
    SD_WEBUI_HOST,
    TIMEOUT_MEDIUM,
)
from vetinari.types import AgentType

logger = logging.getLogger(__name__)


class BuilderAgent(MultiModeAgent):
    """Builder agent — code scaffolding, boilerplate, and image generation.

    Consolidates the former IMAGE_GENERATOR agent into a single build agent
    with two modes: ``build`` (code scaffolding) and ``image_generation``
    (Stable Diffusion / SVG fallback).
    """

    MODES = {
        "build": "_execute_build",
        "image_generation": "_execute_image_generation",
    }
    DEFAULT_MODE = "build"
    MODE_KEYWORDS = {
        "build": [
            "scaffold",
            "boilerplate",
            "generate code",
            "code gen",
            "project structure",
            "create project",
            "implement",
            "build",
            "code",
        ],
        "image_generation": [
            "image",
            "logo",
            "icon",
            "mockup",
            "diagram",
            "illustration",
            "banner",
            "background",
            "svg",
            "picture",
            "visual",
            "asset",
        ],
    }
    LEGACY_TYPE_TO_MODE = {
        "IMAGE_GENERATOR": "image_generation",
    }

    # ── Image generation constants ────────────────────────────────────
    STYLE_PRESETS = {
        "logo": "clean vector logo, flat design, minimalist, professional, white background, "
        "high contrast, suitable for brand identity",
        "icon": "flat icon, simple shapes, clean lines, professional, vector style, 512x512, transparent background",
        "ui_mockup": "UI wireframe, clean layout, modern web design, annotated, low fidelity mockup",
        "diagram": "technical diagram, clean flowchart, professional infographic, clear labels, minimal color palette",
        "banner": "professional banner, marketing design, clean typography, modern gradient",
        "background": "abstract background, subtle texture, professional, suitable for UI, dark theme, minimal",
    }
    NEGATIVE_PROMPT = (
        "blurry, low quality, artifacts, watermark, text errors, distorted, "
        "ugly, deformed, bad anatomy, duplicate, extra elements, noisy, grainy"
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(AgentType.BUILDER, config)
        self._language = self._config.get("language", "python")
        # Image generation config (absorbed from ImageGeneratorAgent)
        self._sd_host = self._config.get("sd_host", SD_WEBUI_HOST)
        self._sd_enabled = self._config.get("sd_enabled", SD_WEBUI_ENABLED)
        self._default_width = self._config.get("width", SD_DEFAULT_WIDTH)
        self._default_height = self._config.get("height", SD_DEFAULT_HEIGHT)
        self._steps = self._config.get("steps", SD_DEFAULT_STEPS)
        self._cfg_scale = self._config.get("cfg_scale", SD_DEFAULT_CFG)
        self._output_dir = Path(self._config.get("output_dir", "./outputs/images"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Builder. You handle code scaffolding, boilerplate "
            "generation, test scaffolding, project structure creation, and visual "
            "asset generation (images, logos, icons, diagrams via Stable Diffusion "
            "or SVG fallback)."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "build": (
                "You are Vetinari's Master Builder — an expert software engineer specialising in\n"
                "code generation, project scaffolding, and production-ready implementation. You\n"
                "translate specifications into working, tested, documented code that a professional\n"
                "developer would be proud to ship. You apply clean code principles (SOLID, DRY,\n"
                "KISS, YAGNI), idiomatic language patterns, and security-by-default practices.\n"
                "You never generate placeholder comments like 'TODO: implement this' — every\n"  # noqa: VET034
                "function you generate actually does something meaningful. Your test suite covers\n"
                "the happy path, all error paths, and critical edge cases. You write code for\n"
                "the maintainer who will read it in 2 years, not just for the compiler.\n\n"
                "OUTPUT SCHEMA:\n"
                "{\n"
                '  "scaffold_code": "string — complete, syntactically valid Python module",\n'
                '  "tests": [\n'
                "    {\n"
                '      "filename": "string — test_{feature}.py",\n'
                '      "content": "string — complete pytest test file"\n'
                "    }\n"
                "  ],\n"
                '  "artifacts": [\n'
                "    {\n"
                '      "filename": "string — e.g. README.md, config.yaml, .gitignore",\n'
                '      "content": "string — complete file content"\n'
                "    }\n"
                "  ],\n"
                '  "implementation_notes": ["list of key implementation decisions and rationale"],\n'
                '  "summary": "string — 1-2 sentence description of what was generated",\n'
                '  "dependencies": ["list of pip packages required"],\n'
                '  "ci_hints": ["list of CI/CD steps required to build and test"],\n'
                '  "known_limitations": ["list of explicit limitations or TODOs"]\n'
                "}\n\n"
                "DECISION FRAMEWORK — code generation:\n"
                "1. Is this a class with state? -> Include __init__, __repr__, and property accessors\n"
                "2. Is this an external-facing API function? -> Include input validation and typed return\n"
                "3. Is this I/O (file, network, DB)? -> Wrap in try/except with specific exception types\n"
                "4. Is this a long-running operation? -> Add logging at start, completion, and error\n"
                "5. Are there secrets needed? -> Use os.environ.get() with clear error if missing\n"
                "6. Is this a utility function? -> Keep it pure (no side effects, same input same output)\n\n"
                "CODE QUALITY STANDARDS:\n"
                "- All functions must have type hints on parameters and return types\n"
                "- All public classes and functions must have docstrings (Google style)\n"
                "- No bare except clauses — always specify exception type\n"
                "- No hardcoded credentials, tokens, secrets, or magic numbers\n"
                "- No mutable default arguments (def f(x=[]) is forbidden)\n"
                "- Import only what is used — no wildcard imports\n\n"
                "SECURITY BY DEFAULT:\n"
                "- Never hardcode passwords, API keys, tokens, or any secrets\n"
                "- Use parameterized queries for all database access (never string concatenation)\n"
                "- Validate and sanitize all inputs that come from outside the system\n"
                "- Default to HTTPS, read-only permissions, least-privilege credentials\n"
                "- Use secrets.token_urlsafe() not random.random() for security tokens\n\n"
                "TESTING STANDARDS:\n"
                "- Every function must have at minimum: one happy-path test and one error-path test\n"
                "- Edge cases: empty string, None, zero, negative numbers, maximum boundary values\n"
                "- Use descriptive test names: test_function_name_when_condition_then_outcome\n"
                "- Arrange-Act-Assert pattern strictly\n"
                "- Use pytest.raises() for exception tests, never try/except in tests\n"
                "- Include at least one parametrized test for functions with multiple valid inputs\n\n"
                "FEW-SHOT EXAMPLE 1 — HTTP client class:\n"
                'Spec: "HTTP client with retry logic and timeout"\n'
                "Output: scaffold_code includes class HTTPClient with __init__(base_url, timeout, max_retries),\n"
                "get(endpoint) and post(endpoint, data) methods with exponential backoff,\n"
                "full type hints, Google-style docstrings, and specific exception handling.\n"
                "tests include test_get_success, test_get_timeout_retries, test_post_with_auth_header.\n\n"
                "FEW-SHOT EXAMPLE 2 — Database model:\n"
                'Spec: "User model with PostgreSQL using SQLAlchemy"\n'
                "Output: scaffold_code includes User dataclass with id (UUID), email (str), created_at (datetime),\n"
                "repository class with create(), get_by_email(), update(), parameterized queries only.\n"
                "artifacts include migration SQL in 001_create_users.sql.\n\n"
                "FEW-SHOT EXAMPLE 3 — CLI tool:\n"
                'Spec: "CLI tool to process CSV files"\n'
                "Output: scaffold_code uses argparse with --input, --output, --format flags,\n"
                "validates file existence before opening, streams large files in chunks.\n\n"
                "ERROR HANDLING:\n"
                "- If spec is ambiguous, make the simplest reasonable interpretation and note it\n"
                "- If spec requires a library not in stdlib, list it in dependencies\n"
                "- If generated code has known limitations, list them in known_limitations (never silently truncate)\n"
                "- If the spec asks for something insecure (hardcoded credentials), refuse and explain\n\n"
                "QUALITY CRITERIA:\n"
                "- scaffold_code must be syntactically valid Python (mentally parse before outputting)\n"
                "- tests array must have at least one file with at least 3 test functions\n"
                "- artifacts must include README.md with usage instructions\n"
                "- implementation_notes must explain at least one non-obvious design decision\n\n"
                "MICRO-RULES for output stability:\n"
                "- scaffold_code must be a string (complete file content, not a snippet)\n"
                "- test filenames must follow the pattern test_{feature_name}.py\n"
                "- dependencies must list exact package names (pip install names)\n"
                "- known_limitations must be present and be an array (use [] if none)"
            ),
            "image_generation": (
                "You are Vetinari's Visual Asset Engineer — an expert in generative image\n"
                "specification, prompt engineering for diffusion models, SVG vector graphics\n"
                "authoring, and visual design principles. You translate high-level asset\n"
                "descriptions into precise, optimised generation specifications that produce\n"
                "professional-quality results from Stable Diffusion or equivalent SVG code\n"
                "when SD is unavailable. You understand that prompt engineering for image\n"
                "generation is a distinct skill: specificity beats vagueness, technical quality\n"
                "terms improve output, and negative prompts are as important as positive ones.\n"
                "You select the right style preset, dimensions, and parameters for each asset\n"
                "type. When generating SVG fallbacks, you produce clean, valid, professional\n"
                "vector code with no external dependencies.\n\n"
                "OUTPUT SCHEMA:\n"
                "{\n"
                '  "sd_prompt": "string — optimised Stable Diffusion positive prompt",\n'
                '  "negative_prompt": "string — elements to exclude from generation",\n'
                '  "style_preset": "logo|icon|ui_mockup|diagram|banner|background",\n'
                '  "width": "integer — pixel width",\n'
                '  "height": "integer — pixel height",\n'
                '  "steps": "integer — diffusion steps (20-50)",\n'
                '  "cfg_scale": "float — classifier-free guidance scale (7-12)",\n'
                '  "description": "string — human-readable description of the asset",\n'
                '  "svg_fallback": "string — complete, valid SVG code starting with <svg",\n'
                '  "color_palette": {"primary": "#...", "secondary": "#...", "background": "#..."},\n'
                '  "usage_context": "string — where/how this asset will be used",\n'
                '  "format_recommendations": ["list of format/export notes"]\n'
                "}\n\n"
                "DECISION FRAMEWORK — style preset selection:\n"
                "1. Keywords: logo, brand, company, startup, identity -> style_preset=logo, 512x512\n"
                "2. Keywords: icon, button, app icon, action -> style_preset=icon, 256x256\n"
                "3. Keywords: mockup, wireframe, UI, layout, screen -> style_preset=ui_mockup, 1280x720\n"
                "4. Keywords: diagram, flowchart, architecture, process -> style_preset=diagram, 1024x768\n"
                "5. Keywords: banner, header, hero, marketing -> style_preset=banner, 1200x400\n"
                "6. Keywords: background, texture, wallpaper, pattern -> style_preset=background, 1920x1080\n\n"
                "SD PROMPT CONSTRUCTION RULES:\n"
                "- Structure: [subject], [style adjectives], [technical quality], [format hints]\n"
                "- Quality boosters: 'professional', 'high resolution', 'sharp', 'clean', 'vector'\n"
                "- Style anchor: always include the style_preset as a descriptive term\n"
                "- Color specification: name specific colors when they matter to brand identity\n"
                "- Composition: 'centered', 'balanced', 'minimal', 'flat design', 'white background'\n"
                "- Negative prompt: always include 'blurry, low quality, watermark, text errors, distorted'\n\n"
                "SVG FALLBACK REQUIREMENTS:\n"
                "- Must be valid, standalone SVG (no external fonts, no JavaScript, no XLinks)\n"
                "- Must include viewBox attribute matching width/height\n"
                "- Use only basic SVG shapes: rect, circle, path, text, g, defs, linearGradient\n"
                "- Text must use generic font families (sans-serif, monospace) not system fonts\n"
                "- For logos: geometric shapes with abbreviated text (first letter or acronym)\n"
                "- For icons: single-concept, recognisable vector shape\n"
                "- For diagrams: boxes + arrows with clear label text\n\n"
                "FEW-SHOT EXAMPLE 1 — Logo generation:\n"
                'Description: "Logo for a Python developer tools company called Vetinari"\n'
                'Output: sd_prompt="clean vector logo for Vetinari developer tools, snake icon stylised\n'
                "as code brackets, dark teal and white, minimalist flat design, professional brand identity,\n"
                'white background, sharp edges, scalable",\n'
                'color_palette={primary:"#1A535C",secondary:"#4ECDC4",background:"#FFFFFF"}\n\n'
                "FEW-SHOT EXAMPLE 2 — Architecture diagram SVG:\n"
                'Description: "System architecture diagram showing API gateway and 3 services"\n'
                "Output: svg_fallback=complete SVG with labeled rectangles for API Gateway, Service A/B/C,\n"
                "arrows showing data flow, colour-coded by service type\n\n"
                "FEW-SHOT EXAMPLE 3 — UI mockup:\n"
                'Description: "Login page wireframe"\n'
                "Output: style_preset=ui_mockup, width=1280, height=720,\n"
                'sd_prompt="UI wireframe login page, centered card layout, email and password fields,\n'
                'submit button, clean low-fidelity mockup, annotated, professional"\n\n'
                "ERROR HANDLING:\n"
                "- If description is very vague (< 5 words), infer the most likely asset type\n"
                "- If dimensions conflict with style_preset, use style_preset dimensions as override\n"
                "- svg_fallback must always be non-empty — generate a placeholder if description is unclear\n"  # noqa: VET034
                "- Never return null for svg_fallback — it is the critical fallback path\n\n"
                "QUALITY CRITERIA:\n"
                "- sd_prompt must be 20-100 words (specific but not overloaded)\n"
                "- negative_prompt must be at least 10 words\n"
                "- svg_fallback must be valid XML starting with <svg and ending with </svg>\n"
                "- color_palette must include all three keys: primary, secondary, background\n\n"
                "MICRO-RULES for output stability:\n"
                "- style_preset must be one of: logo, icon, ui_mockup, diagram, banner, background\n"
                "- width and height must be positive integers\n"
                "- steps must be integer 20-50\n"
                "- cfg_scale must be float 7.0-12.0\n"
                "- svg_fallback must start with <svg (not <!DOCTYPE or any other tag)"
            ),
        }
        return prompts.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        """Verify output — mode-aware.

        Returns:
            The VerificationResult result.
        """
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"message": "Output must be a dict"}], score=0.0)

        # Image generation output
        images = output.get("images")
        if images is not None:
            if not images:
                return VerificationResult(passed=False, issues=[{"message": "No images generated"}], score=0.0)
            score = 1.0
            issues = []
            for img in images:
                if img.get("type") == "svg" and not img.get("code"):
                    issues.append({"message": f"SVG asset has no code: {img.get('description', '')}"})
                    score -= 0.2
                elif img.get("type") == "png" and not Path(img.get("path", "")).exists():
                    issues.append({"message": f"Image file not found: {img.get('path', '')}"})
                    score -= 0.3
            return VerificationResult(passed=score >= 0.5, issues=issues, score=max(0.0, score))

        # Build output
        issues = []
        score = 1.0
        if not output.get("scaffold_code"):
            issues.append({"type": "missing_code", "message": "Scaffold code missing"})
            score -= 0.3
        if not output.get("tests"):
            issues.append({"type": "missing_tests", "message": "Test scaffolding missing"})
            score -= 0.2
        if not output.get("artifacts"):
            issues.append({"type": "missing_artifacts", "message": "Artifact files missing"})
            score -= 0.15
        if not output.get("implementation_notes"):
            issues.append({"type": "missing_notes", "message": "Implementation notes missing"})
            score -= 0.1
        return VerificationResult(passed=score >= 0.5, issues=issues, score=max(0, score))

    def get_capabilities(self) -> list[str]:
        return [
            "code_scaffolding",
            "test_generation",
            "boilerplate_creation",
            "project_structure",
            "configuration_templates",
            "documentation_generation",
            "image_generation",
            "logo_design",
            "icon_creation",
            "ui_mockup",
            "diagram_generation",
            "svg_generation",
            "asset_creation",
        ]

    # ==================================================================
    # Build mode
    # ==================================================================

    def _execute_build(self, task: AgentTask) -> AgentResult:
        """Execute the build scaffolding task."""
        spec = task.context.get("spec", task.description)
        feature_name = task.context.get("feature_name", "feature")
        output_dir = task.context.get("output_dir", "")

        scaffold = self._generate_scaffold(spec, feature_name)

        written_files: list[str] = []
        if output_dir or task.context.get("write_files", False):
            written_files = self._write_scaffold_to_disk(scaffold, output_dir or ".")

        syntax_errors = self._check_syntax(scaffold.get("scaffold_code", ""))

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
            },
        )

    def _generate_scaffold(self, spec: str, feature_name: str) -> dict[str, Any]:
        """Generate code scaffold using LLM-powered code generation."""
        prompt = f"""You are a code generation expert. Generate a complete, production-ready code scaffold.

FEATURE NAME: {feature_name}
SPECIFICATION: {spec}

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

        result = self._infer_json(prompt, temperature=0.2)

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
        generated_code = self._infer(code_prompt, temperature=0.2)

        return {
            "scaffold_code": generated_code
            or f'"""Auto-generated {feature_name} module."""\n\nclass {class_name}:\n    pass\n',
            "tests": [
                {
                    "filename": f"test_{safe_name}.py",
                    "content": f"import unittest\n\nclass Test{class_name}(unittest.TestCase):\n    pass\n\nif __name__ == '__main__':\n    unittest.main()\n",
                }
            ],
            "artifacts": [
                {"filename": "README.md", "content": f"# {class_name}\n\n{spec}\n"},
                {"filename": "config.yaml", "content": f"feature:\n  name: {safe_name}\n  version: 1.0.0\n"},
                {"filename": ".gitignore", "content": "__pycache__/\n*.pyc\n.pytest_cache/\nvenv/\n"},
            ],
            "implementation_notes": ["Review and customize the generated scaffold", "Run tests with: pytest"],
            "summary": f"Scaffold generated for {feature_name}",
        }

    # ------------------------------------------------------------------
    # Tool-aware file writing
    # ------------------------------------------------------------------

    def _tool_write_file(self, path: str, content: str) -> bool:
        """Write a file via the file_operations tool (auditable, permission-gated).

        Falls back to direct I/O if the tool is unavailable.

        Returns:
            True if the file was written successfully.
        """
        if self._has_tool("file_operations"):
            result = self._use_tool("file_operations", operation="write", path=path, content=content)
            if result and result.get("success"):
                return True
            # Tool failed — fall through to direct write

        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return True
        except Exception as exc:
            self._log("error", f"File write failed for {path}: {exc}")
            return False

    def _write_scaffold_to_disk(self, scaffold: dict[str, Any], output_dir: str) -> list[str]:
        """Write all scaffold files to ``output_dir``. Returns list of written paths."""
        written: list[str] = []
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)

        code = scaffold.get("scaffold_code", "")
        if code:
            feature_name = scaffold.get("summary", "feature").split()[-1]
            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in feature_name.lower())
            code_path = str(base / f"{safe_name}.py")
            if self._tool_write_file(code_path, code):
                written.append(code_path)
                logger.info("[BuilderAgent] Wrote %s", code_path)

        for test in scaffold.get("tests", []):
            fname = test.get("filename", "test_generated.py")
            content = test.get("content", "")
            if content:
                test_path = str(base / fname)
                if self._tool_write_file(test_path, content):
                    written.append(test_path)

        for artifact in scaffold.get("artifacts", []):
            fname = artifact.get("filename", "artifact.txt")
            content = artifact.get("content", "")
            if content:
                artifact_path = str(base / fname)
                if self._tool_write_file(artifact_path, content):
                    written.append(artifact_path)

        return written

    @staticmethod
    def _check_syntax(code: str) -> list[str]:
        """Return list of syntax error messages for Python code, or empty list."""
        if not code or not code.strip():
            return []
        try:
            ast.parse(code)
            return []
        except SyntaxError as e:
            return [f"SyntaxError at line {e.lineno}: {e.msg}"]
        except Exception as e:
            return [str(e)]

    def write_and_execute(
        self,
        code: str,
        timeout: int = 30,
        working_dir: str | None = None,
    ) -> dict[str, Any]:
        """Write code to a temp file and execute it safely.

        Args:
            code: The code.
            timeout: The timeout.
            working_dir: The working dir.

        Returns:
            The result string.
        """
        syntax_errors = self._check_syntax(code)
        if syntax_errors:
            return {
                "success": False,
                "stdout": "",
                "stderr": "\n".join(syntax_errors),
                "returncode": -1,
                "syntax_errors": syntax_errors,
            }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir or ".",
            )
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout[:5000],
                "stderr": proc.stderr[:2000],
                "returncode": proc.returncode,
                "syntax_errors": [],
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "returncode": -1,
                "syntax_errors": [],
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "syntax_errors": [],
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.debug("Failed to remove temporary file %s", tmp_path, exc_info=True)

    # ==================================================================
    # Image Generation mode (absorbed from ImageGeneratorAgent)
    # ==================================================================

    def _execute_image_generation(self, task: AgentTask) -> AgentResult:
        """Generate an image asset from the task description."""
        description = task.prompt or task.description
        context = task.context or {}

        # Step 1: Use LLM to build an optimized SD prompt
        image_spec = self._build_image_spec(description, context)

        # Step 2: Try to generate via Stable Diffusion
        generated_images: list[dict[str, Any]] = []
        sd_error = None

        if self._sd_enabled:
            try:
                generated_images = self._generate_via_sd(image_spec)
            except Exception as e:
                sd_error = str(e)
                logger.warning("SD generation failed: %s", e)

        # Step 3: If SD failed/unavailable, generate SVG fallback
        if not generated_images:
            svg_code = image_spec.get("svg_fallback") or self._generate_svg_fallback(description, image_spec)
            if svg_code:
                svg_path = self._save_svg(svg_code, description)
                generated_images = [
                    {
                        "type": "svg",
                        "path": str(svg_path),
                        "description": image_spec.get("description", description),
                        "code": svg_code,
                    }
                ]

        result = {
            "images": generated_images,
            "spec": image_spec,
            "sd_available": self._sd_enabled and not sd_error,
            "sd_error": sd_error,
            "count": len(generated_images),
        }

        return AgentResult(
            success=bool(generated_images),
            output=result,
            metadata={
                "mode": "image_generation",
                "image_count": len(generated_images),
                "backend": "stable_diffusion" if (self._sd_enabled and not sd_error) else "svg_fallback",
                "description": description,
            },
        )

    def _build_image_spec(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Use LLM to build an optimized image specification."""
        style = self._detect_style(description)

        prompt = f"""Create an image generation specification for this asset:

Description: {description}
Context: {json.dumps(context, default=str)[:300] if context else "none"}

Infer the best style preset from: {", ".join(self.STYLE_PRESETS.keys())}
For the SD prompt, use descriptive terms that produce professional results.
Include an SVG code fallback for simple icons/logos that can be represented as vectors.

Respond with valid JSON only."""

        spec = self._infer_json(prompt) or {}

        if not spec.get("style_preset"):
            spec["style_preset"] = style
        if not spec.get("sd_prompt"):
            style_hint = self.STYLE_PRESETS.get(style, "")
            spec["sd_prompt"] = f"{description}, {style_hint}"
        if not spec.get("negative_prompt"):
            spec["negative_prompt"] = self.NEGATIVE_PROMPT
        if not spec.get("width"):
            spec["width"] = self._get_default_size(style)[0]
        if not spec.get("height"):
            spec["height"] = self._get_default_size(style)[1]
        if not spec.get("steps"):
            spec["steps"] = self._steps
        if not spec.get("description"):
            spec["description"] = description

        return spec

    def _detect_style(self, description: str) -> str:
        """Detect appropriate style preset from description keywords."""
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in ["logo", "brand", "identity", "company", "startup"]):
            return "logo"
        if any(kw in desc_lower for kw in ["icon", "button icon", "app icon", "fa icon"]):
            return "icon"
        if any(kw in desc_lower for kw in ["mockup", "wireframe", "ui", "layout", "screen"]):
            return "ui_mockup"
        if any(kw in desc_lower for kw in ["diagram", "flowchart", "architecture", "flow"]):
            return "diagram"
        if any(kw in desc_lower for kw in ["banner", "header image", "hero image"]):
            return "banner"
        if any(kw in desc_lower for kw in ["background", "texture", "wallpaper", "pattern"]):
            return "background"
        return "logo"

    def _get_default_size(self, style: str) -> tuple:
        """Return default (width, height) for a style."""
        sizes = {
            "logo": (512, 512),
            "icon": (256, 256),
            "ui_mockup": (1280, 720),
            "diagram": (1024, 768),
            "banner": (1200, 400),
            "background": (1920, 1080),
        }
        return sizes.get(style, (512, 512))

    def _generate_via_sd(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Call the Stable Diffusion WebUI API to generate images."""
        try:
            import requests as req
        except ImportError:
            raise RuntimeError("requests library required for SD generation") from None

        url = f"{self._sd_host}/sdapi/v1/txt2img"
        payload = {
            "prompt": spec.get("sd_prompt", ""),
            "negative_prompt": spec.get("negative_prompt", self.NEGATIVE_PROMPT),
            "width": spec.get("width", self._default_width),
            "height": spec.get("height", self._default_height),
            "steps": spec.get("steps", self._steps),
            "cfg_scale": spec.get("cfg_scale", self._cfg_scale),
            "sampler_index": spec.get("sampler", "DPM++ 2M Karras"),
            "seed": spec.get("seed", -1),
            "batch_size": 1,
        }

        response = req.post(url, json=payload, timeout=TIMEOUT_MEDIUM * 4)
        response.raise_for_status()
        data = response.json()

        images = []
        for _i, img_b64 in enumerate(data.get("images", [])):
            filename = f"img_{uuid.uuid4().hex[:8]}.png"
            out_path = self._output_dir / filename
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            images.append(
                {
                    "type": "png",
                    "path": str(out_path),
                    "filename": filename,
                    "description": spec.get("description", ""),
                    "prompt": spec.get("sd_prompt", ""),
                }
            )
            logger.info("Generated image: %s", out_path)

        return images

    def _generate_svg_fallback(self, description: str, spec: dict[str, Any]) -> str:
        """Ask the LLM to generate SVG code as a fallback."""
        style = spec.get("style_preset", "logo")
        size = self._get_default_size(style)

        prompt = f"""Generate a clean, professional SVG for the following:

Description: {description}
Style: {style}
Size: {size[0]}x{size[1]}

Requirements:
- Complete, valid SVG code only
- No external dependencies or scripts
- Clean vector shapes, appropriate for {style}
- Professional quality
- Include viewBox="0 0 {size[0]} {size[1]}"

Output ONLY the SVG code, starting with <svg and ending with </svg>."""

        result = self._infer(prompt)
        if not result:
            return self._minimal_svg_placeholder(description, size)

        svg_match = re.search(r"<svg[\s\S]*?</svg>", result, re.IGNORECASE)
        if svg_match:
            return svg_match.group(0)

        if result.strip().startswith("<svg"):
            return result.strip()

        return self._minimal_svg_placeholder(description, size)

    def _minimal_svg_placeholder(self, description: str, size: tuple) -> str:
        """Generate a descriptive placeholder SVG with keyword-based theming."""  # noqa: VET034
        w, h = size
        colors = {
            "landscape": "#4CAF50",
            "portrait": "#2196F3",
            "abstract": "#9C27B0",
            "diagram": "#FF9800",
            "chart": "#00BCD4",
            "icon": "#E91E63",
            "logo": "#3F51B5",
            "photo": "#795548",
        }
        bg_color = "#607D8B"
        for keyword, color in colors.items():
            if keyword in description.lower():
                bg_color = color
                break

        safe_desc = description.replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")
        lines = [safe_desc[i : i + 40] for i in range(0, min(len(safe_desc), 120), 40)]
        text_elements = "\n".join(
            f'  <text x="{w // 2}" y="{h // 2 + i * 20 - len(lines) * 10}" '
            f'text-anchor="middle" fill="white" font-family="sans-serif" '
            f'font-size="13">{line}</text>'
            for i, line in enumerate(lines)
        )

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" data-placeholder="true">\n'  # noqa: VET034
            f'  <rect width="{w}" height="{h}" fill="{bg_color}" rx="8"/>\n'
            f'  <text x="{w // 2}" y="28" text-anchor="middle" fill="white" font-family="sans-serif"\n'
            f'        font-size="15" font-weight="bold">Image Placeholder ({w}x{h})</text>\n'  # noqa: VET034
            f"{text_elements}\n"
            f"</svg>"
        )

    def _save_svg(self, svg_code: str, description: str) -> Path:
        """Save SVG code to a file and return the path."""
        filename = f"img_{uuid.uuid4().hex[:8]}.svg"
        out_path = self._output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
        logger.info("Generated SVG: %s", out_path)
        return out_path


# Singleton instance
_builder_agent: BuilderAgent | None = None


def get_builder_agent(config: dict[str, Any] | None = None) -> BuilderAgent:
    """Get the singleton Builder agent instance.

    Returns:
        The BuilderAgent result.
    """
    global _builder_agent
    if _builder_agent is None:
        _builder_agent = BuilderAgent(config)
    return _builder_agent
