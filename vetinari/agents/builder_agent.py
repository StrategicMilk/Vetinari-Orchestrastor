"""Vetinari Builder Agent — consolidated from UI Planner + Data Engineer + DevOps.

The Builder agent is responsible for code scaffolding, boilerplate generation,
test scaffolding, UI/UX design, data pipeline engineering, and DevOps/CI-CD.
"""

import ast
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class BuilderAgent(BaseAgent):
    """Builder agent — code scaffolding, UI/UX, data engineering, and DevOps.

    Absorbs:
        - UIPlannerAgent: UI/UX design, front-end patterns, responsive layouts, accessibility
        - DataEngineerAgent: data pipelines, schemas, migrations, ETL, SQL
        - DevOpsAgent: CI/CD pipelines, containerisation, deployment, monitoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.BUILDER, config)
        self._language = self._config.get("language", "python")

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Builder. You combine code scaffolding with UI/UX design,
data engineering, and DevOps capabilities for end-to-end implementation.

Your responsibilities:
1. Generate clean, well-structured scaffold code with tests
2. Design accessible, responsive UI components and interaction flows
3. Build data pipelines, schemas, migrations, and ETL processes
4. Configure CI/CD pipelines, Docker containers, and deployment automation
5. Provide CI/CD configuration hints and infrastructure-as-code
6. Include error handling patterns and monitoring setup

Output format must include scaffold_code, tests, artifacts (readme, config), and implementation_notes."""

    def get_capabilities(self) -> List[str]:
        return [
            "code_scaffolding",
            "test_generation",
            "boilerplate_creation",
            "project_structure",
            "configuration_templates",
            "documentation_generation",
            # From UIPlannerAgent
            "ui_design",
            "responsive_layout",
            "accessibility",
            "css_generation",
            "component_architecture",
            # From DataEngineerAgent
            "data_pipelines",
            "schema_design",
            "etl",
            "sql_generation",
            "data_validation",
            # From DevOpsAgent
            "ci_cd_pipelines",
            "containerisation",
            "deployment_automation",
            "monitoring_setup",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute task, delegating to UI planner, data engineer, or devops based on keywords."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("ui", "ux", "frontend", "front-end", "css", "responsive", "layout", "component", "interface design", "mockup", "wireframe", "accessibility")):
                result = self._delegate_to_ui_planner(task)
            elif any(kw in desc for kw in ("data pipeline", "etl", "schema", "migration", "sql", "database design", "data model", "warehouse", "data layer")):
                result = self._delegate_to_data_engineer(task)
            elif any(kw in desc for kw in ("devops", "ci/cd", "cicd", "docker", "kubernetes", "deploy", "container", "terraform", "ansible", "monitoring", "infrastructure")):
                result = self._delegate_to_devops(task)
            else:
                result = self._execute_build(task)

            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"BuilderAgent execution failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)],
            )

    def _delegate_to_ui_planner(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.ui_planner_agent import UIPlannerAgent
        agent = UIPlannerAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_data_engineer(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.data_engineer_agent import DataEngineerAgent
        agent = DataEngineerAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_devops(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.devops_agent import DevOpsAgent
        agent = DevOpsAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _execute_build(self, task: AgentTask) -> AgentResult:
        """Execute code scaffolding (original BuilderAgent logic)."""
        spec = task.context.get("spec", task.description)
        feature_name = task.context.get("feature_name", "feature")
        output_dir = task.context.get("output_dir", "")

        # Generate scaffold using LLM
        scaffold = self._generate_scaffold(spec, feature_name)

        # Write files to disk if output_dir is specified or auto-detect project root
        written_files: List[str] = []
        if output_dir or task.context.get("write_files", False):
            written_files = self._write_scaffold_to_disk(scaffold, output_dir or ".")

        # Run syntax check on generated code
        syntax_errors = self._check_syntax(scaffold.get("scaffold_code", ""))

        return AgentResult(
            success=True,
            output=scaffold,
            metadata={
                "feature_name": feature_name,
                "files_generated": len(scaffold.get("artifacts", [])),
                "test_count": len(scaffold.get("tests", [])),
                "written_files": written_files,
                "syntax_errors": syntax_errors,
            },
        )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the scaffold output meets quality standards.
        
        Args:
            output: The scaffold to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for scaffold code
        if not output.get("scaffold_code"):
            issues.append({"type": "missing_code", "message": "Scaffold code missing"})
            score -= 0.3
        
        # Check for tests
        if not output.get("tests"):
            issues.append({"type": "missing_tests", "message": "Test scaffolding missing"})
            score -= 0.2
        
        # Check for artifacts (readme, config)
        if not output.get("artifacts"):
            issues.append({"type": "missing_artifacts", "message": "Artifact files missing"})
            score -= 0.15
        
        # Check for implementation notes
        if not output.get("implementation_notes"):
            issues.append({"type": "missing_notes", "message": "Implementation notes missing"})
            score -= 0.1
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _generate_scaffold(self, spec: str, feature_name: str) -> Dict[str, Any]:
        """Generate code scaffold using LLM-powered code generation.

        Calls the LLM to produce complete, functional scaffold code and
        tests tailored to the specification. Falls back to minimal stubs.
        """
        prompt = f"""You are a code generation expert. Generate a complete, production-ready code scaffold.

FEATURE NAME: {feature_name}
SPECIFICATION: {spec}

Produce a JSON response with this exact structure:
{{
  "scaffold_code": "complete Python module code as a string",
  "tests": [
    {{"filename": "test_{feature_name.lower().replace(' ','_')}.py", "content": "complete test code"}}
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

        # Fallback: minimal scaffold via direct code generation
        self._log("warning", "JSON scaffold failed, attempting plain text generation")
        safe_name = feature_name.lower().replace(" ", "_")
        class_name = feature_name.replace(" ", "").capitalize()

        # Try to get at least good code
        code_prompt = f"Write a complete Python class named {class_name} that implements: {spec}\nInclude __init__, execute(), validate() methods with full docstrings and error handling."
        generated_code = self._infer(code_prompt, temperature=0.2)

        return {
            "scaffold_code": generated_code or f'"""Auto-generated {feature_name} module."""\n\nclass {class_name}:\n    pass\n',
            "tests": [{"filename": f"test_{safe_name}.py", "content": f"import unittest\n\nclass Test{class_name}(unittest.TestCase):\n    pass\n\nif __name__ == '__main__':\n    unittest.main()\n"}],
            "artifacts": [
                {"filename": "README.md", "content": f"# {class_name}\n\n{spec}\n"},
                {"filename": "config.yaml", "content": f"feature:\n  name: {safe_name}\n  version: 1.0.0\n"},
                {"filename": ".gitignore", "content": "__pycache__/\n*.pyc\n.pytest_cache/\nvenv/\n"}
            ],
            "implementation_notes": ["Review and customize the generated scaffold", "Run tests with: pytest"],
            "summary": f"Scaffold generated for {feature_name}"
        }

    # ------------------------------------------------------------------
    # File I/O helpers
    # ------------------------------------------------------------------

    def _write_scaffold_to_disk(
        self, scaffold: Dict[str, Any], output_dir: str
    ) -> List[str]:
        """Write all scaffold files to ``output_dir``. Returns list of written paths."""
        written: List[str] = []
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)

        # Write main scaffold code
        code = scaffold.get("scaffold_code", "")
        if code:
            feature_name = scaffold.get("summary", "feature").split()[-1]
            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in feature_name.lower())
            code_path = base / f"{safe_name}.py"
            code_path.write_text(code, encoding="utf-8")
            written.append(str(code_path))
            logger.info(f"[BuilderAgent] Wrote {code_path}")

        # Write test files
        for test in scaffold.get("tests", []):
            fname = test.get("filename", "test_generated.py")
            content = test.get("content", "")
            if content:
                test_path = base / fname
                test_path.write_text(content, encoding="utf-8")
                written.append(str(test_path))
                logger.info(f"[BuilderAgent] Wrote {test_path}")

        # Write artifact files (README, config, etc.)
        for artifact in scaffold.get("artifacts", []):
            fname = artifact.get("filename", "artifact.txt")
            content = artifact.get("content", "")
            if content:
                artifact_path = base / fname
                artifact_path.write_text(content, encoding="utf-8")
                written.append(str(artifact_path))
                logger.info(f"[BuilderAgent] Wrote {artifact_path}")

        return written

    @staticmethod
    def _check_syntax(code: str) -> List[str]:
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
        working_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write code to a temp file and execute it safely.

        Returns dict with: ``stdout``, ``stderr``, ``returncode``, ``success``.
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = subprocess.run(
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
                pass


# Singleton instance
_builder_agent: Optional[BuilderAgent] = None


def get_builder_agent(config: Optional[Dict[str, Any]] = None) -> BuilderAgent:
    """Get the singleton Builder agent instance."""
    global _builder_agent
    if _builder_agent is None:
        _builder_agent = BuilderAgent(config)
    return _builder_agent
