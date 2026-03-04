"""
Vetinari Test Automation Agent

LLM-powered test generation agent that creates real, meaningful unit and integration
tests by analysing actual code artifacts — not assert True stubs.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class TestAutomationAgent(BaseAgent):
    """Test Automation agent — LLM-powered, artifact-aware test generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.TEST_AUTOMATION, config)
        self._test_framework = self._config.get("test_framework", "pytest")
        self._language = self._config.get("language", "python")

    def get_system_prompt(self) -> str:
        return f"""You are Vetinari's Test Automation Engineer — an expert in test-driven development
and comprehensive test strategy for {self._language} projects using {self._test_framework}.

You MUST generate REAL tests based on the actual code provided — not assert True stubs.
Every test must:
1. Test a specific, named function or behaviour from the code
2. Use realistic test data (not just {{"data": "test"}})
3. Assert meaningful conditions (not just True)
4. Cover both happy path and error cases

Required output (JSON):
{{
  "test_files": [
    {{
      "name": "test_feature_name.py",
      "feature": "...",
      "module_under_test": "...",
      "test_count": 0,
      "test_categories": ["unit", "integration", "edge_case"]
    }}
  ],
  "test_scripts": [
    {{
      "name": "test_feature_name.py",
      "content": "\"\"\"Tests for ...\"\"\"\nimport pytest\n...",
      "imports_needed": []
    }}
  ],
  "test_data": {{
    "fixtures": [],
    "factory_functions": [],
    "mock_definitions": []
  }},
  "coverage_report": {{
    "target": 0.8,
    "estimated_coverage": 0.0,
    "covered_functions": [],
    "uncovered_functions": [],
    "status": "estimated"
  }},
  "test_results": {{
    "total_tests": 0,
    "note": "Tests must be run with pytest to get actual results",
    "command": "pytest tests/ -v --cov"
  }},
  "ci_config": {{
    "command": "pytest tests/ -v --cov --cov-report=xml",
    "environment": [],
    "parallel": true
  }},
  "summary": "..."
}}

Test writing rules:
- Use descriptive test names: test_function_name_when_condition_then_result
- Use pytest fixtures for setup/teardown
- Mock external dependencies (HTTP, DB, filesystem)
- Test boundary conditions (empty inputs, None, max values)
- Include at least one negative test per function
"""

    def get_capabilities(self) -> List[str]:
        return [
            "unit_test_generation",
            "integration_test_design",
            "test_data_creation",
            "coverage_analysis",
            "test_execution",
            "ci_integration",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute test generation using LLM analysis of actual code."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)

        try:
            features = task.context.get("features", [])
            coverage_target = task.context.get("coverage_target", 0.8)
            code_content = task.context.get("code", "")
            artifacts = task.context.get("artifacts", {})

            # Extract function signatures from code for targeted test generation
            functions = self._extract_functions(code_content)

            # Build prompt with actual code context
            code_excerpt = code_content[:2500] if len(code_content) > 2500 else code_content
            feature_list = ", ".join(features) if features else task.description
            function_list = "\n".join(functions[:20]) if functions else "No functions extracted"

            prompt = (
                f"Generate comprehensive {self._test_framework} tests for:\n"
                f"Features: {feature_list}\n\n"
                f"Functions to test:\n{function_list}\n\n"
                f"Code to test:\n```python\n{code_excerpt}\n```\n\n"
                f"Coverage target: {coverage_target*100:.0f}%\n"
                f"Test framework: {self._test_framework}\n\n"
                "Generate REAL tests with meaningful assertions — not assert True stubs. "
                "Return JSON matching the required output format."
            )

            tests = self._infer_json(
                prompt=prompt,
                fallback=self._fallback_tests(features, functions, coverage_target),
            )

            # Ensure required keys
            tests.setdefault("test_files", [])
            tests.setdefault("test_scripts", [])
            tests.setdefault("test_data", {"fixtures": [], "factory_functions": []})
            tests.setdefault(
                "coverage_report",
                {
                    "target": coverage_target,
                    "estimated_coverage": 0.0,
                    "status": "estimated",
                },
            )
            tests.setdefault(
                "test_results",
                {
                    "total_tests": sum(f.get("test_count", 0) for f in tests.get("test_files", [])),
                    "note": "Run pytest to get actual results",
                    "command": "pytest tests/ -v --cov",
                },
            )
            tests.setdefault("summary", f"Generated tests for {feature_list}")

            result = AgentResult(
                success=True,
                output=tests,
                metadata={
                    "features_tested": len(features),
                    "test_files_count": len(tests.get("test_files", [])),
                    "coverage_target": coverage_target,
                    "functions_found": len(functions),
                },
            )
            task = self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Test generation failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0

        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            return VerificationResult(passed=False, issues=issues, score=0.0)

        if not output.get("test_files"):
            issues.append({"type": "missing_tests", "message": "Test files missing"})
            score -= 0.3
        if not output.get("test_scripts"):
            issues.append({"type": "missing_scripts", "message": "Test scripts missing"})
            score -= 0.2

        # Check test scripts actually contain real assertions
        for script in output.get("test_scripts", []):
            content = script.get("content", "")
            if content.count("assert True") > content.count("assert ") / 2:
                issues.append({"type": "stub_tests", "message": f"{script.get('name')} contains mostly assert True stubs"})
                score -= 0.1

        if not output.get("test_data"):
            issues.append({"type": "missing_data", "message": "Test data missing"})
            score -= 0.15
        if not output.get("coverage_report"):
            issues.append({"type": "missing_coverage", "message": "Coverage report missing"})
            score -= 0.15

        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0.0, score))

    def _extract_functions(self, code: str) -> List[str]:
        """Extract function signatures from Python code."""
        if not code:
            return []
        patterns = [
            r"def\s+(\w+)\s*\([^)]*\)",
            r"async def\s+(\w+)\s*\([^)]*\)",
            r"class\s+(\w+)\s*[:(]",
        ]
        found = []
        for pattern in patterns:
            matches = re.findall(pattern, code)
            found.extend(matches)
        return list(dict.fromkeys(found))  # Deduplicate preserving order

    def _fallback_tests(
        self, features: List[str], functions: List[str], coverage_target: float
    ) -> Dict[str, Any]:
        """Generate meaningful test stubs when LLM inference is unavailable."""
        test_files = []
        test_scripts = []

        for feature in (features or ["core"]):
            safe_name = re.sub(r"[^a-z0-9_]", "_", feature.lower())
            test_name = f"test_{safe_name}.py"

            # Generate tests for discovered functions
            test_cases = []
            for func_name in functions[:5]:
                test_cases.append(
                    f"""
def test_{func_name}_returns_expected_type():
    \"\"\"Test that {func_name} returns the expected type.\"\"\"
    # TODO: Instantiate class/module and call {func_name} with valid args
    # result = module.{func_name}(...)
    # assert result is not None
    pytest.skip("Implement with actual module import")


def test_{func_name}_handles_none_input():
    \"\"\"Test that {func_name} handles None/empty input gracefully.\"\"\"
    # TODO: Test edge case behaviour
    pytest.skip("Implement with actual module import")
"""
                )

            content = f'''"""
Tests for {feature}
Auto-generated by Vetinari TestAutomationAgent — replace pytest.skip() with real assertions.
"""

import pytest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_{safe_name}_data():
    \"\"\"Provide sample data for {feature} tests.\"\"\"
    return {{
        "id": 1,
        "name": "{feature} test",
        "status": "active",
    }}


# ============================================================
# Unit Tests
# ============================================================

{"".join(test_cases) if test_cases else f"""
def test_{safe_name}_basic(sample_{safe_name}_data):
    \"\"\"Basic test for {feature}.\"\"\"
    data = sample_{safe_name}_data
    assert data is not None
    assert data["status"] == "active"


def test_{safe_name}_empty_input():
    \"\"\"Test with empty input.\"\"\"
    # Verify graceful handling of empty/None inputs
    pytest.skip("Implement: test empty input handling")


def test_{safe_name}_invalid_input():
    \"\"\"Test with invalid input.\"\"\"
    pytest.skip("Implement: test invalid input rejection")
"""}
'''
            test_files.append(
                {
                    "name": test_name,
                    "feature": feature,
                    "module_under_test": safe_name,
                    "test_count": len(test_cases) + 3,
                    "test_categories": ["unit"],
                }
            )
            test_scripts.append({"name": test_name, "content": content, "imports_needed": ["pytest"]})

        return {
            "test_files": test_files,
            "test_scripts": test_scripts,
            "test_data": {
                "fixtures": [{"name": f"sample_{re.sub(r'[^a-z0-9_]', '_', f.lower())}_data", "feature": f} for f in features],
                "factory_functions": [],
                "mock_definitions": [
                    "Mock LLM responses: use unittest.mock.patch('vetinari.agents.base_agent.BaseAgent._infer')",
                    "Mock HTTP calls: use responses library or unittest.mock.patch('requests.Session.post')",
                ],
            },
            "coverage_report": {
                "target": coverage_target,
                "estimated_coverage": 0.0,
                "covered_functions": functions[:10],
                "uncovered_functions": [],
                "status": "estimated — run pytest --cov for actual coverage",
            },
            "test_results": {
                "total_tests": sum(f["test_count"] for f in test_files),
                "note": "Tests contain pytest.skip() placeholders — implement before running",
                "command": f"pytest tests/ -v --cov={self._language} --cov-report=html",
            },
            "ci_config": {
                "command": "pytest tests/ -v --cov --cov-report=xml -x",
                "environment": ["pytest", "pytest-cov", "pytest-mock"],
                "parallel": True,
            },
            "summary": f"Generated test scaffolds for {len(test_files)} features. Implement pytest.skip() stubs with real assertions.",
        }


_test_automation_agent: Optional[TestAutomationAgent] = None


def get_test_automation_agent(config: Optional[Dict[str, Any]] = None) -> TestAutomationAgent:
    global _test_automation_agent
    if _test_automation_agent is None:
        _test_automation_agent = TestAutomationAgent(config)
    return _test_automation_agent
