"""
Vetinari Builder Agent

The Builder agent is responsible for code scaffolding, boilerplate generation,
and test scaffolding.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class BuilderAgent(BaseAgent):
    """Builder agent - code scaffolding, boilerplate, and test scaffolding."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.BUILDER, config)
        self._language = self._config.get("language", "python")
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Builder. Generate scaffolding for features from a provided spec. 
Produce boilerplate code with tests and CI hints, plus a minimal README and usage instructions.

You must:
1. Generate clean, well-structured scaffold code
2. Include comprehensive unit tests
3. Provide CI/CD configuration hints
4. Create a README with usage instructions
5. Add configuration file templates
6. Include error handling patterns

Output format must include scaffold_code, tests, artifacts (readme, config), and implementation_notes."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "code_scaffolding",
            "test_generation",
            "boilerplate_creation",
            "project_structure",
            "configuration_templates",
            "documentation_generation"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the build scaffolding task.
        
        Args:
            task: The task containing the specification
            
        Returns:
            AgentResult containing the scaffolded code
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            spec = task.context.get("spec", task.description)
            feature_name = task.context.get("feature_name", "feature")
            
            # Generate scaffold (simulated - in production would use actual code generation)
            scaffold = self._generate_scaffold(spec, feature_name)
            
            return AgentResult(
                success=True,
                output=scaffold,
                metadata={
                    "feature_name": feature_name,
                    "files_generated": len(scaffold.get("artifacts", [])),
                    "test_count": len(scaffold.get("tests", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"Scaffolding generation failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
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
        """Generate code scaffold from specification.
        
        Args:
            spec: The specification to generate from
            feature_name: Name of the feature
            
        Returns:
            Dictionary containing scaffold code and tests
        """
        # This is a simplified implementation
        # In production, this would use actual code generation templates
        
        # Generate main module code
        scaffold_code = f'''"""
{feature_name} module

Auto-generated scaffold for {feature_name}.
"""

class {feature_name.capitalize()}:
    """Main class for {feature_name} feature."""
    
    def __init__(self, config=None):
        """Initialize the {feature_name} feature."""
        self.config = config or {{}}
    
    def execute(self, input_data):
        """Execute the {feature_name} feature."""
        # TODO: Implement feature logic
        return {{"status": "success", "data": input_data}}
    
    def validate(self, input_data):
        """Validate input for {feature_name}."""
        if not input_data:
            raise ValueError("Input data is required")
        return True
'''
        
        # Generate test scaffold
        test_code = f'''"""
Tests for {feature_name} module
"""

import unittest
from {feature_name.lower()} import {feature_name.capitalize()}


class Test{feature_name.capitalize()}(unittest.TestCase):
    """Test cases for {feature_name}."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature = {feature_name.capitalize()}()
    
    def test_initialization(self):
        """Test feature initialization."""
        self.assertIsNotNone(self.feature)
    
    def test_execute_success(self):
        """Test successful execution."""
        result = self.feature.execute({{"test": "data"}})
        self.assertEqual(result["status"], "success")
    
    def test_validate_input(self):
        """Test input validation."""
        with self.assertRaises(ValueError):
            self.feature.validate(None)


if __name__ == "__main__":
    unittest.main()
'''
        
        # Generate artifacts
        readme = f'''# {feature_name.capitalize()} Feature

## Overview
{spec}

## Installation
```bash
pip install -e .
```

## Usage
```python
from {feature_name.lower()} import {feature_name.capitalize()}

feature = {feature_name.capitalize()}()
result = feature.execute(data)
print(result)
```

## Configuration
See `config.yaml` for configuration options.

## Testing
```bash
pytest tests/test_{feature_name.lower()}.py
```
'''
        
        config = f'''# Configuration for {feature_name}

feature:
  name: {feature_name}
  version: 1.0.0
  debug: false

# Add more configuration as needed
'''
        
        return {
            "scaffold_code": scaffold_code,
            "tests": [
                {"filename": f"test_{feature_name.lower()}.py", "content": test_code}
            ],
            "artifacts": [
                {"filename": "README.md", "content": readme},
                {"filename": "config.yaml", "content": config},
                {"filename": ".gitignore", "content": "__pycache__/\n*.pyc\n.pytest_cache/\nvenv/\n"}
            ],
            "implementation_notes": [
                "Generated scaffold provides basic structure",
                "Implement feature logic in the execute() method",
                "Add validation rules in the validate() method",
                "Update tests as you add functionality"
            ],
            "summary": f"Generated scaffold for {feature_name} feature"
        }


# Singleton instance
_builder_agent: Optional[BuilderAgent] = None


def get_builder_agent(config: Optional[Dict[str, Any]] = None) -> BuilderAgent:
    """Get the singleton Builder agent instance."""
    global _builder_agent
    if _builder_agent is None:
        _builder_agent = BuilderAgent(config)
    return _builder_agent
