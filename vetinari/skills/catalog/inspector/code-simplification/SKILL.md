---
name: Code Simplification
description: Dead code detection, over-abstraction identification, complexity reduction, and YAGNI enforcement
mode: simplification
agent: inspector
version: "1.0.0"
capabilities:
  - simplification
  - complexity_analysis
tags:
  - quality
  - simplification
  - complexity
  - yagni
  - dead-code
---

# Code Simplification

## Purpose

Code Simplification identifies unnecessary complexity in the codebase and recommends concrete simplifications. It detects dead code (defined but never called), over-abstraction (abstractions serving a single use case), excessive indirection (wrappers that add no value), and violations of YAGNI (You Ain't Gonna Need It). The goal is to reduce the cognitive load of understanding and maintaining the codebase by removing everything that does not earn its complexity. Simpler code has fewer bugs, is easier to test, and is cheaper to maintain.

## When to Use

- After a code review identifies over-engineering or unnecessary complexity
- During periodic codebase health reviews
- After a major feature is complete and cleanup is needed
- When new developers report difficulty understanding a module
- When the continuous improvement skill identifies maintainability as a concern
- When a module has grown large and needs decomposition or pruning
- When refactoring to pay down technical debt

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| code            | string          | Yes      | Code to analyze for simplification opportunities                   |
| mode            | string          | No       | "simplification" (default)                                         |
| context         | dict            | No       | Usage data, call graph, module dependencies                        |
| focus_areas     | list[string]    | No       | Specific areas: "dead_code", "over_abstraction", "complexity"      |
| thinking_mode   | string          | No       | Thinking budget tier                                               |

## Process Steps

1. **Dead code detection** -- Identify code that is defined but never used:
   - Functions and methods never called from anywhere in the codebase
   - Classes never instantiated or subclassed
   - Variables assigned but never read
   - Imports that are not used (beyond what ruff catches)
   - Conditional branches that can never be reached
   - Code behind feature flags that have been permanently on/off

2. **Over-abstraction identification** -- Find abstractions that serve only one use case:
   - Abstract base classes with a single concrete implementation
   - Factory functions/classes that produce only one type
   - Strategy patterns with a single strategy
   - Configuration systems for values that never change
   - Generic interfaces that are always used with the same concrete type
   - Wrapper classes that delegate every method without adding value

3. **Indirection analysis** -- Map call chains and identify unnecessary intermediaries:
   - Functions that simply call another function with the same arguments
   - Classes that wrap another class without adding behavior
   - Import re-exports that could be direct imports
   - Decorator chains where a single decorator would suffice

4. **Complexity measurement** -- Quantify code complexity:
   - Cyclomatic complexity per function (target: < 10)
   - Nesting depth per function (target: < 4 levels)
   - Function length (target: < 50 lines)
   - Class size (target: < 300 lines)
   - Parameter count (target: < 5 parameters)
   - Module coupling (number of imports from other modules)

5. **YAGNI assessment** -- Identify features that were built "just in case":
   - Configuration options that are never changed from defaults
   - Extensibility hooks that have no extensions
   - Generic solutions for problems that only have one instance
   - Pluggable architectures where nothing is ever plugged in
   - Defensive code for scenarios that the system design prevents

6. **Simplification proposal** -- For each finding, propose a specific simplification:
   - Dead code: delete it (version control preserves history)
   - Over-abstraction: inline the single implementation, remove the abstraction
   - Unnecessary indirection: replace the intermediary with direct calls
   - High complexity: extract helper functions, flatten conditionals, use early returns
   - YAGNI: remove the unused capability, document the decision

7. **Impact assessment** -- For each proposal, assess:
   - Effort to implement the simplification
   - Risk of breaking existing functionality
   - Maintenance savings over time
   - Code size reduction (lines removed)

8. **Prioritization** -- Rank simplifications by value (maintenance savings / effort):
   - Quick wins: dead code removal (zero risk, immediate benefit)
   - Medium value: flatten complex conditionals (low risk, readability benefit)
   - High effort: remove over-abstractions (requires updating callers)

## Output Format

The skill produces a simplification analysis report:

```json
{
  "passed": true,
  "grade": "B",
  "score": 0.75,
  "issues": [
    {
      "severity": "medium",
      "category": "dead_code",
      "description": "Function _legacy_plan_format() defined on line 234 but never called from anywhere in the codebase",
      "file": "vetinari/orchestration/plan_generator.py",
      "line": 234,
      "suggestion": "Delete the function -- git history preserves it if needed later"
    },
    {
      "severity": "medium",
      "category": "over_abstraction",
      "description": "AbstractPlanValidator has only one implementation (DefaultPlanValidator). The abstraction adds complexity without benefit.",
      "file": "vetinari/validation/plan_validator.py",
      "line": 12,
      "suggestion": "Inline DefaultPlanValidator logic into the single caller, remove the ABC"
    },
    {
      "severity": "low",
      "category": "complexity",
      "description": "Function process_plan_step has cyclomatic complexity of 14 (target: <10)",
      "file": "vetinari/orchestration/two_layer.py",
      "line": 89,
      "suggestion": "Extract the error classification logic into a separate function, use early returns to flatten conditions"
    }
  ],
  "metrics": {
    "dead_code_functions": 3,
    "over_abstractions": 1,
    "high_complexity_functions": 2,
    "total_lines_removable": 145,
    "simplification_effort": "M"
  },
  "suggestions": [
    "Consider merging plan_generator.py and plan_validator.py -- they are tightly coupled and always used together"
  ]
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-INS-001**: Code review MUST check all 5 dimensions (maintainability covers simplification)
- **STD-INS-005**: Every issue MUST have a severity level and actionable description
- **STD-INS-006**: Inspector MUST NOT modify code -- only report findings and suggestions
- **STD-INS-007**: Gate decision MUST be based on objective criteria
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-INS-001**: Inspector is READ-ONLY -- cannot modify production files
- **GDL-INS-004**: Use simplification mode after code_review identifies over-engineered areas

## Examples

### Example: Simplifying an over-engineered config loader

**Input:**
```
code: |
  class ConfigLoaderFactory:
      @staticmethod
      def create(config_type: str) -> ConfigLoader:
          if config_type == "yaml":
              return YAMLConfigLoader()
          raise ValueError(f"Unknown config type: {config_type}")

  class ConfigLoader(ABC):
      @abstractmethod
      def load(self, path: Path) -> dict: ...

  class YAMLConfigLoader(ConfigLoader):
      def load(self, path: Path) -> dict:
          with open(path, encoding="utf-8") as f:
              return yaml.safe_load(f)
context: {callers: ["load_config('yaml', path) -- only call site in entire codebase"]}
```

**Output (abbreviated):**
```
issues:
  - [medium/over_abstraction] "Factory pattern with only one product type. ConfigLoaderFactory always creates YAMLConfigLoader."
    suggestion: "Replace with a simple function"

  - [medium/over_abstraction] "ABC ConfigLoader with single implementation YAMLConfigLoader"
    suggestion: "Inline the YAML loading directly"

proposed_simplification:
  Before: 3 classes, 15 lines, factory + ABC + implementation
  After:
    def load_config(path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
  Result: 1 function, 3 lines (80% reduction)
```
