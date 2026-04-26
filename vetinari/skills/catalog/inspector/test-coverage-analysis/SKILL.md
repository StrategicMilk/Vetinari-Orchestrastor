---
name: Test Coverage Analysis
description: Identify untested code paths, generate gap-filling tests, and measure coverage improvement
mode: test_generation
agent: inspector
version: "1.0.0"
capabilities:
  - test_generation
  - complexity_analysis
tags:
  - quality
  - testing
  - coverage
  - gap-analysis
---

# Test Coverage Analysis

## Purpose

Test Coverage Analysis identifies code paths that lack test coverage, prioritizes gaps by risk (untested error handlers are more dangerous than untested logging), and generates test specifications to fill those gaps. It goes beyond line coverage to analyze branch coverage, path coverage, and semantic coverage (are the tests testing the right things, or just executing lines?). The output is either a gap report for planning purposes or concrete test specifications that the Worker's test-writing skill can implement.

## When to Use

- After a feature implementation, to verify test completeness
- During periodic quality reviews to assess overall test health
- When the code review identifies missing test coverage
- Before a release, to ensure critical paths are tested
- When refactoring, to verify existing tests cover the changed behavior
- When investigating why a bug was not caught by existing tests

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| code            | string          | Yes      | Source code to analyze for coverage gaps                           |
| mode            | string          | No       | "test_generation" (default)                                        |
| context         | dict            | No       | Existing test files, coverage report, risk classification          |
| focus_areas     | list[string]    | No       | Specific areas to focus: "error_handling", "edge_cases", "integration"|
| thinking_mode   | string          | No       | Thinking budget tier                                               |

## Process Steps

1. **Coverage data collection** -- Gather existing coverage data:
   - Line coverage: which lines are executed by existing tests
   - Branch coverage: which conditional branches are taken
   - Function coverage: which functions are called
   - If no coverage data exists, analyze test files to determine which functions are tested

2. **Gap identification** -- Find untested code paths:
   - Functions with no corresponding test
   - Branches never taken (always-true or always-false conditions in tests)
   - Error handlers (except blocks) never triggered
   - Default cases in match/case never exercised
   - Configuration-dependent paths (only one config tested)

3. **Risk-weighted prioritization** -- Not all coverage gaps are equal. Prioritize by risk:
   - **Critical**: untested error handlers in production code paths
   - **High**: untested input validation (security boundary)
   - **High**: untested state mutations (data integrity)
   - **Medium**: untested happy path branches
   - **Low**: untested logging or debug code
   - **Info**: untested cosmetic code (string formatting, display)

4. **Semantic coverage analysis** -- Check that tests are testing the right things:
   - Do tests verify outputs, not just execution? (assertion coverage)
   - Do tests check error conditions, not just success? (negative testing)
   - Do tests use realistic inputs, not just trivial ones? (input quality)
   - Do tests verify side effects (database writes, file changes)? (effect coverage)
   - Are there tests that always pass regardless of code changes? (tautological tests)

5. **Test specification generation** -- For each gap, generate a test specification:
   - Test name following project convention: `test_{function}_{scenario}`
   - Input data that exercises the untested path
   - Expected output or expected exception
   - Required mocking or fixture setup
   - Group: happy_path, edge_case, or error_case

6. **Coverage improvement estimation** -- Calculate expected coverage improvement:
   - Current coverage percentage
   - Expected coverage after implementing suggested tests
   - Number of new tests needed
   - Effort estimate for test implementation

7. **Test quality assessment** -- Evaluate existing tests for quality issues:
   - Tests that depend on execution order (shared state)
   - Tests that depend on external resources (network, file system)
   - Tests that are flaky (non-deterministic)
   - Tests that test implementation details rather than behavior

8. **Report compilation** -- Produce the coverage analysis report with gaps, specifications, and quality assessment.

## Output Format

The skill produces a coverage analysis report:

```json
{
  "passed": false,
  "grade": "C",
  "score": 0.65,
  "issues": [
    {
      "severity": "high",
      "category": "coverage",
      "description": "Error handler on line 45 never tested -- FileNotFoundError path",
      "file": "vetinari/config/loader.py",
      "line": 45,
      "suggestion": "Add test: test_load_config_with_missing_file that verifies FileNotFoundError is raised"
    },
    {
      "severity": "medium",
      "category": "coverage",
      "description": "Branch on line 23 (empty input list) never exercised",
      "file": "vetinari/planning/decomposition.py",
      "line": 23,
      "suggestion": "Add test: test_decompose_with_empty_goal_list"
    }
  ],
  "metrics": {
    "line_coverage": 0.78,
    "branch_coverage": 0.62,
    "function_coverage": 0.85,
    "projected_coverage_after_fixes": 0.91,
    "tests_needed": 8,
    "effort_estimate": "M"
  },
  "test_specifications": [
    {
      "name": "test_load_config_with_missing_file",
      "target": "vetinari/config/loader.py:load_config",
      "category": "error_case",
      "input": "Path('/nonexistent/config.yaml')",
      "expected": "raises FileNotFoundError",
      "mocking": "none",
      "priority": "high"
    },
    {
      "name": "test_decompose_with_empty_goal_list",
      "target": "vetinari/planning/decomposition.py:decompose",
      "category": "edge_case",
      "input": "goals=[]",
      "expected": "returns empty plan with no tasks",
      "mocking": "none",
      "priority": "medium"
    }
  ],
  "quality_issues": [
    {
      "test": "test_agent_graph.py::test_full_pipeline",
      "issue": "Uses time.sleep(1) -- fragile timing dependency",
      "recommendation": "Use event-based synchronization instead of sleep"
    }
  ]
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-INS-004**: Test generation MUST cover happy path, edge cases, and error paths
- **STD-INS-005**: Every issue MUST have a severity level and actionable description
- **STD-INS-006**: Inspector MUST NOT modify code -- only report findings and suggestions
- **STD-INS-007**: Gate decision MUST be based on objective criteria
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-INS-001**: Inspector is READ-ONLY -- cannot modify production files
- **CON-INS-002**: Inspector cannot be the same entity that produced the code

## Examples

### Example: Analyzing coverage for a new module

**Input:**
```
code: [contents of vetinari/orchestration/stagnation.py]
context: {
  test_file: "tests/test_stagnation.py (does not exist)",
  module_purpose: "Detect and handle stagnant tasks in the execution pipeline"
}
```

**Output (abbreviated):**
```
score: 0.0 (no tests exist)
grade: F

gaps:
  - [critical] detect_stagnation() -- core function, completely untested
  - [critical] StagnationPolicy.__init__() -- configuration validation untested
  - [high] _compute_progress_derivative() -- numerical computation untested
  - [medium] _format_stagnation_alert() -- output formatting untested

test_specifications:
  1. test_detect_stagnation_with_progressing_task (happy path)
  2. test_detect_stagnation_with_stuck_task (happy path)
  3. test_detect_stagnation_with_no_tasks (edge case)
  4. test_detect_stagnation_with_negative_progress (edge case)
  5. test_stagnation_policy_with_invalid_threshold (error case)
  6. test_compute_progress_derivative_with_single_point (edge case)
  7. test_format_stagnation_alert_includes_task_id (correctness)

projected_coverage: 85% (from 0%) with 7 tests
effort: S
```
