---
name: Comprehensive Code Review
description: Five-pass review covering correctness, style compliance, security, performance, and maintainability
mode: code_review
agent: inspector
version: "1.0.0"
capabilities:
  - code_review
  - performance_review
  - best_practices_enforcement
tags:
  - quality
  - review
  - gate
  - correctness
---

# Comprehensive Code Review

## Purpose

Comprehensive Code Review is the Inspector's primary quality gate. It performs a structured 5-pass review of production code, evaluating correctness, style compliance, security, performance, and maintainability as independent dimensions. Each pass uses different criteria and tools, ensuring that a strong showing in one dimension does not mask weaknesses in another. The review is read-only -- it produces findings and a pass/fail verdict but never modifies code. This separation of producer and reviewer is a core safety principle of the Vetinari pipeline.

## When to Use

- After every build task before the output is accepted into the plan
- When code has been modified as part of a bug fix and needs re-verification
- When a Worker task produces code that self-check flagged as uncertain
- Before merging any change that touches shared modules (types.py, contracts.py)
- When a previous review was failed and the fix needs re-review
- As part of the definition of done for any implementation task

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|------------------------------------------------------------------- |
| code            | string          | Yes      | Code to review (file contents or diff)                             |
| mode            | string          | No       | "code_review" (default)                                            |
| context         | dict            | No       | PR description, task spec, affected files, self_check result       |
| focus_areas     | list[string]    | No       | Specific areas to emphasize (e.g., "error_handling", "thread_safety")|
| thinking_mode   | string          | No       | Thinking budget: "low", "medium", "high", "xhigh"                 |

## Process Steps

1. **Pass 1: Correctness** -- Does the code do what it is supposed to do?
   - Logic correctness: conditions, loops, edge cases handled properly
   - Data flow: inputs are validated, outputs match return type declarations
   - Error handling: exceptions caught specifically, error paths tested
   - State management: no uninitialized variables, no stale state
   - Concurrency: thread safety for shared state, no race conditions
   - Algorithm correctness: boundary conditions, off-by-one errors

2. **Pass 2: Style compliance** -- Does the code follow project conventions?
   - `from __future__ import annotations` present in all files
   - Import order: stdlib, third-party, local (enforced by ruff isort)
   - Canonical imports: enums from `vetinari.types`, specs from `contracts`
   - Naming: snake_case functions, PascalCase classes, UPPER_SNAKE constants
   - Type annotations on all function signatures
   - Google-style docstrings on all public APIs
   - `encoding="utf-8"` on all file I/O
   - No print() in production code (use logging)
   - %-style formatting in logger calls (not f-strings)
   - No magic numbers without named constants

3. **Pass 3: Security** -- Is the code safe from common vulnerabilities?
   - Input validation: all external input validated before use
   - Injection: no SQL injection, command injection, or path traversal
   - Secrets: no hardcoded credentials, API keys, or tokens
   - Data exposure: no sensitive data in logs or error messages
   - Deserialization: no pickle or eval on untrusted data
   - File operations: path canonicalization, no symlink following

4. **Pass 4: Performance** -- Will the code perform acceptably?
   - Algorithmic complexity: no O(n^2) on potentially large collections
   - String operations: no concatenation in loops (use join)
   - I/O efficiency: buffered reads, connection pooling, resource cleanup
   - Memory: no unbounded growth, no unnecessary copies of large objects
   - Caching: appropriate use of caching for expensive computations
   - Database: no N+1 queries, proper index usage

5. **Pass 5: Maintainability** -- Will a future developer understand and modify this easily?
   - Single responsibility: each function/class does one thing
   - Coupling: minimal dependencies between modules
   - Cohesion: related functionality grouped together
   - Complexity: no deeply nested conditionals (max 3 levels)
   - Dead code: no unused functions, imports, or variables
   - Duplication: no copy-paste code that should be extracted
   - Documentation: intent is clear from names and comments

6. **Issue compilation** -- For each finding, record:
   - Severity: critical (must fix), high (should fix), medium (improve), low (suggestion), info (observation)
   - File and line number
   - Category (which pass found it)
   - Description (what is wrong)
   - Suggestion (how to fix it)

7. **Grade calculation** -- Compute an overall grade based on issue distribution:
   - **A**: 0 critical, 0 high, <= 2 medium
   - **B**: 0 critical, <= 1 high, <= 5 medium
   - **C**: 0 critical, <= 3 high, <= 10 medium
   - **D**: 0 critical, > 3 high or > 10 medium
   - **F**: any critical issue

8. **Gate decision** -- Determine pass/fail:
   - **Pass**: grade A or B (no critical, <=1 high)
   - **Conditional pass**: grade C (no critical, fixable issues)
   - **Fail**: grade D or F (critical issues or too many high-severity issues)
   - Gate decisions are objective and cannot be overridden by non-human agents

## Output Format

The skill produces a structured review report:

```json
{
  "passed": false,
  "grade": "D",
  "score": 0.55,
  "issues": [
    {
      "severity": "high",
      "category": "correctness",
      "description": "Race condition: dict modified during iteration in concurrent access path",
      "file": "vetinari/memory/blackboard.py",
      "line": 45,
      "suggestion": "Use threading.Lock to protect dict access or copy dict before iteration"
    },
    {
      "severity": "medium",
      "category": "style",
      "description": "Missing type annotation on return value",
      "file": "vetinari/memory/blackboard.py",
      "line": 32,
      "suggestion": "Add -> dict[str, Any] return type annotation"
    },
    {
      "severity": "low",
      "category": "maintainability",
      "description": "Function exceeds 50 lines -- consider extracting helper",
      "file": "vetinari/memory/blackboard.py",
      "line": 60,
      "suggestion": "Extract the serialization logic into _serialize_entry()"
    }
  ],
  "suggestions": [
    "Consider adding a thread-safety section to the module docstring"
  ],
  "metrics": {
    "lines_reviewed": 245,
    "issues_per_100_lines": 1.63,
    "pass_dimensions": {
      "correctness": false,
      "style": true,
      "security": true,
      "performance": true,
      "maintainability": true
    }
  },
  "self_check_passed": true
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-INS-001**: Code review MUST check all 5 dimensions: correctness, style, security, performance, maintainability
- **STD-INS-005**: Every issue MUST have a severity level and actionable description
- **STD-INS-006**: Inspector MUST NOT modify code -- only report findings and suggestions
- **STD-INS-007**: Gate decision (passed=true/false) MUST be based on objective criteria
- **STD-INS-008**: Gate decisions cannot be overridden by any other agent -- only humans can bypass
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-INS-001**: Inspector is READ-ONLY -- cannot modify production files
- **CON-INS-002**: Inspector cannot be the same entity that produced the code under review
- **CON-INS-003**: Gate decisions cannot be overridden by non-human agents

## Examples

### Example: Reviewing a new utility function

**Input:**
```
code: |
  def merge_configs(base, override):
      result = base
      for k, v in override.items():
          result[k] = v
      return result
context: {task: "Add configuration merging utility", self_check_passed: false}
```

**Output (abbreviated):**
```
passed: false
grade: D
issues:
  - [high/correctness] "Mutates the 'base' dict in-place (result = base creates alias, not copy). Callers who pass their config dict will have it silently modified."
    suggestion: "Use result = {**base, **override} or result = base.copy()"

  - [high/style] "Missing type annotations on both parameters and return type"
    suggestion: "def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:"

  - [high/style] "Missing docstring on public function"
    suggestion: "Add Google-style docstring with Args and Returns sections"

  - [medium/correctness] "Does not handle nested dicts -- override={'db': {'port': 5432}} will replace entire 'db' key, not merge into it"
    suggestion: "Add recursive merging for nested dicts if that's the intended behavior"

  - [medium/style] "Missing from __future__ import annotations"
```
