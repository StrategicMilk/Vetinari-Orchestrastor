---
name: Feature Implementation
description: Implement features with full type safety, tests, documentation, and proper wiring as the sole producer of production code
mode: build
agent: worker
version: "1.0.0"
capabilities:
  - feature_implementation
  - code_generation
tags:
  - build
  - implementation
  - production-code
  - feature
---

# Feature Implementation

## Purpose

Feature Implementation is the sole producer of production code in the Vetinari pipeline. It takes a fully specified task (from the Foreman's decomposition) and produces complete, tested, documented code that follows all project conventions. This skill enforces the single-writer rule: only build mode may create or modify files under `vetinari/`. It is responsible for the entire implementation lifecycle: writing code, adding type annotations, creating tests, writing docstrings, and ensuring proper wiring (every new symbol is imported and used somewhere).

## When to Use

- When a Foreman plan includes a build task for new functionality
- When implementing a feature that has been fully specified with acceptance criteria
- When adding new modules, classes, functions, or endpoints
- When extending existing functionality with new capabilities
- When the implementation requires coordinated changes across multiple files
- Only after research and architecture skills have established patterns and made decisions

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | Detailed task description with acceptance criteria                 |
| files           | list[string]    | No       | Files to create or modify                                          |
| context         | dict            | No       | Codebase context, patterns from code_discovery, ADRs from architecture|
| acceptance      | list[string]    | No       | Given-When-Then acceptance criteria to satisfy                     |
| thinking_mode   | string          | No       | Thinking budget: "low", "medium", "high", "xhigh"                 |

## Process Steps

1. **Pre-implementation check** -- Verify all prerequisites are met: task is fully specified, patterns have been discovered (via code_discovery), design decisions are made (via architecture), and the scope is clear. If any prerequisite is missing, request it rather than guessing.

2. **Pattern alignment** -- Review the patterns discovered by code_discovery for the target area. Note: naming conventions, import style, error handling approach, logging format, test structure. All new code must match these patterns exactly.

3. **Interface design** -- Define the public API surface first: function signatures with full type annotations, class interfaces, and docstrings. This establishes the contract before writing implementation details.

4. **Implementation** -- Write the production code following project conventions:
   - `from __future__ import annotations` as first import
   - Module docstring explaining purpose
   - `logger = logging.getLogger(__name__)` for logging
   - Specific exception handling with `from` chaining
   - `encoding="utf-8"` on all file I/O
   - No print(), no magic numbers, no placeholder code

5. **Test creation** -- Write tests alongside the implementation:
   - Test file mirrors source: `vetinari/foo.py` -> `tests/test_foo.py`
   - Test naming: `test_{function}_{scenario}`
   - Cover happy path, edge cases, and error cases
   - Use pytest fixtures, not setUp/tearDown
   - Mock external dependencies
   - Each test independent and isolated

6. **Documentation** -- Write Google-style docstrings for all public symbols:
   - Functions with 2+ params get Args section
   - Functions with return values get Returns section
   - Functions that raise get Raises section
   - Module-level docstring explaining purpose

7. **Wiring verification** -- Ensure every new symbol is connected to the existing system:
   - New modules are imported in `__init__.py`
   - New functions are called from at least one place
   - New classes are instantiated or referenced
   - New CLI commands are registered
   - New config options are documented

8. **Import validation** -- Verify all imports use canonical sources:
   - Enums from `vetinari.types`
   - Specs from `vetinari.agents.contracts`
   - No wildcard imports
   - No circular imports
   - Every third-party import has a pyproject.toml entry

9. **Quality gate preparation** -- Self-check before submitting to Inspector:
   - Run ruff check and ruff format
   - Run pytest on affected test files
   - Verify import chain works: `python -c "import vetinari; print('OK')"`
   - Check for TODO, FIXME, pass bodies, placeholder strings

10. **Deliverable assembly** -- Compile the list of files changed, tests added, and any follow-up tasks identified during implementation.

## Output Format

The skill produces implementation results with file change manifest:

```json
{
  "success": true,
  "output": "Implemented rate limiting middleware with token bucket algorithm",
  "files_changed": [
    "vetinari/web/middleware.py (new file - rate limiting middleware)",
    "vetinari/web/__init__.py (updated - registered middleware)",
    "vetinari/config/rules.yaml (updated - added rate limit defaults)",
    "tests/test_middleware.py (new file - 8 tests)"
  ],
  "tests_added": 8,
  "tests_passed": 8,
  "warnings": [],
  "metadata": {
    "lines_added": 245,
    "lines_modified": 12,
    "coverage_added": "vetinari/web/middleware.py: 95%"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-011**: Build mode is the SOLE writer of production files -- no other mode may modify code
- **STD-WRK-012**: All new code MUST have type annotations, Google-style docstrings, and tests
- **STD-WRK-013**: Imports MUST use canonical sources (enums from vetinari.types, specs from contracts)
- **STD-WRK-014**: No print() in production code -- use logging module with %-style formatting
- **STD-WRK-015**: File I/O MUST specify encoding='utf-8'
- **STD-WRK-016**: No TODO, FIXME, pass bodies, NotImplementedError, or placeholder strings
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-WRK-004**: Build mode is the SOLE production file writer -- enforced by execution context
- **CON-WRK-005**: Destructive operations (delete, overwrite) require confirmation
- **GDL-WRK-005**: Write tests before implementation (TDD) for complex logic

## Examples

### Example: Implementing a new API endpoint

**Input:**
```
task: "Add GET /api/health endpoint that reports system status"
acceptance:
  - "Given the server is running, When GET /api/health is called, Then return 200 with {status: 'healthy', components: {...}}"
  - "Given the database is unreachable, When GET /api/health is called, Then return 200 with {status: 'degraded', components: {database: 'unreachable'}}"
context: {patterns: "Flask blueprints, JSON responses, @login_required for protected routes"}
```

**Output (abbreviated):**
```
success: true
files_changed:
  - vetinari/web/health_api.py (new - health check endpoint with component checks)
  - vetinari/web/__init__.py (updated - registered health blueprint)
  - tests/test_health_api.py (new - 6 tests covering healthy, degraded, error scenarios)

implementation_notes:
  - "Health endpoint is public (no @login_required) -- consistent with industry practice"
  - "Each component check has a 5-second timeout to prevent health check from hanging"
  - "Returns 200 even when degraded (per RFC, health endpoints should not return 5xx)"
```
