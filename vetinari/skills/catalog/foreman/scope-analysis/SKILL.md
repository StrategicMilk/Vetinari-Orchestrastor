---
name: Scope Analysis
description: Map all affected files, tests, and dependencies via LSP references before planning any change
mode: plan
agent: foreman
version: "1.0.0"
capabilities:
  - dependency_mapping
  - specification
tags:
  - planning
  - scope
  - blast-radius
  - impact-analysis
---

# Scope Analysis

## Purpose

Scope Analysis determines the full blast radius of a proposed change before any planning or implementation begins. It uses LSP references, import graphs, and test coverage data to map every file, function, test, and dependency that could be affected. This prevents the most common planning failure -- underestimating the scope of a change, leading to broken downstream code, missed test updates, and integration failures that surface late in the pipeline.

## When to Use

- Before decomposing any task that modifies shared modules (types.py, contracts.py, interfaces.py)
- When a change touches public APIs, function signatures, or data structures
- When renaming or moving symbols across modules
- Before any refactoring that spans more than 2 files
- When the Foreman needs to determine if a task is Trivial, Scoped, or Complex
- To validate that a proposed plan covers all affected areas

## Inputs

| Parameter       | Type            | Required | Description                                                          |
|-----------------|-----------------|----------|----------------------------------------------------------------------|
| goal            | string          | Yes      | Description of the proposed change                                   |
| target_files    | list[string]    | No       | Files known to require changes                                       |
| target_symbols  | list[string]    | No       | Functions, classes, or constants being modified                       |
| context         | dict            | No       | Codebase context including module graph, test mapping                 |
| include_tests   | bool            | No       | Whether to include test file impact (default: true)                  |
| depth           | int (1-5)       | No       | How many levels of transitive dependencies to trace (default: 2)     |

## Process Steps

1. **Entry point identification** -- From the goal and target_files/symbols, identify the primary code locations that will be directly modified. These are the "epicenter" of the change.

2. **Direct reference scan** -- Use LSP find-references on each target symbol to find all direct callers, importers, and users. Record file path, line number, and usage type (call, import, type annotation, test assertion).

3. **Transitive dependency tracing** -- For each direct reference, check if the referencing code is itself referenced by other modules. Trace up to `depth` levels. This catches second-order and third-order impacts.

4. **Import graph analysis** -- Build the import subgraph centered on affected modules. Identify circular import risks, re-export chains, and modules that re-expose the changing symbols.

5. **Test mapping** -- Map each affected production file to its corresponding test file(s). Identify tests that directly test the changing functions, tests that use fixtures involving the changing code, and integration tests that exercise the affected code paths.

6. **Configuration impact** -- Check if the change affects config files (YAML, TOML), environment variables, or deployment manifests. Flag any config-level changes needed.

7. **Risk classification** -- Classify the overall change scope:
   - **Trivial**: 1 file, no external references, tests obvious
   - **Scoped**: 2-5 files, clear boundaries, tests identifiable
   - **Complex**: 6+ files, cross-cutting concerns, test coverage uncertain

8. **Blast radius report** -- Compile the full list of affected files categorized by: directly modified, callers needing updates, tests needing updates, config needing updates, and documentation needing updates.

9. **Missing coverage detection** -- Identify any affected code paths that have no test coverage. These are high-risk areas where changes could introduce silent regressions.

10. **Scope boundary recommendation** -- Based on the analysis, recommend whether to proceed as planned, narrow the scope, or split into multiple smaller changes.

## Output Format

The skill produces a scope report used by task decomposition and effort estimation:

```json
{
  "classification": "scoped",
  "epicenter": ["vetinari/types.py:AgentType"],
  "direct_impact": {
    "files": [
      {"path": "vetinari/agents/contracts.py", "lines": [45, 112], "usage": "import"},
      {"path": "vetinari/agents/base_agent.py", "lines": [23], "usage": "type_annotation"}
    ],
    "count": 12
  },
  "transitive_impact": {
    "files": [
      {"path": "vetinari/orchestration/agent_graph.py", "lines": [67], "usage": "call"}
    ],
    "count": 5
  },
  "test_impact": {
    "files": [
      {"path": "tests/test_agent_contracts.py", "tests": ["test_agent_type_values"]},
      {"path": "tests/test_base_agent.py", "tests": ["test_init_with_type"]}
    ],
    "count": 8,
    "uncovered_paths": ["vetinari/drift/capability_auditor.py:line 34"]
  },
  "config_impact": [],
  "risks": [
    "Adding a new AgentType value requires updating 12 files that switch on agent type",
    "Two test files have no coverage for the affected code paths"
  ],
  "recommendation": "Proceed with scoped change. Create tasks for all 12 direct-impact files."
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-FMN-002**: Dependency graphs MUST be acyclic (DAG); circular dependencies are forbidden
- **STD-FMN-004**: Plans MUST include risk assessment and rollback strategy for destructive operations
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **CON-FMN-004**: Foreman MUST NOT execute tasks directly -- only plan, clarify, and delegate
- **GDL-FMN-001**: Prefer depth-first decomposition for complex goals; limit to 3 levels for simple goals

## Examples

### Example: Renaming a shared enum value

**Input:**
```
goal: "Rename MemoryType.WARNING to MemoryType.RISK_WARNING"
target_symbols: ["MemoryType.WARNING"]
depth: 3
```

**Output (abbreviated):**
```
classification: complex
epicenter: [vetinari/types.py:MemoryType.WARNING]
direct_impact:
  - vetinari/memory/unified.py (3 references)
  - vetinari/memory/_schema.py (2 references)
  - vetinari/orchestration/pipeline_helpers.py (1 reference)
  - docs/reference/memory.md (1 reference)
  count: 7 references across 4 files

transitive_impact:
  - vetinari/cli_commands.py (exposes memory search filters)
  - vetinari/web/memory_api.py (serializes memory entries)
  count: 4 references across 2 files

test_impact:
  - tests/test_memory_store.py (4 tests)
  - tests/test_memory_cli.py (2 tests)
  count: 6 tests across 2 files

risks:
  - "This is a high-impact shared module change (safety rule: types.py)"
  - "Config YAML files may use string representation that LSP cannot trace"
  - "Serialized data in databases or logs may contain old enum value"

recommendation: "Complex shared enum change. Split into: (1) add compatibility alias if needed, (2) migrate all references, (3) update docs and serialized examples, (4) remove compatibility only after verification."
```
