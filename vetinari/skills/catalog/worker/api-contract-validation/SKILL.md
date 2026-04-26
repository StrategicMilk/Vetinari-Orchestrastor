---
name: API Contract Validation
description: Validate backward compatibility of API changes including schema evolution, deprecation handling, and consumer impact
mode: architecture
agent: worker
version: "1.0.0"
capabilities:
  - api_contract_validation
  - architecture_review
tags:
  - architecture
  - api
  - contracts
  - compatibility
---

# API Contract Validation

## Purpose

API Contract Validation ensures that changes to public interfaces (REST endpoints, function signatures, data schemas, agent contracts) maintain backward compatibility or follow proper deprecation procedures. It detects breaking changes before they reach production by comparing the proposed interface against the current contract, identifying consumers that would break, and verifying that any intentional breaking changes follow the project's evolution strategy. This skill prevents the most disruptive class of integration failure -- silently breaking callers who depend on stable interfaces.

## When to Use

- Before modifying any public function signature, return type, or exception set
- Before changing REST endpoint URLs, request schemas, or response schemas
- Before modifying dataclass fields in contracts.py or types in types.py
- When adding, removing, or renaming fields in agent input/output schemas
- Before releasing a new version that changes the public API surface
- When evaluating whether a change is backward-compatible or requires a migration

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What API change to validate                                        |
| current_contract| dict            | No       | Current API schema or function signatures                          |
| proposed_change | dict            | No       | Proposed modifications to the API                                  |
| consumers       | list[string]    | No       | Known consumers of the API (files, modules, external clients)      |
| files           | list[string]    | No       | Files containing the API to validate                               |
| context         | dict            | No       | Versioning strategy, deprecation policy                            |

## Process Steps

1. **Contract extraction** -- Extract the current API contract from code: function signatures, type annotations, docstrings, JSON schemas, REST endpoint definitions. Build a structured representation of the public API surface.

2. **Change identification** -- Compare proposed changes against the current contract. Classify each change as:
   - **Additive**: new endpoint, new optional field, new function (always safe)
   - **Modification**: changed type, changed validation, changed behavior (potentially breaking)
   - **Removal**: deleted endpoint, removed field, removed function (always breaking)
   - **Rename**: changed name without changing behavior (breaking for callers)

3. **Breaking change detection** -- For each modification and removal, determine if it is backward-incompatible:
   - Adding a required field to a request schema: BREAKING (existing callers don't send it)
   - Adding an optional field to a response schema: SAFE (existing callers ignore unknown fields)
   - Changing a field type from string to int: BREAKING (existing callers parse as string)
   - Removing a response field: BREAKING (existing callers may depend on it)

4. **Consumer impact analysis** -- For each breaking change, identify all consumers and the specific impact on each. Use LSP find-references and grep to locate all callers. Distinguish between internal consumers (can be updated) and external consumers (must be given migration time).

5. **Deprecation compliance check** -- If the project has a deprecation policy, verify that breaking changes follow it:
   - Deprecated features marked with warnings in prior version
   - Migration guide provided for each breaking change
   - Sunset period allows consumers time to adapt
   - Old behavior still available during deprecation period

6. **Schema evolution validation** -- For data schemas (Pydantic models, JSON schemas), verify evolution rules:
   - New required fields have defaults (backward-compatible deserialization)
   - Enum values can be added but not removed (forward-compatible)
   - Field types can be widened (string to string|int) but not narrowed

7. **Version compatibility matrix** -- Build a compatibility matrix showing which client versions work with which server versions. Identify the compatibility window and any versions that would break.

8. **Migration path documentation** -- For each breaking change, document the migration path: what callers need to change, automated migration tools available, and timeline for support of the old behavior.

9. **Validation report assembly** -- Compile findings into a contract validation report with: pass/fail verdict, breaking changes list, consumer impact, and migration requirements.

## Output Format

The skill produces a contract validation report:

```json
{
  "success": true,
  "output": {
    "verdict": "breaking_changes_detected",
    "compatible": false,
    "changes": [
      {
        "type": "modification",
        "location": "vetinari/agents/contracts.py:AgentResult",
        "field": "output",
        "current": "str",
        "proposed": "str | dict[str, Any]",
        "breaking": false,
        "reason": "Widening type is backward-compatible for producers but consumers must handle new type"
      },
      {
        "type": "removal",
        "location": "vetinari/agents/contracts.py:AgentResult",
        "field": "legacy_score",
        "breaking": true,
        "consumers": [
          "vetinari/orchestration/agent_graph.py:line 45",
          "tests/test_agent_contracts.py:line 123"
        ],
        "migration": "Replace legacy_score with quality_metrics.overall_score"
      }
    ],
    "impact_summary": {
      "internal_consumers_affected": 3,
      "external_consumers_affected": 0,
      "tests_affected": 2,
      "migration_effort": "S"
    },
    "recommendations": [
      "Add legacy_score as deprecated alias that delegates to quality_metrics.overall_score",
      "Mark with DeprecationWarning for 2 versions before removal",
      "Update 3 internal consumers to use new field"
    ]
  },
  "metadata": {
    "adr_reference": "ADR-XXXX (AgentResult schema evolution)"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-003**: Architecture mode MUST produce or reference an ADR for every design decision
- **STD-WRK-004**: Architecture modes are READ-ONLY -- MUST NOT modify production files
- **STD-WRK-006**: Component designs MUST define clear boundaries (inputs, outputs, dependencies)
- **STD-WRK-008**: API designs MUST include authentication, authorization, and rate limiting
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-WRK-002**: Architecture modes are READ-ONLY -- produce ADRs, not code changes

## Examples

### Example: Validating a field type change

**Input:**
```
task: "Validate changing Task.effort from string to Enum"
current_contract: {field: "effort", type: "str", values: ["XS", "S", "M", "L", "XL"]}
proposed_change: {field: "effort", type: "EffortLevel (Enum)", values: ["XS", "S", "M", "L", "XL"]}
```

**Output (abbreviated):**
```
verdict: "breaking_changes_detected"
compatible: false

changes:
  - type: modification
    field: Task.effort
    current: str
    proposed: EffortLevel (Enum)
    breaking: true
    reason: "Callers that assign string values (task.effort = 'M') will get TypeError"
    consumers: 8 files, 15 call sites

migration:
  1. "Add EffortLevel enum to vetinari/types.py"
  2. "Change Task.effort type annotation but accept both str and Enum via validator"
  3. "Update all 15 call sites to use EffortLevel.M instead of 'M'"
  4. "After all callers updated, remove str acceptance from validator"
  5. "Total effort: M (8 files, 15 call sites, straightforward string-to-enum)"
```
