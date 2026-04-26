---
name: Task Decomposition
description: Break high-level goals into executable DAGs with dependency analysis, wave sequencing, and parallelism identification
mode: plan
agent: foreman
version: "1.0.0"
capabilities:
  - goal_decomposition
  - task_scheduling
  - dependency_mapping
tags:
  - planning
  - decomposition
  - dag
  - scheduling
---

# Task Decomposition

## Purpose

Task Decomposition converts an ambiguous or complex user goal into a directed acyclic graph (DAG) of atomic, executable tasks. It follows a progressive refinement approach: first identify major phases, then break each phase into atomic tasks, then establish dependency edges and compute the critical path. This is the foundational skill that enables Vetinari's entire execution pipeline -- without a well-formed DAG, downstream Worker and Inspector agents cannot operate efficiently or correctly.

## When to Use

- A user submits a multi-step goal that cannot be completed by a single agent invocation
- The goal spans multiple files, modules, or subsystems in the codebase
- There are implicit ordering constraints (e.g., "add a feature" implies research, implement, test, document)
- The Foreman needs to identify which tasks can run in parallel versus which must be sequential
- A prior plan was too coarse-grained and needs finer decomposition
- The system needs to estimate total effort before committing resources

## Inputs

| Parameter     | Type            | Required | Description                                                        |
|---------------|-----------------|----------|--------------------------------------------------------------------|
| goal          | string          | Yes      | The high-level goal to decompose into tasks                        |
| constraints   | list[string]    | No       | Constraints to respect (e.g., "no breaking changes", "Python only")|
| context       | dict            | No       | Codebase state, prior plans, relevant memories                     |
| max_depth     | int (1-5)       | No       | Maximum decomposition depth (default: 3)                           |
| scope_report  | dict            | No       | Output from scope-analysis skill if already computed               |
| effort_budget | string          | No       | Maximum effort budget (e.g., "M", "L") to constrain plan size      |

## Process Steps

1. **Goal parsing** -- Extract the core intent, desired outcome, and implicit requirements from the natural-language goal. Identify any ambiguity that would require clarification before proceeding.

2. **Phase identification** -- Break the goal into 2-5 major phases (e.g., Research, Design, Implement, Test, Document). Each phase represents a logical stage of work that can be planned independently.

3. **Atomic task extraction** -- Within each phase, identify the smallest meaningful units of work. An atomic task has a single assigned agent, clear inputs, clear outputs, and can be verified independently. Target 1-15 minutes of agent time per task.

4. **Dependency analysis** -- For each pair of tasks, determine if there is a data dependency (Task B needs output from Task A), a resource dependency (both tasks modify the same file), or no dependency (can run in parallel). Build the dependency edge list.

5. **DAG validation** -- Verify the dependency graph is acyclic using topological sort. If cycles are detected, break them by introducing intermediate checkpoint tasks or restructuring the decomposition.

6. **Wave sequencing** -- Group tasks into execution waves. Wave 0 contains all tasks with no dependencies (roots). Wave N contains tasks whose dependencies are all in waves 0 through N-1. This naturally identifies the maximum parallelism at each stage.

7. **Critical path computation** -- Identify the longest chain of sequential dependencies. This is the minimum wall-clock time for the plan. Flag critical-path tasks as high priority since any delay on them delays the entire plan.

8. **Effort estimation** -- Assign an effort estimate (XS/S/M/L/XL) to each task based on historical data and task characteristics. Aggregate to produce a total plan effort estimate.

9. **Verification task injection** -- For every implementation task, insert a corresponding Inspector verification task as a downstream dependency. This ensures the plan enforces quality gates.

10. **Plan assembly** -- Compile the final plan object with task list, dependency graph, critical path, wave assignments, effort estimates, and risk notes.

## Output Format

The skill produces a plan object conforming to the Foreman output schema:

```json
{
  "plan_id": "plan-abc123",
  "goal": "Add rate limiting to the REST API",
  "version": "1",
  "tasks": [
    {
      "id": "T1",
      "description": "Research existing rate limiting patterns in the codebase",
      "assigned_agent": "WORKER",
      "inputs": ["goal description"],
      "outputs": ["research report with patterns found"],
      "dependencies": [],
      "effort": "S",
      "acceptance_criteria": "Report lists all existing middleware and identifies extension points"
    },
    {
      "id": "T2",
      "description": "Design rate limiting middleware with token bucket algorithm",
      "assigned_agent": "WORKER",
      "inputs": ["research report from T1"],
      "outputs": ["ADR with design decision", "interface specification"],
      "dependencies": ["T1"],
      "effort": "M",
      "acceptance_criteria": "ADR accepted, interface matches existing middleware pattern"
    }
  ],
  "critical_path": ["T1", "T2", "T3", "T5", "T6"],
  "risks": [
    "Rate limiting configuration may vary per endpoint, increasing task count",
    "Existing tests may need updating if middleware order changes"
  ],
  "estimated_duration": "2-3 hours agent time"
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-FMN-001**: Every generated task MUST have a unique ID and an assigned_agent from {FOREMAN, WORKER, INSPECTOR}
- **STD-FMN-002**: Dependency graphs MUST be acyclic (DAG); circular dependencies are forbidden
- **STD-FMN-003**: Plans MUST include at least one verification task per implementation task
- **STD-FMN-004**: Plans MUST include risk assessment and rollback strategy for destructive operations
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-002**: All skill executions MUST return a ToolResult; exceptions MUST be caught and reported via error field
- **CON-FMN-001**: Maximum 50 tasks in a single plan
- **CON-FMN-002**: Maximum decomposition depth of 5 levels
- **CON-FMN-004**: Foreman MUST NOT execute tasks directly -- only plan, clarify, and delegate

## Examples

### Example: Multi-file feature addition

**Input:**
```
goal: "Add a health check endpoint to the Vetinari web API that reports system status including database connectivity, model availability, and scheduler state"
constraints: ["Must follow existing Flask route patterns", "No new dependencies"]
```

**Output (abbreviated):**
```
plan_id: plan-health-001
tasks:
  Wave 0 (parallel):
    T1: [WORKER/code_discovery] Explore existing route patterns in vetinari/web/
    T2: [WORKER/code_discovery] Identify all subsystems that need health reporting

  Wave 1 (parallel, depends on T1+T2):
    T3: [WORKER/architecture] Design health check response schema and endpoint contract
    T4: [WORKER/code_discovery] Map existing health/status functions across subsystems

  Wave 2 (sequential, depends on T3):
    T5: [WORKER/build] Implement /api/health endpoint with subsystem checks
    T6: [WORKER/build] Write tests for health endpoint (happy, degraded, failure scenarios)

  Wave 3 (parallel, depends on T5+T6):
    T7: [INSPECTOR/code_review] Review health endpoint implementation
    T8: [WORKER/documentation] Update API documentation with new endpoint

critical_path: [T1, T3, T5, T7]
estimated_duration: "1-2 hours"
risks:
  - "Subsystem health checks may have side effects if not carefully isolated"
  - "Response time may be slow if health checks are synchronous"
```
