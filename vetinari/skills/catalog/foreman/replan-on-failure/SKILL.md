---
name: Replan on Failure
description: Dynamically adjust DAGs when tasks fail or scope changes, identifying salvageable work and creating recovery plans
mode: prune
agent: foreman
version: "1.0.0"
capabilities:
  - task_scheduling
  - dependency_mapping
  - token_management
tags:
  - planning
  - recovery
  - replanning
  - pruning
---

# Replan on Failure

## Purpose

Replan on Failure handles the inevitable situation where a plan's execution diverges from expectations. When a task fails, produces unexpected output, or reveals that the scope was wrong, this skill analyzes the impact on the remaining DAG and produces a recovery plan. It determines which downstream tasks are invalidated, which can be salvaged with modified inputs, and what new tasks are needed. This prevents the pipeline from either halting entirely on a single failure or blindly continuing with invalid assumptions.

## When to Use

- A Worker task fails with an unrecoverable error after retry attempts are exhausted
- A task succeeds but its output contradicts assumptions made by downstream tasks
- An Inspector rejects a deliverable, invalidating tasks that depend on it
- New information surfaces mid-execution that changes the scope or requirements
- Token or cost budget is being exceeded and the plan needs pruning
- A task reveals that the original decomposition was wrong (e.g., expected 1 file, found 10)
- Stagnation detection triggers -- a task has been running too long without progress

## Inputs

| Parameter       | Type            | Required | Description                                                          |
|-----------------|-----------------|----------|----------------------------------------------------------------------|
| plan            | dict            | Yes      | The current plan with task statuses                                  |
| failed_task_id  | string          | Yes      | ID of the task that failed or triggered replanning                   |
| failure_reason  | string          | Yes      | Description of why the task failed                                   |
| failure_type    | string          | No       | Classification: transient, decomposition, delegation, unsolvable     |
| completed_tasks | list[dict]      | No       | Tasks already completed with their outputs                           |
| budget_remaining| dict            | No       | Remaining token and cost budget                                      |
| context         | dict            | No       | Updated context including any new information from failed task        |

## Process Steps

1. **Failure classification** -- Categorize the failure as: transient (retry may succeed), decomposition (task was incorrectly defined), delegation (wrong agent assigned), unsolvable (cannot be done as specified), or policy_violation (breaks a safety rule). Each category has a different recovery strategy.

2. **Impact propagation** -- Walk the dependency graph forward from the failed task. Mark all direct dependents as "blocked". For each blocked task, check if it has alternative input sources or if the dependency is truly hard. Mark transitively blocked tasks.

3. **Salvage assessment** -- For each blocked task, determine if it can be: (a) executed with modified inputs, (b) executed with a different approach, (c) skipped without compromising the goal, or (d) must wait for the failed task to be resolved.

4. **Completed work preservation** -- Catalog all completed tasks and their outputs. These are sunk costs that should be preserved. Ensure the recovery plan does not redo completed work unless the failure invalidates their outputs.

5. **Recovery strategy selection** -- Based on failure type:
   - **Transient**: Requeue the failed task with exponential backoff
   - **Decomposition**: Re-decompose the failed task into smaller subtasks
   - **Delegation**: Reassign to a different agent mode or thinking level
   - **Unsolvable**: Prune the failed branch and notify the user of scope reduction
   - **Policy violation**: Reformulate the task to comply with safety rules

6. **DAG surgery** -- Modify the dependency graph: remove invalidated edges, add new tasks for recovery, reroute dependencies from failed tasks to recovery tasks. Maintain DAG invariant (no cycles).

7. **Budget recalculation** -- Recompute the effort estimate for the modified plan. If the remaining budget is insufficient, identify tasks to prune by priority (cut nice-to-haves first, preserve critical-path tasks).

8. **Critical path recomputation** -- The critical path may have shifted due to task removal or addition. Recompute it and flag any new bottlenecks.

9. **Stagnation guard** -- Set a maximum number of replan cycles (default: 3) for the same task. If a task has been replanned 3 times, escalate to the user rather than continuing to retry.

10. **Recovery plan emission** -- Output the modified plan with: updated task statuses, new/modified tasks, removed tasks, updated critical path, and a human-readable summary of what changed and why.

## Output Format

The skill produces a modified plan with recovery annotations:

```json
{
  "plan_id": "plan-abc123",
  "version": "3",
  "recovery_summary": "Task T4 (implement caching) failed due to decomposition error. Split into T4a (cache interface) and T4b (cache backend). T5 (tests) rescheduled to depend on T4b.",
  "changes": {
    "removed": ["T4"],
    "added": [
      {"id": "T4a", "description": "Define cache interface", "assigned_agent": "WORKER"},
      {"id": "T4b", "description": "Implement cache with TTL", "assigned_agent": "WORKER", "dependencies": ["T4a"]}
    ],
    "modified": [
      {"id": "T5", "change": "dependencies updated from [T4] to [T4b]"}
    ],
    "preserved": ["T1", "T2", "T3"]
  },
  "tasks": ["...full updated task list..."],
  "critical_path": ["T1", "T2", "T4a", "T4b", "T5", "T7"],
  "replan_count": 1,
  "budget_impact": "+2 tasks, estimated +15min agent time"
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-FMN-002**: Dependency graphs MUST be acyclic (DAG); circular dependencies are forbidden (critical after DAG surgery)
- **STD-FMN-001**: Every generated task MUST have a unique ID and an assigned_agent
- **STD-FMN-004**: Plans MUST include risk assessment and rollback strategy for destructive operations
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-002**: All skill executions MUST return a ToolResult; exceptions MUST be caught and reported
- **CON-FMN-001**: Maximum 50 tasks in a single plan (even after replan additions)
- **CON-FMN-003**: Plans involving file deletion or destructive ops require explicit confirmation

## Examples

### Example: Inspector rejection triggers replan

**Input:**
```
plan: {plan_id: "plan-feat-001", tasks: [T1..T8]}
failed_task_id: "T6"
failure_reason: "Inspector rejected: security audit found SQL injection in query builder"
failure_type: "delegation"
completed_tasks: [T1, T2, T3, T4, T5]
```

**Output:**
```
recovery_summary: "Inspector T6 rejected T5's output (SQL injection). T5 reassigned with security-hardened prompt. T6 re-queued after T5 re-execution. T7 (docs) and T8 (deploy) remain blocked."

changes:
  removed: []
  added:
    - T5-fix: [WORKER/build] "Rewrite query builder using parameterized queries"
      dependencies: [T4]  # same as original T5
    - T6-re: [INSPECTOR/security_audit] "Re-audit fixed query builder"
      dependencies: [T5-fix]
  modified:
    - T7: dependencies changed from [T6] to [T6-re]
    - T8: dependencies changed from [T7] to [T7] (unchanged, still blocked)
  preserved: [T1, T2, T3, T4]

replan_count: 1
budget_impact: "+1 build task, +1 review task, estimated +20min"
```
