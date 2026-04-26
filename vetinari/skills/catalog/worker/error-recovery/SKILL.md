---
name: Error Recovery
description: Classify failures by type and execute appropriate recovery strategies for transient, decomposition, and delegation errors
mode: error_recovery
agent: worker
version: "1.0.0"
capabilities:
  - error_recovery
tags:
  - operations
  - recovery
  - resilience
  - error-handling
---

# Error Recovery

## Purpose

Error Recovery classifies runtime failures into actionable categories and executes the appropriate recovery strategy for each. Rather than treating all failures the same (retry and hope), it distinguishes between transient errors (retry with backoff), decomposition errors (task was incorrectly defined), delegation errors (wrong agent or mode assigned), unsolvable errors (fundamentally impossible), and policy violations (breaks a safety rule). Each category has a specific recovery protocol that maximizes the chance of successful recovery while minimizing wasted tokens on futile retries.

## When to Use

- When an agent task fails during execution
- When a task times out without producing output
- When a task produces output that fails validation
- When the orchestration layer detects stagnation (no progress over time)
- When the Inspector rejects output and the fix requires re-execution
- When a cascade of failures indicates a systemic issue rather than isolated defects

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | Description of the failure to recover from                         |
| error           | dict            | Yes      | Error details: type, message, stack trace, context                 |
| failed_task     | dict            | No       | The task that failed (id, description, assigned_agent, mode)       |
| retry_count     | int             | No       | How many times this task has already been retried                  |
| context         | dict            | No       | System state, recent task results, resource availability           |

## Process Steps

1. **Error classification** -- Analyze the error to determine its category:
   - **Transient** (recoverable by retry): network timeout, rate limit, temporary resource unavailability, model overload. Indicators: HTTP 429/503, connection errors, timeout errors.
   - **Decomposition** (task definition is wrong): task too large, contradictory requirements, missing prerequisites. Indicators: output exceeds token limit, task produces partial results, asks for clarification.
   - **Delegation** (wrong agent or mode): task assigned to agent without required capability, wrong thinking tier. Indicators: "I don't know how to do this", capability mismatch errors, consistently low-quality output.
   - **Unsolvable** (fundamentally impossible): violates physics, requires information that does not exist, contradicts hard constraints. Indicators: logical impossibility detected, required resource does not exist.
   - **Policy violation** (breaks safety rule): attempting destructive operation without confirmation, trying to modify files outside scope, exceeding cost budget. Indicators: enforcement module rejections.

2. **Recovery strategy selection** -- Based on classification:
   - **Transient**: Retry with exponential backoff (wait 1s, 2s, 4s, max 3 retries)
   - **Decomposition**: Signal Foreman to re-decompose the task into smaller subtasks
   - **Delegation**: Reassign to different agent mode or escalate thinking tier
   - **Unsolvable**: Report to user with explanation of why it cannot be done and suggest alternatives
   - **Policy violation**: Reformulate task to comply with policy, then re-execute

3. **Retry budget check** -- Before executing recovery, check:
   - Has this task already been retried `retry_count` times?
   - Is there remaining token/cost budget for recovery?
   - Is the overall plan still valid or has scope changed?
   - Maximum 3 retries per task; after that, escalate

4. **Pre-recovery cleanup** -- Before retrying or re-executing:
   - Clean up any partial outputs from the failed attempt
   - Release any resources held by the failed task
   - Update the blackboard with failure information
   - Log the failure classification and recovery strategy

5. **Recovery execution** -- Execute the chosen strategy:
   - For retry: re-queue the task with backoff delay and incremented retry count
   - For re-decomposition: create a Foreman replan task targeting the failed task
   - For reassignment: modify the task's assigned agent or mode and re-queue
   - For unsolvable: mark the task as failed and trigger Foreman replan for the plan
   - For policy violation: modify the task to comply and re-queue

6. **Recovery verification** -- After recovery execution:
   - Check if the recovered task produces valid output
   - Verify downstream tasks are not invalidated by the recovery approach
   - Update plan status to reflect the recovery

7. **Pattern detection** -- Check for systemic failure patterns:
   - Same task type failing repeatedly (indicates a design issue, not bad luck)
   - Multiple tasks failing in the same wave (indicates a shared dependency issue)
   - Increasing failure rate over time (indicates resource exhaustion or degradation)

8. **Escalation** -- If recovery fails after maximum retries, escalate:
   - Log full failure context including all retry attempts
   - Notify the Foreman with failure details for plan-level replanning
   - If the failure is systemic, recommend pausing the plan for human review

## Output Format

The skill produces a recovery report:

```json
{
  "success": true,
  "output": {
    "classification": "transient",
    "recovery_strategy": "retry_with_backoff",
    "recovery_action": {
      "action": "requeue",
      "task_id": "T5",
      "delay_seconds": 4,
      "retry_count": 2,
      "max_retries": 3,
      "modifications": "none"
    },
    "root_cause": "Model API returned HTTP 429 (rate limited)",
    "pattern_detected": false,
    "escalation_needed": false
  },
  "metadata": {
    "failure_type": "HTTPError",
    "error_code": 429,
    "time_to_classify": "0.2s"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-019**: Error recovery MUST classify failure type before prescribing fixes
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-002**: All skill executions MUST return a ToolResult; exceptions MUST be caught and reported
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **CON-WRK-007**: Error recovery MUST NOT execute destructive operations without confirmation
- **CON-WRK-009**: Operations modes run post-execution -- MUST NOT modify already-reviewed code

## Examples

### Example: Recovering from a decomposition error

**Input:**
```
task: "Recover from failed task T5: 'Implement entire authentication system'"
error: {type: "TokenLimitExceeded", message: "Output exceeded 32768 tokens", context: "Task too large for single execution"}
failed_task: {id: "T5", description: "Implement entire authentication system", assigned_agent: "WORKER", mode: "build"}
retry_count: 0
```

**Output:**
```
classification: "decomposition"
recovery_strategy: "redecompose"
root_cause: "Task 'Implement entire authentication system' is too large for a single build task. The output exceeded the token limit, indicating the task needs to be broken into smaller units."

recovery_action:
  action: "signal_foreman_replan"
  task_id: "T5"
  suggestion: "Decompose into: (1) auth models, (2) JWT token generation, (3) login endpoint, (4) middleware, (5) tests for each component"
  estimated_subtasks: 5
  retry_count: 0

escalation_needed: false
pattern_detected: false
```
