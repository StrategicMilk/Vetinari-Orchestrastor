# Vetinari Escalation Rules

When to stop, when to ask, when to delegate, and when to halt the pipeline.

## When to Stop and Ask the User

Trigger `metadata.needs_user_input = true` when:

- **Ambiguous requirements**: The task description can be interpreted in 2+ mutually exclusive ways and no context narrows it down
- **Conflicting constraints**: Two requirements contradict each other (e.g., "make it faster" vs. "add comprehensive validation")
- **Scope exceeds estimate by 2x+**: The task will take significantly more work than initially scoped — confirm before proceeding
- **Missing credentials or config**: The task requires API keys, tokens, or configuration values not available in the environment
- **Destructive operations**: The task would delete data, drop tables, or perform irreversible changes without explicit authorization
- **External service down**: A required external service (local inference, database) is unreachable after 3 retry attempts

## When to Delegate to Another Agent

| If You Are | And You Need | Delegate To |
|------------|-------------|-------------|
| Builder | Context about existing code patterns | Researcher (code_discovery mode) |
| Builder | External API documentation | Researcher (api_lookup mode) |
| Builder | Architecture decision on approach | Oracle (architecture mode) |
| Quality | Code changes to fix issues | Builder (build mode) via rework |
| Planner | Risk analysis before committing | Oracle (risk_assessment mode) |
| Planner | Existing ADR check | Oracle (ontological_analysis mode) |
| Any agent | Cost estimate for proposed approach | Operations (cost_analysis mode) |
| Any agent | Documentation for completed work | Operations (documentation mode) |
| Any agent | Status monitoring | Operations (monitor mode) |

### Delegation Protocol
1. Set `metadata.delegate_to` with target agent type and mode
2. Include specific questions — not open-ended exploration
3. Include what you already know to avoid redundant work
4. Maximum 3 levels of delegation depth — if deeper needed, route through Planner

## When to Halt the Pipeline (Andon Trigger)

These conditions trigger an immediate pipeline halt:

- **Security violation detected**: Credentials exposed, injection vulnerability in generated code, unauthorized file access attempt
- **Repeated failures**: Same task fails 3+ times (max rework cycles exceeded) — indicates a fundamental problem, not a transient error
- **Data integrity at risk**: Operation could corrupt the memory store, lose episode data, or produce inconsistent state
- **Model unavailable**: All models in the fallback chain are unreachable — no inference possible
- **Resource exhaustion**: Token budget, cost budget, or timeout exceeded with no viable fallback

### Andon Protocol
1. Log the trigger reason with severity CRITICAL
2. Record affected task IDs and agent states
3. Set pipeline status to HALTED
4. Wait for human resolution — do not auto-resume
5. Resume only after the trigger condition is confirmed resolved

## Rework Protocol

When Quality rejects output:
1. Quality provides specific, actionable feedback with file paths and line numbers
2. Builder receives the rejection with the original task + Quality feedback
3. Builder fixes ONLY the issues identified — no scope expansion
4. Maximum 3 rework cycles per task before Andon trigger
5. If the same issue recurs across 2+ rework cycles, escalate to Oracle for architectural review

## Confidence-Based Routing

| Agent Confidence | Action |
|-----------------|--------|
| HIGH (>80%) | Proceed to next pipeline stage |
| MEDIUM (50-80%) | Proceed with explicit assumptions documented |
| LOW (<50%) | Escalate: request research, human review, or additional context |
