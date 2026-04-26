---
name: Requirement Specification
description: Convert ambiguous user requests into precise Given-When-Then acceptance criteria using Socratic questioning
mode: plan
agent: foreman
version: "1.0.0"
capabilities:
  - specification
  - clarification
tags:
  - planning
  - requirements
  - acceptance-criteria
  - specification
---

# Requirement Specification

## Purpose

Requirement Specification transforms vague or incomplete user requests into precise, testable acceptance criteria using the Given-When-Then format. It applies Socratic questioning to surface hidden assumptions, implicit requirements, and edge cases that the user may not have considered. This skill prevents the most expensive class of defect -- building the wrong thing -- by ensuring shared understanding before any work begins.

## When to Use

- A user request is ambiguous, underspecified, or could be interpreted multiple ways
- The goal mentions outcomes but not constraints (e.g., "make it faster" without defining target metrics)
- The request involves user-facing behavior where acceptance criteria must be objectively verifiable
- Prior decomposition attempts revealed gaps in understanding
- The team needs a shared definition of "done" before committing resources
- Integration with external systems where interface contracts must be explicit

## Inputs

| Parameter       | Type            | Required | Description                                                       |
|-----------------|-----------------|----------|-------------------------------------------------------------------|
| goal            | string          | Yes      | The raw user request or goal statement                            |
| context         | dict            | No       | Existing codebase context, prior conversations, domain knowledge  |
| constraints     | list[string]    | No       | Known constraints (budget, timeline, technology stack)             |
| stakeholders    | list[string]    | No       | Who cares about this feature (roles, not names)                   |
| prior_questions | list[dict]      | No       | Previously asked clarification questions and their answers        |

## Process Steps

1. **Intent extraction** -- Parse the raw goal to identify the core user intent. Distinguish between what the user asked for (surface request) and what they actually need (underlying problem). Flag any divergence for clarification.

2. **Assumption surfacing** -- List every assumption embedded in the request. For each assumption, classify it as: confirmed (explicitly stated), implicit (reasonable default), or dangerous (could cause misalignment). Dangerous assumptions become clarification questions.

3. **Socratic questioning** -- Generate targeted questions that resolve ambiguity without overwhelming the user. Questions must be specific and answerable (not open-ended). Prioritize questions by impact -- ask the ones that could invalidate the entire approach first.

4. **Scope boundary definition** -- Define what is IN scope and what is explicitly OUT of scope. This prevents scope creep during execution and sets clear expectations.

5. **Persona and scenario identification** -- Identify the actors (users, systems, agents) who interact with the feature. For each actor, identify their primary scenarios (happy path) and alternate scenarios (error, edge case).

6. **Given-When-Then formulation** -- For each scenario, write acceptance criteria in Given-When-Then format. "Given" establishes preconditions, "When" describes the action, "Then" describes the observable outcome. Each criterion must be objectively testable.

7. **Edge case enumeration** -- Systematically consider boundary conditions: empty inputs, maximum inputs, concurrent access, partial failures, invalid data, timeout scenarios. Add acceptance criteria for each relevant edge case.

8. **Non-functional requirement capture** -- Identify performance targets (latency, throughput), security requirements (auth, input validation), reliability requirements (error handling, recovery), and operational requirements (logging, monitoring).

9. **Dependency mapping** -- List external dependencies the feature requires and any features that depend on this one. This feeds into the Foreman's dependency analysis during task decomposition.

10. **Specification assembly** -- Compile the final specification with: goal summary, scope boundaries, acceptance criteria, edge cases, non-functional requirements, open questions, and assumptions.

## Output Format

The skill produces a structured specification that becomes input to task decomposition:

```yaml
specification:
  goal: "Rate limit the REST API to prevent abuse"
  scope:
    in:
      - Per-endpoint rate limiting with configurable thresholds
      - HTTP 429 response with Retry-After header
      - Rate limit state stored in-memory (not distributed)
    out:
      - Distributed rate limiting across multiple instances
      - Per-user rate limiting (requires auth integration)
      - Rate limit dashboard or admin UI

  acceptance_criteria:
    - given: "A client has made fewer than N requests in the window"
      when: "The client sends a valid API request"
      then: "The request is processed normally with X-RateLimit-Remaining header"

    - given: "A client has reached the rate limit"
      when: "The client sends another request"
      then: "HTTP 429 is returned with Retry-After header indicating seconds until reset"

  edge_cases:
    - "Concurrent requests arriving simultaneously at the limit boundary"
    - "Server restart clears in-memory rate limit state"

  non_functional:
    performance: "Rate limit check adds <1ms latency per request"
    reliability: "Rate limiter failure must not block requests (fail-open)"

  open_questions:
    - "Should rate limits be per-IP, per-API-key, or per-endpoint?"
    - "What are the default rate limit thresholds?"

  assumptions:
    confirmed:
      - "API is served by a single Flask instance"
    implicit:
      - "In-memory storage is acceptable (no persistence across restarts)"
    dangerous: []
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-FMN-005**: Clarification mode MUST produce specific, answerable questions -- not open-ended prompts
- **STD-FMN-001**: Output specifications must be structured with unique identifiers for traceability
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-004**: Skill executions SHOULD be idempotent -- running twice with same input produces same output
- **CON-FMN-004**: Foreman MUST NOT execute tasks directly -- only plan, clarify, and delegate
- **GDL-FMN-004**: Use clarify mode before plan when requirements have >2 ambiguous elements

## Examples

### Example: Underspecified performance request

**Input:**
```
goal: "Make the agent pipeline faster"
context: {current_latency: "~4s per task", pipeline: "Foreman -> Worker -> Inspector"}
```

**Socratic Questions Generated:**
```
1. What is your target latency? (Current: ~4s per task)
2. Which phase feels slowest -- planning, execution, or review?
3. Is this about wall-clock time (user waiting) or throughput (tasks per hour)?
4. Are you willing to trade quality (fewer review passes) for speed?
5. Should this optimize for simple tasks, complex tasks, or both?
```

**Acceptance Criteria (after answers received):**
```
- Given: A single-file bug fix task
  When: Submitted to the pipeline
  Then: Total wall-clock time is under 2 seconds

- Given: A multi-file feature task with 5+ subtasks
  When: Submitted to the pipeline
  Then: Parallelizable subtasks execute concurrently, reducing wall-clock by 40%+

- Given: Pipeline optimization is applied
  When: Inspector reviews code quality
  Then: Quality gate pass rate does not decrease compared to baseline
```
