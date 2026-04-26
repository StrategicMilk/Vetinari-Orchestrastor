---
name: Contrarian Analysis
description: Devil's advocate review that surfaces failure modes, hidden assumptions, and worst-case scenarios
mode: contrarian_review
agent: worker
version: "1.0.0"
capabilities:
  - contrarian_analysis
  - risk_assessment
tags:
  - architecture
  - contrarian
  - failure-analysis
  - assumptions
---

# Contrarian Analysis

## Purpose

Contrarian Analysis provides a systematic devil's advocate review of proposed designs, plans, or implementations. While architecture review evaluates designs optimistically (does it meet requirements?), contrarian analysis asks pessimistically: how will this fail? What assumptions are wrong? What does the worst case look like? This skill counteracts the natural optimism bias in design and planning, catching failure modes that standard review misses because no one was actively trying to break the design.

## When to Use

- After an architecture review approves a design, before implementation begins
- When a plan seems "too clean" or when the team has high confidence that deserves stress-testing
- For high-stakes decisions where the cost of failure is severe (security, data integrity, production stability)
- When groupthink may be at play -- everyone agrees, but no one has argued the other side
- Before finalizing an ADR, to ensure consequences section is honest and complete
- When evaluating proposals from external sources or unfamiliar domains

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What to challenge and the adversarial objective                    |
| proposal        | dict            | No       | The design, plan, or ADR to challenge                              |
| assumptions     | list[string]    | No       | Stated assumptions to stress-test                                  |
| optimistic_view | string          | No       | The "everything goes right" scenario to counter                    |
| context         | dict            | No       | System context and constraints                                    |

## Process Steps

1. **Assumption extraction** -- Identify every assumption underlying the proposal, both explicit and implicit. For each assumption, classify it: structural (must be true for the approach to work), convenience (simplifies the approach but not required), or cosmetic (nice to have but irrelevant to success).

2. **Assumption inversion** -- For each structural assumption, ask "what if this is wrong?" and trace the consequences. If the assumption "users send well-formed JSON" is inverted, what breaks? If the assumption "database queries complete in <100ms" is wrong, what cascades?

3. **Failure mode enumeration** -- Systematically enumerate failure modes using the FMEA approach:
   - What can fail? (each component, interface, data flow)
   - How can it fail? (crash, hang, corrupt, slow, wrong answer)
   - What is the effect? (user impact, data impact, system impact)
   - How would we detect it? (monitoring, alerts, user reports)

4. **Adversarial scenario construction** -- Build concrete scenarios where the proposal fails badly. These are not random failures but targeted attacks on the weakest points. Include: malicious input, resource exhaustion, race conditions, cascading failures, and Byzantine behavior.

5. **Pre-mortem analysis** -- Imagine the project has failed spectacularly. Work backwards from the failure to construct a plausible narrative of how it happened. This reveals failure paths that forward-looking analysis misses.

6. **Second-order effect identification** -- Trace ripple effects beyond the immediate scope. A "simple" caching change might affect memory usage, which affects garbage collection pauses, which affects request latency, which triggers false alerts, which causes alert fatigue, which causes real alerts to be ignored.

7. **Hidden cost discovery** -- Identify costs the proposal does not account for: maintenance burden, on-call load, documentation debt, training costs, opportunity cost of not doing something else, and technical debt accrual.

8. **Counter-proposal sketching** -- For each significant weakness found, sketch an alternative approach that avoids that weakness. This is not a full design -- just enough to show that alternatives exist and what trade-offs they make differently.

9. **Challenge report assembly** -- Compile findings into a structured challenge report. Grade the proposal's robustness. Clearly distinguish between fatal flaws (must fix), significant concerns (should address), and nitpicks (could improve).

## Output Format

The skill produces a contrarian analysis report:

```json
{
  "success": true,
  "output": {
    "robustness_grade": "C",
    "summary": "Design handles the happy path well but has 2 significant blind spots in error handling and 1 fatal assumption about data consistency",
    "fatal_flaws": [
      {
        "assumption": "All tasks complete within timeout",
        "inversion": "A task hangs indefinitely, blocking downstream tasks and consuming budget",
        "evidence": "No timeout enforcement in current execution context",
        "fix_required": "Add hard timeout with forced termination and cleanup"
      }
    ],
    "significant_concerns": [
      {
        "failure_mode": "Cache inconsistency after failed write",
        "scenario": "Write to database succeeds, cache update fails, reads return stale data",
        "detection": "No monitoring for cache-database divergence",
        "mitigation": "Write-through caching or cache invalidation on any write failure"
      }
    ],
    "hidden_costs": [
      "Cache adds ~500 lines of code that need maintenance",
      "On-call team needs to understand cache invalidation logic"
    ],
    "pre_mortem": "Six months from now: cache grows unbounded because TTL was never enforced. OOM kill in production. Restart loses all cached data. Cold cache causes 10x spike in latency. Users report slowness. Team adds 'temporary' cache warming script that becomes permanent.",
    "counter_proposals": [
      "Use functools.lru_cache instead of custom cache (simpler, bounded, thread-safe)"
    ]
  },
  "metadata": {
    "proposal_reviewed": "ADR-0062 (in-memory prompt cache)",
    "challenge_level": "thorough"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-004**: Architecture modes are READ-ONLY -- MUST NOT modify production files
- **STD-WRK-005**: Every design MUST state the chosen pattern and its rationale
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-WRK-002**: Architecture modes are READ-ONLY -- produce ADRs, not code changes
- **GDL-WRK-003**: Always run contrarian_review after architecture for high-stakes decisions

## Examples

### Example: Challenging a "simple" feature addition

**Input:**
```
task: "Challenge the proposal to add in-memory session storage for the web UI"
proposal: {pattern: "dict[session_id, SessionData]", ttl: "30 minutes", max_sessions: 1000}
optimistic_view: "Simple dict with TTL cleanup thread, handles all our use cases"
```

**Output (abbreviated):**
```
robustness_grade: D

fatal_flaws:
  - "No thread safety on the dict. Flask runs threaded. Concurrent requests will corrupt the session dict."
  - "TTL cleanup thread holds GIL during cleanup, causing request latency spikes."

significant_concerns:
  - "Sessions lost on server restart -- user forced to re-authenticate mid-workflow"
  - "Max 1000 sessions is arbitrary -- no analysis of expected concurrent users"
  - "Session data in memory means it cannot be shared across multiple workers"

pre_mortem: "Week 3: user reports losing their work mid-session. Investigation reveals server restarted for a deployment. Team adds 'don't deploy during business hours' rule. This holds until an urgent security patch requires immediate deployment."

counter_proposals:
  - "Use Flask-Session with filesystem backend (survives restarts, no external dependency)"
  - "Use signed cookies for session data <4KB (stateless, no server storage)"
```
