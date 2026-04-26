---
name: Lateral Exploration
description: Cross-pollinate solutions from other domains when standard approaches fail, applying analogies from biology, physics, and manufacturing
mode: lateral_thinking
agent: worker
version: "1.0.0"
capabilities:
  - lateral_thinking
  - domain_research
tags:
  - research
  - creativity
  - analogies
  - problem-solving
---

# Lateral Exploration

## Purpose

Lateral Exploration applies cross-domain analogies to technical problems when conventional approaches are insufficient or when a novel perspective could yield a significantly better solution. It draws from domains like biology (immune systems, evolution), physics (feedback loops, entropy), manufacturing (lean, six sigma, SPC), and organizational theory (Conway's Law, queueing theory) to surface approaches invisible to domain-locked thinking. This is Vetinari's creative problem-solving skill -- used sparingly but powerfully when standard patterns do not fit.

## When to Use

- A conventional approach has been tried and failed or produced unsatisfactory results
- The problem has no clear precedent in the codebase or standard software patterns
- The team is stuck in a local optimum and needs a fundamentally different perspective
- The problem involves emergent behavior, complex interactions, or non-linear dynamics
- Design decisions need fresh framing to avoid groupthink or anchoring bias
- The Foreman has flagged a task as requiring "creative" or "novel" approaches

## Inputs

| Parameter       | Type            | Required | Description                                                       |
|-----------------|-----------------|----------|-------------------------------------------------------------------|
| task            | string          | Yes      | The problem to explore laterally                                  |
| failed_attempts | list[string]    | No       | Previous approaches that did not work and why                     |
| domain_hints    | list[string]    | No       | Suggested domains to draw from (e.g., "biology", "manufacturing") |
| constraints     | list[string]    | No       | Hard constraints that any solution must satisfy                   |
| context         | dict            | No       | Technical context about the system and problem                    |

## Process Steps

1. **Problem abstraction** -- Strip the problem of domain-specific language to reveal its abstract structure. A "rate limiting" problem becomes a "flow control" problem. A "caching" problem becomes a "temporal locality exploitation" problem. This abstraction enables cross-domain matching.

2. **Failed approach analysis** -- If previous approaches exist, identify why they failed at the abstract level. Was it a scaling issue? A coordination issue? A feedback delay issue? The abstract failure mode points toward domains that have solved similar abstract problems.

3. **Domain scanning** -- Survey candidate domains for analogous problems:
   - **Biology**: immune systems (anomaly detection), evolution (optimization), homeostasis (self-regulation), neural networks (pattern recognition)
   - **Physics**: feedback loops (control systems), entropy (information theory), phase transitions (tipping points), wave interference (constructive/destructive)
   - **Manufacturing**: lean (waste reduction), SPC (statistical process control), kanban (flow management), poka-yoke (error-proofing)
   - **Economics**: market mechanisms (resource allocation), game theory (multi-agent coordination), auction theory (priority allocation)

4. **Analogy mapping** -- For the most promising domain analogies, map the domain concepts to the technical problem. Create a correspondence table: domain entity to code entity, domain process to code process, domain metric to code metric.

5. **Solution synthesis** -- Using the analogy mapping, propose concrete technical solutions inspired by the cross-domain approach. Each proposal must include: the analogy used, the concrete implementation approach, and why this is better than the failed conventional approaches.

6. **Feasibility filtering** -- Filter proposals through the hard constraints. Discard any that violate constraints. For remaining proposals, assess implementation effort, risk, and expected benefit.

7. **Validation strategy** -- For each viable proposal, describe how to validate it: what experiment to run, what metrics to measure, what success looks like, and what would falsify the approach.

8. **Recommendation ranking** -- Rank proposals by expected value (benefit times probability of success divided by effort). Present the top 2-3 with full rationale.

## Output Format

The skill produces a lateral analysis report with ranked proposals:

```json
{
  "success": true,
  "output": {
    "problem_abstract": "Multi-agent coordination with heterogeneous task types and variable completion times",
    "failed_approaches": [
      {"approach": "Round-robin assignment", "failure": "Does not account for task affinity or agent capability"}
    ],
    "proposals": [
      {
        "rank": 1,
        "title": "Stigmergy-based task selection (ant colony analogy)",
        "domain": "biology/entomology",
        "analogy": "Ants select tasks based on pheromone gradients rather than central assignment. Tasks emit 'urgency pheromones' that decay over time. Agents select the highest-pheromone task they are capable of.",
        "implementation": "Add a priority_score field to tasks that increases with wait time and decreases when an agent picks it up. Agents select tasks by max(priority_score * capability_match).",
        "benefits": ["Self-balancing load distribution", "Graceful degradation when agents fail", "No central coordinator bottleneck"],
        "risks": ["Priority inversion possible", "Requires tuning decay rate"],
        "effort": "M",
        "validation": "Simulate with 100 tasks and 3 agents, measure completion time vs round-robin baseline"
      }
    ]
  },
  "metadata": {
    "domains_consulted": ["biology", "manufacturing", "economics"],
    "analogies_evaluated": 5,
    "proposals_generated": 3,
    "proposals_after_filtering": 2
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-001**: Research modes MUST cite sources -- file paths, URLs, or commit SHAs (cite the domain source)
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-004**: Skill executions SHOULD be idempotent -- running twice with same input produces same output
- **CON-WRK-001**: Research modes are READ-ONLY -- MUST NOT modify production files
- **GDL-WRK-002**: Use lateral_thinking for novel problems with no clear precedent

## Examples

### Example: Solving agent stagnation with biological analogy

**Input:**
```
task: "Agent tasks sometimes get stuck in retry loops with no progress. Need a detection and recovery mechanism."
failed_attempts: ["Fixed retry count (too rigid)", "Timeout-based (kills slow but progressing tasks)"]
domain_hints: ["biology", "manufacturing"]
```

**Output (abbreviated):**
```
problem_abstract: "Detecting stagnation vs slow progress in an autonomous process"

proposals:
  1. "Heartbeat with progress gradient (biology: vital signs monitoring)"
     analogy: "Medical monitors don't just check alive/dead -- they track vital sign TRENDS. A declining heart rate is alarming even if still in normal range."
     implementation: "Track a progress metric per task (e.g., files_touched, tests_passed, tokens_generated). Compute the derivative. If progress derivative drops below threshold for N intervals, classify as stagnating."
     effort: S
     validation: "Replay 50 historical task executions, verify stagnation detector catches known-stuck tasks without false-alarming on slow-but-progressing tasks."

  2. "Andon cord with escalation (manufacturing: Toyota Production System)"
     analogy: "Any worker on the Toyota line can pull the Andon cord to stop the line when they detect a quality problem. The line does not restart until the problem is resolved by the right expert."
     implementation: "Allow agents to self-report stagnation with a 'pull_andon' signal. The Foreman receives the signal and dispatches a diagnostic task before deciding to retry, replan, or escalate to human."
     effort: M
```
