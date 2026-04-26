---
name: Architecture Review
description: Evaluate designs against established patterns and produce ADRs documenting decisions (read-only analysis, not code)
mode: architecture
agent: worker
version: "1.0.0"
capabilities:
  - architecture_review
  - api_contract_validation
tags:
  - architecture
  - design
  - adr
  - patterns
---

# Architecture Review

## Purpose

Architecture Review evaluates proposed or existing designs against established architectural patterns, project conventions, and quality attributes (scalability, maintainability, security). It produces Architecture Decision Records (ADRs) that document what was decided, why, and what alternatives were considered. This skill is strictly read-only -- it analyzes and recommends but never modifies production code. Its output feeds into the Foreman's planning and the Worker's build skills.

## When to Use

- Before implementing a new feature that introduces a new pattern or module
- When evaluating a proposed design against existing codebase conventions
- When a design decision has trade-offs that need formal documentation
- When adding a new dependency, service, or integration point
- When an existing architecture shows signs of degradation (growing complexity, coupling)
- When the Foreman's plan includes architectural changes that need validation

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What design to review and the review objective                     |
| design          | dict            | No       | Proposed design document (components, interfaces, data flow)       |
| files           | list[string]    | No       | Existing files to review for architectural compliance              |
| patterns        | list[string]    | No       | Specific patterns to evaluate against                              |
| quality_attrs   | list[string]    | No       | Quality attributes to prioritize (e.g., "maintainability")        |
| context         | dict            | No       | Project context, existing ADRs, constraints                       |

## Process Steps

1. **Design comprehension** -- Read and internalize the proposed design or existing architecture. Map components, their responsibilities, interfaces, and data flows. Build a mental model of how the system works.

2. **Pattern identification** -- Identify which architectural patterns are in use or proposed: layered architecture, hexagonal/ports-and-adapters, event-driven, pipeline, repository pattern, etc. Note whether the patterns are consistently applied.

3. **Convention compliance** -- Check the design against Vetinari's established conventions:
   - Single-writer rule (only Builder/Worker-build modifies production code)
   - Agent pipeline flow (Foreman -> Worker -> Inspector)
   - Canonical import sources (enums from types.py, specs from contracts.py)
   - Module organization rules (no more than 3 levels deep)
   - Composition over inheritance

4. **Quality attribute analysis** -- Evaluate the design against key quality attributes:
   - **Maintainability**: Can a new contributor understand and modify this?
   - **Testability**: Can each component be tested in isolation?
   - **Extensibility**: Can new capabilities be added without modifying existing code?
   - **Performance**: Are there obvious bottlenecks or scalability concerns?
   - **Security**: Are trust boundaries clearly defined?

5. **Alternative evaluation** -- For each significant design decision, identify at least 2 alternative approaches. For high-stakes decisions (architecture, security, data_flow categories), evaluate at least 3 alternatives. Document trade-offs for each.

6. **Coupling and cohesion assessment** -- Measure coupling (how much components depend on each other) and cohesion (how focused each component is). Flag tight coupling between components that should be independent and low cohesion in components that mix responsibilities.

7. **ADR drafting** -- For each significant decision, draft an Architecture Decision Record with: context (problem and constraints), decision (chosen approach), consequences (positive and negative trade-offs), and alternatives (what else was considered).

8. **Risk identification** -- Identify architectural risks: single points of failure, scalability ceilings, technology lock-in, knowledge concentration (bus factor), and security vulnerabilities at the architectural level.

9. **Recommendation synthesis** -- Compile findings into a structured review with: approval/rejection recommendation, required changes before approval, and suggested improvements.

## Output Format

The skill produces an architecture review with ADR drafts:

```json
{
  "success": true,
  "output": {
    "verdict": "approve_with_changes",
    "summary": "Design is sound but has two coupling issues that should be addressed before implementation",
    "pattern_compliance": {
      "pipeline_pattern": "compliant",
      "single_writer": "compliant",
      "canonical_imports": "non-compliant -- design uses local enum definition"
    },
    "quality_assessment": {
      "maintainability": "B",
      "testability": "A",
      "extensibility": "B",
      "performance": "A",
      "security": "C -- trust boundary not defined for external input"
    },
    "required_changes": [
      "Move enum definition to vetinari/types.py (canonical source)",
      "Define trust boundary for user-provided query input"
    ],
    "adr_drafts": [
      {
        "title": "Use token bucket algorithm for rate limiting",
        "context": "Need to limit API request rate. Options: fixed window, sliding window, token bucket, leaky bucket.",
        "decision": "Token bucket -- provides burst tolerance while maintaining average rate limit",
        "consequences": {
          "positive": ["Allows burst traffic", "Simple implementation", "Memory efficient"],
          "negative": ["Slightly more complex than fixed window", "Burst size needs tuning"]
        },
        "alternatives": ["Fixed window (simpler but allows 2x burst at boundary)", "Sliding window (precise but memory intensive)"]
      }
    ]
  },
  "metadata": {
    "adr_id": "ADR-0062",
    "adr_reference": "ADR-0061 (factory pipeline architecture)"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-003**: Architecture mode MUST produce or reference an ADR for every design decision
- **STD-WRK-004**: Architecture modes are READ-ONLY -- MUST NOT modify production files
- **STD-WRK-005**: Every design MUST state the chosen pattern and its rationale
- **STD-WRK-006**: Component designs MUST define clear boundaries (inputs, outputs, dependencies)
- **STD-WRK-010**: Designs MUST list alternatives considered with trade-off analysis
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-WRK-002**: Architecture modes are READ-ONLY -- produce ADRs, not code changes
- **CON-WRK-003**: High-stakes categories require 3+ alternatives evaluated

## Examples

### Example: Reviewing a proposed caching layer

**Input:**
```
task: "Review the proposed in-memory caching layer for the prompt assembler"
design: {component: "PromptCache", location: "vetinari/prompts/cache.py", pattern: "LRU cache with TTL"}
quality_attrs: ["performance", "maintainability"]
```

**Output (abbreviated):**
```
verdict: "approve_with_changes"

pattern_compliance:
  - "LRU + TTL is appropriate for prompt assembly (temporal locality + invalidation)"
  - "Location in vetinari/prompts/ is correct (same module as consumer)"

required_changes:
  - "Add thread safety (Flask runs threaded) -- use threading.Lock or functools.lru_cache"
  - "Make cache size configurable via config YAML, not hardcoded"

adr_drafts:
  - title: "Use in-memory LRU cache for prompt assembly"
    context: "Prompt assembly reconstructs the same prompt for repeated similar tasks. Options: no cache, LRU, Redis, file-based."
    decision: "In-memory LRU with TTL -- simple, no external dependency, sufficient for single-instance deployment"
    consequences:
      positive: ["~10x speedup for cache hits", "No new dependencies"]
      negative: ["Cache lost on restart", "Memory overhead for large prompts", "Not shared across instances"]
```
