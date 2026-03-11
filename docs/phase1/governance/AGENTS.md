# Vetinari Agent System — Governance Reference

> **Canonical source**: See [`/AGENTS.md`](../../../AGENTS.md) at the repository root for the
> full agent specification (694 lines, 11 sections).  This file is a summary
> for quick reference within the docs tree.

**Architecture**: 6-agent consolidated (ADR-001) — 33 total modes
**Last Updated**: 2026-03-10 | **Version**: 2.0

---

## Cognitive Pipeline

```
Plan → Research → Advise → Build → Verify → Operate
  ↓        ↓         ↓       ↓        ↓        ↓
Planner  Researcher  Oracle  Builder  Quality  Operations
```

---

## Agent Roster

| # | Agent | Class | Modes | Role |
|---|-------|-------|-------|------|
| 1 | **Planner** | `PlannerAgent` | plan, clarify, summarise, prune, extract, consolidate | Orchestration, task decomposition, context management |
| 2 | **Researcher** | `ConsolidatedResearcherAgent` | code_discovery, domain_research, api_lookup, lateral_thinking, ui_design, database, devops, git_workflow | Evidence gathering across 8 domains |
| 3 | **Oracle** | `ConsolidatedOracleAgent` | architecture, risk_assessment, ontological_analysis, contrarian_review | Strategic decisions with explicit reasoning |
| 4 | **Builder** | `BuilderAgent` | build, image_generation | **Sole code writer** — production source files |
| 5 | **Quality** | `QualityAgent` | code_review, security_audit, test_generation, simplification | Mandatory review gate on all Builder output |
| 6 | **Operations** | `ConsolidatedOperationsAgent` | documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, improvement, monitor, devops_ops | Documentation, synthesis, system health |

---

## Key Rules

1. **Builder is the only agent that writes production source files.**
2. **Quality must review all Builder output** — mandatory pass/fail gate.
3. **Maximum delegation depth**: 3 levels (Planner → Agent → Sub-task).
4. **All agents route through Planner** for task assignment.
5. **Oracle consulted for architectural decisions** before Builder implements.

---

## File Jurisdiction (Summary)

| Directory | Primary Owner |
|-----------|--------------|
| `vetinari/orchestration/`, `vetinari/planning/`, `vetinari/adapters/`, `vetinari/config/` | **Planner** |
| `vetinari/skills/`, `vetinari/tools/`, `vetinari/rag/`, `vetinari/web/`, `vetinari/dashboard/` | **Researcher** |
| `vetinari/constraints/`, `vetinari/drift/`, `vetinari/safety/` | **Oracle** |
| `vetinari/coding_agent/`, `vetinari/mcp/`, `vetinari/sandbox.py` | **Builder** |
| `tests/` | **Quality** |
| `vetinari/analytics/`, `vetinari/learning/`, `docs/` | **Operations** |
| `vetinari/exceptions.py`, `vetinari/types.py`, `vetinari/constants.py` | **Shared** |

> See `/AGENTS.md` § File Jurisdiction Map for the complete directory listing.

---

## Quality Gates

| Agent | Gate Metric | Threshold |
|-------|------------|-----------|
| Planner | plan_coherence | ≥ 0.7 |
| Researcher | evidence_quality | ≥ 0.6 |
| Oracle | reasoning_depth | ≥ 0.7 |
| Builder | code_correctness | ≥ 0.8 |
| Quality | review_coverage | ≥ 0.9 |
| Operations | doc_completeness | ≥ 0.7 |

---

## Resource Constraints

| Agent | Token Budget | Timeout | Max Retries |
|-------|-------------|---------|-------------|
| Planner | 4,096 | 60s | 2 |
| Researcher | 8,192 | 120s | 3 |
| Oracle | 4,096 | 90s | 2 |
| Builder | 16,384 | 300s | 3 |
| Quality | 8,192 | 120s | 2 |
| Operations | 8,192 | 120s | 2 |

---

## Legacy Deprecation

22 legacy agents have been consolidated into the 6 target agents:

| Legacy Agent | Replaced By |
|-------------|-------------|
| Explorer, Librarian, Architect, UIPlanner, DataEngineer, DevOps, VersionControl | **Researcher** |
| Evaluator, SecurityAuditor, TestAutomation | **Quality** |
| Synthesizer, DocumentationAgent, CostPlanner, ExperimentationManager, ImprovementAgent, ErrorRecovery | **Operations** |
| ImageGenerator | **Builder** |
| UserInteraction, ContextManager | **Planner** |
| Ponder | **Oracle** (renamed) |

---

## Workflow Pipelines

| Pipeline | Stages |
|----------|--------|
| **default** | Planner → Researcher → Builder → Quality → Operations |
| **code_review** | Researcher → Quality |
| **security_audit** | Researcher → Quality (security_audit mode) → Operations |
| **research_only** | Planner → Researcher → Operations |
| **build_and_test** | Planner → Builder → Quality |
| **full_stack** | Planner → Researcher → Oracle → Builder → Quality → Operations |
| **documentation** | Researcher → Operations |
| **incident_response** | Researcher → Oracle → Builder → Quality |
| **optimization** | Researcher → Oracle → Builder → Quality → Operations |

---

## Further Reading

- **Full specification**: [`/AGENTS.md`](../../../AGENTS.md) (694 lines, 11 sections)
- **Agent prompt files**: [`.claude/agents/`](../../../.claude/agents/) (6 files, 266-309 lines each)
- **Architecture overview**: [`CLAUDE.md`](../../../CLAUDE.md) § Agent System Summary
- **Agent contracts**: `vetinari/agents/contracts.py` (AGENT_REGISTRY)
- **Agent interfaces**: `vetinari/agents/interfaces.py` (AgentInterface ABC)
