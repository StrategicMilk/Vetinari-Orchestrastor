# Vetinari Agent System

This document is the authoritative reference for Vetinari's multi-agent architecture. All agents, workflows, file ownership, collaboration protocols, and deprecation mappings are defined here.

**Version**: 3.x (Phase 3 — Consolidated Architecture)
**Last Updated**: 2026-03-10

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Agent Roster](#2-agent-roster)
3. [File Jurisdiction Map](#3-file-jurisdiction-map)
4. [Three-Role Pattern](#4-three-role-pattern)
5. [Delegation Rules](#5-delegation-rules)
6. [Quality Gates](#6-quality-gates)
7. [Resource Constraints](#7-resource-constraints)
8. [Collaboration Matrix](#8-collaboration-matrix)
9. [Legacy Deprecation](#9-legacy-deprecation)
10. [Workflow Pipelines](#10-workflow-pipelines)
11. [System Prompt Standards](#11-system-prompt-standards)

---

## 1. Architecture Overview

Vetinari uses a **six-agent consolidated architecture** built around a cognitive pipeline that mirrors human problem-solving:

```
Plan → Research → Advise → Build → Verify → Operate
  ↓        ↓         ↓       ↓        ↓        ↓
Planner  Researcher  Oracle  Builder  Quality  Operations
```

Each stage in the pipeline has a defined cognitive role:

| Stage | Agent | Cognitive Role | Output Type |
|---|---|---|---|
| Plan | Planner | Decompose and sequence | Task graph (DAG) |
| Research | Researcher | Gather evidence | Structured findings |
| Advise | Oracle | Deliberate and decide | Architecture decisions |
| Build | Builder | Implement | Source code |
| Verify | Quality | Judge and gate | Pass/fail report |
| Operate | Operations | Synthesise and sustain | Artefacts and docs |

### 33 Total Modes

The six agents collectively expose **33 modes** across the pipeline:

- **Planner**: plan, clarify, summarise, prune, extract, consolidate (6 modes)
- **Researcher**: code_discovery, domain_research, api_lookup, lateral_thinking, ui_design, database, devops, git_workflow (8 modes)
- **Oracle**: architecture, risk_assessment, ontological_analysis, contrarian_review (4 modes)
- **Builder**: build, image_generation (2 modes)
- **Quality**: code_review, security_audit, test_generation, simplification (4 modes)
- **Operations**: documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, image_generation, improvement, monitoring (9 modes)

### Key Architectural Principles

1. **Single writer principle**: Only Builder writes production source files. All other agents read.
2. **Gate authority**: Quality's gate decisions cannot be overridden by other agents. Only a human can bypass a Quality gate.
3. **Planner primacy**: Planner owns the task graph. Agents do not self-assign work or spawn sub-tasks without Planner approval.
4. **Canonical type source**: All enums and shared types are imported from `vetinari/types.py`. No agent defines its own copies.
5. **Max delegation depth**: 3. Planner → Worker → Sub-agent. No deeper chaining.

---

## 2. Agent Roster

| Agent | Class | Modes | Model | Thinking | Status | Source File |
|---|---|---|---|---|---|---|
| **Planner** | `PlannerAgent` | 6 | qwen2.5-72b | high | Active | `vetinari/agents/planner_agent.py` |
| **Researcher** | `ConsolidatedResearcherAgent` | 8 | qwen2.5-72b | medium | Active | `vetinari/agents/consolidated/researcher_agent.py` |
| **Oracle** | `ConsolidatedOracleAgent` | 4 | qwen2.5-72b | high | Active | `vetinari/agents/consolidated/oracle_agent.py` |
| **Builder** | `BuilderAgent` | 2 | qwen2.5-72b | medium | Active | `vetinari/agents/builder_agent.py` |
| **Quality** | `QualityAgent` | 4 | qwen2.5-72b | medium | Active | `vetinari/agents/consolidated/quality_agent.py` |
| **Operations** | `ConsolidatedOperationsAgent` | 9 | qwen2.5-72b | low | Active | `vetinari/agents/consolidated/operations_agent.py` |

### AgentType Enum Values (from `vetinari/types.py`)

```python
# Active consolidated agents
AgentType.ORCHESTRATOR             # Planner
AgentType.CONSOLIDATED_RESEARCHER  # Researcher
AgentType.CONSOLIDATED_ORACLE      # Oracle
AgentType.ARCHITECT                # (sub-mode of Oracle)
AgentType.QUALITY                  # Quality
AgentType.OPERATIONS               # Operations
```

### Thinking Depth by Mode

| Agent | Mode | Thinking Depth | Rationale |
|---|---|---|---|
| Planner | plan | high | Must reason through full dependency graph |
| Planner | clarify/summarise/prune | low | Speed-prioritised tasks |
| Researcher | code_discovery/api_lookup | low | Lookup tasks |
| Researcher | domain_research/database/devops | medium | Analysis tasks |
| Oracle | all modes | high | Oracle always deliberates fully |
| Builder | build (simple) | medium | Routine implementation |
| Builder | build (security/algorithm) | high | High-stakes code |
| Quality | code_review | medium | Structured pattern matching |
| Quality | security_audit | high | Must reason about threat scenarios |
| Operations | documentation/synthesis | low | Writing speed prioritised |
| Operations | cost_analysis/improvement | high | Quantitative reasoning needed |

---

## 3. File Jurisdiction Map

The following table maps every major directory to its owning agent. "Owner" means primary write authority. "Reader" means read access only. Multiple owners requires coordination.

| Path | Owner | Readers | Notes |
|---|---|---|---|
| `vetinari/agents/base_agent.py` | Planner | All | Base class definition |
| `vetinari/agents/contracts.py` | Planner | All | AgentSpec, Task, Plan dataclasses |
| `vetinari/agents/interfaces.py` | Planner | All | AgentInterface ABC |
| `vetinari/agents/consolidated/researcher_agent.py` | Researcher | Planner, Quality | Mode implementation |
| `vetinari/agents/consolidated/oracle_agent.py` | Oracle | Planner | Mode implementation |
| `vetinari/agents/consolidated/quality_agent.py` | Quality | Planner | Mode implementation |
| `vetinari/agents/consolidated/operations_agent.py` | Operations | Planner | Mode implementation |
| `vetinari/agents/builder_agent.py` | Builder | Planner, Quality | Mode implementation |
| `vetinari/agents/coding_bridge.py` | Builder | Planner | Coding sub-agent bridge |
| `vetinari/coding_agent/` | Builder | Quality | Coding execution harness |
| `vetinari/mcp/` | Builder | All | MCP tool integration |
| `vetinari/sandbox.py` | Builder | Quality | Sandbox execution |
| `vetinari/orchestration/` | Planner | All | Orchestration logic |
| `vetinari/planning/` | Planner | All | Planning engine |
| `vetinari/adapters/` | Planner | All | Model adapter registry |
| `vetinari/config/` | Planner | All | System configuration |
| `vetinari/memory/` | Planner | All | Shared memory store |
| `vetinari/two_layer_orchestration.py` | Planner+Operations | All | Co-owned; coordinate changes |
| `vetinari/types.py` | Planner | All | Read-only for non-Planner agents |
| `vetinari/constraints/` | Oracle | Planner, Quality | Constraint definitions |
| `vetinari/drift/` | Oracle | Planner | Drift detection |
| `vetinari/safety/` | Oracle | Quality, Planner | Safety policies |
| `vetinari/skills/` | Researcher | All | Skill definitions |
| `vetinari/tools/` | Researcher | All | Tool wrappers |
| `vetinari/rag/` | Researcher | Builder | RAG index and client |
| `vetinari/analytics/` | Operations | Planner | Analytics data |
| `vetinari/learning/` | Operations | Planner | Learning records |
| `vetinari/training/` | Operations | Planner | Training data |
| `vetinari/benchmarks/` | Operations | Planner | Benchmark definitions |
| `vetinari/adr.py` | Oracle | All | Architecture decision records |
| `tests/` | Quality | Builder | Test files |
| `docs/` | Operations | All | Documentation |
| `config/` | Planner | All | Top-level YAML configs |
| `config/guardrails/` | Oracle | Planner | Guardrail policies |
| `ui/` | Researcher (design) + Builder (impl) | Quality, Operations | UI components |
| `vetinari/web_ui.py` | Builder | Quality, Operations | Flask web server |
| `vetinari/migrations/` | Researcher (research) + Builder (write) | Operations | DB migrations |
| `skills/` | Researcher | All | Agent skill prompt files |

---

## 4. Three-Role Pattern

Vetinari's six agents map cleanly onto the **Planner-Worker-Judge** three-role pattern:

```
┌─────────────────────────────────────────────────────────┐
│                       PLANNER                           │
│  Orchestrates, decomposes, sequences, routes            │
│  Agent: Planner                                         │
└────────────────────┬────────────────────────────────────┘
                     │ delegates tasks
         ┌───────────┼───────────────┐
         ▼           ▼               ▼
┌──────────────┐ ┌────────┐ ┌───────────────┐
│  WORKERS     │ │        │ │               │
│              │ │        │ │               │
│  Researcher  │ │Builder │ │  Operations   │
│  (discover)  │ │(build) │ │  (operate)    │
│              │ │        │ │               │
│  Oracle      │ │        │ │               │
│  (advise)    │ │        │ │               │
└──────────────┘ └────────┘ └───────────────┘
         │           │               │
         └───────────┴───────────────┘
                     │ submits for review
                     ▼
┌─────────────────────────────────────────────────────────┐
│                       JUDGE                             │
│  Reviews, gates, blocks, approves                       │
│  Agent: Quality                                         │
└─────────────────────────────────────────────────────────┘
```

### Role Responsibilities

**Planner (Orchestrator)**:
- Receives user goals and translates them into structured task graphs.
- Assigns each task to the appropriate worker agent and mode.
- Monitors execution progress; replans on failure.
- Manages shared context and memory lifecycle.
- Does not implement code or judge quality.

**Workers (Specialist Executors)**:
- **Researcher**: Evidence gathering. Answers "what exists?" and "what should we use?"
- **Oracle**: Strategic advisor. Answers "what should we decide?" and "what could go wrong?"
- **Builder**: Implementation. Answers "how do we build it?" — the only agent that writes production code.
- **Operations**: Synthesis and sustainability. Answers "what did we produce?" and "how do we sustain it?"

**Quality (Judge)**:
- Reviews all Builder outputs before they are marked complete.
- Issues mandatory pass/fail gate decisions.
- Cannot be bypassed by other agents; only a human can override.
- Does not implement fixes — returns findings to Builder for remediation.

### Pattern Benefits

1. **Separation of concerns**: Planner never judges; Quality never implements; Builder never plans.
2. **Deadlock prevention**: No circular authority chains.
3. **Clear escalation**: Judge decisions are final within the system; only humans escalate further.
4. **Composability**: Workers can be added without changing Planner or Quality logic.

---

## 5. Delegation Rules

### Can Delegate

| From | To | Permitted Modes |
|---|---|---|
| Planner | Researcher | All 8 modes |
| Planner | Oracle | All 4 modes |
| Planner | Builder | All 2 modes |
| Planner | Quality | All 4 modes |
| Planner | Operations | All 9 modes |
| Builder | (via Planner only) | Never directly |
| Researcher | (via Planner only) | Never directly |
| Oracle | (via Planner only) | Never directly |
| Quality | Escalates to Planner | Gate failures only |
| Operations | Escalates to Planner | Error recovery failures only |

### Cannot Delegate

- No agent can delegate directly to another agent without going through Planner.
- Builder cannot spawn sub-tasks; it requests additional context through its implementation report.
- Quality cannot instruct Builder directly; all Quality findings route through Planner.
- Oracle cannot instruct Builder directly; architecture decisions become Planner tasks.

### Delegation Depth

```
Depth 1: Planner → Worker (Researcher, Oracle, Builder, Quality, Operations)
Depth 2: Worker → Sub-operation (e.g., Builder → sandbox execution)
Depth 3: Sub-operation → Tool (e.g., sandbox → subprocess)
Depth 4+: PROHIBITED — system raises DelegationDepthError
```

### Self-Delegation Rules

- Planner **cannot** create another Planner instance (no recursive planning).
- Agents **cannot** invoke their own modes recursively.
- An agent that needs a different mode must request it through Planner.

---

## 6. Quality Gates

Every agent has defined quality thresholds. Gate failures trigger a mandatory replan.

### Per-Agent Quality Scores

**Planner**
| Metric | Pass | Warn | Fail |
|---|---|---|---|
| Task DAG validity (no cycles) | true | — | false |
| All task inputs satisfied | true | — | false |
| Clarification items resolved | 0 | — | >0 |
| Plan generation retries | ≤2 | — | 3 (hard fail) |

**Researcher**
| Metric | Pass | Warn | Fail |
|---|---|---|---|
| Findings per task | ≥3 | 1-2 | 0 |
| Verified file paths | 100% | — | any unverified |
| Confidence documented | all | — | any missing |

**Oracle**
| Metric | Pass | Warn | Fail |
|---|---|---|---|
| Architecture candidates evaluated | ≥3 | 2 | 1 |
| Risk scores (L×I) computed | all | — | any missing |
| Assumptions documented | all | — | any missing |

**Builder**
| Metric | Pass | Warn | Fail |
|---|---|---|---|
| Type hints on new functions | 100% | — | any missing |
| Docstrings on public API | 100% | — | any missing |
| Tests run and passing | all pass | — | any fail |
| Hardcoded secrets | 0 | — | ≥1 |

**Quality** (gate thresholds for code it reviews)
| Metric | Pass | Warn | Fail |
|---|---|---|---|
| Code review overall score | ≥7.0 | 5.0-6.9 | <5.0 |
| CRITICAL security findings | 0 | — | ≥1 |
| HIGH security findings | 0 (or mitigated) | 1-2 | ≥3 |
| Test coverage (new code) | ≥80% | 60-79% | <60% |
| Cyclomatic complexity | ≤10 | 11-15 | ≥16 |

**Operations**
| Metric | Pass | Warn | Fail |
|---|---|---|---|
| Public API documented | 100% | — | any gap |
| Broken doc links | 0 | — | ≥1 |
| Error recovery classified | all errors | — | any "unknown" without escalation |

### Gate Decision Flow

```
Builder completes implementation
         ↓
Quality reviews (code_review + security_audit)
         ↓
    gate_decision?
   /              \
PASS              FAIL
  ↓                 ↓
Operations      Planner requeues
(documentation)  Builder with findings
```

---

## 7. Resource Constraints

### Per-Agent Token and Timeout Limits

| Agent | Max Tokens/Turn | Timeout (s) | Max Retries | Notes |
|---|---|---|---|---|
| Planner | 8192 | 120 | 3 | Shorter timeout; planning is fast |
| Researcher | 6144 | 180 | 2 | Web fetch adds latency |
| Oracle | 8192 | 240 | 1 | Full deliberation; no fast retry |
| Builder | 10240 | 300 | 3 | Largest budget for implementation |
| Quality | 8192 | 240 | 2 | Security audit is thorough |
| Operations | 8192 | 240 | 2 | Documentation can be large |

### Model Routing

All agents default to `qwen2.5-72b` for local execution. The dynamic model router (`vetinari/dynamic_model_router.py`) may substitute based on:

- Task complexity score
- Available VRAM (`vetinari/vram_manager.py`)
- Current model pool state (`vetinari/model_pool.py`)
- Cost policy (`vetinari/model_relay.py`)

Overrides are specified via `model_override` in the task spec. Use `null` to allow automatic routing.

### Memory TTL

| Memory Type | TTL | Notes |
|---|---|---|
| Active plan | 7200s (2h) | Extended if plan is paused |
| Task results | 3600s (1h) | Pruned after wave completion |
| Architecture decisions (ADRs) | Permanent | Never pruned |
| Error records | 86400s (24h) | Retained for improvement analysis |
| Research findings | 1800s (30m) | Short-lived; re-fetch if stale |

---

## 8. Collaboration Matrix

### Who Talks to Whom

| Sender | Receiver | Message Type | Trigger |
|---|---|---|---|
| Planner | Researcher | Task assignment | Plan wave begins |
| Planner | Oracle | Task assignment | Architecture/risk decision needed |
| Planner | Builder | Task assignment | Implementation wave begins |
| Planner | Quality | Task assignment | Review wave begins |
| Planner | Operations | Task assignment | Post-build wave begins |
| Researcher | Planner | Research results | Task completed |
| Oracle | Planner | Decisions/ADRs | Analysis completed |
| Builder | Planner | Implementation report | Code written, tests run |
| Quality | Planner | Gate decision (pass/fail) | Review completed |
| Operations | Planner | Completion confirmation | Artefacts produced |
| Quality | Planner (escalation) | CRITICAL finding | Security severity ≥ CRITICAL |
| Operations | Planner (escalation) | Error recovery failure | Max retries exceeded |

### Handoff Protocols

**Planner → Worker handoff**:
```json
{
  "task_id": "uuid",
  "agent": "CONSOLIDATED_RESEARCHER",
  "mode": "code_discovery",
  "description": "Find all JWT-related code",
  "inputs": [],
  "outputs": ["jwt_code_map"],
  "timeout_seconds": 120,
  "memory_context": ["plan:abc123"]
}
```

**Worker → Planner handoff**:
```json
{
  "task_id": "uuid",
  "status": "completed",
  "output_key": "jwt_code_map",
  "output": {...},
  "memory_id": "mem:xyz789",
  "follow_up_tasks": []
}
```

**Quality gate decision handoff**:
```json
{
  "task_id": "uuid",
  "gate_decision": "pass | fail",
  "gate_rationale": "string",
  "findings": [...],
  "remediation_tasks": [...],
  "blocker": true
}
```

---

## 9. Legacy Deprecation

The following 22 legacy agents from Phase 1 and Phase 2 are superseded by the 6 consolidated agents. They remain in `vetinari/agents/` for backward compatibility but **should not be used in new code or plans**.

| Legacy Agent | Class | Replaced By | Migration Mode |
|---|---|---|---|
| Explorer | `ExplorerAgent` | Researcher | `code_discovery` |
| Researcher (legacy) | `ResearcherAgent` | Researcher | `domain_research` |
| Librarian | `LibrarianAgent` | Researcher | `api_lookup` |
| UI Planner | `UIPlannerAgent` | Researcher | `ui_design` |
| Data Engineer | `DataEngineerAgent` | Researcher | `database` |
| DevOps Agent | `DevOpsAgent` | Researcher | `devops` |
| Version Control | `VersionControlAgent` | Researcher | `git_workflow` |
| Oracle (legacy) | `OracleAgent` (non-consolidated) | Oracle | `architecture` |
| Ponder | `PonderAgent` | Oracle | `ontological_analysis` |
| Evaluator | `EvaluatorAgent` | Quality | `code_review` |
| Security Auditor | `SecurityAuditorAgent` | Quality | `security_audit` |
| Test Automation | `TestAutomationAgent` | Quality | `test_generation` |
| Synthesizer | `SynthesizerAgent` | Operations | `synthesis` |
| Documentation Agent | `DocumentationAgent` | Operations | `documentation` |
| Cost Planner | `CostPlannerAgent` | Operations | `cost_analysis` |
| Experimentation Manager | `ExperimentationManagerAgent` | Operations | `experiment` |
| Error Recovery | `ErrorRecoveryAgent` | Operations | `error_recovery` |
| Image Generator | `ImageGeneratorAgent` | Operations (also Builder) | `image_generation` |
| Improvement | `ImprovementAgent` | Operations | `improvement` |
| Context Manager | `ContextManagerAgent` | Planner | `prune` / `extract` |
| User Interaction | `UserInteractionAgent` | Planner | `clarify` |
| Explain Agent | `ExplainAgent` | Operations | `creative_writing` |

### Migration Path

To migrate a plan using legacy agent types:

1. Replace the `assigned_agent` enum value using `LEGACY_TYPE_TO_MODE` mapping in each consolidated agent's class definition.
2. Add the `mode` field matching the target mode.
3. Update memory keys to use the new agent's namespace.

Example migration:
```python
# BEFORE (legacy)
task.assigned_agent = AgentType.EXPLORER
task.mode = None

# AFTER (consolidated)
task.assigned_agent = AgentType.CONSOLIDATED_RESEARCHER
task.mode = "code_discovery"
```

### Deprecation Timeline

- **Phase 3 (current)**: Legacy agents callable but deprecated. New plans must use consolidated agents.
- **Phase 4**: Legacy agents removed from `AGENT_REGISTRY`. Plans using legacy types will fail validation.
- **Phase 5**: Legacy agent class files removed from codebase.

---

## 10. Workflow Pipelines

The following 9 predefined pipelines cover the most common Vetinari use cases. Each pipeline is a wave sequence with agent assignments.

### Pipeline 1: `code_implementation`

Full software feature implementation from spec to documented code.

```
Wave 0 (Research):   Researcher.code_discovery → Researcher.api_lookup
Wave 1 (Advise):     Oracle.architecture
Wave 2 (Build):      Builder.build
Wave 3 (Review):     Quality.code_review → Quality.security_audit
Wave 4 (Operate):    Operations.documentation → Operations.synthesis
```

**Trigger**: User requests a new feature or code change.
**Success criteria**: Quality gate passes; documentation updated; tests passing.

---

### Pipeline 2: `research_analysis`

Deep investigation without implementation.

```
Wave 0 (Research):   Researcher.domain_research
Wave 1 (Advise):     Oracle.architecture (synthesise findings into recommendations)
Wave 2 (Operate):    Operations.synthesis → Operations.documentation
```

**Trigger**: User asks "how should we approach X?" or "research Y for me".
**Success criteria**: Synthesis report delivered with cited findings and ranked recommendations.

---

### Pipeline 3: `security_audit`

Security-focused review of existing code.

```
Wave 0 (Research):   Researcher.code_discovery (map attack surface)
Wave 1 (Advise):     Oracle.risk_assessment
Wave 2 (Review):     Quality.security_audit (all files in scope)
Wave 3 (Operate):    Operations.documentation (security report)
```

**Trigger**: Periodic security review or pre-release security check.
**Success criteria**: No CRITICAL findings; all HIGH findings have documented mitigations.

---

### Pipeline 4: `ui_development`

Frontend feature development with design research.

```
Wave 0 (Research):   Researcher.ui_design
Wave 1 (Build):      Builder.build (implementation)
Wave 2 (Review):     Quality.code_review
Wave 3 (Operate):    Operations.documentation
```

**Trigger**: User requests UI/UX changes or new frontend components.
**Success criteria**: WCAG 2.1 AA compliance verified; Quality gate passes.

---

### Pipeline 5: `documentation`

Update or generate documentation for existing code.

```
Wave 0 (Research):   Researcher.code_discovery (map public API)
Wave 1 (Operate):    Operations.documentation
```

**Trigger**: Docs are stale; new module added without docs; docs audit requested.
**Success criteria**: All public API symbols documented; no broken internal links.

---

### Pipeline 6: `architecture_review`

Evaluate an architectural question or proposed design.

```
Wave 0 (Research):   Researcher.domain_research → Researcher.code_discovery
Wave 1 (Advise):     Oracle.architecture → Oracle.contrarian_review
Wave 2 (Review):     Quality.code_review (existing relevant code)
Wave 3 (Operate):    Operations.documentation (ADR)
```

**Trigger**: Significant design decision or architectural change proposed.
**Success criteria**: ADR produced; contrarian review passed; decision documented.

---

### Pipeline 7: `data_pipeline`

Database schema design and implementation.

```
Wave 0 (Research):   Researcher.database
Wave 1 (Advise):     Oracle.architecture
Wave 2 (Build):      Builder.build (schema + migrations)
Wave 3 (Review):     Quality.code_review → Quality.security_audit
Wave 4 (Operate):    Operations.documentation
```

**Trigger**: New data model needed; schema migration required.
**Success criteria**: Migration runs cleanly; indexes match query patterns; Quality gate passes.

---

### Pipeline 8: `devops_setup`

CI/CD pipeline or infrastructure configuration.

```
Wave 0 (Research):   Researcher.devops
Wave 1 (Advise):     Oracle.risk_assessment
Wave 2 (Build):      Builder.build (pipeline config files)
Wave 3 (Review):     Quality.security_audit (secrets and permissions)
Wave 4 (Operate):    Operations.documentation
```

**Trigger**: New deployment target; CI/CD setup; container configuration.
**Success criteria**: Pipeline runs successfully; no secrets in config files; documentation complete.

---

### Pipeline 9: `full_project`

End-to-end project execution for complex multi-faceted goals.

```
Wave 0 (Discover):   Researcher.code_discovery → Researcher.domain_research
Wave 1 (Advise):     Oracle.architecture → Oracle.risk_assessment
Wave 2 (Build):      Builder.build (iterative, may repeat)
Wave 3 (Review):     Quality.code_review → Quality.security_audit → Quality.test_generation
Wave 4 (Operate):    Operations.synthesis → Operations.documentation → Operations.improvement
```

**Trigger**: Large, ambiguous, or multi-domain requests.
**Success criteria**: All Quality gates passed; full documentation delivered; improvement recommendations logged.

---

## 11. System Prompt Standards

All agent system prompts must meet the following standards to ensure consistent behaviour across model providers and versions.

### Minimum Length

Each mode must have a system prompt of **at least 40 lines**. Shorter prompts lead to inconsistent output formatting and missed edge cases.

### Required Sections (per mode)

Every mode's system prompt must include:

1. **Identity statement** (2-4 lines): Who you are, what your role is in this mode.
2. **Capability scope** (3-5 lines): What you can do and — critically — what you cannot do.
3. **Step-by-step procedure** (10-20 lines): Numbered steps to complete the mode's task.
4. **Output format specification** (5-10 lines): Exact JSON schema or document structure expected.
5. **Error handling** (3-5 lines): What to do when the task cannot be completed normally.
6. **Do not** list (3-5 lines): Explicit prohibitions relevant to this mode.

### Context Engineering Principles

**1. Front-load authority**: State the agent's role and constraints in the first 5 lines. Models are most likely to follow instructions they encounter early.

**2. Explicit over implicit**: Never rely on the model to infer what format you want. State it explicitly with a JSON schema example.

**3. Enumerate steps, not intentions**: "Step 1: Read file X. Step 2: Extract Y from X." is better than "Understand the codebase and extract relevant information."

**4. Constrain scope with negatives**: Every prompt should state what the agent does NOT do. This prevents scope creep across turns.

**5. Schema-first output design**: Define the output schema before describing the task. The model will write to fit the schema.

**6. Idempotent prompts**: A prompt should produce the same structure on repeated invocations with the same input. Avoid temporal language ("first time", "as before") in system prompts.

**7. Mode isolation**: Each mode must be self-contained. Never reference another mode in a mode's system prompt — the model should not need to know about other modes to execute the current one.

### Prompt Versioning

System prompts are versioned independently of agent code. Version format: `{agent}.{mode}.v{N}`.

Prompt versions are stored in `system_prompts/` and referenced in the agent's `AgentSpec.system_prompt` field in `vetinari/agents/contracts.py`.

When updating a prompt:
1. Create a new version file: `system_prompts/{agent}_{mode}_v2.txt`
2. Update `AgentSpec.system_prompt` in `contracts.py`.
3. Run regression tests: `python -m pytest tests/regression/ -x -q`
4. Document the change and motivation in the prompt file header.

### Anti-Patterns to Avoid

| Anti-Pattern | Problem | Fix |
|---|---|---|
| "Be helpful and do your best" | No actionable guidance | Replace with specific steps |
| "Use your judgment" | Non-deterministic | Specify criteria for each decision |
| "See previous context" | Breaks prompt isolation | Include relevant context inline |
| "Output JSON or Markdown" | Ambiguous format | Specify exactly one format |
| Prompts < 40 lines | Insufficient structure | Expand with examples and edge cases |
| No explicit prohibitions | Scope creep | Add "Do not:" section |
| No error handling instructions | Silent failures | Add explicit fallback behaviour |

---

*This document is maintained by the Planner agent and updated during each major architecture phase. For the operational history of agent changes, see `vetinari/adr.py` and `docs/archive/`.*

*Generated: 2026-03-10 | Phase 3 Consolidated Architecture*
