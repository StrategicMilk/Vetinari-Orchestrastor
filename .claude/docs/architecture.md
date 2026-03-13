# Architecture Overview

Vetinari is a **multi-agent orchestration system** that decomposes user goals into structured task graphs and executes them via specialist AI agents. The system runs locally using LM Studio models.

## Six-Agent Architecture

The system uses 6 consolidated agents (Phase 3 architecture):

| Agent | Role | Modes | Class |
|---|---|---|---|
| **Planner** | Orchestration, task decomposition | 6 | `PlannerAgent` |
| **Researcher** | Code discovery, domain research, API lookup | 8 | `ConsolidatedResearcherAgent` |
| **Oracle** | Architecture decisions, risk assessment | 4 | `ConsolidatedOracleAgent` |
| **Builder** | Code implementation (sole writer) | 2 | `BuilderAgent` |
| **Quality** | Code review, security audit, test generation | 4 | `QualityAgent` |
| **Operations** | Documentation, synthesis, error recovery | 9 | `ConsolidatedOperationsAgent` |

See `AGENTS.md` at the repository root for the full agent specification.

## Core Components

```
vetinari/
├── types.py                    # CANONICAL enum source — import all enums from here
├── agents/
│   ├── contracts.py            # AgentSpec, Task, Plan dataclasses + AGENT_REGISTRY
│   ├── interfaces.py           # AgentInterface ABC
│   ├── base_agent.py           # BaseAgent implementation
│   ├── consolidated/
│   │   ├── researcher_agent.py # ConsolidatedResearcherAgent (8 modes)
│   │   ├── oracle_agent.py     # ConsolidatedOracleAgent (4 modes)
│   │   ├── quality_agent.py    # QualityAgent (4 modes)
│   │   └── operations_agent.py # ConsolidatedOperationsAgent (9 modes)
│   ├── builder_agent.py        # BuilderAgent (2 modes)
│   └── planner_agent.py        # PlannerAgent (6 modes)
├── two_layer_orchestration.py  # TwoLayerOrchestrator — main execution engine
├── planning_engine.py          # Plan generation and wave decomposition
├── memory/                     # DualMemoryStore — shared agent memory
├── web_ui.py                   # Flask web server (entry point for UI)
├── adapters/                   # LM Studio model adapters
├── model_pool.py               # Model pool management
├── model_relay.py              # Model routing and selection
└── safety/                     # Safety policies and guardrails
```

## TwoLayerOrchestrator

The main execution engine in `vetinari/two_layer_orchestration.py`. It manages:
- Plan lifecycle (DRAFT → APPROVED → EXECUTING → COMPLETED/FAILED)
- Wave-by-wave task execution
- Agent invocation and result collection
- Shared memory updates between waves

Instantiate via:
```python
from vetinari.two_layer_orchestration import TwoLayerOrchestrator
orchestrator = TwoLayerOrchestrator(config)
result = orchestrator.execute(plan)
```

## DualMemoryStore

Shared memory system used by all agents. Located in `vetinari/memory/`. Agents read and write to a shared blackboard keyed by memory IDs.

```python
from vetinari.shared_memory import SharedMemory
memory = SharedMemory()
memory.store("key", value, ttl=3600)
value = memory.retrieve("key")
```

## Flask Web UI

The web interface is served by `vetinari/web_ui.py` on port 5000 by default. API endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/plans` | POST | Create a new plan |
| `/api/plans` | GET | List all plans |
| `/api/plans/{id}/start` | POST | Start plan execution |
| `/api/plans/{id}/pause` | POST | Pause execution |
| `/api/models` | GET | List available models |
| `/api/sandbox/execute` | POST | Execute code in sandbox |

## Key File Locations

| File | Purpose |
|---|---|
| `vetinari/types.py` | **Canonical enum source** — all enums live here |
| `vetinari/agents/contracts.py` | AgentSpec, Task, Plan, AGENT_REGISTRY |
| `vetinari/agents/interfaces.py` | AgentInterface ABC |
| `vetinari/agents/base_agent.py` | BaseAgent base class |
| `vetinari/two_layer_orchestration.py` | Main execution engine |
| `vetinari/planning_engine.py` | Plan generation logic |
| `vetinari/web_ui.py` | Flask application factory |
| `vetinari/model_relay.py` | Model routing policy |
| `vetinari/shared_memory.py` | Shared agent memory |
| `vetinari/exceptions.py` | Custom exception hierarchy |
| `vetinari/safety/` | Safety policies and content filters |
| `config/` | YAML configuration files |
| `tests/` | Test suite (5119 tests, 0 failures) |
| `docs/` | Documentation |
| `AGENTS.md` | Full agent system specification |

## Agent System Summary

Vetinari uses a **six-agent pipeline** with a clear cognitive division of labour:

```
Planner → Researcher → Oracle → Builder → Quality → Operations
```

- **Planner**: Decomposes goals into task DAGs; routes work to agents.
- **Researcher**: Gathers evidence (code, docs, research) before decisions are made.
- **Oracle**: Makes architecture and risk decisions with explicit reasoning.
- **Builder**: The **only** agent that writes production source files.
- **Quality**: Reviews all Builder output; issues mandatory pass/fail gate decisions.
- **Operations**: Produces documentation, synthesis reports, and manages system health.

For the complete specification — modes, file jurisdiction, delegation rules, workflow pipelines, and deprecation mappings — see **`AGENTS.md`** at the repository root.

For the individual agent prompt specifications (used by Claude Code sub-agents), see **`.claude/agents/`**.

---

*This file is a reference copy of the architecture section from the Vetinari Development Guide. Keep it in sync when the architecture changes.*
