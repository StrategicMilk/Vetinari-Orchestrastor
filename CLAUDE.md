# Vetinari Development Guide

This is the authoritative development guide for the Vetinari codebase. All contributors and AI agents working on this project must follow these conventions.

---

## Quick Reference

```bash
# Run tests
python -m pytest tests/ -x -q

# Run tests with coverage
python -m pytest tests/ -x -q --cov=vetinari --cov-report=term-missing

# Verify package imports correctly
python -c "import vetinari; print('OK')"

# Run a specific test file
python -m pytest tests/test_contracts.py -x -q

# Run regression tests
python -m pytest tests/regression/ -x -q

# Start the web UI
python -m vetinari

# Check for type errors (if pyright/mypy available)
python -m mypy vetinari/ --ignore-missing-imports
```

---

## Build and Test Commands

| Task | Command |
|---|---|
| Run full test suite | `python -m pytest tests/ -x -q` |
| Run with verbose output | `python -m pytest tests/ -v` |
| Run single test file | `python -m pytest tests/test_<module>.py -x -q` |
| Run regression suite | `python -m pytest tests/regression/ -x -q` |
| Coverage report | `python -m pytest tests/ --cov=vetinari --cov-report=term-missing` |
| Verify import | `python -c "import vetinari; print('OK')"` |
| Verify types file | `python -c "from vetinari.types import AgentType; print(AgentType.PLANNER)"` |
| Start server | `python -m vetinari` |
| Lint (if configured) | `python -m flake8 vetinari/ --max-line-length=120` |

**Test discovery**: pytest discovers tests in `tests/` matching `test_*.py`. Test functions must be prefixed `test_`.

**Test isolation**: Each test must be independent. No shared mutable state between tests. Use `pytest.fixture` for setup.

---

## Project Conventions

### Python Version

**Python 3.10+** is required. Use Python 3.10+ language features:
- `X | Y` union syntax (not `Union[X, Y]`)
- `list[str]` (not `List[str]`)
- `dict[str, Any]` (not `Dict[str, Any]`)
- `match/case` statements where appropriate
- `from __future__ import annotations` at the top of all new files

### Style

- **PEP 8** compliance is mandatory.
- Line length: soft limit 88 characters, hard limit 120 characters.
- Use 4-space indentation (never tabs).
- Two blank lines between top-level definitions.
- One blank line between methods within a class.
- No trailing whitespace.

### Type Hints

All new function signatures must be fully annotated:
```python
# CORRECT
def process_task(task: Task, config: dict[str, Any]) -> AgentResult:
    ...

# WRONG — missing type hints
def process_task(task, config):
    ...
```

### Docstrings

Use **Google-style docstrings** for all public functions, methods, and classes:

```python
def verify_token(token: str, secret_key: str) -> dict[str, Any]:
    """Verify a JWT token and return its decoded payload.

    Args:
        token: The JWT token string to verify.
        secret_key: The secret key used to sign the token.

    Returns:
        Decoded token payload as a dictionary.

    Raises:
        ValueError: If the token is invalid or expired.

    Example:
        >>> payload = verify_token("eyJ...", "my-secret")
        >>> print(payload["sub"])
    """
```

Private functions (prefixed `_`) may use shorter docstrings but must still document non-obvious behaviour.

### Error Handling

- Never use bare `except:` — always catch specific exception types.
- Always chain exceptions: `raise ValueError("message") from exc`
- Do not swallow exceptions silently. If you catch and discard, add a log call.
- Use custom exception classes from `vetinari/exceptions.py` where appropriate.

### Imports

**Import order** (enforced by convention):
1. Standard library imports
2. Third-party imports
3. Local imports (`vetinari.*`)

**Canonical import sources** — always use these, never redefine locally:
```python
# Enums — ALWAYS from vetinari.types
from vetinari.types import AgentType, TaskStatus, ExecutionMode, PlanStatus

# Agent specs and dataclasses — from vetinari.agents.contracts
from vetinari.agents.contracts import AgentSpec, Task, Plan, AgentResult

# Agent interface — from vetinari.agents.interfaces
from vetinari.agents.interfaces import AgentInterface
```

Never use wildcard imports (`from module import *`).

---

## Architecture Overview

Vetinari is a **multi-agent orchestration system** that decomposes user goals into structured task graphs and executes them via specialist AI agents. The system runs locally using LM Studio models.

### Six-Agent Architecture

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

### Core Components

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

### TwoLayerOrchestrator

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

### DualMemoryStore

Shared memory system used by all agents. Located in `vetinari/memory/`. Agents read and write to a shared blackboard keyed by memory IDs.

```python
from vetinari.shared_memory import SharedMemory
memory = SharedMemory()
memory.store("key", value, ttl=3600)
value = memory.retrieve("key")
```

### Flask Web UI

The web interface is served by `vetinari/web_ui.py` on port 5000 by default. API endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/plans` | POST | Create a new plan |
| `/api/plans` | GET | List all plans |
| `/api/plans/{id}/start` | POST | Start plan execution |
| `/api/plans/{id}/pause` | POST | Pause execution |
| `/api/models` | GET | List available models |
| `/api/sandbox/execute` | POST | Execute code in sandbox |

---

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
| `tests/` | Test suite (6169 tests, 0 failures) |
| `docs/` | Documentation |
| `AGENTS.md` | Full agent system specification |

---

## Import Patterns

### Correct Import Patterns

```python
# Types and enums — ALWAYS from vetinari.types
from vetinari.types import AgentType, TaskStatus, ExecutionMode, PlanStatus

# Agent contracts — from vetinari.agents.contracts
from vetinari.agents.contracts import AgentSpec, Task, Plan, AgentResult, AgentTask

# Agent interface — from vetinari.agents.interfaces
from vetinari.agents.interfaces import AgentInterface

# Base class — from vetinari.agents.base_agent
from vetinari.agents.base_agent import BaseAgent

# Orchestrator
from vetinari.two_layer_orchestration import TwoLayerOrchestrator

# Shared memory
from vetinari.shared_memory import SharedMemory

# Exceptions
from vetinari.exceptions import VetinariError, AgentError, PlanError
```

### Anti-Patterns (do not use)

```python
# WRONG — redefining an enum that exists in types.py
class AgentType(Enum):
    PLANNER = "PLANNER"  # Already defined in vetinari.types

# WRONG — importing AgentType from contracts (it re-exports from types, but use the canonical source)
from vetinari.agents.contracts import AgentType

# WRONG — wildcard import
from vetinari.types import *

# WRONG — relative import beyond one level
from ....types import AgentType
```

### Adding a New Agent Type

1. Add the enum value to `AgentType` in `vetinari/types.py`.
2. Create the agent class in `vetinari/agents/` or `vetinari/agents/consolidated/`.
3. Add an `AgentSpec` entry to `AGENT_REGISTRY` in `vetinari/agents/contracts.py`.
4. Register the agent in the orchestrator's dispatch table.
5. Add tests in `tests/test_<agent_name>.py`.

---

## Development Workflow

### Branch Naming

```
feature/short-description        # New features
fix/bug-description              # Bug fixes
refactor/what-is-changing        # Refactoring
docs/what-is-documented          # Documentation only
test/what-is-being-tested        # Test additions
chore/what-is-being-done         # Maintenance tasks
```

### Commit Conventions

Follow Conventional Commits format:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`

Examples:
```
feat(researcher): add git_workflow mode to ConsolidatedResearcherAgent
fix(quality): correct security pattern matching for f-string SQL injection
test(builder): add tests for image_generation mode
docs(agents): update AGENTS.md with Phase 3 delegation rules
```

### Before Committing

1. Run `python -m pytest tests/ -x -q` — all tests must pass.
2. Verify `python -c "import vetinari; print('OK')"` succeeds.
3. Check no new `TODO`/`FIXME`/`HACK` comments without issue references.
4. Confirm no hardcoded secrets or credentials.
5. Confirm all new public functions have type hints and docstrings.

### Pull Request Requirements

- All CI checks must pass.
- At least one test added for every new function.
- AGENTS.md updated if agent roles, modes, or file jurisdiction changed.
- CHANGELOG.md entry added under `[Unreleased]`.

---

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

*This file describes the Vetinari project to AI coding agents and human developers alike. Keep it accurate and up to date when the architecture changes.*
