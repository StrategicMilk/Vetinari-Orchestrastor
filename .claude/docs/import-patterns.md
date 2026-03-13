# Import Patterns

## Correct Import Patterns

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

## Anti-Patterns (do not use)

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

## Adding a New Agent Type

1. Add the enum value to `AgentType` in `vetinari/types.py`.
2. Create the agent class in `vetinari/agents/` or `vetinari/agents/consolidated/`.
3. Add an `AgentSpec` entry to `AGENT_REGISTRY` in `vetinari/agents/contracts.py`.
4. Register the agent in the orchestrator's dispatch table.
5. Add tests in `tests/test_<agent_name>.py`.

---

*This file is a reference copy of the import patterns section from the Vetinari Development Guide.*
