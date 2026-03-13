# Preferred Code Patterns

## File I/O

```python
# CORRECT — always use pathlib + encoding
from pathlib import Path

content = Path("config.yaml").read_text(encoding="utf-8")
Path("output.json").write_text(json.dumps(data), encoding="utf-8")

# CORRECT — context manager for streaming
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        process(line)

# WRONG — no encoding, uses os.path
import os
with open(os.path.join("config", "file.yaml")) as f:
    content = f.read()
```

## Exception Handling

```python
# CORRECT — specific exception, chained, logged
try:
    result = agent.execute(task)
except AgentError as exc:
    logger.exception("Agent %s failed on task %s", agent.name, task.id)
    raise PlanError(f"Task {task.id} failed: {exc}") from exc

# WRONG — bare except, swallowed, print
try:
    result = agent.execute(task)
except:
    print("something went wrong")
```

## Configuration Access

```python
# CORRECT — use config dict with defaults
timeout = config.get("timeout", 30)
model_name = config.get("model", "qwen2.5-72b")

# WRONG — hardcoded values
timeout = 30
model_name = "qwen2.5-72b"
```

## Agent Implementation

```python
# CORRECT — follow the established pattern
from __future__ import annotations

import logging
from typing import Any

from vetinari.types import AgentType
from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult

logger = logging.getLogger(__name__)

class MyAgent(BaseAgent):
    """One-line description of what this agent does.

    Extended description with design rationale.
    """

    def execute(self, task: dict[str, Any]) -> AgentResult:
        """Execute the given task and return a result.

        Args:
            task: Task specification dictionary.

        Returns:
            AgentResult with output and metadata.
        """
        logger.info("Executing task %s in mode %s", task.get("id"), task.get("mode"))
        # ... implementation ...
        return AgentResult(output=result, success=True)
```

## Test Implementation

```python
# CORRECT — descriptive name, isolated, uses fixtures
import pytest
from vetinari.agents.contracts import AgentResult

class TestMyAgent:
    def test_execute_returns_success_for_valid_task(self, mock_model):
        agent = MyAgent(config={"model": mock_model})
        result = agent.execute({"id": "t1", "mode": "build"})
        assert result.success is True
        assert result.output is not None

    def test_execute_raises_agent_error_for_invalid_mode(self, mock_model):
        agent = MyAgent(config={"model": mock_model})
        with pytest.raises(AgentError, match="Unknown mode"):
            agent.execute({"id": "t2", "mode": "invalid"})
```

## Enum Usage

```python
# CORRECT — import from canonical source
from vetinari.types import AgentType, TaskStatus

if agent_type == AgentType.BUILDER:
    ...

task.status = TaskStatus.COMPLETED

# WRONG — string comparison
if agent_type == "BUILDER":
    ...
```

---

*This file shows preferred patterns for common operations in the Vetinari codebase.*
