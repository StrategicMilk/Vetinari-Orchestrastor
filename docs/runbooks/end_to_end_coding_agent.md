# End-to-End Coding Agent Runbook

This runbook demonstrates the full flow of using Vetinari's coding agent to generate a multi-step coding task (scaffold + module + tests).

## Prerequisites

- Vetinari installed and running
- DualMemoryStore enabled (default)
- Plan mode enabled
- Admin token configured

## Step 1: Create a Plan with Coding Tasks

```bash
curl -X POST http://localhost:5000/api/plan/generate \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Create a Python calculator module with tests",
    "dry_run": true,
    "domain_hint": "coding"
  }'
```

Response:
```json
{
  "success": true,
  "plan_id": "plan_abc123",
  "status": "draft",
  "risk_score": 0.15,
  "subtasks": [...]
}
```

## Step 2: Check Approval Requirements

For coding tasks in Plan mode, approval is required:

```bash
curl -X GET http://localhost:5000/api/plan/plan_abc123 \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

## Step 3: Approve the Plan

```bash
curl -X POST http://localhost:5000/api/plan/plan_abc123/approve \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "approver": "admin",
    "reason": "Low risk dry-run task"
  }'
```

## Step 4: Execute Multi-Step Coding Task

### Option A: Using the API

```bash
curl -X POST http://localhost:5000/api/coding/multi-step \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "plan_id": "plan_abc123",
    "subtasks": [
      {
        "subtask_id": "scaffold_001",
        "type": "scaffold",
        "language": "python",
        "description": "Scaffold calculator package",
        "target_files": ["calculator"]
      },
      {
        "subtask_id": "implement_001",
        "type": "implement", 
        "language": "python",
        "description": "Implement Calculator class",
        "target_files": ["calculator"]
      },
      {
        "subtask_id": "test_001",
        "type": "test",
        "language": "python",
        "description": "Write unit tests",
        "target_files": ["calculator"]
      }
    ]
  }'
```

### Option B: Using Python

```python
from vetinari.coding_agent import (
    CodeAgentEngine, CodeTask, CodingTaskType, get_coding_agent
)

agent = get_coding_agent()

tasks = [
    CodeTask(
        subtask_id="scaffold_001",
        type=CodingTaskType.SCAFFOLD,
        language="python",
        target_files=["calculator"]
    ),
    CodeTask(
        subtask_id="implement_001",
        type=CodingTaskType.IMPLEMENT,
        language="python", 
        target_files=["calculator"],
        description="Implement Calculator class with add, sub, mul, div methods"
    ),
    CodeTask(
        subtask_id="test_001",
        type=CodingTaskType.TEST,
        language="python",
        target_files=["calculator"]
    )
]

artifacts = agent.run_multi_step_task(tasks)

for art in artifacts:
    print(f"{art.path}: {len(art.content)} bytes")
```

## Step 5: Verify Artifacts

The generated artifacts will include:

1. **Scaffold** (`calculator/__init__.py`)
```python
"""
calculator - Auto-generated scaffold.
"""

__version__ = "0.1.0"

def main():
    print("Hello from calculator!")

if __name__ == "__main__":
    main()
```

2. **Implementation** (`calculator.py`)
```python
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    def sub(self, x, y):
        return x - y
    
    def mul(self, x, y):
        return x * y
    
    def div(self, x, y):
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
```

3. **Tests** (`test_calculator.py`)
```python
import pytest
from calculator import Calculator

class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5
    
    # ... more tests
```

## Step 6: Check Memory Logs

```python
from vetinari.memory import get_dual_memory_store

store = get_dual_memory_store()

# Search for coding artifacts
results = store.search("coding_agent")

for entry in results:
    print(f"Agent: {entry.agent}")
    print(f"Type: {entry.entry_type}")
    print(f"Provenance: {entry.provenance}")
```

## Troubleshooting

### Coding Agent Not Available

```python
from vetinari.coding_agent import get_coding_agent

agent = get_coding_agent()
print(agent.is_available())  # Should be True

# If False, check:
# - CODING_AGENT_ENABLED=true in environment
```

### Bridge Not Working

- Ensure `CODE_BRIDGE_ENABLED=true`
- Check bridge endpoint is accessible
- Verify API key if required

### Memory Not Logging

- Check dual memory is enabled: `USE_DUAL_MEMORY=true`
- Verify imports: `from vetinari.memory import DUAL_MEMORY_AVAILABLE`

## Files Generated

- `dry_run_demo_pkg/` - Demo package scaffold
- `tests/test_coding_agent.py` - Unit tests
- `docs/coding_agent.md` - Documentation

## Next Steps

1. Extend with more complex implementations
2. Add external bridge integration (CodeNomad)
3. Implement sandboxed execution
4. Add more test frameworks
