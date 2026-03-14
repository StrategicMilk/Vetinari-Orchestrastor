# Vetinari Coding Agent

The Vetinari Coding Agent is an in-process coding agent that can generate code scaffolds, implementations, tests, and reviews. It integrates with Vetinari's plan mode and memory system.

## Architecture

### Components

1. **CodeAgentEngine** (`vetinari/coding_agent/engine.py`)
   - In-process coding agent using internal LM
   - Generates scaffolds, implementations, tests, and reviews
   - Multi-step task support

2. **CodeBridge** (`vetinari/coding_agent/bridge.py`)
   - External service bridge for offloading heavier tasks
   - Optional integration with CodeNomad-like services

3. **Data Models**
   - `CodeTask`: Coding task specification
   - `CodeArtifact`: Generated code artifact

## Usage

### Basic Usage

```python
from vetinari.coding_agent import CodeAgentEngine, CodeTask, CodingTaskType

agent = CodeAgentEngine()

# Create a scaffold task
task = CodeTask(
    type=CodingTaskType.SCAFFOLD,
    language="python",
    target_files=["my_module"]
)

# Execute
artifact = agent.run_task(task)
print(artifact.path)  # Generated file path
print(artifact.content)  # Generated code
```

### Multi-Step Tasks

```python
tasks = [
    CodeTask(type=CodingTaskType.SCAFFOLD, target_files=["demo"]),
    CodeTask(type=CodingTaskType.IMPLEMENT, target_files=["demo"]),
    CodeTask(type=CodingTaskType.TEST, target_files=["demo"])
]

artifacts = agent.run_multi_step_task(tasks)
```

### API Usage

```bash
# Create a coding task
curl -X POST http://localhost:5000/api/coding/task \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "scaffold",
    "language": "python",
    "description": "Create a new module",
    "target_files": ["my_module"]
  }'

# Multi-step task
curl -X POST http://localhost:5000/api/coding/multi-step \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "plan_id": "plan_123",
    "subtasks": [
      {"subtask_id": "s1", "type": "scaffold", "target_files": ["demo"]},
      {"subtask_id": "s2", "type": "implement", "target_files": ["demo"]},
      {"subtask_id": "s3", "type": "test", "target_files": ["demo"]}
    ]
  }'
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODING_AGENT_ENABLED` | `true` | Enable the coding agent |
| `CODING_AGENT_USE_BRIDGE` | `false` | Use external bridge instead of in-process |
| `CODING_BRIDGE_ENDPOINT` | `http://localhost:4096` | External bridge URL |
| `CODE_BRIDGE_ENABLED` | `false` | Enable external bridge |

## Task Types

- **SCAFFOLD**: Generate project skeleton
- **IMPLEMENT**: Generate implementation code
- **TEST**: Generate unit tests
- **REVIEW**: Generate code review
- **REFACTOR**: Generate refactored code
- **FIX**: Generate bug fix

## Integration with Plan Mode

The coding agent integrates with Vetinari's plan mode:

1. Plan includes coding subtasks
2. Approval required for coding tasks (in Plan mode)
3. After approval, coding agent executes tasks
4. Artifacts logged to dual memory

```python
from vetinari.plan_mode import PlanModeEngine

engine = PlanModeEngine()

# Execute coding task as part of plan
result = engine.execute_coding_task(plan, subtask)
print(result["artifact"])
```

## Memory Integration

Coding artifacts are logged to dual memory (OcMemoryStore + MnemosyneMemoryStore):

- Entry type: `FEATURE`
- Agent: `coding_agent`
- Provenance: `plan:{plan_id},task:{task_id}`

## Security

- Plan gating applies to all coding tasks
- Approvals logged with audit trail
- Sandbox execution for generated code (future)
