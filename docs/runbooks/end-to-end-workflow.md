# End-to-End Vetinari Workflow Runbook

This runbook demonstrates the complete "Golden Path" for Vetinari's orchestration system, covering plan generation, approval gating, code generation, dual memory, distributed tracing, and telemetry.

---

## Prerequisites

- Vetinari installed: `pip install -e .`
- DualMemoryStore enabled (default)
- Plan mode enabled (default)
- Admin token configured (`PLAN_ADMIN_TOKEN`)
- LM Studio running (or cloud provider credentials configured)

### Cloud Provider Credentials (Optional)

```bash
# .env file
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GEMINI_API_KEY="AIza..."
COHERE_API_KEY="co_..."
LMSTUDIO_BASE_URL="http://localhost:1234"
```

Or use the credential vault:

```python
from vetinari.credentials import CredentialManager
cm = CredentialManager()
cm.set_credential('openai', 'sk-...', credential_type='bearer', note='OpenAI GPT-4')
```

---

## Core Workflow: Plan, Approve, Execute, Verify

All Vetinari workflows follow this pattern:

```
User Input -> Plan Generation -> Risk Assessment -> Approval -> Execution -> Memory Logging -> Verification
```

### Step 1: Generate a Plan

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

### Step 2: Check Approval Requirements

```bash
curl -X GET http://localhost:5000/api/plan/plan_abc123 \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

Risk scoring determines approval requirements:
- **LOW** (0.0-0.25): Auto-approved in dry-run mode
- **MEDIUM** (0.25-0.5): May require approval
- **HIGH** (0.5-0.75): Requires approval
- **CRITICAL** (0.75-1.0): Requires explicit approval

### Step 3: Approve the Plan

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

### Approval JSON Schemas

**Plan Approval:**
```json
{
  "plan_id": "string (required)",
  "approved": "boolean (required)",
  "approver": "string (required)",
  "reason": "string (optional)",
  "risk_score": "float (optional)",
  "timestamp": "string (auto-generated)"
}
```

**Subtask Approval:**
```json
{
  "approved": "boolean (required)",
  "approver": "string (required)",
  "reason": "string (optional)",
  "risk_score": "float (optional)"
}
```

---

## Workflow A: Multi-Step Coding Agent

Execute scaffold, implementation, and test generation as a pipeline.

### Using the API

```bash
curl -X POST http://localhost:5000/api/coding/multi-step \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "plan_id": "plan_abc123",
    "subtasks": [
      {"subtask_id": "scaffold_001", "type": "scaffold", "language": "python", "target_files": ["calculator"]},
      {"subtask_id": "implement_001", "type": "implement", "language": "python", "target_files": ["calculator"]},
      {"subtask_id": "test_001", "type": "test", "language": "python", "target_files": ["calculator"]}
    ]
  }'
```

### Using Python

```python
from vetinari.coding_agent import CodeAgentEngine, CodeTask, CodingTaskType

agent = CodeAgentEngine()

tasks = [
    CodeTask(subtask_id="scaffold_001", type=CodingTaskType.SCAFFOLD, language="python", target_files=["calculator"]),
    CodeTask(subtask_id="implement_001", type=CodingTaskType.IMPLEMENT, language="python", target_files=["calculator"],
             description="Implement Calculator class with add, sub, mul, div methods"),
    CodeTask(subtask_id="test_001", type=CodingTaskType.TEST, language="python", target_files=["calculator"]),
]

artifacts = agent.run_multi_step_task(tasks)
for art in artifacts:
    print(f"{art.path}: {len(art.content)} bytes")
```

### Verify Artifacts

Generated artifacts are automatically logged to dual memory:

```python
from vetinari.memory import get_dual_memory_store

store = get_dual_memory_store()
results = store.search("coding_agent")
for entry in results:
    print(f"Agent: {entry.agent}, Type: {entry.entry_type}, Provenance: {entry.provenance}")
```

---

## Workflow B: Cloud Orchestration with Distributed Tracing

Full workflow with cloud providers, tracing, secret filtering, and telemetry.

### Complete Python Script

```python
"""
End-to-end example: Plan Mode -> Approval -> Code Generation -> Memory -> Telemetry
"""
import json
from vetinari.structured_logging import configure_logging, CorrelationContext, get_logger
from vetinari.plan_mode import PlanModeEngine
from vetinari.memory import get_dual_memory_store
from vetinari.memory.interfaces import MemoryEntry, MemoryEntryType
from vetinari.telemetry import get_telemetry_collector
from vetinari.security import get_secret_scanner
from vetinari.coding_agent.engine import CodeAgentEngine

configure_logging()
logger = get_logger("example")

def main():
    plan_engine = PlanModeEngine()
    memory = get_dual_memory_store()
    telemetry = get_telemetry_collector()
    code_engine = CodeAgentEngine()

    with CorrelationContext() as ctx:
        # Generate plan
        plan = plan_engine.generate_plan({
            "objective": "Add health check endpoint",
            "context": "Enhance demo package",
            "constraints": "Maintain backward compatibility"
        })
        logger.info("Generated plan: %s", plan.id)

        # Process subtasks
        for subtask in plan.subtasks:
            telemetry.record_plan_decision(
                "approve", risk_score=subtask.risk_score,
                auto_approved=subtask.risk_score < 0.3
            )

            if subtask.domain == "coding":
                result = code_engine.execute(
                    task_type=subtask.subtask_type,
                    description=subtask.description,
                    context={"subtask_id": subtask.id}
                )
                if result.success:
                    memory.remember(MemoryEntry(
                        content=result.generated_code,
                        entry_type=MemoryEntryType.CODE,
                        agent="code_engine",
                        metadata={"subtask_id": subtask.id}
                    ))

        # Export telemetry
        telemetry.export_json("logs/telemetry.json")
        logger.info("Workflow completed")

if __name__ == "__main__":
    main()
```

### Secret Filtering Verification

```python
from vetinari.security import get_secret_scanner

scanner = get_secret_scanner()
unsafe_content = "api_key: sk-proj-1234567890abcdefghijk"
unsafe_entry = MemoryEntry(
    content=unsafe_content, entry_type=MemoryEntryType.CONFIG,
    agent="example", metadata={"type": "api_config"}
)
memory_id = memory.remember(unsafe_entry)

# Verify redaction
retrieved = memory.get_entry(memory_id)
assert "[REDACTED]" in retrieved.content  # Secret was filtered
```

### Viewing Distributed Traces

```bash
# View JSON logs with trace IDs
cat logs/vetinari.log | jq '.trace_id'

# Filter by specific trace
cat logs/vetinari.log | jq 'select(.trace_id == "YOUR_TRACE_ID")'
```

---

## Workflow C: CodingBridge Scaffold Generation

Use CodingBridge for scaffold generation with plan gating and dual memory.

```python
from vetinari.agents.coding_bridge import CodingBridge, CodingTask, CodingTaskType

bridge = CodingBridge()
bridge.enabled = True

task = CodingTask(
    task_type=CodingTaskType.SCAFFOLD,
    description="Scaffold demo_app package",
    language="python",
    context={"project_name": "demo_app"},
    output_path="./demo_app"
)

result = bridge.generate_task(task)
print(f"Success: {result.success}")
print(f"Output files: {result.output_files}")
```

### Dry-Run Demo Package

A toy package scaffold is included at `dry_run_demo_pkg/` for testing:

```bash
cd dry_run_demo_pkg
pip install -e .
python -m dry_run_demo_pkg
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Coding agent not available | Check `CODING_AGENT_ENABLED=true` |
| CodingBridge not generating | Check `CODE_BRIDGE_ENABLED=true` |
| Plan mode disabled | Set `PLAN_MODE_ENABLE=true` and `PLAN_MODE_DEFAULT=true` |
| Approval not logging to memory | Verify `DUAL_MEMORY_AVAILABLE` is True |
| No trace IDs in logs | Wrap code in `CorrelationContext()` |
| Memory backend failures | Check logs for "backend write failed" errors |
| Telemetry not recording | Ensure `telemetry.record_*()` calls before `export_json()` |
| Secret filtering not working | Check `vetinari/security.py` patterns |
| Bridge endpoint unreachable | Verify bridge URL and check API key |

---

## Related Documents

- [Coding Agent Reference](../reference/coding-agent.md) - Coding agent architecture and API
- [Memory System Reference](../reference/memory.md) - Dual-memory architecture
- [Approval Workflow Reference](../reference/approval-workflow.md) - Approval states and API
- [Configuration Reference](../reference/config.md) - Environment variables
- [Dashboard Guide](dashboard-guide.md) - Monitoring dashboard operations
- [Ponder Runbook](ponder.md) - Model selection operations
