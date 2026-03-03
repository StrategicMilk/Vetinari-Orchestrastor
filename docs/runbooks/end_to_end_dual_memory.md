# End-to-End Dry-Run: Dual Memory + Plan Gate + Coding Scaffold

This runbook demonstrates Vetinari's end-to-end flow using:
- Dual memory backends (OcMemoryStore + MnemosyneMemoryStore)
- Plan gating across all domains
- Approval workflow with audit logging
- CodingBridge scaffold generation

## Overview

The flow follows these steps:
1. **Plan** - Generate a plan for a task
2. **Approve** - Human approval via plan API (or auto-approve for low-risk)
3. **Execute** - CodingBridge generates scaffold files
4. **Verify** - Confirm files exist and log to dual memory

## Prerequisites

- Vetinari installed and configured
- DualMemoryStore enabled (default)
- Plan mode enabled
- Admin token configured (for approval endpoints)

## Step 1: Generate a Plan

Create a plan for scaffolding a Python package:

```bash
curl -X POST http://localhost:5000/api/plan/generate \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Scaffold a Python package named demo_app",
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

Check if the plan requires approval:

```bash
curl -X GET http://localhost:5000/api/plan/plan_abc123 \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

For coding tasks in Plan mode, approval is required.

## Step 3: Approve the Plan

If approval is required, approve via the API:

```bash
curl -X POST http://localhost:5000/api/plan/plan_abc123/approve \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "approver": "admin",
    "reason": "Low risk dry-run task",
    "risk_score": 0.15
  }'
```

Response:
```json
{
  "success": true,
  "plan_id": "plan_abc123",
  "status": "approved",
  "approved_by": "admin",
  "approved_at": "2026-03-03T10:30:00Z",
  "audit_id": "mem_abc123"
}
```

## Step 4: Execute (Generate Scaffold)

Use CodingBridge to generate the scaffold:

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

## Step 5: Verify and Log to Memory

The scaffold generation logs to dual memory automatically:

```python
from vetinari.memory import get_dual_memory_store, MemoryEntry, MemoryEntryType

store = get_dual_memory_store()

# Search for the scaffold generation entry
results = store.search("scaffold")
for entry in results:
    print(f"Agent: {entry.agent}")
    print(f"Content: {entry.content}")
    print(f"Source: {entry.source_backends}")
```

## Approval JSON Schema

### Plan Approval Request

```json
{
  "plan_id": "string (required)",
  "approved": "boolean (required)",
  "approver": "string (required)",
  "reason": "string (optional)",
  "audit_id": "string (optional, auto-generated)",
  "risk_score": "float (optional)",
  "timestamp": "string (optional, auto-generated)",
  "approval_schema_version": 1
}
```

### Subtask Approval Request

```json
{
  "approved": "boolean (required)",
  "approver": "string (required)",
  "reason": "string (optional)",
  "audit_id": "string (optional)",
  "risk_score": "float (optional)"
}
```

## Dry-Run Demo Package

A toy package scaffold is included at `dry_run_demo_pkg/` for testing:

```bash
cd dry_run_demo_pkg
pip install -e .
python -m dry_run_demo_pkg
```

## Troubleshooting

### Approval Not Logging to Memory

Check that `DUAL_MEMORY_AVAILABLE` is True:
```python
from vetinari.memory import DUAL_MEMORY_AVAILABLE
print(DUAL_MEMORY_AVAILABLE)
```

### CodingBridge Not Generating Scaffold

Ensure the bridge is enabled:
```python
bridge = CodingBridge()
print(bridge.enabled)  # Should be True
```

Or set the environment variable:
```bash
export CODING_BRIDGE_ENABLED=true
```

### Plan Mode Disabled

Enable plan mode:
```bash
export PLAN_MODE_ENABLE=true
export PLAN_MODE_DEFAULT=true
```

## Files Created

- `dry_run_demo_pkg/setup.py` - Package setup
- `dry_run_demo_pkg/dry_run_demo_pkg/__init__.py` - Package init
- `dry_run_demo_pkg/dry_run_demo_pkg/__main__.py` - CLI entry
- `dry_run_demo_pkg/README.md` - Documentation

## Next Steps

1. Extend CodingBridge to call actual external coding agents
2. Add more domain-specific templates
3. Implement parallel task execution
4. Add more sophisticated approval rules
