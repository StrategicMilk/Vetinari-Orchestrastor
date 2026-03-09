# End-to-End Vetinari Cloud Orchestration Workflow

## Overview

This runbook demonstrates the complete "Golden Path" for transforming a Python package using Vetinari's cloud-native AI orchestration system. We'll walk through:

1. **Environment Setup** - Configure cloud provider credentials
2. **Plan Mode** - Decompose a task into subtasks with approval gating
3. **Distributed Tracing** - Track execution across multiple agents
4. **Memory Integration** - Persist decisions and artifacts with secret filtering
5. **Telemetry** - Observe system performance

## Prerequisites

### 1. Install Dependencies

```bash
# From the Vetinari project root
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Cloud Provider Credentials

Create a `.env` file in the project root with your API keys:

```bash
# OpenAI
OPENAI_API_KEY="sk-..."

# Anthropic
ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
GEMINI_API_KEY="AIza..."

# Cohere
COHERE_API_KEY="co_..."

# Optional: LM Studio local endpoint
LMSTUDIO_BASE_URL="http://localhost:8000"
```

Alternatively, use the credential vault:

```bash
python -c "
from vetinari.credentials import CredentialManager
cm = CredentialManager()
cm.set_credential('openai', 'sk-...', credential_type='bearer', note='OpenAI GPT-4')
cm.set_credential('anthropic', 'sk-ant-...', credential_type='bearer', note='Anthropic Claude')
"
```

## Scenario: Enhance `dry_run_demo_pkg` with Health Check Endpoint

We'll guide the package through the full orchestration pipeline.

### Step 1: Examine the Package

```bash
cd dry_run_demo_pkg
tree dry_run_demo_pkg/
```

Expected structure:
```
dry_run_demo_pkg/
├── pyproject.toml
├── setup.py
└── dry_run_demo_pkg/
    ├── __init__.py
    └── __main__.py
```

### Step 2: Create a Vetinari Plan

Start a Python REPL or script in the Vetinari project root:

```python
from vetinari.structured_logging import configure_logging, CorrelationContext, get_logger
from vetinari.planning.plan_mode import PlanModeEngine
from vetinari.memory import get_dual_memory_store
from vetinari.telemetry import get_telemetry_collector

# Initialize logging with distributed tracing
configure_logging()
logger = get_logger("example_workflow")

# Start a trace for the entire workflow
with CorrelationContext() as ctx:
    logger.info("Starting Vetinari end-to-end example")
    
    # Initialize components
    plan_engine = PlanModeEngine()
    memory = get_dual_memory_store()
    telemetry = get_telemetry_collector()
    
    # Create a plan for enhancing the package
    plan_request = {
        "objective": "Add a health check endpoint to dry_run_demo_pkg",
        "context": "The package is a minimal Python demo package",
        "constraints": "Maintain backward compatibility; use FastAPI if adding HTTP endpoint"
    }
    
    # Generate the plan
    plan = plan_engine.generate_plan(plan_request)
    logger.info(f"Generated plan with {len(plan.subtasks)} subtasks", 
                plan_id=plan.id)
    
    # Store the plan in memory
    from vetinari.memory.interfaces import MemoryEntry, MemoryEntryType
    plan_entry = MemoryEntry(
        content=str(plan),
        entry_type=MemoryEntryType.PLAN,
        agent="example_workflow",
        metadata={"plan_id": plan.id, "objective": plan_request["objective"]}
    )
    memory.remember(plan_entry)
```

### Step 3: Review and Approve Subtasks

The PlanModeEngine will evaluate each subtask's risk level and request approval for high-risk tasks:

```python
    # Review subtasks
    for i, subtask in enumerate(plan.subtasks):
        logger.info(f"Subtask {i+1}: {subtask.name}", 
                    subtask_id=subtask.id,
                    risk_score=subtask.risk_score)
        
        # Risk scoring is automatic:
        # - risk_score < 0.3: auto-approve (low risk)
        # - 0.3 <= risk_score < 0.7: require approval
        # - risk_score >= 0.7: reject (high risk) or require escalation
        
        if subtask.requires_approval():
            # Simulate approval (in production, use web UI or API)
            logger.info(f"Requesting approval for {subtask.name}")
            ctx.set_span_id(f"approval_{subtask.id}")
            
            # Record the decision in memory
            approval_entry = MemoryEntry(
                content=f"Approved: {subtask.name}",
                entry_type=MemoryEntryType.APPROVAL,
                agent="example_workflow",
                metadata={
                    "subtask_id": subtask.id,
                    "risk_score": subtask.risk_score,
                    "decision": "approve"
                }
            )
            memory.remember(approval_entry)
            telemetry.record_plan_decision("approve", risk_score=subtask.risk_score)
        else:
            logger.info(f"Auto-approved (low risk): {subtask.name}")
            telemetry.record_plan_decision("approve", risk_score=subtask.risk_score, 
                                          auto_approved=True)
```

### Step 4: Execute Coding Tasks

Use the in-process CodeAgentEngine to generate code for approved subtasks:

```python
    from vetinari.coding_agent.engine import CodeAgentEngine
    
    code_engine = CodeAgentEngine()
    
    for subtask in plan.subtasks:
        if subtask.domain == "coding":
            logger.info(f"Executing coding task: {subtask.name}",
                       subtask_id=subtask.id)
            ctx.set_span_id(f"coding_{subtask.id}")
            
            # Generate code
            task_type = subtask.subtask_type  # e.g., "implement", "test"
            result = code_engine.execute(
                task_type=task_type,
                description=subtask.description,
                context={
                    "package_path": "dry_run_demo_pkg",
                    "subtask_id": subtask.id
                }
            )
            
            # Store result in memory
            if result.success:
                code_entry = MemoryEntry(
                    content=result.generated_code,
                    entry_type=MemoryEntryType.CODE,
                    agent="code_agent_engine",
                    metadata={
                        "subtask_id": subtask.id,
                        "file_path": result.file_path,
                        "task_type": task_type
                    }
                )
                memory.remember(code_entry)
                logger.info(f"Code generated: {result.file_path}")
            else:
                logger.error(f"Code generation failed: {result.error}",
                           subtask_id=subtask.id)
```

### Step 5: Verify Secrets Filtering

Demonstrate that the security module prevents credentials from being stored:

```python
    from vetinari.security import get_secret_scanner
    
    scanner = get_secret_scanner()
    
    # This content has an API key embedded
    unsafe_content = """
    Configuration for OpenAI:
    api_key: sk-proj-1234567890abcdefghijk
    endpoint: https://api.openai.com/v1
    """
    
    # Attempt to store it (secret will be filtered)
    unsafe_entry = MemoryEntry(
        content=unsafe_content,
        entry_type=MemoryEntryType.CONFIG,
        agent="example_workflow",
        metadata={"type": "api_config"}
    )
    
    memory_id = memory.remember(unsafe_entry)
    logger.info("Stored config entry (secrets automatically filtered)", 
                memory_id=memory_id)
    
    # Verify the stored content
    retrieved = memory.get_entry(memory_id)
    if "[REDACTED]" in retrieved.content:
        logger.info("✓ Secret filtering works - API key was redacted")
    else:
        logger.warning("⚠ Secret filtering may not have worked")
```

### Step 6: Export Telemetry

View system performance metrics:

```python
    # Export telemetry as JSON
    telemetry.export_json("logs/telemetry.json")
    logger.info("Telemetry exported to logs/telemetry.json")
    
    # Print summary
    plan_metrics = telemetry.get_plan_metrics()
    logger.info(f"Plan Metrics:",
                total_decisions=plan_metrics.total_decisions,
                approval_rate=f"{plan_metrics.approval_rate:.1f}%",
                avg_risk_score=f"{plan_metrics.average_risk_score:.2f}")
    
    # View memory metrics
    memory_metrics = telemetry.get_memory_metrics()
    for backend_name, metrics in memory_metrics.items():
        logger.info(f"Memory {backend_name}:",
                   total_ops=metrics.total_writes + metrics.total_reads,
                   dedup_hit_rate=f"{metrics.dedup_hit_rate:.1f}%")
```

### Step 7: View Distributed Traces

Extract logs with trace IDs for analysis:

```bash
# View JSON logs
cat logs/vetinari.log | jq '.trace_id'

# Or filter by trace ID
export TRACE_ID="<from output above>"
cat logs/vetinari.log | jq "select(.trace_id == \"$TRACE_ID\")"
```

## Complete Script Example

Save this as `example_workflow.py` in the Vetinari project root:

```python
"""
Complete end-to-end example of Vetinari cloud orchestration.
Demonstrates: Plan Mode -> Approval -> Code Generation -> Memory -> Telemetry
"""

import json
from vetinari.structured_logging import (
    configure_logging, CorrelationContext, get_logger, traced_operation
)
from vetinari.planning.plan_mode import PlanModeEngine
from vetinari.memory import get_dual_memory_store
from vetinari.memory.interfaces import MemoryEntry, MemoryEntryType
from vetinari.telemetry import get_telemetry_collector
from vetinari.security import get_secret_scanner
from vetinari.coding_agent.engine import CodeAgentEngine

# Initialize
configure_logging()
logger = get_logger("example")

@traced_operation("end_to_end_workflow")
def main():
    """Run the complete workflow."""
    
    # Initialize components
    plan_engine = PlanModeEngine()
    memory = get_dual_memory_store()
    telemetry = get_telemetry_collector()
    code_engine = CodeAgentEngine()
    scanner = get_secret_scanner()
    
    logger.info("Starting end-to-end orchestration example")
    
    # Step 1: Create a plan
    plan = plan_engine.generate_plan({
        "objective": "Add health check endpoint",
        "context": "Enhance dry_run_demo_pkg",
        "constraints": "Maintain backward compatibility"
    })
    
    logger.info(f"Generated plan: {plan.id}", 
                subtask_count=len(plan.subtasks))
    
    # Step 2: Process subtasks
    for subtask in plan.subtasks:
        logger.info(f"Processing: {subtask.name}")
        telemetry.record_plan_decision(
            "approve",
            risk_score=subtask.risk_score,
            auto_approved=subtask.risk_score < 0.3
        )
        
        # Generate code for coding tasks
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
    
    # Step 3: Export results
    telemetry.export_json("logs/telemetry.json")
    logger.info("Workflow completed")
    
    return {"status": "success", "plan_id": plan.id}

if __name__ == "__main__":
    with CorrelationContext() as ctx:
        result = main()
        print(json.dumps(result, indent=2))
```

Run it:

```bash
python example_workflow.py
```

## Troubleshooting

### Secret Filtering Not Working
- Ensure `vetinari/security.py` patterns match your credential format
- Check logs for "Entry content contained secrets - sanitized"

### No Trace IDs in Logs
- Verify `CorrelationContext()` wraps your code
- Check that `VETINARI_STRUCTURED_LOGGING=true` (default)

### Memory Backend Failures
- Check `logs/vetinari.log` for "backend write failed" errors
- Verify OC Memory and Mnemosyne Memory are initialized

### Telemetry Not Recording
- Ensure `telemetry.record_*()` calls are made before `export_json()`
- Check that adapter/memory/plan metrics are being tracked

## Next Steps

1. **Integrate with CI/CD** - See `.github/workflows/vetinari-ci.yml`
2. **Monitor in Production** - Use Prometheus exporter: `telemetry.export_prometheus()`
3. **Custom Secret Patterns** - Add domain-specific patterns: `scanner.add_pattern()`
4. **Approval UI** - Use `vetinari/plan_api.py` REST endpoints for web-based approval

---

For more information, see:
- `docs/memory_merge_policy.md` - Memory backend coordination
- `docs/plan_mode_architecture.md` - Plan generation and approval flow
- `docs/cloud_adapters.md` - Multi-provider model discovery
