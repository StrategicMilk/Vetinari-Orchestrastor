# Logging Standards

## The Rule

**NEVER use `print()` in production code under `vetinari/`.** Use the `logging` module instead.

Exceptions (where `print()` is allowed):
- `vetinari/__main__.py` — CLI entry point user-facing output
- `vetinari/cli.py` — CLI command output
- `scripts/` — standalone scripts
- `tests/` — test output

## Setting Up a Logger

Every module that needs logging MUST create a module-level logger:

```python
import logging

logger = logging.getLogger(__name__)
```

**Do NOT use:**
- `logging.info(...)` (root logger — pollutes global namespace)
- `print(...)` (no level, no filtering, no structured output)
- `logger = logging.getLogger("custom_name")` (breaks hierarchy)

## Log Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| `logger.debug()` | Verbose diagnostic info for development | `logger.debug("Processing task %s with config %s", task_id, config)` |
| `logger.info()` | Normal operational events | `logger.info("Plan %s started with %d tasks", plan_id, len(tasks))` |
| `logger.warning()` | Something unexpected but recoverable | `logger.warning("Model %s not available, falling back to %s", primary, fallback)` |
| `logger.error()` | An error that prevents an operation | `logger.error("Failed to execute task %s: %s", task_id, exc)` |
| `logger.exception()` | Error with full traceback (use inside except blocks) | `logger.exception("Unhandled error in agent %s", agent_name)` |
| `logger.critical()` | System-wide failure | `logger.critical("Cannot connect to LM Studio: %s", exc)` |

## Formatting Rules

Use %-style formatting (lazy evaluation), NOT f-strings:

```python
# CORRECT — lazy evaluation, string only formatted if message is logged
logger.info("Processing %d items in batch %s", count, batch_id)

# WRONG — f-string always evaluated even if log level is disabled
logger.info(f"Processing {count} items in batch {batch_id}")
```

## Structured Data

For complex data, use `extra` parameter or structured logging:

```python
logger.info("Agent completed task", extra={
    "agent_type": agent.type.value,
    "task_id": task.id,
    "duration_ms": elapsed_ms,
})
```

## Exception Logging

Inside `except` blocks, always use `logger.exception()` or `exc_info=True`:

```python
try:
    result = agent.execute(task)
except AgentError:
    logger.exception("Agent %s failed on task %s", agent.name, task.id)
    raise
```

## Common Migration Pattern (print to logging)

```python
# BEFORE (wrong)
print(f"Starting plan {plan.id}")
print(f"WARNING: Model not found: {model_name}")
print(f"ERROR: {e}")

# AFTER (correct)
logger.info("Starting plan %s", plan.id)
logger.warning("Model not found: %s", model_name)
logger.error("Operation failed: %s", e)
```

---

*This file defines logging standards for the Vetinari project. See CLAUDE.md Section 4.3 for the mandatory rules summary.*
