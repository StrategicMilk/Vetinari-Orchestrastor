# JSON-Structured Logging Refactor Plan

**Status:** Phase 2 (Observability) - Ready for Implementation
**Priority:** High
**Estimated Effort:** 20-30 hours of development + 10-15 hours testing

## Overview

This document outlines a comprehensive strategy to replace ad-hoc Python logging with JSON-structured logging across the Vetinari orchestration platform. JSON-structured logging enables:

- **End-to-end tracing** of orchestration flows (plan_id → waves → tasks → artifacts)
- **Machine-readable log aggregation** (Datadog, ELK, Splunk, CloudWatch)
- **Contextual debugging** with structured fields (timestamps, identifiers, event types)
- **Observability dashboards** for latency, success rates, retry counts
- **Audit trails** for compliance and troubleshooting

---

## Log Structure Format

### Standard JSON Log Entry

```json
{
  "timestamp": "2026-03-03T10:30:45.123456Z",
  "level": "INFO",
  "logger": "vetinari.orchestrator",
  "event": "task_started",
  "plan_id": "plan_20260303_001",
  "wave_id": "wave_1",
  "task_id": "task_001",
  "status": "running",
  "details": {
    "model_id": "qwen3-coder-next",
    "task_description": "Implement user authentication",
    "inputs": ["requirements.txt"],
    "outputs": ["auth_module.py"]
  },
  "trace_id": "tr_abc123def456",
  "span_id": "sp_xyz789",
  "duration_ms": null,
  "memory_mb": null,
  "error": null
}
```

### Field Definitions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `timestamp` | ISO8601 | Event timestamp (UTC) | `2026-03-03T10:30:45Z` |
| `level` | enum | Log level | `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL` |
| `logger` | string | Logger name (module path) | `vetinari.orchestrator` |
| `event` | string | Event type (PascalCase) | `TaskStarted`, `ModelDiscovered`, `ExecutionFailed` |
| `plan_id` | string | Unique plan identifier | `plan_20260303_001` |
| `wave_id` | string | Wave identifier (nested under plan) | `wave_1` |
| `task_id` | string | Task identifier (nested under wave) | `task_001` |
| `status` | enum | Current status | `pending`, `running`, `completed`, `failed`, `retrying` |
| `details` | object | Event-specific metadata | `{model_id, task_desc, ...}` |
| `trace_id` | string | Distributed trace ID (for multi-service flows) | `tr_abc123def456` |
| `span_id` | string | Span ID within trace | `sp_xyz789` |
| `duration_ms` | int | Execution duration (milliseconds) | `1234` |
| `memory_mb` | float | Memory usage (megabytes) | `256.5` |
| `error` | object | Error details (if failed) | `{type, message, stack}` |

---

## Implementation Strategy

### Phase 1: Logging Infrastructure (Week 1)

Create `vetinari/structured_logging.py` with:

```python
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

class StructuredFormatter(logging.Formatter):
    """Formats log records as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Convert LogRecord to JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "event": getattr(record, 'event', None),
            "plan_id": getattr(record, 'plan_id', None),
            "wave_id": getattr(record, 'wave_id', None),
            "task_id": getattr(record, 'task_id', None),
            "status": getattr(record, 'status', None),
            "details": getattr(record, 'details', None),
            "trace_id": getattr(record, 'trace_id', None),
            "span_id": getattr(record, 'span_id', None),
            "duration_ms": getattr(record, 'duration_ms', None),
            "memory_mb": getattr(record, 'memory_mb', None),
        }
        
        # Add error info if present
        if record.exc_info:
            log_data["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stack": self.formatException(record.exc_info)
            }
        
        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        return json.dumps(log_data, default=str)

def get_logger(name: str) -> logging.Logger:
    """Get a logger configured for structured logging."""
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    
    return logger

def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    plan_id: Optional[str] = None,
    wave_id: Optional[str] = None,
    task_id: Optional[str] = None,
    status: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    duration_ms: Optional[int] = None,
    memory_mb: Optional[float] = None,
    **kwargs
):
    """Log a structured event."""
    record = logger.makeRecord(
        logger.name,
        level,
        "",
        0,
        "",
        (),
        None
    )
    
    # Attach structured fields
    record.event = event
    record.plan_id = plan_id
    record.wave_id = wave_id
    record.task_id = task_id
    record.status = status
    record.details = details or {}
    record.trace_id = trace_id
    record.span_id = span_id
    record.duration_ms = duration_ms
    record.memory_mb = memory_mb
    
    logger.handle(record)
```

### Phase 2: Integration Points (Week 2-3)

#### A. Orchestrator (`vetinari/orchestrator.py`)

```python
from vetinari.structured_logging import get_logger, log_event

logger = get_logger(__name__)

def run_all(self):
    plan_id = f"plan_{int(time.time())}"
    
    log_event(
        logger, logging.INFO, "OrchestrationStarted",
        plan_id=plan_id,
        status="running",
        details={
            "task_count": len(self.config.get("tasks", [])),
            "max_concurrent": self.max_concurrent
        }
    )
    
    try:
        # ... orchestration logic ...
        
        log_event(
            logger, logging.INFO, "OrchestrationCompleted",
            plan_id=plan_id,
            status="completed",
            details={
                "tasks_completed": len(completed_tasks),
                "tasks_failed": len(failed_tasks)
            }
        )
    except Exception as e:
        log_event(
            logger, logging.ERROR, "OrchestrationFailed",
            plan_id=plan_id,
            status="failed",
            details={"error": str(e)}
        )
        raise
```

#### B. ModelPool (`vetinari/model_pool.py`)

```python
logger = get_logger(__name__)

def discover_models(self):
    plan_id = getattr(self, '_current_plan_id', None)
    
    log_event(
        logger, logging.INFO, "ModelDiscoveryStarted",
        plan_id=plan_id,
        status="running",
        details={
            "host": self.host,
            "memory_budget_gb": self.memory_budget_gb,
            "attempt": self._discovery_retry_count
        }
    )
    
    # ... discovery logic ...
    
    log_event(
        logger, logging.INFO, "ModelDiscoveryCompleted",
        plan_id=plan_id,
        status="completed",
        details={
            "models_discovered": len(self.models),
            "fallback_active": self._fallback_active
        }
    )
```

#### C. Executor (`vetinari/executor.py`)

```python
logger = get_logger(__name__)

def execute_task(self, task_id: str, plan_id: str = None, wave_id: str = None) -> dict:
    start_time = time.time()
    
    log_event(
        logger, logging.INFO, "TaskExecutionStarted",
        plan_id=plan_id,
        wave_id=wave_id,
        task_id=task_id,
        status="running",
        details={
            "model_id": model_id,
            "task_description": task.get("description", "")
        }
    )
    
    try:
        result = self.adapter.chat(lm_model_name, system_prompt, prompt_to_send)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        log_event(
            logger, logging.INFO, "TaskExecutionCompleted",
            plan_id=plan_id,
            wave_id=wave_id,
            task_id=task_id,
            status="completed",
            duration_ms=duration_ms,
            details={
                "output_length": len(result.get("output", "")),
                "model_used": lm_model_name
            }
        )
        
        return {"status": "completed", "task_id": task_id, ...}
    
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        
        log_event(
            logger, logging.ERROR, "TaskExecutionFailed",
            plan_id=plan_id,
            wave_id=wave_id,
            task_id=task_id,
            status="failed",
            duration_ms=duration_ms,
            details={"error": str(e)}
        )
        raise
```

#### D. Scheduler (`vetinari/scheduler.py`)

```python
logger = get_logger(__name__)

def build_schedule_layers(self, config: dict, plan_id: str = None):
    log_event(
        logger, logging.DEBUG, "ScheduleLayerBuildingStarted",
        plan_id=plan_id,
        status="running",
        details={
            "task_count": len(config.get("tasks", [])),
            "max_concurrent": self.max_concurrent
        }
    )
    
    # ... scheduling logic ...
    
    log_event(
        logger, logging.INFO, "ScheduleLayerBuildingCompleted",
        plan_id=plan_id,
        status="completed",
        details={
            "layer_count": len(layers),
            "total_tasks_scheduled": sum(len(layer) for layer in layers)
        }
    )
    
    return layers
```

#### E. Sandbox (`vetinari/sandbox.py`)

```python
logger = get_logger(__name__)

def execute(self, code: str, context: Dict[str, Any] = None) -> SandboxResult:
    execution_id = f"exec_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    
    log_event(
        logger, logging.DEBUG, "SandboxExecutionStarted",
        task_id=execution_id,
        status="running",
        details={
            "code_length": len(code),
            "context_vars": list(context.keys()) if context else []
        }
    )
    
    # ... execution logic ...
    
    duration_ms = int((time.time() - start_time) * 1000)
    
    log_event(
        logger, logging.INFO, "SandboxExecutionCompleted",
        task_id=execution_id,
        status="completed" if result.success else "failed",
        duration_ms=duration_ms,
        memory_mb=result.memory_used_mb,
        details={
            "success": result.success,
            "timeout": duration_ms >= self.timeout * 1000
        }
    )
    
    return result
```

### Phase 3: Metrics Collection (Week 3)

Create `vetinari/metrics.py`:

```python
import time
from typing import Dict, List, Optional
from collections import defaultdict

class MetricsCollector:
    """Collects metrics for observability."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
    
    def record_latency(self, metric_name: str, duration_ms: int):
        """Record execution latency."""
        self.metrics[f"{metric_name}_latency"].append(duration_ms)
    
    def record_success(self, metric_name: str):
        """Increment success counter."""
        self.counters[f"{metric_name}_success"] += 1
    
    def record_failure(self, metric_name: str):
        """Increment failure counter."""
        self.counters[f"{metric_name}_failure"] += 1
    
    def record_retry(self, metric_name: str):
        """Increment retry counter."""
        self.counters[f"{metric_name}_retry"] += 1
    
    def get_metrics(self) -> Dict:
        """Export metrics summary."""
        summary = {
            "counters": dict(self.counters),
            "latencies": {}
        }
        
        for metric_name, values in self.metrics.items():
            if values:
                summary["latencies"][metric_name] = {
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "avg_ms": sum(values) / len(values),
                    "p95_ms": sorted(values)[int(len(values) * 0.95)],
                    "p99_ms": sorted(values)[int(len(values) * 0.99)],
                    "count": len(values)
                }
        
        return summary

# Global metrics instance
metrics = MetricsCollector()
```

### Phase 4: Testing & Validation (Week 4)

Create `tests/test_structured_logging.py`:

```python
import json
import logging
from io import StringIO
from vetinari.structured_logging import StructuredFormatter, get_logger, log_event

def test_json_formatting():
    """Verify logs are valid JSON."""
    logger = get_logger("test")
    
    log_event(
        logger, logging.INFO, "TestEvent",
        plan_id="test_plan_1",
        task_id="test_task_1",
        status="completed",
        duration_ms=100,
        details={"result": "success"}
    )
    
    # Parse and verify
    # (In real test, capture log output and parse JSON)

def test_all_fields_present():
    """Verify all structured fields are present."""
    # Test implementation...

def test_backward_compatibility():
    """Verify legacy logging still works."""
    # Test implementation...
```

---

## Migration Path

### Week 1: Infrastructure
- ✅ Create `structured_logging.py`
- ✅ Create base `log_event()` function
- ✅ Set up JSON formatter
- ✅ Create example tests

### Week 2-3: Integration
- **Day 1-2**: Integrate into `orchestrator.py`
- **Day 3-4**: Integrate into `model_pool.py`
- **Day 5**: Integrate into `executor.py`
- **Day 6**: Integrate into `scheduler.py`
- **Day 7**: Integrate into `sandbox.py`
- **Day 8-10**: Integrate into remaining modules

### Week 4: Metrics & Validation
- Create `metrics.py`
- Add metrics recording
- Write comprehensive tests
- Validate end-to-end

---

## Feature Flags

Add environment variable to control logging mode:

```bash
# Enable structured logging (default: false for backward compatibility)
export VETINARI_STRUCTURED_LOGGING=true

# Control log level
export VETINARI_LOG_LEVEL=INFO

# Log output format (json or text)
export VETINARI_LOG_FORMAT=json

# Enable metric collection
export VETINARI_METRICS_ENABLED=true
```

---

## Backward Compatibility

**Option 1: Dual Logging** (Recommended)
- Keep existing `logging` calls
- Add structured logging in parallel
- Gradually migrate modules

**Option 2: Gradual Migration**
- Start with critical path (orchestrator → executor → model_pool)
- Migrate supporting modules later
- Phase out old logging over time

**Option 3: Hybrid Mode**
- Use feature flag to toggle JSON formatting
- Support both old and new simultaneously
- Allow gradual rollout

---

## Log Aggregation Examples

### ELK Stack Query
```json
{
  "query": {
    "bool": {
      "must": [
        {"match": {"event": "TaskExecutionFailed"}},
        {"range": {"timestamp": {"gte": "now-1h"}}}
      ]
    }
  }
}
```

### Datadog Dashboard
```
avg:vetinari.task.latency_ms{plan_id:*} over plan_id
sum:vetinari.task.failures{plan_id:*}.as_count() over plan_id
```

### CloudWatch Insights Query
```
fields @timestamp, task_id, status, duration_ms
| stats avg(duration_ms) as avg_latency by task_id
| sort avg_latency desc
```

---

## Success Metrics

After implementation, we should achieve:

1. **Observability**: 100% of critical events logged with structured context
2. **Traceability**: End-to-end plan_id → wave_id → task_id correlation
3. **Performance Insight**: Latency percentiles (p50, p95, p99) visible
4. **Reliability**: Retry counts and failure patterns tracked
5. **Audit Trail**: All state changes permanently recorded

---

## Dependencies

- `json` (stdlib)
- `logging` (stdlib)
- Optional: datadog, elk, splunk client libraries (for export)

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Performance overhead | Use lazy evaluation, batch writes, async logging |
| Storage/bandwidth | Implement log rotation, sampling for high-volume events |
| Breaking changes | Maintain backward compatibility with feature flags |
| Debugging difficulty | Provide log parsing tools, dashboards for analysis |

---

## Success Criteria

- [ ] All critical modules emit structured JSON logs
- [ ] End-to-end traces visible (plan → waves → tasks)
- [ ] Latency metrics available (min/max/avg/p95/p99)
- [ ] Success/failure rates tracked per task/model
- [ ] Sample dashboard demonstrates value
- [ ] Tests pass with >80% coverage
- [ ] No breaking changes to existing APIs
- [ ] Documentation updated

---

## Next Steps

1. **Review & Approval** - Stakeholder sign-off on design
2. **Create Infrastructure** - Implement `structured_logging.py`
3. **Integration** - Migrate modules in dependency order
4. **Testing** - Comprehensive test suite
5. **Deployment** - Feature flag rollout
6. **Monitoring** - Validate metrics collection

---

**Document Version:** 1.0
**Created:** 2026-03-03
**Status:** Ready for Implementation
