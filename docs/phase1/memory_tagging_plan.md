# Phase 1 Memory Tagging Plan

## Overview

This document outlines the memory tagging strategy for Phase 1, extending the existing SharedMemory with plan-aware tagging.

---

## 1. Memory Type Taxonomy

### Existing Types (Phase 0)

```python
class MemoryType(Enum):
    INTENT = "intent"           # User goals
    DISCOVERY = "discovery"     # Findings from exploration
    DECISION = "decision"       # Decisions made
    PROBLEM = "problem"         # Issues identified
    SOLUTION = "solution"       # Solutions found
    PATTERN = "pattern"         # Recurring patterns
    WARNING = "warning"         # Warnings
    SUCCESS = "success"        # Successes
    REFACTOR = "refactor"      # Refactoring notes
    BUGFIX = "bugfix"          # Bug fixes
    FEATURE = "feature"        # Feature notes
```

### New Phase 1 Types

```python
class MemoryType(Enum):
    # ... existing types ...
    
    # Planning
    PLAN = "plan"              # Plan created
    WAVE = "wave"              # Wave status
    TASK = "task"              # Task status
    
    # Results
    PLAN_RESULT = "plan_result"      # Plan completion
    WAVE_RESULT = "wave_result"      # Wave completion
    TASK_RESULT = "task_result"      # Task result
    
    # Model Selection
    MODEL_SELECTION = "model_selection"  # Model picked
    
    # Sandbox
    SANDBOX_EVENT = "sandbox_event"    # Sandbox execution
    SANDBOX_ERROR = "sandbox_error"    # Sandbox errors
    
    # Governance
    GOVERNANCE = "governance"      # Governance decisions
```

---

## 2. Memory Schema

### Extended MemoryEntry

```python
@dataclass
class MemoryEntry:
    # Existing fields
    entry_id: str
    agent_name: str
    memory_type: str
    summary: str
    content: str
    timestamp: str
    tags: List[str]
    project_id: str
    session_id: str
    
    # New Phase 1 fields
    plan_id: str = ""           # Link to plan
    wave_id: str = ""          # Link to wave
    task_id: str = ""         # Link to task
    
    # Provenance
    provenance: str = "agent"  # "user", "agent", "system"
    confidence: float = 1.0     # 0.0-1.0
    
    # Metadata
    metadata: dict = field(default_factory=dict)
```

---

## 3. Tagging Rules

### By Event Type

| Event | Memory Type | Tags |
|-------|-------------|------|
| Plan created | PLAN | ["plan", "created", plan_id] |
| Wave started | WAVE | ["wave", "started", plan_id, wave_id] |
| Wave completed | WAVE | ["wave", "completed", plan_id, wave_id] |
| Task assigned | TASK | ["task", "assigned", plan_id, task_id] |
| Task completed | TASK | ["task", "completed", plan_id, task_id] |
| Task failed | TASK | ["task", "failed", plan_id, task_id] |
| Model selected | MODEL_SELECTION | ["model", "selected", model_id] |
| Sandbox executed | SANDBOX_EVENT | ["sandbox", "executed", execution_id] |
| Sandbox error | SANDBOX_ERROR | ["sandbox", "error", execution_id] |

### By Agent

| Agent | Memory Types |
|-------|--------------|
| Explorer | DISCOVERY, SEARCH |
| Librarian | DISCOVERY, RESEARCH |
| Oracle | DECISION, SOLUTION |
| Builder | FEATURE, REFACTOR, BUGFIX |
| Researcher | DISCOVERY, PATTERN |
| Evaluator | PROBLEM, WARNING, SUCCESS |
| Synthesizer | SOLUTION, PATTERN |

---

## 4. Memory Storage

### File Structure

```
memory/
├── memory.json              # All memories
├── by_plan/
│   ├── plan_001.json      # Memories for plan_001
│   └── plan_002.json
├── by_agent/
│   ├── explorer.json
│   ├── librarian.json
│   └── ...
└── by_type/
    ├── decision.json
    ├── solution.json
    └── ...
```

---

## 5. API Endpoints

### Store Memory

```python
POST /api/memory
{
  "agent_name": "explorer",
  "memory_type": "discovery",
  "summary": "Found auth files",
  "content": "Found: src/auth/middleware.ts, lib/jwt.ts",
  "tags": ["auth", "security"],
  "plan_id": "plan_001",
  "task_id": "task_1",
  "provenance": "agent"
}
```

### Query by Plan

```python
GET /api/memory?plan_id=plan_001
```

### Query by Type

```python
GET /api/memory?type=decision
```

### Timeline

```python
GET /api/memory/timeline?plan_id=plan_001
```

---

## 6. Memory Auto-Tagging

### On Plan Creation

```python
def on_plan_created(plan):
    memory.add(
        agent_name="system",
        memory_type="plan",
        summary=f"Plan created: {plan.title}",
        content=plan.prompt,
        tags=["plan", "created"],
        plan_id=plan.plan_id,
        provenance="system"
    )
```

### On Task Completion

```python
def on_task_completed(task, result):
    memory.add(
        agent_name=task.agent_type,
        memory_type="task_result",
        summary=f"Task completed: {task.description}",
        content=result,
        tags=["task", "completed", task.status],
        plan_id=task.plan_id,
        wave_id=task.wave_id,
        task_id=task.task_id,
        provenance="agent"
    )
```

---

## 7. Success Criteria

- [ ] Memory types extended for plans/waves/tasks
- [ ] Memory entries link to plans/waves/tasks
- [ ] Query by plan_id works
- [ ] Query by type works
- [ ] Timeline view shows plan progression
- [ ] Memory auto-tagging on plan events

---

*Document Version: 1.0*
*Phase: 1*
