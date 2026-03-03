# Vetinari Architecture Guide

## Overview

Vetinari is an AI orchestration agent that automatically plans, assigns, and executes tasks using local and cloud LLM models. The system implements a **Plan-First** architecture where every task goes through explicit planning before execution.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                          │
├─────────────────────────────────────────────────────────────────────┤
│  CLI (cli.py)          │  Web UI (web_ui.py)      │  API Endpoints  │
└────────────────────────┬──────────────────────────┬──────────────────┘
                         │
┌────────────────────────▼──────────────────────────▼──────────────────┐
│                    Orchestration Layer (orchestrator.py)             │
├─────────────────────────────────────────────────────────────────────┤
│  Main Workflow Engine                                               │
│  ├─ Plan Mode Integration (plan_mode.py)                           │
│  ├─ run_all(): Full pipeline                                       │
│  └─ Parallel execution (ThreadPoolExecutor)                        │
└─────────┬────────────┬──────────────┬─────────┬──────────────┬─────┘
          │            │              │         │              │
    ┌─────▼──┐   ┌────▼───┐   ┌─────▼──┐  ┌───▼───┐  ┌──────▼──┐
    │Planning │   │Ponder  │   │Executor│  │Builder│  │Upgrader │
    │Engine   │   │Engine  │   │        │  │       │  │        │
    └─────┬──┘   └────┬───┘   └─────┬──┘  └───┬───┘  └──────┬──┘
          │           │             │         │            │
┌─────────▼───────────▼─────────────▼─────────▼─────────────▼──────┐
│                      Plan Mode Layer                                 │
├───────────────────────────────────────────────────────────────────┤
│  PlanModeEngine (plan_mode.py)                                    │
│  ├─ generate_plan(): Create plan from goal                        │
│  ├─ dry_run_plan(): Generate without execution                   │
│  └─ approve_plan(): Approve/reject plan                          │
│                                                                   │
│  MemoryStore (memory.py)                                          │
│  ├─ SQLite primary storage                                       │
│  ├─ JSON fallback (development)                                  │
│  └─ Plan history & subtask tracking                              │
│                                                                   │
│  Plan API (plan_api.py)                                           │
│  ├─ Admin-gated endpoints                                        │
│  └─ Plan management REST API                                     │
└──────────────┬──────────────────────────────┬──────────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼──────────────────┐
│                      Task Management Layer                        │
├───────────────────────────────────────────────────────────────────┤
│  Scheduler (scheduler.py)   │  Executor (executor.py)            │
│  ├─ build_schedule_layers() │  ├─ execute_task()               │
│  └─ Topological sort (DAG)  │  └─ Prompt loading & execution   │
│                             │                                   │
│  SubtaskTree (subtask_tree.py)  │ Validator (validator.py)     │
│  ├─ Task hierarchy mgmt         │ ├─ Syntax validation         │
│  ├─ Dependency tracking         │ ├─ Format checking           │
│  └─ Audit fields               │ └─ Content validation       │
└──────────────┬──────────────────────────────┬──────────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼──────────────────────┐
│                      Model Layer                                │
├───────────────────────────────────────────────────────────────┤
│  ModelPool (model_pool.py)                                   │
│  ├─ discover_models(): LM Studio auto-discovery               │
│  ├─ get_cloud_models(): Cloud provider enumeration            │
│  ├─ assign_tasks_to_models(): Task-model mapping              │
│  └─ Scoring (multi-factor)                                   │
│                                                              │
│  Ponder (ponder.py)                                         │
│  ├─ Phase 1: Local model scoring                            │
│  ├─ Phase 2: Cloud augmentation                            │
│  └─ score_models_with_cloud(): Augmented ranking            │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Orchestrator (orchestrator.py)

Central workflow coordinator that:
- Initializes all subsystems
- Executes workflow pipeline (discover → plan → assign → schedule → execute → validate → build)
- Handles Plan Mode integration
- Manages parallel task execution

### 2. Plan Mode Engine (plan_mode.py)

Intelligent plan generation and management:
- **generate_plan()**: Creates Plan from goal with multiple candidates
- **dry_run_plan()**: Generates plan without execution (for evaluation)
- **approve_plan()**: Approves or rejects plans
- **Risk scoring**: Calculates risk based on depth, cost, dependencies
- **Auto-approval**: Low-risk plans auto-approved in dry-run mode

### 3. Memory Store (memory.py)

Long-term persistence for plans and outcomes:
- **SQLite** (primary): ACID-compliant, indexed storage
- **JSON fallback** (development): Quick prototyping
- **PlanHistory**: Stores all plans with status, risk scores
- **SubtaskMemory**: Stores subtask outcomes and metrics
- **ModelPerformance**: Tracks model success rates and latency

### 4. Plan API (plan_api.py)

REST endpoints for plan management:
- `POST /api/plan/generate`: Generate a plan
- `GET /api/plan/{plan_id}`: Get plan details
- `POST /api/plan/{plan_id}/approve`: Approve/reject plan
- `GET /api/plan/{plan_id}/history`: Get plan history
- All endpoints require admin token (PLAN_ADMIN_TOKEN)

### 5. Ponder Engine (ponder.py)

Two-pass model selection:
- **Phase 1**: Local model scoring (capability, context, memory, heuristic)
- **Phase 2**: Cloud augmentation (Claude, Gemini, HF, Replicate)
- Returns ranked models with scores

## Data Flow

### Plan-First Workflow

```
User Input / Task
    ↓
Orchestrator.run_all()
    ↓
[If PLAN_MODE_ENABLED]
    PlanModeEngine.generate_plan()
        ├─ Infer domain from goal
        ├─ Generate plan candidates
        ├─ Calculate risk scores
        ├─ Auto-approve if low-risk (dry-run)
        └─ Store in Memory
    ↓
ModelPool.discover_models()
    ↓
Ponder.score_models_with_cloud()
    ↓
Scheduler.build_schedule_layers()
    ↓
Executor.execute_task() [parallel]
    ↓
Validator.validate()
    ↓
Builder.build_artifact()
    ↓
Results + Memory Update
```

### Plan Mode Flow

```
Goal + Constraints
    ↓
PlanModeEngine.generate_plan()
   
    ├─ Template selection (coding ├─ Domain inference, data processing, etc.)
    ├─ Candidate generation (1-3 variants)
    ├─ Risk scoring
    └─ Subtask creation
    ↓
Plan + Candidates
    ↓
[If DRY_RUN_ENABLED]
    ├─ Risk <= Threshold → Auto-approve
    └─ Risk > Threshold → Require approval
    ↓
Execution or Wait for Approval
```

## Configuration

### Plan Mode Settings (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| PLAN_MODE_ENABLE | true | Enable plan-first orchestration |
| PLAN_MODE_DEFAULT | true | Default mode for all tasks |
| DRY_RUN_ENABLED | false | Generate plans without execution |
| DRY_RUN_RISK_THRESHOLD | 0.25 | Auto-approval risk threshold |
| PLAN_DEPTH_CAP | 16 | Maximum subtask depth |
| PLAN_MAX_CANDIDATES | 3 | Max plan variants to generate |
| PLAN_ADMIN_TOKEN | - | Token for admin endpoints |
| PLAN_MEMORY_DB_PATH | ./vetinari_memory.db | SQLite database path |
| PLAN_RETENTION_DAYS | 90 | Plan retention period |

### Memory Store

- **Primary**: SQLite at `./vetinari_memory.db`
- **Fallback**: JSON at `./vetinari_memory.json` (if SQLite unavailable)
- **Pruning**: Automatic deletion of plans older than 90 days

## Security

### Admin Token

Plan management endpoints require admin authentication:
```bash
# Set admin token
export PLAN_ADMIN_TOKEN="your-secret-token"

# API call with token
curl -H "Authorization: Bearer your-secret-token" \
     -X POST http://localhost:5000/api/plan/generate \
     -H "Content-Type: application/json" \
     -d '{"goal": "Build a web app"}'
```

### Data Privacy

- Plans stored locally (SQLite/JSON)
- Prompts can be sanitized before storage
- Admin-only access to plan history

## Extension Points

### Adding Domain Templates

Edit `plan_mode.py` → `_load_domain_templates()`:

```python
TaskDomain.NEW_DOMAIN: [
    {
        "description": "Step 1",
        "domain": TaskDomain.NEW_DOMAIN,
        "definition_of_done": DefinitionOfDone(criteria=["..."]),
        "definition_of_ready": DefinitionOfReady(prerequisites=["..."])
    },
    # ... more steps
]
```

### Adding Plan Endpoints

Add to `plan_api.py`:

```python
@plan_api.route('/api/plan/<plan_id>/custom', methods=['POST'])
@require_admin_token
def custom_endpoint(plan_id):
    # Your custom logic
    return jsonify({"success": True})
```

## See Also

- [CONFIG.md](CONFIG.md) - Configuration reference
- [cloud-ponder.md](cloud-ponder.md) - Ponder model selection
- [api-contracts.md](api-contracts.md) - REST API documentation
