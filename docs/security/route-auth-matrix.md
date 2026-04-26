# Route Auth Matrix

Mutating routes (POST / PUT / DELETE) across all live Litestar handlers.

**Auth levels:**
- `admin` — requires valid `VETINARI_ADMIN_TOKEN` via `X-Admin-Token` or `Bearer` header (guard: `admin_guard`)
- `csrf` — requires `X-Requested-With` header only (CSRF protection, no token)
- `public` — no auth required (read-only or low-risk utility)
- `exempt` — internal/framework path (A2A machine-to-machine, MCP transport — separately guarded)

**Recent hardening note:** Routes marked `*` had `admin_guard` added.

---

## Approvals & Autonomy (`approvals_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/approvals/{action_id}/approve` | POST | `admin` * | Changed: was csrf-only |
| `/api/v1/approvals/{action_id}/reject` | POST | `admin` * | Changed: was csrf-only |
| `/api/v1/autonomy/promote/{action_type}` | POST | `admin` * | Changed: was csrf-only |
| `/api/v1/autonomy/veto/{action_type}` | POST | `admin` * | Changed: was csrf-only |
| `/api/v1/autonomy/veto/{action_type}` | DELETE | `admin` * | Changed: was csrf-only |

## Autonomy Policy (`litestar_autonomy_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/autonomy/policies/{action_type}` | PUT | `admin` * | Changed: was csrf-only |
| `/api/v1/autonomy/promotions/{action_type}/veto` | POST | `admin` * | Changed: was csrf-only |
| `/api/v1/undo/{action_id}` | POST | `admin` * | Changed: was csrf-only |

## Milestones (`litestar_milestones_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/milestones/approve` | POST | `admin` * | Changed: was csrf-only |

## ADR (`litestar_adr_routes.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/adr` | POST | `admin` | |
| `/api/adr/{adr_id}` | PUT | `admin` | |
| `/api/adr/{adr_id}/deprecate` | POST | `admin` | |
| `/api/adr/propose` | POST | `admin` | |
| `/api/adr/propose/accept` | POST | `admin` | |
| `/api/adr/from-plan` | POST | `admin` | |
| `/api/adr/{adr_id}/link-plan` | POST | `admin` | |

## Admin Credentials (`litestar_admin_routes.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/admin/credentials/{source_type}` | POST | `admin` | |
| `/api/admin/credentials/{source_type}/rotate` | POST | `admin` | |
| `/api/admin/credentials/{source_type}` | DELETE | `admin` | |

## Agents (`litestar_agents_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/agents/{agent_id}/pause` | POST | `admin` | |
| `/api/v1/agents/{agent_id}/redirect` | POST | `admin` | |
| `/api/v1/agents/{agent_id}/resume` | POST | `admin` | |
| `/api/v1/agents/initialize` | POST | `admin` | |
| `/api/v1/decisions` | POST | `admin` | |

## Analytics (`litestar_analytics.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/analytics/sla/breach` | POST | `admin` | |

## A2A Transport (`litestar_app.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/a2a` | POST | `public` | Machine-to-machine; no browser token expected |
| `/api/v1/a2a/raw` | POST | `public` | Raw A2A passthrough |

## Chat (`litestar_chat_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/chat/attachments` | POST | `admin` | |
| `/api/v1/chat/feedback` | POST | `admin` | |
| `/api/v1/chat/retry/{project_id}/{task_id}` | POST | `admin` | |

## Dashboard Quality (`litestar_dashboard_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/dashboard/quality/batch` | POST | `csrf` | User-facing quality score submission |
| `/api/v1/dashboard/quality/scores-batch` | POST | `csrf` | User-facing batch score submission |

## Dashboard Metrics (`litestar_dashboard_metrics.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/traces` | DELETE | `csrf` | Trace data purge; no admin token required |

## Decomposition (`litestar_decomposition_routes.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/decomposition/decompose` | POST | `admin` | |
| `/api/v1/decomposition/decompose-agent` | POST | `admin` | |

## MCP Transport (`litestar_mcp_transport.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/mcp/message` | POST | `admin` | MCP protocol; separately guarded |

## Memory (`litestar_memory_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/memory` | POST | `csrf` | User-facing memory creation |
| `/api/v1/memory/{entry_id}` | PUT | `csrf` | User-facing memory update |
| `/api/v1/memory/{entry_id}` | DELETE | `csrf` | User-facing memory deletion |

## Model Management (`litestar_model_mgmt.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/models/assign-tasks` | POST | `admin` | |
| `/api/v1/models` | POST | `admin` | |
| `/api/v1/models/{model_id}/delete` | POST | `admin` | |
| `/api/v1/models/chat-stream` | POST | `admin` | SSE stream |
| `/api/v1/vram/phase` | POST | `admin` | |
| `/api/v1/models/cascade-router/build` | POST | `admin` | |
| `/api/v1/models/cascade/disable` | POST | `admin` | |

## Models Catalog (`litestar_models_catalog.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/models/select` | POST | `admin` | |
| `/api/v1/models/policy` | PUT | `admin` | |
| `/api/v1/models/reload` | POST | `admin` | |
| `/api/v1/project/{project_id}/model-search` | POST | `admin` | |
| `/api/v1/project/{project_id}/refresh-models` | POST | `admin` | |
| `/api/v1/models/search` | POST | `admin` | |
| `/api/v1/models/download` | POST | `admin` | |

## Models Discovery (`litestar_models_discovery.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/models/refresh` | POST | `admin` | |
| `/api/v1/score-models` | POST | `admin` | |
| `/api/v1/model-config` | POST | `admin` | |
| `/api/v1/swap-model` | POST | `admin` | |

## Plans (`litestar_plans_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/plan` | POST | `admin` | |
| `/api/v1/plans` | POST | `admin` | |
| `/api/v1/plans/{plan_id}` | PUT | `admin` | |
| `/api/v1/plans/{plan_id}` | DELETE | `admin` | |
| `/api/v1/plans/{plan_id}/start` | POST | `admin` | |
| `/api/v1/plans/{plan_id}/pause` | POST | `admin` | |
| `/api/v1/plans/{plan_id}/resume` | POST | `admin` | |

## Plan (legacy path) (`litestar_plan_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/plan/generate` | POST | `admin` | |
| `/api/plan/{plan_id}/approve` | POST | `admin` | |
| `/api/plan/{plan_id}/subtasks/{subtask_id}/approve` | POST | `admin` | |
| `/api/coding/task` | POST | `admin` | |
| `/api/coding/multi-step` | POST | `admin` | |

## Ponder (`litestar_ponder_routes.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/ponder/choose-model` | POST | `admin` | |
| `/api/ponder/plan/{plan_id}` | POST | `admin` | |

## Projects (`litestar_projects_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/new-project` | POST | `admin` | |
| `/api/project/{project_id}/rename` | POST | `admin` | |
| `/api/project/{project_id}/archive` | POST | `admin` | |
| `/api/project/{project_id}` | DELETE | `admin` | |
| `/api/project/{project_id}/task` | POST | `admin` | |
| `/api/project/{project_id}/task/{task_id}` | PUT | `admin` | |
| `/api/project/{project_id}/task/{task_id}` | DELETE | `admin` | |
| `/api/project/{project_id}/task/{task_id}/rerun` | POST | `admin` | |
| `/api/project/{project_id}/message` | POST | `admin` | |
| `/api/project/{project_id}/execute` | POST | `admin` | |
| `/api/project/{project_id}/verify-goal` | POST | `admin` | |
| `/api/project/{project_id}/approve` | POST | `admin` | |
| `/api/project/{project_id}/assemble` | POST | `admin` | |
| `/api/project/{project_id}/merge` | POST | `admin` | |
| `/api/project/{project_id}/files/read` | POST | `public` | Read-only file access, no mutation |
| `/api/project/{project_id}/files/write` | POST | `admin` | |
| `/api/project/{project_id}/model-search` | POST | `admin` | |
| `/api/project/{project_id}/task/{task_id}/override` | POST | `admin` | |
| `/api/project/{project_id}/refresh-models` | POST | `admin` | |

## Project Execution (`litestar_projects_execution.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/project/{project_id}/cancel` | POST | `admin` | |
| `/api/project/{project_id}/pause` | POST | `admin` | |
| `/api/project/{project_id}/resume` | POST | `admin` | |

## Project Git (`litestar_project_git.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/project/git/commit-message` | POST | `admin` | |
| `/api/v1/project/git/commit-message-path` | POST | `admin` | |
| `/api/v1/project/git/conflicts` | POST | `admin` | |

## Replay (`litestar_replay_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/replay` | POST | `admin` | |

## Rules (`litestar_rules_routes.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/rules/global` | POST | `admin` | |
| `/api/v1/rules/global-prompt` | POST | `admin` | |
| `/api/v1/rules/project/{project_id}` | POST | `admin` | |
| `/api/v1/rules/model/{model_id}` | POST | `admin` | |

## Sandbox (`litestar_sandbox_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/sandbox/execute` | POST | `admin` | Arbitrary code execution — must stay admin |
| `/api/sandbox/plugins/hook` | POST | `admin` | |

## Search (`litestar_search_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/search/index` | POST | `admin` | |

## Settings & Preferences (`litestar_system_content.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/system-prompts` | POST | `admin` | |
| `/api/v1/system-prompts/{name}` | DELETE | `admin` | |
| `/api/v1/preferences` | PUT | `csrf` | User-facing preferences |
| `/api/v1/settings` | PUT | `csrf` | User-facing settings |
| `/api/v1/variant` | PUT | `admin` | |

## Skills (`litestar_skills_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/skills/{skill_id}/validate` | POST | `csrf` | User-facing skill validation |
| `/api/v1/skills/propose` | POST | `csrf` | User-facing skill proposal |

## Subtasks (`litestar_subtasks_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/subtasks/{plan_id}` | POST | `csrf` | Subtask creation (plan-scoped) |
| `/api/v1/subtasks/{plan_id}/{subtask_id}` | PUT | `csrf` | Subtask field update |
| `/api/v1/assignments/execute-pass` | POST | `admin` | |
| `/api/v1/assignments/{plan_id}/{subtask_id}` | PUT | `admin` | Agent assignment override |
| `/api/v1/plans/{plan_id}/migrate_templates` | POST | `admin` | |
| `/api/v1/project/{project_id}/verify-goal` (subtasks copy) | POST | `csrf` | Verify goal — subtasks module copy |

## System (`litestar_system_status.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/config` | POST | `admin` | |
| `/api/v1/config` | PUT | `admin` | |
| `/api/v1/validate-path` | POST | `public` | Path existence check; no mutation |
| `/api/v1/browse-directory` | POST | `public` | Directory listing; no mutation |

## System Hardware (`litestar_system_hardware.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/server/shutdown` | POST | `admin` | |
| `/api/v1/system/vram/phase` | POST | `admin` | |

## Tasks (`litestar_tasks_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/run-task` | POST | `admin` | |
| `/api/v1/run-all` | POST | `admin` | |
| `/api/v1/run-prompt` | POST | `admin` | |
| `/api/v1/project/{project_id}/task/{task_id}/override` (tasks copy) | POST | `admin` | |

## Training (`litestar_training_api.py`, `_part2.py`, `_part3.py`, `_routes.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/training/start` | POST | `admin` | |
| `/api/v1/training/pause` | POST | `admin` | |
| `/api/v1/training/resume` | POST | `admin` | |
| `/api/v1/training/stop` | POST | `admin` | |
| `/api/v1/training/data/seed` | POST | `admin` | |
| `/api/v1/training/dry-run` | POST | `admin` | |
| `/api/v1/training/rules` | POST | `admin` | |
| `/api/v1/training/sync-data` | POST | `admin` | |
| `/api/v1/training/generate-synthetic` | POST | `admin` | |
| `/api/training/export` (legacy) | POST | `admin` | |
| `/api/training/start` (legacy) | POST | `admin` | |
| `/api/generate-image` (legacy) | POST | `admin` | |

## Training Experiments (`litestar_training_experiments_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/training/experiments/compare` | POST | `csrf` | Read-like comparison operation |
| `/api/v1/training/automation/rules` | POST | `admin` | |
| `/api/v1/training/data/upload` | POST | `admin` | File upload |

## Visualization (`litestar_visualization.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/plans/{plan_id}/approve-gate` | POST | `csrf` | Plan gate approval — user-facing workflow step |

## Workflow Gates (`litestar_manufacturing_api.py`)

| Route | Method | Auth Level | Notes |
|---|---|---|---|
| `/api/v1/workflow/gates/{stage}` | POST | `csrf` | Workflow gate creation — user-facing |
| `/api/v1/workflow/gates/{stage}` | PUT | `csrf` | Workflow gate update — user-facing |
| `/api/v1/workflow/gates/{stage}` | DELETE | `csrf` | Workflow gate deletion — user-facing |

---

## Summary

| Auth Level | Route Count | Rationale |
|---|---|---|
| `admin` | 85 | Server control, data mutation, model management, pipeline execution |
| `csrf` | 17 | User-facing operations where browser CSRF protection is sufficient |
| `public` | 5 | Read-only or machine-to-machine paths with no meaningful mutation risk |

## Routes Still csrf-Only (not escalated to admin)

These routes were identified as CSRF-only. They were intentionally left as
`csrf` because they are user-facing UI operations, not admin-level server
control:

| Route | Rationale for csrf-only |
|---|---|
| `POST /api/v1/dashboard/quality/batch` | Quality score submission from UI |
| `POST /api/v1/dashboard/quality/scores-batch` | Batch quality score from UI |
| `POST /api/v1/memory` | User memory creation |
| `PUT /api/v1/memory/{entry_id}` | User memory update |
| `DELETE /api/v1/memory/{entry_id}` | User memory deletion |
| `POST /api/v1/skills/propose` | User skill proposal |
| `POST /api/v1/skills/{skill_id}/validate` | User skill validation |
| `PUT /api/v1/preferences` | User preference update |
| `PUT /api/v1/settings` | User settings update |
| `POST /api/v1/workflow/gates/{stage}` | Workflow gate mutation |
| `PUT /api/v1/workflow/gates/{stage}` | Workflow gate mutation |
| `DELETE /api/v1/workflow/gates/{stage}` | Workflow gate mutation |
| `POST /api/v1/subtasks/{plan_id}` | Subtask creation |
| `PUT /api/v1/subtasks/{plan_id}/{subtask_id}` | Subtask update |
| `POST /api/v1/training/experiments/compare` | Training comparison (read-like) |
| `POST /api/plans/{plan_id}/approve-gate` | Plan gate step — user workflow |
| `DELETE /api/v1/traces` | Trace purge — non-critical telemetry |
