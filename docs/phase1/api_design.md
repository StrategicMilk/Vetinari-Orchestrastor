# Vetinari Phase 1 API Design Specification

## Overview

This document defines the REST API surface for Phase 1, covering:
- Plan management (CRUD, execution control)
- Model Relay (model listing, selection, policy)
- Sandbox (execution, status, audit)
- Code Search (search, index, status)

---

## 1. Plan Management API

### 1.1 Create Plan

**Endpoint**: `POST /api/plans`

**Description**: Create a new plan from a prompt

**Request Body**:
```json
{
  "title": "Add authentication to API",
  "prompt": "Add JWT authentication to our Express API with login/logout/register endpoints",
  "waves": [
    {
      "milestone": "Research and Planning",
      "description": "Research JWT libraries and plan implementation",
      "order": 1,
      "tasks": [
        {
          "agent_type": "explorer",
          "description": "Find existing auth code",
          "prompt": "Search for existing authentication code in the codebase",
          "priority": 8
        },
        {
          "agent_type": "librarian",
          "description": "Research JWT best practices",
          "prompt": "Find best practices for JWT authentication in Express",
          "priority": 7
        }
      ]
    },
    {
      "milestone": "Implementation",
      "description": "Implement authentication endpoints",
      "order": 2,
      "dependencies": ["wave_1"],
      "tasks": [
        {
          "agent_type": "builder",
          "description": "Create auth routes",
          "prompt": "Create login, logout, and register routes",
          "priority": 10
        }
      ]
    }
**Response** (201 Created):
```  ]
}
```

json
{
  "plan_id": "plan_001",
  "title": "Add authentication to API",
  "status": "pending",
  "created_at": "2026-03-02T10:30:00Z",
  "total_tasks": 4,
  "completed_tasks": 0,
  "progress_percent": 0.0,
  "waves": [
    {
      "wave_id": "wave_1",
      "milestone": "Research and Planning",
      "status": "pending",
      "order": 1,
      "tasks": [
        {
          "task_id": "task_1",
          "status": "pending",
          "priority": 8
        }
      ]
    }
  ]
}
```

### 1.2 List Plans

**Endpoint**: `GET /api/plans`

**Query Parameters**:
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| status | string | Filter by status | all |
| limit | int | Max results | 50 |
| offset | int | Pagination offset | 0 |

**Response** (200 OK):
```json
{
  "plans": [
    {
      "plan_id": "plan_001",
      "title": "Add authentication to API",
      "status": "active",
      "created_at": "2026-03-02T10:30:00Z",
      "progress_percent": 45.0,
      "total_tasks": 4,
      "completed_tasks": 2
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### 1.3 Get Plan Details

**Endpoint**: `GET /api/plans/{plan_id}`

**Response** (200 OK):
```json
{
  "plan_id": "plan_001",
  "title": "Add authentication to API",
  "prompt": "Add JWT authentication to our Express API...",
  "created_by": "user_001",
  "created_at": "2026-03-02T10:30:00Z",
  "updated_at": "2026-03-02T11:45:00Z",
  "status": "active",
  "total_tasks": 4,
  "completed_tasks": 2,
  "progress_percent": 45.0,
  "waves": [
    {
      "wave_id": "wave_1",
      "milestone": "Research and Planning",
      "description": "Research JWT libraries",
      "order": 1,
      "status": "completed",
      "tasks": [
        {
          "task_id": "task_1",
          "agent_type": "explorer",
          "description": "Find existing auth code",
          "status": "completed",
          "result": {
            "files": ["src/auth/middleware.ts", "lib/jwt.ts"]
          }
        }
      ]
    },
    {
      "wave_id": "wave_2",
      "milestone": "Implementation",
      "description": "Implement auth endpoints",
      "order": 2,
      "status": "running",
      "tasks": [
        {
          "task_id": "task_3",
          "agent_type": "builder",
          "description": "Create auth routes",
          "status": "running",
          "assigned_agent": "agent_builder_1"
        }
      ]
    }
  ]
}
```

### 1.4 Update Plan

**Endpoint**: `PUT /api/plans/{plan_id}`

**Request Body**:
```json
{
  "title": "Updated title",
  "waves": [
    {
      "wave_id": "wave_1",
      "milestone": "Updated milestone",
      "tasks": [
        {
          "task_id": "task_1",
          "priority": 10
        }
      ]
    }
  ]
}
```

### 1.5 Delete Plan

**Endpoint**: `DELETE /api/plans/{plan_id}`

**Response**: 204 No Content

### 1.6 Start Plan Execution

**Endpoint**: `POST /api/plans/{plan_id}/start`

**Response** (200 OK):
```json
{
  "plan_id": "plan_001",
  "status": "active",
  "started_at": "2026-03-02T12:00:00Z"
}
```

### 1.7 Pause Plan

**Endpoint**: `POST /api/plans/{plan_id}/pause`

**Response** (200 OK):
```json
{
  "plan_id": "plan_001",
  "status": "paused",
  "paused_at": "2026-03-02T12:30:00Z"
}
```

### 1.8 Resume Plan

**Endpoint**: `POST /api/plans/{plan_id}/resume`

**Response** (200 OK):
```json
{
  "plan_id": "plan_001",
  "status": "active",
  "resumed_at": "2026-03-02T12:35:00Z"
}
```

### 1.9 Cancel Plan

**Endpoint**: `POST /api/plans/{plan_id}/cancel`

**Response** (200 OK):
```json
{
  "plan_id": "plan_001",
  "status": "cancelled",
  "cancelled_at": "2026-03-02T12:40:00Z"
}
```

### 1.10 Get Plan Status

**Endpoint**: `GET /api/plans/{plan_id}/status`

**Response** (200 OK):
```json
{
  "plan_id": "plan_001",
  "status": "active",
  "current_wave": "wave_2",
  "completed_tasks": 2,
  "running_tasks": 1,
  "pending_tasks": 1,
  "failed_tasks": 0,
  "progress_percent": 45.0,
  "started_at": "2026-03-02T12:00:00Z",
  "estimated_completion": "2026-03-02T14:00:00Z"
}
```

### 1.11 Get Plan Waves

**Endpoint**: `GET /api/plans/{plan_id}/waves`

**Response** (200 OK):
```json
{
  "plan_id": "plan_001",
  "waves": [
    {
      "wave_id": "wave_1",
      "milestone": "Research and Planning",
      "status": "completed",
      "order": 1,
      "tasks_count": 2,
      "completed_count": 2
    }
  ]
}
```

### 1.12 Get Wave Tasks

**Endpoint**: `GET /api/plans/{plan_id}/waves/{wave_id}/tasks`

**Response** (200 OK):
```json
{
  "wave_id": "wave_1",
  "tasks": [
    {
      "task_id": "task_1",
      "agent_type": "explorer",
      "description": "Find existing auth code",
      "status": "completed",
      "result": {...}
    }
  ]
}
```

---

## 2. Model Relay API

### 2.1 List Available Models

**Endpoint**: `GET /api/models`

**Response** (200 OK):
```json
{
  "models": [
    {
      "model_id": "qwen2.5-coder-7b",
      "provider": "lmstudio",
      "display_name": "Qwen 2.5 Coder 7B",
      "capabilities": ["coding", "fast"],
      "context_window": 32768,
      "latency_hint": "fast",
      "privacy_level": "local",
      "status": "available"
    },
    {
      "model_id": "gpt-4o",
      "provider": "openai",
      "display_name": "GPT-4o",
      "capabilities": ["reasoning", "vision"],
      "context_window": 128000,
      "latency_hint": "medium",
      "privacy_level": "public",
      "cost_per_1k_tokens": 0.005,
      "status": "available"
    }
  ]
}
```

### 2.2 Get Model Details

**Endpoint**: `GET /api/models/{model_id}`

**Response** (200 OK):
```json
{
  "model_id": "qwen2.5-coder-7b",
  "provider": "lmstudio",
  "display_name": "Qwen 2.5 Coder 7B",
  "capabilities": ["coding", "fast"],
  "context_window": 32768,
  "latency_hint": "fast",
  "privacy_level": "local",
  "memory_requirements_gb": 8,
  "status": "available",
  "current_load": 0.3
}
```

### 2.3 Select Model for Task

**Endpoint**: `POST /api/models/select`

**Request Body**:
```json
{
  "task_type": "coding",
  "context": {
    "prompt": "Write a function to authenticate users",
    "required_capabilities": ["coding"]
  }
}
```

**Response** (200 OK):
```json
{
  "model_id": "qwen2.5-coder-7b",
  "provider": "lmstudio",
  "endpoint": "http://localhost:1234/v1/chat/completions",
  "reasoning": "Local model selected due to local_first policy and matching capabilities",
  "confidence": 0.9,
  "latency_estimate": "fast"
}
```

### 2.4 Get Routing Policy

**Endpoint**: `GET /api/models/policy`

**Response** (200 OK):
```json
{
  "local_first": true,
  "privacy_weight": 1.0,
  "latency_weight": 0.5,
  "cost_weight": 0.3,
  "max_cost_per_1k_tokens": 0.0,
  "preferred_providers": ["lmstudio", "ollama"]
}
```

### 2.5 Update Routing Policy

**Endpoint**: `PUT /api/models/policy`

**Request Body**:
```json
{
  "local_first": false,
  "privacy_weight": 0.5,
  "latency_weight": 1.0,
  "cost_weight": 0.8,
  "max_cost_per_1k_tokens": 0.01
}
```

**Response** (200 OK):
```json
{
  "status": "updated",
  "policy": {
    "local_first": false,
    "privacy_weight": 0.5,
    "latency_weight": 1.0,
    "cost_weight": 0.8
  }
}
```

### 2.6 Reload Model Catalog

**Endpoint**: `POST /api/models/reload`

**Response** (200 OK):
```json
{
  "status": "reloaded",
  "models_loaded": 5,
  "loaded_at": "2026-03-02T12:00:00Z"
}
```

---

## 3. Sandbox API

### 3.1 Execute in Sandbox

**Endpoint**: `POST /api/sandbox/execute`

**Request Body**:
```json
{
  "code": "def hello():\n    return 'Hello, World!'",
  "language": "python",
  "timeout": 30,
  "sandbox_type": "in_process"
}
```

**Response** (200 OK):
```json
{
  "execution_id": "exec_001",
  "status": "completed",
  "result": "Hello, World!",
  "execution_time_ms": 15,
  "memory_used_mb": 2.5
}
```

**Error Response** (400 Bad Request):
```json
{
  "error": "ExecutionError",
  "message": "Code execution timeout after 30s",
  "execution_id": "exec_001"
}
```

### 3.2 Sandbox Status

**Endpoint**: `GET /api/sandbox/status`

**Response** (200 OK):
```json
{
  "in_process": {
    "available": true,
    "current_load": 0.2,
    "max_concurrent": 5,
    "queue_length": 0
  },
  "external": {
    "available": true,
    "plugins_loaded": 3,
    "isolation": "process"
  }
}
```

### 3.3 Sandbox Audit Log

**Endpoint**: `GET /api/sandbox/audit`

**Query Parameters**:
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| limit | int | Max results | 100 |
| execution_id | string | Filter by execution | - |

**Response** (200 OK):
```json
{
  "audit_entries": [
    {
      "timestamp": "2026-03-02T12:00:00Z",
      "execution_id": "exec_001",
      "operation": "execute_code",
      "sandbox_type": "in_process",
      "status": "success",
      "duration_ms": 15
    }
  ],
  "total": 1
}
```

---

## 4. Code Search API

### 4.1 Search Code

**Endpoint**: `GET /api/search`

**Query Parameters**:
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| q | string | Search query | required |
| limit | int | Max results | 10 |
| language | string | Filter by language | - |
| backend | string | Search backend | cocoindex |

**Response** (200 OK):
```json
{
  "query": "JWT authentication",
  "backend": "cocoindex",
  "results": [
    {
      "file_path": "src/auth/jwt.ts",
      "language": "typescript",
      "content": "export function verifyToken(token: string) {...",
      "line_start": 42,
      "line_end": 58,
      "score": 0.95,
      "context_before": "...",
      "context_after": "..."
    }
  ],
  "total": 15,
  "search_time_ms": 125
}
```

### 4.2 Index Project

**Endpoint**: `POST /api/search/index`

**Request Body**:
```json
{
  "project_path": "/path/to/project",
  "backend": "cocoindex"
}
```

**Response** (200 OK):
```json
{
  "status": "indexing",
  "project_path": "/path/to/project",
  "backend": "cocoindex",
  "estimated_time_seconds": 120
}
```

### 4.3 Search Backend Status

**Endpoint**: `GET /api/search/status`

**Response** (200 OK):
```json
{
  "backends": {
    "cocoindex": {
      "available": true,
      "version": "0.1.6",
      "indexed_projects": ["project_001", "project_002"]
    }
  },
  "default_backend": "cocoindex"
}
```

---

## 5. Error Responses

### 5.1 Standard Error Format

```json
{
  "error": "ErrorType",
  "message": "Human-readable message",
  "code": 400,
  "details": {}
}
```

### 5.2 Common Error Codes

| Code | Error | Description |
|------|-------|-------------|
| 400 | BadRequest | Invalid request format |
| 401 | Unauthorized | Missing/invalid auth |
| 403 | Forbidden | Permission denied |
| 404 | NotFound | Resource not found |
| 409 | Conflict | Resource conflict |
| 422 | ValidationError | Validation failed |
| 429 | RateLimited | Too many requests |
| 500 | InternalError | Server error |
| 503 | ServiceUnavailable | Service temporarily unavailable |

---

*Document Version: 1.0*
*Phase: 1*
*Last Updated: 2026-03-02*
