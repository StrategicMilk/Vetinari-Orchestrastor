# Vetinari Ponder API Contracts

This document defines key API endpoints introduced for plan-wide ponder orchestration and cloud provider health visibility.

## POST /api/ponder/plan/<plan_id>

Triggers a plan-wide ponder pass for the given plan.

**Request:**
```bash
POST /api/ponder/plan/plan_123
Content-Type: application/json
{}
```

**Response (success):**
```json
HTTP/1.1 200 OK
{
  "plan_id": "plan_123",
  "total_subtasks": 5,
  "updated_subtasks": 3,
  "errors": [],
  "success": true
}
```

**Response (error):**
```json
HTTP/1.1 400 Bad Request
{
  "error": "Plan plan_123 not found",
  "success": false
}
```

## GET /api/ponder/plan/<plan_id>

Fetch per-subtask ponder audit data for the plan.

**Response (success):**
```json
HTTP/1.1 200 OK
{
  "plan_id": "plan_123",
  "total_subtasks": 5,
  "subtasks_with_ponder": 5,
  "subtasks": [
    {
      "subtask_id": "st_0001",
      "description": "Write Python function",
      "agent_type": "builder",
      "ponder_ranking": [
        {"rank": 1, "model_id": "qwen2.5-coder-14b", "total_score": 0.95},
        {"rank": 2, "model_id": "claude:3.5-sonnet", "total_score": 0.88}
      ],
      "ponder_scores": {"qwen2.5-coder-14b": 0.95, "claude:3.5-sonnet": 0.88},
      "ponder_used": true
    }
  ]
}
```

## GET /api/ponder/health

Returns provider health status and token presence.

**Response (success):**
```json
HTTP/1.1 200 OK
{
  "enable_model_search": true,
  "cloud_weight": 0.2,
  "providers": {
    "huggingface_inference": {
      "available": true,
      "name": "HuggingFace Inference API",
      "has_token": true
    },
    "replicate": {
      "available": true,
      "name": "Replicate",
      "has_token": true
    },
    "claude": {
      "available": true,
      "name": "Claude (Anthropic)",
      "has_token": true
    },
    "gemini": {
      "available": false,
      "name": "Gemini (Google)",
      "has_token": false
    }
  }
}
```

## POST /api/ponder/choose-model

Get top-N model rankings for a specific task description.

**Request:**
```bash
POST /api/ponder/choose-model
Content-Type: application/json
{
  "task_description": "Write Python code for data processing",
  "top_n": 3,
  "template_version": "v1"
}
```

**Response (success):**
```json
HTTP/1.1 200 OK
{
  "task_id": "ponder_20240101_120000",
  "task_description": "Write Python code for data processing",
  "rankings": [
    {
      "rank": 1,
      "model_id": "qwen2.5-coder-14b-instruct",
      "model_name": "Qwen2.5 Coder 14B",
      "total_score": 0.95,
      "capability_score": 0.9,
      "context_score": 1.0,
      "memory_score": 1.0,
      "heuristic_score": 0.8,
      "policy_penalty": 0,
      "reasoning": "capability match: 0.90, context fit: 1.00"
    }
  ],
  "timestamp": "2024-01-01T12:00:00",
  "phase": "result"
}
```

## GET /api/ponder/templates

Get available Ponder templates.

**Response (success):**
```json
HTTP/1.1 200 OK
{
  "templates": [
    {
      "template_id": "ponder_rank_default",
      "name": "Default Model Ranking",
      "description": "Standard model selection for subtask"
    }
  ],
  "total": 5,
  "version": "v1"
}
```

## Security Notes
- Tokens are never returned in API responses
- Tokens are sourced exclusively from environment variables
- Use proper RBAC for plan-level endpoints; restrict to admins where necessary
