# Phase 1 Sample Plan Representation

## Overview

This document shows a sample plan representation for testing the planning engine.

---

## Sample Plan: Add Authentication to API

### Plan Creation Request

```json
{
  "title": "Add JWT Authentication to API",
  "prompt": "Add JWT authentication to our Express.js API with login, logout, and register endpoints. Include password hashing with bcrypt and token refresh functionality.",
  "waves": [
    {
      "milestone": "Research and Analysis",
      "description": "Research JWT libraries and analyze existing codebase",
      "order": 1,
      "tasks": [
        {
          "task_id": "task_1",
          "agent_type": "explorer",
          "description": "Find existing auth code",
          "prompt": "Search the codebase for existing authentication code, middleware, and user models. List all relevant files.",
          "dependencies": [],
          "priority": 8
        },
        {
          "task_id": "task_2",
          "agent_type": "librarian",
          "description": "Research JWT best practices",
          "prompt": "Find best practices for JWT authentication in Express.js. Include token refresh strategies, security considerations, and popular libraries.",
          "dependencies": [],
          "priority": 8
        }
      ]
    },
    {
      "milestone": "Design",
      "description": "Design authentication architecture",
      "order": 2,
      "dependencies": ["wave_1"],
      "tasks": [
        {
          "task_id": "task_3",
          "agent_type": "oracle",
          "description": "Design auth architecture",
          "prompt": "Based on research, recommend authentication architecture. Consider: JWT vs sessions, password hashing, token refresh, error handling.",
          "dependencies": ["task_1", "task_2"],
          "priority": 10
        }
      ]
    },
    {
      "milestone": "Implementation",
      "description": "Implement authentication endpoints",
      "order": 3,
      "dependencies": ["wave_2"],
      "tasks": [
        {
          "task_id": "task_4",
          "agent_type": "builder",
          "description": "Create user model",
          "prompt": "Create User model with email, password hash, and timestamps. Use bcrypt for hashing.",
          "dependencies": ["task_3"],
          "priority": 9
        },
        {
          "task_id": "task_5",
          "agent_type": "builder",
          "description": "Create auth routes",
          "prompt": "Create auth routes: POST /register, POST /login, POST /logout, POST /refresh. Use JWT tokens.",
          "dependencies": ["task_3", "task_4"],
          "priority": 10
        },
        {
          "task_id": "task_6",
          "agent_type": "builder",
          "description": "Create auth middleware",
          "prompt": "Create auth middleware to verify JWT tokens on protected routes.",
          "dependencies": ["task_5"],
          "priority": 9
        }
      ]
    },
    {
      "milestone": "Testing",
      "description": "Write tests for authentication",
      "order": 4,
      "dependencies": ["wave_3"],
      "tasks": [
        {
          "task_id": "task_7",
          "agent_type": "builder",
          "description": "Write unit tests",
          "prompt": "Write unit tests for auth routes and middleware. Cover: registration, login, token refresh, protected routes.",
          "dependencies": ["task_6"],
          "priority": 8
        }
      ]
    },
    {
      "milestone": "Review",
      "description": "Review implementation",
      "order": 5,
      "dependencies": ["wave_4"],
      "tasks": [
        {
          "task_id": "task_8",
          "agent_type": "evaluator",
          "description": "Security review",
          "prompt": "Review authentication implementation for security issues. Check: password handling, token validation, error messages, edge cases.",
          "dependencies": ["task_7"],
          "priority": 10
        }
      ]
    }
  ]
}
```

---

## Plan State After Creation

```json
{
  "plan_id": "plan_001",
  "title": "Add JWT Authentication to API",
  "prompt": "Add JWT authentication to our Express.js API with login, logout, and register endpoints. Include password hashing with bcrypt and token refresh functionality.",
  "created_by": "user_001",
  "created_at": "2026-03-02T10:30:00Z",
  "updated_at": "2026-03-02T10:30:00Z",
  "status": "pending",
  "total_tasks": 8,
  "completed_tasks": 0,
  "progress_percent": 0.0,
  "waves": [
    {
      "wave_id": "wave_1",
      "milestone": "Research and Analysis",
      "description": "Research JWT libraries and analyze existing codebase",
      "order": 1,
      "status": "pending",
      "tasks": [
        {
          "task_id": "task_1",
          "agent_type": "explorer",
          "description": "Find existing auth code",
          "status": "pending",
          "dependencies": [],
          "priority": 8
        },
        {
          "task_id": "task_2",
          "agent_type": "librarian",
          "description": "Research JWT best practices",
          "status": "pending",
          "dependencies": [],
          "priority": 8
        }
      ]
    },
    {
      "wave_id": "wave_2",
      "milestone": "Design",
      "description": "Design authentication architecture",
      "order": 2,
      "status": "pending",
      "dependencies": ["wave_1"],
      "tasks": [...]
    },
    {
      "wave_id": "wave_3",
      "milestone": "Implementation",
      "description": "Implement authentication endpoints",
      "order": 3,
      "status": "pending",
      "dependencies": ["wave_2"],
      "tasks": [...]
    },
    {
      "wave_id": "wave_4",
      "milestone": "Testing",
      "description": "Write tests for authentication",
      "order": 4,
      "status": "pending",
      "dependencies": ["wave_3"],
      "tasks": [...]
    },
    {
      "wave_id": "wave_5",
      "milestone": "Review",
      "description": "Review implementation",
      "order": 5,
      "status": "pending",
      "dependencies": ["wave_4"],
      "tasks": [...]
    }
  ]
}
```

---

## Plan State After Wave 1 Completion

```json
{
  "plan_id": "plan_001",
  "status": "active",
  "current_wave": "wave_2",
  "completed_tasks": 2,
  "running_tasks": 1,
  "pending_tasks": 5,
  "progress_percent": 25.0,
  "waves": [
    {
      "wave_id": "wave_1",
      "milestone": "Research and Analysis",
      "status": "completed",
      "completed_tasks": 2,
      "total_tasks": 2,
      "tasks": [
        {
          "task_id": "task_1",
          "agent_type": "explorer",
          "status": "completed",
          "result": {
            "files": [
              "src/models/User.ts",
              "src/middleware/auth.ts"
            ],
            "summary": "Found existing auth middleware and User model"
          },
          "completed_at": "2026-03-02T10:45:00Z"
        },
        {
          "task_id": "task_2",
          "agent_type": "librarian",
          "status": "completed",
          "result": {
            "recommendations": [
              "Use jsonwebtoken library",
              "Implement token refresh with refresh tokens",
              "Use bcrypt for password hashing"
            ],
            "sources": ["https://jwt.io", "expressjs.com"]
          },
          "completed_at": "2026-03-02T10:50:00Z"
        }
      ]
    },
    {
      "wave_id": "wave_2",
      "milestone": "Design",
      "status": "running",
      "completed_tasks": 0,
      "total_tasks": 1,
      "tasks": [
        {
          "task_id": "task_3",
          "agent_type": "oracle",
          "status": "running",
          "assigned_agent": "agent_oracle_1"
        }
      ]
    }
  ]
}
```

---

## API Usage Example

### Create Plan

```bash
curl -X POST http://localhost:5000/api/plans \
  -H "Content-Type: application/json" \
  -d @sample_plan.json
```

### Start Plan

```bash
curl -X POST http://localhost:5000/api/plans/plan_001/start
```

### Get Status

```bash
curl http://localhost:5000/api/plans/plan_001/status
```

---

*Document Version: 1.0*
*Phase: 1*
