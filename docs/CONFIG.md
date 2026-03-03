# Vetinari Configuration Reference

Complete guide to configuring Vetinari including Plan Mode, Ponder, and memory settings.

---

## Environment Variables

### Plan Mode Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| PLAN_MODE_ENABLE | true | Enable plan-first orchestration |
| PLAN_MODE_DEFAULT | true | Default mode for all tasks |
| DRY_RUN_ENABLED | false | Generate plans without execution |
| DRY_RUN_RISK_THRESHOLD | 0.25 | Auto-approval risk threshold (0.0-1.0) |
| PLAN_DEPTH_CAP | 16 | Maximum subtask decomposition depth |
| PLAN_MAX_CANDIDATES | 3 | Maximum plan variants to generate |
| PLAN_ADMIN_TOKEN | (none) | Admin token for plan API endpoints |
| PLAN_MEMORY_DB_PATH | ./vetinari_memory.db | SQLite database path |
| PLAN_USE_JSON_FALLBACK | false | Use JSON if SQLite unavailable |
| PLAN_RETENTION_DAYS | 90 | Days to keep plan history |

### Ponder Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| ENABLE_PONDER_MODEL_SEARCH | true | Enable cloud model search |
| PONDER_CLOUD_WEIGHT | 0.20 | Cloud model influence (0.0-1.0) |

### Cloud Provider API Keys

| Variable | Provider | Description |
|----------|----------|-------------|
| CLAUDE_API_KEY | Anthropic Claude | API key for Claude models |
| GEMINI_API_KEY | Google Gemini | API key for Gemini models |
| HF_HUB_TOKEN | HuggingFace | Token for HF Inference API |
| REPLICATE_API_TOKEN | Replicate | Token for Replicate API |

### LM Studio Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| LM_STUDIO_HOST | http://localhost:1234 | LM Studio server URL |
| LM_STUDIO_API_TOKEN | (none) | API token if required |

---

## Configuration Examples

### Minimal Configuration (.env)

```bash
# Plan Mode (defaults enabled)
PLAN_MODE_ENABLE=true
PLAN_MODE_DEFAULT=true

# Admin token for plan endpoints
PLAN_ADMIN_TOKEN=my-secret-token

# Cloud providers (optional)
CLAUDE_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

### Development Configuration

```bash
# Plan Mode
PLAN_MODE_ENABLE=true
PLAN_MODE_DEFAULT=true
DRY_RUN_ENABLED=true
DRY_RUN_RISK_THRESHOLD=0.5

# Memory store (JSON for quick prototyping)
PLAN_USE_JSON_FALLBACK=true
PLAN_MEMORY_DB_PATH=./dev_memory.json

# Admin
PLAN_ADMIN_TOKEN=dev-token

# Logging
DEBUG=true
LOG_LEVEL=DEBUG
```

### Production Configuration

```bash
# Plan Mode
PLAN_MODE_ENABLE=true
PLAN_MODE_DEFAULT=true
DRY_RUN_ENABLED=false
DRY_RUN_RISK_THRESHOLD=0.25
PLAN_DEPTH_CAP=16
PLAN_MAX_CANDIDATES=3
PLAN_RETENTION_DAYS=180

# Memory store (SQLite)
PLAN_MEMORY_DB_PATH=/var/lib/vetinari/vetinari_memory.db
PLAN_USE_JSON_FALLBACK=false

# Admin (use secure token)
PLAN_ADMIN_TOKEN=<secure-random-token>

# Cloud providers
CLAUDE_API_KEY=<key>
GEMINI_API_KEY=<key>

# LM Studio
LM_STUDIO_HOST=http://internal-lmstudio:1234

# Logging
LOG_LEVEL=INFO
```

---

## Plan Mode

### How It Works

1. **Goal Analysis**: When a task is submitted, Plan Mode generates a plan
2. **Candidate Generation**: Creates 1-3 plan variants with different approaches
3. **Risk Scoring**: Calculates risk based on depth, cost, dependencies
4. **Auto-Approval**: Low-risk plans (score <= 0.25) auto-approved in dry-run
5. **Manual Approval**: High-risk plans require admin approval via API

### Risk Scoring

Risk factors:
- Plan depth (max 16 levels)
- Number of subtasks
- Estimated cost
- Dependency complexity

Risk levels:
- **LOW** (0.0-0.25): Auto-approved
- **MEDIUM** (0.25-0.5): May require approval
- **HIGH** (0.5-0.75): Requires approval
- **CRITICAL** (0.75-1.0): Requires explicit approval

### Dry-Run Mode

When `DRY_RUN_ENABLED=true`:
- Plans generated but not executed
- Auto-approved if risk_score <= DRY_RUN_RISK_THRESHOLD
- Useful for testing and evaluation

---

## Memory Store

### SQLite (Primary)

Location: `./vetinari_memory.db` (or custom path)

Schema:
- `PlanHistory`: Plans, goals, status, risk scores
- `SubtaskMemory`: Subtasks, outcomes, metrics
- `ModelPerformance`: Model success rates, latency

### JSON Fallback

Location: `./vetinari_memory.json`

Used when:
- SQLite unavailable
- `PLAN_USE_JSON_FALLBACK=true`

### Retention

- Default: 90 days
- Configurable via `PLAN_RETENTION_DAYS`
- Automatic pruning on startup

---

## Plan API Endpoints

All endpoints require admin token in `Authorization` header:

```bash
Authorization: Bearer <PLAN_ADMIN_TOKEN>
```

### POST /api/plan/generate

Generate a plan from a goal.

**Request:**
```json
{
  "goal": "Build a web application",
  "constraints": "Use Python and Flask",
  "plan_depth_cap": 16,
  "max_candidates": 3,
  "domain_hint": "coding",
  "dry_run": false,
  "risk_threshold": 0.25
}
```

**Response:**
```json
{
  "success": true,
  "plan_id": "plan_abc123",
  "version": 1,
  "goal": "Build a web application",
  "status": "draft",
  "risk_score": 0.15,
  "risk_level": "low",
  "subtask_count": 5,
  "auto_approved": false,
  "plan_candidates": [...]
}
```

### GET /api/plan/{plan_id}

Get plan details.

### POST /api/plan/{plan_id}/approve

Approve or reject a plan.

**Request:**
```json
{
  "approved": true,
  "approver": "admin",
  "reason": "Looks good"
}
```

### GET /api/plan/{plan_id}/history

Get plan history and subtask outcomes.

### GET /api/plan/status

Get plan mode status and memory statistics.

---

## Domain Templates

### Coding
- Define API surface and data models
- Implement core functionality
- Write unit tests
- Integrate with existing components
- Refactor for clarity

### Data Processing
- Define data schema and sources
- Build data ingestion pipeline
- Implement transformation logic
- Implement data quality checks
- Create deployment and scheduling

### Infra
- Define metrics and observability requirements
- Implement health checks and readiness probes
- Create monitoring dashboards
- Configure alerting rules
- Document runbooks

### Docs
- Outline documentation structure
- Draft main sections
- Add usage examples and code snippets
- Review and get feedback
- Finalize and publish

### AI Experiments
- Define experiment metrics and success criteria
- Design experiment configuration
- Run experiments
- Analyze results
- Document insights and recommendations

### Research
- Gather sources and literature
- Summarize findings
- Compare approaches
- Propose recommendations
- Validate against goals

---

## Security

### Admin Token

Set a secure token for plan API access:
```bash
# Generate secure token
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in environment
export PLAN_ADMIN_TOKEN=<generated-token>
```

### API Requests

All plan API calls must include:
```bash
curl -H "Authorization: Bearer <PLAN_ADMIN_TOKEN>" \
     -H "Content-Type: application/json" \
     ...
```

---

## Troubleshooting

### Plan Mode Not Working

1. Check PLAN_MODE_ENABLE=true
2. Check logs for plan generation errors
3. Verify memory store is accessible

### Memory Store Errors

1. Check PLAN_MEMORY_DB_PATH is writable
2. If SQLite fails, JSON fallback activates automatically
3. Check PLAN_USE_JSON_FALLBACK for fallback behavior

### API 401 Unauthorized

1. Verify PLAN_ADMIN_TOKEN is set
2. Check Authorization header format: `Bearer <token>`
3. Token must match exactly

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [cloud-ponder.md](cloud-ponder.md) - Ponder model selection
- [api-contracts.md](api-contracts.md) - REST API documentation
