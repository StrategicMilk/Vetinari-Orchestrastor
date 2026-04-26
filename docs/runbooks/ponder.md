# Ponder Orchestration Runbook

This runbook documents how to operate, monitor, and troubleshoot the cloud-augmented Ponder flow in Vetinari.

## Overview

Ponder is the mechanism Vetinari uses to select models for subtasks with cloud augmentation. The system supports:
- Plan-wide ponder pass after decomposition
- Per-subtask auditing fields stored in SubtaskTree
- Token-based cloud providers with graceful degradation to local-only mode

## Prerequisites

Environment variables for cloud providers:
```bash
# Required for cloud providers (optional - system works without)
export HF_HUB_TOKEN=your_huggingface_token
export REPLICATE_API_TOKEN=your_replicate_token
export CLAUDE_API_KEY=your_anthropic_key
export GEMINI_API_KEY=your_google_key

# Ponder configuration (optional)
export ENABLE_PONDER_MODEL_SEARCH=true
export PONDER_CLOUD_WEIGHT=0.20
```

## Operations

### Trigger Plan-wide Ponder

```bash
curl -X POST http://localhost:5000/api/ponder/plan/plan_123
```

Response:
```json
{
  "plan_id": "plan_123",
  "total_subtasks": 5,
  "updated_subtasks": 5,
  "errors": [],
  "success": true
}
```

### Inspect Ponder Data

```bash
curl http://localhost:5000/api/ponder/plan/plan_123
```

### Health Check

```bash
curl http://localhost:5000/api/ponder/health
```

Response:
```json
{
  "enable_model_search": true,
  "cloud_weight": 0.2,
  "providers": {
    "huggingface_inference": {"available": true, "has_token": true},
    "replicate": {"available": true, "has_token": true},
    "claude": {"available": true, "has_token": true},
    "gemini": {"available": false, "has_token": false}
  }
}
```

## Operational Guidelines

1. **Feature Flags**: Use environment variables to enable/disable cloud-ponder
   - `ENABLE_PONDER_MODEL_SEARCH=true` - Enable cloud model search
   - `PONDER_CLOUD_WEIGHT=0.20` - Cloud signal weight (0.0-0.5)

2. **Monitoring**: Monitor cloud latency and TTL cache hits/misses
   - Check logs for `cloud_latency_ms`, `cloud_calls`, `ponder_duration`

3. **For Long-running Projects**: Prefer asynchronous ponder with plan status endpoint
   - Prevents blocking UI or API calls

4. **Token Rotation**: Regularly rotate API keys via your secret management pipeline
   - Never store keys in code or logs

## Troubleshooting

### Ponder Results Are Empty
- Verify token presence: `GET /api/ponder/health`
- Check provider health status
- Review subtask tree for `ponder_ranking`/`ponder_scores` fields

### Cloud Signals Stale
- Check per-provider cache TTL (default: 60 seconds)
- Clear cache: Remove files from `~/.cache/vetinari/ponder/`

### Migration Issues
- Run migration script: `python vetinari/migrations/upgrade_subtask_schema_v1_to_v2.py`
- Verify older subtask JSONs load with defaults for ponder fields

### High Latency
- Consider disabling cloud providers for time-sensitive tasks
- Adjust `PONDER_CLOUD_WEIGHT` to reduce cloud influence

## Rollback Procedure

If cloud-ponder causes issues:

1. Disable cloud augmentation:
   ```bash
   export ENABLE_PONDER_MODEL_SEARCH=false
   ```

2. Restart Vetinari

3. Verify local-only mode works:
   ```bash
   curl http://localhost:5000/api/ponder/health
   # Should show all providers as "available": false
   ```

4. Ponder audit history remains available for debugging

## Performance Tuning

- **Cache TTL**: Adjust in `vetinari/model_search.py` (default: 60s)
- **Cloud Weight**: Tune `PONDER_CLOUD_WEIGHT` (default: 0.20)
- **Provider Priority**: Modify provider order in `CLOUD_PROVIDERS` dict
