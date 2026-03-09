# Vetinari Database Architecture

## Overview

Vetinari uses **SQLite** for all persistent storage, chosen for its zero-configuration,
serverless design that runs anywhere Python does. Each subsystem owns its own database
file to avoid cross-module coupling and simplify independent schema evolution.

All databases live under `VETINARI_DATA_DIR` (default: `~/.vetinari/`).

## Database Inventory

| Database File | Module | Purpose | Tables |
|---|---|---|---|
| `vetinari_memory.db` | `memory/__init__.py`, `memory/enhanced.py` | Plan history, subtasks, model performance, semantic memory | `PlanHistory`, `SubtaskMemory`, `ModelPerformance`, `memory_entries`, `memory_fts` |
| `vetinari_episodes.db` | `learning/episode_memory.py` | Episodic agent execution records with similarity search | `episodes` |
| `vetinari_quality_scores.db` | `learning/quality_scorer.py` | Quality assessment scores from LLM-as-judge and heuristics | `quality_scores` |
| `vetinari_benchmark_metrics.db` | `benchmarks/runner.py` | Benchmark run history and per-case results | `benchmark_runs`, `benchmark_results` |
| `.vetinari/mnemoria/memories.db` | `memory/oc_memory.py` | OC-mnemoria shared agent memory with tombstone support | `memories` |

## Configuration

All paths are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `VETINARI_DATA_DIR` | `~/.vetinari` | Root directory for all database files |
| `PLAN_MEMORY_DB_PATH` | `$VETINARI_DATA_DIR/vetinari_memory.db` | Plan memory database |
| `VETINARI_MEMORY_DB` | `$VETINARI_DATA_DIR/vetinari_memory.db` | Semantic memory database |
| `VETINARI_EPISODE_DB` | `$VETINARI_DATA_DIR/vetinari_episodes.db` | Episode memory database |
| `VETINARI_QUALITY_DB` | `$VETINARI_DATA_DIR/vetinari_quality_scores.db` | Quality scores database |
| `VETINARI_MAX_EPISODES` | `10000` | Maximum episodes before eviction |
| `PLAN_RETENTION_DAYS` | `90` | Days before old plans are pruned |
| `PLAN_USE_JSON_FALLBACK` | `false` | Use JSON file instead of SQLite |

## SQLite Best Practices (Enforced)

Every database connection in Vetinari applies these PRAGMAs:

```sql
PRAGMA journal_mode=WAL;        -- Write-Ahead Logging for concurrent reads
PRAGMA foreign_keys=ON;         -- Enforce referential integrity
PRAGMA busy_timeout=5000;       -- Wait 5s on lock contention before failing
PRAGMA synchronous=NORMAL;      -- Safe with WAL; balances durability and speed
```

### Why These Settings

- **WAL mode**: Allows concurrent readers while a writer is active. Critical for
  multi-agent workloads where the orchestrator, feedback loop, and dashboard may
  all access the same database simultaneously.

- **Foreign keys**: SQLite disables FK enforcement by default. Enabling it ensures
  cascading deletes work (e.g., pruning plans removes their subtasks) and prevents
  orphaned rows.

- **Busy timeout**: Without this, a concurrent write attempt immediately raises
  `sqlite3.OperationalError: database is locked`. The 5-second timeout allows
  short contention windows to resolve naturally.

- **synchronous=NORMAL**: In WAL mode, `NORMAL` provides crash safety equivalent
  to `FULL` in rollback mode, with significantly better write throughput. Data loss
  only possible during OS-level crash (not application crash).

## Thread Safety

All database classes use `threading.Lock()` or `threading.RLock()` to serialize
access to `sqlite3.Connection` objects:

| Module | Lock Type | Scope |
|---|---|---|
| `OcMemoryStore` | `threading.Lock()` | All read and write operations |
| `SemanticMemoryStore` | `threading.Lock()` | All read and write operations |
| `EpisodeMemory` | `threading.RLock()` | Record, recall, and index operations |
| `MemoryStore` | None (single-threaded) | Uses `check_same_thread=False` |
| `QualityScorer` | None (context-manager connections) | Uses `with sqlite3.connect()` |
| `MetricStore` | None (context-manager connections) | Uses `with sqlite3.connect()` |

Modules using context-manager connections (`with sqlite3.connect(...)`) are
inherently safe for multi-threaded use because each call creates a new connection.
Modules with persistent `self._conn` connections require explicit locking.

## Connection Lifecycle

All database classes follow this pattern:

1. **Directory creation**: `Path(db_path).parent.mkdir(parents=True, exist_ok=True)`
2. **Connection**: `sqlite3.connect(path, check_same_thread=False)` for persistent,
   or `with sqlite3.connect(path)` for transactional
3. **PRAGMA setup**: WAL, FK, busy_timeout, synchronous (applied once at init)
4. **Schema creation**: `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS`
5. **Cleanup**: `close()` method + `__del__` + context manager (`__enter__`/`__exit__`)

## Schema Reference

### vetinari_memory.db

#### PlanHistory
Stores complete plan records with goals, status tracking, and risk assessment.

```sql
CREATE TABLE PlanHistory (
    plan_id              TEXT PRIMARY KEY,
    plan_version         INTEGER DEFAULT 1,
    goal                 TEXT NOT NULL,
    created_at           DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at           DATETIME DEFAULT CURRENT_TIMESTAMP,
    status               TEXT DEFAULT 'draft',      -- draft|approved|executing|completed|failed
    plan_json            TEXT,                       -- Serialized plan structure
    plan_explanation_json TEXT,                      -- JSON PlanExplanation
    chosen_plan_id       TEXT,                       -- Selected plan variant
    plan_justification   TEXT,                       -- Why this plan was chosen
    risk_score           REAL DEFAULT 0.0,           -- 0.0 (safe) to 1.0 (risky)
    dry_run              INTEGER DEFAULT 0,          -- Boolean: simulated run
    auto_approved        INTEGER DEFAULT 0           -- Boolean: bypassed approval
);
-- Indexes: idx_plan_created_at, idx_plan_status
```

#### SubtaskMemory
Stores hierarchical subtask decomposition with execution outcomes.

```sql
CREATE TABLE SubtaskMemory (
    subtask_id             TEXT PRIMARY KEY,
    plan_id                TEXT NOT NULL,             -- FK -> PlanHistory
    parent_subtask_id      TEXT,                      -- Self-referencing hierarchy
    description            TEXT NOT NULL,
    depth                  INTEGER DEFAULT 0,         -- Tree depth level
    status                 TEXT DEFAULT 'pending',    -- pending|running|completed|failed
    assigned_model_id      TEXT,                      -- Which model executed this
    outcome                TEXT,                      -- success|failed|skipped
    duration_seconds       REAL,
    cost_estimate          REAL,
    rationale              TEXT,                      -- Why this subtask exists
    subtask_explanation_json TEXT,                    -- JSON SubtaskExplanation
    domain                 TEXT,                      -- coding|research|testing|etc
    created_at             DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at             DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES PlanHistory(plan_id)
);
-- Indexes: idx_subtask_plan_id, idx_subtask_domain, idx_subtask_status
```

#### ModelPerformance
Running averages of model success rates and latencies per task type.

```sql
CREATE TABLE ModelPerformance (
    model_id     TEXT NOT NULL,
    task_type    TEXT NOT NULL,
    success_rate REAL DEFAULT 0.0,
    avg_latency  REAL DEFAULT 0.0,
    total_uses   INTEGER DEFAULT 0,
    last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, task_type)
);
```

#### memory_entries (SemanticMemoryStore)
Long-term semantic memory with optional vector embeddings and FTS5 full-text search.

```sql
CREATE TABLE memory_entries (
    entry_id       TEXT PRIMARY KEY,
    content        TEXT NOT NULL,
    memory_type    TEXT NOT NULL,      -- plan|task|decision|knowledge|code|conversation|result|context
    metadata_json  TEXT,
    tags_json      TEXT,
    provenance     TEXT,
    embedding_blob BLOB,              -- JSON-serialized float vector
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    access_count   INTEGER DEFAULT 0,
    last_accessed  TEXT
);
-- FTS5 virtual table: memory_fts (synced via triggers)
-- Indexes: idx_memory_type, idx_created_at, idx_tags
```

### vetinari_episodes.db

#### episodes
Stores structured records of past agent executions for experience-based learning.

```sql
CREATE TABLE episodes (
    episode_id    TEXT PRIMARY KEY,
    timestamp     TEXT,
    task_summary  TEXT,                -- Truncated to 300 chars
    agent_type    TEXT,                -- PLANNER|RESEARCHER|BUILDER|etc
    task_type     TEXT,                -- coding|research|analysis|etc
    output_summary TEXT,               -- Truncated to 500 chars
    quality_score REAL,                -- 0.0 - 1.0
    success       INTEGER,             -- Boolean
    model_id      TEXT,
    embedding     TEXT,                -- JSON float vector for similarity search
    metadata      TEXT,                -- JSON blob
    created_at    REAL DEFAULT (unixepoch()),
    importance    REAL DEFAULT 0.5     -- Eviction priority
);
-- Indexes: idx_ep_type, idx_ep_score, idx_ep_agent_type_success,
--          idx_ep_importance, idx_ep_created_at
```

**Eviction policy**: When episode count exceeds `VETINARI_MAX_EPISODES` (default 10,000),
the bottom 10% by importance score are deleted.

### vetinari_quality_scores.db

#### quality_scores
Records from LLM-as-judge and heuristic quality evaluation.

```sql
CREATE TABLE quality_scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id       TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    task_type     TEXT NOT NULL,
    overall_score REAL NOT NULL,       -- 0.0 - 1.0
    correctness   REAL,
    completeness  REAL,
    efficiency    REAL,
    style         REAL,
    dimensions    TEXT,                 -- JSON: per-dimension scores
    issues        TEXT,                 -- JSON: list of identified issues
    method        TEXT,                 -- "llm"|"heuristic"|"hybrid"|"llm+benchmark"
    timestamp     TEXT,
    created_at    REAL DEFAULT (unixepoch())
);
-- Indexes: idx_qs_model, idx_qs_task_type, idx_qs_created_at
```

### vetinari_benchmark_metrics.db

#### benchmark_runs
Aggregated results for benchmark suite executions.

```sql
CREATE TABLE benchmark_runs (
    run_id              TEXT PRIMARY KEY,
    suite_name          TEXT NOT NULL,
    layer               TEXT NOT NULL,    -- AGENT|ORCHESTRATION|PIPELINE
    tier                TEXT NOT NULL,    -- fast|medium|slow
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    total_cases         INTEGER DEFAULT 0,
    passed_cases        INTEGER DEFAULT 0,
    pass_at_1           REAL DEFAULT 0.0,
    pass_k              REAL DEFAULT 0.0,
    avg_score           REAL DEFAULT 0.0,
    avg_latency         REAL DEFAULT 0.0,
    total_tokens        INTEGER DEFAULT 0,
    total_cost          REAL DEFAULT 0.0,
    error_recovery_rate REAL DEFAULT 0.0,
    metadata            TEXT DEFAULT '{}'
);
-- Index: idx_runs_suite
```

#### benchmark_results
Individual case results within a benchmark run.

```sql
CREATE TABLE benchmark_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,         -- FK -> benchmark_runs
    case_id        TEXT NOT NULL,
    suite_name     TEXT NOT NULL,
    passed         INTEGER NOT NULL,
    score          REAL NOT NULL,
    latency_ms     REAL DEFAULT 0.0,
    tokens         INTEGER DEFAULT 0,
    cost_usd       REAL DEFAULT 0.0,
    error_recovery INTEGER DEFAULT 0,
    error          TEXT,
    output         TEXT,
    timestamp      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
);
-- Index: idx_results_run
```

### .vetinari/mnemoria/memories.db

#### memories
OC-mnemoria style shared agent memory with soft-delete tombstones.

```sql
CREATE TABLE memories (
    id             TEXT PRIMARY KEY,
    agent          TEXT NOT NULL,
    entry_type     TEXT NOT NULL,
    content        TEXT NOT NULL,
    summary        TEXT NOT NULL,
    timestamp      INTEGER NOT NULL,     -- Unix timestamp in milliseconds
    provenance     TEXT NOT NULL,
    source_backends TEXT NOT NULL,        -- JSON array of backend names
    forgotten      INTEGER DEFAULT 0,    -- Tombstone flag
    content_hash   TEXT NOT NULL,         -- SHA-256 for deduplication
    created_at     TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Indexes: idx_memories_agent, idx_memories_type, idx_memories_timestamp,
--          idx_memories_hash, idx_memories_forgotten
```

**Soft delete**: Entries are marked `forgotten=1` rather than deleted. The `compact()`
method permanently removes tombstoned entries and optionally prunes by age.

## Data Retention

| Database | Policy | Mechanism |
|---|---|---|
| Plans | 90 days (configurable) | `MemoryStore.prune_old_plans()` |
| Semantic memory | 90 days, access_count < 3 | `SemanticMemoryStore.prune()` |
| Episodes | Max 10,000 entries | Importance-based eviction in `EpisodeMemory._evict()` |
| Quality scores | Unbounded | Manual cleanup only |
| Benchmarks | Unbounded | Manual cleanup only |
| OC memories | Manual compact | `OcMemoryStore.compact(max_age_days=N)` |

## Backup and Recovery

All databases use WAL mode, which produces `*.db-wal` and `*.db-shm` companion files.
When backing up:

```bash
# Safe backup (copies all WAL data into main db first)
sqlite3 ~/.vetinari/vetinari_memory.db ".backup backup_memory.db"

# Or use the export API
python -c "
from vetinari.memory.oc_memory import OcMemoryStore
store = OcMemoryStore()
store.export('memories_backup.json')
"
```

## Migration Strategy

Vetinari uses **additive-only schema evolution**:

- All tables use `CREATE TABLE IF NOT EXISTS`
- All indexes use `CREATE INDEX IF NOT EXISTS`
- New columns are added via `ALTER TABLE ... ADD COLUMN` in migration scripts
- Schema changes are backward-compatible (new columns have defaults)

Migration scripts live in `vetinari/migrations/`:

| Script | Purpose |
|---|---|
| `schema.sql` | Reference schema for `vetinari_memory.db` |
| `upgrade_subtask_schema.py` | Add ponder auditing fields to subtask JSON files |

## Troubleshooting

### "database is locked"
- **Cause**: Multiple processes writing simultaneously without WAL mode
- **Fix**: All connections now use `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`

### Database in wrong location
- **Cause**: Older versions used CWD-relative paths (e.g., `./vetinari_episodes.db`)
- **Fix**: All databases now default to `~/.vetinari/`. Move old files:
  ```bash
  mv ./vetinari_episodes.db ~/.vetinari/
  mv ./vetinari_quality_scores.db ~/.vetinari/
  mv ./vetinari_memory.db ~/.vetinari/
  ```

### Corrupt database
- **Cause**: Unclean shutdown, disk full, or hardware failure
- **Fix**: SQLite's `PRAGMA integrity_check` and WAL checkpointing:
  ```bash
  sqlite3 ~/.vetinari/vetinari_memory.db "PRAGMA integrity_check;"
  sqlite3 ~/.vetinari/vetinari_memory.db "PRAGMA wal_checkpoint(TRUNCATE);"
  ```

### Missing indexes (slow queries)
- **Cause**: Database created before index additions
- **Fix**: Re-run the application; all indexes use `CREATE INDEX IF NOT EXISTS`
