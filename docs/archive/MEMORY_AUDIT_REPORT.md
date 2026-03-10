# VETINARI MEMORY SYSTEMS - COMPREHENSIVE DEEP AUDIT

## EXECUTIVE SUMMARY

The Vetinari project contains **FIVE distinct memory systems** with some overlapping responsibilities:

| System | File | Size | References | Status |
|--------|------|------|------------|--------|
| **SharedMemory** | `vetinari/shared_memory.py` | 284 lines | 17 | Legacy/Auxiliary |
| **DualMemoryStore** | `vetinari/memory/dual_memory.py` | 349 lines | 37 | Modern, Growing |
| **MemoryStore** | `vetinari/memory/__init__.py` (lines 50-548) | 500 lines | 112 | **CRITICAL CORE** |
| **EpisodeMemory** | `vetinari/learning/episode_memory.py` | 416 lines | Specialized | Learning-focused |
| **FeedbackLoop** | `vetinari/learning/feedback_loop.py` | 120+ lines | Integration | Active |

---

## SYSTEM 1: SharedMemory (JSON-Based Agent Memory)

**File:** `/c/Users/darst/.lmstudio/projects/Vetinari/.claude/worktrees/zealous-pasteur/vetinari/shared_memory.py`

**Lines:** 1-284

**Architecture Pattern:** Singleton with JSON persistence

### Data Schema

```python
@dataclass
class MemoryEntry:
    entry_id: str                    # 8-char UUID
    agent_name: str                  # e.g., "plan", "build", "ask"
    memory_type: str                 # e.g., "discovery", "decision"
    summary: str                     # Short description
    content: str                     # Full content
    timestamp: str                   # ISO format
    tags: List[str] = []
    project_id: str = ""
    session_id: str = ""
    options: List[Dict] = []         # For decisions
    resolved: bool = False
    # Phase 1 additions:
    plan_id: str = ""
    wave_id: str = ""
    task_id: str = ""
    provenance: str = "agent"
    confidence: float = 1.0
    metadata: Dict = {}
```

### Persistence
- **Format:** JSON file
- **Location:** `~/.lmstudio/projects/Vetinari/memory/memory.json`
- **Load/Save:** Simple JSON serialization via `asdict()`
- **Backward compatibility:** Auto-adds missing fields during load

### Key Methods (lines 52-280)

| Method | Purpose | Usage |
|--------|---------|-------|
| `get_instance(storage_path)` | Singleton factory | Global initialization |
| `add(agent_name, memory_type, summary, content, ...)` | Create entry with auto UUID | Record agent decisions |
| `search(query, agent_name, memory_type, limit)` | Text search in summary+content | Query by keyword |
| `get_recent(limit)` | Timeline by timestamp desc | Recent entries |
| `get_by_agent(agent_name, limit)` | Filter by agent | Agent-specific memory |
| `get_by_type(memory_type, limit)` | Filter by type | Type-specific queries |
| `get_timeline(limit)` | Chronological order | Event playback |
| `get_stats()` | Count by type/agent | Dashboard metrics |
| `get_by_plan(plan_id, limit)` | Phase 1: filter by plan | Plan-aware memory |
| `get_by_task(task_id, limit)` | Phase 1: filter by task | Task-aware memory |
| `resolve_decision(decision_id, choice)` | Mark decision resolved | Record human choices |
| `forget(entry_id)` | Mark as "forgotten" (not deleted) | Soft delete |
| `compact(max_age_days)` | Prune old entries | Retention policy |

### Usages (17 references total)

**Active usage:**
- **web_ui.py** (3 refs, lines 2360-2400):
  - `/api/memory` endpoints for UI memory display
  - `SharedMemory.get_instance()` for REST access

- **agents/context_manager_agent.py** (2 refs, lines 351-359):
  - Load recent 30 entries for context assembly
  - Error handling with fallback

- **agents/consolidated/orchestrator_agent.py** (2 refs, lines 296-300):
  - Load recent 30 entries for orchestration context
  - Graceful degradation on failure

**Passive references:**
- types.py: Documentation references
- shared_memory.py: Definition + module-level singleton instance

### Strengths
- ✅ Simple, transparent JSON format
- ✅ Minimal dependencies
- ✅ Fast in-memory operations
- ✅ Plan-aware via Phase 1 fields

### Weaknesses
- ❌ No deduplication
- ❌ No semantic search
- ❌ No multiple backends
- ❌ Text search only (no full-text indices)
- ❌ No secret filtering
- ❌ Low active adoption (17 refs vs 112 for MemoryStore)

**Status:** Auxiliary memory system. Primarily UI-facing. Could be deprecated in favor of DualMemoryStore.

---

## SYSTEM 2: DualMemoryStore (Episodic/Semantic Agent Memory)

**File:** `/c/Users/darst/.lmstudio/projects/Vetinari/.claude/worktrees/zealous-pasteur/vetinari/memory/dual_memory.py`

**Lines:** 1-349

**Architecture Pattern:** Dual-backend coordinator with content deduplication

### Dual Backend Architecture

```
┌─────────────────────────────────────────┐
│      DualMemoryStore (Coordinator)      │
├─────────────────┬───────────────────────┤
│  OcMemoryStore  │  MnemosyneMemoryStore │
│  (SQLite)       │  (JSON)               │
│  ~280 lines     │  ~250 lines           │
└─────────────────┴───────────────────────┘
```

### Data Schema (from `interfaces.py`, lines 24-98)

```python
@dataclass
class MemoryEntry:
    id: str                                    # mem_{uuid[:8]}
    agent: str                                 # Agent name
    entry_type: MemoryEntryType                # DISCOVERY, DECISION, OUTCOME, etc.
    content: str                               # Full observation/insight
    summary: str                               # Brief description
    timestamp: int                             # Milliseconds since epoch
    provenance: str                            # Source context
    source_backends: List[str]                 # ["oc"], ["mnemosyne"], or both
    metadata: Optional[Dict[str, Any]]         # Extended data

@dataclass
class MemoryStats:
    total_entries: int
    file_size_bytes: int
    oldest_entry: int
    newest_entry: int
    entries_by_agent: Dict[str, int]
    entries_by_type: Dict[str, int]
```

### MemoryEntryType Enum (canonical from `types.py`)
- DISCOVERY (observation, pattern)
- DECISION (choice point)
- OUTCOME (execution result)
- INSIGHT (synthesis)
- FAILURE (error/lesson)
- SUCCESS (achievement)
- QUESTION (uncertainty)
- ANSWER (resolution)
- MODEL_SELECTION (router decision)
- PERFORMANCE_CHANGE (metric update)
- OPTIMIZATION_FINDING (improvement)

### OcMemoryStore Persistence (lines 24-284 of oc_memory.py)

**Database:** `./.vetinari/mnemoria/memories.db` (SQLite)

**Schema:**
```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    source_backends TEXT NOT NULL,         -- JSON array
    forgotten INTEGER DEFAULT 0,            -- Tombstone flag
    content_hash TEXT NOT NULL,             -- SHA-256 for dedup
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast queries:
CREATE INDEX idx_memories_agent ON memories(agent);
CREATE INDEX idx_memories_type ON memories(entry_type);
CREATE INDEX idx_memories_timestamp ON memories(timestamp);
CREATE INDEX idx_memories_hash ON memories(content_hash);
CREATE INDEX idx_memories_forgotten ON memories(forgotten);
```

### MnemosyneMemoryStore Persistence (lines 23-253 of mnemosyne_memory.py)

**File:** `./.vetinari/mnemosyne/memories.json`

**Schema:**
```json
{
    "memories": [
        {
            "id": "mem_abc123",
            "agent": "builder",
            "entry_type": "discovery",
            "content": "...",
            "summary": "...",
            "timestamp": 1234567890000,
            "provenance": "execution",
            "source_backends": ["mnemosyne"],
            "content_hash": "sha256hex",
            "forgotten": false,
            "created_at": "2026-03-08T..."
        }
    ],
    "metadata": {
        "last_updated": "2026-03-08T...",
        "entry_count": 42
    }
}
```

### Key Methods (lines 25-331)

| Method | Behavior |
|--------|----------|
| `remember(entry)` | Write to BOTH backends, sanitize secrets, track source_backends |
| `search(query, agent, entry_types, limit)` | Search both, merge results, deduplicate by content_hash |
| `timeline(agent, start_time, end_time, limit)` | Browse chronologically, merge from both |
| `ask(question, agent)` | NLP-style query (keyword-based), merge results |
| `export(path)` | Dual JSON export (_oc.json, _mnemo.json) |
| `forget(entry_id, reason)` | Tombstone in both backends |
| `compact(max_age_days)` | Remove forgotten entries, prune old by age |
| `stats()` | Merged statistics from both backends |
| `get_entry(entry_id)` | Fallback lookup (primary backend first, then secondary) |
| `_filter_secrets(entry)` | Sanitize content + metadata before storage |
| `_merge_and_dedupe(entries, limit)` | Dedup by SHA-256 hash, pick winner by timestamp + backend priority |

### Write Behavior
1. **Filter secrets** via `get_secret_scanner()`
2. **Write to OcMemoryStore** (SQLite)
3. **Write to MnemosyneMemoryStore** (JSON)
4. **Track source_backends** list on entry
5. **Return primary ID** (OC ID preferred over mnemosyne)
6. **Handle partial failures** gracefully (log warnings, continue if either succeeds)

### Read Behavior
1. **Query both backends in parallel** (exception-safe)
2. **Combine results** into single list
3. **Deduplicate by content_hash** if enabled (env: MEMORY_DEDUP_ENABLED, default true)
4. **Pick winner** per hash group by:
   - Most recent timestamp
   - Primary backend preference (env: MEMORY_PRIMARY_READ, default "oc")
5. **Merge source_backends** lists across duplicates
6. **Sort by timestamp descending**
7. **Return top N** (respects limit)

### Environment Configuration (interfaces.py, lines 159-165)

```python
MEMORY_BACKEND_MODE = os.environ.get("MEMORY_BACKEND_MODE", "dual")  # or "oc", "mnemosyne"
OC_MEMORY_PATH = os.environ.get("OC_MEMORY_PATH", "./.vetinari/mnemoria")
MNEMOSYNE_PATH = os.environ.get("MNEMOSYNE_PATH", "./.vetinari/mnemosyne")
MEMORY_PRIMARY_READ = os.environ.get("MEMORY_PRIMARY_READ", "oc")  # Fallback priority
MEMORY_DEDUP_ENABLED = os.environ.get("MEMORY_DEDUP_ENABLED", "true") in ("1", "true", "yes")
MEMORY_MERGE_LIMIT = int(os.environ.get("MEMORY_MERGE_LIMIT", "100"))
```

### Usages (37 references total)

**Direct imports:**
- **learning/auto_tuner.py** (line 201-202):
  - `from vetinari.memory import MemoryEntry, MemoryEntryType, get_dual_memory_store`
  - Log hyperparameter tuning DECISIONs

- **tools/tool_registry_integration.py** (3 refs):
  - `get_dual_memory_store()` for tool execution tracking
  - `MemoryEntry` for structured recording

- **memory/__init__.py** (module definition + exports)

**Referenced in:**
- coding_agent/engine.py: Documentation reference
- memory/__init__.py: Re-exports and singleton management

### Strengths
- ✅ Dual persistent backends (resilience)
- ✅ Content-hash deduplication (efficiency)
- ✅ Secret filtering (security)
- ✅ Rich MemoryEntryType enum
- ✅ Merge policy with backend priority
- ✅ Backward compatibility (both backends optional)
- ✅ Comprehensive stats across backends

### Weaknesses
- ❌ Moderate complexity (dual backend overhead)
- ❌ Lower adoption than MemoryStore (37 vs 112 refs)
- ❌ Ask() uses keyword search only (no semantic)
- ❌ Write amplification (every write hits 2 backends)

**Status:** Modern, robust agent memory system. Underutilized but well-designed.

---

## SYSTEM 3: MemoryStore (Plan Execution Tracking) ⭐ CRITICAL

**File:** `/c/Users/darst/.lmstudio/projects/Vetinari/.claude/worktrees/zealous-pasteur/vetinari/memory/__init__.py`

**Lines:** 50-548 (500 lines)

**Architecture Pattern:** Singleton with SQLite/JSON fallback

### Purpose
Track plan execution, subtask outcomes, and model performance metrics for feedback loops.

### Three Core Tables

#### Table 1: PlanHistory

**Purpose:** Track plan versions, goals, approval status, and execution outcomes

**Schema:**
```sql
CREATE TABLE PlanHistory (
    plan_id TEXT PRIMARY KEY,
    plan_version INTEGER DEFAULT 1,
    goal TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'draft',                    -- draft, approved, executing, completed, failed
    plan_json TEXT,                                 -- Serialized plan structure
    plan_explanation_json TEXT,                     -- JSON explanation object
    chosen_plan_id TEXT,                            -- If multiple plans evaluated
    plan_justification TEXT,                        -- Why this plan chosen
    risk_score REAL DEFAULT 0.0,                   -- 0.0-1.0 risk assessment
    dry_run BOOLEAN DEFAULT 0,                      -- Simulation flag
    auto_approved BOOLEAN DEFAULT 0                 -- Human approval bypass
);
```

**Key Methods:**
- `write_plan_history(plan_data)` — INSERT OR REPLACE (lines 153-182)
- `query_plan_history(plan_id, goal_contains, limit)` — Retrieve plans (lines 236-282)

**Usages:**
- plan_api.py: Fetch stats
- plan_mode.py: Plan execution tracking
- feedback_loop.py: Indirectly via plan context

#### Table 2: SubtaskMemory

**Purpose:** Track individual subtask execution, model assignments, outcomes

**Schema:**
```sql
CREATE TABLE SubtaskMemory (
    subtask_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    parent_subtask_id TEXT,                        -- Hierarchical parent
    description TEXT NOT NULL,
    depth INTEGER DEFAULT 0,                       -- Tree depth
    status TEXT DEFAULT 'pending',                 -- pending, in_progress, done, failed
    assigned_model_id TEXT,                        -- Which model executed this
    outcome TEXT,                                  -- success, failed, partial
    duration_seconds REAL,                         -- Execution time
    cost_estimate REAL,                            -- Predicted cost
    rationale TEXT,                                -- Why this model/assignment
    subtask_explanation_json TEXT,                 -- JSON explanation
    domain TEXT,                                   -- Task domain (coding, research, etc.)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES PlanHistory(plan_id)
);
```

**Key Methods:**
- `write_subtask_memory(subtask_data)` — INSERT OR REPLACE (lines 193-225)
- `query_subtasks(plan_id, subtask_id, depth)` — Hierarchical queries (lines 284-337)
- `update_subtask_quality(subtask_id, quality_score, succeeded)` — Outcome annotation (lines 356-380)

**Usages:**
- plan_mode.py: Subtask creation and status tracking
- feedback_loop.py: Quality score recording
- learning/feedback_loop.py: Outcome updates

#### Table 3: ModelPerformance

**Purpose:** Track running performance metrics per (model_id, task_type) pair

**Schema:**
```sql
CREATE TABLE ModelPerformance (
    model_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    success_rate REAL DEFAULT 0.0,                 -- 0.0-1.0 recent success
    avg_latency REAL DEFAULT 0.0,                  -- milliseconds
    total_uses INTEGER DEFAULT 0,                  -- Total task count
    last_used_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, task_type)
);
```

**Key Methods:**
- `get_model_performance(model_id, task_type)` — Read running metrics (lines 339-354)
- `update_model_performance(model_id, task_type, success/dict, latency)` — EMA updates (lines 382-464)
  - Supports two signatures:
    - Legacy: `update_model_performance(model_id, task_type, success: bool, latency: float)`
    - New: `update_model_performance(model_id, task_type, data: dict)` with keys like "success_rate", "avg_latency"
  - Implements exponential moving average (EMA) for running stats
  - If record exists: blends with old values; creates new if missing

**Usages:**
- learning/feedback_loop.py: **Primary consumer** (lines 77-113)
  - Reads existing performance via `get_model_performance()`
  - Blends success + quality into single signal (50/50 weight)
  - Applies EMA alpha=0.3 for recent performance emphasis
  - Writes back via `update_model_performance()`
- dynamic_model_router.py: Indirectly (feedback from router)

### Persistence

**Primary:** SQLite at `./vetinari_memory.db`
- Environment variable: `PLAN_MEMORY_DB_PATH` (lines 22)
- Full ACID compliance
- Row factory enabled for dict-like access

**Fallback:** JSON at `./vetinari_memory.json`
- Triggered if SQLite init fails (lines 130-133)
- Mirror structure: `{"plans": {}, "subtasks": {}, "model_performance": {}}`
- Degrades gracefully but loses indexing benefits

### Retention Policy

```python
PLAN_RETENTION_DAYS = int(os.environ.get("PLAN_RETENTION_DAYS", "90"))  # Default 90 days
```

**Pruning:**
- `prune_old_plans(retention_days)` — Delete cascade (lines 466-510)
  - Finds plans created before cutoff
  - Deletes matching SubtaskMemory records
  - Deletes matching PlanHistory records
  - Logs count

### Key Methods Summary

| Method | Purpose | Lines |
|--------|---------|-------|
| `__init__(db_path, use_json_fallback)` | Initialize storage (SQLite or JSON) | 58-67 |
| `write_plan_history(plan_data)` | Record plan execution | 153-191 |
| `write_subtask_memory(subtask_data)` | Record subtask outcome | 193-234 |
| `query_plan_history(plan_id, goal_contains, limit)` | Retrieve plans | 236-282 |
| `query_subtasks(plan_id, subtask_id, depth)` | Hierarchical subtask queries | 284-337 |
| `get_model_performance(model_id, task_type)` | Read running metrics | 339-354 |
| `update_subtask_quality(subtask_id, quality_score, succeeded)` | Annotate outcome | 356-380 |
| `update_model_performance(model_id, task_type, success_or_dict, latency)` | EMA update | 382-464 |
| `prune_old_plans(retention_days)` | Retention enforcement | 466-510 |
| `get_memory_stats()` | Summary metrics (total, by type) | 517-548 |

### Usages (112 references total) — HEAVIEST USAGE

**Direct Imports:**
- **plan_api.py** (line 417): `get_memory_store()` for stats endpoint
- **plan_mode.py** (line 39): `MemoryStore` parameter in PlanModeEngine.__init__
- **learning/feedback_loop.py** (line 36): `get_memory_store()` for ModelPerformance updates

**Data Structure References:**
- **plan_types.py** (lines 28-120): Subtask dataclass mirrors SubtaskMemory schema
- **learning/feedback_loop.py**: Heavy usage in _update_memory_performance()

**Indirect Uses:**
- Multiple plan execution paths write to SubtaskMemory
- Quality scoring paths update ModelPerformance via FeedbackLoop
- Router selection reads ModelPerformance for decision making

### Strengths
- ✅ **CRITICAL infrastructure** for plan execution tracking
- ✅ Structured SQL tables with proper keys/constraints
- ✅ Hierarchical subtask support (parent_id, depth)
- ✅ Running performance metrics (EMA logic)
- ✅ Cascade deletion (plan delete → subtask delete)
- ✅ JSON fallback for robustness
- ✅ Comprehensive stats
- ✅ Heavy adoption (112 refs) = battle-tested

### Weaknesses
- ❌ JSON fallback loses index benefits
- ❌ Mixing structured (SQLite) + fallback (JSON) adds complexity
- ❌ Legacy signature support in update_model_performance() (dict vs bool)

**Status:** ⭐ **CORE CRITICAL SYSTEM**. Cannot be consolidated without losing plan infrastructure.

---

## SYSTEM 4: EpisodeMemory (Episodic Learning with Similarity Search)

**File:** `/c/Users/darst/.lmstudio/projects/Vetinari/.claude/worktrees/zealous-pasteur/vetinari/learning/episode_memory.py`

**Lines:** 1-416 (partial read: 1-150)

**Architecture Pattern:** Singleton with thread-safe embedding index

### Purpose
Store past agent execution episodes and retrieve similar past experiences via embedding similarity for prompt injection.

### Data Model

```python
@dataclass
class Episode:
    episode_id: str                  # Unique identifier
    timestamp: str                   # ISO format
    task_summary: str                # Truncated description (for display)
    agent_type: str                  # "BUILDER", "RESEARCHER", etc.
    task_type: str                   # "coding", "data_processing", etc.
    output_summary: str              # Truncated key results
    quality_score: float             # 0.0-1.0
    success: bool
    model_id: str                    # Which model executed
    embedding: List[float]           # Vector for similarity (256-dim)
    metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict:
        # Serialization for export
```

### Persistence

**Database:** `./vetinari_episodes.db` (environment: VETINARI_EPISODE_DB)

**Configuration:**
```python
_MAX_EPISODES = int(os.environ.get("VETINARI_MAX_EPISODES", "10000"))  # Bounded storage
```

**Schema:** (inferred from lines 138-150)
- Stores full Episode records with embeddings
- Bounded to MAX_EPISODES (importance-based eviction)
- Thread-safe via RLock

### Embedding Strategy

**Primary:** sentence-transformers "all-MiniLM-L6-v2" (if available)
- Modern semantic embeddings
- 384-dimensional vectors (actual model)
- GPU-accelerated when available

**Fallback:** Character n-gram hashing (CPU-only, no dependencies)
```python
def _simple_embedding(text: str, dim: int = 256) -> List[float]:
    """Character n-gram hashing embedding — CPU only, no dependencies."""
    vec = [0.0] * dim
    text_lower = text.lower()
    for n in (2, 3, 4):  # Trigrams, bigrams, fourgrams
        for i in range(len(text_lower) - n + 1):
            gram = text_lower[i:i + n]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
    # L2-normalize
    norm = (sum(x * x for x in vec) ** 0.5) or 1.0
    return [x / norm for x in vec]
```

**Similarity metric:** Cosine similarity (0.0-1.0)

### Key Methods (inferred from lines 1-150)

| Method | Purpose |
|--------|---------|
| `record(task_description, agent_type, task_type, output_summary, quality_score, success, model_id)` | Store new episode |
| `recall(query, k, min_score)` | Similarity-based retrieval (top-k by cosine) |
| `search(query)` | Keyword search (if available) |
| `_init_db()` | Initialize SQLite schema |
| `get_episode_memory()` | Singleton accessor |

### Usages (Specialized)

- **agents/base_agent.py**: `get_episode_memory()` for prompt injection
  - Recall similar past tasks for in-context examples

- **prompts/assembler.py**: `get_episode_memory()`
  - Inject relevant past episodes into prompt context

- **learning module**: Docstring examples and documentation

### Strengths
- ✅ Specialized for ML learning use case
- ✅ Dual embedding support (modern + fallback)
- ✅ Thread-safe with RLock
- ✅ Bounded storage (importance eviction)
- ✅ Cosine similarity for semantic relevance
- ✅ No external ML dependencies (fallback hashing)

### Weaknesses
- ❌ Separate storage system (non-integrated)
- ❌ Specialized use case (not general-purpose)
- ❌ Lower adoption than MemoryStore/SharedMemory

**Status:** Specialized episodic learning system. Complementary to other memory systems, not overlapping.

---

## SYSTEM 5: FeedbackLoop (Integration Layer)

**File:** `/c/Users/darst/.lmstudio/projects/Vetinari/.claude/worktrees/zealous-pasteur/vetinari/learning/feedback_loop.py`

**Lines:** 1-120+ (partial read)

**Architecture Pattern:** Integration layer (not a storage system itself)

### Purpose
Close the feedback loop: execution outcome → ModelPerformance updates → router cache

### Data Flow

```
Execution completes
       ↓
record_outcome(task_id, model_id, task_type, quality_score, latency_ms, success)
       ↓
[1] _update_memory_performance()  → MemoryStore.ModelPerformance (EMA)
[2] _update_router_cache()        → DynamicModelRouter cache
[3] _update_subtask_quality()     → MemoryStore.SubtaskMemory outcome
```

### Key Methods (lines 50-120+)

**Public API:**
```python
def record_outcome(
    self,
    task_id: str,
    model_id: str,
    task_type: str,
    quality_score: float,
    latency_ms: int = 0,
    cost_usd: float = 0.0,
    success: bool = True,
) -> None:
```

**Internal Implementation:**

1. **_update_memory_performance()** (lines 86-113)
   - Reads: `mem.get_model_performance(model_id, task_type)`
   - Implements EMA with alpha=0.3
   - Signal = 0.5 * success + 0.5 * quality_score (avoid double-counting)
   - Writes: `mem.update_model_performance()` with new running averages
   - Error handling: graceful degradation

2. **_update_router_cache()** (lines 115-120+)
   - Updates DynamicModelRouter in-memory cache
   - Sync with MemoryStore updates

3. **_update_subtask_quality()** (inferred)
   - Annotates SubtaskMemory.outcome with quality_score
   - Records success/failure flag

### EMA Logic (Exponential Moving Average)

```python
EMA_ALPHA = 0.3  # Recent observations get 30% weight

old_rate = existing.get("success_rate", 0.7)
old_latency = existing.get("avg_latency", latency_ms or 1000)
old_uses = existing.get("total_uses", 0)

# Blend success + quality (avoid double-counting)
signal = 0.5 * (1.0 if success else 0.0) + 0.5 * quality

new_rate = (1 - 0.3) * old_rate + 0.3 * signal
new_latency = (1 - 0.3) * old_latency + 0.3 * (latency_ms or old_latency)
new_uses = old_uses + 1
```

**Example:**
- Old: success_rate=0.7, quality_score=0.95, success=True
- Signal = 0.5 * 1.0 + 0.5 * 0.95 = 0.975
- New rate = 0.7 * 0.7 + 0.3 * 0.975 = 0.49 + 0.2925 = 0.7825

### Usages

- **agents/base_agent.py**: `get_feedback_loop()` for outcome recording
- **executor.py**: `feedback.record_outcome()` after task completion
- **orchestration/durable_execution.py**: `record_outcome()` in execution flow
- **learning/feedback_loop.py**: Definition

### Integration Points

**Reads from:**
- MemoryStore.get_model_performance()

**Writes to:**
- MemoryStore.update_model_performance() (via EMA)
- DynamicModelRouter cache (in-memory)
- MemoryStore.update_subtask_quality()

### Strengths
- ✅ Clean separation of concerns
- ✅ Graceful degradation
- ✅ EMA logic is well-designed (time-weighted, double-counting avoidance)
- ✅ Bridges execution → selection loop

### Weaknesses
- ❌ Depends on external systems (MemoryStore, DynamicModelRouter)
- ❌ Could be tightly coupled

**Status:** Active integration layer. Correct architecture, no consolidation needed.

---

## CONSOLIDATION ANALYSIS

### Functional Overlaps

| System Pair | Overlap? | Recommendation |
|-------------|----------|-----------------|
| **SharedMemory vs DualMemoryStore** | YES (both agent memory) | Consolidate: DualMemoryStore is superior |
| **SharedMemory vs MemoryStore** | NO (different domains) | Keep separate |
| **DualMemoryStore vs MemoryStore** | NO (episodic vs planning) | Keep separate |
| **MemoryStore vs EpisodeMemory** | NO (structured vs learning) | Keep separate |
| **MemoryStore vs FeedbackLoop** | NO (storage vs integration) | Keep separate |
| **EpisodeMemory vs DualMemoryStore** | PARTIAL (both episodic) | Consider integration |

### TIER 1: CRITICAL — Do Not Consolidate

**MemoryStore (PlanHistory, SubtaskMemory, ModelPerformance)**
- Core to plan execution tracking
- Heavy usage: 112 references
- Removing = plan system breaks
- Unique structured tables for plan-specific data
- **Verdict:** KEEP AS-IS

**EpisodeMemory**
- Specialized ML learning component
- Different persistence model (embeddings)
- Different use case (experience recall via similarity)
- Low interference with other systems
- **Verdict:** KEEP AS-IS

**FeedbackLoop**
- Clean integration layer
- Well-designed EMA logic
- Correct separation of concerns
- **Verdict:** KEEP AS-IS

### TIER 2: MODERNIZE — Migrate Away From

**SharedMemory (JSON-based agent memory)**
- Low usage: 17 references (mostly UI)
- JSON-only, no dedup, minimal structure
- Lack of semantic search
- No secret filtering
- Legacy approach vs modern DualMemoryStore
- **Replacement:** DualMemoryStore for agent memory needs
- **Risk Level:** LOW
- **Impact:** Affects only UI endpoints + orchestrator context loading

### TIER 3: LEVERAGE — Already Optimal

**DualMemoryStore**
- Modern architecture with dual backends
- Deduplication with content-hash
- Secret filtering integrated
- Used where needed (auto_tuner, tool_registry)
- **Verdict:** KEEP AND EXPAND USAGE

---

## MIGRATION PATH: Consolidating SharedMemory into DualMemoryStore

### Step 1: Identify All SharedMemory Usage (17 references)

```
✗ vetinari/shared_memory.py:283         — Module-level singleton
✗ vetinari/types.py:158-176             — Documentation only
✗ vetinari/web_ui.py:2360-2400          — 3 refs (REST endpoints)
✗ vetinari/agents/context_manager_agent.py:351-359  — 2 refs
✗ vetinari/agents/consolidated/orchestrator_agent.py:296-300  — 2 refs
```

### Step 2: Map Data Types

```python
# OLD: SharedMemory.MemoryEntry
@dataclass
class MemoryEntry:
    entry_id: str
    agent_name: str
    memory_type: str
    summary: str
    content: str
    timestamp: str
    tags: List[str]
    project_id: str
    session_id: str
    plan_id: str
    wave_id: str
    task_id: str
    provenance: str
    confidence: float
    metadata: Dict

# NEW: DualMemoryStore.MemoryEntry
@dataclass
class MemoryEntry:
    id: str
    agent: str                        # ← agent_name → agent
    entry_type: MemoryEntryType       # ← memory_type → entry_type (enum)
    content: str
    summary: str
    timestamp: int                    # ← str → int (milliseconds)
    provenance: str
    source_backends: List[str]        # ← automatic
    metadata: Optional[Dict]
```

### Step 3: Mapping Guide

```python
# SharedMemory.add() → DualMemoryStore.remember()
OLD:
    entry = shared_memory.add(
        agent_name="plan",
        memory_type="discovery",
        summary="Found optimal model",
        content="gpt-4-turbo has best score",
        tags=["model-selection"],
        plan_id="plan_abc123"
    )

NEW:
    from vetinari.memory import MemoryEntryType, get_dual_memory_store
    store = get_dual_memory_store()
    entry = store.remember(MemoryEntry(
        agent="plan",
        entry_type=MemoryEntryType.DISCOVERY,  # or MemoryEntryType.MODEL_SELECTION
        summary="Found optimal model",
        content="gpt-4-turbo has best score",
        provenance="model_selection",
        metadata={"tags": ["model-selection"], "plan_id": "plan_abc123"}
    ))

---

# SharedMemory.get_all() → DualMemoryStore.timeline()
OLD:
    entries = shared_memory.get_all(limit=30)
    for e in entries:
        print(f"{e.agent_name}: {e.summary}")

NEW:
    entries = store.timeline(limit=30)
    for e in entries:
        print(f"{e.agent}: {e.summary}")

---

# SharedMemory.get_by_plan() → DualMemoryStore.search() + filtering
OLD:
    plan_entries = shared_memory.get_by_plan(plan_id="plan_abc123")

NEW:
    # Filter in metadata
    all_entries = store.timeline(limit=100)
    plan_entries = [e for e in all_entries if e.metadata.get("plan_id") == "plan_abc123"]
    # Or store plan context differently
```

### Step 4: Update Files

**File 1: vetinari/web_ui.py** (3 locations, ~50 lines)
- Replace `from vetinari.shared_memory import SharedMemory`
- Replace `SharedMemory.get_instance()`
- Update endpoints to use `get_dual_memory_store().timeline()`
- Update serialization (MemoryEntry.to_dict())

**File 2: vetinari/agents/context_manager_agent.py** (2 locations, ~20 lines)
- Replace imports
- Map `shared_memory.get_all()` → `dual_store.timeline()`

**File 3: vetinari/agents/consolidated/orchestrator_agent.py** (2 locations, ~20 lines)
- Replace imports
- Map `shared_memory.get_all()` → `dual_store.timeline()`

### Step 5: Testing & Validation

```bash
# Run unit tests
python -m pytest tests/test_*memory*.py -v

# Test migration paths
python -m pytest tests/test_dual_memory.py -v
python -m pytest tests/test_web_ui.py::TestMemoryEndpoints -v

# Integration tests
python -m pytest tests/test_orchestrator.py -v
```

### Step 6: Deprecation & Cleanup

**Phase 1 (v3.6):** Mark SharedMemory.add/search/get_* as deprecated
```python
import warnings
def add(self, ...):
    warnings.warn("SharedMemory deprecated. Use DualMemoryStore instead.", DeprecationWarning)
    ...
```

**Phase 2 (v3.7):** Remove SharedMemory entirely
- Delete vetinari/shared_memory.py
- Update CHANGELOG

### Risk Assessment

| Component | Risk | Mitigation |
|-----------|------|-----------|
| Web UI endpoints | LOW | Full test coverage exists |
| Context loading | LOW | Graceful error handling |
| Memory data | LOW | DualMemoryStore is more robust |
| Timeline ordering | MEDIUM | Verify sorting in tests |
| Type conversions | MEDIUM | Add integration tests |

**Overall Risk:** LOW

---

## ARCHITECTURE RECOMMENDATIONS

### Short-term (No Action)
- ✅ Leave MemoryStore, EpisodeMemory, FeedbackLoop intact
- ✅ They are well-structured and serve distinct purposes
- ✅ No consolidation benefit

### Medium-term (Modernization) — RECOMMENDED
- 🔄 Deprecate SharedMemory (legacy approach)
- 🔄 Migrate 17 references to DualMemoryStore
- 🔄 Simplifies codebase (1 fewer memory system)
- 🔄 Improves robustness (dual backends, dedup, secrets)
- **Effort:** ~40 hours (5 files, 100 lines changed)
- **Benefit:** Consistency, robustness, reduced maintenance

### Long-term (Architecture Clarity) — FUTURE
1. **Document memory system boundaries:**
   - **Agent observations/insights** → DualMemoryStore
   - **Plan execution tracking** → MemoryStore
   - **ML experience learning** → EpisodeMemory
   - **Feedback closure** → FeedbackLoop

2. **Add usage guidelines:**
   - When to use which system
   - Migration paths from legacy systems
   - Schema evolution strategy

3. **Consider EpisodeMemory ↔ DualMemoryStore integration:**
   - Episodes could store in DualMemoryStore with `entry_type=EPISODE`
   - Unify embedding infrastructure
   - **Future exploration**, not immediate

---

## SUMMARY TABLE

| System | Purpose | Size | Refs | Status | Action |
|--------|---------|------|------|--------|--------|
| **SharedMemory** | Agent memory (JSON) | 284L | 17 | Legacy | ⚠️ Deprecate |
| **DualMemoryStore** | Agent memory (dual) | 349L | 37 | Modern | ✅ Expand |
| **MemoryStore** | Plan tracking (SQL) | 500L | 112 | Critical | ✅ Keep |
| **EpisodeMemory** | Learning (similarity) | 416L | Spec | Specialized | ✅ Keep |
| **FeedbackLoop** | Integration | 120L+ | Integ | Active | ✅ Keep |

---

## CONCLUSION

Vetinari's memory architecture is **sound but has legacy components**:

- **MemoryStore** is the heavyweight champion (112 refs, critical infrastructure)
- **SharedMemory** is the legacy system (17 refs, low value)
- **DualMemoryStore** is the modern replacement (37 refs, well-designed)
- **EpisodeMemory** serves a specialized niche (learning)
- **FeedbackLoop** is the glue (integration layer)

**Recommended action:** Deprecate SharedMemory and consolidate its 17 references into DualMemoryStore. This removes legacy code, improves robustness, and unifies the agent memory layer.

All other systems are well-designed and should be retained.
