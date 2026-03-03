# Memory Schema

This document describes the data schema for Vetinari's memory layer.

## Core Data Models

### MemoryEntry

The fundamental unit of stored information in Vetinari's memory system.

```python
@dataclass
class MemoryEntry:
    id: str                          # Unique identifier (UUID)
    agent: str                       # Agent that created this entry
    entry_type: MemoryEntryType      # Type of memory entry
    content: str                     # Full content/body of the entry
    summary: str                     # Short summary (≤200 chars)
    timestamp: int                   # Unix timestamp in milliseconds
    provenance: str                  # Source/origin of this entry
    source_backends: List[str]       # Which backends have this entry
```

### MemoryEntryType Enum

```python
class MemoryEntryType(str, Enum):
    INTENT = "intent"                 # User goals or tasks
    DISCOVERY = "discovery"          # Findings during execution
    DECISION = "decision"            # Key decisions made
    PROBLEM = "problem"              # Issues encountered
    SOLUTION = "solution"            # Solutions found
    PATTERN = "pattern"              # Reusable patterns identified
    WARNING = "warning"              # Warnings or cautions
    SUCCESS = "success"              # Successful outcomes
    REFACTOR = "refactor"            # Refactoring notes
    BUGFIX = "bugfix"                # Bug fix details
    FEATURE = "feature"              # Feature implementation notes
    APPROVAL = "approval"            # Plan approval decisions
```

### ApprovalDetails

Embedded in approval-type memory entries.

```python
@dataclass
class ApprovalDetails:
    task_id: str                     # ID of the task being approved
    task_type: str                   # Type: "coding", "docs", etc.
    plan_id: str                     # Parent plan ID
    approval_status: str             # "approved", "rejected", "auto_approved"
    approver: str                    # Who approved (or "system")
    reason: str                      # Optional rejection reason
    risk_score: float                # Risk score at time of approval
    timestamp: str                   # ISO timestamp
```

### MemoryStats

Statistics about the memory store.

```python
@dataclass
class MemoryStats:
    total_entries: int                # Total number of entries
    file_size_bytes: int             # Size of storage in bytes
    oldest_entry: int                # Timestamp of oldest entry
    newest_entry: int                # Timestamp of newest entry
    entries_by_agent: Dict[str, int] # Count per agent
    entries_by_type: Dict[str, int]  # Count per entry type
```

## SQLite Schema (OcMemoryStore)

### memories table

```sql
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    entry_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    provenance TEXT NOT NULL,
    source_backends TEXT NOT NULL,  -- JSON array
    forgotten INTEGER DEFAULT 0,     -- Tombstone marker
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(entry_type);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(forgotten);
```

### approvals table

```sql
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    approval_status TEXT NOT NULL,
    approver TEXT NOT NULL,
    reason TEXT,
    risk_score REAL NOT NULL,
    timestamp TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_approvals_task ON approvals(task_id);
CREATE INDEX IF NOT EXISTS idx_approvals_plan ON approvals(plan_id);
```

## JSON Schema (MnemosyneMemoryStore)

```json
{
  "version": "1.0",
  "memories": [
    {
      "id": "mem_abc123",
      "agent": "planner",
      "entry_type": "decision",
      "content": "Full decision content...",
      "summary": "Decision summary",
      "timestamp": 1709478600000,
      "provenance": "plan_generation",
      "source_backends": ["oc", "mnemosyne"]
    }
  ],
  "approvals": [
    {
      "id": "apr_xyz789",
      "task_id": "subtask_001",
      "task_type": "coding",
      "plan_id": "plan_abc123",
      "approval_status": "approved",
      "approver": "john_doe",
      "reason": "Code review passed",
      "risk_score": 0.15,
      "timestamp": "2026-03-03T10:30:00Z"
    }
  ],
  "metadata": {
    "created_at": "2026-01-01T00:00:00Z",
    "last_updated": "2026-03-03T10:30:00Z",
    "entry_count": 42
  }
}
```

## Content Hash for Deduplication

Entries are deduplicated using SHA-256 of the content:

```python
import hashlib

def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()
```

When merging from dual backends:
1. Compute content hash for each entry
2. Group entries by hash
3. For each group, keep the entry with latest timestamp
4. On timestamp tie, prefer OcMemoryStore entry

## Data Retention

- Default retention: 90 days (configurable via `PLAN_RETENTION_DAYS`)
- Forgotten entries (tombstones) are kept until compaction
- Compaction removes both tombstone markers and entries older than retention period
