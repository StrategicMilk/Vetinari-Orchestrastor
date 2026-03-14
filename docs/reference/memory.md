# Memory System

Vetinari uses a dual-backend memory architecture with a unified interface,
deduplication, and merge semantics.  This document consolidates the memory
abstraction, schema, and merge-policy specifications.

## Architecture

Two backends sit behind a unified `IMemoryStore` interface:

| Backend | Storage | Strength |
|---------|---------|----------|
| **OcMemoryStore** | SQLite | Fast queries, ACID transactions |
| **MnemosyneMemoryStore** | JSON files | Portable, human-readable |

`DualMemoryStore` writes to both and merges reads.

## IMemoryStore Interface

Every backend implements these operations:

| Method | Purpose |
|--------|---------|
| `remember(entry)` | Store a memory entry, returns entry ID |
| `search(query, agent?, types?, limit)` | Keyword / semantic search |
| `timeline(agent?, start?, end?, limit)` | Chronological browse |
| `ask(question, agent?)` | Natural-language query |
| `export(path)` | Export to JSON |
| `forget(entry_id, reason)` | Tombstone an entry |
| `compact(max_age_days?)` | Prune forgotten / old entries |
| `stats()` | Aggregate statistics |

## Data Models

### MemoryEntry

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | UUID |
| `agent` | `str` | Creating agent |
| `entry_type` | `MemoryEntryType` | Category enum |
| `content` | `str` | Full body |
| `summary` | `str` | Short summary (max 200 chars) |
| `timestamp` | `int` | Unix ms |
| `provenance` | `str` | How the entry was created |
| `source_backends` | `list[str]` | Contributing backends |

### MemoryEntryType Values

`intent`, `discovery`, `decision`, `problem`, `solution`, `pattern`,
`warning`, `success`, `refactor`, `bugfix`, `feature`, `approval`.

### ApprovalDetails (embedded in approval entries)

`task_id`, `task_type`, `plan_id`, `approval_status`, `approver`,
`reason`, `risk_score`, `timestamp`.

## Storage Schemas

### SQLite (OcMemoryStore)

Tables: `memories` (indexed on agent, entry_type, timestamp, forgotten)
and `approvals` (indexed on task_id, plan_id).

### JSON (MnemosyneMemoryStore)

Top-level keys: `memories` (array), `approvals` (array), `metadata`
(created_at, last_updated, entry_count).

## Dual-Write Semantics

Every `remember()` call writes to **both** backends.  If one fails,
the other proceeds and `source_backends` reflects what succeeded.
If both fail, `MemoryStoreError` is raised.

## Read-Merge Policy

1. Fetch results from both backends (in parallel).
2. Deduplicate by SHA-256 of `(agent, entry_type, content, summary)`.
3. For duplicates: most recent timestamp wins.
4. On timestamp tie: OcMemoryStore entry wins.
5. Unique entries from either backend are always included.

| Scenario | Resolution |
|----------|------------|
| Same content, different timestamps | Most recent wins |
| Same content, same timestamp | OcMemoryStore wins |
| Unique content from both backends | Both included |
| One backend has entry, other does not | Include available |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_BACKEND_MODE` | `"dual"` | `"dual"`, `"oc"`, or `"mnemosyne"` |
| `OC_MEMORY_PATH` | `./.vetinari/mnemoria` | OcMemoryStore path |
| `MNEMOSYNE_PATH` | `./.vetinari/mnemosyne` | MnemosyneMemoryStore path |
| `MEMORY_PRIMARY_READ` | `"oc"` | Preferred backend on ties |
| `MEMORY_DEDUP_ENABLED` | `true` | Enable deduplication |
| `MEMORY_MERGE_LIMIT` | `100` | Max results to merge |
| `PLAN_RETENTION_DAYS` | `90` | Default retention period |
