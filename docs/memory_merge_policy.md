# Memory Merge Policy

This document describes the read-merge policy for Vetinari's dual memory backends.

## Overview

When using `DualMemoryStore`, reads must merge results from both:
- **OcMemoryStore**: oc-mnemoria-style persistent memory
- **MnemosyneMemoryStore**: Mnemosyne-style memory OS

## Merge Algorithm

### Step 1: Fetch from Both Backends

```python
def read_merge(query: str, agent: Optional[str] = None) -> List[MemoryEntry]:
    oc_results = oc_store.search(query, agent)
    mnemo_results = mnemo_store.search(query, agent)
    
    all_entries = oc_results + mnemo_results
    return resolve_duplicates(all_entries)
```

### Step 2: Deduplicate by Content Hash

```python
def resolve_duplicates(entries: List[MemoryEntry]) -> List[MemoryEntry]:
    # Group by content hash
    by_hash: Dict[str, List[MemoryEntry]] = {}
    for entry in entries:
        h = content_hash(entry.content)
        if h not in by_hash:
            by_hash[h] = []
        by_hash[h].append(entry)
    
    # For each hash group, pick the winner
    resolved = []
    for hash_group in by_hash.values():
        winner = pick_winner(hash_group)
        resolved.append(winner)
    
    # Sort by timestamp descending
    resolved.sort(key=lambda e: e.timestamp, reverse=True)
    return resolved

def pick_winner(entries: List[MemoryEntry]) -> MemoryEntry:
    if len(entries) == 1:
        return entries[0]
    
    # Sort by timestamp descending
    sorted_entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)
    
    # If timestamps differ, most recent wins
    if sorted_entries[0].timestamp != sorted_entries[1].timestamp:
        winner = sorted_entries[0]
        winner.source_backends = list(set(sum([e.source_backends for e in entries], [])))
        return winner
    
    # Timestamps tie: prefer OcMemoryStore
    for entry in sorted_entries:
        if "oc" in entry.source_backends:
            entry.source_backends = list(set(sum([e.source_backends for e in entries], [])))
            return entry
    
    # Fallback: return first
    return sorted_entries[0]
```

## Resolution Rules Summary

| Scenario | Resolution |
|----------|------------|
| Same content, different timestamps | Most recent wins |
| Same content, same timestamp | OcMemoryStore wins |
| Unique content from both backends | Both included in results |
| One backend has entry, other doesn't | Include available entry |
| Multiple duplicates | All deduped to single entry |

## Content Hash Calculation

```python
import hashlib
import json

def content_hash(entry: MemoryEntry) -> str:
    # Normalize content for hashing
    normalized = json.dumps({
        "agent": entry.agent,
        "entry_type": entry.entry_type.value if hasattr(entry.entry_type, 'value') else entry.entry_type,
        "content": entry.content,
        "summary": entry.summary
    }, sort_keys=True)
    return hashlib.sha256(normalized.encode()).hexdigest()
```

## Source Backend Tracking

The `source_backends` field tracks which backends contributed:

```python
# Single source
entry.source_backends = ["oc"]

# Both sources (after merge)
entry.source_backends = ["oc", "mnemosyne"]

# Merged from duplicates
entry.source_backends = ["oc", "mnemosyne"]  # Contains contributions from both
```

## Write Behavior

### Dual Write

```python
def remember(self, entry: MemoryEntry) -> str:
    # Write to both backends
    entry.source_backends = ["oc", "mnemosyne"]
    
    oc_id = self.oc_store.remember(entry)
    mnemo_id = self.mnemo_store.remember(entry)
    
    # Return OC ID as primary (consistent with read preference)
    return oc_id
```

### Write Failure Handling

If one backend fails to write:

```python
def remember(self, entry: MemoryEntry) -> str:
    entry.source_backends = []
    
    try:
        oc_id = self.oc_store.remember(entry)
        entry.source_backends.append("oc")
    except Exception as e:
        logger.warning(f"OC store write failed: {e}")
        oc_id = None
    
    try:
        mnemo_id = self.mnemo_store.remember(entry)
        entry.source_backends.append("mnemosyne")
    except Exception as e:
        logger.warning(f"Mnemosyne store write failed: {e}")
        mnemo_id = None
    
    if not entry.source_backends:
        raise MemoryStoreError("Both backends failed to write")
    
    return oc_id or mnemo_id  # Return whatever succeeded
```

## Configuration

Environment variables affecting merge behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_PRIMARY_READ` | `"oc"` | Preferred backend on timestamp ties |
| `MEMORY_DEDUP_ENABLED` | `true` | Enable deduplication |
| `MEMORY_MERGE_LIMIT` | `100` | Max results to merge |

## Performance Considerations

1. **Parallel Fetch**: Both backends should be queried in parallel
2. **Caching**: Consider caching merged results for repeated queries
3. **Indexing**: Both backends should index on agent, entry_type, timestamp
4. **Pagination**: Large result sets should be paginated

## Conflict Resolution Examples

### Example 1: Timestamp Difference

```
OC:     {content: "Use SQLite", timestamp: 1709478600000}
Mnemo:  {content: "Use SQLite", timestamp: 1709478000000}

Result: {content: "Use SQLite", timestamp: 1709478600000, source_backends: ["oc", "mnemosyne"]}
```

### Example 2: Timestamp Tie

```
OC:     {content: "Use SQLite", timestamp: 1709478600000}
Mnemo:  {content: "Use SQLite", timestamp: 1709478600000}

Result: {content: "Use SQLite", timestamp: 1709478600000, source_backends: ["oc", "mnemosyne"]}
```

### Example 3: Unique Content

```
OC:     {content: "Use SQLite", timestamp: 1709478600000}
Mnemo:  {content: "Use PostgreSQL", timestamp: 1709478500000}

Results: 
  - {content: "Use SQLite", source_backends: ["oc"]}
  - {content: "Use PostgreSQL", source_backends: ["mnemosyne"]}
```
