# Memory Abstraction Layer

This document describes the unified memory interface (IMemoryStore) for Vetinari's dual-memory backend architecture.

## Overview

Vetinari uses a unified memory abstraction layer that supports dual memory backends:
- **OcMemoryStore**: Adapter for oc-mnemoria-style persistent shared memory
- **MnemosyneMemoryStore**: Adapter for Mnemosyne-style memory OS

Both backends are behind a unified `IMemoryStore` interface, with a `DualMemoryStore` that writes to both and merges reads.

## IMemoryStore Interface

```python
class IMemoryStore(ABC):
    """Abstract interface for memory stores."""
    
    @abstractmethod
    def remember(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns entry ID."""
        pass
    
    @abstractmethod
    def search(self, query: str, agent: Optional[str] = None, 
               entry_types: Optional[List[str]] = None,
               limit: int = 10) -> List[MemoryEntry]:
        """Search memories by keyword or semantic similarity."""
        pass
    
    @abstractmethod
    def timeline(self, agent: Optional[str] = None,
                 start_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 limit: int = 100) -> List[MemoryEntry]:
        """Browse memories chronologically."""
        pass
    
    @abstractmethod
    def ask(self, question: str, agent: Optional[str] = None) -> List[MemoryEntry]:
        """Ask a natural language question against the memory store."""
        pass
    
    @abstractmethod
    def export(self, path: str) -> bool:
        """Export memories to a JSON file."""
        pass
    
    @abstractmethod
    def forget(self, entry_id: str, reason: str) -> bool:
        """Mark a memory as obsolete (tombstone)."""
        pass
    
    @abstractmethod
    def compact(self, max_age_days: Optional[int] = None) -> int:
        """Remove forgotten entries and optionally prune old data."""
        pass
    
    @abstractmethod
    def stats(self) -> MemoryStats:
        """Get memory store statistics."""
        pass
```

## MemoryEntry Data Model

```python
@dataclass
class MemoryEntry:
    id: str
    agent: str  # agent name that created this entry
    entry_type: MemoryEntryType
    content: str
    summary: str
    timestamp: int  # Unix timestamp in milliseconds
    provenance: str  # how this entry was created
    source_backends: List[str]  # ["oc", "mnemosyne"]
    
@dataclass
class MemoryStats:
    total_entries: int
    file_size_bytes: int
    oldest_entry: int
    newest_entry: int
    entries_by_agent: Dict[str, int]
    entries_by_type: Dict[str, int]
```

## Entry Types

- `intent`: User goals or tasks
- `discovery`: Findings during execution
- `decision`: Key decisions made
- `problem`: Issues encountered
- `solution`: Solutions found
- `pattern`: Reusable patterns identified
- `warning`: Warnings or cautions
- `success`: Successful outcomes
- `refactor`: Refactoring notes
- `bugfix`: Bug fix details
- `feature`: Feature implementation notes
- `approval`: Plan approval decisions

## DualMemoryStore Behavior

The `DualMemoryStore` class coordinates between both backends:

### Write Operations
- Every `remember()` call writes to **both** backends simultaneously
- Each entry tracks `source_backends` to indicate which backends contributed
- Writes are atomic in concept; in practice, both adapters are called

### Read Operations
- Results from both backends are merged
- Deduplication happens by SHA-256(content) hash
- For duplicate content, the entry with the most recent timestamp wins
- On timestamp ties, OcMemoryStore entries take precedence

### Merge Policy Summary
| Scenario | Resolution |
|----------|------------|
| Same content, different timestamps | Most recent wins |
| Same content, same timestamp | OcMemoryStore wins |
| Unique content from both | Both included |
| One backend has entry, other doesn't | Include available |

## Usage Example

```python
from vetinari.memory import get_dual_memory_store

store = get_dual_memory_store()

# Store a memory
entry_id = store.remember(MemoryEntry(
    agent="planner",
    entry_type=MemoryEntryType.DECISION,
    content="Use SQLite for primary storage",
    summary="Decision: SQLite for primary storage",
    provenance="plan_generation",
    source_backends=["oc", "mnemosyne"]
))

# Search memories
results = store.search("storage", agent="planner")

# Get timeline
timeline = store.timeline(agent="executor", limit=20)
```

## Configuration

Environment variables:
- `MEMORY_BACKEND_MODE`: "dual" (default), "oc", or "mnemosyne"
- `OC_MEMORY_PATH`: Path for oc-mnemoria store (default: "./.vetinari/mnemoria")
- `MNEMOSYNE_PATH`: Path for Mnemosyne store (default: "./.vetinari/mnemosyne")
- `MEMORY_PRIMARY_READ`: "oc" or "mnemosyne" (default: "oc")
