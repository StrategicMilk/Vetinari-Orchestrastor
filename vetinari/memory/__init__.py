"""
Vetinari Memory Module.

This module provides unified memory storage with dual backend support:
- OcMemoryStore: oc-mnemoria style persistent memory
- MnemosyneMemoryStore: Mnemosyne style memory OS
- DualMemoryStore: coordinates between both backends
"""

from .interfaces import (
    IMemoryStore,
    MemoryEntry,
    MemoryStats,
    MemoryEntryType,
    ApprovalDetails,
    content_hash,
    MEMORY_BACKEND_MODE,
    OC_MEMORY_PATH,
    MNEMOSYNE_PATH,
    MEMORY_PRIMARY_READ,
    MEMORY_DEDUP_ENABLED,
    MEMORY_MERGE_LIMIT,
)

from .oc_memory import OcMemoryStore
from .mnemosyne_memory import MnemosyneMemoryStore
from .dual_memory import DualMemoryStore, get_dual_memory_store, init_dual_memory_store

# Enhanced memory re-exports
from .enhanced import (
    EnhancedMemoryManager,
    SemanticMemoryStore,
    ContextMemory,
    MemoryManager,
    MemoryType,
    get_memory_manager,
    init_memory_manager,
)

# Legacy plan-history store (extracted to _store.py)
from ._store import (
    MemoryStore,
    get_memory_store,
    init_memory_store,
    PLAN_MEMORY_DB_PATH,
    PLAN_RETENTION_DAYS,
    PLAN_ADMIN_TOKEN,
)

# Availability flag — True since both backends are always importable
DUAL_MEMORY_AVAILABLE = True

__all__ = [
    # New package exports
    "IMemoryStore",
    "MemoryEntry",
    "MemoryStats",
    "MemoryEntryType",
    "ApprovalDetails",
    "content_hash",
    "OcMemoryStore",
    "MnemosyneMemoryStore",
    "DualMemoryStore",
    "get_dual_memory_store",
    "init_dual_memory_store",
    "MEMORY_BACKEND_MODE",
    "OC_MEMORY_PATH",
    "MNEMOSYNE_PATH",
    "MEMORY_PRIMARY_READ",
    "MEMORY_DEDUP_ENABLED",
    "MEMORY_MERGE_LIMIT",
    "DUAL_MEMORY_AVAILABLE",
    # Enhanced memory exports
    "EnhancedMemoryManager",
    "SemanticMemoryStore",
    "ContextMemory",
    "MemoryManager",
    "MemoryType",
    "get_memory_manager",
    "init_memory_manager",
    # Legacy exports
    "MemoryStore",
    "get_memory_store",
    "init_memory_store",
    "PLAN_MEMORY_DB_PATH",
    "PLAN_RETENTION_DAYS",
    "PLAN_ADMIN_TOKEN",
]
