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
    MEMORY_MERGE_LIMIT
)

from .oc_memory import OcMemoryStore
from .mnemosyne_memory import MnemosyneMemoryStore
from .dual_memory import DualMemoryStore, get_dual_memory_store, init_dual_memory_store

__all__ = [
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
    "MEMORY_MERGE_LIMIT"
]
