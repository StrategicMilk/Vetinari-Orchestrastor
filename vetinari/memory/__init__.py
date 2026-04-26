"""Vetinari Memory Module — unified storage with three access patterns.

Which getter to use:

- ``get_unified_memory_store()`` — core storage (memories, episodes, embeddings).
  Use for agent recall, episode recording, and direct memory CRUD.
- ``get_memory_store()`` — plan execution tracking (plan history, subtask logs).
  Use when recording or querying plan/subtask lifecycle events.
- ``get_shared_memory()`` — facade over all subsystems (store + plan tracking
  + blackboard). Use when you need access to multiple subsystems from one object.
"""

from __future__ import annotations

import logging
import os

from vetinari.constants import _PROJECT_ROOT  # noqa: VET123 - barrel export preserves public import compatibility

logger = logging.getLogger(__name__)

PLAN_MEMORY_DB_PATH = os.environ.get("PLAN_MEMORY_DB_PATH", str(_PROJECT_ROOT / "vetinari_memory.db"))
PLAN_RETENTION_DAYS = int(os.environ.get("PLAN_RETENTION_DAYS", "90"))
PLAN_ADMIN_TOKEN = os.environ.get("PLAN_ADMIN_TOKEN", "")

from .intent_parser import (  # noqa: E402 - late import is required after bootstrap setup
    IntentParser,
    ParsedQuery,
    QueryIntent,
    get_intent_parser,
)
from .interfaces import (  # noqa: E402, VET123 — DUAL_MEMORY_AVAILABLE imported from vetinari.memory in plan_api.py
    DUAL_MEMORY_AVAILABLE,
    ApprovalDetails,
    MemoryEntry,
    MemoryStats,
    MemoryType,
    content_hash,
)
from .plan_tracking import (  # noqa: E402, VET123 — init_memory_store has no external callers but removing causes VET120
    MemoryStore,
    get_memory_store,
    init_memory_store,
)
from .shared import SharedMemory, get_shared_memory  # noqa: E402 - late import is required after bootstrap setup
from .unified import (  # noqa: E402 - late import is required after bootstrap setup
    RecordedEpisode,
    SessionContext,
    UnifiedMemoryStore,
    get_unified_memory_store,
    init_unified_memory_store,
)

__all__ = [
    "DUAL_MEMORY_AVAILABLE",
    "PLAN_ADMIN_TOKEN",
    "PLAN_MEMORY_DB_PATH",
    "PLAN_RETENTION_DAYS",
    "ApprovalDetails",
    "Episode",
    "IntentParser",
    "MemoryEntry",
    "MemoryStats",
    "MemoryStore",
    "MemoryType",
    "ParsedQuery",
    "QueryIntent",
    "RecordedEpisode",
    "SessionContext",
    "SharedMemory",
    "UnifiedMemoryStore",
    "content_hash",
    "get_intent_parser",
    "get_memory_store",
    "get_shared_memory",
    "get_unified_memory_store",
    "init_memory_store",
    "init_unified_memory_store",
]
