"""
Enhanced Memory System for Vetinari - DEPRECATED FACADE

This module is a backward-compatibility shim. New code should use:

    from vetinari.memory import DualMemoryStore, get_dual_memory_store

Unique classes in this module (SemanticMemoryStore, ContextMemory, MemoryManager)
have no direct equivalent in vetinari.memory and are retained here pending
migration. See docs/MIGRATION_INDEX.md for tracking.

Migration tasks outstanding:
  - SemanticMemoryStore: SQLite + optional vector-embedding store with FTS5
    -> needs porting into vetinari/memory/ as a third backend or merged into
       OcMemoryStore's search layer.
  - ContextMemory: in-process short-term key/value context with history
    -> candidate for vetinari/memory/context_memory.py
  - MemoryManager / get_memory_manager / init_memory_manager: unified facade
    combining SemanticMemoryStore + ContextMemory
    -> callers in vetinari/tools/tool_registry_integration.py (lines 283, 352, 365)
       must be updated to use DualMemoryStore + a context side-channel.
"""

import warnings as _warnings

_warnings.warn(
    "vetinari.enhanced_memory is deprecated. Use vetinari.memory.DualMemoryStore instead.",
    DeprecationWarning,
    stacklevel=2,
)

# ---------------------------------------------------------------------------
# Re-export canonical enum so `from vetinari.enhanced_memory import MemoryType`
# continues to work without pulling in legacy code.
# ---------------------------------------------------------------------------
from vetinari.types import MemoryType  # noqa: E402  (import after warning)

# ---------------------------------------------------------------------------
# Re-export canonical memory package symbols for callers that may have used
# them indirectly through this module.
# ---------------------------------------------------------------------------
from vetinari.memory import (  # noqa: E402
    DualMemoryStore,
    get_dual_memory_store,
    init_dual_memory_store,
    MemoryEntry as _CanonicalMemoryEntry,
)

# ---------------------------------------------------------------------------
# Legacy implementation - retained until callers are migrated.
#
# DO NOT add new code here. These classes exist solely so that existing
# import paths do not break while migration is in progress.
# ---------------------------------------------------------------------------

import os
import json
import logging
import sqlite3
import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class MemoryEntry:
    """A single memory entry with metadata.

    DEPRECATED: This class uses a different schema from vetinari.memory.MemoryEntry.
    Migrate to vetinari.memory.MemoryEntry (dataclass-based, dual-backend aware).
    """

    def __init__(self,
                 entry_id: str = None,
                 content: str = "",
                 memory_type: MemoryType = MemoryType.CONTEXT,
                 metadata: Dict[str, Any] = None,
                 tags: List[str] = None,
                 provenance: str = "",
                 embedding: List[float] = None):
        self.entry_id = entry_id or hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.tags = tags or []
        self.provenance = provenance
        self.embedding = embedding

        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.access_count = 0
        self.last_accessed = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "tags": self.tags,
            "provenance": self.provenance,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        entry = cls(
            entry_id=data.get("entry_id"),
            content=data.get("content", ""),
            memory_type=MemoryType(data.get("memory_type", "context")),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            provenance=data.get("provenance", ""),
            embedding=data.get("embedding"),
        )
        entry.created_at = data.get("created_at", entry.created_at)
        entry.updated_at = data.get("updated_at", entry.updated_at)
        entry.access_count = data.get("access_count", 0)
        entry.last_accessed = data.get("last_accessed", entry.last_accessed)
        return entry

    def update_content(self, new_content: str):
        """Update the content of this memory entry."""
        self.content = new_content
        self.updated_at = datetime.now().isoformat()


class SemanticMemoryStore:
    """SQLite-backed memory store with optional vector embeddings.

    DEPRECATED: Pending migration into vetinari/memory/ as a dedicated backend.
    No direct replacement available yet in vetinari.memory.
    """

    def __init__(self,
                 db_path: str = None,
                 enable_embeddings: bool = False,
                 embedding_model: str = "simple"):
        self.db_path = db_path or os.environ.get("VETINARI_MEMORY_DB", "./vetinari_memory.db")
        self.enable_embeddings = enable_embeddings
        self.embedding_model = embedding_model

        self._conn = None
        self._lock = threading.Lock()

        self._init_db()

        self._embedding_provider = None
        if enable_embeddings:
            self._init_embedding_provider()

        logger.info("SemanticMemoryStore initialized (db=%s, embeddings=%s)", self.db_path, enable_embeddings)

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                entry_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                metadata_json TEXT,
                tags_json TEXT,
                provenance TEXT,
                embedding_blob BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                entry_id,
                content,
                tags,
                content=memory_entries,
                content_rowid=rowid
            )
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory_entries BEGIN
                INSERT INTO memory_fts(rowid, entry_id, content, tags)
                VALUES (NEW.rowid, NEW.entry_id, NEW.content, NEW.tags_json);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory_entries BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, entry_id, content, tags)
                VALUES('delete', OLD.rowid, OLD.entry_id, OLD.content, OLD.tags_json);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memory_entries BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, entry_id, content, tags)
                VALUES('delete', OLD.rowid, OLD.entry_id, OLD.content, OLD.tags_json);
                INSERT INTO memory_fts(rowid, entry_id, content, tags)
                VALUES (NEW.rowid, NEW.entry_id, NEW.content, NEW.tags_json);
            END
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON memory_entries(tags_json)")

        self._conn.commit()

    def _init_embedding_provider(self):
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            self._embedding_provider = "sentence_transformers"
            logger.info("Using sentence-transformers for embeddings")
        except ImportError:
            try:
                import openai  # noqa: F401
                self._embedding_provider = "openai"
                logger.info("Using OpenAI for embeddings")
            except ImportError:
                logger.warning("No embedding provider available, using simple hashing")
                self._embedding_provider = "simple"

    def _get_embedding(self, text: str) -> List[float]:
        return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> List[float]:
        hash_val = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_val[:32]]

    def store(self, entry: MemoryEntry) -> bool:
        with self._lock:
            try:
                embedding_blob = None
                if self.enable_embeddings and entry.embedding is None:
                    entry.embedding = self._get_embedding(entry.content)

                if entry.embedding:
                    embedding_blob = json.dumps(entry.embedding)

                cursor = self._conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_entries
                    (entry_id, content, memory_type, metadata_json, tags_json, provenance,
                     embedding_blob, created_at, updated_at, access_count, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.entry_id,
                    entry.content,
                    entry.memory_type.value,
                    json.dumps(entry.metadata),
                    json.dumps(entry.tags),
                    entry.provenance,
                    embedding_blob,
                    entry.created_at,
                    entry.updated_at,
                    entry.access_count,
                    entry.last_accessed
                ))

                self._conn.commit()
                return True

            except sqlite3.Error as e:
                logger.error("Failed to store memory entry: %s", e)
                return False

    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM memory_entries WHERE entry_id = ?", (entry_id,))

            row = cursor.fetchone()
            if not row:
                return None

            cursor.execute("""
                UPDATE memory_entries
                SET access_count = access_count + 1, last_accessed = ?
                WHERE entry_id = ?
            """, (datetime.now().isoformat(), entry_id))
            self._conn.commit()

            return self._row_to_entry(row)

    def _row_to_entry(self, row) -> MemoryEntry:
        embedding = None
        if row["embedding_blob"]:
            embedding = json.loads(row["embedding_blob"])

        return MemoryEntry(
            entry_id=row["entry_id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
            tags=json.loads(row["tags_json"] or "[]"),
            provenance=row["provenance"] or "",
            embedding=embedding,
        )

    def search(self,
               query: str = None,
               memory_type: MemoryType = None,
               tags: List[str] = None,
               limit: int = 10,
               use_semantic: bool = False) -> List[MemoryEntry]:
        with self._lock:
            cursor = self._conn.cursor()

            conditions = []
            params = []

            if query:
                if use_semantic and self.enable_embeddings:
                    query_embedding = self._get_embedding(query)
                    cursor.execute("""
                        SELECT * FROM memory_entries
                        WHERE embedding_blob IS NOT NULL
                        LIMIT 100
                    """)

                    results = []
                    for row in cursor.fetchall():
                        entry = self._row_to_entry(row)
                        if entry.embedding:
                            sim = self._cosine_similarity(query_embedding, entry.embedding)
                            results.append((sim, entry))

                    results.sort(key=lambda x: x[0], reverse=True)
                    return [e for _, e in results[:limit]]
                else:
                    conditions.append("entry_id IN (SELECT entry_id FROM memory_fts WHERE memory_fts MATCH ?)")
                    params.append(query)

            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type.value)

            if tags:
                for tag in tags:
                    conditions.append("tags_json LIKE ?")
                    params.append(f'%"{tag}"%')

            sql = "SELECT * FROM memory_entries"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY access_count DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)

            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def get_recent(self,
                   memory_type: MemoryType = None,
                   since: datetime = None,
                   limit: int = 20) -> List[MemoryEntry]:
        with self._lock:
            cursor = self._conn.cursor()

            conditions = []
            params = []

            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type.value)

            if since:
                conditions.append("created_at >= ?")
                params.append(since.isoformat())

            sql = "SELECT * FROM memory_entries"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute("DELETE FROM memory_entries WHERE entry_id = ?", (entry_id,))
                self._conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.error("Failed to delete memory entry: %s", e)
                return False

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            cursor = self._conn.cursor()

            cursor.execute("SELECT COUNT(*) as total FROM memory_entries")
            total = cursor.fetchone()["total"]

            cursor.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memory_entries
                GROUP BY memory_type
            """)
            by_type = {row["memory_type"]: row["count"] for row in cursor.fetchall()}

            cursor.execute("SELECT SUM(access_count) as total_access FROM memory_entries")
            total_access = cursor.fetchone()["total_access"] or 0

            return {
                "total_entries": total,
                "by_type": by_type,
                "total_accesses": total_access,
                "db_path": self.db_path,
                "embeddings_enabled": self.enable_embeddings,
            }

    def prune(self, retention_days: int = 90) -> int:
        with self._lock:
            cutoff = datetime.now() - timedelta(days=retention_days)

            cursor = self._conn.cursor()
            cursor.execute("""
                DELETE FROM memory_entries
                WHERE created_at < ? AND access_count < 3
            """, (cutoff.isoformat(),))

            deleted = cursor.rowcount
            self._conn.commit()

            logger.info("Pruned %s old memory entries", deleted)
            return deleted

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


class ContextMemory:
    """In-process short-term key/value context store with history.

    DEPRECATED: Pending migration to vetinari/memory/context_memory.py.
    No direct replacement available yet in vetinari.memory.
    """

    def __init__(self):
        self._context: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100

    def set(self, key: str, value: Any):
        self._context[key] = value
        self._history.append({
            "action": "set",
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })

        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        return self._context.copy()

    def delete(self, key: str):
        if key in self._context:
            del self._context[key]
            self._history.append({
                "action": "delete",
                "key": key,
                "timestamp": datetime.now().isoformat()
            })

    def clear(self):
        self._context.clear()
        self._history.append({
            "action": "clear",
            "timestamp": datetime.now().isoformat()
        })

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._history[-limit:]


class MemoryManager:
    """Unified memory combining SemanticMemoryStore and ContextMemory.

    DEPRECATED: Callers should migrate to DualMemoryStore + a context
    side-channel. See vetinari/memory/__init__.py.

    Known callers:
      vetinari/tools/tool_registry_integration.py:283  (get_memory_manager)
      vetinari/tools/tool_registry_integration.py:352  (get_memory_manager)
      vetinari/tools/tool_registry_integration.py:365  (MemoryType)
    """

    def __init__(self,
                 db_path: str = None,
                 enable_semantic: bool = False):
        self.semantic = SemanticMemoryStore(db_path, enable_embeddings=enable_semantic)
        self.context = ContextMemory()

        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

    def remember(self,
                 content: str,
                 memory_type: MemoryType = MemoryType.CONTEXT,
                 metadata: Dict[str, Any] = None,
                 tags: List[str] = None,
                 provenance: str = "") -> str:
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            tags=tags or [],
            provenance=provenance or f"session:{self.session_id}"
        )

        self.semantic.store(entry)

        if metadata and "key" in metadata:
            self.context.set(metadata["key"], content)

        return entry.entry_id

    def recall(self,
               query: str = None,
               memory_type: MemoryType = None,
               tags: List[str] = None,
               limit: int = 5) -> List[MemoryEntry]:
        return self.semantic.search(
            query=query,
            memory_type=memory_type,
            tags=tags,
            limit=limit
        )

    def remember_decision(self,
                          decision: str,
                          rationale: str,
                          context: str = "",
                          tags: List[str] = None) -> str:
        content = f"Decision: {decision}\nRationale: {rationale}"
        if context:
            content += f"\nContext: {context}"

        return self.remember(
            content=content,
            memory_type=MemoryType.DECISION,
            metadata={"decision": decision, "rationale": rationale},
            tags=tags or ["decision"]
        )

    def remember_task_result(self,
                             task_id: str,
                             result: Any,
                             model_used: str = "",
                             tags: List[str] = None) -> str:
        content = f"Task {task_id} completed"
        if isinstance(result, dict):
            content += f"\nResult: {json.dumps(result, indent=2)}"
        else:
            content += f"\nResult: {result}"

        return self.remember(
            content=content,
            memory_type=MemoryType.RESULT,
            metadata={"task_id": task_id, "model_used": model_used},
            tags=tags or ["task", "result"]
        )

    def remember_knowledge(self,
                           topic: str,
                           content: str,
                           source: str = "",
                           tags: List[str] = None) -> str:
        return self.remember(
            content=f"Topic: {topic}\n{content}",
            memory_type=MemoryType.KNOWLEDGE,
            metadata={"topic": topic, "source": source},
            tags=tags or ["knowledge", topic]
        )

    def get_context(self, key: str = None) -> Any:
        if key:
            return self.context.get(key)
        return self.context.get_all()

    def set_context(self, **kwargs):
        for key, value in kwargs.items():
            self.context.set(key, value)

    def get_recent_context(self, limit: int = 10) -> List[Dict]:
        return self.context.get_history(limit)

    def get_stats(self) -> Dict[str, Any]:
        return self.semantic.get_stats()


# Global memory manager (legacy singleton)
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global MemoryManager.

    DEPRECATED: Use ``vetinari.memory.get_dual_memory_store()`` instead.
    """
    global _memory_manager
    if _memory_manager is None:
        enable_semantic = os.environ.get("VETINARI_ENABLE_SEMANTIC_MEMORY", "false").lower() in ("1", "true", "yes")
        db_path = os.environ.get("VETINARI_MEMORY_DB", "./vetinari_memory.db")
        _memory_manager = MemoryManager(db_path=db_path, enable_semantic=enable_semantic)
    return _memory_manager


def init_memory_manager(db_path: str = None, **kwargs) -> MemoryManager:
    """Initialize a new MemoryManager.

    DEPRECATED: Use ``vetinari.memory.init_dual_memory_store()`` instead.
    """
    global _memory_manager
    _memory_manager = MemoryManager(db_path=db_path, **kwargs)
    return _memory_manager


__all__ = [
    # Canonical re-exports (new code should import these directly)
    "MemoryType",
    "DualMemoryStore",
    "get_dual_memory_store",
    "init_dual_memory_store",
    # Legacy classes - pending migration
    "MemoryEntry",
    "SemanticMemoryStore",
    "ContextMemory",
    "MemoryManager",
    "get_memory_manager",
    "init_memory_manager",
]
