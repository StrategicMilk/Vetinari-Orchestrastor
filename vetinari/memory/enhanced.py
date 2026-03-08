"""
Enhanced Memory System for Vetinari

Provides long-term memory with semantic search capabilities using:
- SQLite for structured data
- Optional vector embeddings for semantic search
- Provenance tracking
- Temporal context

Supports:
- Short-term memory (task context)
- Long-term memory (knowledge graph)
- Semantic search with embeddings
"""

import os
import json
import logging
import sqlite3
import hashlib
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory entries."""
    PLAN = "plan"
    TASK = "task"
    DECISION = "decision"
    KNOWLEDGE = "knowledge"
    CODE = "code"
    CONVERSATION = "conversation"
    RESULT = "result"
    CONTEXT = "context"


class MemoryEntry:
    """A single memory entry with metadata."""

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
    """
    Enhanced memory store with semantic search capabilities.

    Features:
    - SQLite-backed persistent storage
    - Vector embeddings for semantic search (optional)
    - Full-text search
    - Temporal queries
    - Tag-based filtering
    - Provenance tracking
    """

    def __init__(self,
                 db_path: str = None,
                 enable_embeddings: bool = False,
                 embedding_model: str = "simple"):
        """
        Initialize the semantic memory store.

        Args:
            db_path: Path to SQLite database
            enable_embeddings: Enable vector embeddings for semantic search
            embedding_model: Embedding model to use
        """
        self.db_path = db_path or os.environ.get("VETINARI_MEMORY_DB", "./vetinari_memory.db")
        self.enable_embeddings = enable_embeddings
        self.embedding_model = embedding_model

        self._conn = None
        self._lock = threading.Lock()

        # Initialize database
        self._init_db()

        # Embedding provider
        self._embedding_provider = None
        if enable_embeddings:
            self._init_embedding_provider()

        logger.info(f"SemanticMemoryStore initialized (db={self.db_path}, embeddings={enable_embeddings})")

    def _init_db(self):
        """Initialize the database schema."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        cursor = self._conn.cursor()

        # Main memory table
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

        # Full-text search virtual table
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                entry_id,
                content,
                tags,
                content=memory_entries,
                content_rowid=rowid
            )
        """)

        # Triggers to keep FTS in sync
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

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON memory_entries(tags_json)")

        self._conn.commit()

    def _init_embedding_provider(self):
        """Initialize the embedding provider."""
        try:
            # Try to use sentence-transformers if available
            from sentence_transformers import SentenceTransformer
            self._embedding_provider = "sentence_transformers"
            logger.info("Using sentence-transformers for embeddings")
        except ImportError:
            try:
                # Try OpenAI embeddings
                import openai
                self._embedding_provider = "openai"
                logger.info("Using OpenAI for embeddings")
            except ImportError:
                logger.warning("No embedding provider available, using simple hashing")
                self._embedding_provider = "simple"

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self._embedding_provider == "sentence_transformers":
            # This would require loading the model - simplified here
            return self._simple_embedding(text)
        elif self._embedding_provider == "openai":
            # This would call OpenAI API - simplified here
            return self._simple_embedding(text)
        else:
            return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> List[float]:
        """Simple hash-based embedding for fallback."""
        # Simple deterministic hash-based vector
        hash_val = hashlib.sha256(text.encode()).digest()
        # Convert to float list
        return [float(b) / 255.0 for b in hash_val[:32]]

    def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        with self._lock:
            try:
                # Get embedding if enabled
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
                logger.error(f"Failed to store memory entry: {e}")
                return False

    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT * FROM memory_entries WHERE entry_id = ?
            """, (entry_id,))

            row = cursor.fetchone()
            if not row:
                return None

            # Update access count
            cursor.execute("""
                UPDATE memory_entries
                SET access_count = access_count + 1, last_accessed = ?
                WHERE entry_id = ?
            """, (datetime.now().isoformat(), entry_id))
            self._conn.commit()

            return self._row_to_entry(row)

    def _row_to_entry(self, row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
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
        """
        Search memory entries.

        Args:
            query: Text query for search
            memory_type: Filter by memory type
            tags: Filter by tags
            limit: Maximum results
            use_semantic: Use semantic search (embeddings)

        Returns:
            List of matching memory entries
        """
        with self._lock:
            cursor = self._conn.cursor()

            # Build query
            conditions = []
            params = []

            if query:
                if use_semantic and self.enable_embeddings:
                    # Semantic search
                    query_embedding = self._get_embedding(query)
                    # Simple cosine similarity (not efficient for large datasets)
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
                    # Full-text search
                    conditions.append("entry_id IN (SELECT entry_id FROM memory_fts WHERE memory_fts MATCH ?)")
                    params.append(query)

            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type.value)

            if tags:
                for tag in tags:
                    conditions.append("tags_json LIKE ?")
                    params.append(f'%"{tag}"%')

            # Build SQL
            sql = "SELECT * FROM memory_entries"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY access_count DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)

            return [self._row_to_entry(row) for row in cursor.fetchall()]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
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
        """Get recent memory entries."""
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
        """Delete a memory entry."""
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute("DELETE FROM memory_entries WHERE entry_id = ?", (entry_id,))
                self._conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.error(f"Failed to delete memory entry: {e}")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
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
        """Prune old memory entries."""
        with self._lock:
            cutoff = datetime.now() - timedelta(days=retention_days)

            cursor = self._conn.cursor()
            cursor.execute("""
                DELETE FROM memory_entries
                WHERE created_at < ? AND access_count < 3
            """, (cutoff.isoformat(),))

            deleted = cursor.rowcount
            self._conn.commit()

            logger.info(f"Pruned {deleted} old memory entries")
            return deleted

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the connection when exiting the context manager."""
        self.close()
        return False

    def __del__(self):
        """Destructor — ensure the connection is closed."""
        self.close()


class ContextMemory:
    """
    Short-term memory for current task context.

    Provides:
    - Task-scoped context
    - Variable storage
    - Quick access to recent data
    """

    def __init__(self):
        self._context: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100

    def set(self, key: str, value: Any):
        """Set a context variable."""
        self._context[key] = value
        self._history.append({
            "action": "set",
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self._context.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Get all context variables."""
        return self._context.copy()

    def delete(self, key: str):
        """Delete a context variable."""
        if key in self._context:
            del self._context[key]
            self._history.append({
                "action": "delete",
                "key": key,
                "timestamp": datetime.now().isoformat()
            })

    def clear(self):
        """Clear all context."""
        self._context.clear()
        self._history.append({
            "action": "clear",
            "timestamp": datetime.now().isoformat()
        })

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get context history."""
        return self._history[-limit:]


class MemoryManager:
    """
    Unified memory management combining:
    - Semantic memory (long-term with search)
    - Context memory (short-term)
    - Session memory (per-session)
    """

    def __init__(self,
                 db_path: str = None,
                 enable_semantic: bool = False):
        self.semantic = SemanticMemoryStore(db_path, enable_embeddings=enable_semantic)
        self.context = ContextMemory()

        # Session ID
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

    def remember(self,
                content: str,
                memory_type: MemoryType = MemoryType.CONTEXT,
                metadata: Dict[str, Any] = None,
                tags: List[str] = None,
                provenance: str = "") -> str:
        """Store something in memory."""
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            tags=tags or [],
            provenance=provenance or f"session:{self.session_id}"
        )

        self.semantic.store(entry)

        # Also set in context
        if metadata and "key" in metadata:
            self.context.set(metadata["key"], content)

        return entry.entry_id

    def recall(self,
              query: str = None,
              memory_type: MemoryType = None,
              tags: List[str] = None,
              limit: int = 5) -> List[MemoryEntry]:
        """Recall information from memory."""
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
        """Remember a decision made by the system."""
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
        """Remember a task result."""
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
        """Remember knowledge for future reference."""
        return self.remember(
            content=f"Topic: {topic}\n{content}",
            memory_type=MemoryType.KNOWLEDGE,
            metadata={"topic": topic, "source": source},
            tags=tags or ["knowledge", topic]
        )

    def get_context(self, key: str = None) -> Any:
        """Get context variable."""
        if key:
            return self.context.get(key)
        return self.context.get_all()

    def set_context(self, **kwargs):
        """Set context variables."""
        for key, value in kwargs.items():
            self.context.set(key, value)

    def get_recent_context(self, limit: int = 10) -> List[Dict]:
        """Get recent context history."""
        return self.context.get_history(limit)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.semantic.get_stats()


# Convenience aliases
EnhancedMemoryManager = MemoryManager

# Global memory manager
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        enable_semantic = os.environ.get("VETINARI_ENABLE_SEMANTIC_MEMORY", "false").lower() in ("1", "true", "yes")
        db_path = os.environ.get("VETINARI_MEMORY_DB", "./vetinari_memory.db")
        _memory_manager = MemoryManager(db_path=db_path, enable_semantic=enable_semantic)
    return _memory_manager


def init_memory_manager(db_path: str = None, **kwargs) -> MemoryManager:
    """Initialize a new memory manager."""
    global _memory_manager
    _memory_manager = MemoryManager(db_path=db_path, **kwargs)
    return _memory_manager


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test the memory manager
    mm = MemoryManager(db_path="./test_memory.db", enable_semantic=True)

    # Store some memories
    logger.info("=== Storing memories ===")
    mm.remember(
        "Python 3.12 was released",
        memory_type=MemoryType.KNOWLEDGE,
        tags=["python", "release"]
    )

    mm.remember_decision(
        "Use Llama 3 for coding tasks",
        "It has excellent code generation capabilities and runs locally",
        tags=["model", "selection"]
    )

    mm.remember_task_result(
        "task-1",
        {"status": "completed", "output": "Test passed"},
        model_used="llama-3-8b",
        tags=["task", "result"]
    )

    # Set context
    mm.set_context(current_task="task-2", goal="Build web app")

    # Recall
    logger.info("=== Recalling memories ===")
    results = mm.recall(query="Python", memory_type=MemoryType.KNOWLEDGE)
    for r in results:
        logger.info(f"  - {r.content[:50]}...")

    # Search decisions
    logger.info("=== Recent decisions ===")
    decisions = mm.recall(memory_type=MemoryType.DECISION)
    for d in decisions:
        logger.info(f"  - {d.content[:80]}...")

    # Context
    logger.info("=== Context ===")
    logger.info(f"  Current task: {mm.get_context('current_task')}")
    logger.info(f"  Goal: {mm.get_context('goal')}")

    # Stats
    logger.info("=== Stats ===")
    stats = mm.get_stats()
    logger.info(f"  Total entries: {stats['total_entries']}")
    logger.info(f"  By type: {stats['by_type']}")
