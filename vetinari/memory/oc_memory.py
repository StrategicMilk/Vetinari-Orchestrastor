"""OcMemoryStore - Adapter for oc-mnemoria-style persistent shared memory.

This adapter provides the IMemoryStore interface using a local file-based
storage with SQLite-like features for indexing and querying.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from .interfaces import OC_MEMORY_PATH, IMemoryStore, MemoryEntry, MemoryEntryType, MemoryStats, content_hash

logger = logging.getLogger(__name__)


class OcMemoryStore(IMemoryStore):
    """Memory store adapter using oc-mnemoria style storage."""

    def __init__(self, path: str = OC_MEMORY_PATH):
        self.path = Path(path)
        self.db_path = self.path / "memories.db"
        self._conn = None
        self._init_storage()

    def _init_storage(self):
        """Initialize the storage."""
        self.path.mkdir(parents=True, exist_ok=True)

        try:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

            cursor = self._conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    agent TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    provenance TEXT NOT NULL,
                    source_backends TEXT NOT NULL,
                    forgotten INTEGER DEFAULT 0,
                    content_hash TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(entry_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(forgotten)
            """)

            self._conn.commit()
            logger.info("OcMemoryStore initialized at %s", self.db_path)

        except sqlite3.Error as e:
            logger.error("Failed to initialize OcMemoryStore: %s", e)
            raise

    def remember(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        if "oc" not in entry.source_backends:
            entry.source_backends = [*entry.source_backends, "oc"]

        entry_hash = content_hash(entry.content)

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, agent, entry_type, content, summary, timestamp, provenance,
             source_backends, content_hash, forgotten)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
            (
                entry.id,
                entry.agent,
                entry.entry_type.value if isinstance(entry.entry_type, MemoryEntryType) else entry.entry_type,
                entry.content,
                entry.summary,
                entry.timestamp,
                entry.provenance,
                json.dumps(entry.source_backends),
                entry_hash,
            ),
        )
        self._conn.commit()

        logger.debug("OcMemoryStore: stored entry %s", entry.id)
        return entry.id

    def search(
        self, query: str, agent: str | None = None, entry_types: list[str] | None = None, limit: int = 10
    ) -> list[MemoryEntry]:
        """Search memories by keyword."""
        cursor = self._conn.cursor()

        sql = "SELECT * FROM memories WHERE forgotten = 0 AND (content LIKE ? OR summary LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]

        if agent:
            sql += " AND agent = ?"
            params.append(agent)

        if entry_types:
            placeholders = ",".join(["?"] * len(entry_types))
            sql += f" AND entry_type IN ({placeholders})"
            params.extend(entry_types)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def timeline(
        self, agent: str | None = None, start_time: int | None = None, end_time: int | None = None, limit: int = 100
    ) -> list[MemoryEntry]:
        """Browse memories chronologically."""
        cursor = self._conn.cursor()

        sql = "SELECT * FROM memories WHERE forgotten = 0"
        params = []

        if agent:
            sql += " AND agent = ?"
            params.append(agent)

        if start_time is not None:
            sql += " AND timestamp >= ?"
            params.append(start_time)

        if end_time is not None:
            sql += " AND timestamp <= ?"
            params.append(end_time)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def ask(self, question: str, agent: str | None = None) -> list[MemoryEntry]:
        """Ask a natural language question (simplified: keyword search)."""
        keywords = question.lower().split()
        if not keywords:
            return []

        query = " ".join(keywords[:3])  # Use first 3 keywords
        return self.search(query, agent=agent, limit=5)

    def export(self, path: str) -> bool:
        """Export memories to JSON."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM memories ORDER BY timestamp DESC")
        rows = cursor.fetchall()

        memories = [self._row_to_entry(row).to_dict() for row in rows]

        with open(path, "w") as f:
            json.dump({"memories": memories, "exported_at": datetime.now().isoformat()}, f, indent=2)

        logger.info("OcMemoryStore: exported %d entries to %s", len(memories), path)
        return True

    def forget(self, entry_id: str, reason: str) -> bool:
        """Mark a memory as forgotten (tombstone)."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            UPDATE memories SET forgotten = 1 WHERE id = ?
        """,
            (entry_id,),
        )
        self._conn.commit()

        logger.debug("OcMemoryStore: marked entry %s as forgotten", entry_id)
        return cursor.rowcount > 0

    def compact(self, max_age_days: int | None = None) -> int:
        """Remove forgotten entries and optionally prune old data."""
        cursor = self._conn.cursor()

        deleted = 0

        cursor.execute("DELETE FROM memories WHERE forgotten = 1")
        deleted += cursor.rowcount

        if max_age_days is not None:
            import time

            cutoff = int((time.time() - max_age_days * 24 * 3600) * 1000)
            cursor.execute("DELETE FROM memories WHERE timestamp < ?", (cutoff,))
            deleted += cursor.rowcount

        self._conn.commit()

        logger.info("OcMemoryStore: compacted, removed %d entries", deleted)
        return deleted

    def stats(self) -> MemoryStats:
        """Get memory store statistics."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories WHERE forgotten = 0")
        total = cursor.fetchone()[0]

        file_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM memories WHERE forgotten = 0")
        row = cursor.fetchone()
        oldest = row[0] or 0
        newest = row[1] or 0

        cursor.execute("""
            SELECT agent, COUNT(*) as cnt FROM memories
            WHERE forgotten = 0 GROUP BY agent
        """)
        by_agent = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT entry_type, COUNT(*) as cnt FROM memories
            WHERE forgotten = 0 GROUP BY entry_type
        """)
        by_type = {row[0]: row[1] for row in cursor.fetchall()}

        return MemoryStats(
            total_entries=total,
            file_size_bytes=file_size,
            oldest_entry=oldest,
            newest_entry=newest,
            entries_by_agent=by_agent,
            entries_by_type=by_type,
        )

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry by ID."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (entry_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_entry(row)
        return None

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            agent=row["agent"],
            entry_type=MemoryEntryType(row["entry_type"]),
            content=row["content"],
            summary=row["summary"],
            timestamp=row["timestamp"],
            provenance=row["provenance"],
            source_backends=json.loads(row["source_backends"]),
        )

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
