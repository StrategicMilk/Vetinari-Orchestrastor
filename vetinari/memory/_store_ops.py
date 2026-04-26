"""Internal CRUD and lifecycle helpers for UnifiedMemoryStore.

Covers row conversion, entry storage, forgetting, updating, compaction,
stats, and eviction for the ``memories`` table.  Episode and search
operations live in ``_store_episode.py`` and ``_store_search.py``
respectively.

Not part of the public API — import only from ``vetinari.memory.unified``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .interfaces import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public re-exports from sub-modules (keeps existing callers working)
# ---------------------------------------------------------------------------

from ._store_episode import (  # noqa: E402 - late import is required after bootstrap setup
    evict_old_episodes,
    get_episode_stats,
    get_failure_patterns,
    insert_episode,
    recall_episodes_from_db,
    record_episode_full,
    row_to_episode_dict,
)
from ._store_search import (  # noqa: E402 - late import is required after bootstrap setup
    build_timeline,
    fts_search,
    is_semantic_duplicate,
    like_search,
    manual_cosine_search,
    vec_knn_search,
)

__all__ = [
    "build_timeline",
    "compact_memories",
    "evict_low_importance_memories",
    "evict_old_episodes",
    "export_memories",
    "filter_entry_secrets",
    "forget_memory",
    "fts_search",
    "get_entry_by_id",
    "get_episode_stats",
    "get_fact_chain",
    "get_failure_patterns",
    "get_memory_stats",
    "get_superseded_ids",
    "insert_episode",
    "is_semantic_duplicate",
    "like_search",
    "manual_cosine_search",
    "recall_episodes_from_db",
    "record_episode_full",
    "row_to_entry",
    "row_to_episode_dict",
    "set_relationship",
    "store_memory_entry",
    "update_memory_content",
    "vec_knn_search",
]


# ---------------------------------------------------------------------------
# Row conversion helper
# ---------------------------------------------------------------------------


def row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    """Convert a memories table row to a MemoryEntry.

    Args:
        row: sqlite3.Row from the memories table.

    Returns:
        Populated :class:`~vetinari.memory.interfaces.MemoryEntry`.
    """
    try:
        entry_type = MemoryType(row["entry_type"])
    except ValueError:
        entry_type = MemoryType.DISCOVERY

    metadata = None
    if row["metadata_json"]:
        try:
            metadata = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            metadata = None

    # Safely read columns that may not exist in pre-migration databases
    def _col(name: str, default: Any = None) -> Any:
        try:
            return row[name]
        except (IndexError, KeyError):
            logger.warning(
                "Column %r not found in memory row — database may predate this schema migration, using default",
                name,
            )
            return default

    return MemoryEntry(
        id=row["id"],
        agent=row["agent"],
        entry_type=entry_type,
        content=row["content"],
        summary=row["summary"],
        timestamp=row["timestamp"],
        provenance=row["provenance"],
        source_backends=["unified"],
        metadata=metadata,
        recall_count=_col("recall_count", 0) or 0,
        supersedes_id=_col("supersedes_id"),
        relationship_type=_col("relationship_type"),
        scope=_col("scope", "global") or "global",
        last_accessed=_col("last_accessed", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Security sanitisation
# ---------------------------------------------------------------------------


def filter_entry_secrets(entry: Any) -> Any:
    """Sanitize a MemoryEntry's content and metadata to remove secrets.

    Uses the global secret scanner singleton.  Mutates and returns the entry
    in place (MemoryEntry is not frozen).

    Args:
        entry: :class:`~vetinari.memory.interfaces.MemoryEntry` to sanitize.

    Returns:
        The same entry with secrets scrubbed from content and metadata.
    """
    from vetinari.security import get_secret_scanner

    scanner = get_secret_scanner()
    if entry.content:
        sanitized = scanner.sanitize(entry.content)
        if sanitized != entry.content:
            logger.debug("Entry content contained secrets — sanitized")
            entry.content = sanitized
    if entry.metadata and isinstance(entry.metadata, dict):
        sanitized_meta = scanner.sanitize_dict(entry.metadata)
        if sanitized_meta != entry.metadata:
            logger.debug("Entry metadata contained secrets — sanitized")
            entry.metadata = sanitized_meta
    return entry


# ---------------------------------------------------------------------------
# Entry storage
# ---------------------------------------------------------------------------


def store_memory_entry(
    conn: sqlite3.Connection,
    entry: Any,
    store_embedding_fn: Any,
) -> str:
    """Persist a memory entry with content hash deduplication.

    Skips the write when an identical content hash already exists.
    Calls ``store_embedding_fn(memory_id, text)`` after a successful insert.

    Args:
        conn: Active SQLite connection.
        entry: :class:`~vetinari.memory.interfaces.MemoryEntry` to persist.
        store_embedding_fn: Callable ``(memory_id: str, text: str) -> None``
            invoked after a successful insert for best-effort embedding storage.

    Returns:
        The ``entry.id`` that was stored, or the existing ID when deduplication
        skips the write.

    Raises:
        vetinari.exceptions.StorageError: If the database INSERT fails.
    """
    from vetinari.exceptions import StorageError

    from .interfaces import content_hash

    now = datetime.now(timezone.utc).isoformat()
    c_hash = content_hash(entry.content)

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM memories WHERE content_hash = ? AND forgotten = 0", (c_hash,))
    existing = cursor.fetchone()
    if existing:
        logger.debug("Skipping duplicate memory (hash match): %s", existing["id"])
        return existing["id"]

    try:
        cursor.execute(
            """INSERT INTO memories
               (id, agent, entry_type, content, summary, timestamp,
                provenance, content_hash, forgotten, access_count,
                quality_score, importance, created_at, updated_at, metadata_json,
                scope, recall_count, supersedes_id, relationship_type, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0.0, 0.5, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.agent,
                entry.entry_type.value if isinstance(entry.entry_type, MemoryType) else str(entry.entry_type),
                entry.content,
                entry.summary,
                entry.timestamp,
                entry.provenance,
                c_hash,
                now,
                now,
                json.dumps(entry.metadata) if entry.metadata else None,
                entry.scope,
                entry.recall_count,
                entry.supersedes_id,
                entry.relationship_type,
                entry.last_accessed,
            ),
        )
        conn.commit()
        store_embedding_fn(entry.id, entry.content)
        logger.debug("Stored memory %s (agent=%s, type=%s)", entry.id, entry.agent, entry.entry_type)
        return entry.id
    except sqlite3.Error as exc:
        logger.error("Failed to store memory %s: %s", entry.id, exc)
        raise StorageError(f"Memory store write failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_memories(conn: sqlite3.Connection, path: str) -> bool:
    """Export all non-forgotten memories to a JSON file.

    Args:
        conn: Active SQLite connection.
        path: Output file path (UTF-8 encoded JSON).

    Returns:
        True on success, False when an error occurs.
    """
    from pathlib import Path

    try:
        rows = conn.execute("SELECT * FROM memories WHERE forgotten = 0 ORDER BY timestamp DESC").fetchall()
        entries = [row_to_entry(row).to_dict() for row in rows]
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump({"exported_at": datetime.now(timezone.utc).isoformat(), "entries": entries}, f, indent=2)
        return True
    except Exception as exc:
        logger.error("Export failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Soft delete and content update
# ---------------------------------------------------------------------------


def forget_memory(conn: sqlite3.Connection, entry_id: str, reason: str) -> bool:
    """Mark a memory as forgotten (soft delete).

    Args:
        conn: Active SQLite connection.
        entry_id: The entry ID to forget.
        reason: Reason for forgetting (logged at INFO).

    Returns:
        True if the entry was found and marked.
    """
    cursor = conn.execute(
        "UPDATE memories SET forgotten = 1, updated_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), entry_id),
    )
    conn.commit()
    if cursor.rowcount > 0:
        logger.info("Forgot memory %s: %s", entry_id, reason)
        return True
    return False


def update_memory_content(conn: sqlite3.Connection, entry_id: str, new_content: str) -> bool:
    """Update the content of an existing memory entry.

    Args:
        conn: Active SQLite connection.
        entry_id: The entry ID to update.
        new_content: Replacement content string.

    Returns:
        True if the entry was found and updated.
    """
    from .interfaces import content_hash

    cursor = conn.execute(
        "UPDATE memories SET content = ?, content_hash = ?, updated_at = ? WHERE id = ?",
        (new_content, content_hash(new_content), datetime.now(timezone.utc).isoformat(), entry_id),
    )
    if cursor.rowcount > 0:
        conn.execute("DELETE FROM embeddings WHERE memory_id = ?", (entry_id,))
        with contextlib.suppress(sqlite3.Error):
            conn.execute("DELETE FROM memory_vec WHERE memory_id = ?", (entry_id,))
    conn.commit()
    if cursor.rowcount > 0:
        logger.debug("Updated memory %s content", entry_id)
        return True
    return False


# ---------------------------------------------------------------------------
# Compaction and capacity management
# ---------------------------------------------------------------------------


def compact_memories(conn: sqlite3.Connection, max_age_days: int | None) -> int:
    """Remove forgotten entries and optionally prune old data.

    Args:
        conn: Active SQLite connection.
        max_age_days: Remove non-frequently-accessed entries older than this.

    Returns:
        Number of entries removed.
    """
    import time as _time

    deleted = 0
    conn.execute("DELETE FROM memories WHERE forgotten = 1")
    deleted += conn.execute("SELECT changes()").fetchone()[0]

    if max_age_days is not None:
        cutoff = int((_time.time() - max_age_days * 86400) * 1000)
        conn.execute(
            "DELETE FROM memories WHERE timestamp < ? AND access_count < 3",
            (cutoff,),
        )
        deleted += conn.execute("SELECT changes()").fetchone()[0]

    conn.commit()
    logger.info("Compacted memory store: removed %d entries", deleted)
    return deleted


def evict_low_importance_memories(conn: sqlite3.Connection, max_entries: int) -> None:
    """Evict the least-important long-term memory entries when over capacity.

    Importance is computed as quality_score * (access_count + 1) * recency_decay.
    Removes entries until the count is ``max_entries`` minus a 5% buffer.

    Args:
        conn: Active SQLite connection.
        max_entries: Maximum allowed memory count.
    """
    total = conn.execute("SELECT COUNT(*) as cnt FROM memories WHERE forgotten = 0").fetchone()["cnt"]
    if total <= max_entries:
        return
    evict_count = total - max_entries + (max_entries // 20)  # 5% buffer
    # Ebbinghaus-based eviction: rank by retention strength (ADR-0071).
    # Uses SQL approximation of the Ebbinghaus formula for in-DB sorting.
    # importance * exp(-0.16 * (1 - importance*0.8) * days) * (1 + recall_count*0.2)
    try:
        conn.execute(
            """DELETE FROM memories WHERE id IN (
                 SELECT id FROM memories
                 WHERE forgotten = 0
                 ORDER BY (
                     importance *
                     EXP(-0.16 * (1.0 - importance * 0.8) *
                         MAX(0, (julianday('now') - julianday(created_at)))) *
                     (1.0 + COALESCE(recall_count, 0) * 0.2)
                 ) ASC LIMIT ?
               )""",
            (evict_count,),
        )
        conn.commit()
        logger.info("Evicted low-importance memories (count=%d)", evict_count)
    except sqlite3.Error as exc:
        logger.warning("Memory eviction failed: %s", exc)


# ---------------------------------------------------------------------------
# Stats and single-entry fetch
# ---------------------------------------------------------------------------


def get_memory_stats(conn: sqlite3.Connection, db_path_fn: Any) -> dict[str, Any]:
    """Compute aggregate statistics from the memories table.

    Args:
        conn: Active SQLite connection.
        db_path_fn: Callable that returns the database Path (for file size).

    Returns:
        Dictionary with total_entries, file_size_bytes, oldest/newest timestamps,
        and per-agent/per-type counts.
    """
    total = conn.execute("SELECT COUNT(*) as total FROM memories WHERE forgotten = 0").fetchone()["total"]
    row = conn.execute(
        "SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM memories WHERE forgotten = 0"
    ).fetchone()
    oldest = row["oldest"] or 0
    newest = row["newest"] or 0
    by_agent = {
        r["agent"]: r["cnt"]
        for r in conn.execute(
            "SELECT agent, COUNT(*) as cnt FROM memories WHERE forgotten = 0 GROUP BY agent"
        ).fetchall()
    }
    by_type = {
        r["entry_type"]: r["cnt"]
        for r in conn.execute(
            "SELECT entry_type, COUNT(*) as cnt FROM memories WHERE forgotten = 0 GROUP BY entry_type"
        ).fetchall()
    }
    try:
        file_size = db_path_fn().stat().st_size
    except OSError:
        file_size = 0
    return {
        "total_entries": total,
        "file_size_bytes": file_size,
        "oldest_entry": oldest,
        "newest_entry": newest,
        "entries_by_agent": by_agent,
        "entries_by_type": by_type,
    }


def get_fact_chain(conn: sqlite3.Connection, entry_id: str) -> list[MemoryEntry]:
    """Walk the supersedes_id chain from newest to oldest.

    Starting from *entry_id*, follows supersedes_id links until the chain
    ends (NULL) or a cycle is detected.  Only non-forgotten entries are
    included.

    Args:
        conn: Active SQLite connection.
        entry_id: Starting entry ID (typically the newest in the chain).

    Returns:
        Ordered list of MemoryEntry from newest to oldest.
    """
    chain: list[MemoryEntry] = []
    visited: set[str] = set()
    current_id: str | None = entry_id
    while current_id and current_id not in visited:
        visited.add(current_id)
        row = conn.execute("SELECT * FROM memories WHERE id = ? AND forgotten = 0", (current_id,)).fetchone()
        if row is None:
            break
        chain.append(row_to_entry(row))
        try:
            current_id = row["supersedes_id"]
        except (IndexError, KeyError):
            break
    return chain


def set_relationship(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    relationship_type: str,
) -> bool:
    """Create a typed relationship from source to target memory.

    Sets ``supersedes_id`` and ``relationship_type`` on the source entry
    to link it to the target.

    Args:
        conn: Active SQLite connection.
        source_id: The newer entry that supersedes/relates to target.
        target_id: The older entry being referenced.
        relationship_type: One of :class:`~vetinari.types.RelationshipType` values.

    Returns:
        True if the source entry was found and updated.
    """
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE memories SET supersedes_id = ?, relationship_type = ?, updated_at = ? WHERE id = ? AND forgotten = 0",
        (target_id, relationship_type, now, source_id),
    )
    conn.commit()
    if cursor.rowcount > 0:
        logger.debug("Relationship %s -> %s (%s)", source_id, target_id, relationship_type)
        return True
    return False


def get_superseded_ids(conn: sqlite3.Connection) -> set[str]:
    """Return IDs of entries that have been superseded by another live entry.

    An entry is superseded when another non-forgotten entry references it
    via ``supersedes_id``.

    Args:
        conn: Active SQLite connection.

    Returns:
        Set of memory IDs that are superseded.
    """
    rows = conn.execute(
        "SELECT supersedes_id FROM memories WHERE supersedes_id IS NOT NULL AND forgotten = 0"
    ).fetchall()
    return {row[0] for row in rows}


def get_entry_by_id(conn: sqlite3.Connection, entry_id: str) -> Any | None:
    """Fetch a memory entry by ID, incrementing its access count.

    Args:
        conn: Active SQLite connection.
        entry_id: The entry ID to fetch.

    Returns:
        The sqlite3.Row if found and not forgotten, otherwise None.
    """
    row = conn.execute("SELECT * FROM memories WHERE id = ? AND forgotten = 0", (entry_id,)).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE memories SET access_count = access_count + 1, recall_count = COALESCE(recall_count, 0) + 1, updated_at = ? WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), entry_id),
    )
    conn.commit()
    return row
