"""Embedding helpers for the memory subsystem.

Handles interaction with an OpenAI-compatible /v1/embeddings endpoint
for generating dense vector representations of memory content, and
provides binary pack/unpack utilities for storing those vectors in SQLite
BLOB columns.

sqlite-vec extension management is handled by the caller (``unified.py``)
which owns the connection lifecycle and optional KNN virtual table setup.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct

from vetinari.constants import DEFAULT_EMBEDDING_API_URL, EMBEDDING_API_TIMEOUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrors unified.py module-level env reads)
# ---------------------------------------------------------------------------

EMBEDDING_API_URL = DEFAULT_EMBEDDING_API_URL
EMBEDDING_MODEL = os.environ.get("VETINARI_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
# Expected output dimension from the default embedding model.
EMBED_DIM = 768  # nomic-embed-text-v1.5 produces 768-dim vectors


# ---------------------------------------------------------------------------
# Binary pack/unpack
# ---------------------------------------------------------------------------


def pack_embedding(vec: list[float]) -> bytes:
    """Pack a float list into compact float32 bytes (little-endian).

    Args:
        vec: Float list to pack.

    Returns:
        Binary blob of ``len(vec) * 4`` bytes.
    """
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_embedding(blob: bytes) -> list[float]:
    """Unpack float32 bytes back to a float list.

    Args:
        blob: Binary blob previously produced by :func:`pack_embedding`.

    Returns:
        Float list of length ``len(blob) // 4``.
    """
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


# ---------------------------------------------------------------------------
# Local inference endpoint
# ---------------------------------------------------------------------------


def embed_via_local_inference(
    text: str,
    api_url: str = EMBEDDING_API_URL,
    model: str = EMBEDDING_MODEL,
) -> list[float] | None:
    """Get an embedding from an OpenAI-compatible /v1/embeddings endpoint.

    Configured via ``VETINARI_EMBEDDING_API_URL`` and
    ``VETINARI_EMBEDDING_MODEL`` environment variables.  Returns ``None``
    when the endpoint is unreachable so callers can fall back gracefully
    to FTS5 text search.

    Args:
        text: Text to embed.
        api_url: Base URL of the embedding API (default from env).
        model: Embedding model identifier (default from env).

    Returns:
        Float list embedding vector, or None if the endpoint is unreachable.
    """
    if not text.strip():
        # Empty text produces an all-zeros vector from most endpoints, which
        # makes cosine_similarity return 0.0 for every comparison — undefined
        # similarity. Return a unit vector so comparisons are at least defined.
        return [1.0] + [0.0] * (EMBED_DIM - 1)

    try:
        import httpx

        payload = json.dumps({"input": text, "model": model}).encode("utf-8")
        resp = httpx.post(
            f"{api_url}/v1/embeddings",
            content=payload,
            headers={"Content-Type": "application/json"},
            timeout=EMBEDDING_API_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as exc:
        logger.warning("Embedding endpoint unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# sqlite-vec extension helpers
# ---------------------------------------------------------------------------


def sqlite_vec_available() -> bool:
    """Return True if the sqlite-vec Python package can be imported.

    Only checks importability -- does not attempt to load the extension into
    a connection.  Use :func:`load_sqlite_vec` to actually enable KNN search.

    Returns:
        True when ``import sqlite_vec`` succeeds without error.
    """
    try:
        import sqlite_vec  # noqa: F401 - import intentionally probes or re-exports API surface

        return True
    except ImportError:
        logger.warning("sqlite_vec not installed — KNN vector search unavailable")
        return False


def load_sqlite_vec(conn: sqlite3.Connection | None) -> bool:
    """Load the sqlite-vec extension into *conn*, enabling KNN vector search.

    Must be called before creating or querying the ``memory_vec`` virtual
    table.  Silently returns False when the extension is unavailable or when
    *conn* is None so callers can treat it as optional.

    Args:
        conn: Active SQLite connection to load the extension into, or None.

    Returns:
        True if the extension was loaded successfully, False otherwise.
    """
    if conn is None:
        return False
    try:
        import sqlite_vec

        conn.load_extension(sqlite_vec.loadable_path())
        return True
    except (ImportError, AttributeError, sqlite3.OperationalError) as exc:
        logger.warning("sqlite-vec extension not available: %s — falling back to linear scan for memory search", exc)
        return False


def embed_all_missing(
    conn: sqlite3.Connection,
    api_url: str,
    model: str,
    has_vec: bool,
) -> int:
    """Generate embeddings for all memory entries that currently lack them.

    Skips forgotten entries.  Stores results in the ``embeddings`` table and,
    when ``has_vec`` is True, also in the sqlite-vec ``memory_vec`` table.

    Args:
        conn: Active SQLite connection.
        api_url: Embedding endpoint base URL.
        model: Embedding model identifier.
        has_vec: Whether the sqlite-vec extension is loaded.

    Returns:
        Number of embeddings successfully generated and stored.
    """
    from datetime import datetime, timezone

    rows = conn.execute(
        """SELECT m.id, m.content FROM memories m
           LEFT JOIN embeddings e ON m.id = e.memory_id
           WHERE e.memory_id IS NULL AND m.forgotten = 0""",
    ).fetchall()

    count = 0
    for row in rows:
        vec = embed_via_local_inference(row["content"], api_url, model)
        if vec is None:
            continue
        blob = pack_embedding(vec)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(memory_id, embedding_blob, model, dimensions, created_at) VALUES (?, ?, ?, ?, ?)",
                (row["id"], blob, model, len(vec), datetime.now(timezone.utc).isoformat()),
            )
            if has_vec:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                    (row["id"], blob),
                )
            conn.commit()
            count += 1
        except sqlite3.Error as exc:
            logger.warning("Embedding store failed for %s: %s", row["id"], exc)

    logger.info("Generated %d embeddings for %d memories", count, len(rows))
    return count
