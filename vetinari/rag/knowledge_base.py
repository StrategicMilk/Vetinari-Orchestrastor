r"""Vetinari RAG Knowledge Base.

Vector-backed knowledge base for Retrieval-Augmented Generation using
SQLite + FTS5 + sqlite-vec for vector search.

Agents query this to get relevant context before execution:
- Project documentation
- Past successful outputs
- Code patterns and templates
- Error resolution guides
- LLM best practices

Architecture
------------
- SQLite + FTS5 for full-text search (always available)
- sqlite-vec for fast KNN vector search (optional, graceful fallback)
- Embeddings via OpenAI-compatible ``/v1/embeddings`` endpoint (shared with UnifiedMemoryStore)
- Automatic document ingestion from project docs/ directory
- Context-window-aware retrieval (returns only what fits)

Decision: sqlite-vec replaces ChromaDB for unified SQLite storage (ADR-0063)

Usage::

    from vetinari.rag.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()
    kb.ingest_directory("docs/")

    results = kb.query(
        "How do I implement exponential backoff?",
        k=5,
        max_chars=3000,
    )
    context = "\n---\n".join(r.content for r in results)
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from vetinari.constants import DATABASE_BUSY_TIMEOUT_MS
from vetinari.database import get_connection
from vetinari.rag.knowledge_base_helpers import (  # noqa: F401 - import intentionally probes or re-exports API surface
    _EMBEDDING_DIMENSIONS,
    _MAX_DOC_CHARS,
    KBDocument,
    load_sqlite_vec,
    pack_embedding,
    unpack_embedding,
)
from vetinari.rag.knowledge_base_helpers import (
    kb_embed as _embed,  # aliased as _embed so tests can patch vetinari.rag.knowledge_base._embed
)
from vetinari.rag.knowledge_base_search import KnowledgeBaseSearchMixin

logger = logging.getLogger(__name__)

_MAX_DIRECTORY_FILES = 1000
_MAX_DIRECTORY_BYTES = 10_000_000
_MAX_DIRECTORY_CHUNKS = 5000
_MAX_URL_RESPONSE_BYTES = 2_000_000
_MAX_URL_CHUNKS = 500
_REQUIRE_HTTPS_URL_INGEST = os.environ.get("VETINARI_RAG_REQUIRE_HTTPS", "true").lower() in {"1", "true", "yes"}


# ── Knowledge Base ────────────────────────────────────────────────────────


class KnowledgeBase(KnowledgeBaseSearchMixin):
    """SQLite-backed knowledge base with FTS5 and optional sqlite-vec KNN search.

    In production (``db_path=None``) uses the unified database via
    ``vetinari.database.get_connection()``. When ``db_path`` is provided
    (test isolation), opens a dedicated persistent connection to that file.
    """

    _instance: KnowledgeBase | None = None
    _cls_lock = threading.Lock()

    def __init__(self, db_path: str | None = None):
        self._db_path: str | None = db_path
        self._has_vec = False
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._embedding_attempts: int = 0
        self._embedding_fallbacks: int = 0
        self._init_db()

    @classmethod
    def get_instance(cls) -> KnowledgeBase:
        """Get or create the singleton instance.

        Returns:
            The shared KnowledgeBase instance.
        """
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ── Database Setup ────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create database schema with FTS5 and optional sqlite-vec support."""
        if self._db_path is not None:
            db_dir = Path(self._db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={DATABASE_BUSY_TIMEOUT_MS}")
        else:
            self._conn = get_connection()

        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doc_embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding_blob BLOB NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
                doc_id,
                content,
                source,
                category,
                content=documents,
                content_rowid=rowid
            )
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS doc_fts_ai AFTER INSERT ON documents BEGIN
                INSERT INTO doc_fts(rowid, doc_id, content, source, category)
                VALUES (NEW.rowid, NEW.doc_id, NEW.content, NEW.source, NEW.category);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS doc_fts_ad AFTER DELETE ON documents BEGIN
                INSERT INTO doc_fts(doc_fts, rowid, doc_id, content, source, category)
                VALUES ('delete', OLD.rowid, OLD.doc_id, OLD.content, OLD.source, OLD.category);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS doc_fts_au AFTER UPDATE ON documents BEGIN
                INSERT INTO doc_fts(doc_fts, rowid, doc_id, content, source, category)
                VALUES ('delete', OLD.rowid, OLD.doc_id, OLD.content, OLD.source, OLD.category);
                INSERT INTO doc_fts(rowid, doc_id, content, source, category)
                VALUES (NEW.rowid, NEW.doc_id, NEW.content, NEW.source, NEW.category);
            END
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_category ON documents(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_source ON documents(source)")

        if load_sqlite_vec(self._conn):
            self._has_vec = True
            try:
                cursor.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS doc_vec USING vec0(
                        doc_id TEXT PRIMARY KEY,
                        embedding float[{_EMBEDDING_DIMENSIONS}]
                    )
                """)
            except sqlite3.OperationalError as exc:
                logger.warning("KB sqlite-vec table creation failed: %s", exc)
                self._has_vec = False

        self._conn.commit()
        logger.info("KnowledgeBase initialized (db=%s, sqlite_vec=%s)", self._db_path, self._has_vec)

    def close(self) -> None:
        """Close the database connection and release resources.

        Only closes the connection when using a dedicated file (test-isolation
        mode). In production the unified connection lifecycle is managed by
        ``vetinari.database``.
        """
        if self._conn is not None and self._db_path is not None:
            self._conn.close()
            self._conn = None

    def __del__(self) -> None:
        """Safety-net cleanup if close() was not called explicitly."""
        with contextlib.suppress(Exception):
            self.close()

    # ── Ingestion ─────────────────────────────────────────────────────

    def add_document(
        self,
        content: str,
        source: str,
        category: str = "general",
        doc_id: str | None = None,
    ) -> str:
        """Add a document chunk to the knowledge base.

        Args:
            content: Document text content.
            source: Source file path or URL.
            category: Document category (docs, code, pattern, error, etc.).
            doc_id: Optional deterministic document ID.

        Returns:
            The document ID assigned to this chunk.
        """
        if not doc_id:
            doc_id = f"doc_{hashlib.md5(f'{source}:{content[:50]}'.encode(), usedforsecurity=False).hexdigest()[:8]}"

        truncated = content[:_MAX_DOC_CHARS]
        vec = _embed(truncated)

        with self._lock:
            cursor = self._conn.cursor()

            cursor.execute(
                "INSERT OR REPLACE INTO documents (doc_id, content, source, category) VALUES (?, ?, ?, ?)",
                (doc_id, truncated, source, category),
            )

            if vec is not None:
                blob = pack_embedding(vec)
                cursor.execute(
                    "INSERT OR REPLACE INTO doc_embeddings (doc_id, embedding_blob) VALUES (?, ?)",
                    (doc_id, blob),
                )
                if self._has_vec:
                    try:
                        cursor.execute(
                            "INSERT OR REPLACE INTO doc_vec (doc_id, embedding) VALUES (?, ?)",
                            (doc_id, blob),
                        )
                    except sqlite3.Error as exc:
                        logger.warning("KB vec0 upsert failed for %s: %s", doc_id, exc)
            else:
                # Embedding unavailable — delete any stale embedding rows for
                # this doc so that vector search cannot return outdated results
                # for updated content.
                cursor.execute("DELETE FROM doc_embeddings WHERE doc_id = ?", (doc_id,))
                if self._has_vec:
                    try:
                        cursor.execute("DELETE FROM doc_vec WHERE doc_id = ?", (doc_id,))
                    except sqlite3.Error as exc:
                        logger.warning("KB vec0 stale-row delete failed for %s: %s", doc_id, exc)

            self._conn.commit()

        return doc_id

    def ingest_directory(
        self,
        directory: str,
        extensions: list[str] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> int:
        """Ingest all documents from a directory.

        Args:
            directory: Path to directory to ingest.
            extensions: File extensions to include (default: .md, .txt, .py, .yaml, .json).
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.

        Returns:
            Number of chunks added.
        """
        extensions = extensions or [".md", ".txt", ".py", ".yaml", ".json"]
        try:
            base = Path(directory).resolve(strict=True)
        except OSError:
            logger.warning("[KnowledgeBase] Directory %s not found", directory)
            return 0
        if not base.is_dir():
            logger.debug("[KnowledgeBase] Directory %s is not a directory", directory)
            return 0

        count = 0
        files_seen = 0
        bytes_seen = 0
        for path in base.rglob("*"):
            if count >= _MAX_DIRECTORY_CHUNKS or files_seen >= _MAX_DIRECTORY_FILES:
                logger.warning("[KnowledgeBase] Directory ingest cap reached for %s", base)
                break
            if path.is_dir() or path.suffix.lower() not in extensions:
                continue
            try:
                relative_path = path.relative_to(base)
            except ValueError as exc:
                logger.warning("[KnowledgeBase] Skipped path outside ingest root %s: %s", path, exc)
                continue
            if path.is_symlink():
                logger.warning("[KnowledgeBase] Skipped symlink during directory ingest: %s", relative_path)
                continue
            if any(
                part.startswith(".") or part in ("__pycache__", "venv", "node_modules") for part in relative_path.parts
            ):
                continue
            try:
                resolved_path = path.resolve(strict=True)
                if not resolved_path.is_relative_to(base):
                    logger.warning("[KnowledgeBase] Skipped path outside ingest root: %s", relative_path)
                    continue
                files_seen += 1
                remaining_bytes = _MAX_DIRECTORY_BYTES - bytes_seen
                if remaining_bytes <= 0:
                    logger.warning("[KnowledgeBase] Directory ingest byte cap reached for %s", base)
                    break
                with resolved_path.open("rb") as handle:
                    raw = handle.read(remaining_bytes + 1)
                if len(raw) > remaining_bytes:
                    logger.warning("[KnowledgeBase] Skipped %s because ingest byte budget is exhausted", relative_path)
                    break
                bytes_seen += len(raw)
                content = raw.decode("utf-8", errors="ignore")
                chunks = self._chunk(content, chunk_size, chunk_overlap)[: _MAX_DIRECTORY_CHUNKS - count]
                source = relative_path.as_posix()
                # Delete all existing chunks for this source path before
                # writing new ones.  If the file shrank, the old higher-index
                # chunks would otherwise remain as stale documents.
                with self._lock:
                    cursor = self._conn.cursor()
                    cursor.execute(
                        "DELETE FROM documents WHERE source = ?",
                        (source,),
                    )
                    cursor.execute(
                        "DELETE FROM doc_embeddings WHERE doc_id NOT IN (SELECT doc_id FROM documents)",
                    )
                    if self._has_vec:
                        try:
                            cursor.execute(
                                "DELETE FROM doc_vec WHERE doc_id NOT IN (SELECT doc_id FROM documents)",
                            )
                        except sqlite3.Error as exc:
                            logger.warning("KB vec0 stale-chunk cleanup failed for %s: %s", path, exc)
                    self._conn.commit()
                for i, chunk in enumerate(chunks):
                    category = "code" if path.suffix == ".py" else "docs"
                    self.add_document(
                        content=chunk,
                        source=source,
                        category=category,
                        doc_id=(f"doc_{hashlib.md5(f'{source}_{i}'.encode(), usedforsecurity=False).hexdigest()[:8]}"),
                    )
                    count += 1
            except Exception as e:
                logger.warning("[KnowledgeBase] Skipped %s: %s", path, e)

        logger.info("[KnowledgeBase] Ingested %s chunks from %s", count, directory)
        return count

    def ingest_url(
        self,
        url: str,
        category: str = "docs",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> int:
        """Fetch a URL and ingest its text content into the knowledge base.

        Validates the URL against SSRF attack vectors before making any
        network request.  Only ``http`` and ``https`` schemes are allowed;
        private/loopback/cloud-metadata addresses are rejected with a
        ``ValueError``.

        Args:
            url: The URL to fetch.  Must use http or https and must not
                point to a private or internal address.
            category: Document category to assign to the ingested chunks.
            chunk_size: Maximum size of each text chunk in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.

        Returns:
            Number of chunks ingested from the URL, or 0 if the fetch fails.

        Raises:
            ValueError: If ``url`` fails SSRF validation (private IP,
                metadata endpoint, disallowed scheme, etc.).
        """
        from vetinari.security import validate_url_no_ssrf

        # This raises ValueError / SecurityError for blocked URLs — intentionally
        # not caught here so the caller knows the URL was rejected.
        fetch_url = validate_url_no_ssrf(url)
        _validate_ingest_fetch_url(fetch_url)

        try:
            status_code, headers, raw_content = _fetch_url_bytes(fetch_url)
            if 300 <= status_code < 400:
                location = headers.get("location", "")
                redirected = validate_url_no_ssrf(urljoin(fetch_url, location))
                _validate_ingest_fetch_url(redirected)
                fetch_url = redirected
                status_code, _headers, raw_content = _fetch_url_bytes(fetch_url)
                if 300 <= status_code < 400:
                    logger.warning("KnowledgeBase URL redirect chain rejected for %s", _redact_url(fetch_url))
                    return 0
            content = raw_content.decode("utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("KnowledgeBase URL fetch failed for %s — skipping ingest: %s", _redact_url(url), exc)
            return 0

        chunks = self._chunk(content, chunk_size, chunk_overlap)[:_MAX_URL_CHUNKS]
        count = 0
        for i, chunk in enumerate(chunks):
            self.add_document(
                content=chunk,
                source=fetch_url,
                category=category,
                doc_id=f"doc_{hashlib.md5(f'{fetch_url}_{i}'.encode(), usedforsecurity=False).hexdigest()[:8]}",
            )
            count += 1

        logger.info("KnowledgeBase ingested %d chunks from URL %s", count, _redact_url(fetch_url))
        return count

    # ── Querying ──────────────────────────────────────────────────────

    def query(
        self,
        query: str,
        k: int = 5,
        max_chars: int = 3000,
        category: str | None = None,
    ) -> list[KBDocument]:
        """Retrieve the k most relevant documents for a query.

        Uses sqlite-vec KNN search when available, falls back to manual
        cosine similarity, then to FTS5 keyword search.

        Args:
            query: The search query string.
            k: Number of results to return.
            max_chars: Maximum total characters in returned results.
            category: Optional category filter.

        Returns:
            List of KBDocument sorted by relevance, truncated to max_chars.
        """
        with self._lock:
            self._embedding_attempts += 1
            query_vec = _embed(query)
            if query_vec is not None:
                if self._has_vec:
                    results = self._query_vec_knn(query_vec, k, category, fallback_query=query)
                else:
                    results = self._query_cosine(query_vec, k, category, fallback_query=query)
            else:
                self._embedding_fallbacks += 1
                fallback_rate = self._embedding_fallbacks / self._embedding_attempts
                if self._embedding_fallbacks % 10 == 1:
                    logger.warning(
                        "KB embedding fallback: %d/%d queries (%.0f%%) using FTS5 instead of vectors",
                        self._embedding_fallbacks,
                        self._embedding_attempts,
                        fallback_rate * 100,
                    )
                results = self._query_fts(query, k, category)

        filtered: list[KBDocument] = []
        total_chars = 0
        for doc in results:
            if total_chars + len(doc.content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 0:
                    # Trim this document to fit within the budget rather than
                    # dropping it entirely.  The old threshold of 100 caused
                    # the first result to be discarded whenever the budget was
                    # small, returning nothing at all.
                    doc.content = doc.content[:remaining]
                    filtered.append(doc)
                break
            filtered.append(doc)
            total_chars += len(doc.content)

        return self._compress_results(filtered, max_chars)

    # ── Statistics ────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics.

        Returns:
            Dict with document_count, backend type, database path, and
            embedding attempt/fallback counts.
        """
        count = 0
        with self._lock:
            try:
                cursor = self._conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                count = cursor.fetchone()[0]
            except Exception:
                logger.warning("Failed to get KB document count", exc_info=True)

        fallback_rate = (
            round(self._embedding_fallbacks / self._embedding_attempts, 3) if self._embedding_attempts > 0 else 0.0
        )
        return {
            "document_count": count,
            "backend": "sqlite_vec" if self._has_vec else "sqlite_fts5",
            "db_path": self._db_path if self._db_path is not None else "unified",
            "embedding_attempts": self._embedding_attempts,
            "embedding_fallbacks": self._embedding_fallbacks,
            "embedding_fallback_rate": fallback_rate,
        }


# ── Module-level Accessors ────────────────────────────────────────────────


def _validate_ingest_fetch_url(url: str) -> None:
    parsed = urlparse(url)
    if _REQUIRE_HTTPS_URL_INGEST and parsed.scheme != "https":
        raise ValueError("KnowledgeBase URL ingestion requires https")


def _redact_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or "<missing-host>"
    return f"{parsed.scheme or '<missing-scheme>'}://{host}"


def _fetch_url_bytes(url: str) -> tuple[int, dict[str, str], bytes]:
    import httpx

    chunks: list[bytes] = []
    total = 0
    with httpx.stream("GET", url, timeout=10, follow_redirects=False) as resp:
        status_code = int(resp.status_code)
        headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        if 300 <= status_code < 400:
            return status_code, headers, b""
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_URL_RESPONSE_BYTES:
                raise ValueError("URL response exceeded KnowledgeBase byte cap")
            chunks.append(chunk)
    return status_code, headers, b"".join(chunks)


_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def get_knowledge_base() -> KnowledgeBase:
    """Return the global KnowledgeBase singleton.

    Returns:
        The shared KnowledgeBase instance.
    """
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                _kb = KnowledgeBase.get_instance()
    return _kb


def ingest_project_docs() -> int:
    """Ingest all Vetinari project documentation into the knowledge base.

    Returns:
        Number of chunks ingested.
    """
    kb = get_knowledge_base()
    project_root = Path(__file__).parent.parent.parent
    total = 0
    for d in ["docs", "skills", "system_prompts", "prompts"]:
        p = project_root / d
        if p.exists():
            total += kb.ingest_directory(str(p))
    logger.info("[KnowledgeBase] Project docs ingestion complete: %s chunks", total)
    return total
