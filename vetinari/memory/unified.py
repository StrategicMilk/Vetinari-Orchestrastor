"""UnifiedMemoryStore — the single long-term memory backend for all agents.

Owns the SQLite database, FTS5 index, and vector embeddings.  Every
memory operation (store, search, forget, consolidate) flows through this
module.  It is the "warehouse" in the factory metaphor: raw materials
(episodic observations) arrive, get consolidated into knowledge (semantic
patterns and skills), and are served back to agents on demand.

**Memory ontology — who owns what:**

- ``UnifiedMemoryStore`` (this module): Long-term storage, search, fact
  graph, episode recording, Ebbinghaus decay eviction, and episodic →
  semantic promotion.  Single SQLite connection behind an RLock.
- ``SharedMemory`` (``shared.py``): Facade that unifies access to
  UnifiedMemoryStore, PlanTracking, and Blackboard behind one API so
  agent code never imports storage internals directly.
- ``EpisodeMemory`` (``_store_episode.py``): Insert/recall/evict helpers
  for the ``memory_episodes`` table — episodic records of task executions.
- ``Blackboard`` (``blackboard.py``): Pub/sub message board for inter-agent
  coordination.  Short-lived messages, not persistent knowledge.
- ``SessionContext`` (``session_context.py``): LRU cache of recent entries
  for the current session, promoted to long-term on consolidation.

Sub-modules: ``_store_ops`` (CRUD), ``_store_search`` (FTS5/KNN/cosine),
``_store_episode`` (episode insert/recall), ``memory_embeddings`` (embeddings),
``session_context`` (LRU session), ``episode_recorder``, ``_schema`` (DDL).

Decision: Ebbinghaus decay over simple temporal decay (ADR-0092).
Decision: Structured fact relationships with chain-aware search (ADR-0092).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import DEFAULT_EMBEDDING_API_URL
from vetinari.database import _get_db_path, get_connection
from vetinari.exceptions import StorageError

from ._schema import create_schema, create_vec_tables
from ._store_episode import get_episode_stats, get_failure_patterns, recall_episodes_from_db, row_to_episode_dict
from ._store_episode import record_episode_full as _record_episode_full
from ._store_ops import (
    compact_memories,
    evict_low_importance_memories,
    export_memories,
    filter_entry_secrets,
    forget_memory,
    get_entry_by_id,
    get_fact_chain,
    get_memory_stats,
    get_superseded_ids,
    row_to_entry,
    set_relationship,
    store_memory_entry,
    update_memory_content,
)
from ._store_search import (
    build_timeline,
    fts_search,
    is_semantic_duplicate,
    like_search,
    manual_cosine_search,
    vec_knn_search,
)
from .episode_recorder import RecordedEpisode
from .intent_parser import IntentParser, QueryIntent
from .interfaces import MemoryEntry, MemoryStats, MemoryType
from .memory_embeddings import embed_via_local_inference as _embed_via_local_inference
from .memory_embeddings import load_sqlite_vec as _load_sqlite_vec
from .memory_embeddings import pack_embedding as _pack_embedding
from .memory_embeddings import sqlite_vec_available as _sqlite_vec_available
from .memory_embeddings import unpack_embedding as _unpack_embedding  # noqa: F401 — re-exported for callers
from .session_context import SessionContext

logger = logging.getLogger(__name__)

EMBEDDING_API_URL = DEFAULT_EMBEDDING_API_URL
EMBEDDING_MODEL = os.environ.get("VETINARI_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
MAX_LONG_TERM_ENTRIES = int(os.environ.get("VETINARI_MAX_MEMORY_ENTRIES", "10000"))
SEMANTIC_DEDUP_THRESHOLD = float(os.environ.get("VETINARI_SEMANTIC_DEDUP_THRESHOLD", "0.85"))
CONSOLIDATION_QUALITY_THRESHOLD = float(os.environ.get("VETINARI_CONSOLIDATION_QUALITY_THRESHOLD", "0.7"))
EPISODE_PROMOTION_THRESHOLD = int(
    os.environ.get("VETINARI_EPISODE_PROMOTION_THRESHOLD", "10")
)  # min similar episodes to trigger semantic promotion
SESSION_MAX_ENTRIES = int(os.environ.get("VETINARI_SESSION_MAX_ENTRIES", "100"))
EMBEDDING_DIMENSIONS = int(os.environ.get("VETINARI_EMBEDDING_DIMENSIONS", "768"))  # nomic-embed default


class UnifiedMemoryStore:
    """Single SQLite + FTS5 memory backend.

    Owns the DB connection, RLock, and config; all SQL/business logic is
    in the split sub-modules.  The public API is unchanged.
    """

    _ask_deprecation_warned: bool = False

    def __init__(
        self,
        db_path: str | None = None,
        embedding_api_url: str = EMBEDDING_API_URL,
        embedding_model: str = EMBEDDING_MODEL,
        max_entries: int = MAX_LONG_TERM_ENTRIES,
        dedup_threshold: float = SEMANTIC_DEDUP_THRESHOLD,
        session_max: int = SESSION_MAX_ENTRIES,
    ):
        """Open the store, initialise the schema, and probe for sqlite-vec."""
        self._embedding_api_url = embedding_api_url
        self._embedding_model = embedding_model
        self._max_entries = max_entries
        self._dedup_threshold = dedup_threshold
        self._lock = threading.RLock()
        self.session = SessionContext(max_entries=session_max)
        self._has_vec = False

        if db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._private_conn: sqlite3.Connection | None = sqlite3.connect(db_path, check_same_thread=False)
            self._private_conn.row_factory = sqlite3.Row
            self._private_conn.execute("PRAGMA journal_mode=WAL")
            self._owns_connection = True
            create_schema(self._private_conn)
        else:
            self._private_conn = None
            self._owns_connection = False

        self._try_load_sqlite_vec()
        logger.info("UnifiedMemoryStore initialized (sqlite_vec=%s)", self._has_vec)

    @property
    def _conn(self) -> sqlite3.Connection | None:
        """Active connection: private when store owns one, shared thread-local otherwise."""
        if self._owns_connection:
            return self._private_conn
        return get_connection()

    def _try_load_sqlite_vec(self) -> None:
        """Load sqlite-vec and create KNN virtual tables; sets ``_has_vec``."""
        if not (_sqlite_vec_available() and _load_sqlite_vec(self._conn)):
            return
        self._has_vec = create_vec_tables(self._conn, EMBEDDING_DIMENSIONS)

    def embeddings_available(self) -> bool:
        """Return True if the embedding endpoint responds to a probe request."""
        return _embed_via_local_inference("ping", self._embedding_api_url, self._embedding_model) is not None

    def remember(self, entry: MemoryEntry) -> str:
        """Scan for secrets, dedup by content hash, then persist.

        Returns:
            The unique ID assigned to the stored entry.
        """
        entry = filter_entry_secrets(entry)
        with self._lock:
            return store_memory_entry(self._conn, entry, self._store_embedding_for)

    def search(
        self,
        query: str,
        agent: str | None = None,
        entry_types: list[str] | None = None,
        limit: int = 10,
        use_semantic: bool = False,
        include_superseded: bool = False,
    ) -> list[MemoryEntry]:
        """FTS5 or embedding similarity search over long-term memories.

        By default, entries that have been superseded by a newer entry in
        the fact-graph chain are excluded from results.

        Args:
            query: Free-text or semantic search query.
            agent: Optional agent-name filter.
            entry_types: Optional entry-type filter list.
            limit: Maximum results.
            use_semantic: Use embedding similarity instead of FTS5.
            include_superseded: When True, include entries superseded by
                newer entries in the fact-graph chain.

        Returns:
            Up to *limit* MemoryEntries ranked by relevance to *query*.
        """
        fetch_limit = limit if include_superseded else limit * 2
        if use_semantic:
            results = self._semantic_search(query, agent, entry_types, fetch_limit)
        else:
            with self._lock:
                results = fts_search(self._conn, query, agent, entry_types, fetch_limit, like_fallback=like_search)
        if not include_superseded:
            with self._lock:
                superseded = get_superseded_ids(self._conn)
            results = [e for e in results if e.id not in superseded]
        final = results[:limit]
        self._touch_accessed([e.id for e in final if e.id])
        return final

    def timeline(
        self, agent: str | None = None, start_time: int | None = None, end_time: int | None = None, limit: int = 100
    ) -> list[MemoryEntry]:
        """Return memories in reverse chronological order with optional filters.

        Args:
            agent: Restrict results to entries recorded by this agent name. Pass
                ``None`` to include all agents.
            start_time: Inclusive lower bound as a Unix millisecond timestamp.
                Pass ``None`` for no lower bound.
            end_time: Inclusive upper bound as a Unix millisecond timestamp.
                Pass ``None`` for no upper bound.
            limit: Maximum number of entries to return (default 100).

        Returns:
            Up to *limit* MemoryEntries ordered newest-first within the requested time window.
        """
        with self._lock:
            return build_timeline(self._conn, agent, start_time, end_time, limit)

    def query(self, question: str, agent: str | None = None) -> list[MemoryEntry]:
        """Dispatch a natural language question to the best retrieval backend.

        Uses :class:`IntentParser` to classify the question, then routes to
        ``recall_episodes`` (episode queries), ``timeline`` (time-range queries),
        or ``search`` (knowledge-base and general semantic queries).

        Args:
            question: Natural language question from an agent or user.
            agent: Optional agent-name filter passed through to timeline/search.

        Returns:
            Matching MemoryEntries (or Episodes converted to MemoryEntries).
        """
        parsed = IntentParser().parse(question)

        if parsed.intent == QueryIntent.EPISODE_RECALL:
            episodes = self.recall_episodes(
                query=question,
                task_type=parsed.task_type,
                successful_only=bool(parsed.success_filter),
            )
            return [self._episode_to_entry(ep) for ep in episodes]

        if parsed.intent == QueryIntent.TIMELINE:
            start, end = parsed.time_range
            return self.timeline(agent=agent, start_time=start, end_time=end)

        if parsed.intent == QueryIntent.KNOWLEDGE_BASE:
            return self.search(
                question,
                entry_types=["discovery", "rule", "pattern"],
                use_semantic=True,
            )

        # SEMANTIC_SEARCH — general fallback
        return self.search(question, use_semantic=True)

    def _episode_to_entry(self, ep: RecordedEpisode) -> MemoryEntry:
        """Convert an Episode record to a MemoryEntry for uniform return types."""
        return MemoryEntry(
            id=ep.episode_id,
            agent=ep.agent_type,
            entry_type=MemoryType.SUCCESS if ep.success else MemoryType.PROBLEM,
            content=ep.task_summary,
            summary=ep.output_summary,
            metadata={"task_type": ep.task_type},
        )

    def ask(self, question: str, agent: str | None = None) -> list[MemoryEntry]:
        """Deprecated — use :meth:`query` instead.

        Delegates to ``query()`` and emits a one-time deprecation warning.

        Args:
            question: Natural language question.
            agent: Optional agent-name filter.

        Returns:
            Results from :meth:`query`.
        """
        if not UnifiedMemoryStore._ask_deprecation_warned:
            logger.warning(
                "ask() is deprecated — use query() for intent-aware dispatch. Falling back to query() automatically."
            )
            UnifiedMemoryStore._ask_deprecation_warned = True
        return self.query(question, agent=agent)

    def export(self, path: str) -> bool:
        """Dump all non-forgotten memories to *path* as JSON.

        Returns:
            True when the file was written successfully, False on any I/O error.
        """
        with self._lock:
            return export_memories(self._conn, path)

    def forget(self, entry_id: str, reason: str) -> bool:
        """Soft-delete an entry by marking it as a tombstone so it is excluded from all future queries.

        Args:
            entry_id: The unique ID of the memory entry to tombstone.
            reason: Human-readable explanation for why the entry is being forgotten
                (stored for audit purposes).

        Returns:
            True when the entry existed and was tombstoned, False when the ID was not found.
        """
        with self._lock:
            return forget_memory(self._conn, entry_id, reason)

    def update_content(self, entry_id: str, new_content: str) -> bool:
        """Replace the stored text of an existing memory entry in place.

        Args:
            entry_id: The unique ID of the memory entry to modify.
            new_content: The replacement content string; must be non-empty.

        Returns:
            True when the entry existed and was updated, False when the ID was not found.
        """
        with self._lock:
            return update_memory_content(self._conn, entry_id, new_content)

    # -- Fact graph --------------------------------------------------------

    def fact_graph(self, entry_id: str) -> list[MemoryEntry]:
        """Walk the supersession chain from *entry_id* back to its origin.

        Follows ``supersedes_id`` links through non-forgotten entries,
        returning the full lineage from newest to oldest.  Useful for
        understanding how a fact evolved over time.

        Args:
            entry_id: Starting entry ID (typically the newest revision).

        Returns:
            Ordered list of MemoryEntry from newest to oldest in the chain.
        """
        with self._lock:
            return get_fact_chain(self._conn, entry_id)

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
    ) -> bool:
        """Link two memory entries with a typed relationship.

        Sets ``supersedes_id`` and ``relationship_type`` on the source entry
        so that chain-aware search can filter or walk the resulting graph.

        Args:
            source_id: The newer entry that references *target_id*.
            target_id: The older entry being referenced.
            relationship_type: One of :class:`~vetinari.types.RelationshipType` values
                (``supersedes``, ``contradicts``, ``caused_by``, ``elaborates``).

        Returns:
            True when the source entry was found and updated, False otherwise.
        """
        with self._lock:
            return set_relationship(self._conn, source_id, target_id, relationship_type)

    def compact(self, max_age_days: int | None = None) -> int:
        """Delete forgotten entries and optionally prune by age.

        Returns:
            Number of entries permanently removed from the store.
        """
        with self._lock:
            return compact_memories(self._conn, max_age_days)

    def stats(self) -> MemoryStats:
        """Return aggregate counts, timestamps, and file size for the store.

        Returns:
            MemoryStats snapshot covering total entries, type breakdown, and disk usage.
        """
        with self._lock:
            return MemoryStats(**get_memory_stats(self._conn, _get_db_path))

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Fetch entry by ID (increments access count).

        Returns:
            The matching MemoryEntry, or None when the ID does not exist or is tombstoned.
        """
        with self._lock:
            row = get_entry_by_id(self._conn, entry_id)
        return row_to_entry(row) if row is not None else None

    def _touch_accessed(self, entry_ids: list[str]) -> None:
        """Update last_accessed timestamp for retrieved memory entries."""
        if not entry_ids:
            return
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        with self._lock:
            try:
                for eid in entry_ids:
                    self._conn.execute(
                        "UPDATE memories SET last_accessed = ? WHERE id = ?",
                        (now_ms, eid),
                    )
                self._conn.commit()
            except Exception:
                logger.warning("Could not update last_accessed for retrieved entries — timestamps may be stale")

    def _store_embedding_for(self, memory_id: str, text: str) -> None:
        # Calls _embed_via_local_inference by module-level name so test patches on
        # vetinari.memory.unified._embed_via_local_inference take effect.
        vec = _embed_via_local_inference(text, self._embedding_api_url, self._embedding_model)
        if vec is None:
            return
        blob = _pack_embedding(vec)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(memory_id, embedding_blob, model, dimensions, created_at) VALUES (?, ?, ?, ?, ?)",
                (memory_id, blob, self._embedding_model, len(vec), datetime.now(timezone.utc).isoformat()),
            )
            if self._has_vec:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, blob),
                )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Failed to store embedding for %s — semantic search will degrade", memory_id)
            logger.debug("Embedding store error detail: %s", exc)

    def embed_all(self) -> int:
        """Generate embeddings for every memory that currently lacks one.

        Returns:
            Count of embeddings newly generated during this call.
        """
        from .memory_embeddings import embed_all_missing

        with self._lock:
            return embed_all_missing(
                self._conn,
                api_url=self._embedding_api_url,
                model=self._embedding_model,
                has_vec=self._has_vec,
            )

    def _semantic_search(
        self, query: str, agent: str | None, entry_types: list[str] | None, limit: int
    ) -> list[MemoryEntry]:
        """Embedding similarity search, falling back to FTS5 when unavailable."""
        query_vec = _embed_via_local_inference(query, self._embedding_api_url, self._embedding_model)
        if query_vec is None:
            with self._lock:
                return fts_search(self._conn, query, agent, entry_types, limit, like_fallback=like_search)

        def _fts(conn: sqlite3.Connection, q: str, ag: Any, et: Any, lim: int) -> list[MemoryEntry]:
            return fts_search(conn, q, ag, et, lim, like_fallback=like_search)

        def _cosine(conn: sqlite3.Connection, qv: list[float], ag: Any, et: Any, lim: int) -> list[MemoryEntry]:
            return manual_cosine_search(
                conn,
                qv,
                ag,
                et,
                lim,
                fallback_query=query,
                embedding_model=self._embedding_model,
                embedding_dimensions=EMBEDDING_DIMENSIONS,
                fts_fallback=_fts,
            )

        with self._lock:
            if self._has_vec:
                return vec_knn_search(
                    self._conn,
                    query_vec,
                    agent,
                    entry_types,
                    limit,
                    fallback_query=query,
                    embedding_model=self._embedding_model,
                    embedding_dimensions=EMBEDDING_DIMENSIONS,
                    fts_fallback=_fts,
                    manual_fallback=_cosine,
                )
            return _cosine(self._conn, query_vec, agent, entry_types, limit)

    def _check_semantic_duplicate(self, content: str) -> bool:
        """Return True when *content* cosine-similarity exceeds the dedup threshold."""
        query_vec = _embed_via_local_inference(content, self._embedding_api_url, self._embedding_model)
        if query_vec is None:
            return False
        with self._lock:
            return is_semantic_duplicate(self._conn, query_vec, self._dedup_threshold)

    def record_episode(
        self,
        task_description: str,
        agent_type: str,
        task_type: str,
        output_summary: str,
        quality_score: float,
        success: bool,
        model_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist an agent execution episode and store its embedding for recall.

        Args:
            task_description: Plain-text description of the task that was executed.
            agent_type: Identifier for the agent that ran the task (e.g. ``"worker"``).
            task_type: Category of work (e.g. ``"code_generation"``, ``"review"``).
            output_summary: Short summary of what the agent produced.
            quality_score: Numeric quality rating in ``[0.0, 1.0]`` assigned by the inspector.
            success: Whether the episode ended in a successful outcome.
            model_id: Identifier of the model used for inference; empty string if unknown.
            metadata: Optional free-form key/value pairs stored alongside the episode record.

        Returns:
            The unique episode ID assigned to the new record.
        """
        with self._lock:
            return _record_episode_full(
                self._conn,
                task_description=task_description,
                agent_type=agent_type,
                task_type=task_type,
                output_summary=output_summary,
                quality_score=quality_score,
                success=success,
                model_id=model_id,
                metadata=metadata or {},  # noqa: VET112 — Optional per func param
                max_entries=self._max_entries,
                api_url=self._embedding_api_url,
                model=self._embedding_model,
            )

    def recall_episodes(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        task_type: str | None = None,
        successful_only: bool = False,
    ) -> list[RecordedEpisode]:
        """Return the *k* most relevant past episodes for *query*.

        Args:
            query: Natural language description of the task to find similar episodes for.
            k: Maximum number of episodes to return (default 5).
            min_score: Minimum cosine-similarity score; episodes below this threshold
                are excluded (default 0.0, meaning no filtering).
            task_type: Optional task category filter applied before similarity ranking.
            successful_only: When ``True``, restrict results to episodes that succeeded.

        Returns:
            Up to *k* Episodes ranked by similarity to the query, filtered by *min_score*.
        """
        query_vec = _embed_via_local_inference(
            f"{task_type or ''}: {query}",  # noqa: VET112 — Optional per func param
            self._embedding_api_url,
            self._embedding_model,
        )
        with self._lock:
            return recall_episodes_from_db(
                self._conn,
                query_vec=query_vec,
                query_text=query,
                k=k,
                min_score=min_score,
                task_type=task_type,
                successful_only=successful_only,
                row_to_episode_fn=self._row_to_episode,
            )

    def get_failure_patterns(self, agent_type: str, task_type: str) -> list[str]:
        """Return recent failure output summaries for an agent/task combination.

        Args:
            agent_type: Agent identifier to filter by (e.g. ``"worker"``).
            task_type: Task category to filter by (e.g. ``"code_generation"``).

        Returns:
            Output summary strings from failed episodes, most recent first.
        """
        with self._lock:
            return get_failure_patterns(self._conn, agent_type, task_type)

    def get_episode_stats(self) -> dict[str, Any]:
        """Return total, successful, and avg_quality_score for stored episodes.

        Returns:
            Dict with total, successful, failed, and avg_quality_score keys.
        """
        with self._lock:
            return get_episode_stats(self._conn)

    def consolidate(self, quality_threshold: float = CONSOLIDATION_QUALITY_THRESHOLD) -> int:
        """Promote session entries (quality >= threshold) to long-term memory.

        Returns:
            Number of session entries successfully promoted to the long-term store.
        """
        entries = self.session.get_all()
        promoted = 0
        for session_entry in entries:
            if session_entry.quality_score < quality_threshold:
                continue
            value = session_entry.value
            if isinstance(value, dict) and "content" in value:
                mem_entry = MemoryEntry.from_dict(value)
            elif isinstance(value, str):
                mem_entry = MemoryEntry(content=value, provenance="session_consolidation")
            else:
                continue
            if self._check_semantic_duplicate(mem_entry.content):
                logger.debug("Skipping semantic duplicate during consolidation: %s", session_entry.key)
                continue
            try:
                self.remember(mem_entry)
                promoted += 1
            except (RuntimeError, StorageError):
                logger.warning("Consolidation failed for entry %s — skipping", session_entry.key)
        if promoted > 0:
            with self._lock:
                evict_low_importance_memories(self._conn, self._max_entries)
        logger.info("Consolidated %s of %s session entries to long-term", promoted, len(entries))
        return promoted

    # -- Episode-to-semantic promotion ------------------------------------

    def promote_episodes_to_semantic(
        self,
        threshold: int = EPISODE_PROMOTION_THRESHOLD,
    ) -> int:
        """Extract recurring patterns from episodic memory into semantic rules.

        Groups successful episodes by task_type.  When a group reaches
        *threshold* members, a PATTERN memory is created that summarises
        the common approach, and source episodes are marked consolidated
        (``promoted=1`` in metadata) so they are not promoted again.

        Args:
            threshold: Minimum number of similar successful episodes required
                before a pattern is extracted (default from env / 10).

        Returns:
            Number of new semantic pattern entries created.
        """
        import json
        import uuid

        with self._lock:
            rows = self._conn.execute("SELECT * FROM memory_episodes WHERE success = 1").fetchall()

        if not rows:
            return 0

        # Group by task_type, filtering out already-promoted episodes
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            ep = row_to_episode_dict(row)
            meta = ep.get("metadata") or {}
            if meta.get("promoted"):
                continue
            key = ep.get("task_type", "unknown")
            groups.setdefault(key, []).append(ep)

        promoted = 0
        for task_type, episodes in groups.items():
            if len(episodes) < threshold:
                continue

            # Build a summary from the top-quality episodes
            sorted_eps = sorted(episodes, key=lambda e: e["quality_score"], reverse=True)
            top = sorted_eps[:5]
            summaries = [e["task_summary"] for e in top]
            avg_quality = sum(e["quality_score"] for e in episodes) / len(episodes)
            source_ids = [e["episode_id"] for e in episodes]

            pattern_content = (
                f"Recurring pattern for task_type={task_type} "
                f"(observed {len(episodes)} times, avg quality {avg_quality:.2f}):\n"
                + "\n".join(f"- {s}" for s in summaries)
            )

            entry = MemoryEntry(
                id=f"pattern_{uuid.uuid4().hex[:8]}",
                agent="system",
                entry_type=MemoryType.PATTERN,
                content=pattern_content,
                summary=f"Extracted pattern: {task_type} ({len(episodes)} episodes)",
                provenance="episode_promotion",
                metadata={
                    "source_episode_ids": source_ids,
                    "task_type": task_type,
                    "episode_count": len(episodes),
                    "avg_quality": round(avg_quality, 3),
                },
            )

            try:
                self.remember(entry)
                promoted += 1
            except StorageError:
                logger.warning(
                    "Failed to store promoted pattern for task_type=%s — skipping",
                    task_type,
                )
                continue

            # Mark source episodes as promoted so they don't trigger again
            with self._lock:
                for ep_id in source_ids:
                    try:
                        row = self._conn.execute(
                            "SELECT metadata_json FROM memory_episodes WHERE episode_id = ?",
                            (ep_id,),
                        ).fetchone()
                        meta = json.loads(row["metadata_json"]) if row and row["metadata_json"] else {}
                        meta["promoted"] = True
                        self._conn.execute(
                            "UPDATE memory_episodes SET metadata_json = ? WHERE episode_id = ?",
                            (json.dumps(meta), ep_id),
                        )
                    except (sqlite3.Error, json.JSONDecodeError):
                        logger.warning(
                            "Could not update metadata for episode %s during promotion — episode promoted but metadata flag not set",
                            ep_id,
                        )
                self._conn.commit()

        if promoted > 0:
            logger.info("Promoted %d episode groups to semantic patterns", promoted)
        return promoted

    def _filter_secrets(self, entry: MemoryEntry) -> MemoryEntry:
        """Delegate to ``filter_entry_secrets`` — kept for backward compatibility."""
        return filter_entry_secrets(entry)

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return row_to_entry(row)

    def _row_to_episode(self, row: sqlite3.Row) -> RecordedEpisode:
        return RecordedEpisode(**row_to_episode_dict(row))

    def close(self) -> None:
        """Close the private connection (no-op when using the shared connection)."""
        if getattr(self, "_private_conn", None) is not None and getattr(self, "_owns_connection", False):
            with contextlib.suppress(sqlite3.Error):
                self._private_conn.close()
            self._private_conn = None

    def __enter__(self) -> UnifiedMemoryStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        self.close()


_unified_store: UnifiedMemoryStore | None = None
_store_lock = threading.Lock()


def get_unified_memory_store() -> UnifiedMemoryStore:
    """Get or create the global UnifiedMemoryStore singleton (double-checked locking).

    Returns:
        The process-wide UnifiedMemoryStore instance, initialised on first call.
    """
    global _unified_store
    if _unified_store is None:
        with _store_lock:
            if _unified_store is None:
                _unified_store = UnifiedMemoryStore()
    return _unified_store


# Alias used by vetinari.training.idle_scheduler
get_unified_store = get_unified_memory_store


def init_unified_memory_store(**kwargs: Any) -> UnifiedMemoryStore:
    """Replace the global singleton with a freshly constructed store.

    Returns:
        The newly created UnifiedMemoryStore that is now the active singleton.
    """
    global _unified_store
    with _store_lock:
        if _unified_store is not None:
            _unified_store.close()
        _unified_store = UnifiedMemoryStore(**kwargs)
    return _unified_store
