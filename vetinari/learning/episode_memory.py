"""Vetinari Episodic Memory System.

==================================
Stores structured records of past agent executions and retrieves relevant
past experiences via lightweight embedding similarity.

Agents use this to:
- Avoid repeating past mistakes
- Reuse successful patterns
- Inject relevant examples into prompts

Architecture
------------
- SQLite backend for persistence
- Lightweight embedding via character n-gram hashing (no GPU/VRAM used)
- Optional sentence-transformers upgrade when available
- Bounded storage with importance-based eviction (max 10,000 episodes)
- Thread-safe via RLock

Usage::

    from vetinari.learning.episode_memory import get_episode_memory

    mem = get_episode_memory()

    # Record a successful execution
    mem.record(
        task_description="Write a Redis cache wrapper",
        agent_type="BUILDER",
        task_type="coding",
        output_summary="Generated RedisCacheWrapper class with get/set/delete/ttl methods",
        quality_score=0.92,
        success=True,
        model_id="qwen3-vl-32b",
    )

    # Recall relevant past episodes for a new task
    episodes = mem.recall("Implement a cache layer for our API", k=3, min_score=0.7)
    for ep in episodes:
        logger.debug(ep.task_summary, ep.output_summary)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get("VETINARI_EPISODE_DB", "./vetinari_episodes.db")
_MAX_EPISODES = int(os.environ.get("VETINARI_MAX_EPISODES", "10000"))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _simple_embedding(text: str, dim: int = 256) -> list[float]:
    """Character n-gram hashing embedding — CPU only, no dependencies."""
    vec = [0.0] * dim
    text_lower = text.lower()
    for n in (2, 3, 4):
        for i in range(len(text_lower) - n + 1):
            gram = text_lower[i : i + n]
            h = int(hashlib.md5(gram.encode()).hexdigest(), 16)  # noqa: S324
            vec[h % dim] += 1.0
    # L2-normalise
    norm = (sum(x * x for x in vec) ** 0.5) or 1.0
    return [x / norm for x in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))


def _get_embedder():
    """Try to load sentence-transformers; fall back to simple hashing."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        def encode(text: str) -> list[float]:
            return model.encode([text], show_progress_bar=False)[0].tolist()

        return encode
    except Exception:
        return _simple_embedding


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """A single past execution record."""

    episode_id: str
    timestamp: str
    task_summary: str  # Truncated task description (for display)
    agent_type: str
    task_type: str
    output_summary: str  # Truncated output (key facts)
    quality_score: float
    success: bool
    model_id: str
    embedding: list[float]  # For similarity search
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "task_summary": self.task_summary,
            "agent_type": self.agent_type,
            "task_type": self.task_type,
            "output_summary": self.output_summary,
            "quality_score": self.quality_score,
            "success": self.success,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class EpisodeMemory:
    """Persistent episodic memory with similarity-based retrieval."""

    _instance: EpisodeMemory | None = None
    _cls_lock = threading.Lock()

    def __init__(self, db_path: str = _DB_PATH):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._embedder = _get_embedder()
        # In-memory embedding index: list of (episode_id, embedding)
        self._index: list[tuple] = []
        self._init_db()
        self._load_index()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, db_path: str = _DB_PATH) -> EpisodeMemory:
        """Get instance.

        Returns:
            The EpisodeMemory result.
        """
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls(db_path=db_path)
        return cls._instance

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    task_summary TEXT,
                    agent_type TEXT,
                    task_type TEXT,
                    output_summary TEXT,
                    quality_score REAL,
                    success INTEGER,
                    model_id TEXT,
                    embedding TEXT,
                    metadata TEXT,
                    created_at REAL DEFAULT (unixepoch()),
                    importance REAL DEFAULT 0.5
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_type ON episodes(task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_score ON episodes(quality_score)")

    def _load_index(self) -> None:
        """Load embeddings from DB into memory for fast search."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT episode_id, embedding FROM episodes ORDER BY created_at DESC LIMIT ?", (_MAX_EPISODES,)
                ).fetchall()
            with self._lock:
                self._index = [(row[0], json.loads(row[1])) for row in rows if row[1]]
        except Exception as e:
            logger.debug("[EpisodeMemory] Index load failed: %s", e)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
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
        """Record a new episode. Returns the episode_id.

        Args:
            task_description: The task description.
            agent_type: The agent type.
            task_type: The task type.
            output_summary: The output summary.
            quality_score: The quality score.
            success: The success.
            model_id: The model id.
            metadata: The metadata.

        Returns:
            The result string.
        """
        import uuid

        episode_id = f"ep_{uuid.uuid4().hex[:8]}"
        task_summary = task_description[:300]
        output_summary = output_summary[:500]

        embedding = self._embedder(f"{task_type}: {task_summary}")

        # Importance score: quality * recency weight
        importance = round(quality_score * (1.0 if success else 0.5), 3)

        ep = Episode(
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            task_summary=task_summary,
            agent_type=agent_type,
            task_type=task_type,
            output_summary=output_summary,
            quality_score=quality_score,
            success=success,
            model_id=model_id,
            embedding=embedding,
            metadata=metadata or {},
        )

        with self._lock:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO episodes
                           (episode_id, timestamp, task_summary, agent_type, task_type,
                            output_summary, quality_score, success, model_id, embedding,
                            metadata, importance)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ep.episode_id,
                            ep.timestamp,
                            ep.task_summary,
                            ep.agent_type,
                            ep.task_type,
                            ep.output_summary,
                            ep.quality_score,
                            int(ep.success),
                            ep.model_id,
                            json.dumps(ep.embedding),
                            json.dumps(ep.metadata),
                            importance,
                        ),
                    )
                self._index.append((episode_id, embedding))
                # Evict if over limit
                if len(self._index) > _MAX_EPISODES:
                    self._evict()
            except Exception as e:
                logger.debug("[EpisodeMemory] Record failed: %s", e)

        return episode_id

    def _evict(self) -> None:
        """Remove the least-important episodes to stay within MAX_EPISODES."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                # Delete bottom 10% by importance
                max(0, _MAX_EPISODES - _MAX_EPISODES // 10)
                conn.execute(
                    "DELETE FROM episodes WHERE episode_id IN "
                    "(SELECT episode_id FROM episodes ORDER BY importance ASC LIMIT ?)",
                    (_MAX_EPISODES // 10,),
                )
            # Rebuild index
            self._load_index()
        except Exception as e:
            logger.debug("[EpisodeMemory] Eviction failed: %s", e)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    # Episodes with benchmark_score > this threshold get a relevance boost
    BENCHMARK_BOOST_THRESHOLD = 0.8
    BENCHMARK_BOOST_FACTOR = 1.5

    def recall(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        task_type: str | None = None,
        successful_only: bool = False,
    ) -> list[Episode]:
        """Return the k most relevant past episodes for a query.

        Episodes with ``benchmark_score > 0.8`` in their metadata receive a
        1.5x relevance boost, making benchmark-validated episodes surface
        more readily.

        Args:
            query:           The current task description to match against
            k:               Number of episodes to return
            min_score:       Minimum quality_score filter
            task_type:       Filter to a specific task type
            successful_only: Only return successful episodes

        Returns:
            List of results.
        """
        query_emb = self._embedder(query)

        with self._lock:
            index_snapshot = list(self._index)

        # Score all index entries
        scored = [(ep_id, _cosine_similarity(query_emb, emb)) for ep_id, emb in index_snapshot]
        scored.sort(key=lambda x: -x[1])
        top_ids = [ep_id for ep_id, _ in scored[: k * 3]]  # Fetch 3x for filtering

        if not top_ids:
            return []

        # Build a similarity lookup for re-ranking with benchmark boost
        similarity_map = dict(scored[: k * 3])

        # Fetch full records from DB
        try:
            placeholders = ",".join("?" for _ in top_ids)
            query_sql = f"SELECT episode_id, timestamp, task_summary, agent_type, task_type, output_summary, quality_score, success, model_id, embedding, metadata FROM episodes WHERE episode_id IN ({placeholders})"  # noqa: S608
            params = top_ids

            if min_score > 0:
                query_sql += " AND quality_score >= ?"
                params = [*list(params), min_score]
            if task_type:
                query_sql += " AND task_type = ?"
                params = [*list(params), task_type]
            if successful_only:
                query_sql += " AND success = 1"

            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(query_sql, params).fetchall()

            episodes = [
                Episode(
                    episode_id=row[0],
                    timestamp=row[1],
                    task_summary=row[2],
                    agent_type=row[3],
                    task_type=row[4],
                    output_summary=row[5],
                    quality_score=row[6],
                    success=bool(row[7]),
                    model_id=row[8],
                    embedding=json.loads(row[9] or "[]"),
                    metadata=json.loads(row[10] or "{}"),
                )
                for row in rows
            ]

            # Re-rank with benchmark boost: episodes with high benchmark_score
            # in metadata get 1.5x relevance boost
            def _boosted_score(ep: Episode) -> float:
                base = similarity_map.get(ep.episode_id, 0.0)
                bench_score = ep.metadata.get("benchmark_score", 0.0)
                if isinstance(bench_score, (int, float)) and bench_score > self.BENCHMARK_BOOST_THRESHOLD:
                    return base * self.BENCHMARK_BOOST_FACTOR
                return base

            episodes.sort(key=_boosted_score, reverse=True)
            return episodes[:k]

        except Exception as e:
            logger.debug("[EpisodeMemory] Recall failed: %s", e)
            return []

    def get_failure_patterns(self, agent_type: str, task_type: str) -> list[str]:
        """Return common failure summaries for an agent/task combination.

        Args:
            agent_type: The agent type.
            task_type: The task type.

        Returns:
            The result string.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    """SELECT output_summary FROM episodes
                       WHERE agent_type = ? AND task_type = ? AND success = 0
                       ORDER BY created_at DESC LIMIT 10""",
                    (agent_type, task_type),
                ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def get_stats(self) -> dict[str, Any]:
        """Return memory statistics.

        Returns:
            The result string.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                successful = conn.execute("SELECT COUNT(*) FROM episodes WHERE success = 1").fetchone()[0]
                avg_score = conn.execute("SELECT AVG(quality_score) FROM episodes").fetchone()[0] or 0.0
            return {
                "total_episodes": total,
                "successful": successful,
                "avg_quality_score": round(avg_score, 3),
                "index_size": len(self._index),
                "db_path": self._db_path,
            }
        except Exception as e:
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_episode_memory: EpisodeMemory | None = None
_mem_lock = threading.Lock()


def get_episode_memory(db_path: str = _DB_PATH) -> EpisodeMemory:
    """Return the global EpisodeMemory singleton.

    Returns:
        The EpisodeMemory result.
    """
    global _episode_memory
    if _episode_memory is None:
        with _mem_lock:
            if _episode_memory is None:
                _episode_memory = EpisodeMemory.get_instance(db_path=db_path)
    return _episode_memory
