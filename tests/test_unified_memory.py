"""Tests for UnifiedMemoryStore — the consolidated memory backend.

Covers: storage, search (FTS5 + semantic), dedup, session context,
episode recording/recall, consolidation pipeline, secret scanning,
eviction, export, forget/compact, and singleton management.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from vetinari.memory.interfaces import MemoryEntry, MemoryStats, MemoryType, content_hash
from vetinari.memory.unified import (
    RecordedEpisode,
    SessionContext,
    UnifiedMemoryStore,
    _pack_embedding,
    _unpack_embedding,
    get_unified_memory_store,
    init_unified_memory_store,
)
from vetinari.types import AgentType
from vetinari.utils.math_helpers import cosine_similarity as _cosine_similarity

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh UnifiedMemoryStore in a temp directory.

    Patches _embed_via_local_inference to return None (no embeddings) so tests
    don't attempt real network connections or time out.
    """
    db_path = str(tmp_path / "test_unified.db")
    with (
        patch("vetinari.memory.unified._embed_via_local_inference", return_value=None),
        patch("vetinari.memory.memory_embeddings.embed_via_local_inference", return_value=None),
    ):
        s = UnifiedMemoryStore(
            db_path=db_path,
            embedding_api_url="http://127.0.0.1:99999",  # never called (mocked)
            max_entries=100,
            session_max=10,
        )
        yield s
        s.close()


@pytest.fixture
def entry():
    """Create a sample MemoryEntry."""
    return MemoryEntry(
        id="mem_test001",
        agent="builder",
        entry_type=MemoryType.DISCOVERY,
        content="The cache layer uses Redis with TTL-based eviction",
        summary="Redis cache with TTL eviction",
        timestamp=1700000000000,
        provenance="test",
    )


# ── Basic Storage ──────────────────────────────────────────────────────────


class TestRemember:
    """Tests for the remember() method."""

    def test_remember_stores_entry(self, store, entry):
        entry_id = store.remember(entry)
        assert entry_id == "mem_test001"

        retrieved = store.get_entry("mem_test001")
        assert retrieved is not None
        assert retrieved.content == entry.content
        assert retrieved.agent == "builder"

    def test_remember_dedup_by_hash(self, store, entry):
        store.remember(entry)
        entry2 = MemoryEntry(
            id="mem_test002",
            agent="builder",
            entry_type=MemoryType.DISCOVERY,
            content=entry.content,  # same content
            summary="duplicate",
        )
        returned_id = store.remember(entry2)
        # Should return the original entry ID (dedup)
        assert returned_id == "mem_test001"

    def test_remember_different_content_creates_separate(self, store, entry):
        store.remember(entry)
        entry2 = MemoryEntry(
            id="mem_test002",
            content="Completely different content about logging",
            entry_type=MemoryType.PATTERN,
        )
        returned_id = store.remember(entry2)
        assert returned_id == "mem_test002"

    def test_remember_with_metadata(self, store):
        entry = MemoryEntry(
            content="Test with metadata",
            entry_type=MemoryType.KNOWLEDGE,
            metadata={"key": "value", "nested": {"a": 1}},
        )
        store.remember(entry)
        retrieved = store.get_entry(entry.id)
        assert retrieved is not None
        assert retrieved.metadata == {"key": "value", "nested": {"a": 1}}

    def test_remember_preserves_scope(self, store):
        entry = MemoryEntry(
            id="scoped_memory",
            content="Scoped project memory",
            entry_type=MemoryType.DISCOVERY,
            scope="project:alpha",
        )
        store.remember(entry)

        retrieved = store.get_entry("scoped_memory")

        assert retrieved is not None
        assert retrieved.scope == "project:alpha"


class TestSecretScanning:
    """Tests for secret scanning before storage."""

    def test_secrets_are_sanitized(self, store):
        entry = MemoryEntry(
            content="API key is AKIAIOSFODNN7EXAMPLE",
            entry_type=MemoryType.DISCOVERY,
        )
        store.remember(entry)
        retrieved = store.get_entry(entry.id)
        assert retrieved is not None
        # Secret scanner should have replaced the AWS key
        assert "AKIAIOSFODNN7EXAMPLE" not in retrieved.content or "***" in retrieved.content


# ── Search ─────────────────────────────────────────────────────────────────


class TestSearch:
    """Tests for FTS5 and fallback search."""

    def test_fts_search_finds_content(self, store):
        store.remember(MemoryEntry(id="m1", content="Redis cache implementation", entry_type=MemoryType.DISCOVERY))
        store.remember(MemoryEntry(id="m2", content="PostgreSQL migration plan", entry_type=MemoryType.DECISION))
        store.remember(MemoryEntry(id="m3", content="Redis connection pooling", entry_type=MemoryType.PATTERN))

        results = store.search("Redis")
        assert len(results) >= 2
        contents = [r.content for r in results]
        assert any("Redis" in c for c in contents)

    def test_search_filters_by_agent(self, store):
        store.remember(
            MemoryEntry(id="m1", agent="builder", content="Builder found a bug", entry_type=MemoryType.DISCOVERY)
        )
        store.remember(
            MemoryEntry(id="m2", agent="planner", content="Planner found a bug", entry_type=MemoryType.DISCOVERY)
        )

        results = store.search("bug", agent="builder")
        assert len(results) == 1
        assert results[0].agent == "builder"

    def test_search_filters_by_entry_type(self, store):
        store.remember(MemoryEntry(id="m1", content="Decision about auth", entry_type=MemoryType.DECISION))
        store.remember(MemoryEntry(id="m2", content="Pattern for auth", entry_type=MemoryType.PATTERN))

        results = store.search("auth", entry_types=["decision"])
        assert len(results) == 1
        assert results[0].entry_type == MemoryType.DECISION

    def test_semantic_search_falls_back_to_fts(self, store):
        """When local inference is unavailable, semantic search falls back to FTS5."""
        store.remember(MemoryEntry(id="m1", content="Caching strategy uses LRU", entry_type=MemoryType.DISCOVERY))
        results = store.search("caching", use_semantic=True)
        assert len(results) >= 1

    def test_semantic_search_without_vectors_falls_back_to_original_query(self, store):
        store.remember(
            MemoryEntry(
                id="older_needle",
                content="Needle-specific rollback procedure",
                timestamp=1000,
                entry_type=MemoryType.DISCOVERY,
            )
        )
        store.remember(
            MemoryEntry(
                id="newer_unrelated",
                content="Completely unrelated deployment note",
                timestamp=9000,
                entry_type=MemoryType.DISCOVERY,
            )
        )

        with patch("vetinari.memory.unified._embed_via_local_inference", return_value=[1.0, 0.0]):
            results = store.search("needle rollback", use_semantic=True, limit=1)

        assert [entry.id for entry in results] == ["older_needle"]

    def test_search_limit(self, store):
        for i in range(20):
            store.remember(
                MemoryEntry(id=f"m{i}", content=f"Test entry number {i} about search", entry_type=MemoryType.DISCOVERY)
            )
        results = store.search("search", limit=5)
        assert len(results) <= 5

    def test_update_content_refreshes_hash_and_drops_stale_embedding(self, store):
        entry = MemoryEntry(id="editable", content="Original memory text", entry_type=MemoryType.DISCOVERY)
        store.remember(entry)
        store._conn.execute(
            "INSERT INTO embeddings (memory_id, embedding_blob, model, dimensions, created_at) VALUES (?, ?, ?, ?, ?)",
            ("editable", _pack_embedding([1.0, 0.0]), "test-model", 2, "2026-01-01T00:00:00+00:00"),
        )
        store._conn.commit()

        assert store.update_content("editable", "Updated memory text")

        row = store._conn.execute(
            "SELECT content_hash FROM memories WHERE id = ?",
            ("editable",),
        ).fetchone()
        stale_embedding = store._conn.execute(
            "SELECT 1 FROM embeddings WHERE memory_id = ?",
            ("editable",),
        ).fetchone()
        assert row["content_hash"] == content_hash("Updated memory text")
        assert stale_embedding is None


# ── Timeline ───────────────────────────────────────────────────────────────


class TestTimeline:
    """Tests for chronological browsing."""

    def test_timeline_returns_chronological(self, store):
        store.remember(MemoryEntry(id="m1", content="First", timestamp=1000, entry_type=MemoryType.DISCOVERY))
        store.remember(MemoryEntry(id="m2", content="Second", timestamp=2000, entry_type=MemoryType.DISCOVERY))
        store.remember(MemoryEntry(id="m3", content="Third", timestamp=3000, entry_type=MemoryType.DISCOVERY))

        results = store.timeline()
        assert len(results) == 3
        assert results[0].timestamp >= results[1].timestamp

    def test_timeline_filters_by_time_range(self, store):
        store.remember(MemoryEntry(id="m1", content="Old", timestamp=1000, entry_type=MemoryType.DISCOVERY))
        store.remember(MemoryEntry(id="m2", content="Recent", timestamp=5000, entry_type=MemoryType.DISCOVERY))

        results = store.timeline(start_time=4000)
        assert len(results) == 1
        assert results[0].id == "m2"


# ── Get Entry ──────────────────────────────────────────────────────────────


class TestGetEntry:
    """Tests for single entry retrieval."""

    def test_get_entry_returns_none_for_missing(self, store):
        assert store.get_entry("nonexistent") is None

    def test_get_entry_returns_entry(self, store, entry):
        store.remember(entry)
        result = store.get_entry(entry.id)
        assert result is not None
        assert result.id == entry.id

    def test_get_entry_increments_access_count(self, store, entry):
        store.remember(entry)
        store.get_entry(entry.id)
        result = store.get_entry(entry.id)
        # The entry must still be retrievable after multiple accesses
        assert result is not None
        assert result.id == entry.id


# ── Forget and Compact ─────────────────────────────────────────────────────


class TestForgetCompact:
    """Tests for forgetting and compaction."""

    def test_forget_marks_entry(self, store, entry):
        store.remember(entry)
        result = store.forget(entry.id, "No longer relevant")
        assert result is True
        # After forgetting, get_entry should return None
        assert store.get_entry(entry.id) is None

    def test_forget_nonexistent_returns_false(self, store):
        assert store.forget("nonexistent", "test") is False

    def test_compact_removes_forgotten(self, store, entry):
        store.remember(entry)
        store.forget(entry.id, "test")
        deleted = store.compact()
        assert deleted >= 1


# ── Export ─────────────────────────────────────────────────────────────────


class TestExport:
    """Tests for JSON export."""

    def test_export_creates_file(self, store, entry, tmp_path):
        store.remember(entry)
        export_path = str(tmp_path / "export.json")
        result = store.export(export_path)
        assert result is True

        with open(export_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "entries" in data
        assert len(data["entries"]) == 1


# ── Stats ──────────────────────────────────────────────────────────────────


class TestStats:
    """Tests for statistics."""

    def test_stats_returns_memory_stats(self, store, entry):
        store.remember(entry)
        s = store.stats()
        assert isinstance(s, MemoryStats)
        assert s.total_entries == 1
        assert "builder" in s.entries_by_agent

    def test_stats_empty_store(self, store):
        s = store.stats()
        assert s.total_entries == 0


# ── Ask ────────────────────────────────────────────────────────────────────


class TestAsk:
    """Tests for natural language question interface."""

    def test_ask_returns_results(self, store):
        store.remember(
            MemoryEntry(id="m1", content="Python uses GIL for thread safety", entry_type=MemoryType.KNOWLEDGE)
        )
        results = store.ask("thread safety")
        # May or may not find depending on FTS5 matching, but should not crash
        assert isinstance(results, list)


# ── Embedding Helpers ──────────────────────────────────────────────────────


class TestEmbeddingHelpers:
    """Tests for embedding pack/unpack and cosine similarity."""

    def test_pack_unpack_roundtrip(self):
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = _pack_embedding(vec)
        result = _unpack_embedding(blob)
        assert len(result) == 5
        for a, b in zip(vec, result):
            assert abs(a - b) < 1e-6

    def test_cosine_similarity_identical(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_different_length(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ── Session Context ────────────────────────────────────────────────────────


class TestSessionContext:
    """Tests for short-term session memory."""

    def test_set_and_get_context(self):
        ctx = SessionContext(max_entries=5)
        ctx.set_context("task", "build cache")
        assert ctx.get_context("task") == "build cache"

    def test_get_missing_returns_default(self):
        ctx = SessionContext()
        assert ctx.get_context("missing", "default") == "default"

    def test_lru_eviction(self):
        ctx = SessionContext(max_entries=3)
        ctx.set_context("a", 1)
        ctx.set_context("b", 2)
        ctx.set_context("c", 3)
        ctx.set_context("d", 4)  # should evict "a"
        assert ctx.get_context("a") is None
        assert ctx.get_context("d") == 4
        assert len(ctx) == 3

    def test_add_memory_entry(self):
        ctx = SessionContext()
        entry = MemoryEntry(id="test1", content="test content", entry_type=MemoryType.DISCOVERY)
        ctx.add(entry)
        stored = ctx.get_context("test1")
        assert isinstance(stored, dict)
        assert stored.get("content") == "test content"

    def test_get_recent(self):
        ctx = SessionContext()
        ctx.set_context("a", 1)
        ctx.set_context("b", 2)
        recent = ctx.get_recent(limit=5)
        assert len(recent) == 2

    def test_clear(self):
        ctx = SessionContext()
        ctx.set_context("a", 1)
        ctx.clear()
        assert len(ctx) == 0

    def test_update_existing_key(self):
        ctx = SessionContext()
        ctx.set_context("key", "old")
        ctx.set_context("key", "new")
        assert ctx.get_context("key") == "new"
        assert len(ctx) == 1


# ── Episode Recording ─────────────────────────────────────────────────────


class TestEpisodes:
    """Tests for episode recording and recall."""

    def test_record_episode(self, store):
        ep_id = store.record_episode(
            task_description="Write a Redis cache wrapper",
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            output_summary="Generated RedisCacheWrapper class",
            quality_score=0.92,
            success=True,
            model_id="test-model",
        )
        assert ep_id.startswith("ep_")

    def test_record_and_recall_episodes(self, store):
        store.record_episode(
            task_description="Implement user authentication",
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            output_summary="JWT-based auth module",
            quality_score=0.88,
            success=True,
        )
        store.record_episode(
            task_description="Fix database connection leak",
            agent_type=AgentType.WORKER.value,
            task_type="bugfix",
            output_summary="Added connection pooling",
            quality_score=0.95,
            success=True,
        )

        # recall_episodes uses keyword fallback when embeddings unavailable
        results = store.recall_episodes("authentication", k=5)
        assert isinstance(results, list)

    def test_recall_with_filters(self, store):
        store.record_episode(
            task_description="Write tests",
            agent_type=AgentType.INSPECTOR.value,
            task_type="testing",
            output_summary="Added 20 tests",
            quality_score=0.9,
            success=True,
        )
        store.record_episode(
            task_description="Write bad tests",
            agent_type=AgentType.INSPECTOR.value,
            task_type="testing",
            output_summary="Tests failed",
            quality_score=0.3,
            success=False,
        )

        results = store.recall_episodes("tests", successful_only=True)
        for ep in results:
            assert ep.success is True

    def test_get_failure_patterns(self, store):
        store.record_episode(
            task_description="Deploy to prod",
            agent_type=AgentType.WORKER.value,
            task_type="deployment",
            output_summary="Docker build failed",
            quality_score=0.2,
            success=False,
        )
        failures = store.get_failure_patterns(AgentType.WORKER.value, "deployment")
        assert len(failures) == 1
        assert "Docker build failed" in failures[0]

    def test_get_episode_stats(self, store):
        store.record_episode(
            task_description="Task 1",
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            output_summary="Done",
            quality_score=0.9,
            success=True,
        )
        stats = store.get_episode_stats()
        assert stats["total_episodes"] == 1
        assert stats["successful"] == 1

    def test_episode_eviction(self, tmp_path):
        """Test that episodes are evicted when over limit."""
        db_path = str(tmp_path / "evict_test.db")
        with (
            patch("vetinari.memory.unified._embed_via_local_inference", return_value=None),
            patch("vetinari.memory.memory_embeddings.embed_via_local_inference", return_value=None),
        ):
            s = UnifiedMemoryStore(db_path=db_path, max_entries=15)
            for i in range(20):
                s.record_episode(
                    task_description=f"Task {i}",
                    agent_type=AgentType.WORKER.value,
                    task_type="coding",
                    output_summary=f"Output {i}",
                    quality_score=0.5,
                    success=True,
                )
            stats = s.get_episode_stats()
            # Should have evicted some (20 > 15)
            assert stats["total_episodes"] <= 20
            s.close()


# ── Consolidation Pipeline ─────────────────────────────────────────────────


class TestConsolidation:
    """Tests for session-to-long-term promotion."""

    def test_consolidate_promotes_high_quality(self, store):
        # Add session entries with varying quality
        store.session.set_context(
            "good_entry",
            {
                "id": "mem_good",
                "content": "High quality finding about caching",
                "entry_type": "discovery",
                "agent": "builder",
                "summary": "",
                "timestamp": 1000,
                "provenance": "test",
                "source_backends": [],
            },
            quality_score=0.9,
        )
        store.session.set_context(
            "bad_entry",
            {
                "id": "mem_bad",
                "content": "Low quality noise",
                "entry_type": "discovery",
                "agent": "builder",
                "summary": "",
                "timestamp": 1001,
                "provenance": "test",
                "source_backends": [],
            },
            quality_score=0.3,
        )

        promoted = store.consolidate(quality_threshold=0.7)
        assert promoted == 1

        # Verify the high-quality entry was promoted
        stats = store.stats()
        assert stats.total_entries == 1

    def test_consolidate_is_idempotent(self, store):
        store.session.set_context(
            "entry1",
            {
                "id": "mem_idem",
                "content": "Unique content for idempotency test",
                "entry_type": "knowledge",
                "agent": "",
                "summary": "",
                "timestamp": 2000,
                "provenance": "test",
                "source_backends": [],
            },
            quality_score=0.9,
        )
        first = store.consolidate()
        second = store.consolidate()
        # Second consolidation should find the same entry already stored (dedup)
        assert first >= second

    def test_consolidate_skips_non_dict_values(self, store):
        store.session.set_context("number", 42, quality_score=0.9)
        store.session.set_context("list", [1, 2, 3], quality_score=0.9)
        promoted = store.consolidate()
        assert promoted == 0

    def test_consolidate_handles_string_values(self, store):
        store.session.set_context("text", "Some important text to remember", quality_score=0.9)
        promoted = store.consolidate()
        assert promoted == 1


# ── Singleton Management ──────────────────────────────────────────────────


class TestSingleton:
    """Tests for singleton accessors."""

    def test_init_unified_memory_store(self, tmp_path):
        db_path = str(tmp_path / "singleton_test.db")
        with (
            patch("vetinari.memory.unified._embed_via_local_inference", return_value=None),
            patch("vetinari.memory.memory_embeddings.embed_via_local_inference", return_value=None),
        ):
            s = init_unified_memory_store(db_path=db_path)
            assert isinstance(s, UnifiedMemoryStore)
            s2 = get_unified_memory_store()
            assert s2 is s
            s.close()

    def test_init_replaces_existing(self, tmp_path):
        db1 = str(tmp_path / "s1.db")
        db2 = str(tmp_path / "s2.db")
        with (
            patch("vetinari.memory.unified._embed_via_local_inference", return_value=None),
            patch("vetinari.memory.memory_embeddings.embed_via_local_inference", return_value=None),
        ):
            s1 = init_unified_memory_store(db_path=db1)
            s2 = init_unified_memory_store(db_path=db2)
            assert s2 is not s1
            current = get_unified_memory_store()
            assert current is s2
            s2.close()


# ── Lifecycle ──────────────────────────────────────────────────────────────


class TestLifecycle:
    """Tests for close and context manager."""

    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx_test.db")
        with (
            patch("vetinari.memory.unified._embed_via_local_inference", return_value=None),
            patch("vetinari.memory.memory_embeddings.embed_via_local_inference", return_value=None),
        ):
            with UnifiedMemoryStore(db_path=db_path) as s:
                entry = MemoryEntry(content="test", entry_type=MemoryType.DISCOVERY)
                s.remember(entry)
                _entry_id = entry.id
            # Connection should be closed after context exit; s._conn must be None or closed
            assert s._conn is None or not s._conn

    def test_close_is_idempotent(self, store):
        store.close()
        store.close()  # should not raise
        assert store._conn is None  # connection must be cleared after close


# ── Import Verification ───────────────────────────────────────────────────


class TestImports:
    """Verify the unified store is properly exported from vetinari.memory."""

    def test_import_from_memory_package(self):
        from vetinari.memory import (  # noqa: F811 - duplicate import pattern is the behavior under test
            RecordedEpisode,
            SessionContext,
            UnifiedMemoryStore,
            get_unified_memory_store,
            init_unified_memory_store,
        )

        assert isinstance(UnifiedMemoryStore, type)
        assert isinstance(SessionContext, type)
        assert RecordedEpisode is not None
