"""Tests for Session 19: Memory Consolidation and Skill Library.

Covers: Ebbinghaus decay, fact graph / chain-aware search, episodic-to-semantic
promotion, SAGE skill library, and enhanced idle-time consolidation.
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import pytest

from vetinari.memory.interfaces import MemoryEntry, MemoryType
from vetinari.memory.memory_storage import (
    DECAY_RATE,
    IMPORTANCE_FACTOR,
    PRUNE_THRESHOLD,
    RECALL_BOOST,
    ebbinghaus_strength,
)
from vetinari.memory.unified import UnifiedMemoryStore

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Fresh UnifiedMemoryStore with embeddings disabled."""
    db_path = str(tmp_path / "test_consolidation.db")
    with (
        patch("vetinari.memory.unified._embed_via_local_inference", return_value=None),
        patch("httpx.post") as mock_httpx_post,
    ):
        mock_httpx_post.return_value = MagicMock(
            status_code=500, raise_for_status=MagicMock(side_effect=Exception("blocked"))
        )
        s = UnifiedMemoryStore(
            db_path=db_path,
            embedding_api_url="http://127.0.0.1:99999",
            max_entries=1000,
            session_max=50,
        )
        yield s
        s.close()


def _make_entry(entry_id: str, content: str, **kwargs) -> MemoryEntry:
    """Build a MemoryEntry with sensible defaults."""
    defaults = {
        "id": entry_id,
        "agent": "builder",
        "entry_type": MemoryType.DISCOVERY,
        "content": content,
        "summary": content[:60],
        "timestamp": int(time.time() * 1000),
        "provenance": "test",
    }
    defaults.update(kwargs)
    return MemoryEntry(**defaults)


# ── US-003: Ebbinghaus Decay ──────────────────────────────────────────────


class TestEbbinghausDecay:
    """Tests for the Ebbinghaus forgetting curve implementation."""

    def test_constants_are_named(self):
        assert DECAY_RATE == 0.16
        assert IMPORTANCE_FACTOR == 0.8
        assert RECALL_BOOST == 0.2
        assert PRUNE_THRESHOLD == 0.1

    def test_fresh_memory_full_strength(self):
        now = int(time.time() * 1000)
        strength = ebbinghaus_strength(importance=1.0, created_ts_ms=now, now_ms=now)
        assert strength == pytest.approx(1.0, abs=0.01)

    def test_importance_05_day30_no_recall(self):
        now = int(time.time() * 1000)
        thirty_days_ago = now - (30 * 86400 * 1000)
        strength = ebbinghaus_strength(importance=0.5, created_ts_ms=thirty_days_ago, recall_count=0, now_ms=now)
        # Manual: 0.5 * exp(-0.16 * (1 - 0.5*0.8) * 30) * (1 + 0*0.2)
        #       = 0.5 * exp(-0.16 * 0.6 * 30) = 0.5 * exp(-2.88) ≈ 0.028
        assert strength < PRUNE_THRESHOLD  # should be prunable

    def test_high_recall_boosts_strength(self):
        now = int(time.time() * 1000)
        thirty_days_ago = now - (30 * 86400 * 1000)
        without_recall = ebbinghaus_strength(importance=0.5, created_ts_ms=thirty_days_ago, recall_count=0, now_ms=now)
        with_recall = ebbinghaus_strength(importance=0.5, created_ts_ms=thirty_days_ago, recall_count=10, now_ms=now)
        assert with_recall > without_recall

    def test_high_importance_decays_slower(self):
        now = int(time.time() * 1000)
        thirty_days_ago = now - (30 * 86400 * 1000)
        high_imp = ebbinghaus_strength(importance=1.0, created_ts_ms=thirty_days_ago, recall_count=0, now_ms=now)
        low_imp = ebbinghaus_strength(importance=0.2, created_ts_ms=thirty_days_ago, recall_count=0, now_ms=now)
        assert high_imp > low_imp

    def test_formula_matches_specification(self):
        now_ms = 1700000000000
        created_ms = now_ms - (10 * 86400 * 1000)  # 10 days ago
        importance = 0.7
        recall_count = 3
        days = 10.0

        expected = (
            importance
            * math.exp(-DECAY_RATE * (1.0 - importance * IMPORTANCE_FACTOR) * days)
            * (1.0 + recall_count * RECALL_BOOST)
        )
        actual = ebbinghaus_strength(
            importance=importance,
            created_ts_ms=created_ms,
            recall_count=recall_count,
            now_ms=now_ms,
        )
        assert actual == pytest.approx(min(expected, 1.0), abs=0.001)


# ── US-007: Fact Graph & Chain-Aware Search ───────────────────────────────


class TestFactGraph:
    """Tests for fact_graph() and chain-aware search."""

    def test_fact_graph_returns_chain(self, store):
        """A -> supersedes B -> supersedes C returns [A, B, C]."""
        c = _make_entry("mem_c", "Version 1 of the fact")
        b = _make_entry("mem_b", "Version 2 of the fact", supersedes_id="mem_c")
        a = _make_entry("mem_a", "Version 3 of the fact", supersedes_id="mem_b")

        store.remember(c)
        store.remember(b)
        store.remember(a)

        # Set relationships
        store.create_relationship("mem_b", "mem_c", "supersedes")
        store.create_relationship("mem_a", "mem_b", "supersedes")

        chain = store.fact_graph("mem_a")
        chain_ids = [e.id for e in chain]
        assert chain_ids == ["mem_a", "mem_b", "mem_c"]

    def test_fact_graph_single_entry(self, store):
        """Entry with no supersedes_id returns single-element list."""
        entry = _make_entry("mem_solo", "Standalone fact")
        store.remember(entry)

        chain = store.fact_graph("mem_solo")
        assert len(chain) == 1
        assert chain[0].id == "mem_solo"

    def test_fact_graph_nonexistent_id(self, store):
        chain = store.fact_graph("nonexistent_id")
        assert chain == []

    def test_search_excludes_superseded_by_default(self, store):
        """search() with include_superseded=False filters out superseded entries."""
        # Distinct content avoids content-hash dedup; single words for FTS phrase matching
        old = _make_entry("mem_old", "eviction old policy")
        new = _make_entry("mem_new", "eviction new policy")

        store.remember(old)
        store.remember(new)
        store.create_relationship("mem_new", "mem_old", "supersedes")

        results = store.search("eviction", include_superseded=False)
        result_ids = [r.id for r in results]
        assert "mem_old" not in result_ids
        assert "mem_new" in result_ids

    def test_search_includes_superseded_when_requested(self, store):
        """search(include_superseded=True) returns all entries."""
        old = _make_entry("mem_old2", "schema original version")
        new = _make_entry("mem_new2", "schema updated version")

        store.remember(old)
        store.remember(new)
        store.create_relationship("mem_new2", "mem_old2", "supersedes")

        results = store.search("schema", include_superseded=True)
        result_ids = [r.id for r in results]
        assert "mem_old2" in result_ids
        assert "mem_new2" in result_ids

    def test_create_relationship_returns_true(self, store):
        a = _make_entry("mem_rel_a", "Fact A")
        b = _make_entry("mem_rel_b", "Fact B")
        store.remember(a)
        store.remember(b)

        assert store.create_relationship("mem_rel_a", "mem_rel_b", "elaborates") is True

    def test_create_relationship_nonexistent_returns_false(self, store):
        assert store.create_relationship("nonexistent", "also_nonexistent", "supersedes") is False


# ── US-004: Episodic-to-Semantic Promotion ────────────────────────────────


class TestEpisodePromotion:
    """Tests for promote_episodes_to_semantic()."""

    def _insert_episodes(self, store, task_type: str, count: int):
        """Insert *count* successful episodes of the given task_type."""
        for i in range(count):
            store.record_episode(
                task_description=f"Test task {task_type} #{i}",
                agent_type="WORKER",
                task_type=task_type,
                output_summary=f"Completed {task_type} task #{i} successfully",
                quality_score=0.85,
                success=True,
                model_id="test-model",
                metadata={},
            )

    def test_promotion_triggers_at_threshold(self, store):
        """10 similar episodes triggers pattern extraction."""
        self._insert_episodes(store, "coding", 10)
        promoted = store.promote_episodes_to_semantic(threshold=10)
        assert promoted == 1

    def test_no_promotion_below_threshold(self, store):
        """9 episodes does NOT trigger promotion."""
        self._insert_episodes(store, "coding", 9)
        promoted = store.promote_episodes_to_semantic(threshold=10)
        assert promoted == 0

    def test_promoted_entry_is_pattern_type(self, store):
        """Promoted entry has entry_type=pattern and meaningful content."""
        self._insert_episodes(store, "research", 10)
        promoted = store.promote_episodes_to_semantic(threshold=10)
        assert promoted == 1

        # Query DB directly — FTS may not index without embedding service
        row = store._conn.execute(
            "SELECT id, content, entry_type, metadata_json FROM memories WHERE entry_type = 'pattern' AND forgotten = 0"
        ).fetchone()
        assert row is not None, "Pattern entry should exist in DB after promotion"
        assert "research" in row["content"].lower()
        assert row["entry_type"] == "pattern"

        import json

        meta = json.loads(row["metadata_json"])
        assert "source_episode_ids" in meta
        assert len(meta["source_episode_ids"]) == 10

    def test_double_promotion_is_idempotent(self, store):
        """Running promotion twice doesn't create duplicate patterns."""
        self._insert_episodes(store, "coding", 10)
        first = store.promote_episodes_to_semantic(threshold=10)
        second = store.promote_episodes_to_semantic(threshold=10)
        assert first == 1
        assert second == 0  # already promoted


# ── US-006: Skill Library ─────────────────────────────────────────────────


class TestSkillLibrary:
    """Tests for the SAGE-inspired skill library."""

    def test_extract_skill_from_successful_episodes(self):
        from vetinari.learning.skill_library import extract_skill

        episodes = [
            {
                "episode_id": f"ep_{i:04d}",
                "task_summary": f"Implement feature #{i}",
                "output_summary": f"Feature #{i} implemented with tests",
                "quality_score": 0.9,
                "success": True,
                "task_type": "coding",
            }
            for i in range(5)
        ]
        skill = extract_skill(episodes)
        assert skill is not None
        assert skill.task_type == "coding"
        assert skill.success_count == 5
        assert skill.avg_quality == pytest.approx(0.9, abs=0.01)
        assert len(skill.source_episodes) == 5
        assert "coding" in skill.template.lower()

    def test_extract_skill_insufficient_episodes(self):
        from vetinari.learning.skill_library import MIN_EPISODES_FOR_SKILL, extract_skill

        episodes = [
            {
                "episode_id": "ep_0001",
                "task_summary": "Single task",
                "output_summary": "Done",
                "quality_score": 0.9,
                "success": True,
                "task_type": "coding",
            }
        ]
        assert extract_skill(episodes) is None

    def test_extract_skill_low_quality_rejected(self):
        from vetinari.learning.skill_library import extract_skill

        episodes = [
            {
                "episode_id": f"ep_{i:04d}",
                "task_summary": f"Low quality task #{i}",
                "output_summary": f"Mediocre result #{i}",
                "quality_score": 0.3,
                "success": True,
                "task_type": "coding",
            }
            for i in range(10)
        ]
        assert extract_skill(episodes) is None

    def test_skill_roundtrip_dict(self):
        from vetinari.learning.skill_library import Skill

        skill = Skill(
            id="skill_test01",
            name="test skill",
            task_type="coding",
            template="Step 1: do X\nStep 2: do Y",
            source_episodes=["ep_001", "ep_002"],
            success_count=2,
            avg_quality=0.85,
        )
        d = skill.to_dict()
        restored = Skill.from_dict(d)
        assert restored.id == skill.id
        assert restored.template == skill.template
        assert restored.source_episodes == skill.source_episodes

    def test_find_matching_skill_returns_none_when_empty(self, store):
        from vetinari.learning.skill_library import find_matching_skill

        with patch("vetinari.memory.unified.get_unified_store", return_value=store):
            result = find_matching_skill("implement a REST API endpoint")
            assert result is None

    def test_store_and_get_skill(self, store):
        from vetinari.learning.skill_library import Skill, get_skill, store_skill

        skill = Skill(
            id="skill_abc123",
            name="API endpoint skill",
            task_type="coding",
            template="1. Define route\n2. Implement handler\n3. Add tests",
            source_episodes=["ep_001"],
            success_count=5,
            avg_quality=0.88,
        )

        with patch("vetinari.memory.unified.get_unified_store", return_value=store):
            stored_id = store_skill(skill)
            assert stored_id == "skill_abc123"

            retrieved = get_skill("skill_abc123")
            assert retrieved is not None
            assert retrieved.name == "API endpoint skill"
            assert retrieved.avg_quality == pytest.approx(0.88, abs=0.01)


# ── US-005: Enhanced Idle Consolidation ───────────────────────────────────


class TestIdleConsolidation:
    """Tests for the enhanced _consolidate_memory in TrainingScheduler."""

    def test_consolidation_calls_promote_episodes(self, store):
        """_consolidate_memory should call promote_episodes_to_semantic."""
        with patch("vetinari.memory.unified.get_unified_store", return_value=store):
            from vetinari.training.idle_scheduler import TrainingScheduler

            scheduler = TrainingScheduler.__new__(TrainingScheduler)
            with (
                patch.object(scheduler, "_prune_weak_memories") as mock_prune,
                patch.object(scheduler, "_flag_contradictions") as mock_flag,
                patch.object(store, "consolidate", return_value=0),
                patch.object(store, "promote_episodes_to_semantic", return_value=0) as mock_promote,
            ):
                scheduler._consolidate_memory()
                mock_promote.assert_called_once()
                mock_prune.assert_called_once_with(store)
                mock_flag.assert_called_once_with(store)

    def test_prune_weak_memories_removes_decayed(self, store):
        """_prune_weak_memories should forget entries below Ebbinghaus threshold."""
        # Insert an old, low-importance, never-recalled memory
        old_entry = _make_entry(
            "mem_weak",
            "Stale observation from long ago",
            timestamp=1000000000000,  # ~2001, very old
        )
        store.remember(old_entry)

        from vetinari.training.idle_scheduler import TrainingScheduler

        scheduler = TrainingScheduler.__new__(TrainingScheduler)
        scheduler._prune_weak_memories(store)

        # Entry should still exist if timestamp is ISO (our logic skips ISO)
        # But the method is tested to not crash
        retrieved = store.get_entry("mem_weak")
        # The entry uses epoch-ms timestamp, so it should be prunable
        # depending on the importance stored. Default importance is 0.5.
        # With ~25 years of decay at importance 0.5, strength ≈ 0
        # But the SQL INSERT sets importance=0.5. Let's verify it was forgotten.
        # Note: get_entry returns None for forgotten entries
        # This depends on the actual strength calculation — at 25 years with 0.5 importance,
        # it will definitely be below threshold.
        assert retrieved is None

    def test_flag_contradictions_logs_conflicts(self, store, caplog):
        """_flag_contradictions should log entries with contradicts relationship."""
        a = _make_entry("mem_contra_a", "The API uses REST")
        b = _make_entry("mem_contra_b", "The API uses GraphQL")
        store.remember(a)
        store.remember(b)
        store.create_relationship("mem_contra_b", "mem_contra_a", "contradicts")

        from vetinari.training.idle_scheduler import TrainingScheduler

        scheduler = TrainingScheduler.__new__(TrainingScheduler)
        with caplog.at_level("INFO", logger="vetinari.training.idle_scheduler"):
            scheduler._flag_contradictions(store)
        assert "contradiction detected" in caplog.text
