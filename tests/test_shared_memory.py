"""Tests for agent memory integration — S28 wiring of UnifiedMemoryStore into agent execution.

Covers: UnifiedMemoryStore search delegation, episode recording,
embeddings_available probing, and agent memory recall integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_memory_entry
from vetinari.memory.interfaces import MemoryEntry
from vetinari.memory.unified import UnifiedMemoryStore
from vetinari.types import AgentType

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh UnifiedMemoryStore backed by a temp database."""
    db_path = str(tmp_path / "test_shared.db")
    with patch("vetinari.memory.unified._embed_via_local_inference", return_value=None):
        s = UnifiedMemoryStore(
            db_path=db_path,
            embedding_api_url="http://127.0.0.1:99999",
            max_entries=100,
            session_max=10,
        )
        yield s
        s.close()


# ── Search Tests ───────────────────────────────────────────────────────────


class TestSearch:
    """UnifiedMemoryStore.search() used by agent memory recall."""

    def test_search_returns_entries(self, store):
        """search() returns list of MemoryEntry objects."""
        with patch("vetinari.memory.unified._embed_via_local_inference", return_value=None):
            store.remember(make_memory_entry(content="python error handling best practices"))
        results = store.search("error handling")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, MemoryEntry)

    def test_search_with_agent_filter(self, store):
        """search() passes agent filter through to store."""
        with patch("vetinari.memory.unified._embed_via_local_inference", return_value=None):
            store.remember(make_memory_entry(content="task routing logic", agent="foreman"))
            store.remember(make_memory_entry(content="code review feedback", agent="inspector"))
        results = store.search("routing", agent="foreman")
        for r in results:
            assert r.agent == "foreman"

    def test_search_respects_limit(self, store):
        """search() respects the limit parameter."""
        with patch("vetinari.memory.unified._embed_via_local_inference", return_value=None):
            for i in range(10):
                store.remember(make_memory_entry(content=f"memory entry number {i}"))
        results = store.search("memory entry", limit=3)
        assert len(results) <= 3


# ── Episode Recording Tests ────────────────────────────────────────────────


class TestRecordEpisode:
    """UnifiedMemoryStore.record_episode() used by agent execution."""

    def test_record_episode_returns_id(self, store):
        """record_episode() returns a non-empty episode ID."""
        with patch("vetinari.memory.unified._embed_via_local_inference", return_value=None):
            ep_id = store.record_episode(
                task_description="test task",
                agent_type="worker",
                task_type="build",
                output_summary="built successfully",
                quality_score=0.9,
                success=True,
                model_id="test-model",
            )
        assert ep_id
        assert ep_id.startswith("ep_")


# ── Embeddings Availability Tests ──────────────────────────────────────────


class TestStoreEmbeddingsAvailable:
    """UnifiedMemoryStore.embeddings_available() method."""

    def test_store_embeddings_unavailable(self, store):
        """Returns False when endpoint is unreachable."""
        with patch("vetinari.memory.unified._embed_via_local_inference", return_value=None):
            assert store.embeddings_available() is False

    def test_store_embeddings_available(self, store):
        """Returns True when endpoint responds."""
        with patch("vetinari.memory.unified._embed_via_local_inference", return_value=[0.1]):
            assert store.embeddings_available() is True


# ── Agent Integration Tests ────────────────────────────────────────────────


class TestAgentMemoryRecall:
    """Test that base_agent._recall_relevant_memories uses UnifiedMemoryStore."""

    def test_recall_uses_unified_memory_store(self, store):
        """_recall_relevant_memories delegates to UnifiedMemoryStore.search()."""
        with patch("vetinari.memory.unified._embed_via_local_inference", return_value=None):
            store.remember(make_memory_entry(content="deployment strategy for microservices"))

        mock_store = MagicMock()
        mock_entry = MagicMock()
        mock_entry.to_dict.return_value = {"content": "deployment tip"}
        mock_store.search.return_value = [mock_entry]

        # Patch the module-level cache in base_agent directly so the cached
        # function reference is replaced regardless of import order.
        import vetinari.agents.base_agent as _ba

        with patch.object(_ba, "_cached_shared_memory_cls", new=lambda: mock_store):
            from vetinari.agents.base_agent import BaseAgent
            from vetinari.agents.contracts import AgentResult, VerificationResult

            class _TestAgent(BaseAgent):
                def execute(self, task):
                    return AgentResult(success=True, output="stub")

                def verify(self, output):
                    return VerificationResult(passed=True, issues=[])

                def get_system_prompt(self):
                    return "test"

            agent = _TestAgent(agent_type=AgentType.WORKER)
            results = agent._recall_relevant_memories("deployment")

            mock_store.search.assert_called_once_with(
                "deployment",
                agent=AgentType.WORKER.value,
                limit=5,
            )
            assert results == [{"content": "deployment tip"}]

    def test_recall_handles_error_gracefully(self):
        """_recall_relevant_memories returns [] on any exception."""
        import vetinari.agents.base_agent as _ba

        def _raise():
            raise RuntimeError("init failed")

        with patch.object(_ba, "_cached_shared_memory_cls", new=_raise):
            from vetinari.agents.base_agent import BaseAgent
            from vetinari.agents.contracts import AgentResult, VerificationResult

            class _TestAgent(BaseAgent):
                def execute(self, task):
                    return AgentResult(success=True, output="stub")

                def verify(self, output):
                    return VerificationResult(passed=True, issues=[])

                def get_system_prompt(self):
                    return "test"

            agent = _TestAgent(agent_type=AgentType.WORKER)
            results = agent._recall_relevant_memories("anything")
            assert results == []
