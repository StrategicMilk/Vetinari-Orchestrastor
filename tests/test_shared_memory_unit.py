"""Tests for vetinari.memory.shared_memory — legacy SharedMemory."""

from __future__ import annotations

import tempfile
import warnings

# Suppress the deprecation warning when importing shared_memory
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from vetinari.memory.shared_memory import (
        AgentName,
        MemoryEntry,
        SharedMemory,
    )


class TestAgentName:
    """Tests for the AgentName enum."""

    def test_core_agents_present(self):
        names = {m.value for m in AgentName}
        assert "plan" in names
        assert "build" in names
        assert "oracle" in names
        assert "quality" in names
        assert "operations" in names
        assert "researcher" in names


class TestMemoryEntry:
    """Tests for the MemoryEntry dataclass."""

    def test_required_fields(self):
        entry = MemoryEntry(
            entry_id="abc",
            agent_name="plan",
            memory_type="fact",
            summary="test summary",
            content="test content",
            timestamp="2026-01-01T00:00:00",
        )
        assert entry.entry_id == "abc"
        assert entry.agent_name == "plan"

    def test_defaults(self):
        entry = MemoryEntry("id", "agent", "type", "sum", "content", "ts")
        assert entry.tags == []
        assert entry.resolved is False
        assert entry.confidence == 1.0
        assert entry.provenance == "agent"

    def test_to_dict(self):
        entry = MemoryEntry("id", "agent", "fact", "sum", "content", "ts")
        d = entry.to_dict()
        assert d["entry_id"] == "id"
        assert d["agent_name"] == "agent"
        assert isinstance(d, dict)


class TestSharedMemory:
    """Tests for the SharedMemory class."""

    def test_init_creates_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SharedMemory(storage_path=tmpdir)
            assert sm.storage_path.exists()
            assert sm.entries == [] or isinstance(sm.entries, list)

    def test_add_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SharedMemory(storage_path=tmpdir)
            entry = sm.add(
                agent_name="plan",
                memory_type="fact",
                summary="test fact",
                content="detailed content",
            )
            assert entry.entry_id
            assert entry.agent_name == "plan"
            assert len(sm.entries) >= 1

    def test_search_by_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SharedMemory(storage_path=tmpdir)
            sm.add("plan", "fact", "authentication design", "JWT tokens chosen")
            sm.add("build", "fact", "database schema", "PostgreSQL tables")
            results = sm.search("authentication")
            assert len(results) >= 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm1 = SharedMemory(storage_path=tmpdir)
            sm1.add("plan", "fact", "persisted item", "content here")

            # Reset singleton for fresh load
            SharedMemory._instance = None
            sm2 = SharedMemory(storage_path=tmpdir)
            assert len(sm2.entries) >= 1
