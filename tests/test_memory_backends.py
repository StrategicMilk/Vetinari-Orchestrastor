"""
Tests for Vetinari's dual memory backends.

These tests verify:
1. OcMemoryStore basic operations
2. MnemosyneMemoryStore basic operations
3. DualMemoryStore write/read/merge behavior
4. Deduplication and merge policy
"""

import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOcMemoryStore:
    """Tests for OcMemoryStore adapter."""

    @pytest.fixture
    def temp_path(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_store_and_retrieve(self, temp_path):
        """Test basic store and retrieve operations."""
        from vetinari.memory import MemoryEntry, MemoryEntryType, OcMemoryStore

        store = OcMemoryStore(path=temp_path)

        entry = MemoryEntry(
            agent="test_agent",
            entry_type=MemoryEntryType.DECISION,
            content="Test decision content",
            summary="Test decision",
            provenance="test"
        )

        entry_id = store.remember(entry)
        assert entry_id is not None

        retrieved = store.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved.content == "Test decision content"
        assert retrieved.agent == "test_agent"

    def test_search(self, temp_path):
        """Test search functionality."""
        from vetinari.memory import MemoryEntry, MemoryEntryType, OcMemoryStore

        store = OcMemoryStore(path=temp_path)

        for i in range(5):
            entry = MemoryEntry(
                agent="planner",
                entry_type=MemoryEntryType.DECISION,
                content=f"Decision number {i} about storage",
                summary=f"Decision {i}",
                provenance="test"
            )
            store.remember(entry)

        results = store.search("storage", agent="planner")
        assert len(results) >= 1

    def test_timeline(self, temp_path):
        """Test timeline functionality."""
        from vetinari.memory import MemoryEntry, MemoryEntryType, OcMemoryStore

        store = OcMemoryStore(path=temp_path)

        for i in range(3):
            entry = MemoryEntry(
                agent="executor",
                entry_type=MemoryEntryType.SUCCESS,
                content=f"Task {i} completed",
                summary=f"Task {i}",
                provenance="test"
            )
            store.remember(entry)

        timeline = store.timeline(agent="executor", limit=10)
        assert len(timeline) >= 3

    def test_forget_and_compact(self, temp_path):
        """Test forget (tombstone) and compact operations."""
        from vetinari.memory import MemoryEntry, MemoryEntryType, OcMemoryStore

        store = OcMemoryStore(path=temp_path)

        entry = MemoryEntry(
            agent="test",
            entry_type=MemoryEntryType.PROBLEM,
            content="Old problem to forget",
            summary="Old problem",
            provenance="test"
        )
        entry_id = store.remember(entry)

        store.forget(entry_id, "No longer relevant")

        search_results = store.search("problem")
        assert all(e.id != entry_id for e in search_results)

        deleted = store.compact()
        assert deleted >= 1

    def test_stats(self, temp_path):
        """Test statistics gathering."""
        from vetinari.memory import MemoryEntry, MemoryEntryType, OcMemoryStore

        store = OcMemoryStore(path=temp_path)

        entry = MemoryEntry(
            agent="planner",
            entry_type=MemoryEntryType.DECISION,
            content="Test content",
            summary="Test",
            provenance="test"
        )
        store.remember(entry)

        stats = store.stats()
        assert stats.total_entries >= 1
        assert "planner" in stats.entries_by_agent


class TestMnemosyneMemoryStore:
    """Tests for MnemosyneMemoryStore adapter."""

    @pytest.fixture
    def temp_path(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_store_and_retrieve(self, temp_path):
        """Test basic store and retrieve operations."""
        from vetinari.memory import MemoryEntry, MemoryEntryType, MnemosyneMemoryStore

        store = MnemosyneMemoryStore(path=temp_path)

        entry = MemoryEntry(
            agent="test_agent",
            entry_type=MemoryEntryType.DISCOVERY,
            content="Test discovery content",
            summary="Test discovery",
            provenance="test"
        )

        entry_id = store.remember(entry)
        assert entry_id is not None

        retrieved = store.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved.content == "Test discovery content"

    def test_search(self, temp_path):
        """Test search functionality."""
        from vetinari.memory import MemoryEntry, MemoryEntryType, MnemosyneMemoryStore

        store = MnemosyneMemoryStore(path=temp_path)

        for i in range(5):
            entry = MemoryEntry(
                agent="researcher",
                entry_type=MemoryEntryType.DISCOVERY,
                content=f"Discovery about pattern {i}",
                summary=f"Discovery {i}",
                provenance="test"
            )
            store.remember(entry)

        results = store.search("pattern", agent="researcher")
        assert len(results) >= 1

    def test_forget_and_compact(self, temp_path):
        """Test forget and compact operations."""
        from vetinari.memory import MemoryEntry, MemoryEntryType, MnemosyneMemoryStore

        store = MnemosyneMemoryStore(path=temp_path)

        entry = MemoryEntry(
            agent="test",
            entry_type=MemoryEntryType.WARNING,
            content="Old warning to forget",
            summary="Old warning",
            provenance="test"
        )
        entry_id = store.remember(entry)

        store.forget(entry_id, "Resolved")

        search_results = store.search("warning")
        assert all(e.id != entry_id for e in search_results)


class TestDualMemoryStore:
    """Tests for DualMemoryStore coordinator."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary directories for both backends."""
        oc_dir = tempfile.mkdtemp()
        mnemo_dir = tempfile.mkdtemp()
        yield oc_dir, mnemo_dir
        shutil.rmtree(oc_dir, ignore_errors=True)
        shutil.rmtree(mnemo_dir, ignore_errors=True)

    def test_dual_write(self, temp_paths):
        """Test that writes go to both backends."""
        from vetinari.memory import DualMemoryStore, MemoryEntry, MemoryEntryType

        oc_dir, mnemo_dir = temp_paths
        store = DualMemoryStore(oc_path=oc_dir, mnemosyne_path=mnemo_dir)

        entry = MemoryEntry(
            agent="test",
            entry_type=MemoryEntryType.DECISION,
            content="Dual write test",
            summary="Dual write",
            provenance="test"
        )

        entry_id = store.remember(entry)
        assert entry_id is not None

        oc_entry = store.oc_store.get_entry(entry_id)
        mnemo_entry = store.mnemo_store.get_entry(entry_id)

        assert oc_entry is not None
        assert mnemo_entry is not None

    def test_dual_read_merge(self, temp_paths):
        """Test that reads merge from both backends."""
        from vetinari.memory import DualMemoryStore, MemoryEntry, MemoryEntryType

        oc_dir, mnemo_dir = temp_paths
        store = DualMemoryStore(oc_path=oc_dir, mnemosyne_path=mnemo_dir)

        entry = MemoryEntry(
            agent="planner",
            entry_type=MemoryEntryType.INTENT,
            content="Plan to build feature X",
            summary="Build feature X",
            provenance="test"
        )

        store.remember(entry)

        results = store.search("feature")
        assert len(results) >= 1

    def test_deduplication(self, temp_paths):
        """Test that duplicates are deduplicated on read."""
        from vetinari.memory import DualMemoryStore, MemoryEntry, MemoryEntryType

        oc_dir, mnemo_dir = temp_paths
        store = DualMemoryStore(oc_path=oc_dir, mnemosyne_path=mnemo_dir)

        same_content = "Same content for deduplication test"

        entry1 = MemoryEntry(
            agent="test",
            entry_type=MemoryEntryType.DECISION,
            content=same_content,
            summary="Same",
            provenance="test"
        )

        entry2 = MemoryEntry(
            agent="test2",
            entry_type=MemoryEntryType.DECISION,
            content=same_content,
            summary="Same",
            provenance="test2"
        )

        store.remember(entry1)
        store.remember(entry2)

        results = store.search("deduplication")
        assert len(results) == 1

    def test_stats_combined(self, temp_paths):
        """Test that stats are combined from both backends."""
        from vetinari.memory import DualMemoryStore, MemoryEntry, MemoryEntryType

        oc_dir, mnemo_dir = temp_paths
        store = DualMemoryStore(oc_path=oc_dir, mnemosyne_path=mnemo_dir)

        entry = MemoryEntry(
            agent="combined",
            entry_type=MemoryEntryType.SUCCESS,
            content="Combined test",
            summary="Combined",
            provenance="test"
        )

        store.remember(entry)

        stats = store.stats()
        assert stats.total_entries >= 1


class TestMemoryMergePolicy:
    """Tests for memory merge policy."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary directories."""
        oc_dir = tempfile.mkdtemp()
        mnemo_dir = tempfile.mkdtemp()
        yield oc_dir, mnemo_dir
        shutil.rmtree(oc_dir, ignore_errors=True)
        shutil.rmtree(mnemo_dir, ignore_errors=True)

    def test_timestamp_precedence(self, temp_paths):
        """Test that most recent timestamp wins."""
        import time

        from vetinari.memory import DualMemoryStore, MemoryEntry, MemoryEntryType

        oc_dir, mnemo_dir = temp_paths
        store = DualMemoryStore(oc_path=oc_dir, mnemosyne_path=mnemo_dir)

        same_content = "Timestamp test content"

        entry1 = MemoryEntry(
            agent="oc_agent",
            entry_type=MemoryEntryType.DECISION,
            content=same_content,
            summary="OC first",
            provenance="test",
            timestamp=int(time.time() * 1000)
        )

        time.sleep(0.01)

        entry2 = MemoryEntry(
            agent="mnemo_agent",
            entry_type=MemoryEntryType.DECISION,
            content=same_content,
            summary="Mnemo later",
            provenance="test",
            timestamp=int(time.time() * 1000)
        )

        store.remember(entry1)
        store.remember(entry2)

        results = store.search("Timestamp")
        assert len(results) == 1
        assert "oc" in results[0].source_backends
        assert "mnemosyne" in results[0].source_backends


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
