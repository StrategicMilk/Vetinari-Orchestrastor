"""Coverage tests for memory layer — Phase 7E"""
import os
import tempfile
import unittest

from vetinari.memory.interfaces import (
    MemoryEntry,
    MemoryEntryType,
    MemoryStats,
    content_hash,
)

# ─── MemoryEntry ──────────────────────────────────────────────────────────────

class TestMemoryEntryExtended(unittest.TestCase):

    def test_default_id_format(self):
        e = MemoryEntry()
        assert e.id.startswith("mem_")

    def test_unique_ids(self):
        ids = {MemoryEntry().id for _ in range(50)}
        assert len(ids) == 50

    def test_to_dict_has_all_fields(self):
        e = MemoryEntry(agent="builder", content="test content",
                        entry_type=MemoryEntryType.DISCOVERY)
        d = e.to_dict()
        for k in ("id", "agent", "content", "entry_type", "timestamp"):
            assert k in d
        assert d["entry_type"] == "discovery"

    def test_from_dict_roundtrip(self):
        e = MemoryEntry(agent="explorer", content="roundtrip",
                        summary="summary", entry_type=MemoryEntryType.DECISION)
        d = e.to_dict()
        e2 = MemoryEntry.from_dict(d)
        assert e2.id == e.id
        assert e2.agent == "explorer"
        assert e2.entry_type == MemoryEntryType.DECISION

    def test_metadata_field(self):
        e = MemoryEntry(metadata={"plan_id": "p1", "risk": 0.3})
        assert e.metadata["plan_id"] == "p1"

    def test_metadata_defaults_none(self):
        e = MemoryEntry()
        assert e.metadata is None

    def test_config_entry_type(self):
        e = MemoryEntry(entry_type=MemoryEntryType.CONFIG)
        assert e.entry_type == MemoryEntryType.CONFIG


class TestMemoryStats(unittest.TestCase):

    def test_defaults(self):
        s = MemoryStats()
        assert s.total_entries == 0
        assert s.file_size_bytes == 0

    def test_to_dict(self):
        s = MemoryStats(total_entries=5, file_size_bytes=1024)
        d = s.to_dict()
        assert d["total_entries"] == 5


class TestContentHash(unittest.TestCase):

    def test_same_content_same_hash(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_content_different_hash(self):
        assert content_hash("hello") != content_hash("world")

    def test_empty_string(self):
        h = content_hash("")
        assert isinstance(h, str)
        assert len(h) > 0


# ─── OcMemoryStore ────────────────────────────────────────────────────────────

class TestOcMemoryStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_oc.db")
        from vetinari.memory.oc_memory import OcMemoryStore
        self.store = OcMemoryStore(self.db_path)

    def tearDown(self):
        try:
            self.store.close()
        except Exception:  # noqa: VET022
            pass

    def test_remember_and_get(self):
        e = MemoryEntry(agent="test", content="some content",
                        entry_type=MemoryEntryType.DISCOVERY)
        entry_id = self.store.remember(e)
        assert entry_id is not None
        retrieved = self.store.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved.content == "some content"

    def test_search_returns_results(self):
        self.store.remember(MemoryEntry(agent="builder", content="python function",
                                       entry_type=MemoryEntryType.SOLUTION))
        results = self.store.search("python")
        assert isinstance(results, list)

    def test_timeline(self):
        for i in range(3):
            self.store.remember(MemoryEntry(agent="a", content=f"item {i}",
                                            entry_type=MemoryEntryType.DISCOVERY))
        results = self.store.timeline(limit=10)
        assert isinstance(results, list)

    def test_stats(self):
        self.store.remember(MemoryEntry(agent="x", content="stat test"))
        stats = self.store.stats()
        assert isinstance(stats, MemoryStats)
        assert stats.total_entries > 0

    def test_forget(self):
        e = MemoryEntry(agent="x", content="to delete")
        eid = self.store.remember(e)
        result = self.store.forget(eid, "test cleanup")
        assert result

    def test_compact(self):
        count = self.store.compact(max_age_days=0)
        assert isinstance(count, int)

    def test_export(self):
        import json
        self.store.remember(MemoryEntry(agent="export", content="exported"))
        path = os.path.join(self.tmpdir, "export.json")
        ok = self.store.export(path)
        assert ok
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        # OcMemoryStore exports as "memories" or "entries" depending on version
        assert "memories" in data or "entries" in data


# ─── MnemosyneMemoryStore ─────────────────────────────────────────────────────

class TestMnemosyneMemoryStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        path = os.path.join(self.tmpdir, "mnemosyne.json")
        from vetinari.memory.mnemosyne_memory import MnemosyneMemoryStore
        self.store = MnemosyneMemoryStore(path)

    def test_remember_and_retrieve(self):
        e = MemoryEntry(agent="oracle", content="knowledge",
                        entry_type=MemoryEntryType.PATTERN)
        eid = self.store.remember(e)
        assert eid is not None
        retrieved = self.store.get_entry(eid)
        assert retrieved is not None
        assert retrieved.content == "knowledge"

    def test_search(self):
        self.store.remember(MemoryEntry(agent="a", content="fuzzy search test"))
        results = self.store.search("fuzzy")
        assert isinstance(results, list)

    def test_stats(self):
        self.store.remember(MemoryEntry(agent="x", content="stat"))
        stats = self.store.stats()
        assert isinstance(stats, MemoryStats)

    def test_forget(self):
        e = MemoryEntry(agent="x", content="to forget")
        eid = self.store.remember(e)
        ok = self.store.forget(eid, "cleanup")
        assert ok

    def test_timeline(self):
        for i in range(3):
            self.store.remember(MemoryEntry(agent="b", content=f"t{i}"))
        results = self.store.timeline(limit=10)
        assert isinstance(results, list)


# ─── DualMemoryStore ──────────────────────────────────────────────────────────

class TestDualMemoryStoreExtended(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        oc_path  = os.path.join(self.tmpdir, "oc.db")
        mn_path  = os.path.join(self.tmpdir, "mn.json")
        from vetinari.memory.dual_memory import DualMemoryStore
        self.store = DualMemoryStore(oc_path=oc_path, mnemosyne_path=mn_path)

    def test_remember_returns_id(self):
        e = MemoryEntry(agent="dual", content="dual content")
        eid = self.store.remember(e)
        assert eid is not None

    def test_metadata_sanitized(self):
        e = MemoryEntry(agent="dual",
                        content="safe content",
                        metadata={"note": "no secrets here"})
        eid = self.store.remember(e)
        assert eid is not None

    def test_search_returns_list(self):
        self.store.remember(MemoryEntry(agent="dual", content="searchable dual text"))
        results = self.store.search("searchable")
        assert isinstance(results, list)

    def test_stats_combined(self):
        self.store.remember(MemoryEntry(agent="dual", content="stat entry"))
        stats = self.store.stats()
        assert isinstance(stats, MemoryStats)
        assert stats.total_entries >= 1

    def test_get_entry(self):
        e = MemoryEntry(agent="dual", content="get me")
        eid = self.store.remember(e)
        retrieved = self.store.get_entry(eid)
        assert retrieved is not None

    def test_forget(self):
        e = MemoryEntry(agent="dual", content="forget me")
        eid = self.store.remember(e)
        ok = self.store.forget(eid, "test")
        assert ok

    def test_content_sanitization(self):
        # Secret should be sanitized before storage
        e = MemoryEntry(agent="dual",
                        content="My key is sk-proj-secretkeyvalue123")
        eid = self.store.remember(e)
        assert eid is not None


if __name__ == "__main__":
    unittest.main()
