"""Coverage tests for memory layer."""

from __future__ import annotations

from vetinari.memory.interfaces import (
    MemoryEntry,
    MemoryStats,
    MemoryType,
    content_hash,
)

# ─── MemoryEntry ──────────────────────────────────────────────────────────────


class TestMemoryEntryExtended:
    def test_default_id_format(self):
        e = MemoryEntry()
        assert e.id.startswith("mem_")

    def test_unique_ids(self):
        ids = {MemoryEntry().id for _ in range(50)}
        assert len(ids) == 50

    def test_to_dict_has_all_fields(self):
        e = MemoryEntry(agent="builder", content="test content", entry_type=MemoryType.DISCOVERY)
        d = e.to_dict()
        for k in ("id", "agent", "content", "entry_type", "timestamp"):
            assert k in d
        assert d["entry_type"] == "discovery"

    def test_from_dict_roundtrip(self):
        e = MemoryEntry(agent="explorer", content="roundtrip", summary="summary", entry_type=MemoryType.DECISION)
        d = e.to_dict()
        e2 = MemoryEntry.from_dict(d)
        assert e2.id == e.id
        assert e2.agent == "explorer"
        assert e2.entry_type == MemoryType.DECISION

    def test_metadata_field(self):
        e = MemoryEntry(metadata={"plan_id": "p1", "risk": 0.3})
        assert e.metadata["plan_id"] == "p1"

    def test_metadata_defaults_none(self):
        e = MemoryEntry()
        assert e.metadata is None

    def test_config_entry_type(self):
        e = MemoryEntry(entry_type=MemoryType.CONFIG)
        assert e.entry_type == MemoryType.CONFIG


class TestMemoryStats:
    def test_defaults(self):
        s = MemoryStats()
        assert s.total_entries == 0
        assert s.file_size_bytes == 0

    def test_to_dict(self):
        s = MemoryStats(total_entries=5, file_size_bytes=1024)
        d = s.to_dict()
        assert d["total_entries"] == 5


class TestContentHash:
    def test_same_content_same_hash(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_content_different_hash(self):
        assert content_hash("hello") != content_hash("world")

    def test_empty_string(self):
        h = content_hash("")
        assert isinstance(h, str)
        assert len(h) > 0


# ─── MemoryStorage telemetry helpers ──────────────────────────────────────────


class TestMemoryStorageTelemetryHelpers:
    """Verify the private telemetry helpers do not raise even when telemetry is absent."""

    def test_record_dedup_hit_ignores_import_error(self):
        """_record_dedup_hit swallows exceptions when telemetry is unavailable."""
        from unittest.mock import patch

        from vetinari.memory.memory_storage import _record_dedup_hit

        with patch("vetinari.telemetry.get_telemetry_collector", side_effect=RuntimeError("unavailable")):
            assert _record_dedup_hit("test_backend") is None

    def test_record_dedup_miss_ignores_import_error(self):
        """_record_dedup_miss swallows exceptions when telemetry is unavailable."""
        from unittest.mock import patch

        from vetinari.memory.memory_storage import _record_dedup_miss

        with patch("vetinari.telemetry.get_telemetry_collector", side_effect=RuntimeError("unavailable")):
            assert _record_dedup_miss("test_backend") is None

    def test_record_sync_failure_ignores_import_error(self):
        """_record_sync_failure swallows exceptions when telemetry is unavailable."""
        from unittest.mock import patch

        from vetinari.memory.memory_storage import _record_sync_failure

        with patch("vetinari.telemetry.get_telemetry_collector", side_effect=RuntimeError("unavailable")):
            assert _record_sync_failure("test_backend") is None

    def test_record_dedup_hit_calls_collector(self):
        """_record_dedup_hit invokes TelemetryCollector.record_dedup_hit."""
        from unittest.mock import MagicMock, patch

        from vetinari.memory.memory_storage import _record_dedup_hit

        mock_collector = MagicMock()
        with patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_collector):
            with patch(
                "vetinari.memory.memory_storage.get_telemetry_collector", return_value=mock_collector, create=True
            ):
                _record_dedup_hit("my_agent")
        mock_collector.record_dedup_hit.assert_called_once_with("my_agent")

    def test_record_dedup_miss_calls_collector(self):
        """_record_dedup_miss invokes TelemetryCollector.record_dedup_miss."""
        from unittest.mock import MagicMock, patch

        from vetinari.memory.memory_storage import _record_dedup_miss

        mock_collector = MagicMock()
        # The lazy import inside the helper resolves vetinari.telemetry.get_telemetry_collector
        with patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_collector):
            _record_dedup_miss("my_agent")
        mock_collector.record_dedup_miss.assert_called_once_with("my_agent")

    def test_record_sync_failure_calls_collector(self):
        """_record_sync_failure invokes TelemetryCollector.record_sync_failure."""
        from unittest.mock import MagicMock, patch

        from vetinari.memory.memory_storage import _record_sync_failure

        mock_collector = MagicMock()
        with patch("vetinari.telemetry.get_telemetry_collector", return_value=mock_collector):
            _record_sync_failure("my_agent")
        mock_collector.record_sync_failure.assert_called_once_with("my_agent")
