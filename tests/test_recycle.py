"""Tests for RecycleStore — short-grace-window recycle bin facade.

Covers: retire+restore round trip; purge_expired deletes past-grace records;
purge_expired skips within-grace records; concurrent retire safety (one wins,
one raises); purge_expired is the only hard-delete path (negative test).
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vetinari.lifecycle.store import _MANIFEST_FILENAME
from vetinari.safety.recycle import RecycleStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> RecycleStore:
    """RecycleStore backed by a temp directory with a 72h grace window."""
    return RecycleStore(root=tmp_path / "recycle", grace_hours=72)


@pytest.fixture
def short_grace_store(tmp_path: Path) -> RecycleStore:
    """RecycleStore with a 1-hour grace window for expiry testing."""
    return RecycleStore(root=tmp_path / "recycle_short", grace_hours=1)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A small file to retire."""
    f = tmp_path / "entity.txt"
    f.write_text("recycle me", encoding="utf-8")
    return f


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """A small directory to retire."""
    d = tmp_path / "entity_dir"
    d.mkdir()
    (d / "output.txt").write_text("done", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# retire() + restore() round trip
# ---------------------------------------------------------------------------


class TestRetireRestoreRoundTrip:
    """Basic retire and restore cycle."""

    def test_retire_moves_file_out_of_original_path(self, store: RecycleStore, sample_file: Path) -> None:
        """retire() removes the file from its original location."""
        original = sample_file
        store.retire(sample_file, reason="test retire")
        assert not original.exists()

    def test_restore_returns_file_to_original_path(self, store: RecycleStore, sample_file: Path) -> None:
        """restore() puts the file back with original content."""
        original = sample_file
        content = original.read_text(encoding="utf-8")
        record = store.retire(sample_file, reason="restore round trip")
        store.restore(record.record_id)
        assert original.exists()
        assert original.read_text(encoding="utf-8") == content

    def test_retire_directory_round_trip(self, store: RecycleStore, sample_dir: Path) -> None:
        """Directories can be retired and restored."""
        original = sample_dir
        record = store.retire(sample_dir, reason="dir round trip")
        assert not original.exists()
        store.restore(record.record_id)
        assert original.is_dir()
        assert (original / "output.txt").exists()

    def test_retire_nonexistent_raises(self, store: RecycleStore, tmp_path: Path) -> None:
        """Retiring a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            store.retire(tmp_path / "missing.txt", reason="should fail")

    def test_list_all_returns_retired_record(self, store: RecycleStore, sample_file: Path) -> None:
        """list_all() includes the newly retired record."""
        record = store.retire(sample_file, reason="list check")
        records = store.list_all()
        assert any(r.record_id == record.record_id for r in records)


# ---------------------------------------------------------------------------
# purge_expired() — only hard-delete path
# ---------------------------------------------------------------------------


class TestPurgeExpired:
    """purge_expired() hard-deletes past-grace records; skips within-grace."""

    def test_purge_expired_deletes_past_grace_record(self, store: RecycleStore, sample_file: Path) -> None:
        """A record whose retired_at is >72h ago is deleted by purge_expired."""
        import json

        record = store.retire(sample_file, reason="past grace")

        # Back-date the manifest to simulate expiry.
        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["retired_at_utc"] = (datetime.now(timezone.utc) - timedelta(hours=73)).isoformat()
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        purged = store.purge_expired()
        assert len(purged) == 1
        assert purged[0].record_id == record.record_id
        assert not record.store_path.exists()

    def test_purge_expired_skips_within_grace_record(self, store: RecycleStore, sample_file: Path) -> None:
        """A freshly retired record (within grace) is NOT purged."""
        record = store.retire(sample_file, reason="within grace")
        purged = store.purge_expired()
        assert purged == []
        assert record.store_path.exists()

    def test_purge_expired_custom_grace_override(self, store: RecycleStore, sample_file: Path) -> None:
        """purge_expired(grace_hours=1) expires a record retired 2h ago."""
        import json

        record = store.retire(sample_file, reason="custom grace")

        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["retired_at_utc"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        purged = store.purge_expired(grace_hours=1)
        assert len(purged) == 1
        assert purged[0].record_id == record.record_id

    def test_purge_expired_returns_empty_on_empty_store(self, store: RecycleStore) -> None:
        """purge_expired() returns empty list when store is empty."""
        assert store.purge_expired() == []


# ---------------------------------------------------------------------------
# Concurrent retire safety
# ---------------------------------------------------------------------------


class TestConcurrentRetire:
    """Two threads retiring the same path: one wins, one raises."""

    def test_concurrent_retire_one_wins_one_raises(self, store: RecycleStore, sample_file: Path) -> None:
        """Concurrent retire is safe: exactly one succeeds."""
        results: list[str] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def do_retire() -> None:
            barrier.wait()
            try:
                record = store.retire(sample_file, reason="concurrent")
                results.append(record.record_id)
            except (FileNotFoundError, OSError) as exc:
                errors.append(exc)

        t1 = threading.Thread(target=do_retire)
        t2 = threading.Thread(target=do_retire)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 1, f"expected exactly 1 winner, got {results}"
        assert len(errors) == 1, f"expected exactly 1 failure, got {errors}"
        assert not sample_file.exists()


# ---------------------------------------------------------------------------
# purge_expired is the only hard-delete path (negative tests)
# ---------------------------------------------------------------------------


class TestOnlyHardDeletePath:
    """No RecycleStore method other than purge_expired permanently deletes bytes."""

    def test_retire_does_not_delete_bytes(self, store: RecycleStore, sample_file: Path) -> None:
        """retire() moves the payload into the store — bytes are not lost."""
        record = store.retire(sample_file, reason="bytes preserved")
        # Payload directory must be non-empty
        payload_dir = record.store_path / "payload"
        assert any(payload_dir.iterdir()), "bytes must exist in store after retire"

    def test_restore_does_not_delete_from_store(self, store: RecycleStore, sample_file: Path) -> None:
        """restore() moves bytes back — the record directory may persist."""
        record = store.retire(sample_file, reason="restore safe")
        store.restore(record.record_id)
        # Original path is back; record dir may still exist with empty payload
        assert sample_file.exists()

    def test_recycle_store_has_no_public_delete_method(self, store: RecycleStore) -> None:
        """RecycleStore public API exposes no 'delete' method."""
        public_methods = [name for name in dir(store) if not name.startswith("_") and callable(getattr(store, name))]
        assert "delete" not in public_methods
        # purge_expired is the only purge-flavoured public method
        purge_methods = [m for m in public_methods if "purge" in m]
        assert purge_methods == ["purge_expired"], f"Only purge_expired should be public, got: {purge_methods}"
