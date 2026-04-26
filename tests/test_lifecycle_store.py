"""Tests for the shared LifecycleStore primitive.

Covers: retire+restore round trip across both policies, manifest schema
validation, concurrent retire safety (one wins, one raises), atomic rollback
on move failure, and layout matching the policy root.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from vetinari.lifecycle.policies import ArchivePolicy, PolicyFilter, RecyclePolicy
from vetinari.lifecycle.store import _MANIFEST_FILENAME, LifecycleRecord, LifecycleStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recycle_store(tmp_path: Path) -> LifecycleStore:
    """LifecycleStore configured with RecyclePolicy."""
    return LifecycleStore(root=tmp_path / "recycle", policy=RecyclePolicy())


@pytest.fixture
def archive_store(tmp_path: Path) -> LifecycleStore:
    """LifecycleStore configured with ArchivePolicy."""
    return LifecycleStore(root=tmp_path / "archive", policy=ArchivePolicy())


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A small file to retire."""
    f = tmp_path / "entity.txt"
    f.write_text("lifecycle test", encoding="utf-8")
    return f


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """A small directory to retire."""
    d = tmp_path / "entity_dir"
    d.mkdir()
    (d / "output.txt").write_text("done", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLifecycleStoreLayout:
    """Store layout matches <root>/<policy>/<yyyy-mm-dd>/<uuid>/."""

    def test_retire_creates_policy_subdirectory(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """Retired records land inside <root>/recycle/<date>/<uuid>/."""
        record = recycle_store.retire(sample_file, reason="layout test")
        assert record.store_path.is_dir()
        # policy name in path
        assert "recycle" in str(record.store_path)

    def test_retire_creates_manifest(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """manifest.json is written inside the record directory."""
        record = recycle_store.retire(sample_file, reason="manifest test")
        manifest = record.store_path / _MANIFEST_FILENAME
        assert manifest.exists()

    def test_retire_creates_payload_directory(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """payload/ directory exists inside the record directory."""
        record = recycle_store.retire(sample_file, reason="payload dir test")
        payload_dir = record.store_path / "payload"
        assert payload_dir.is_dir()
        # The original file should be inside payload/
        assert any(payload_dir.iterdir())

    def test_archive_policy_uses_archive_subdirectory(self, archive_store: LifecycleStore, sample_file: Path) -> None:
        """Archive policy places records in <root>/archive/..."""
        record = archive_store.retire(sample_file, reason="archive layout")
        assert "archive" in str(record.store_path)


# ---------------------------------------------------------------------------
# Retire + restore round trip
# ---------------------------------------------------------------------------


class TestRetireRestoreRoundTrip:
    """retire() moves file; restore() brings it back with original content."""

    def test_retire_moves_file(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """File is gone from original path after retire."""
        original = sample_file
        recycle_store.retire(sample_file, reason="move test")
        assert not original.exists()

    def test_restore_returns_file_to_original_path(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """restore() puts the file back at original_path."""
        original = sample_file
        content = original.read_text(encoding="utf-8")
        record = recycle_store.retire(sample_file, reason="restore test")
        recycle_store.restore(record.record_id)
        assert original.exists()
        assert original.read_text(encoding="utf-8") == content

    def test_retire_directory_round_trip(self, recycle_store: LifecycleStore, sample_dir: Path) -> None:
        """Directories are retired and restored correctly."""
        original = sample_dir
        record = recycle_store.retire(sample_dir, reason="dir round trip")
        assert not original.exists()
        recycle_store.restore(record.record_id)
        assert original.is_dir()
        assert (original / "output.txt").exists()

    def test_restore_updates_manifest_with_restored_at(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """restored_at_utc is written into the manifest after restore."""
        record = recycle_store.retire(sample_file, reason="manifest update")
        recycle_store.restore(record.record_id)
        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["restored_at_utc"] is not None

    def test_retire_nonexistent_raises(self, recycle_store: LifecycleStore, tmp_path: Path) -> None:
        """Retiring a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            recycle_store.retire(tmp_path / "ghost.txt", reason="should fail")

    def test_work_receipt_id_stored_in_manifest(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """work_receipt_id is preserved in the manifest."""
        record = recycle_store.retire(sample_file, reason="receipt id test", work_receipt_id="wr-abc-123")
        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["work_receipt_id"] == "wr-abc-123"


# ---------------------------------------------------------------------------
# Manifest schema validation
# ---------------------------------------------------------------------------


class TestManifestSchema:
    """Manifest JSON contains all required fields with correct types."""

    def test_manifest_has_all_required_keys(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """Manifest contains all 8 required schema keys."""
        record = recycle_store.retire(sample_file, reason="schema test")
        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        required_keys = {
            "record_id",
            "original_path",
            "sha256",
            "retired_at_utc",
            "reason",
            "work_receipt_id",
            "policy",
            "restored_at_utc",
        }
        assert required_keys.issubset(data.keys())

    def test_manifest_policy_name_matches(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """Manifest policy field is 'recycle'."""
        record = recycle_store.retire(sample_file, reason="policy field")
        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["policy"] == "recycle"

    def test_archive_manifest_policy_name(self, archive_store: LifecycleStore, sample_file: Path) -> None:
        """Archive manifest policy field is 'archive'."""
        record = archive_store.retire(sample_file, reason="archive policy field")
        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["policy"] == "archive"

    def test_sha256_is_hex_string_for_file(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """sha256 is a 64-character hex string for a file."""
        record = recycle_store.retire(sample_file, reason="sha256 test")
        assert len(record.sha256) == 64
        assert all(c in "0123456789abcdef" for c in record.sha256)

    def test_sha256_is_empty_for_directory(self, recycle_store: LifecycleStore, sample_dir: Path) -> None:
        """sha256 is empty string for a directory."""
        record = recycle_store.retire(sample_dir, reason="dir sha256")
        assert record.sha256 == ""


# ---------------------------------------------------------------------------
# Atomic rollback on move failure
# ---------------------------------------------------------------------------


class TestAtomicRollback:
    """On move failure, the original path is left untouched."""

    def test_rollback_on_move_failure_leaves_original_intact(
        self, recycle_store: LifecycleStore, sample_file: Path
    ) -> None:
        """When shutil.move raises, original file still exists."""
        with patch("vetinari.lifecycle.store.shutil.move", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="original is untouched"):
                recycle_store.retire(sample_file, reason="rollback test")
        assert sample_file.exists(), "original must be intact after move failure"

    def test_rollback_leaves_no_dest_dir(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """Dest directory is cleaned up after move failure."""
        policy_root = recycle_store._root / recycle_store._policy.name
        with patch("vetinari.lifecycle.store.shutil.move", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="original is untouched"):
                recycle_store.retire(sample_file, reason="no dest dir")
        # No record directories should exist (root may not even be created)
        if policy_root.exists():
            for date_dir in policy_root.iterdir():
                assert not list(date_dir.iterdir()), "no partial record dirs should remain"


# ---------------------------------------------------------------------------
# Concurrent retire
# ---------------------------------------------------------------------------


class TestConcurrentRetire:
    """Concurrent retire on the same path: one wins, one raises."""

    def test_concurrent_retire_one_wins_one_raises(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """Two threads retiring the same path: exactly one succeeds."""
        results: list[str] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def do_retire() -> None:
            barrier.wait()
            try:
                record = recycle_store.retire(sample_file, reason="concurrent test")
                results.append(record.record_id)
            except (FileNotFoundError, OSError) as exc:
                errors.append(exc)

        t1 = threading.Thread(target=do_retire)
        t2 = threading.Thread(target=do_retire)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one should succeed and one should fail
        assert len(results) + len(errors) == 2
        assert len(results) == 1, f"expected 1 winner, got {results}"
        assert len(errors) == 1, f"expected 1 failure, got {errors}"
        # Original file should be gone
        assert not sample_file.exists()


# ---------------------------------------------------------------------------
# list() and PolicyFilter
# ---------------------------------------------------------------------------


class TestListAndPolicyFilter:
    """list() returns records; PolicyFilter narrows by bucket."""

    def test_list_returns_retired_record(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """list() returns the newly retired record."""
        record = recycle_store.retire(sample_file, reason="list test")
        records = recycle_store.list()
        assert any(r.record_id == record.record_id for r in records)

    def test_list_empty_when_no_records(self, recycle_store: LifecycleStore) -> None:
        """list() returns empty list when store is empty."""
        assert recycle_store.list() == []

    def test_filter_by_reason_contains(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """PolicyFilter reason_contains narrows list to matching records."""
        record = recycle_store.retire(sample_file, reason="unique_reason_xyz")
        filt = PolicyFilter(reason_contains="unique_reason_xyz")
        records = recycle_store.list(filter=filt)
        assert any(r.record_id == record.record_id for r in records)


# ---------------------------------------------------------------------------
# purge()
# ---------------------------------------------------------------------------


class TestPurge:
    """purge() hard-deletes recycle records; raises for archive policy."""

    def test_purge_recycle_record_removes_store_path(self, recycle_store: LifecycleStore, sample_file: Path) -> None:
        """purge() removes the record directory from disk."""
        record = recycle_store.retire(sample_file, reason="purge test")
        store_path = record.store_path
        recycle_store.purge(record.record_id)
        assert not store_path.exists()

    def test_purge_archive_record_raises_permission_error(
        self, archive_store: LifecycleStore, sample_file: Path
    ) -> None:
        """purge() raises PermissionError for archive policy (no hard delete)."""
        record = archive_store.retire(sample_file, reason="archive no purge")
        with pytest.raises(PermissionError, match="does not permit hard delete"):
            archive_store.purge(record.record_id)

    def test_purge_unknown_record_raises_key_error(self, recycle_store: LifecycleStore) -> None:
        """purge() raises KeyError for an unknown record_id."""
        with pytest.raises(KeyError):
            recycle_store.purge("nonexistent-record-id")
