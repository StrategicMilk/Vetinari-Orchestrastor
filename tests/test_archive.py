"""Tests for ArchiveStore — view-tiered archive facade.

Covers: archive+unarchive round trip; list_by_tier returns correct tier;
sweep archives only past-cooldown candidates; no direct hard-delete path
exposed; search matches by path/reason/receipt.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vetinari.lifecycle.archive import ArchiveCandidate, ArchiveStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ArchiveStore:
    """ArchiveStore backed by a temp directory."""
    return ArchiveStore(root=tmp_path / "archive", recent_days=7, cooling_days=30)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """A small file to archive."""
    f = tmp_path / "entity.txt"
    f.write_text("archive me", encoding="utf-8")
    return f


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """A small directory to archive."""
    d = tmp_path / "entity_dir"
    d.mkdir()
    (d / "output.txt").write_text("done", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# archive() + unarchive() round trip
# ---------------------------------------------------------------------------


class TestArchiveUnarchive:
    """Basic archive and unarchive cycle."""

    def test_archive_moves_file_out_of_original_path(self, store: ArchiveStore, sample_file: Path) -> None:
        """archive() removes the file from its original location."""
        original = sample_file
        store.archive(sample_file, reason="test archive")
        assert not original.exists()

    def test_unarchive_returns_file_to_original_path(self, store: ArchiveStore, sample_file: Path) -> None:
        """unarchive() restores the file with original content."""
        original = sample_file
        content = original.read_text(encoding="utf-8")
        record = store.archive(sample_file, reason="unarchive round trip")
        store.unarchive(record.record_id)
        assert original.exists()
        assert original.read_text(encoding="utf-8") == content

    def test_archive_directory_round_trip(self, store: ArchiveStore, sample_dir: Path) -> None:
        """Directories can be archived and unarchived."""
        original = sample_dir
        record = store.archive(sample_dir, reason="dir round trip")
        assert not original.exists()
        store.unarchive(record.record_id)
        assert original.is_dir()
        assert (original / "output.txt").exists()

    def test_archive_nonexistent_raises(self, store: ArchiveStore, tmp_path: Path) -> None:
        """Archiving a non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            store.archive(tmp_path / "missing.txt", reason="should fail")

    def test_archive_records_work_receipt_id(self, store: ArchiveStore, sample_file: Path) -> None:
        """work_receipt_id is stored in the manifest."""
        record = store.archive(sample_file, reason="receipt test", work_receipt_id="receipt-abc-123")
        assert record.work_receipt_id == "receipt-abc-123"


# ---------------------------------------------------------------------------
# list_by_tier — view tier classification
# ---------------------------------------------------------------------------


class TestListByTier:
    """list_by_tier classifies records by age."""

    def _backdate_manifest(self, record, days: int) -> None:
        """Backdate retired_at_utc in a manifest by the given number of days."""
        import json

        from vetinari.lifecycle.store import _MANIFEST_FILENAME

        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["retired_at_utc"] = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def test_fresh_record_in_recent_tier(self, store: ArchiveStore, sample_file: Path) -> None:
        """A just-archived record appears in the 'recent' tier."""
        record = store.archive(sample_file, reason="recent test")
        recent = store.list_by_tier("recent")
        assert any(r.record_id == record.record_id for r in recent)

    def test_eight_day_old_record_in_cooling_tier(self, store: ArchiveStore, sample_file: Path, tmp_path: Path) -> None:
        """A record 8 days old is in the 'cooling' tier (not 'recent')."""
        record = store.archive(sample_file, reason="cooling test")
        self._backdate_manifest(record, days=8)

        recent = store.list_by_tier("recent")
        cooling = store.list_by_tier("cooling")

        assert not any(r.record_id == record.record_id for r in recent)
        assert any(r.record_id == record.record_id for r in cooling)

    def test_thirty_one_day_old_record_in_cold_tier(
        self, store: ArchiveStore, sample_file: Path, tmp_path: Path
    ) -> None:
        """A record 31 days old is in the 'cold' tier."""
        record = store.archive(sample_file, reason="cold test")
        self._backdate_manifest(record, days=31)

        cooling = store.list_by_tier("cooling")
        cold = store.list_by_tier("cold")

        assert not any(r.record_id == record.record_id for r in cooling)
        assert any(r.record_id == record.record_id for r in cold)

    def test_empty_tier_returns_empty_list(self, store: ArchiveStore) -> None:
        """list_by_tier returns an empty list when no records match."""
        assert store.list_by_tier("recent") == []
        assert store.list_by_tier("cooling") == []
        assert store.list_by_tier("cold") == []


# ---------------------------------------------------------------------------
# sweep() — archives past-cooldown candidates
# ---------------------------------------------------------------------------


class TestSweep:
    """sweep() archives past-cooldown candidates; skips within-cooldown or absent."""

    def test_sweep_archives_past_cooldown_candidate(self, store: ArchiveStore, sample_file: Path) -> None:
        """Candidate with elapsed cooldown is archived by sweep."""
        completed_at = datetime.now(timezone.utc) - timedelta(hours=25)
        candidate = ArchiveCandidate(
            path=sample_file,
            completed_at=completed_at,
            cooldown_hours=24,
            reason="past cooldown",
        )
        archived = store.sweep([candidate])
        assert len(archived) == 1
        assert not sample_file.exists()

    def test_sweep_skips_within_cooldown_candidate(self, store: ArchiveStore, sample_file: Path) -> None:
        """Candidate within cooldown is not archived."""
        completed_at = datetime.now(timezone.utc) - timedelta(hours=1)
        candidate = ArchiveCandidate(
            path=sample_file,
            completed_at=completed_at,
            cooldown_hours=24,
            reason="still cooling",
        )
        archived = store.sweep([candidate])
        assert archived == []
        assert sample_file.exists()

    def test_sweep_skips_absent_path(self, store: ArchiveStore, tmp_path: Path) -> None:
        """Candidate whose path doesn't exist is silently skipped."""
        completed_at = datetime.now(timezone.utc) - timedelta(hours=48)
        candidate = ArchiveCandidate(
            path=tmp_path / "nonexistent.txt",
            completed_at=completed_at,
            cooldown_hours=1,
            reason="already gone",
        )
        archived = store.sweep([candidate])
        assert archived == []

    def test_sweep_is_idempotent(self, store: ArchiveStore, sample_file: Path) -> None:
        """Running sweep twice on same candidates is safe (second run skips absent paths)."""
        completed_at = datetime.now(timezone.utc) - timedelta(hours=25)
        candidate = ArchiveCandidate(
            path=sample_file,
            completed_at=completed_at,
            cooldown_hours=24,
            reason="idempotent test",
        )
        archived_first = store.sweep([candidate])
        archived_second = store.sweep([candidate])
        assert len(archived_first) == 1
        assert archived_second == []

    def test_sweep_accepts_naive_datetime(self, store: ArchiveStore, sample_file: Path) -> None:
        """sweep() treats naive datetimes as UTC (does not raise)."""
        completed_at = datetime.utcnow() - timedelta(hours=25)  # naive
        candidate = ArchiveCandidate(
            path=sample_file,
            completed_at=completed_at,
            cooldown_hours=24,
            reason="naive datetime",
        )
        archived = store.sweep([candidate])
        assert len(archived) == 1


# ---------------------------------------------------------------------------
# No direct hard-delete path
# ---------------------------------------------------------------------------


class TestNoDirectHardDelete:
    """ArchiveStore exposes no public delete or purge method."""

    def test_archive_store_has_no_public_delete_method(self, store: ArchiveStore) -> None:
        """ArchiveStore public API exposes no 'delete' or 'purge' method."""
        public_methods = [name for name in dir(store) if not name.startswith("_") and callable(getattr(store, name))]
        assert "delete" not in public_methods, "delete must not be public on ArchiveStore"
        purge_methods = [m for m in public_methods if "purge" in m]
        assert purge_methods == [], f"No purge methods should be public on ArchiveStore; got: {purge_methods}"

    def test_purge_archive_record_is_private(self, store: ArchiveStore) -> None:
        """_purge_archive_record exists but is private (underscore-prefixed)."""
        assert hasattr(store, "_purge_archive_record"), "_purge_archive_record must exist on ArchiveStore"
        assert not hasattr(store, "purge_archive_record"), "purge_archive_record must NOT be public"

    def test_purge_archive_record_raises_permission_error(self, store: ArchiveStore, sample_file: Path) -> None:
        """Calling _purge_archive_record raises PermissionError (ArchivePolicy.allows_hard_delete=False)."""
        record = store.archive(sample_file, reason="test purge blocked")
        with pytest.raises(PermissionError):
            store._purge_archive_record(record.record_id)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestSearch:
    """search() finds records by path/reason/receipt substring."""

    def test_search_finds_by_reason_substring(self, store: ArchiveStore, sample_file: Path) -> None:
        """search() returns records matching a reason substring."""
        store.archive(sample_file, reason="quarterly cleanup run")
        results = store.search("quarterly")
        assert len(results) == 1

    def test_search_finds_by_work_receipt_id(self, store: ArchiveStore, sample_file: Path) -> None:
        """search() returns records matching a work_receipt_id substring."""
        store.archive(sample_file, reason="receipt search test", work_receipt_id="receipt-xyz-789")
        results = store.search("xyz-789")
        assert len(results) == 1

    def test_search_is_case_insensitive(self, store: ArchiveStore, sample_file: Path) -> None:
        """search() is case-insensitive."""
        store.archive(sample_file, reason="UPPERCASE REASON")
        results = store.search("uppercase")
        assert len(results) == 1

    def test_search_returns_empty_on_no_match(self, store: ArchiveStore, sample_file: Path) -> None:
        """search() returns empty list when no records match."""
        store.archive(sample_file, reason="something else")
        results = store.search("zzznomatch")
        assert results == []
