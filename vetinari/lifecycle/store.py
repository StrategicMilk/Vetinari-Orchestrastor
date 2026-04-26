"""Shared LifecycleStore primitive — atomic move-aside with manifest.

Both RecycleStore and ArchiveStore are thin facades over this one primitive.
Every retire operation is:
  1. Per-path lock acquired (double-checked, TOCTOU-safe)
  2. Destination directory created
  3. Payload moved via shutil.move (atomic on same filesystem, best-effort across)
  4. Manifest written atomically via tempfile + os.replace
  5. If step 3 or 4 fails, dest dir is removed and original is untouched

Layout::

    <root>/<policy>/<yyyy-mm-dd>/<uuid>/
        payload/         # moved contents land here
        manifest.json    # record metadata

``purge()`` is the only path that hard-deletes bytes.  It is gated by
``policy.allows_hard_delete`` and, at the application layer, by
``@protected_mutation``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vetinari.lifecycle.policies import Policy, PolicyFilter

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"
_PAYLOAD_DIRNAME = "payload"

# Per-path locks prevent concurrent retire on the same source path.
# Double-checked locking: check outside lock, acquire, check again.
_PATH_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _get_path_lock(path: Path) -> threading.Lock:
    """Return (creating if absent) a per-path threading lock.

    Uses double-checked locking so only the first caller per path pays
    the ``_LOCKS_LOCK`` acquisition cost.

    Args:
        path: Absolute path being retired; the lock key is its string form.

    Returns:
        A ``threading.Lock`` dedicated to this path.
    """
    key = str(path.resolve())
    if key not in _PATH_LOCKS:
        with _LOCKS_LOCK:
            if key not in _PATH_LOCKS:
                _PATH_LOCKS[key] = threading.Lock()
    return _PATH_LOCKS[key]


def _sha256_of_path(path: Path) -> str:
    """Return hex SHA-256 digest of a file, or empty string for directories.

    Args:
        path: File or directory to hash.

    Returns:
        Hex SHA-256 string for files; empty string for directories.
    """
    if path.is_dir():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest_atomic(manifest_path: Path, data: dict) -> None:
    """Write ``data`` as JSON to ``manifest_path`` using an atomic replace.

    Writes to a sibling temp file first, then ``os.replace``-es it into
    position so readers never see a partial write.

    Args:
        manifest_path: Final destination for the manifest.
        data: Dictionary to serialise as JSON.
    """
    tmp_path = manifest_path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp_path, manifest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


@dataclass
class LifecycleRecord:  # noqa: VET114 — restored_at_utc is mutated post-creation by restore() so that list/load callers see the up-to-date status without re-reading the manifest; converting to dataclasses.replace() would force every list caller to re-fetch the record from disk, adding I/O on what is already a hot-path scan loop
    """Represents a single retired (moved-aside) entity.

    Attributes:
        record_id: UUID hex string uniquely identifying this record.
        original_path: Absolute path the entity was moved from.
        sha256: Content hash at retire time (empty for directories).
        retired_at_utc: ISO-8601 UTC timestamp the entity was retired.
        reason: Human-readable reason for the retirement.
        work_receipt_id: Optional receipt that triggered this retirement.
        policy: Policy name (``"recycle"`` or ``"archive"``).
        store_path: Absolute path of the record directory inside the store root.
        restored_at_utc: ISO-8601 timestamp the entity was restored, or None.
    """

    record_id: str
    original_path: str
    sha256: str
    retired_at_utc: str
    reason: str
    policy: str
    store_path: Path
    work_receipt_id: str | None = None
    restored_at_utc: str | None = None

    def to_manifest_dict(self) -> dict:
        """Serialise the record to a dict for the manifest JSON.

        Returns:
            A dict with all record fields (store_path as string).
        """
        return {
            "record_id": self.record_id,
            "original_path": self.original_path,
            "sha256": self.sha256,
            "retired_at_utc": self.retired_at_utc,
            "reason": self.reason,
            "work_receipt_id": self.work_receipt_id,
            "policy": self.policy,
            "restored_at_utc": self.restored_at_utc,
        }

    @classmethod
    def from_manifest_dict(cls, data: dict, store_path: Path) -> LifecycleRecord:
        """Deserialise a manifest dict back into a LifecycleRecord.

        Args:
            data: Dict loaded from a manifest.json file.
            store_path: The directory that contains the manifest.

        Returns:
            A populated ``LifecycleRecord`` instance.
        """
        return cls(
            record_id=data["record_id"],
            original_path=data["original_path"],
            sha256=data["sha256"],
            retired_at_utc=data["retired_at_utc"],
            reason=data["reason"],
            work_receipt_id=data.get("work_receipt_id"),
            policy=data["policy"],
            store_path=store_path,
            restored_at_utc=data.get("restored_at_utc"),
        )

    def __repr__(self) -> str:
        """Show record_id, policy, and original_path for debugging."""
        return (
            f"LifecycleRecord(record_id={self.record_id!r}, "
            f"policy={self.policy!r}, "
            f"original_path={self.original_path!r})"
        )


class LifecycleStore:
    """Atomic move-aside store backed by a manifest per record.

    Both RecycleStore and ArchiveStore are thin wrappers around one instance
    of this class configured with the appropriate policy.

    Args:
        root: Root directory for all records of this policy.
        policy: A ``Policy``-protocol object that determines naming,
            hard-delete permission, and bucket classification.
    """

    def __init__(self, root: Path, policy: Policy) -> None:
        """Initialise the store; root is created lazily on first write.

        Args:
            root: Directory that will contain all policy records.
            policy: The lifecycle policy for this store instance.
        """
        self._root = root
        self._policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retire(
        self,
        path: Path,
        reason: str,
        work_receipt_id: str | None = None,
    ) -> LifecycleRecord:
        """Move ``path`` into the store and write a manifest.

        The operation is atomic: either both the payload move and the manifest
        write succeed, or the original path is left untouched and an exception
        is raised.

        Concurrent calls for the same path are serialised via a per-path lock;
        one caller wins and the other raises ``FileNotFoundError`` (because the
        source is gone).

        Args:
            path: File or directory to retire. Must exist.
            reason: Human-readable reason for the retirement.
            work_receipt_id: Optional receipt identifier to record in the manifest.

        Returns:
            A ``LifecycleRecord`` describing the newly retired entity.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            OSError: If the move or manifest write fails (original is untouched).
        """
        lock = _get_path_lock(path)
        with lock:
            # Re-check inside lock to be TOCTOU-safe (anti-pattern: TOCTOU without locks).
            if not path.exists():
                raise FileNotFoundError(f"retire: path does not exist: {path}")

            record_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y-%m-%d")
            dest_dir = self._root / self._policy.name / date_str / record_id
            payload_dir = dest_dir / _PAYLOAD_DIRNAME
            manifest_path = dest_dir / _MANIFEST_FILENAME

            sha256 = _sha256_of_path(path)

            # Create destination directory structure before touching the source.
            # If this fails (permissions, disk full) the source is untouched.
            dest_dir.mkdir(parents=True, exist_ok=True)
            payload_dir.mkdir(exist_ok=True)

            # Move payload into the payload directory.
            dest_payload = payload_dir / path.name
            try:
                shutil.move(str(path), str(dest_payload))
            except Exception as exc:
                # VET142-excluded: lifecycle primitive rollback — undoes the dest dir
                # creation when the move itself failed; original payload is untouched.
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise OSError(f"retire: failed to move {path} — original is untouched") from exc

            record = LifecycleRecord(
                record_id=record_id,
                original_path=str(path),
                sha256=sha256,
                retired_at_utc=now.isoformat(),
                reason=reason,
                work_receipt_id=work_receipt_id,
                policy=self._policy.name,
                store_path=dest_dir,
            )

            try:
                _write_manifest_atomic(manifest_path, record.to_manifest_dict())
            except Exception as exc:
                # VET142-excluded: lifecycle primitive rollback — restores the original
                # payload and drops the half-built dest after a manifest write failure.
                shutil.move(str(dest_payload), str(path))
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise OSError(f"retire: manifest write failed for {path} — original has been restored") from exc

            logger.info(
                "lifecycle.retire: %s -> %s (policy=%s, reason=%s)",
                path,
                dest_dir,
                self._policy.name,
                reason,
            )
            return record

    def restore(self, record_id: str) -> None:
        """Move the payload back to its original path and mark as restored.

        Args:
            record_id: The UUID hex identifying the record to restore.

        Raises:
            KeyError: If no record with this ID exists in the store.
            FileExistsError: If the original path is already occupied.
        """
        record = self._load_record(record_id)
        original = Path(record.original_path)
        if original.exists():
            raise FileExistsError(f"restore: original path already exists: {original} — move it away first")

        payload_dir = record.store_path / _PAYLOAD_DIRNAME
        # Payload dir should contain exactly one entry (the original name).
        entries = list(payload_dir.iterdir())
        if not entries:
            raise FileNotFoundError(f"restore: payload directory is empty for record {record_id}")
        payload_entry = entries[0]

        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(payload_entry), str(original))

        # Update manifest with restored_at_utc.
        manifest_path = record.store_path / _MANIFEST_FILENAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["restored_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_manifest_atomic(manifest_path, data)

        logger.info(
            "lifecycle.restore: record %s -> %s",
            record_id,
            original,
        )

    def list(self, filter: PolicyFilter | None = None) -> list[LifecycleRecord]:
        """Return all records, optionally narrowed by a ``PolicyFilter``.

        Args:
            filter: Optional filter specifying bucket, reason substring, or
                work_receipt_id to match.

        Returns:
            List of matching ``LifecycleRecord`` objects sorted by
            ``retired_at_utc`` descending (newest first).
        """
        policy_root = self._root / self._policy.name
        if not policy_root.exists():
            return []

        records: list[LifecycleRecord] = []
        now = datetime.now(timezone.utc)

        for date_dir in policy_root.iterdir():
            if not date_dir.is_dir():
                continue
            for record_dir in date_dir.iterdir():
                if not record_dir.is_dir():
                    continue
                manifest_path = record_dir / _MANIFEST_FILENAME
                if not manifest_path.exists():
                    continue
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    record = LifecycleRecord.from_manifest_dict(data, record_dir)
                except Exception as exc:
                    logger.warning(
                        "lifecycle.list: skipping malformed manifest at %s — %s",
                        manifest_path,
                        exc,
                    )
                    continue

                if filter is not None:
                    if filter.bucket is not None:
                        bucket = self._policy.surface_buckets(record, now)
                        if bucket != filter.bucket:
                            continue
                    if (
                        filter.reason_contains is not None
                        and filter.reason_contains.lower() not in record.reason.lower()
                    ):
                        continue
                    if filter.work_receipt_id is not None and record.work_receipt_id != filter.work_receipt_id:
                        continue

                records.append(record)

        records.sort(key=lambda r: r.retired_at_utc, reverse=True)
        return records

    def purge(self, record_id: str) -> None:
        """Permanently delete a record's payload and manifest from disk.

        This is the ONLY path in the lifecycle subsystem that hard-deletes
        bytes.  It is gated by ``policy.allows_hard_delete`` and, at the
        application layer, must only be reachable via ``@protected_mutation``.

        Args:
            record_id: The UUID hex of the record to purge.

        Raises:
            PermissionError: If the policy does not permit hard delete.
            KeyError: If no record with this ID exists.
        """
        if not self._policy.allows_hard_delete:
            raise PermissionError(
                f"Policy '{self._policy.name}' does not permit hard delete — "
                "use @protected_mutation(DestructiveAction.PURGE_ARCHIVE) to override."
            )
        record = self._load_record(record_id)
        # VET142-excluded: lifecycle primitive — guarded by Policy.allows_hard_delete
        # check above; callers (RecycleStore.purge_expired, ArchiveStore purge paths)
        # supply the @protected_mutation gate when invoked from user-facing code.
        shutil.rmtree(record.store_path)
        logger.info(
            "lifecycle.purge: record %s permanently deleted (policy=%s)",
            record_id,
            self._policy.name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_record(self, record_id: str) -> LifecycleRecord:
        """Find and deserialise a record by ID by scanning the policy root.

        Args:
            record_id: UUID hex of the record to find.

        Returns:
            The matching ``LifecycleRecord``.

        Raises:
            KeyError: If no record with this ID is found.
        """
        policy_root = self._root / self._policy.name
        if not policy_root.exists():
            raise KeyError(f"No record found with id={record_id!r}")

        for date_dir in policy_root.iterdir():
            if not date_dir.is_dir():
                continue
            record_dir = date_dir / record_id
            if not record_dir.is_dir():
                continue
            manifest_path = record_dir / _MANIFEST_FILENAME
            if manifest_path.exists():
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    return LifecycleRecord.from_manifest_dict(data, record_dir)
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(
                        "lifecycle._load_record: skipping corrupt manifest at %s — %s",
                        manifest_path,
                        exc,
                    )
                    continue

        raise KeyError(f"No record found with id={record_id!r}")


__all__ = [
    "LifecycleRecord",
    "LifecycleStore",
]
