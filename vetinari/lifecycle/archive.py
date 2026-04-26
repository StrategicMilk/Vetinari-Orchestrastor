"""ArchiveStore facade — view-tiered archive built on LifecycleStore.

Provides a high-level API for archiving completed entities with no automatic
hard-delete path.  Records are classified into view tiers by age:

- ``"recent"``  : age <= 7 days
- ``"cooling"`` : 7 < age <= 30 days
- ``"cold"``    : age > 30 days

Tier thresholds are configurable via ``safety_defaults.yaml``.  Hard delete
of an archive record requires explicit ``@protected_mutation`` — the default
UX never exposes a purge button.

``ArchiveStore.sweep()`` is the LW-UX-08 auto-archive hook; call it with a
list of ``ArchiveCandidate`` objects and it archives only those whose cooldown
has elapsed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from vetinari.lifecycle.policies import ArchivePolicy
from vetinari.lifecycle.store import LifecycleRecord, LifecycleStore
from vetinari.safety.safety_defaults import load_safety_defaults  # runtime — used in ArchiveStore.__init__

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ArchiveCandidate:
    """Descriptor for a path that the sweep job should consider archiving.

    Attributes:
        path: The file or directory to potentially archive.
        completed_at: When the work producing this path finished.
        cooldown_hours: How long to wait after completion before archiving.
        reason: Human-readable reason (written into the manifest).
        work_receipt_id: Optional receipt identifier to record in the manifest.
    """

    path: Path
    completed_at: datetime
    cooldown_hours: float
    reason: str
    work_receipt_id: str | None = None

    def __repr__(self) -> str:
        """Show path and reason for debugging."""
        return f"ArchiveCandidate(path={self.path!r}, reason={self.reason!r})"


class ArchiveStore:
    """View-tiered archive facade over LifecycleStore.

    Entities are moved aside with a manifest and classified into view tiers
    based on their age.  No hard-delete path is exposed via the public API.

    When called with no arguments, ``root``, ``recent_days``, and
    ``cooling_days`` are read from ``config/safety_defaults.yaml`` via
    ``load_safety_defaults()`` so that all callers automatically pick up
    per-deployment configuration.

    Args:
        root: Root directory for archive records.  Defaults to the value
            from ``safety_defaults.yaml`` (``outputs/archive``).
        recent_days: Records younger than this are in the ``"recent"`` tier.
            Defaults to the value from ``safety_defaults.yaml`` (7).
        cooling_days: Records younger than this (but > recent) are
            ``"cooling"``; older records are ``"cold"``.  Defaults to the
            value from ``safety_defaults.yaml`` (30).
    """

    def __init__(
        self,
        root: Path | None = None,
        *,
        recent_days: int | None = None,
        cooling_days: int | None = None,
    ) -> None:
        """Initialise the store, reading defaults from safety_defaults.yaml when args are omitted.

        Args:
            root: Root directory for all archive records.  If ``None``, reads
                ``archive_policy.archive_root`` from ``config/safety_defaults.yaml``.
            recent_days: Age threshold (days) for the ``"recent"`` tier.  If
                ``None``, reads ``archive_policy.recent_days`` from the YAML.
            cooling_days: Age threshold (days) for the ``"cooling"`` tier;
                records older than this fall into ``"cold"``.  If ``None``,
                reads ``archive_policy.cooling_days`` from the YAML.
        """
        if root is None or recent_days is None or cooling_days is None:
            defaults = load_safety_defaults()
            effective_root = root if root is not None else defaults.archive_root
            effective_recent = recent_days if recent_days is not None else defaults.recent_days
            effective_cooling = cooling_days if cooling_days is not None else defaults.cooling_days
        else:
            effective_root = root
            effective_recent = recent_days
            effective_cooling = cooling_days
        policy = ArchivePolicy(recent_days=effective_recent, cooling_days=effective_cooling)
        self._store = LifecycleStore(root=effective_root, policy=policy)
        self._policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def archive(
        self,
        path: Path,
        reason: str,
        work_receipt_id: str | None = None,
    ) -> LifecycleRecord:
        """Move ``path`` into the archive store and record a manifest.

        Args:
            path: File or directory to archive.  Must exist.
            reason: Human-readable reason for archiving.
            work_receipt_id: Optional work receipt identifier to embed in the
                manifest for audit linkage.

        Returns:
            A ``LifecycleRecord`` describing the archived entity.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            OSError: If the move or manifest write fails.
        """
        return self._store.retire(path, reason=reason, work_receipt_id=work_receipt_id)

    def unarchive(self, record_id: str) -> None:
        """Restore an archived entity to its original path.

        Args:
            record_id: UUID hex of the record to restore.

        Raises:
            KeyError: If no record with this ID exists.
            FileExistsError: If the original path is already occupied.
        """
        self._store.restore(record_id)

    def list_by_tier(
        self,
        tier: str,
    ) -> list[LifecycleRecord]:
        """Return all records in the given view tier.

        Args:
            tier: One of ``"recent"``, ``"cooling"``, or ``"cold"``.

        Returns:
            List of ``LifecycleRecord`` objects in the requested tier,
            sorted by ``retired_at_utc`` descending.
        """
        from vetinari.lifecycle.policies import PolicyFilter

        return self._store.list(filter=PolicyFilter(bucket=tier))

    def search(self, query: str) -> list[LifecycleRecord]:
        """Find records by case-insensitive substring match.

        Matches against ``original_path``, ``reason``, and
        ``work_receipt_id``.  Slow path is acceptable per design (archive
        queries are infrequent).

        Args:
            query: Substring to search for (case-insensitive).

        Returns:
            List of matching ``LifecycleRecord`` objects.
        """
        q = query.lower()
        return [
            record
            for record in self._store.list()
            if (
                q in record.original_path.lower()
                or q in record.reason.lower()
                or (record.work_receipt_id is not None and q in record.work_receipt_id.lower())
            )
        ]

    def sweep(self, candidates: Iterable[ArchiveCandidate]) -> list[LifecycleRecord]:
        """Archive candidates whose cooldown has elapsed; idempotent.

        Candidates that are already absent (already archived or deleted by
        other means) are silently skipped.  Running sweep twice on the same
        list is safe.

        Args:
            candidates: Iterable of ``ArchiveCandidate`` objects.

        Returns:
            List of ``LifecycleRecord`` objects for newly archived entities.
        """
        archived: list[LifecycleRecord] = []
        now = datetime.now(timezone.utc)

        for candidate in candidates:
            # Skip if the path no longer exists (already archived or removed).
            if not candidate.path.exists():
                continue

            completed_at = candidate.completed_at
            if completed_at.tzinfo is None:
                completed_at = completed_at.replace(tzinfo=timezone.utc)

            elapsed = now - completed_at
            if elapsed < timedelta(hours=candidate.cooldown_hours):
                # Cooldown not yet elapsed — leave in place.
                continue

            try:
                record = self._store.retire(
                    candidate.path,
                    reason=candidate.reason,
                    work_receipt_id=candidate.work_receipt_id,
                )
                archived.append(record)
                logger.info(
                    "archive.sweep: archived %s (reason=%s)",
                    candidate.path,
                    candidate.reason,
                )
            except Exception as exc:
                logger.warning(
                    "archive.sweep: failed to archive %s — %s",
                    candidate.path,
                    exc,
                )

        return archived

    # ------------------------------------------------------------------
    # Protected hard-delete (only reachable via @protected_mutation)
    # ------------------------------------------------------------------

    def _purge_archive_record(self, record_id: str) -> None:
        """Hard-delete an archive record.  Only callable via @protected_mutation.

        This method is intentionally private (underscore-prefixed).  Callers
        must wrap it with ``@protected_mutation(DestructiveAction.PURGE_ARCHIVE)``
        to ensure a confirmed intent is present.

        Args:
            record_id: UUID hex of the record to permanently delete.

        Raises:
            PermissionError: Always — ArchivePolicy.allows_hard_delete is False.
        """
        # LifecycleStore.purge() enforces policy.allows_hard_delete check.
        self._store.purge(record_id)


__all__ = [
    "ArchiveCandidate",
    "ArchiveStore",
]
