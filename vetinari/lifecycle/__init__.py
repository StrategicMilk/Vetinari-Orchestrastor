"""Vetinari lifecycle subsystem — atomic move-aside with policy-backed manifests.

Two policies are provided:

- ``RecyclePolicy`` / ``RecycleStore`` — short grace window (72h default),
  hard-delete allowed via ``purge_expired()``.
- ``ArchivePolicy`` / ``ArchiveStore`` — unbounded retention, view-tiered by
  age (recent / cooling / cold), no hard-delete without ``@protected_mutation``.

Both are thin facades over the shared ``LifecycleStore`` primitive.
"""

from __future__ import annotations

from vetinari.lifecycle.archive import ArchiveCandidate, ArchiveStore
from vetinari.lifecycle.policies import ArchivePolicy, PolicyFilter, RecyclePolicy
from vetinari.lifecycle.store import LifecycleRecord, LifecycleStore

__all__ = [
    "ArchiveCandidate",
    "ArchivePolicy",
    "ArchiveStore",
    "LifecycleRecord",
    "LifecycleStore",
    "PolicyFilter",
    "RecyclePolicy",
]
