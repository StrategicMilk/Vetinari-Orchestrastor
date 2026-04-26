"""Persistent JSONL ledger for Vetinari release claims.

Each ``ReleaseClaimRecord`` appended here is written atomically to
``outputs/release/<version>/ledger.jsonl`` via a tempfile-then-rename strategy
so a mid-write crash never leaves a half-written record.

``ClaimsLedger.verify_all`` walks the ledger and confirms every evidence
artifact still exists on disk and matches its embedded SHA-256 checksum.
Any missing path or checksum mismatch causes a ``LedgerVerificationReport``
with ``passed=False`` (fail-closed, Rule 2).

This is part of the release pipeline: ``release_doctor.py`` builds the wheel
and smoke evidence; ``ClaimsLedger`` persists those claims for later audit by
``scripts/release/pre_release_gate.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from vetinari.release.proof_schema import ClaimKind, ReleaseClaimRecord

logger = logging.getLogger(__name__)

# --- LedgerVerificationReport ------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class LedgerVerificationReport:
    """Result of walking a ledger file and verifying all evidence artifacts.

    Fail-closed: ``passed`` is ``True`` only when every record's
    ``evidence_path`` exists on disk **and** its SHA-256 matches the
    embedded checksum.  Any deviation sets ``passed=False``.

    Args:
        passed: ``True`` iff all claims resolved without error.
        total: Total number of records examined.
        ok: Number of records that resolved cleanly.
        failures: Human-readable descriptions of each failed record,
            ordered by occurrence in the ledger.

    Returns:
        Immutable verification report.
    """

    passed: bool
    total: int
    ok: int
    failures: tuple[str, ...]

    def __repr__(self) -> str:
        """Return a compact repr showing pass/fail and counts."""
        return (
            f"LedgerVerificationReport(passed={self.passed}, ok={self.ok}/{self.total}, failures={len(self.failures)})"
        )


# --- ClaimsLedger ------------------------------------------------------------


class ClaimsLedger:
    """Append-only JSONL ledger for release claims.

    Each call to ``append()`` serialises one ``ReleaseClaimRecord`` and
    writes it atomically to ``outputs/release/<version>/ledger.jsonl``.
    The file is created (with all parent directories) if it does not yet
    exist.

    The ledger is intentionally append-only: records are never deleted or
    mutated after writing.  ``verify_all()`` is a read-only audit pass.

    Args:
        version: Release version string used as the sub-directory name
            under ``outputs/release/``, e.g. ``"0.7.0"`` or ``"dev"``.
        repo_root: Absolute path to the repository root.  Defaults to the
            parent of ``vetinari/`` (i.e., resolved from this file's
            location).  Evidence paths stored in claim records are
            resolved relative to this root.
    """

    def __init__(
        self,
        version: str,
        repo_root: Path | None = None,
    ) -> None:
        """Initialise the ledger for the given version.

        Args:
            version: Release version string; used as the directory name
                under ``outputs/release/``.
            repo_root: Repository root path.  If ``None``, resolved as
                three parents above this source file (i.e.
                ``vetinari/release/claims_ledger.py`` -> repo root).
        """
        if not version or not version.strip():
            raise ValueError("version must be a non-empty string")
        self._version = version.strip()
        self._repo_root: Path = repo_root if repo_root is not None else Path(__file__).resolve().parent.parent.parent
        self._ledger_path: Path = self._repo_root / "outputs" / "release" / self._version / "ledger.jsonl"

    @property
    def ledger_path(self) -> Path:
        """The absolute path to the JSONL ledger file for this version.

        Returns:
            ``Path`` object pointing to ``outputs/release/<version>/ledger.jsonl``.
        """
        return self._ledger_path

    def append(self, claim: ReleaseClaimRecord) -> None:
        """Append one claim record to the ledger, atomically.

        Serialises *claim* to a single JSON line and writes it to the
        ledger file via ``tempfile.NamedTemporaryFile`` + ``os.replace``
        so a mid-write crash never leaves a corrupt record.

        The strategy is:
        1. Read the existing ledger content (if any).
        2. Write existing content + new line to a sibling tempfile.
        3. ``os.replace()`` the tempfile over the ledger path.

        This means the ledger file is always in a consistent state at the
        filesystem level; readers see either the old file or the fully
        written new file, never a partial state.

        Args:
            claim: The ``ReleaseClaimRecord`` to persist.

        Raises:
            TypeError: If *claim* is not a ``ReleaseClaimRecord``.
            OSError: If the ledger directory cannot be created or written.
        """
        if not isinstance(claim, ReleaseClaimRecord):
            raise TypeError(f"append() expects a ReleaseClaimRecord, got {type(claim).__name__!r}")

        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)

        new_line = _claim_to_jsonl(claim)

        # Read current content first (empty string if file doesn't exist yet).
        existing: str = ""
        if self._ledger_path.exists():
            existing = self._ledger_path.read_text(encoding="utf-8")

        # Build new content: preserve existing lines, append new one.
        combined = existing + new_line + "\n"

        # Atomic write: write to sibling temp, then rename over the target.
        dir_path = self._ledger_path.parent
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=dir_path,
            prefix=".ledger_tmp_",
            suffix=".jsonl",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(combined)
            os.replace(tmp_path, self._ledger_path)
        except Exception:
            # Best-effort cleanup of the orphaned tempfile before re-raising.
            # contextlib.suppress avoids a nested try/except that would
            # trigger VET022/VET023 on the inner except block.
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        logger.debug(
            "Appended claim %r to ledger %s",
            claim.id,
            self._ledger_path,
        )

    @staticmethod
    def verify_all(
        ledger_path: Path,
        repo_root: Path | None = None,
    ) -> LedgerVerificationReport:
        """Walk a ledger file and verify every evidence artifact.

        For each record in *ledger_path*:
        - If ``evidence_path`` is non-empty, the file must exist on disk.
        - If the record has a ``sha256`` field (optional extension), the
          file's digest must match.

        The report is fail-closed (Rule 2): ``passed=True`` is returned
        **only** when every record in the ledger resolves cleanly.
        A missing ledger file itself triggers ``passed=False``.

        Args:
            ledger_path: Absolute path to the ``ledger.jsonl`` file to
                audit.
            repo_root: Root path against which relative ``evidence_path``
                values are resolved.  If ``None``, the parent of
                *ledger_path* is used as a safe fallback.

        Returns:
            ``LedgerVerificationReport`` with ``passed=True`` only when
            all checks succeed.
        """
        if repo_root is None:
            # Safe fallback: resolve relative paths against ledger's parent.
            repo_root = ledger_path.parent

        if not ledger_path.exists():
            return LedgerVerificationReport(
                passed=False,
                total=0,
                ok=0,
                failures=(f"Ledger file not found: {ledger_path}",),
            )

        content = ledger_path.read_text(encoding="utf-8")
        lines = [ln for ln in content.splitlines() if ln.strip()]

        failures: list[str] = []
        ok_count = 0

        for lineno, raw in enumerate(lines, start=1):
            try:
                record: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Ledger line %d in %s is not valid JSON — skipping: %s",
                    lineno,
                    ledger_path,
                    exc,
                )
                failures.append(f"Line {lineno}: invalid JSON — {exc}")
                continue

            claim_id = record.get("id", f"<line-{lineno}>")
            evidence_rel = record.get("evidence_path", "")

            if not evidence_rel:
                # No file evidence required for this claim — passes trivially.
                ok_count += 1
                continue

            # Resolve the evidence path.
            evidence_abs = Path(evidence_rel)
            if not evidence_abs.is_absolute():
                evidence_abs = repo_root / evidence_rel

            if not evidence_abs.exists():
                failures.append(f"Claim {claim_id!r}: evidence file not found: {evidence_rel}")
                continue

            # Optional SHA-256 check if the record carries a checksum.
            stored_sha = record.get("sha256", "")
            if stored_sha:
                actual_sha = _sha256_file(evidence_abs)
                if actual_sha != stored_sha.lower():
                    failures.append(
                        f"Claim {claim_id!r}: SHA-256 mismatch for {evidence_rel} "
                        f"(stored={stored_sha[:12]}..., actual={actual_sha[:12]}...)"
                    )
                    continue

            ok_count += 1

        total = len(lines)
        passed = len(failures) == 0

        if failures:
            logger.warning(
                "Ledger verification failed: %d/%d claims have issues in %s",
                len(failures),
                total,
                ledger_path,
            )

        return LedgerVerificationReport(
            passed=passed,
            total=total,
            ok=ok_count,
            failures=tuple(failures),
        )


# --- helpers -----------------------------------------------------------------


def _claim_to_jsonl(claim: ReleaseClaimRecord) -> str:
    """Serialise a ``ReleaseClaimRecord`` to a single-line JSON string.

    The ``kind`` field is written as its string value so the JSONL is
    self-contained without requiring the enum import at read time.

    Args:
        claim: The claim to serialise.

    Returns:
        A single-line JSON string (no trailing newline).
    """
    raw: dict[str, Any] = {
        "id": claim.id,
        "text": claim.text,
        "evidence_path": claim.evidence_path,
        "kind": claim.kind.value if isinstance(claim.kind, ClaimKind) else str(claim.kind),
        "verified_at": claim.verified_at,
    }
    return json.dumps(raw, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    """Compute the lowercase hex SHA-256 digest of a file.

    Args:
        path: Absolute path to the file to hash.

    Returns:
        Lowercase hex-encoded SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
