"""Tests for vetinari/release/claims_ledger.py and scripts/release/pre_release_gate.py.

Covers:
    - ClaimsLedger.append: happy-path round-trip (Task 3.1)
    - ClaimsLedger.append: atomicity under simulated mid-write failure (Task 3.1)
    - ClaimsLedger.verify_all: missing evidence returns passed=False (Task 3.1)
    - ClaimsLedger.verify_all: SHA-256 mismatch returns passed=False (Task 3.1)
    - ClaimsLedger.verify_all: missing ledger file returns passed=False (Task 3.1)
    - pre_release_gate._run_gate: clean CHANGELOG + complete ledger -> exit 0 (Task 3.3)
    - pre_release_gate._run_gate: missing ledger tag -> exit 2 (Task 3.3)
    - pre_release_gate._run_gate: missing ledger file -> exit 2 (Task 3.3)
    - pre_release_gate._run_gate: ledger id not in ledger -> exit 2 (Task 3.3)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repo root and release scripts are importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts" / "release") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "release"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pre_release_gate as prg

from vetinari.release.claims_ledger import ClaimsLedger, LedgerVerificationReport, _claim_to_jsonl
from vetinari.release.proof_schema import ClaimKind, ReleaseClaimRecord

# ── Shared helpers ─────────────────────────────────────────────────────────


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_claim(
    claim_id: str = "test-claim-001",
    text: str = "Wheel SHA-256 matches build artefact",
    evidence_path: str = "",
    kind: ClaimKind = ClaimKind.TOOL_EVIDENCE,
) -> ReleaseClaimRecord:
    """Create a minimal ReleaseClaimRecord for testing."""
    return ReleaseClaimRecord(
        id=claim_id,
        text=text,
        evidence_path=evidence_path,
        kind=kind,
        verified_at=_utcnow(),
    )


# ── Task 3.1: ClaimsLedger.append ─────────────────────────────────────────


class TestClaimsLedgerAppend:
    """append() writes valid JSONL and the file is readable afterwards."""

    def test_append_creates_ledger_and_roundtrip(self, tmp_path: Path) -> None:
        """Appending a claim creates the ledger file and the record is recoverable."""
        ledger = ClaimsLedger(version="0.7.0-test", repo_root=tmp_path)
        claim = make_claim(claim_id="wheel-sha-match", text="Wheel SHA-256 matches")

        ledger.append(claim)

        assert ledger.ledger_path.exists(), "Ledger file must be created after first append"
        lines = [ln for ln in ledger.ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1, "One append should produce exactly one JSONL line"

        record = json.loads(lines[0])
        assert record["id"] == "wheel-sha-match"
        assert record["text"] == "Wheel SHA-256 matches"
        assert record["kind"] == ClaimKind.TOOL_EVIDENCE.value
        assert record["evidence_path"] == ""

    def test_append_multiple_claims(self, tmp_path: Path) -> None:
        """Multiple appends accumulate all records in order."""
        ledger = ClaimsLedger(version="0.7.0-multi", repo_root=tmp_path)
        claims = [make_claim(claim_id=f"claim-{i}") for i in range(3)]
        for c in claims:
            ledger.append(c)

        lines = [ln for ln in ledger.ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 3
        ids = [json.loads(ln)["id"] for ln in lines]
        assert ids == ["claim-0", "claim-1", "claim-2"]

    def test_append_rejects_non_claim(self, tmp_path: Path) -> None:
        """append() raises TypeError for non-ReleaseClaimRecord arguments."""
        ledger = ClaimsLedger(version="bad-type", repo_root=tmp_path)
        with pytest.raises(TypeError, match="ReleaseClaimRecord"):
            ledger.append({"id": "not-a-claim"})  # type: ignore[arg-type]

    def test_append_kind_serialised_as_string(self, tmp_path: Path) -> None:
        """ClaimKind enum is serialised to its string value, not the Python repr."""
        ledger = ClaimsLedger(version="kind-check", repo_root=tmp_path)
        claim = make_claim(kind=ClaimKind.HUMAN_ATTESTED)
        ledger.append(claim)

        line = ledger.ledger_path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["kind"] == "human_attested", "kind must serialise to string value"


# ── Task 3.1: atomicity under simulated failure ────────────────────────────


class TestClaimsLedgerAtomicity:
    """Verify that a write failure leaves the ledger in its prior consistent state."""

    def test_failed_write_does_not_corrupt_existing_ledger(self, tmp_path: Path) -> None:
        """If the file rename fails, the pre-existing ledger is not modified."""
        ledger = ClaimsLedger(version="atomic-test", repo_root=tmp_path)

        # Write a good record first.
        first_claim = make_claim(claim_id="first-claim")
        ledger.append(first_claim)
        original_content = ledger.ledger_path.read_text(encoding="utf-8")

        # Simulate os.replace raising mid-write.
        with patch("vetinari.release.claims_ledger.os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                ledger.append(make_claim(claim_id="second-claim"))

        # Ledger must still contain exactly the original content.
        after_content = ledger.ledger_path.read_text(encoding="utf-8")
        assert after_content == original_content, "Ledger must not be modified when os.replace fails"

    def test_no_orphan_tempfile_after_failure(self, tmp_path: Path) -> None:
        """Orphaned tempfiles are cleaned up when os.replace fails."""
        ledger = ClaimsLedger(version="orphan-check", repo_root=tmp_path)
        dir_path = ledger.ledger_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        before_files = set(dir_path.iterdir())

        with patch("vetinari.release.claims_ledger.os.replace", side_effect=OSError("no space")):
            with pytest.raises(OSError, match="no space"):
                ledger.append(make_claim())

        after_files = set(dir_path.iterdir())
        new_files = after_files - before_files
        # Any new file should not be a leftover tempfile (no .ledger_tmp_ prefix).
        for f in new_files:
            assert not f.name.startswith(".ledger_tmp_"), f"Orphan tempfile left behind: {f.name}"


# ── Task 3.1: verify_all ──────────────────────────────────────────────────


class TestVerifyAll:
    """verify_all returns passed=False for missing / mismatched evidence."""

    def test_verify_all_empty_ledger_passes(self, tmp_path: Path) -> None:
        """An empty ledger (no records) passes trivially."""
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text("", encoding="utf-8")

        report = ClaimsLedger.verify_all(ledger_path)
        assert isinstance(report, LedgerVerificationReport)
        assert report.passed is True
        assert report.total == 0

    def test_verify_all_no_evidence_path_passes(self, tmp_path: Path) -> None:
        """Claims with empty evidence_path do not require a file on disk."""
        ledger = ClaimsLedger(version="no-ev", repo_root=tmp_path)
        ledger.append(make_claim(claim_id="no-file", evidence_path=""))

        report = ClaimsLedger.verify_all(ledger.ledger_path)
        assert report.passed is True
        assert report.ok == 1

    def test_verify_all_missing_evidence_file_fails_closed(self, tmp_path: Path) -> None:
        """A claim referencing a non-existent evidence file causes passed=False."""
        ledger = ClaimsLedger(version="missing-ev", repo_root=tmp_path)
        # Write the JSONL directly since the evidence file won't exist.
        line = json.dumps({
            "id": "missing-ev-001",
            "text": "evidence gone",
            "evidence_path": "outputs/release/missing-ev/proof.log",
            "kind": "tool_evidence",
            "verified_at": _utcnow(),
        })
        ledger.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger.ledger_path.write_text(line + "\n", encoding="utf-8")

        report = ClaimsLedger.verify_all(ledger.ledger_path, repo_root=tmp_path)
        assert report.passed is False, "Missing evidence file must fail closed"
        assert report.ok == 0
        assert len(report.failures) == 1
        assert "missing-ev-001" in report.failures[0]

    def test_verify_all_sha256_mismatch_fails_closed(self, tmp_path: Path) -> None:
        """A SHA-256 mismatch in the ledger causes passed=False."""
        ev_file = tmp_path / "outputs" / "proof.log"
        ev_file.parent.mkdir(parents=True, exist_ok=True)
        ev_file.write_text("real content", encoding="utf-8")

        # Compute a wrong hash.
        wrong_sha = hashlib.sha256(b"wrong content").hexdigest()

        line = json.dumps({
            "id": "sha-mismatch-001",
            "text": "wheel matches",
            "evidence_path": str(ev_file.relative_to(tmp_path)),
            "sha256": wrong_sha,
            "kind": "tool_evidence",
            "verified_at": _utcnow(),
        })
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text(line + "\n", encoding="utf-8")

        report = ClaimsLedger.verify_all(ledger_path, repo_root=tmp_path)
        assert report.passed is False, "SHA-256 mismatch must fail closed"
        assert "sha-mismatch-001" in report.failures[0]
        assert "mismatch" in report.failures[0].lower()

    def test_verify_all_correct_sha256_passes(self, tmp_path: Path) -> None:
        """A matching SHA-256 checksum allows the claim to pass."""
        ev_file = tmp_path / "outputs" / "proof.log"
        ev_file.parent.mkdir(parents=True, exist_ok=True)
        ev_file.write_bytes(b"verified content")

        correct_sha = hashlib.sha256(b"verified content").hexdigest()

        line = json.dumps({
            "id": "sha-ok-001",
            "text": "wheel matches",
            "evidence_path": str(ev_file.relative_to(tmp_path)),
            "sha256": correct_sha,
            "kind": "tool_evidence",
            "verified_at": _utcnow(),
        })
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text(line + "\n", encoding="utf-8")

        report = ClaimsLedger.verify_all(ledger_path, repo_root=tmp_path)
        assert report.passed is True
        assert report.ok == 1

    def test_verify_all_missing_ledger_file_fails_closed(self, tmp_path: Path) -> None:
        """A non-existent ledger file causes passed=False (fail-closed)."""
        absent_path = tmp_path / "does_not_exist.jsonl"
        report = ClaimsLedger.verify_all(absent_path)
        assert report.passed is False, "Missing ledger file must fail closed"
        assert report.total == 0
        assert "not found" in report.failures[0].lower()


# ── Task 3.3: pre_release_gate ────────────────────────────────────────────


def _write_changelog(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_ledger(path: Path, claim_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "id": cid,
            "text": f"Evidence for {cid}",
            "evidence_path": "",
            "kind": "tool_evidence",
            "verified_at": _utcnow(),
        })
        for cid in claim_ids
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_CLEAN_CHANGELOG = """\
# Changelog

## [0.7.0] - 2026-04-25

### Added

- Inspector fails closed on missing citations. [ledger:claim-001]
- Wheel SHA-256 verified against build artefact. [ledger:claim-002]

## [0.6.0] - 2026-04-22

### Added

- Initial release.
"""

_MISSING_TAG_CHANGELOG = """\
# Changelog

## [0.7.0] - 2026-04-25

### Added

- Inspector fails closed on missing citations. [ledger:claim-001]
- This line has no ledger tag at all.

## [0.6.0] - 2026-04-22

### Added

- Initial release.
"""

_WRONG_ID_CHANGELOG = """\
# Changelog

## [0.7.0] - 2026-04-25

### Added

- Inspector fails closed on missing citations. [ledger:claim-001]
- Wheel SHA-256 verified. [ledger:claim-MISSING-ID]

## [0.6.0] - 2026-04-22

### Added

- Initial release.
"""


class TestPreReleaseGate:
    """_run_gate returns 0 on pass and 2 on any failure."""

    def test_clean_changelog_complete_ledger_exits_0(self, tmp_path: Path) -> None:
        """All claims tagged and all ids in ledger: gate passes with exit code 0."""
        changelog = tmp_path / "CHANGELOG.md"
        _write_changelog(changelog, _CLEAN_CHANGELOG)

        ledger = tmp_path / "ledger.jsonl"
        _write_ledger(ledger, ["claim-001", "claim-002"])

        result = prg._run_gate(changelog_path=changelog, ledger_path=ledger)
        assert result == 0, f"Expected exit 0, got {result}"

    def test_missing_ledger_tag_exits_2(self, tmp_path: Path) -> None:
        """A claim line with no [ledger:...] tag causes exit 2."""
        changelog = tmp_path / "CHANGELOG.md"
        _write_changelog(changelog, _MISSING_TAG_CHANGELOG)

        ledger = tmp_path / "ledger.jsonl"
        _write_ledger(ledger, ["claim-001"])

        result = prg._run_gate(changelog_path=changelog, ledger_path=ledger)
        assert result == 2, f"Expected exit 2 for missing tag, got {result}"

    def test_missing_ledger_file_exits_2(self, tmp_path: Path) -> None:
        """A missing ledger file causes exit 2 (fail-closed, not pass-through)."""
        changelog = tmp_path / "CHANGELOG.md"
        _write_changelog(changelog, _CLEAN_CHANGELOG)

        absent_ledger = tmp_path / "no_such_ledger.jsonl"
        assert not absent_ledger.exists()

        result = prg._run_gate(changelog_path=changelog, ledger_path=absent_ledger)
        assert result == 2, f"Expected exit 2 for missing ledger file, got {result}"

    def test_ledger_id_not_found_exits_2(self, tmp_path: Path) -> None:
        """A [ledger:<id>] tag that does not match any record in the ledger exits 2."""
        changelog = tmp_path / "CHANGELOG.md"
        _write_changelog(changelog, _WRONG_ID_CHANGELOG)

        ledger = tmp_path / "ledger.jsonl"
        # Only claim-001 is in the ledger; claim-MISSING-ID is not.
        _write_ledger(ledger, ["claim-001"])

        result = prg._run_gate(changelog_path=changelog, ledger_path=ledger)
        assert result == 2, f"Expected exit 2 for unresolved ledger id, got {result}"

    def test_convention_start_skips_historical_version(self, tmp_path: Path) -> None:
        """When --convention-start points to a future version, historical entry passes."""
        changelog = tmp_path / "CHANGELOG.md"
        # Latest entry is 0.7.0, convention starts at 0.8.0 — should pass.
        _write_changelog(changelog, _MISSING_TAG_CHANGELOG)

        absent_ledger = tmp_path / "absent.jsonl"

        result = prg._run_gate(
            changelog_path=changelog,
            ledger_path=absent_ledger,
            convention_start_version="0.8.0",
        )
        # 0.7.0 < 0.8.0 so the gate should skip enforcement and return 0.
        assert result == 0, f"Expected exit 0 when latest version predates convention start, got {result}"

    @pytest.mark.parametrize(
        "changelog_text, ledger_ids, expected_code",
        [
            # All claims tagged and ledger complete.
            (_CLEAN_CHANGELOG, ["claim-001", "claim-002"], 0),
            # One claim has no tag.
            (_MISSING_TAG_CHANGELOG, ["claim-001"], 2),
            # Tag present but id absent from ledger.
            (_WRONG_ID_CHANGELOG, ["claim-001"], 2),
        ],
    )
    def test_gate_parametrized(
        self,
        tmp_path: Path,
        changelog_text: str,
        ledger_ids: list[str],
        expected_code: int,
    ) -> None:
        """Parametrized sweep of the three main gate scenarios."""
        changelog = tmp_path / "CHANGELOG.md"
        _write_changelog(changelog, changelog_text)
        ledger = tmp_path / "ledger.jsonl"
        _write_ledger(ledger, ledger_ids)

        result = prg._run_gate(changelog_path=changelog, ledger_path=ledger)
        assert result == expected_code
