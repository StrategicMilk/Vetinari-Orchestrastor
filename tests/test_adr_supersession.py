"""Tests for ADRSystem.supersede_adr — supersession protocol wiring.

Covers the new supersede_adr helper added to close the review finding S3
(no write-site helper uses the superseded_by field). Verifies that calling
supersede_adr sets the right status, links both sides, and persists to disk.

Part of the wave-0.5 follow-up landing the review recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path

from vetinari.adr import ADRStatus, ADRSystem


def _make_pair(tmp_path: Path) -> tuple[ADRSystem, str, str]:
    """Create an ADRSystem with an old and a replacement ADR ready to supersede.

    Args:
        tmp_path: Pytest tmp_path fixture root used as the storage directory.

    Returns:
        Tuple of (system, old_id, new_id) — the old ADR is the target of
        supersession, the new ADR is the replacement.
    """
    system = ADRSystem(storage_path=str(tmp_path))
    old = system.create_adr(
        title="Use direct HTTP for inference",
        category="integration",
        context="We need an inference backend.",
        decision="Call provider HTTP APIs directly.",
        consequences="Simple; but provider-specific.",
        status=ADRStatus.ACCEPTED.value,
    )
    new = system.create_adr(
        title="Use adapter registry for inference",
        category="integration",
        context="Direct HTTP was too provider-specific.",
        decision="Route all inference through a pluggable adapter registry.",
        consequences="Harder to debug, but provider-agnostic.",
        status=ADRStatus.ACCEPTED.value,
    )
    return system, old.adr_id, new.adr_id


class TestSupersedeAdrHappyPath:
    """supersede_adr sets status, superseded_by, and bidirectional links."""

    def test_status_becomes_superseded(self, tmp_path: Path) -> None:
        system, old_id, new_id = _make_pair(tmp_path)

        result = system.supersede_adr(old_id, new_id)

        assert result is not None
        assert result.adr_id == old_id
        assert result.status == ADRStatus.SUPERSEDED.value, (
            f"expected status={ADRStatus.SUPERSEDED.value!r}, got {result.status!r}"
        )

    def test_superseded_by_field_set(self, tmp_path: Path) -> None:
        """The new superseded_by field MUST be populated — this is the wiring."""
        system, old_id, new_id = _make_pair(tmp_path)

        system.supersede_adr(old_id, new_id)

        old = system.get_adr(old_id)
        assert old is not None
        assert old.superseded_by == new_id, f"superseded_by must be set to {new_id!r}, got {old.superseded_by!r}"

    def test_bidirectional_related_adrs(self, tmp_path: Path) -> None:
        """Both sides of the supersession MUST reference each other."""
        system, old_id, new_id = _make_pair(tmp_path)

        system.supersede_adr(old_id, new_id)

        old = system.get_adr(old_id)
        new = system.get_adr(new_id)
        assert old is not None and new is not None
        assert new_id in old.related_adrs, f"old.related_adrs must contain {new_id!r}, got {old.related_adrs!r}"
        assert old_id in new.related_adrs, f"new.related_adrs must contain {old_id!r}, got {new.related_adrs!r}"

    def test_persists_superseded_by_to_disk(self, tmp_path: Path) -> None:
        """Round-trip the superseded_by field through the JSON file."""
        system, old_id, new_id = _make_pair(tmp_path)

        system.supersede_adr(old_id, new_id)

        # Re-read the JSON file directly — do not trust the in-memory copy.
        raw = json.loads((tmp_path / f"{old_id}.json").read_text(encoding="utf-8"))
        assert raw["status"] == ADRStatus.SUPERSEDED.value
        assert raw["superseded_by"] == new_id

    def test_reload_from_disk_preserves_superseded_by(self, tmp_path: Path) -> None:
        """A fresh ADRSystem loading the same directory sees the superseded_by link."""
        system, old_id, new_id = _make_pair(tmp_path)
        system.supersede_adr(old_id, new_id)

        fresh = ADRSystem(storage_path=str(tmp_path))
        reloaded = fresh.get_adr(old_id)

        assert reloaded is not None
        assert reloaded.status == ADRStatus.SUPERSEDED.value
        assert reloaded.superseded_by == new_id

    def test_idempotent_related_adrs(self, tmp_path: Path) -> None:
        """Calling supersede_adr twice MUST NOT duplicate entries in related_adrs."""
        system, old_id, new_id = _make_pair(tmp_path)

        system.supersede_adr(old_id, new_id)
        system.supersede_adr(old_id, new_id)

        old = system.get_adr(old_id)
        new = system.get_adr(new_id)
        assert old is not None and new is not None
        assert old.related_adrs.count(new_id) == 1, f"duplicate entry on repeated supersession: {old.related_adrs!r}"
        assert new.related_adrs.count(old_id) == 1


class TestSupersedeAdrFailClosed:
    """Missing ADRs MUST return None, not crash or silently succeed."""

    def test_missing_old_adr_returns_none(self, tmp_path: Path) -> None:
        system, _, new_id = _make_pair(tmp_path)

        result = system.supersede_adr("ADR-9999", new_id)

        assert result is None

    def test_missing_replacement_returns_none(self, tmp_path: Path) -> None:
        system, old_id, _ = _make_pair(tmp_path)

        result = system.supersede_adr(old_id, "ADR-9999")

        assert result is None

    def test_missing_replacement_does_not_mutate_old(self, tmp_path: Path) -> None:
        """On failure, the old ADR MUST NOT have its status flipped."""
        system, old_id, _ = _make_pair(tmp_path)
        original_status = system.get_adr(old_id).status  # type: ignore[union-attr]

        system.supersede_adr(old_id, "ADR-9999")

        still = system.get_adr(old_id)
        assert still is not None
        assert still.status == original_status, (
            "Missing replacement must leave the old ADR unchanged — no partial mutation."
        )
        assert still.superseded_by is None
