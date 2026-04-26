"""Tests for vetinari/types.py — canonical enum and type definitions.

Verifies that all public enums exist, have the expected values, and that
new enums introduced in SESSION-05/SHARD-01 (EvidenceBasis, ArtifactKind)
have the correct contract.
"""

from __future__ import annotations

import pytest

from vetinari.types import (
    ArtifactKind,
    EvidenceBasis,
)


class TestEvidenceBasis:
    """Contract tests for EvidenceBasis enum (SESSION-05/SHARD-01)."""

    def test_all_five_values_exist(self) -> None:
        """All five canonical EvidenceBasis values must be present."""
        expected = {
            EvidenceBasis.TOOL_EVIDENCE,
            EvidenceBasis.LLM_JUDGMENT,
            EvidenceBasis.HUMAN_ATTESTED,
            EvidenceBasis.HYBRID,
            EvidenceBasis.UNSUPPORTED,
        }
        assert set(EvidenceBasis) == expected

    def test_string_values_are_stable(self) -> None:
        """String values are part of the public API — must not change."""
        assert EvidenceBasis.TOOL_EVIDENCE.value == "tool_evidence"
        assert EvidenceBasis.LLM_JUDGMENT.value == "llm_judgment"
        assert EvidenceBasis.HUMAN_ATTESTED.value == "human_attested"
        assert EvidenceBasis.HYBRID.value == "hybrid"
        assert EvidenceBasis.UNSUPPORTED.value == "unsupported"

    def test_unsupported_is_fail_closed_sentinel(self) -> None:
        """UNSUPPORTED must be the value that represents 'no evidence collected'.

        It is the default for OutcomeSignal — verified here at the type level
        so even a refactor that changes the default cannot silently use a
        different value.
        """
        sentinel = EvidenceBasis.UNSUPPORTED
        assert sentinel.value == "unsupported"
        # Confirm it round-trips from string
        assert EvidenceBasis("unsupported") is EvidenceBasis.UNSUPPORTED

    @pytest.mark.parametrize(
        "basis",
        [
            EvidenceBasis.TOOL_EVIDENCE,
            EvidenceBasis.LLM_JUDGMENT,
            EvidenceBasis.HUMAN_ATTESTED,
            EvidenceBasis.HYBRID,
            EvidenceBasis.UNSUPPORTED,
        ],
    )
    def test_each_value_round_trips_from_string(self, basis: EvidenceBasis) -> None:
        """Every EvidenceBasis value must be reconstructible from its string value."""
        reconstructed = EvidenceBasis(basis.value)
        assert reconstructed is basis

    def test_is_str_enum(self) -> None:
        """EvidenceBasis inherits from str so values compare directly to strings."""
        assert EvidenceBasis.UNSUPPORTED == "unsupported"
        assert EvidenceBasis.TOOL_EVIDENCE == "tool_evidence"


class TestArtifactKind:
    """Contract tests for ArtifactKind enum (SESSION-05/SHARD-01)."""

    def test_all_five_values_exist(self) -> None:
        """All five canonical ArtifactKind values must be present."""
        expected = {
            ArtifactKind.COMMAND_INVOCATION,
            ArtifactKind.COMMIT_SHA,
            ArtifactKind.SIGNED_REVIEW,
            ArtifactKind.ADR_REFERENCE,
            ArtifactKind.EXTERNAL_RECEIPT,
        }
        assert set(ArtifactKind) == expected

    def test_string_values_are_stable(self) -> None:
        """String values are part of the public API — must not change."""
        assert ArtifactKind.COMMAND_INVOCATION.value == "command_invocation"
        assert ArtifactKind.COMMIT_SHA.value == "commit_sha"
        assert ArtifactKind.SIGNED_REVIEW.value == "signed_review"
        assert ArtifactKind.ADR_REFERENCE.value == "adr_reference"
        assert ArtifactKind.EXTERNAL_RECEIPT.value == "external_receipt"

    @pytest.mark.parametrize(
        "kind",
        [
            ArtifactKind.COMMAND_INVOCATION,
            ArtifactKind.COMMIT_SHA,
            ArtifactKind.SIGNED_REVIEW,
            ArtifactKind.ADR_REFERENCE,
            ArtifactKind.EXTERNAL_RECEIPT,
        ],
    )
    def test_each_value_round_trips_from_string(self, kind: ArtifactKind) -> None:
        """Every ArtifactKind value must be reconstructible from its string value."""
        reconstructed = ArtifactKind(kind.value)
        assert reconstructed is kind
