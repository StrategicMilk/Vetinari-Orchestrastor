"""Tests for vetinari/agents/contracts.py — evidence-signal and outcome contracts.

Verifies the SESSION-05/SHARD-01 additions:
  - OutcomeSignal default is UNSUPPORTED / passed=False / score=0.0 (Rule 2 compliance)
  - HUMAN_ATTESTED invariant: bare attestation raises InsufficientEvidenceError
    unless use_case='INTENT_CONFIRMATION' or attested_artifacts is non-empty
  - Every EvidenceBasis value can produce a valid OutcomeSignal
  - AttestedArtifact, ToolEvidence, LLMJudgment, Provenance construction
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from vetinari.agents.contracts import (
    AttestedArtifact,
    LLMJudgment,
    OutcomeSignal,
    Provenance,
    ToolEvidence,
)
from vetinari.exceptions import InsufficientEvidenceError
from vetinari.types import ArtifactKind, EvidenceBasis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc).isoformat()


def _make_artifact(kind: ArtifactKind = ArtifactKind.COMMIT_SHA) -> AttestedArtifact:
    """Create a minimal valid AttestedArtifact for use in tests."""
    payloads: dict[ArtifactKind, dict] = {
        ArtifactKind.COMMAND_INVOCATION: {"command": "pytest -q", "stdout_hash": "abc123", "exit_code": 0},
        ArtifactKind.COMMIT_SHA: {"repo": "https://github.com/example/repo", "sha": "deadbeef", "signed": False},
        ArtifactKind.SIGNED_REVIEW: {"reviewer_id": "reviewer-1", "signature": "sig", "reviewed_at": _NOW},
        ArtifactKind.ADR_REFERENCE: {"adr_id": "ADR-0061", "status": "accepted"},
        ArtifactKind.EXTERNAL_RECEIPT: {
            "issuer": "ci.example.com",
            "receipt_id": "run-42",
            "url": "https://ci.example.com/42",
        },
    }
    return AttestedArtifact(
        kind=kind,
        attested_by="test-user",
        attested_at_utc=_NOW,
        payload=payloads[kind],
    )


def _make_provenance(source: str = "test-suite") -> Provenance:
    return Provenance(source=source, timestamp_utc=_NOW)


# ---------------------------------------------------------------------------
# OutcomeSignal — Rule 2 fail-closed defaults
# ---------------------------------------------------------------------------


class TestOutcomeSignalDefaults:
    """OutcomeSignal() default constructor must be the fail-closed sentinel."""

    def test_default_passed_is_false(self) -> None:
        sig = OutcomeSignal()
        assert sig.passed is False

    def test_default_score_is_zero(self) -> None:
        sig = OutcomeSignal()
        assert sig.score == 0.0

    def test_default_basis_is_unsupported(self) -> None:
        sig = OutcomeSignal()
        assert sig.basis is EvidenceBasis.UNSUPPORTED

    def test_default_tool_evidence_is_empty(self) -> None:
        sig = OutcomeSignal()
        assert len(sig.tool_evidence) == 0

    def test_default_attested_artifacts_is_empty(self) -> None:
        sig = OutcomeSignal()
        assert len(sig.attested_artifacts) == 0

    def test_default_issues_is_empty(self) -> None:
        sig = OutcomeSignal()
        assert len(sig.issues) == 0

    def test_default_llm_judgment_is_none(self) -> None:
        sig = OutcomeSignal()
        assert sig.llm_judgment is None

    def test_repr_shows_key_fields(self) -> None:
        """__repr__ must expose passed, score, basis for quick inspection."""
        r = repr(OutcomeSignal())
        assert "passed=False" in r
        assert "score=0.0" in r
        assert "unsupported" in r

    def test_is_frozen(self) -> None:
        """OutcomeSignal must be immutable."""
        sig = OutcomeSignal()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            sig.passed = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OutcomeSignal — HUMAN_ATTESTED narrowing invariant
# ---------------------------------------------------------------------------


class TestOutcomeSignalHumanAttestedInvariant:
    """Bare HUMAN_ATTESTED without artifacts must raise InsufficientEvidenceError."""

    def test_bare_human_attested_raises(self) -> None:
        """Constructing HUMAN_ATTESTED with no artifacts and no use_case is forbidden."""
        with pytest.raises(InsufficientEvidenceError):
            OutcomeSignal(
                passed=True,
                score=0.9,
                basis=EvidenceBasis.HUMAN_ATTESTED,
                # attested_artifacts defaults to empty — forbidden on factual paths
            )

    def test_human_attested_with_artifact_is_valid(self) -> None:
        """HUMAN_ATTESTED with at least one AttestedArtifact must not raise."""
        artifact = _make_artifact(ArtifactKind.COMMIT_SHA)
        sig = OutcomeSignal(
            passed=True,
            score=0.95,
            basis=EvidenceBasis.HUMAN_ATTESTED,
            attested_artifacts=(artifact,),
        )
        assert sig.basis is EvidenceBasis.HUMAN_ATTESTED
        assert sig.passed is True
        assert len(sig.attested_artifacts) == 1

    def test_human_attested_intent_confirmation_allows_empty_artifacts(self) -> None:
        """use_case='INTENT_CONFIRMATION' permits bare human attestation."""
        sig = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.HUMAN_ATTESTED,
            use_case="INTENT_CONFIRMATION",
            # no attested_artifacts — valid for consent / override paths
        )
        assert sig.passed is True
        assert sig.use_case == "INTENT_CONFIRMATION"
        assert len(sig.attested_artifacts) == 0

    def test_unsupported_basis_allows_empty_artifacts(self) -> None:
        """Default UNSUPPORTED basis must not trigger the HUMAN_ATTESTED check."""
        sig = OutcomeSignal()
        assert sig.basis is EvidenceBasis.UNSUPPORTED

    def test_tool_evidence_basis_allows_empty_artifacts(self) -> None:
        """TOOL_EVIDENCE basis must not trigger the HUMAN_ATTESTED check."""
        te = ToolEvidence(tool_name="pytest", command="pytest -q", exit_code=0, passed=True)
        sig = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            tool_evidence=(te,),
        )
        assert sig.basis is EvidenceBasis.TOOL_EVIDENCE
        assert sig.passed is True


# ---------------------------------------------------------------------------
# OutcomeSignal — every EvidenceBasis produces a valid signal
# ---------------------------------------------------------------------------


class TestOutcomeSignalAllBases:
    """Parametrized test: every EvidenceBasis must be usable in a valid OutcomeSignal."""

    @pytest.mark.parametrize(
        "basis, extra_kwargs",
        [
            (
                EvidenceBasis.TOOL_EVIDENCE,
                {"tool_evidence": (ToolEvidence(tool_name="ruff", command="ruff check .", exit_code=0, passed=True),)},
            ),
            (
                EvidenceBasis.LLM_JUDGMENT,
                {"llm_judgment": LLMJudgment(model_id="qwen-7b", summary="Looks correct.", score=0.85)},
            ),
            (
                EvidenceBasis.HUMAN_ATTESTED,
                {
                    "attested_artifacts": (
                        AttestedArtifact(
                            kind=ArtifactKind.COMMIT_SHA,
                            attested_by="alice",
                            attested_at_utc=_NOW,
                            payload={"repo": "https://github.com/x/y", "sha": "abc123", "signed": True},
                        ),
                    )
                },
            ),
            (
                EvidenceBasis.HYBRID,
                {
                    "tool_evidence": (ToolEvidence(tool_name="pytest", command="pytest -q", exit_code=0, passed=True),),
                    "llm_judgment": LLMJudgment(
                        model_id="qwen-7b", summary="Tests pass and logic is sound.", score=0.9
                    ),
                },
            ),
            (EvidenceBasis.UNSUPPORTED, {}),  # default / no-evidence sentinel
        ],
    )
    def test_valid_signal_for_each_basis(self, basis: EvidenceBasis, extra_kwargs: dict) -> None:
        """A correctly-constructed OutcomeSignal must be accepted for every basis value."""
        sig = OutcomeSignal(
            passed=basis is not EvidenceBasis.UNSUPPORTED,
            score=0.0 if basis is EvidenceBasis.UNSUPPORTED else 0.8,
            basis=basis,
            **extra_kwargs,
        )
        assert sig.basis is basis
        assert isinstance(sig.passed, bool)
        assert 0.0 <= sig.score <= 1.0


# ---------------------------------------------------------------------------
# AttestedArtifact construction
# ---------------------------------------------------------------------------


class TestAttestedArtifact:
    """AttestedArtifact must be constructible for every ArtifactKind."""

    @pytest.mark.parametrize("kind", list(ArtifactKind))
    def test_construction_for_every_kind(self, kind: ArtifactKind) -> None:
        """Every ArtifactKind produces a valid AttestedArtifact."""
        artifact = _make_artifact(kind)
        assert artifact.kind is kind
        assert artifact.attested_by == "test-user"
        assert artifact.attested_at_utc == _NOW
        assert isinstance(artifact.payload, dict)

    def test_repr_shows_kind_and_attester(self) -> None:
        artifact = _make_artifact(ArtifactKind.ADR_REFERENCE)
        r = repr(artifact)
        assert "adr_reference" in r
        assert "test-user" in r

    def test_is_frozen(self) -> None:
        artifact = _make_artifact()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            artifact.attested_by = "intruder"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ToolEvidence construction
# ---------------------------------------------------------------------------


class TestToolEvidence:
    """ToolEvidence carries tool name, command, exit code, and pass flag."""

    def test_basic_construction(self) -> None:
        te = ToolEvidence(tool_name="pytest", command="pytest tests/ -q", exit_code=0, passed=True)
        assert te.tool_name == "pytest"
        assert te.exit_code == 0
        assert te.passed is True

    def test_repr_shows_tool_exit_and_passed(self) -> None:
        te = ToolEvidence(tool_name="ruff", command="ruff check .", exit_code=1, passed=False)
        r = repr(te)
        assert "ruff" in r
        assert "exit_code=1" in r
        assert "passed=False" in r

    def test_failed_tool_has_passed_false_by_default(self) -> None:
        te = ToolEvidence(tool_name="mypy", command="mypy vetinari/", exit_code=1)
        assert te.passed is False


# ---------------------------------------------------------------------------
# LLMJudgment construction
# ---------------------------------------------------------------------------


class TestLLMJudgment:
    """LLMJudgment captures model id, summary, score, and optional reasoning."""

    def test_basic_construction(self) -> None:
        j = LLMJudgment(model_id="qwen2.5-72b", summary="Output is correct and well-structured.", score=0.92)
        assert j.model_id == "qwen2.5-72b"
        assert j.score == 0.92
        assert j.summary.startswith("Output is correct")

    def test_repr_truncates_summary(self) -> None:
        long_summary = "A" * 200
        j = LLMJudgment(model_id="m", summary=long_summary, score=0.5)
        r = repr(j)
        # repr truncates summary to 60 chars
        assert long_summary not in r
        assert "A" * 60 in r or "A" * 59 in r  # allow for quote chars


# ---------------------------------------------------------------------------
# Provenance construction
# ---------------------------------------------------------------------------


class TestProvenance:
    """Provenance records source and timestamp with optional tool/model/human fields."""

    def test_minimal_construction(self) -> None:
        p = _make_provenance()
        assert p.source == "test-suite"
        assert p.model_id == ""
        assert p.tool_name == ""
        assert p.attested_by == ""

    def test_tool_provenance(self) -> None:
        p = Provenance(source="pytest-runner", timestamp_utc=_NOW, tool_name="pytest", tool_version="8.1.0")
        assert p.tool_name == "pytest"
        assert p.tool_version == "8.1.0"

    def test_repr_shows_source_and_timestamp(self) -> None:
        p = _make_provenance("ci-pipeline")
        r = repr(p)
        assert "ci-pipeline" in r
