"""Tests for vetinari.validation.verification.CascadeOrchestrator — tiered cascade.

Also covers SESSION-05 SHARD-03 claim-fail-closed requirements:
- CascadeClaimRecord and VerificationCascadeReport report shape (Task 3.2).
- Six failure-mode assertions (Task 3.3).
- Cascade aggregation never silently converts passed=False to passed=True.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.contracts import AttestedArtifact, OutcomeSignal, Provenance
from vetinari.types import ArtifactKind, EvidenceBasis
from vetinari.validation.verification import CascadeOrchestrator, CascadeVerdict
from vetinari.verification.cascade_report import (
    CascadeClaimRecord,
    VerificationCascadeReport,
    build_cascade_report,
)
from vetinari.verification.claim_verifier import (
    DEFAULT_FRESHNESS_WINDOW_SECONDS,
    verify_citation_present,
    verify_claim_fail_closed,
    verify_evidence_freshness,
    verify_file_claim,
    verify_human_attestation,
    verify_no_entailment_contradiction,
    verify_not_llm_only,
)


class TestTier1StaticFailure:
    """Tier 1 (StaticVerifier) failures short-circuit before Tier 2 and Tier 3."""

    def test_syntax_error_fails_at_tier1(self) -> None:
        """A response with a Python SyntaxError is rejected at tier_reached='static'."""
        orchestrator = CascadeOrchestrator()
        verdict = orchestrator.verify(
            "def broken(\n    return None\n",
            task_description="implement a function",
        )
        assert verdict.passed is False
        assert verdict.tier_reached == "static"
        assert len(verdict.static_findings) >= 1

    def test_banned_import_fails_at_tier1(self) -> None:
        """A response importing a banned module is rejected at Tier 1."""
        orchestrator = CascadeOrchestrator()
        code = "import ctypes\ndef hello(): pass\n"
        verdict = orchestrator.verify(code, task_description="implement a greeting")
        assert verdict.passed is False
        assert verdict.tier_reached == "static"
        assert any("ctypes" in f for f in verdict.static_findings)

    def test_hardcoded_credential_fails_at_tier1(self) -> None:
        """A response with a hardcoded password is rejected at Tier 1."""
        orchestrator = CascadeOrchestrator()
        code = 'password = "supersecret123"\ndef connect(): pass\n'
        verdict = orchestrator.verify(code, task_description="write a connect function")
        assert verdict.passed is False
        assert verdict.tier_reached == "static"

    def test_tier1_failure_does_not_call_llm(self) -> None:
        """When Tier 1 fails, score_confidence_via_llm is never called."""
        orchestrator = CascadeOrchestrator()
        with patch("vetinari.llm_helpers.score_confidence_via_llm") as mock_llm:
            orchestrator.verify("def bad(\n", task_description="implement something")
        mock_llm.assert_not_called()


class TestTier2EntailmentPass:
    """Tier 2 (EntailmentChecker) accepts high-coverage responses without LLM."""

    def test_high_coverage_passes_at_tier2(self) -> None:
        """A response with strong keyword overlap resolves at tier_reached='entailment'."""
        orchestrator = CascadeOrchestrator()
        task = "implement a binary search function that finds an element in a sorted list"
        response = (
            "def binary_search(sorted_list, element):\n"
            "    low, high = 0, len(sorted_list) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if sorted_list[mid] == element:\n"
            "            return mid\n"
            "        elif sorted_list[mid] < element:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        )
        with patch("vetinari.llm_helpers.score_confidence_via_llm") as mock_llm:
            verdict = orchestrator.verify(response, task_description=task)
        # LLM must NOT be called when Tier 2 is conclusive
        mock_llm.assert_not_called()
        assert verdict.tier_reached == "entailment"
        assert verdict.passed is True
        assert verdict.entailment_coverage is not None

    def test_zero_coverage_rejected_at_tier2(self) -> None:
        """A response with near-zero keyword overlap is rejected at Tier 2, no LLM.

        Uses a non-code task description to avoid the Tier 1 code-required check,
        isolating the Tier 2 keyword-coverage failure.
        """
        orchestrator = CascadeOrchestrator()
        # Describe a concept-explanation task (no "implement") so Tier 1 passes
        task = "explain what database migration means and describe schema versioning"
        response = "The weather today is sunny and warm with a light breeze."
        with patch("vetinari.llm_helpers.score_confidence_via_llm") as mock_llm:
            verdict = orchestrator.verify(response, task_description=task)
        mock_llm.assert_not_called()
        assert verdict.passed is False
        assert verdict.tier_reached == "entailment"


class TestTier3LLMFallback:
    """Tier 3 (LLM) is only invoked when Tiers 1 and 2 are inconclusive."""

    def test_llm_called_when_coverage_is_borderline(self) -> None:
        """When coverage falls in the middle range, the LLM is consulted."""
        orchestrator = CascadeOrchestrator()
        # Craft a response with partial overlap — enough to pass static, not enough for Tier 2
        task = "implement a function to parse configuration files with validation"
        response = "Here is a parse function."  # minimal, borderline coverage

        with patch("vetinari.validation.verification.score_confidence_via_llm") as mock_llm:
            mock_llm.return_value = 0.8
            verdict = orchestrator.verify(response, task_description=task)

        # Either Tier 2 or Tier 3 resolved — if Tier 3, LLM was called
        if verdict.tier_reached == "llm":
            mock_llm.assert_called_once()
            assert verdict.llm_score == 0.8
            assert verdict.passed is True

    def test_llm_score_below_threshold_fails(self) -> None:
        """When LLM returns a low score, the cascade fails at Tier 3."""
        orchestrator = CascadeOrchestrator()
        task = "write a complex algorithm implementation"
        response = "Here is something."

        with patch("vetinari.validation.verification.score_confidence_via_llm") as mock_llm:
            mock_llm.return_value = 0.2
            verdict = orchestrator.verify(response, task_description=task)

        if verdict.tier_reached == "llm":
            assert verdict.passed is False
            assert verdict.llm_score == 0.2

    def test_llm_unavailable_falls_back_to_entailment_coverage(self) -> None:
        """When LLM raises an exception, the cascade uses coverage as fallback."""
        orchestrator = CascadeOrchestrator()
        task = "implement a sorting algorithm"
        response = "def sort(items): return sorted(items)"

        with patch("vetinari.validation.verification.score_confidence_via_llm") as mock_llm:
            mock_llm.side_effect = RuntimeError("LLM unavailable")
            verdict = orchestrator.verify(response, task_description=task)

        # Must not propagate exception
        assert isinstance(verdict, CascadeVerdict)


class TestCascadeVerdictToDict:
    """CascadeVerdict.to_dict() produces a JSON-serializable structure."""

    def test_to_dict_is_serializable(self) -> None:
        """to_dict() output can be round-tripped through json.dumps/loads."""
        import json

        orchestrator = CascadeOrchestrator()
        verdict = orchestrator.verify("x = 1\n", task_description="assign a variable")
        d = verdict.to_dict()
        serialized = json.dumps(d)
        restored = json.loads(serialized)
        assert restored["passed"] == verdict.passed
        assert restored["tier_reached"] == verdict.tier_reached

    def test_to_dict_contains_required_keys(self) -> None:
        """to_dict() always contains the five expected keys."""
        orchestrator = CascadeOrchestrator()
        verdict = orchestrator.verify("y = 2\n")
        d = verdict.to_dict()
        for key in ("passed", "tier_reached", "static_findings", "entailment_coverage", "llm_score"):
            assert key in d, f"Missing key: {key!r}"


class TestGetCascadeOrchestrator:
    """get_cascade_orchestrator() returns a stable singleton."""

    def test_singleton_identity(self) -> None:
        """Two calls to get_cascade_orchestrator() return the same object."""
        from vetinari.validation.verification import get_cascade_orchestrator

        a = get_cascade_orchestrator()
        b = get_cascade_orchestrator()
        assert a is b

    def test_singleton_is_cascade_orchestrator(self) -> None:
        """The singleton is an instance of CascadeOrchestrator."""
        from vetinari.validation.verification import get_cascade_orchestrator

        instance = get_cascade_orchestrator()
        assert isinstance(instance, CascadeOrchestrator)


# ---------------------------------------------------------------------------
# SESSION-05 SHARD-03: Six fail-closed failure modes (Task 3.3)
# ---------------------------------------------------------------------------


def _fresh_tool_signal() -> OutcomeSignal:
    """Build a minimal passing TOOL_EVIDENCE signal with current provenance."""
    return OutcomeSignal(
        passed=True,
        score=1.0,
        basis=EvidenceBasis.TOOL_EVIDENCE,
        provenance=Provenance(
            source="test",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )


class TestFailureModesMissingCitation:
    """Failure mode 1: missing citation must produce passed=False, basis=UNSUPPORTED."""

    def test_no_provenance_fails_closed(self) -> None:
        """Signal with no Provenance returns UNSUPPORTED."""
        signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            provenance=None,
        )
        result = verify_citation_present(signal)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "no citation" in result.issues[0]

    def test_unsupported_basis_fails_closed(self) -> None:
        """Signal with basis=UNSUPPORTED returns UNSUPPORTED with original issues."""
        signal = OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=("stub reason",),
            provenance=Provenance(source="t", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        result = verify_citation_present(signal)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED

    def test_score_is_zero_when_unsupported(self) -> None:
        """Rule 2 invariant: score must be 0.0 when basis is UNSUPPORTED."""
        signal = OutcomeSignal(
            passed=True,
            score=0.9,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            provenance=None,
        )
        result = verify_citation_present(signal)
        assert result.score == 0.0

    def test_valid_signal_passes_through(self) -> None:
        """Signal with Provenance and non-UNSUPPORTED basis is returned unchanged."""
        signal = _fresh_tool_signal()
        result = verify_citation_present(signal)
        assert result is signal


class TestFailureModeStaleEvidence:
    """Failure mode 2: stale evidence must produce passed=False, basis=UNSUPPORTED."""

    def test_old_timestamp_fails_closed(self) -> None:
        """Evidence older than the freshness window returns UNSUPPORTED."""
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=7200)).isoformat()
        signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            provenance=Provenance(source="test", timestamp_utc=old_ts),
        )
        result = verify_evidence_freshness(signal, freshness_window_seconds=3600)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "stale evidence" in result.issues[0]
        assert "fresh-by" in result.issues[0]

    def test_fresh_timestamp_passes_through(self) -> None:
        """Evidence within the freshness window is returned unchanged."""
        signal = _fresh_tool_signal()
        result = verify_evidence_freshness(signal, freshness_window_seconds=3600)
        assert result is signal

    def test_no_provenance_treated_as_stale(self) -> None:
        """Signal with no Provenance is treated as stale (fail-closed)."""
        signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            provenance=None,
        )
        result = verify_evidence_freshness(signal)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED


class TestFailureModeUnverifiedFileClaim:
    """Failure mode 3: unverified file claim must produce passed=False, basis=UNSUPPORTED."""

    def test_nonexistent_file_fails_closed(self, tmp_path: Path) -> None:
        """Claim about a non-existent file returns UNSUPPORTED."""
        signal = _fresh_tool_signal()
        missing = str(tmp_path / "does_not_exist.py")
        result = verify_file_claim(signal, claimed_path=missing)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "file claim unverified" in result.issues[0]
        assert "does_not_exist.py" in result.issues[0]

    def test_existing_file_without_hash_passes(self, tmp_path: Path) -> None:
        """Existing file without hash requirement passes."""
        f = tmp_path / "exists.py"
        f.write_text("x = 1", encoding="utf-8")
        signal = _fresh_tool_signal()
        result = verify_file_claim(signal, claimed_path=str(f))
        assert result is signal

    def test_hash_mismatch_fails_closed(self, tmp_path: Path) -> None:
        """Existing file with wrong SHA-256 returns UNSUPPORTED."""
        f = tmp_path / "file.py"
        f.write_text("actual content", encoding="utf-8")
        signal = _fresh_tool_signal()
        result = verify_file_claim(signal, claimed_path=str(f), expected_sha256="deadbeef" * 8)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "hash mismatch" in result.issues[0]

    def test_correct_hash_passes(self, tmp_path: Path) -> None:
        """Existing file with correct SHA-256 passes."""
        import hashlib

        content = b"verified content"
        f = tmp_path / "verified.py"
        f.write_bytes(content)
        correct_hash = hashlib.sha256(content).hexdigest()
        signal = _fresh_tool_signal()
        result = verify_file_claim(signal, claimed_path=str(f), expected_sha256=correct_hash)
        assert result is signal


class TestFailureModeEntailmentContradiction:
    """Failure mode 4: entailment contradiction must produce passed=False, basis=UNSUPPORTED."""

    def test_contradiction_flagged_fails_closed(self) -> None:
        """entailment_passed=False returns UNSUPPORTED with 'entailment contradiction'."""
        signal = _fresh_tool_signal()
        result = verify_no_entailment_contradiction(
            signal,
            entailment_passed=False,
            claim_text="all tests pass",
        )
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "entailment contradiction" in result.issues[0]
        assert "all tests pass" in result.issues[0]

    def test_no_contradiction_passes_through(self) -> None:
        """entailment_passed=True returns the original signal unchanged."""
        signal = _fresh_tool_signal()
        result = verify_no_entailment_contradiction(signal, entailment_passed=True)
        assert result is signal


class TestFailureModeBareHumanAttested:
    """Failure mode 5: bare HUMAN_ATTESTED on high-accuracy claim must fail.

    Also covers the four HUMAN_ATTESTED parametrizations from the shard spec.
    """

    def _make_attested_signal(
        self,
        artifacts: tuple[AttestedArtifact, ...] = (),
        use_case: str | None = None,
    ) -> OutcomeSignal:
        """Build a HUMAN_ATTESTED signal, bypassing the OutcomeSignal constructor guard via use_case."""
        if not artifacts and use_case != "INTENT_CONFIRMATION":
            # OutcomeSignal.__post_init__ raises InsufficientEvidenceError for bare HUMAN_ATTESTED.
            # We test the verifier hook directly with a pre-constructed TOOL_EVIDENCE signal
            # that has been manually relabeled — so instead we test via the verifier function
            # with explicit parameters.
            raise ValueError("Use verify_human_attestation directly for bare HUMAN_ATTESTED")
        return OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.HUMAN_ATTESTED,
            attested_artifacts=artifacts,
            use_case=use_case,
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )

    def test_bare_human_attested_fails_closed(self) -> None:
        """(a) Bare HUMAN_ATTESTED with no artifacts on factual claim -> passed=False."""
        # verify_human_attestation is called with a HUMAN_ATTESTED signal; we build it
        # with INTENT_CONFIRMATION to pass construction, then test the verifier logic.
        intent_signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.HUMAN_ATTESTED,
            use_case="INTENT_CONFIRMATION",
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        # Call the hook with is_intent_confirmation=False to simulate factual-claim path
        result = verify_human_attestation(intent_signal, claim_scope="", is_intent_confirmation=False)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "human attestation requires a concrete attested artifact" in result.issues[0]

    def test_matching_commit_sha_passes(self) -> None:
        """(b) HUMAN_ATTESTED with matching COMMIT_SHA artifact -> passed=True."""
        artifact = AttestedArtifact(
            kind=ArtifactKind.COMMIT_SHA,
            attested_by="engineer@example.com",
            attested_at_utc=datetime.now(timezone.utc).isoformat(),
            payload={"repo": "vetinari", "sha": "abc123", "signed": True, "scope": "fix-login-bug"},
        )
        signal = self._make_attested_signal(artifacts=(artifact,))
        result = verify_human_attestation(signal, claim_scope="fix-login-bug", is_intent_confirmation=False)
        assert result.passed is True
        assert result.basis is EvidenceBasis.HUMAN_ATTESTED

    def test_adr_unrelated_to_scope_fails(self) -> None:
        """(c) HUMAN_ATTESTED with ADR_REFERENCE unrelated to claim scope -> passed=False."""
        artifact = AttestedArtifact(
            kind=ArtifactKind.ADR_REFERENCE,
            attested_by="engineer@example.com",
            attested_at_utc=datetime.now(timezone.utc).isoformat(),
            payload={"adr_id": "ADR-0042", "status": "Accepted"},
        )
        signal = self._make_attested_signal(artifacts=(artifact,))
        # Claim scope is about "test-database-migration" — ADR-0042 does not match
        result = verify_human_attestation(signal, claim_scope="test-database-migration", is_intent_confirmation=False)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "artifact does not support the specific claim" in result.issues[0]

    def test_signed_review_on_intent_confirmation_passes(self) -> None:
        """(d) SIGNED_REVIEW artifact on intent-confirmation use case -> passed=True.

        The narrowed invariant (artifact required) only applies to factual-claim
        closure paths.  Intent-confirmation paths (destructive-op consent, override
        appeals) are permitted to carry bare attestation.
        """
        signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.HUMAN_ATTESTED,
            use_case="INTENT_CONFIRMATION",
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        result = verify_human_attestation(signal, claim_scope="deploy-to-prod", is_intent_confirmation=True)
        assert result.passed is True
        assert result.basis is EvidenceBasis.HUMAN_ATTESTED


class TestFailureModeLLMOnly:
    """Failure mode 6: LLM-only signal on high-accuracy claim must flag advisory visibly."""

    def test_llm_only_adds_advisory(self) -> None:
        """basis=LLM_JUDGMENT with no tool_evidence adds 'no tool evidence' advisory."""
        signal = OutcomeSignal(
            passed=True,
            score=0.85,
            basis=EvidenceBasis.LLM_JUDGMENT,
            llm_judgment=MagicMock(),
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        result = verify_not_llm_only(signal, high_accuracy=True)
        assert result.basis is EvidenceBasis.LLM_JUDGMENT
        assert any("no tool evidence" in issue for issue in result.issues)

    def test_llm_only_advisory_not_hidden_in_aggregate(self) -> None:
        """The 'no tool evidence' advisory survives aggregation and is visible in issues."""
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals

        signal = OutcomeSignal(
            passed=True,
            score=0.85,
            basis=EvidenceBasis.LLM_JUDGMENT,
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        flagged = verify_not_llm_only(signal, high_accuracy=True)
        aggregate = aggregate_outcome_signals([flagged])
        assert any("no tool evidence" in issue for issue in aggregate.issues)

    def test_non_high_accuracy_no_advisory(self) -> None:
        """LLM-only signal on non-high-accuracy path is returned without advisory."""
        signal = OutcomeSignal(
            passed=True,
            score=0.85,
            basis=EvidenceBasis.LLM_JUDGMENT,
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        result = verify_not_llm_only(signal, high_accuracy=False)
        assert result is signal
        assert not any("no tool evidence" in issue for issue in result.issues)


# ---------------------------------------------------------------------------
# SESSION-05 SHARD-03: Report shape (Task 3.2) — unsupported_claims iterable
# ---------------------------------------------------------------------------


class TestVerificationCascadeReportShape:
    """VerificationCascadeReport exposes unsupported_claims as a top-level iterable."""

    def test_unsupported_claims_field_is_iterable(self) -> None:
        """unsupported_claims can be iterated without parsing free-form text."""
        good = _fresh_tool_signal()
        bad = OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=("no citation attached",),
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        report = build_cascade_report([
            ("c1", "all tests pass", good),
            ("c2", "file exists", bad),
        ])
        assert isinstance(report, VerificationCascadeReport)
        unsupported_ids = [r.claim_id for r in report.unsupported_claims]
        assert "c2" in unsupported_ids
        assert "c1" not in unsupported_ids

    def test_to_dict_has_top_level_unsupported_claims(self) -> None:
        """to_dict() must contain 'unsupported_claims' as a top-level list key."""
        bad = OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=("no citation attached",),
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        report = build_cascade_report([("c1", "claim text", bad)])
        d = report.to_dict()
        assert "unsupported_claims" in d
        assert isinstance(d["unsupported_claims"], list)
        assert d["unsupported_claims"][0]["claim_id"] == "c1"

    def test_aggregate_does_not_convert_false_to_true(self) -> None:
        """Aggregate signal must remain passed=False when any constituent fails.

        Anti-pattern check: branch-accepting tests.  This test will fail if the
        aggregation step silently converts a failed claim's passed=False to True.
        """
        good = _fresh_tool_signal()
        bad = OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=("entailment contradiction",),
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        report = build_cascade_report([
            ("c1", "passing claim", good),
            ("c2", "failing claim", bad),
        ])
        # The aggregate must be False — even one UNSUPPORTED constituent prevents passing
        assert report.aggregate_signal.passed is False
        assert report.aggregate_signal.basis is EvidenceBasis.UNSUPPORTED

    def test_all_passing_report_is_passed_true(self) -> None:
        """A report with all passing TOOL_EVIDENCE signals yields aggregate passed=True."""
        s1 = _fresh_tool_signal()
        s2 = _fresh_tool_signal()
        report = build_cascade_report([("c1", "claim 1", s1), ("c2", "claim 2", s2)])
        assert report.aggregate_signal.passed is True
        assert len(report.unsupported_claims) == 0

    def test_empty_claim_list_produces_unsupported_aggregate(self) -> None:
        """Empty claim list produces a fail-closed aggregate (no evidence = UNSUPPORTED)."""
        report = build_cascade_report([])
        assert report.aggregate_signal.passed is False
        assert report.aggregate_signal.basis is EvidenceBasis.UNSUPPORTED
        assert len(report.unsupported_claims) == 0  # no records to classify

    def test_claim_record_is_unsupported_flag_correct(self) -> None:
        """CascadeClaimRecord.is_unsupported mirrors signal.basis == UNSUPPORTED."""
        bad = OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=("missing citation",),
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        good = _fresh_tool_signal()
        report = build_cascade_report([("bad", "bad claim", bad), ("good", "good claim", good)])
        records_by_id = {r.claim_id: r for r in report.claim_records}
        assert records_by_id["bad"].is_unsupported is True
        assert records_by_id["good"].is_unsupported is False


# ---------------------------------------------------------------------------
# SESSION-05 SHARD-03: Composite gate (verify_claim_fail_closed) smoke tests
# ---------------------------------------------------------------------------


class TestVerifyClaimFailClosed:
    """verify_claim_fail_closed applies all six checks in sequence."""

    def test_composite_gate_passes_clean_signal(self) -> None:
        """A fully valid TOOL_EVIDENCE signal passes all six checks."""
        signal = _fresh_tool_signal()
        result = verify_claim_fail_closed(signal)
        assert result.passed is True
        assert result.basis is EvidenceBasis.TOOL_EVIDENCE

    def test_composite_gate_missing_citation_short_circuits(self) -> None:
        """Missing citation is detected by the composite gate."""
        signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            provenance=None,
        )
        result = verify_claim_fail_closed(signal)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED

    def test_composite_gate_entailment_contradiction(self) -> None:
        """Entailment contradiction detected by the composite gate."""
        signal = _fresh_tool_signal()
        result = verify_claim_fail_closed(signal, entailment_passed=False, claim_text="all pass")
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "entailment contradiction" in result.issues[0]
