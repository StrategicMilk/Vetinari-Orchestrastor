"""Regression tests for vetinari.verification.entailment_checker fail-closed semantics.

Proves Rule 2 compliance: when no entailment model is loaded, factual claims and
unverifiable structural claims MUST return trust_score=0.0 with method="stub",
not a neutral default-pass score.

This is a separate file from tests/test_entailment_checker.py (which exercises the
unrelated vetinari.validation.entailment_checker module with the EntailmentChecker
class). Two modules share the 'entailment_checker' name and must not be conflated.

Part of SESSION-01 SHARD-01 (governance theater closure).
"""

from __future__ import annotations

import pytest

from vetinari.verification.claim_extractor import Claim, ClaimType
from vetinari.verification.entailment_checker import (
    ClaimVerdict,
    verify_claim,
    verify_claims,
)


def _make_claim(
    claim_type: ClaimType,
    text: str = "placeholder claim text",
    confidence: float = 0.9,
    code_snippet: str | None = None,
) -> Claim:
    """Build a Claim with sensible defaults for a test case.

    Args:
        claim_type: Claim classification driving verification strategy.
        text: Claim text — long enough to satisfy prefix matching when needed.
        confidence: Extraction confidence in [0.0, 1.0].
        code_snippet: Optional code snippet attached to the claim.

    Returns:
        Fully-initialized Claim dataclass instance.
    """
    return Claim(
        text=text,
        claim_type=claim_type,
        source_span=(0, len(text)),
        confidence=confidence,
        code_snippet=code_snippet,
    )


class TestFactualClaimFailsClosed:
    """Rule 2: factual claims without a loaded entailment model MUST fail closed."""

    def test_factual_claim_returns_trust_zero(self) -> None:
        """Without an entailment model, factual verdicts MUST report trust_score=0.0."""
        claim = _make_claim(ClaimType.FACTUAL, text="The algorithm has O(n log n) complexity.")

        verdict = verify_claim(claim)

        assert isinstance(verdict, ClaimVerdict)
        assert verdict.trust_score == 0.0, (
            f"Rule 2 violation: factual claim returned trust_score={verdict.trust_score}. "
            "Without a loaded entailment model the verdict MUST fail closed at 0.0."
        )
        assert verdict.method == "stub", (
            f"Factual claim returned method={verdict.method!r}, expected 'stub'. "
            "Method must flag the verdict as a stub so callers can distinguish it."
        )

    def test_factual_explanation_states_unsupported(self) -> None:
        """Explanation MUST state the claim is unsupported, not 'neutral' or 'half-true'."""
        claim = _make_claim(ClaimType.FACTUAL, text="This approach is theoretically optimal.")

        verdict = verify_claim(claim)

        explanation_lower = verdict.explanation.lower()
        assert "unsupported" in explanation_lower or "not loaded" in explanation_lower, (
            f"Factual claim explanation {verdict.explanation!r} must name the missing model "
            "and the unsupported outcome."
        )
        assert "neutral" not in explanation_lower, (
            "Explanation must not frame an unverified factual claim as 'neutral'."
        )

    def test_factual_emits_warning_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """A WARNING log line MUST fire so operators know the model is absent."""
        claim = _make_claim(ClaimType.FACTUAL, text="The bayesian update converges in O(1).")

        with caplog.at_level("WARNING", logger="vetinari.verification.entailment_checker"):
            verify_claim(claim)

        assert any("entailment model unavailable" in rec.getMessage().lower() for rec in caplog.records), (
            "WARNING log must explicitly name the missing entailment model and its impact. "
            f"Captured: {[rec.getMessage() for rec in caplog.records]}"
        )


class TestStructuralClaimUnknownCountFailsClosed:
    """Structural claims without extractable counts MUST fail closed, not default to 0.5."""

    def test_structural_claim_without_count_returns_trust_zero(self) -> None:
        """When no count can be extracted from the claim, trust MUST be 0.0."""
        # No numeric count phrase — the structural checker falls back to unverified.
        claim = _make_claim(
            ClaimType.STRUCTURAL,
            text="The output is well organised across several parts.",
        )

        verdict = verify_claim(claim)

        assert verdict.trust_score == 0.0
        assert verdict.method == "stub"

    def test_structural_claim_with_count_but_no_snippet_fails_closed(self) -> None:
        """Count present but no evidence — trust MUST be 0.0, not 0.5."""
        claim = _make_claim(
            ClaimType.STRUCTURAL,
            text="The response contains 3 sections covering the topic.",
            code_snippet=None,
        )

        verdict = verify_claim(claim)

        assert verdict.trust_score == 0.0, (
            "Rule 2 violation: structural claim with an expected count but no content "
            f"returned trust_score={verdict.trust_score}. It MUST fail closed at 0.0."
        )
        assert verdict.method == "stub"


class TestAggregateTrustFailsClosed:
    """verify_claims() MUST NOT return overall_trust=0.5 when confidence evidence is absent."""

    def test_empty_claims_returns_trust_zero(self) -> None:
        """No claims to verify yields zero trust, not 1.0 and not 0.5."""
        report = verify_claims([])

        assert report.total_claims == 0
        assert report.overall_trust == 0.0

    def test_zero_confidence_claims_return_trust_zero(self) -> None:
        """Claims with confidence=0.0 carry no weight — overall_trust MUST be 0.0."""
        claim = _make_claim(ClaimType.FACTUAL, text="Zero-confidence factual claim.", confidence=0.0)

        report = verify_claims([claim])

        assert report.total_claims == 1
        assert report.overall_trust == 0.0, (
            f"Rule 2 violation: zero-confidence claims returned overall_trust={report.overall_trust}. "
            "Without confidence weight the aggregate MUST fail closed at 0.0, not default to 0.5."
        )


class TestDeterministicBranchStillReturnsPositiveEvidence:
    """Positive evidence paths MUST still return a passing verdict — the gate is a real discriminator."""

    def test_valid_python_code_claim_returns_high_trust(self) -> None:
        """A CODE claim with a valid snippet and 'valid' assertion returns trust_score=1.0."""
        claim = _make_claim(
            ClaimType.CODE,
            text="The function is valid Python.",
            code_snippet="def add(a, b):\n    return a + b\n",
        )

        verdict = verify_claim(claim)

        assert verdict.trust_score == 1.0, (
            "Valid-code CODE claim should return trust_score=1.0. The gate is a discriminator, not always-false."
        )
        assert verdict.method == "ast"

    def test_invalid_python_code_claim_returns_low_trust(self) -> None:
        """A CODE claim asserting validity but with a syntax error returns trust_score=0.0."""
        claim = _make_claim(
            ClaimType.CODE,
            text="The function is valid Python.",
            code_snippet="def add(a, b:\n    return a + b\n",  # missing paren
        )

        verdict = verify_claim(claim)

        assert verdict.trust_score == 0.0
        assert verdict.method == "ast"
