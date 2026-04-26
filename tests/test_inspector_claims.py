"""Tests for vetinari.verification.claim_extractor and entailment_checker.

Verifies claim extraction from Worker output, claim classification,
deterministic verification (AST, regex, structural), and aggregate
trust scoring.

Part of US-013: Claim-Level Verification Pipeline.
"""

from __future__ import annotations

import pytest

from vetinari.verification.claim_extractor import (
    Claim,
    ClaimType,
    extract_claims,
    extract_claims_by_type,
)
from vetinari.verification.entailment_checker import (
    ClaimVerdict,
    VerificationReport,
    verify_claim,
    verify_claims,
)

# -- Claim extraction: code blocks -------------------------------------------


class TestCodeBlockExtraction:
    """Tests for extracting CODE claims from fenced code blocks."""

    def test_valid_python_code_block_produces_code_claim(self) -> None:
        """A fenced Python code block must produce a CODE claim with high confidence."""
        text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\n"
        claims = extract_claims(text)

        code_claims = [c for c in claims if c.claim_type == ClaimType.CODE]
        assert len(code_claims) == 1
        assert code_claims[0].code_snippet == "def add(a, b):\n    return a + b"
        assert "valid" in code_claims[0].text.lower()
        assert code_claims[0].confidence == pytest.approx(0.95)

    def test_invalid_python_code_block_flagged(self) -> None:
        """A code block with syntax errors must be flagged as containing errors."""
        text = "```python\ndef broken(\n```\n"
        claims = extract_claims(text)

        code_claims = [c for c in claims if c.claim_type == ClaimType.CODE]
        assert len(code_claims) == 1
        assert "error" in code_claims[0].text.lower() or "invalid" in code_claims[0].text.lower()
        assert code_claims[0].confidence == pytest.approx(0.8)

    def test_multiple_code_blocks_each_produce_claim(self) -> None:
        """Each fenced code block must produce its own CODE claim."""
        text = "First:\n```python\nx = 1\n```\nSecond:\n```python\ny = 2\n```\n"
        claims = extract_claims(text)
        code_claims = [c for c in claims if c.claim_type == ClaimType.CODE]
        assert len(code_claims) == 2

    def test_unfenced_code_not_extracted_as_code_block(self) -> None:
        """Inline code without fences must not produce CODE claims from the code-block pass."""
        text = "The variable x = 1 is assigned correctly."
        claims = extract_claims(text)
        code_claims = [c for c in claims if c.claim_type == ClaimType.CODE]
        assert len(code_claims) == 0


# -- Claim extraction: prose classification ----------------------------------


class TestProseClassification:
    """Tests for classifying prose sentences into claim types."""

    def test_function_return_classified_as_code(self) -> None:
        """A sentence about a function returning a value must be CODE."""
        text = "The function process_data returns a list of results."
        claims = extract_claims(text)
        assert len(claims) >= 1
        assert claims[0].claim_type == ClaimType.CODE

    def test_valid_json_classified_as_format(self) -> None:
        """A sentence about valid JSON must be FORMAT."""
        text = "The output is valid JSON conforming to the schema."
        claims = extract_claims(text)
        format_claims = [c for c in claims if c.claim_type == ClaimType.FORMAT]
        assert len(format_claims) >= 1

    def test_sections_count_classified_as_structural(self) -> None:
        """A sentence about number of sections must be STRUCTURAL."""
        text = "The response contains 3 sections covering the main topics."
        claims = extract_claims(text)
        structural = [c for c in claims if c.claim_type == ClaimType.STRUCTURAL]
        assert len(structural) >= 1

    def test_generic_assertion_classified_as_factual(self) -> None:
        """A generic assertion with no special patterns must be FACTUAL."""
        text = "The algorithm has quadratic time complexity in the worst case."
        claims = extract_claims(text)
        assert len(claims) >= 1
        assert claims[0].claim_type == ClaimType.FACTUAL


# -- Claim extraction: filtering ---------------------------------------------


class TestClaimFiltering:
    """Tests for filtering non-assertive text out of claim extraction."""

    def test_empty_text_returns_empty(self) -> None:
        """Empty or whitespace-only text must produce no claims."""
        assert extract_claims("") == []
        assert extract_claims("   ") == []

    def test_questions_filtered_out(self) -> None:
        """Questions must not be extracted as claims."""
        text = "What is the best approach for this problem?"
        claims = extract_claims(text)
        assert len(claims) == 0

    def test_hedging_phrases_filtered_out(self) -> None:
        """Hedging language must not produce claims."""
        text = "I think this might work but I'm not sure about the edge cases."
        claims = extract_claims(text)
        # "I think" is a hedge pattern — should be filtered
        hedged = [c for c in claims if "I think" in c.text]
        assert len(hedged) == 0

    def test_short_fragments_filtered(self) -> None:
        """Fragments shorter than 10 characters must be filtered."""
        text = "OK. Fine. Yes."
        claims = extract_claims(text)
        assert len(claims) == 0


# -- extract_claims_by_type --------------------------------------------------


class TestExtractClaimsByType:
    """Tests for the grouping convenience wrapper."""

    def test_groups_by_type(self) -> None:
        """Claims must be correctly grouped by their ClaimType."""
        text = (
            "The function process returns a list. "
            "The output is valid JSON. "
            "The response contains 5 sections for the report."
        )
        grouped = extract_claims_by_type(text)

        assert isinstance(grouped, dict)
        # At least code and format should appear
        found_types = set(grouped.keys())
        assert len(found_types) >= 1

    def test_empty_text_returns_empty_dict(self) -> None:
        """Empty text must produce an empty dict."""
        assert extract_claims_by_type("") == {}


# -- Entailment: CODE claim verification ------------------------------------


class TestCodeClaimVerification:
    """Tests for AST-based CODE claim verification."""

    def test_valid_code_claim_gets_trust_1(self) -> None:
        """A claim asserting 'valid' code with a parseable snippet must get trust=1.0."""
        claim = Claim(
            text="Code block is valid Python",
            claim_type=ClaimType.CODE,
            source_span=(0, 10),
            confidence=0.95,
            code_snippet="def hello():\n    return 'world'",
        )
        verdict = verify_claim(claim)

        assert isinstance(verdict, ClaimVerdict)
        assert verdict.trust_score == pytest.approx(1.0)
        assert verdict.method == "ast"

    def test_invalid_code_claimed_valid_gets_trust_0(self) -> None:
        """A claim asserting 'valid' for broken code must get trust=0.0."""
        claim = Claim(
            text="Code block is valid Python",
            claim_type=ClaimType.CODE,
            source_span=(0, 10),
            confidence=0.9,
            code_snippet="def broken(\n",
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(0.0)
        assert verdict.method == "ast"

    def test_no_snippet_gets_neutral_trust(self) -> None:
        """A CODE claim with no attached snippet must get trust=0.5."""
        claim = Claim(
            text="The code is valid Python",
            claim_type=ClaimType.CODE,
            source_span=(0, 10),
            confidence=0.8,
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(0.5)
        assert verdict.method == "ast"

    def test_syntax_error_claimed_invalid_gets_trust_1(self) -> None:
        """A claim asserting 'syntax error' for broken code must get trust=1.0."""
        claim = Claim(
            text="Code block contains syntax errors",
            claim_type=ClaimType.CODE,
            source_span=(0, 10),
            confidence=0.8,
            code_snippet="def broken(\n",
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(1.0)
        assert verdict.method == "ast"


# -- Entailment: FORMAT claim verification -----------------------------------


class TestFormatClaimVerification:
    """Tests for regex/format-based FORMAT claim verification."""

    def test_valid_json_claim_verified(self) -> None:
        """A JSON format claim with parseable JSON snippet must get trust=1.0."""
        claim = Claim(
            text="Output is valid JSON",
            claim_type=ClaimType.FORMAT,
            source_span=(0, 10),
            confidence=0.9,
            code_snippet='{"key": "value", "count": 42}',
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(1.0)
        assert verdict.method == "regex"

    def test_invalid_json_claim_disproved(self) -> None:
        """A JSON format claim with broken JSON must get trust=0.0."""
        claim = Claim(
            text="Output is valid JSON",
            claim_type=ClaimType.FORMAT,
            source_span=(0, 10),
            confidence=0.9,
            code_snippet="{broken json",
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(0.0)
        assert verdict.method == "regex"

    def test_unknown_format_unverified(self) -> None:
        """A format claim about an unknown format must get trust=0.5."""
        claim = Claim(
            text="The output is formatted as custom-format-xyz",
            claim_type=ClaimType.FORMAT,
            source_span=(0, 10),
            confidence=0.9,
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(0.5)
        assert verdict.method == "unverified"

    def test_snake_case_naming_detected(self) -> None:
        """A naming convention claim with snake_case code must get trust=0.9."""
        claim = Claim(
            text="Code follows snake_case naming convention",
            claim_type=ClaimType.FORMAT,
            source_span=(0, 10),
            confidence=0.9,
            code_snippet="my_variable = some_function(other_arg)",
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(0.9)
        assert verdict.method == "regex"


# -- Entailment: STRUCTURAL claim verification --------------------------------


class TestStructuralClaimVerification:
    """Tests for structural count verification."""

    def test_correct_count_verified(self) -> None:
        """A structural claim with matching line count must get trust=0.8."""
        claim = Claim(
            text="The output contains 3 sections",
            claim_type=ClaimType.STRUCTURAL,
            source_span=(0, 10),
            confidence=0.9,
            code_snippet="Section A\nSection B\nSection C",
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(0.8)
        assert verdict.method == "structural"

    def test_no_count_in_claim_unverified(self) -> None:
        """A structural claim without a numeric count reports as unsupported (fail-closed).

        SHARD-01 changed structural claims without a parseable count from the
        old neutral 0.5/unverified to 0.0/stub (fail-closed Rule 2).  A
        structural claim that cannot be verified must NOT be given a neutral
        passing score — it must be treated as unsupported.
        """
        claim = Claim(
            text="The output is well-organized into logical units",
            claim_type=ClaimType.STRUCTURAL,
            source_span=(0, 10),
            confidence=0.9,
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(0.0)
        assert verdict.method == "stub"


# -- Entailment: FACTUAL claim verification -----------------------------------


class TestFactualClaimVerification:
    """Tests for factual (model-based) claim handling."""

    def test_factual_claim_marked_unverified(self) -> None:
        """Factual claims report as unsupported (fail-closed) when no model is loaded.

        SHARD-01 changed the factual-claim handler from the old neutral
        0.5/unverified to 0.0/stub (fail-closed Rule 2).  A factual claim
        without a loaded entailment model must NOT receive a neutral passing
        score — it must be reported as unsupported until real model-based
        verification is wired.
        """
        claim = Claim(
            text="The algorithm has O(n log n) complexity",
            claim_type=ClaimType.FACTUAL,
            source_span=(0, 10),
            confidence=0.9,
        )
        verdict = verify_claim(claim)

        assert verdict.trust_score == pytest.approx(0.0)
        assert verdict.method == "stub"
        assert "unsupported" in verdict.explanation.lower() or "model" in verdict.explanation.lower()


# -- Aggregate verification --------------------------------------------------


class TestVerifyClaims:
    """Tests for the aggregate verify_claims() function."""

    def test_empty_claims_returns_zero_trust(self) -> None:
        """An empty claim list must return trust=0.0 — no evidence of correctness means no trust.

        Returning 1.0 for empty input is the default-pass anti-pattern: a verifier that
        certifies arbitrary content as valid when given nothing to inspect.
        """
        report = verify_claims([])

        assert isinstance(report, VerificationReport)
        assert report.overall_trust == pytest.approx(0.0)
        assert report.total_claims == 0
        assert report.deterministic_count == 0
        assert report.model_count == 0
        assert report.unverified_count == 0

    def test_all_deterministic_claims_counted(self) -> None:
        """Claims verified by AST/regex must be counted as deterministic."""
        claims = [
            Claim(
                text="Code block is valid Python",
                claim_type=ClaimType.CODE,
                source_span=(0, 10),
                confidence=0.95,
                code_snippet="x = 1",
            ),
            Claim(
                text="Output is valid JSON",
                claim_type=ClaimType.FORMAT,
                source_span=(20, 30),
                confidence=0.9,
                code_snippet='{"a": 1}',
            ),
        ]
        report = verify_claims(claims)

        assert report.total_claims == 2
        assert report.deterministic_count == 2
        assert report.unverified_count == 0

    def test_weighted_trust_score(self) -> None:
        """Overall trust must be a weighted average of individual claim scores.

        SHARD-01 changed factual claims from 0.5/unverified to 0.0/stub (fail-closed).
        Weighted average is now (1.0 + 0.0) / 2 = 0.5.
        """
        claims = [
            Claim(
                text="Code block is valid Python",
                claim_type=ClaimType.CODE,
                source_span=(0, 10),
                confidence=1.0,
                code_snippet="x = 1",
            ),
            Claim(
                text="The algorithm is efficient",
                claim_type=ClaimType.FACTUAL,
                source_span=(20, 30),
                confidence=1.0,
            ),
        ]
        report = verify_claims(claims)

        # First claim: trust=1.0, confidence=1.0 => weighted 1.0
        # Second claim: trust=0.0, confidence=1.0 => weighted 0.0 (fail-closed, SHARD-01)
        # Average: (1.0 + 0.0) / 2 = 0.5
        assert report.overall_trust == pytest.approx(0.5)

    def test_report_repr_shows_trust(self) -> None:
        """VerificationReport repr must include trust score."""
        report = verify_claims([])
        assert "trust=" in repr(report)

    def test_verdict_repr_shows_type_and_method(self) -> None:
        """ClaimVerdict repr must include claim type, trust, and method."""
        claim = Claim(
            text="Code block is valid Python",
            claim_type=ClaimType.CODE,
            source_span=(0, 10),
            confidence=0.95,
            code_snippet="x = 1",
        )
        verdict = verify_claim(claim)
        r = repr(verdict)
        assert "code" in r
        assert "trust=" in r
        assert "method=" in r


# -- Deterministic-before-model ordering -------------------------------------


class TestDeterministicFirst:
    """Verify that deterministic checks run before model-based checks."""

    def test_code_claim_uses_ast_not_model(self) -> None:
        """CODE claims must be verified by AST, never by model."""
        claim = Claim(
            text="Code block is valid Python",
            claim_type=ClaimType.CODE,
            source_span=(0, 10),
            confidence=0.95,
            code_snippet="x = 1",
        )
        verdict = verify_claim(claim)
        assert verdict.method == "ast"

    def test_format_claim_uses_regex_not_model(self) -> None:
        """FORMAT claims with JSON must be verified by regex, never by model."""
        claim = Claim(
            text="Output is valid JSON",
            claim_type=ClaimType.FORMAT,
            source_span=(0, 10),
            confidence=0.9,
            code_snippet='{"a": 1}',
        )
        verdict = verify_claim(claim)
        assert verdict.method == "regex"

    def test_mixed_claims_deterministic_count_correct(self) -> None:
        """In a mixed batch, deterministic and unverified counts must be correct."""
        claims = [
            Claim(
                text="Code block is valid Python",
                claim_type=ClaimType.CODE,
                source_span=(0, 10),
                confidence=0.95,
                code_snippet="x = 1",
            ),
            Claim(
                text="The design pattern is correct",
                claim_type=ClaimType.FACTUAL,
                source_span=(20, 30),
                confidence=0.9,
            ),
        ]
        report = verify_claims(claims)

        assert report.deterministic_count == 1
        assert report.unverified_count == 1


# -- Module wiring -----------------------------------------------------------


class TestModuleWiring:
    """Verify exports and imports are correct."""

    def test_claim_extractor_exports(self) -> None:
        """Key types must be importable from vetinari.verification.claim_extractor."""
        from vetinari.verification.claim_extractor import (
            Claim,
            ClaimType,
            extract_claims,
            extract_claims_by_type,
        )

        assert Claim is not None
        assert ClaimType is not None
        assert extract_claims is not None
        assert extract_claims_by_type is not None

    def test_entailment_checker_exports(self) -> None:
        """Key types must be importable from vetinari.verification.entailment_checker."""
        from vetinari.verification.entailment_checker import (
            ClaimVerdict,
            VerificationReport,
            verify_claim,
            verify_claims,
        )

        assert ClaimVerdict is not None
        assert VerificationReport is not None
        assert verify_claim is not None
        assert verify_claims is not None

    def test_imports_from_verification_init(self) -> None:
        """Key types must be importable from vetinari.verification."""
        from vetinari.verification import (
            Claim,
            ClaimType,
            ClaimVerdict,
            VerificationReport,
            extract_claims,
            verify_claims,
        )

        assert Claim is not None
        assert ClaimType is not None
        assert ClaimVerdict is not None
        assert VerificationReport is not None
        assert extract_claims is not None
        assert verify_claims is not None
