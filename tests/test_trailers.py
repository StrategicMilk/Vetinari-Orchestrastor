"""Tests for vetinari.git.trailers — Decision-Ref commit trailer generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.git.trailers import (
    format_trailers_for_commit,
    generate_trailers,
)

# -- generate_trailers ---------------------------------------------------------


class TestGenerateTrailers:
    def test_explicit_decision_ids(self) -> None:
        trailers = generate_trailers(decision_ids=["abc123", "def456"])
        assert "Decision-Ref: @DJ-abc123" in trailers
        assert "Decision-Ref: @DJ-def456" in trailers
        assert len(trailers) == 2

    def test_explicit_adr_ids(self) -> None:
        trailers = generate_trailers(adr_ids=["0061", "0076"])
        assert "Decision-Ref: @ADR-0061" in trailers
        assert "Decision-Ref: @ADR-0076" in trailers
        assert len(trailers) == 2

    def test_mixed_decision_and_adr_ids(self) -> None:
        trailers = generate_trailers(decision_ids=["abc"], adr_ids=["0061"])
        assert "Decision-Ref: @DJ-abc" in trailers
        assert "Decision-Ref: @ADR-0061" in trailers

    def test_deduplicates_trailers(self) -> None:
        trailers = generate_trailers(decision_ids=["abc", "abc"])
        dj_trailers = [t for t in trailers if "DJ-abc" in t]
        assert len(dj_trailers) == 1

    def test_no_args_returns_empty(self) -> None:
        trailers = generate_trailers()
        assert trailers == []

    @patch("vetinari.git.trailers._get_decisions_for_trace")
    @patch("vetinari.git.trailers._extract_adr_refs_from_decisions")
    def test_trace_id_lookup(self, mock_adr_refs: MagicMock, mock_trace: MagicMock) -> None:
        mock_trace.return_value = ["trace-dec-1", "trace-dec-2"]
        mock_adr_refs.return_value = []
        trailers = generate_trailers(trace_id="trace-001")
        assert "Decision-Ref: @DJ-trace-dec-1" in trailers
        assert "Decision-Ref: @DJ-trace-dec-2" in trailers
        mock_trace.assert_called_once_with("trace-001")

    @patch("vetinari.git.trailers._get_decisions_for_trace")
    @patch("vetinari.git.trailers._extract_adr_refs_from_decisions")
    def test_trace_extracts_adr_refs(self, mock_adr_refs: MagicMock, mock_trace: MagicMock) -> None:
        mock_trace.return_value = ["dec-1"]
        mock_adr_refs.return_value = ["0072"]
        trailers = generate_trailers(trace_id="trace-001")
        assert "Decision-Ref: @DJ-dec-1" in trailers
        assert "Decision-Ref: @ADR-0072" in trailers


# -- format_trailers_for_commit ------------------------------------------------


class TestFormatTrailersForCommit:
    def test_formats_with_blank_line_separator(self) -> None:
        trailers = ["Decision-Ref: @DJ-abc", "Decision-Ref: @ADR-0061"]
        result = format_trailers_for_commit(trailers)
        assert result.startswith("\n")
        assert "Decision-Ref: @DJ-abc" in result
        assert "Decision-Ref: @ADR-0061" in result

    def test_empty_list_returns_empty_string(self) -> None:
        result = format_trailers_for_commit([])
        assert result == ""

    def test_single_trailer(self) -> None:
        result = format_trailers_for_commit(["Decision-Ref: @DJ-xyz"])
        assert result == "\nDecision-Ref: @DJ-xyz"


# -- Integration with git __init__ -------------------------------------------


class TestGitModuleExports:
    def test_generate_trailers_importable_from_git(self) -> None:
        from vetinari.git import generate_trailers as gt

        assert gt is generate_trailers


# -- D7: _extract_adr_refs_from_decisions bulk-fetch + correct field names --


class TestExtractAdrRefsFromDecisions:
    """D7: _extract_adr_refs_from_decisions must use bulk-fetch (O(n)) and correct field names."""

    def test_finds_adr_ref_beyond_100_records(self) -> None:
        """ADR ref in record at index 103 must be found despite being beyond the first 100."""
        from dataclasses import dataclass, field
        from unittest.mock import MagicMock, patch

        from vetinari.git.trailers import _extract_adr_refs_from_decisions
        from vetinari.observability.decision_journal import DecisionRecord
        from vetinari.types import ConfidenceLevel, DecisionType

        # Build 105 mock DecisionRecord objects; the ADR ref lives at index 103
        def _make_record(i: int) -> DecisionRecord:
            """Create a DecisionRecord with an ADR ref only in the last entry."""
            description = f"Decision number {i}"
            action_taken = "proceeded"
            outcome = "success"
            context: dict = {}
            confidence_factors: dict = {}
            if i == 103:
                description = "Switched strategy ADR-0099 for resilience"
            return DecisionRecord(
                decision_id=f"dec-{i:04d}",
                decision_type=DecisionType.ROUTING,
                description=description,
                confidence_score=0.8,
                confidence_level=ConfidenceLevel.HIGH,
                action_taken=action_taken,
                outcome=outcome,
                context=context,
                confidence_factors=confidence_factors,
            )

        all_records = [_make_record(i) for i in range(105)]
        decision_ids = [r.decision_id for r in all_records]

        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = all_records

        with patch(
            "vetinari.observability.decision_journal.get_decision_journal",
            return_value=mock_journal,
        ):
            result = _extract_adr_refs_from_decisions(decision_ids)

        assert "0099" in result, (
            "D7 fix: _extract_adr_refs_from_decisions must find ADR-0099 referenced at record index 103"
        )
        # Confirm get_decisions was called once (bulk fetch, not per-ID)
        mock_journal.get_decisions.assert_called_once()

    def test_uses_correct_field_names(self) -> None:
        """Verifies the function reads description/action_taken/outcome/context/confidence_factors."""
        from unittest.mock import MagicMock, patch

        from vetinari.git.trailers import _extract_adr_refs_from_decisions
        from vetinari.observability.decision_journal import DecisionRecord
        from vetinari.types import ConfidenceLevel, DecisionType

        # ADR ref is in the `outcome` field — would be missed if the function used wrong field name
        rec = DecisionRecord(
            decision_id="dec-field-test",
            decision_type=DecisionType.QUALITY,
            description="no ref here",
            confidence_score=0.5,
            confidence_level=ConfidenceLevel.MEDIUM,
            action_taken="no ref here",
            outcome="Applied approach from ADR-0042",
            context={},
            confidence_factors={},
        )

        mock_journal = MagicMock()
        mock_journal.get_decisions.return_value = [rec]

        with patch(
            "vetinari.observability.decision_journal.get_decision_journal",
            return_value=mock_journal,
        ):
            result = _extract_adr_refs_from_decisions(["dec-field-test"])

        assert "0042" in result, (
            "D7 fix: outcome field must be searched — wrong field name would miss ADR-0042"
        )
