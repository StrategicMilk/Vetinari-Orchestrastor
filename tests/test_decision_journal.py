"""Tests for vetinari.observability.decision_journal — SQLite-backed decision log."""

from __future__ import annotations

from pathlib import Path

import pytest

from vetinari.observability.decision_journal import DecisionJournal, DecisionRecord
from vetinari.types import ConfidenceLevel, DecisionType


@pytest.fixture
def journal(tmp_path: Path) -> DecisionJournal:
    """Create a fresh DecisionJournal backed by a temp database."""
    return DecisionJournal(db_path=tmp_path / "test_decisions.db")


# -- log_decision -------------------------------------------------------------


class TestLogDecision:
    """log_decision() writes records and returns decision_id."""

    def test_returns_decision_id(self, journal: DecisionJournal) -> None:
        """log_decision returns a non-empty string ID."""
        decision_id = journal.log_decision(
            decision_type=DecisionType.ROUTING,
            description="test routing decision",
            confidence_score=-0.5,
            confidence_level=ConfidenceLevel.HIGH,
        )
        assert decision_id.startswith("dec_")
        assert len(decision_id) > 4

    def test_stored_record_is_retrievable(self, journal: DecisionJournal) -> None:
        """A logged decision can be retrieved via get_decisions."""
        decision_id = journal.log_decision(
            decision_type=DecisionType.APPROVAL,
            description="human approval request",
            confidence_score=-2.5,
            confidence_level=ConfidenceLevel.LOW,
            action_taken="defer",
        )
        records = journal.get_decisions()
        assert len(records) == 1
        record = records[0]
        assert record.decision_id == decision_id
        assert record.decision_type == DecisionType.APPROVAL
        assert record.description == "human approval request"
        assert abs(record.confidence_score - (-2.5)) < 1e-9
        assert record.confidence_level == ConfidenceLevel.LOW
        assert record.action_taken == "defer"

    def test_confidence_factors_stored(self, journal: DecisionJournal) -> None:
        """confidence_factors dict is stored and retrieved intact."""
        factors = {"mean_logprob": -0.5, "token_count": 10.0}
        journal.log_decision(
            decision_type=DecisionType.QUALITY,
            description="quality gate decision",
            confidence_score=-0.5,
            confidence_level=ConfidenceLevel.HIGH,
            confidence_factors=factors,
        )
        records = journal.get_decisions()
        assert records[0].confidence_factors == factors

    def test_context_stored(self, journal: DecisionJournal) -> None:
        """context dict is stored and retrieved intact."""
        context = {"project_id": "proj_abc", "task_id": "task_123"}
        journal.log_decision(
            decision_type=DecisionType.ESCALATION,
            description="escalation decision",
            confidence_score=-5.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            context=context,
        )
        records = journal.get_decisions()
        assert records[0].context == context

    def test_timestamp_is_set(self, journal: DecisionJournal) -> None:
        """timestamp is automatically set to a non-empty ISO 8601 string."""
        journal.log_decision(
            decision_type=DecisionType.AUTONOMY,
            description="autonomy change",
            confidence_score=0.9,
            confidence_level=ConfidenceLevel.HIGH,
        )
        records = journal.get_decisions()
        assert records[0].timestamp != ""

    def test_multiple_decisions_stored(self, journal: DecisionJournal) -> None:
        """Multiple log_decision calls each produce independent records."""
        for i in range(5):
            journal.log_decision(
                decision_type=DecisionType.ROUTING,
                description=f"decision {i}",
                confidence_score=float(-i),
                confidence_level=ConfidenceLevel.MEDIUM,
            )
        records = journal.get_decisions()
        assert len(records) == 5


# -- update_outcome -----------------------------------------------------------


class TestUpdateOutcome:
    """update_outcome() patches the outcome field."""

    def test_updates_existing_record(self, journal: DecisionJournal) -> None:
        """update_outcome returns True and updates the outcome field."""
        decision_id = journal.log_decision(
            decision_type=DecisionType.ROUTING,
            description="routing decision",
            confidence_score=-0.5,
            confidence_level=ConfidenceLevel.HIGH,
        )
        updated = journal.update_outcome(decision_id, "output passed quality gate")
        assert updated is True
        records = journal.get_decisions()
        assert records[0].outcome == "output passed quality gate"

    def test_returns_false_for_missing_id(self, journal: DecisionJournal) -> None:
        """update_outcome returns False when decision_id doesn't exist."""
        updated = journal.update_outcome("dec_nonexistent", "some outcome")
        assert updated is False

    def test_outcome_starts_empty(self, journal: DecisionJournal) -> None:
        """Freshly logged decisions have an empty outcome."""
        journal.log_decision(
            decision_type=DecisionType.ROUTING,
            description="new decision",
            confidence_score=-1.0,
            confidence_level=ConfidenceLevel.MEDIUM,
        )
        records = journal.get_decisions()
        assert records[0].outcome == ""


# -- get_decisions ------------------------------------------------------------


class TestGetDecisions:
    """get_decisions() filtering and ordering."""

    def test_no_filter_returns_all(self, journal: DecisionJournal) -> None:
        """No filters returns all records up to limit."""
        for dt in (DecisionType.ROUTING, DecisionType.APPROVAL, DecisionType.QUALITY):
            journal.log_decision(
                decision_type=dt,
                description=f"{dt.value} decision",
                confidence_score=-1.0,
                confidence_level=ConfidenceLevel.MEDIUM,
            )
        records = journal.get_decisions()
        assert len(records) == 3

    def test_filter_by_decision_type(self, journal: DecisionJournal) -> None:
        """filter by decision_type returns only matching records."""
        journal.log_decision(DecisionType.ROUTING, "routing", -0.5, ConfidenceLevel.HIGH)
        journal.log_decision(DecisionType.APPROVAL, "approval", -0.5, ConfidenceLevel.HIGH)
        journal.log_decision(DecisionType.ROUTING, "routing2", -0.5, ConfidenceLevel.HIGH)

        routing = journal.get_decisions(decision_type=DecisionType.ROUTING)
        assert len(routing) == 2
        assert all(r.decision_type == DecisionType.ROUTING for r in routing)

    def test_filter_by_confidence_level(self, journal: DecisionJournal) -> None:
        """filter by confidence_level returns only matching records."""
        journal.log_decision(DecisionType.ROUTING, "high", -0.2, ConfidenceLevel.HIGH)
        journal.log_decision(DecisionType.ROUTING, "low", -2.5, ConfidenceLevel.LOW)
        journal.log_decision(DecisionType.ROUTING, "high2", -0.3, ConfidenceLevel.HIGH)

        high_records = journal.get_decisions(confidence_level=ConfidenceLevel.HIGH)
        assert len(high_records) == 2
        assert all(r.confidence_level == ConfidenceLevel.HIGH for r in high_records)

    def test_limit_restricts_results(self, journal: DecisionJournal) -> None:
        """limit parameter restricts number of returned records."""
        for i in range(10):
            journal.log_decision(DecisionType.ROUTING, f"d{i}", -1.0, ConfidenceLevel.MEDIUM)

        records = journal.get_decisions(limit=3)
        assert len(records) == 3

    def test_returns_most_recent_first(self, journal: DecisionJournal) -> None:
        """get_decisions returns records in descending timestamp order."""
        for i in range(3):
            journal.log_decision(DecisionType.ROUTING, f"d{i}", float(-i), ConfidenceLevel.MEDIUM)
        records = journal.get_decisions()
        # Most recent should be last logged (d2)
        assert records[0].description == "d2"

    def test_combined_filter(self, journal: DecisionJournal) -> None:
        """Both decision_type and confidence_level filters apply together."""
        journal.log_decision(DecisionType.ROUTING, "r+high", -0.2, ConfidenceLevel.HIGH)
        journal.log_decision(DecisionType.ROUTING, "r+low", -2.5, ConfidenceLevel.LOW)
        journal.log_decision(DecisionType.APPROVAL, "a+high", -0.2, ConfidenceLevel.HIGH)

        records = journal.get_decisions(
            decision_type=DecisionType.ROUTING,
            confidence_level=ConfidenceLevel.HIGH,
        )
        assert len(records) == 1
        assert records[0].description == "r+high"


# -- DecisionRecord -----------------------------------------------------------


class TestDecisionRecord:
    """DecisionRecord dataclass repr and identity."""

    def test_repr_contains_key_fields(self) -> None:
        """__repr__ shows decision_id, type, score, and level."""
        record = DecisionRecord(
            decision_id="dec_abc123",
            decision_type=DecisionType.ROUTING,
            description="test",
            confidence_score=-1.5,
            confidence_level=ConfidenceLevel.MEDIUM,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        r = repr(record)
        assert "dec_abc123" in r
        assert "routing" in r
        assert "score=-1.500" in r
        assert "medium" in r
