"""Tests for knowledge linter — contradictions, stale entries, orphans, and vocabulary drift."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vetinari.kaizen.knowledge_lint import (
    _MS_PER_DAY,
    _ORPHAN_THRESHOLD_DAYS,
    _STALE_THRESHOLD_DAYS,
    KnowledgeLinter,
    KnowledgeLintReport,
    LintFinding,
)

# -- Target for patching emit side-effects in lint_all calls ------------------
_EMIT_TARGET = "vetinari.kaizen.improvement_events.emit_lint_finding"


# -- Helpers ------------------------------------------------------------------


def _now_ms() -> int:
    """Return current UTC time as epoch milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _make_entry(
    id: str = "e1",
    content: str = "test content",
    entry_type: str = "discovery",
    last_accessed: int = 0,
    summary: str = "test summary",
    recall_count: int = 0,
    timestamp: int = 0,
) -> SimpleNamespace:
    """Create a minimal memory-entry-like object for testing."""
    return SimpleNamespace(
        id=id,
        content=content,
        entry_type=entry_type,
        last_accessed=last_accessed,
        summary=summary,
        recall_count=recall_count,
        timestamp=timestamp,
    )


def _old_timestamp_ms() -> int:
    """Return an epoch-ms timestamp that is past the orphan threshold."""
    return _now_ms() - (_ORPHAN_THRESHOLD_DAYS + 10) * _MS_PER_DAY


def _recent_timestamp_ms() -> int:
    """Return an epoch-ms timestamp that is within the orphan threshold (5 days old)."""
    return _now_ms() - 5 * _MS_PER_DAY


# -- Tests --------------------------------------------------------------------


def test_no_findings_clean_entries() -> None:
    """Well-separated entries produce no lint findings."""
    entries = [
        _make_entry(
            id="e1",
            content="Python is a high-level programming language with dynamic typing.",
            entry_type="discovery",
            summary="Python language basics",
            recall_count=5,
            last_accessed=_now_ms(),
            timestamp=_now_ms(),
        ),
        _make_entry(
            id="e2",
            content="Rust is a systems language focused on memory safety without garbage collection.",
            entry_type="discovery",
            summary="Rust memory model",
            recall_count=3,
            last_accessed=_now_ms(),
            timestamp=_now_ms(),
        ),
    ]
    linter = KnowledgeLinter()
    with patch(_EMIT_TARGET):
        report = linter.lint_all(entries)

    assert isinstance(report, KnowledgeLintReport)
    assert report.checked_entries == 2
    assert report.findings == []


def test_contradiction_detected() -> None:
    """Two same-type entries with high content similarity trigger a contradiction finding."""
    base = "The deployment pipeline runs on Kubernetes using Helm charts for configuration."
    similar = "The deployment pipeline runs on Kubernetes using Helm chart for configurations."

    entries = [
        _make_entry(id="e1", content=base, entry_type="fact", summary="deploy pipeline"),
        _make_entry(id="e2", content=similar, entry_type="fact", summary="helm deploy"),
    ]
    linter = KnowledgeLinter()
    findings = linter.check_contradictions(entries)

    assert len(findings) == 1
    f = findings[0]
    assert f.category == "contradiction"
    assert f.severity == "warning"
    assert "e1" in f.entry_ids or "e2" in f.entry_ids


def test_stale_entry_detected() -> None:
    """Entry with last_accessed 100+ days ago triggers a stale finding."""
    stale_time = _now_ms() - (_STALE_THRESHOLD_DAYS + 10) * _MS_PER_DAY
    entries = [_make_entry(id="old1", last_accessed=stale_time)]
    linter = KnowledgeLinter()
    findings = linter.check_stale(entries)

    assert len(findings) == 1
    f = findings[0]
    assert f.category == "stale"
    assert f.severity == "info"
    assert "old1" in f.entry_ids


def test_stale_skips_never_accessed() -> None:
    """Entry with last_accessed=0 is NOT flagged as stale (handled by orphan check)."""
    entries = [_make_entry(id="new_entry", last_accessed=0)]
    linter = KnowledgeLinter()
    findings = linter.check_stale(entries)

    assert findings == []


def test_orphaned_entry_detected() -> None:
    """Entry with recall_count=0 and timestamp 40+ days ago triggers an orphan finding."""
    entries = [_make_entry(id="orphan1", recall_count=0, timestamp=_old_timestamp_ms())]
    linter = KnowledgeLinter()
    findings = linter.check_orphaned(entries)

    assert len(findings) == 1
    f = findings[0]
    assert f.category == "orphaned"
    assert f.severity == "warning"
    assert "orphan1" in f.entry_ids


def test_orphaned_skips_recent() -> None:
    """Entry with recall_count=0 but only 5 days old is NOT flagged as orphaned."""
    entries = [_make_entry(id="fresh1", recall_count=0, timestamp=_recent_timestamp_ms())]
    linter = KnowledgeLinter()
    findings = linter.check_orphaned(entries)

    assert findings == []


def test_orphaned_skips_recalled_entries() -> None:
    """Entry with recall_count > 0 is never flagged as orphaned even if old."""
    entries = [_make_entry(id="used1", recall_count=5, timestamp=_old_timestamp_ms())]
    linter = KnowledgeLinter()
    findings = linter.check_orphaned(entries)

    assert findings == []


def test_vocabulary_drift_detected() -> None:
    """Two entries with similar summaries but different content trigger vocabulary_drift."""
    summary = "how to configure database connection pooling"
    entries = [
        _make_entry(
            id="d1",
            summary=summary,
            content="Use SQLAlchemy pool_size=10, max_overflow=20 to configure the connection pool for PostgreSQL.",
        ),
        _make_entry(
            id="d2",
            summary=summary,
            content="Django uses CONN_MAX_AGE setting in DATABASES to manage connection lifetime and pool behaviour.",
        ),
    ]
    linter = KnowledgeLinter()
    findings = linter.check_vocabulary_drift(entries)

    assert len(findings) >= 1
    f = findings[0]
    assert f.category == "vocabulary_drift"
    assert f.severity == "info"


def test_lint_all_aggregates() -> None:
    """Full lint pass returns a report containing all expected finding categories."""
    stale_time = _now_ms() - (_STALE_THRESHOLD_DAYS + 10) * _MS_PER_DAY
    old_ts = _old_timestamp_ms()

    base_content = "The CI/CD pipeline deploys using Docker containers on AWS ECS clusters."
    near_dup = "The CI/CD pipeline deploys using Docker container on AWS ECS cluster."

    entries = [
        # Contradiction pair
        _make_entry(
            id="c1",
            content=base_content,
            entry_type="fact",
            summary="ci pipeline",
            recall_count=1,
            last_accessed=_now_ms(),
            timestamp=_now_ms(),
        ),
        _make_entry(
            id="c2",
            content=near_dup,
            entry_type="fact",
            summary="ci deploy",
            recall_count=1,
            last_accessed=_now_ms(),
            timestamp=_now_ms(),
        ),
        # Stale entry
        _make_entry(id="s1", last_accessed=stale_time, recall_count=1, timestamp=_now_ms()),
        # Orphaned entry
        _make_entry(id="o1", recall_count=0, timestamp=old_ts),
        # Vocabulary drift pair
        _make_entry(
            id="v1",
            summary="database pool configuration",
            content="SQLAlchemy pool_size=10 for PostgreSQL connection pool management.",
            recall_count=1,
            last_accessed=_now_ms(),
            timestamp=_now_ms(),
        ),
        _make_entry(
            id="v2",
            summary="database pool configuration",
            content="Django CONN_MAX_AGE controls persistent connection lifetime in settings.",
            recall_count=1,
            last_accessed=_now_ms(),
            timestamp=_now_ms(),
        ),
    ]

    linter = KnowledgeLinter()
    with patch(_EMIT_TARGET):
        report = linter.lint_all(entries)

    assert isinstance(report, KnowledgeLintReport)
    assert report.checked_entries == len(entries)
    assert len(report.findings) > 0

    categories = {f.category for f in report.findings}
    assert "contradiction" in categories, "Expected contradiction finding"
    assert "stale" in categories, "Expected stale finding"
    assert "orphaned" in categories, "Expected orphaned finding"


def test_lint_emits_events() -> None:
    """Verify emit_lint_finding is called for each finding produced by lint_all."""
    entries = [_make_entry(id="orphan_emit", recall_count=0, timestamp=_old_timestamp_ms())]

    linter = KnowledgeLinter()
    with patch(_EMIT_TARGET) as mock_emit:
        report = linter.lint_all(entries)

    # At least one orphaned finding → emit_lint_finding called at least once
    assert len(report.findings) >= 1
    assert mock_emit.call_count == len(report.findings)

    # Each call must include the required keyword arguments
    for call in mock_emit.call_args_list:
        kwargs = call.kwargs
        assert "finding_id" in kwargs
        assert "category" in kwargs
        assert "description" in kwargs
        assert "severity" in kwargs


def test_lint_finding_dataclass() -> None:
    """LintFinding is frozen and has all required fields populated correctly."""
    f = LintFinding(
        finding_id="lint_abc123",
        category="stale",
        description="Entry was not accessed recently.",
        severity="info",
        entry_ids=("e1",),
    )
    assert f.finding_id == "lint_abc123"
    assert f.category == "stale"
    assert f.severity == "info"
    assert f.entry_ids == ("e1",)
    assert f.timestamp  # auto-set ISO string

    with pytest.raises((AttributeError, TypeError)):
        f.category = "changed"  # type: ignore[misc]


def test_knowledge_lint_report_repr() -> None:
    """KnowledgeLintReport repr shows findings count and checked_entries."""
    report = KnowledgeLintReport(checked_entries=5)
    r = repr(report)
    assert "findings=0" in r
    assert "checked_entries=5" in r
