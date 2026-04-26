"""Tests for vetinari.agents.inspector_extract — retroactive decision surfacing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.inspector_extract import (
    CandidateDecision,
    extract_implicit_decisions,
    log_extracted_decisions,
)
from vetinari.types import DecisionType

# -- CandidateDecision ---------------------------------------------------------


class TestCandidateDecision:
    def test_repr_shows_key_fields(self) -> None:
        cd = CandidateDecision(
            description="New dependency: requests",
            confidence=0.75,
            adr_exists=False,
        )
        assert "New dependency" in repr(cd)
        assert "0.75" in repr(cd)
        assert "adr_exists=False" in repr(cd)


# -- extract_implicit_decisions ------------------------------------------------


_DIFF_NEW_IMPORT = """\
diff --git a/vetinari/foo.py b/vetinari/foo.py
--- a/vetinari/foo.py
+++ b/vetinari/foo.py
@@ -1,3 +1,5 @@
+from redis import Redis
+import requests
 from vetinari.types import AgentType
"""

_DIFF_API_ENDPOINT = """\
diff --git a/vetinari/web/api.py b/vetinari/web/api.py
+++ b/vetinari/web/api.py
@@ -10,0 +11,3 @@
+    @post("/api/v1/tasks")
+    async def create_task(self, data: TaskInput) -> TaskResponse:
+        pass
"""

_DIFF_DATACLASS = """\
diff --git a/vetinari/types.py b/vetinari/types.py
+++ b/vetinari/types.py
@@ -10,0 +11,4 @@
+    @dataclass
+    class NewModel:
+        name: str
+        value: int
"""

_DIFF_SQLITE_TABLE = """\
diff --git a/vetinari/db.py b/vetinari/db.py
+++ b/vetinari/db.py
@@ -5,0 +6,3 @@
+    CREATE TABLE IF NOT EXISTS pipeline_metrics (
+        id TEXT PRIMARY KEY,
+        timestamp TEXT NOT NULL
"""

_DIFF_ERROR_HANDLING = """\
diff --git a/vetinari/agents/worker.py b/vetinari/agents/worker.py
+++ b/vetinari/agents/worker.py
@@ -20,0 +21,3 @@
+    except ConnectionError as exc:
+        raise InferenceError("Model server unreachable") from exc
"""

_DIFF_WITH_ADR_REF = """\
diff --git a/vetinari/foo.py b/vetinari/foo.py
+++ b/vetinari/foo.py
@@ -1,0 +2,2 @@
+# Decision: use Redis for caching (ADR-0042)
+from redis import Redis
"""


class TestExtractImplicitDecisions:
    def test_detects_new_dependency(self) -> None:
        candidates = extract_implicit_decisions(_DIFF_NEW_IMPORT)
        descriptions = [c.description for c in candidates]
        assert any("redis" in d.lower() or "requests" in d.lower() for d in descriptions)

    def test_detects_api_endpoint(self) -> None:
        candidates = extract_implicit_decisions(_DIFF_API_ENDPOINT)
        assert any(c.decision_type == DecisionType.TASK_ROUTING for c in candidates)

    def test_detects_dataclass(self) -> None:
        candidates = extract_implicit_decisions(_DIFF_DATACLASS)
        assert len(candidates) >= 1

    def test_detects_sqlite_table(self) -> None:
        candidates = extract_implicit_decisions(_DIFF_SQLITE_TABLE)
        assert any("table" in c.description.lower() for c in candidates)
        # Schema changes should have high confidence
        schema_candidates = [c for c in candidates if "table" in c.description.lower()]
        assert schema_candidates[0].confidence >= 0.7

    def test_detects_error_handling(self) -> None:
        candidates = extract_implicit_decisions(_DIFF_ERROR_HANDLING)
        assert any(c.decision_type == DecisionType.QUALITY_THRESHOLD for c in candidates)

    def test_sorted_by_confidence_descending(self) -> None:
        candidates = extract_implicit_decisions(_DIFF_NEW_IMPORT + _DIFF_SQLITE_TABLE)
        if len(candidates) >= 2:
            for i in range(len(candidates) - 1):
                assert candidates[i].confidence >= candidates[i + 1].confidence

    def test_deduplicates_descriptions(self) -> None:
        # Same diff twice should not produce duplicate descriptions
        double_diff = _DIFF_NEW_IMPORT + _DIFF_NEW_IMPORT
        candidates = extract_implicit_decisions(double_diff)
        descriptions = [c.description for c in candidates]
        assert len(descriptions) == len(set(descriptions))

    def test_empty_diff_returns_empty(self) -> None:
        candidates = extract_implicit_decisions("")
        assert candidates == []

    def test_no_added_lines_returns_empty(self) -> None:
        diff = "diff --git a/foo.py b/foo.py\n- removed line\n context line\n"
        candidates = extract_implicit_decisions(diff)
        assert candidates == []

    def test_short_match_reduces_confidence(self) -> None:
        """Very short matches (< 10 chars) get confidence reduced by 0.7 factor."""
        diff = "+raise X\n"
        candidates = extract_implicit_decisions(diff)
        if candidates:
            assert candidates[0].confidence < 0.4  # 0.4 * 0.7 = 0.28

    def test_existing_adrs_suppress_adr_flagging(self) -> None:
        candidates = extract_implicit_decisions(
            _DIFF_WITH_ADR_REF,
            context={"existing_adrs": ["0042"]},
        )
        adr_flagged = [c for c in candidates if c.adr_exists]
        # When diff references ADR-0042 and existing_adrs includes "0042", adr_exists should be True
        assert len(adr_flagged) >= 1


# -- log_extracted_decisions ---------------------------------------------------


class TestLogExtractedDecisions:
    @patch("vetinari.observability.decision_journal.get_decision_journal")
    def test_logs_high_confidence_candidates(self, mock_get_journal: MagicMock) -> None:
        mock_journal = MagicMock()
        mock_record = MagicMock()
        mock_record.decision_id = "logged-001"
        mock_journal.log_decision.return_value = mock_record
        mock_get_journal.return_value = mock_journal

        candidates = [
            CandidateDecision(description="New dep: redis", confidence=0.7),
            CandidateDecision(description="Low confidence thing", confidence=0.3),
        ]
        ids = log_extracted_decisions(candidates)
        assert len(ids) == 1  # Only high-confidence logged
        assert ids[0] == "logged-001"
        mock_journal.log_decision.assert_called_once()

    @patch("vetinari.observability.decision_journal.get_decision_journal")
    def test_skips_candidates_with_existing_adr(self, mock_get_journal: MagicMock) -> None:
        mock_journal = MagicMock()
        mock_get_journal.return_value = mock_journal

        candidates = [
            CandidateDecision(description="Already documented", confidence=0.8, adr_exists=True),
        ]
        ids = log_extracted_decisions(candidates)
        assert len(ids) == 0
        mock_journal.log_decision.assert_not_called()

    @patch("vetinari.observability.decision_journal.get_decision_journal")
    def test_logs_with_implicit_needs_review_status(self, mock_get_journal: MagicMock) -> None:
        mock_journal = MagicMock()
        mock_record = MagicMock()
        mock_record.decision_id = "rev-001"
        mock_journal.log_decision.return_value = mock_record
        mock_get_journal.return_value = mock_journal

        candidates = [CandidateDecision(description="Something", confidence=0.6)]
        log_extracted_decisions(candidates)
        call_kwargs = mock_journal.log_decision.call_args
        assert call_kwargs.kwargs.get("status") == "implicit-needs-review" or (
            call_kwargs[1].get("status") == "implicit-needs-review" if len(call_kwargs) > 1 else False
        )

    def test_journal_unavailable_returns_empty(self) -> None:
        """When journal import fails, returns empty list without crashing."""
        with patch(
            "vetinari.observability.decision_journal.get_decision_journal",
            side_effect=RuntimeError("no journal"),
        ):
            candidates = [CandidateDecision(description="Something", confidence=0.8)]
            ids = log_extracted_decisions(candidates)
            assert ids == []
