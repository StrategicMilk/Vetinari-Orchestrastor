"""Tests for rejection_reason fields on TrainingRecord and DPO export wiring.

Covers:
- TrainingRecord stores rejection_reason, rejection_category, inspector_feedback
- to_dict() serialises all three new fields
- TrainingDataCollector.record() accepts and passes through the new fields
- export_dpo_dataset() includes why_chosen_is_better / rejection_category when
  the worst record has a rejection_reason
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vetinari.learning.training_collector import TrainingDataCollector
from vetinari.learning.training_record import TrainingRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**kwargs) -> TrainingRecord:
    """Create a minimal valid TrainingRecord with optional field overrides."""
    defaults: dict = {
        "record_id": "tr_test0001",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "task": "Summarise the README",
        "prompt": "Summarise the README",
        "response": "Here is a summary.",
        "score": 0.9,
        "model_id": "test-model-7b",
        "task_type": "general",
    }
    defaults.update(kwargs)
    return TrainingRecord(**defaults)


def _make_collector(tmp_path: Path) -> TrainingDataCollector:
    """Create a sync TrainingDataCollector writing to a temp file."""
    return TrainingDataCollector(
        output_path=str(tmp_path / "training.jsonl"),
        sync=True,
    )


# ---------------------------------------------------------------------------
# TrainingRecord field tests
# ---------------------------------------------------------------------------


class TestTrainingRecordRejectionFields:
    def test_default_rejection_reason_is_empty(self):
        """rejection_reason defaults to empty string when not supplied."""
        rec = _make_record()
        assert rec.rejection_reason == ""

    def test_default_rejection_category_is_empty(self):
        """rejection_category defaults to empty string when not supplied."""
        rec = _make_record()
        assert rec.rejection_category == ""

    def test_default_inspector_feedback_is_empty(self):
        """inspector_feedback defaults to empty string when not supplied."""
        rec = _make_record()
        assert rec.inspector_feedback == ""

    def test_rejection_reason_stored(self):
        """rejection_reason is stored correctly when provided."""
        rec = _make_record(rejection_reason="Missing error handling")
        assert rec.rejection_reason == "Missing error handling"

    def test_rejection_category_stored(self):
        """rejection_category is stored correctly when provided."""
        rec = _make_record(rejection_category="missing_error_handling")
        assert rec.rejection_category == "missing_error_handling"

    def test_inspector_feedback_stored(self):
        """inspector_feedback is stored correctly when provided."""
        rec = _make_record(inspector_feedback="Output lacks try/except blocks.")
        assert rec.inspector_feedback == "Output lacks try/except blocks."

    def test_to_dict_includes_rejection_reason(self):
        """to_dict() serialises rejection_reason."""
        rec = _make_record(rejection_reason="Bad response")
        d = rec.to_dict()
        assert "rejection_reason" in d
        assert d["rejection_reason"] == "Bad response"

    def test_to_dict_includes_rejection_category(self):
        """to_dict() serialises rejection_category."""
        rec = _make_record(rejection_category="quality_rejection")
        d = rec.to_dict()
        assert "rejection_category" in d
        assert d["rejection_category"] == "quality_rejection"

    def test_to_dict_includes_inspector_feedback(self):
        """to_dict() serialises inspector_feedback."""
        rec = _make_record(inspector_feedback="Full feedback text here.")
        d = rec.to_dict()
        assert "inspector_feedback" in d
        assert d["inspector_feedback"] == "Full feedback text here."

    def test_to_dict_defaults_are_empty_strings(self):
        """to_dict() produces empty strings for unset rejection fields."""
        rec = _make_record()
        d = rec.to_dict()
        assert d["rejection_reason"] == ""
        assert d["rejection_category"] == ""
        assert d["inspector_feedback"] == ""

    def test_record_is_frozen(self):
        """TrainingRecord is frozen — rejection fields cannot be mutated."""
        rec = _make_record(rejection_reason="original")
        with pytest.raises((AttributeError, TypeError)):
            rec.rejection_reason = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TrainingDataCollector.record() passthrough tests
# ---------------------------------------------------------------------------


class TestCollectorRejectionFieldPassthrough:
    def test_record_accepts_rejection_reason(self, tmp_path):
        """record() accepts rejection_reason without error."""
        collector = _make_collector(tmp_path)
        collector.record(
            task="task",
            prompt="prompt",
            response="response text that is not a fallback",
            score=0.3,
            model_id="test-model",
            latency_ms=100,
            tokens_used=50,
            success=False,
            rejection_reason="Output too short",
        )
        jsonl_path = tmp_path / "training.jsonl"
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["rejection_reason"] == "Output too short"

    def test_record_persists_rejection_reason(self, tmp_path):
        """rejection_reason written by record() is readable from the JSONL file."""
        collector = _make_collector(tmp_path)
        collector.record(
            task="summarise",
            prompt="summarise",
            response="response that is not a fallback",
            score=0.2,
            model_id="test-model",
            latency_ms=100,
            tokens_used=50,
            success=False,
            rejection_reason="Missing citations",
            rejection_category="missing_citations",
            inspector_feedback="No sources cited.",
        )
        jsonl_path = tmp_path / "training.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec_dict = json.loads(lines[0])
        assert rec_dict["rejection_reason"] == "Missing citations"
        assert rec_dict["rejection_category"] == "missing_citations"
        assert rec_dict["inspector_feedback"] == "No sources cited."

    def test_record_accepted_output_has_empty_rejection_fields(self, tmp_path):
        """When no rejection kwargs are passed, fields default to empty strings."""
        collector = _make_collector(tmp_path)
        collector.record(
            task="good task",
            prompt="good task",
            response="great response",
            score=0.95,
            model_id="test-model",
            latency_ms=100,
            tokens_used=50,
            success=True,
        )
        jsonl_path = tmp_path / "training.jsonl"
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec_dict = json.loads(lines[0])
        assert rec_dict["rejection_reason"] == ""
        assert rec_dict["rejection_category"] == ""
        assert rec_dict["inspector_feedback"] == ""


# ---------------------------------------------------------------------------
# export_dpo_dataset() rejection signal tests
# ---------------------------------------------------------------------------


class TestDpoExportRejectionSignal:
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def _base_record(self, **overrides) -> dict:
        base = {
            "record_id": "tr_00000001",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "task": "Write a sort function",
            "prompt": "Write a sort function",
            "response": "def sort(x): return sorted(x)",
            "score": 0.9,
            "model_id": "model-a",
            "task_type": "coding",
            "prompt_variant_id": "",
            "agent_type": "",
            "latency_ms": 100,
            "tokens_used": 50,
            "success": True,
            "vram_used_gb": 0.0,
            "benchmark_suite": "",
            "benchmark_pass": False,
            "benchmark_score": 0.0,
            "rejection_reason": "",
            "rejection_category": "",
            "inspector_feedback": "",
            "metadata": {},
        }
        base.update(overrides)
        return base

    def test_why_chosen_is_better_present_when_rejected_has_reason(self, tmp_path):
        """export_dpo_dataset includes why_chosen_is_better from worst record."""
        jsonl = tmp_path / "training.jsonl"
        self._write_jsonl(
            jsonl,
            [
                self._base_record(record_id="tr_best", score=0.9, response="good response"),
                self._base_record(
                    record_id="tr_worst",
                    score=0.3,
                    response="bad response",
                    rejection_reason="Output lacked error handling",
                ),
            ],
        )
        collector = TrainingDataCollector(output_path=str(jsonl), sync=True)
        pairs = collector.export_dpo_dataset(min_score_gap=0.2)

        assert len(pairs) == 1
        pair = pairs[0]
        assert "why_chosen_is_better" in pair
        assert pair["why_chosen_is_better"] == "Output lacked error handling"

    def test_rejection_category_present_when_worst_has_category(self, tmp_path):
        """export_dpo_dataset includes rejection_category from worst record."""
        jsonl = tmp_path / "training.jsonl"
        self._write_jsonl(
            jsonl,
            [
                self._base_record(record_id="tr_best", score=0.9, response="good response"),
                self._base_record(
                    record_id="tr_worst",
                    score=0.3,
                    response="bad response",
                    rejection_reason="Missing try/except",
                    rejection_category="missing_error_handling",
                ),
            ],
        )
        collector = TrainingDataCollector(output_path=str(jsonl), sync=True)
        pairs = collector.export_dpo_dataset(min_score_gap=0.2)

        assert len(pairs) == 1
        assert pairs[0]["rejection_category"] == "missing_error_handling"

    def test_no_rejection_signal_when_worst_has_no_reason(self, tmp_path):
        """export_dpo_dataset omits why_chosen_is_better when worst has no rejection_reason."""
        jsonl = tmp_path / "training.jsonl"
        self._write_jsonl(
            jsonl,
            [
                self._base_record(record_id="tr_best", score=0.9, response="good response"),
                self._base_record(record_id="tr_worst", score=0.3, response="bad response"),
            ],
        )
        collector = TrainingDataCollector(output_path=str(jsonl), sync=True)
        pairs = collector.export_dpo_dataset(min_score_gap=0.2)

        assert len(pairs) == 1
        assert "why_chosen_is_better" not in pairs[0]
        assert "rejection_category" not in pairs[0]

    def test_standard_dpo_fields_always_present(self, tmp_path):
        """Core DPO fields are present regardless of rejection signal."""
        jsonl = tmp_path / "training.jsonl"
        self._write_jsonl(
            jsonl,
            [
                self._base_record(record_id="tr_best", score=0.9, response="good response"),
                self._base_record(
                    record_id="tr_worst",
                    score=0.3,
                    response="bad response",
                    rejection_reason="Too brief",
                ),
            ],
        )
        collector = TrainingDataCollector(output_path=str(jsonl), sync=True)
        pairs = collector.export_dpo_dataset(min_score_gap=0.2)

        pair = pairs[0]
        for key in ("prompt", "chosen", "rejected", "chosen_score", "rejected_score", "task_type"):
            assert key in pair, f"Missing standard DPO field: {key}"
