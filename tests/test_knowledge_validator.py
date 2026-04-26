"""Tests for vetinari/knowledge/validator.py — KnowledgeValidator, ValidationReport, CorrectionRecord."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vetinari.knowledge.validator import (
    CorrectionRecord,
    KnowledgeValidator,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validator(tmp_path: Path) -> KnowledgeValidator:
    """Return a validator whose corrections file lives in a temp directory."""
    return KnowledgeValidator(corrections_path=tmp_path / "corrections.json")


# ---------------------------------------------------------------------------
# CorrectionRecord
# ---------------------------------------------------------------------------


class TestCorrectionRecord:
    def test_fields_stored(self) -> None:
        rec = CorrectionRecord(
            model_id="llama3",
            metric="accuracy",
            old_value=0.80,
            new_value=0.60,
            divergence=-0.20,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert rec.model_id == "llama3"
        assert rec.metric == "accuracy"
        assert rec.old_value == pytest.approx(0.80)
        assert rec.new_value == pytest.approx(0.60)
        assert rec.divergence == pytest.approx(-0.20)
        assert rec.timestamp == "2024-01-01T00:00:00+00:00"

    def test_repr_shows_key_fields(self) -> None:
        rec = CorrectionRecord(
            model_id="qwen2",
            metric="latency",
            old_value=0.5,
            new_value=0.9,
            divergence=0.4,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        r = repr(rec)
        assert "qwen2" in r
        assert "latency" in r
        assert "0.400" in r

    def test_frozen(self) -> None:
        rec = CorrectionRecord(
            model_id="x",
            metric="y",
            old_value=0.1,
            new_value=0.2,
            divergence=0.1,
            timestamp="t",
        )
        with pytest.raises(AttributeError):
            rec.model_id = "z"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_defaults(self) -> None:
        report = ValidationReport()
        assert report.corrections == []
        assert report.checked_models == 0
        # timestamp is auto-set to a non-empty ISO string
        assert "T" in report.timestamp

    def test_repr_shows_counts(self) -> None:
        report = ValidationReport(checked_models=5)
        rec = CorrectionRecord("m", "acc", 0.8, 0.5, -0.3, "t")
        report.corrections.append(rec)
        r = repr(report)
        assert "corrections=1" in r
        assert "checked_models=5" in r

    def test_mutable(self) -> None:
        """ValidationReport is NOT frozen — fields should be writable."""
        report = ValidationReport()
        report.checked_models = 10
        assert report.checked_models == 10


# ---------------------------------------------------------------------------
# KnowledgeValidator.validate — happy path
# ---------------------------------------------------------------------------


class TestValidate:
    def test_no_divergence_no_corrections(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        knowledge = {"llama3": {"accuracy": 0.85}}
        actual = {"llama3": {"accuracy": 0.85}}
        report = v.validate(knowledge, actual)
        assert report.checked_models == 1
        assert report.corrections == []

    def test_small_divergence_no_correction(self, tmp_path: Path) -> None:
        """Divergence strictly below threshold must NOT trigger correction."""
        v = _make_validator(tmp_path)
        knowledge = {"m": {"score": 0.70}}
        actual = {"m": {"score": 0.80}}  # diff = 0.10 — well below 0.15
        report = v.validate(knowledge, actual)
        assert report.corrections == []

    def test_divergence_above_threshold_triggers_correction(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        knowledge = {"llama3": {"accuracy": 0.80}}
        actual = {"llama3": {"accuracy": 0.60}}  # diff = -0.20 > 0.15
        report = v.validate(knowledge, actual)
        assert report.checked_models == 1
        assert len(report.corrections) == 1
        rec = report.corrections[0]
        assert rec.model_id == "llama3"
        assert rec.metric == "accuracy"
        assert rec.old_value == pytest.approx(0.80)
        assert rec.new_value == pytest.approx(0.60)
        assert rec.divergence == pytest.approx(-0.20)

    def test_positive_divergence_triggers_correction(self, tmp_path: Path) -> None:
        """Positive divergence (actual > predicted) should also auto-correct."""
        v = _make_validator(tmp_path)
        knowledge = {"m": {"score": 0.50}}
        actual = {"m": {"score": 0.80}}  # diff = +0.30 > 0.15
        report = v.validate(knowledge, actual)
        assert len(report.corrections) == 1
        assert report.corrections[0].divergence == pytest.approx(0.30)

    def test_multiple_models_multiple_metrics(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        knowledge = {
            "modelA": {"acc": 0.90, "speed": 0.70},
            "modelB": {"acc": 0.75},
        }
        actual = {
            "modelA": {"acc": 0.60, "speed": 0.68},  # acc diverges, speed does not
            "modelB": {"acc": 0.90},  # acc diverges
        }
        report = v.validate(knowledge, actual)
        assert report.checked_models == 2
        corrected_keys = {(r.model_id, r.metric) for r in report.corrections}
        assert ("modelA", "acc") in corrected_keys
        assert ("modelB", "acc") in corrected_keys
        assert ("modelA", "speed") not in corrected_keys

    def test_models_only_in_knowledge_are_skipped(self, tmp_path: Path) -> None:
        """Models absent from actual_data must not appear in the report."""
        v = _make_validator(tmp_path)
        knowledge = {"seen": {"acc": 0.90}, "unseen": {"acc": 0.50}}
        actual = {"seen": {"acc": 0.60}}
        report = v.validate(knowledge, actual)
        assert report.checked_models == 1
        model_ids = {r.model_id for r in report.corrections}
        assert "unseen" not in model_ids

    def test_non_numeric_metric_skipped(self, tmp_path: Path) -> None:
        """Non-numeric metric values must be skipped gracefully, not raised."""
        v = _make_validator(tmp_path)
        knowledge = {"m": {"tag": "fast", "acc": 0.90}}
        actual = {"m": {"tag": "slow", "acc": 0.60}}
        report = v.validate(knowledge, actual)
        # 'tag' is non-numeric — skipped; 'acc' diverges — corrected
        corrected_metrics = {r.metric for r in report.corrections}
        assert "tag" not in corrected_metrics
        assert "acc" in corrected_metrics

    def test_non_dict_model_entry_skipped(self, tmp_path: Path) -> None:
        """If a model entry is not a dict, it must be skipped without error."""
        v = _make_validator(tmp_path)
        knowledge = {"m": "not-a-dict"}
        actual = {"m": "also-not-a-dict"}
        report = v.validate(knowledge, actual)
        assert report.checked_models == 0
        assert report.corrections == []

    def test_empty_datasets(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        report = v.validate({}, {})
        assert report.checked_models == 0
        assert report.corrections == []


# ---------------------------------------------------------------------------
# KnowledgeValidator._auto_correct — corrections file persistence
# ---------------------------------------------------------------------------


class TestAutoCorrect:
    def test_correction_persisted_to_file(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        v._auto_correct("llama3", "accuracy", 0.80, 0.55)
        assert v._corrections_path.exists()
        items = json.loads(v._corrections_path.read_text(encoding="utf-8"))
        assert len(items) == 1
        assert items[0]["model_id"] == "llama3"
        assert items[0]["metric"] == "accuracy"
        assert items[0]["old_value"] == pytest.approx(0.80)
        assert items[0]["new_value"] == pytest.approx(0.55)
        assert items[0]["divergence"] == pytest.approx(-0.25)

    def test_corrections_accumulate(self, tmp_path: Path) -> None:
        """Each call must append rather than overwrite."""
        v = _make_validator(tmp_path)
        v._auto_correct("m1", "acc", 0.9, 0.6)
        v._auto_correct("m2", "speed", 0.5, 0.8)
        items = json.loads(v._corrections_path.read_text(encoding="utf-8"))
        assert len(items) == 2
        assert items[0]["model_id"] == "m1"
        assert items[1]["model_id"] == "m2"

    def test_validate_populates_corrections_file(self, tmp_path: Path) -> None:
        """validate() must write to disk when corrections are triggered."""
        v = _make_validator(tmp_path)
        knowledge = {"m": {"score": 0.90}}
        actual = {"m": {"score": 0.50}}
        v.validate(knowledge, actual)
        assert v._corrections_path.exists()
        items = json.loads(v._corrections_path.read_text(encoding="utf-8"))
        assert len(items) == 1

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        """Corrections file parent directories must be created if absent."""
        deep_path = tmp_path / "a" / "b" / "c" / "corrections.json"
        v = KnowledgeValidator(corrections_path=deep_path)
        v._auto_correct("m", "metric", 0.5, 0.9)
        assert deep_path.exists()

    def test_returned_record_matches_persisted(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        rec = v._auto_correct("llama3", "acc", 0.80, 0.55)
        items = json.loads(v._corrections_path.read_text(encoding="utf-8"))
        assert items[0]["model_id"] == rec.model_id
        assert items[0]["metric"] == rec.metric
        assert items[0]["divergence"] == pytest.approx(rec.divergence)
        assert items[0]["timestamp"] == rec.timestamp


# ---------------------------------------------------------------------------
# KnowledgeValidator.load_corrections
# ---------------------------------------------------------------------------


class TestLoadCorrections:
    def test_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        assert v.load_corrections() == []

    def test_round_trip(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        v._auto_correct("m1", "acc", 0.90, 0.65)
        v._auto_correct("m2", "latency", 0.20, 0.45)
        records = v.load_corrections()
        assert len(records) == 2
        assert all(isinstance(r, CorrectionRecord) for r in records)
        assert records[0].model_id == "m1"
        assert records[1].model_id == "m2"

    def test_returns_empty_on_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / "corrections.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        v = KnowledgeValidator(corrections_path=path)
        assert v.load_corrections() == []

    def test_returns_empty_on_non_list_json(self, tmp_path: Path) -> None:
        path = tmp_path / "corrections.json"
        path.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        v = KnowledgeValidator(corrections_path=path)
        assert v.load_corrections() == []

    def test_skips_malformed_entries(self, tmp_path: Path) -> None:
        """Malformed entries in the JSON list must be skipped, valid ones returned."""
        path = tmp_path / "corrections.json"
        data = [
            {
                "model_id": "good",
                "metric": "acc",
                "old_value": 0.8,
                "new_value": 0.5,
                "divergence": -0.3,
                "timestamp": "2024-01-01T00:00:00+00:00",
            },
            {"broken": True},  # missing required fields
        ]
        path.write_text(json.dumps(data), encoding="utf-8")
        v = KnowledgeValidator(corrections_path=path)
        records = v.load_corrections()
        assert len(records) == 1
        assert records[0].model_id == "good"

    def test_divergence_sign_preserved(self, tmp_path: Path) -> None:
        v = _make_validator(tmp_path)
        v._auto_correct("m", "score", 0.90, 0.60)  # divergence = -0.30
        records = v.load_corrections()
        assert records[0].divergence == pytest.approx(-0.30)


# ---------------------------------------------------------------------------
# Import wiring — public API exposed through __init__
# ---------------------------------------------------------------------------


def test_public_api_importable() -> None:
    """All three public names must be importable from vetinari.knowledge."""
    from vetinari.knowledge import (
        CorrectionRecord,
        KnowledgeValidator,
        ValidationReport,
    )

    assert CorrectionRecord is not None
    assert KnowledgeValidator is not None
    assert ValidationReport is not None


def test_default_corrections_path() -> None:
    """Default corrections_path must point to .vetinari/knowledge_corrections.json."""
    v = KnowledgeValidator()
    assert v._corrections_path == Path(".vetinari") / "knowledge_corrections.json"
