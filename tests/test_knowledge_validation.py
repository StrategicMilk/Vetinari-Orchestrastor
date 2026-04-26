"""Tests for knowledge validator — divergence detection, auto-correction, persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vetinari.knowledge.validator import (
    CorrectionRecord,
    KnowledgeValidator,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_knowledge(model_id: str, **metrics: float) -> dict[str, Any]:
    """Build a single-model knowledge_data dict."""
    return {model_id: dict(metrics)}


def _make_actual(model_id: str, **metrics: float) -> dict[str, Any]:
    """Build a single-model actual_data dict."""
    return {model_id: dict(metrics)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_divergence_no_corrections(tmp_path: Path) -> None:
    """knowledge_data and actual_data match within 0.15 threshold — no corrections produced."""
    corrections_file = tmp_path / "corrections.json"
    validator = KnowledgeValidator(corrections_path=corrections_file)

    knowledge = _make_knowledge("model_a", quality=0.9, latency=0.5)
    actual = _make_actual("model_a", quality=0.9, latency=0.5)

    report = validator.validate(knowledge, actual)

    assert report.corrections == [], "Expected no corrections when values match"
    assert report.checked_models == 1
    # File should not have been created because nothing was written
    assert not corrections_file.exists()


def test_divergence_below_threshold_produces_no_correction(tmp_path: Path) -> None:
    """Divergence of exactly 0.14 (< 0.15 threshold) must NOT trigger a correction."""
    corrections_file = tmp_path / "corrections.json"
    validator = KnowledgeValidator(corrections_path=corrections_file)

    knowledge = _make_knowledge("model_a", quality=0.90)
    actual = _make_actual("model_a", quality=0.76)  # abs(0.76 - 0.90) = 0.14

    report = validator.validate(knowledge, actual)

    assert report.corrections == []
    assert not corrections_file.exists()


def test_divergence_triggers_correction(tmp_path: Path) -> None:
    """knowledge says model_a quality=0.9, actual=0.7 — divergence=0.2 > 0.15 triggers correction."""
    corrections_file = tmp_path / "corrections.json"
    validator = KnowledgeValidator(corrections_path=corrections_file)

    knowledge = _make_knowledge("model_a", quality=0.9)
    actual = _make_actual("model_a", quality=0.7)

    report = validator.validate(knowledge, actual)

    assert len(report.corrections) == 1
    rec = report.corrections[0]
    assert isinstance(rec, CorrectionRecord)
    assert rec.model_id == "model_a"
    assert rec.metric == "quality"
    assert rec.old_value == pytest.approx(0.9)
    assert rec.new_value == pytest.approx(0.7)
    assert rec.divergence == pytest.approx(-0.2)


def test_correction_persisted_to_file(tmp_path: Path) -> None:
    """After a correction, the JSON file at corrections_path contains the record."""
    corrections_file = tmp_path / "subdir" / "corrections.json"
    validator = KnowledgeValidator(corrections_path=corrections_file)

    knowledge = _make_knowledge("model_b", accuracy=0.95)
    actual = _make_actual("model_b", accuracy=0.70)  # divergence = -0.25

    validator.validate(knowledge, actual)

    assert corrections_file.exists(), "Corrections file should be created"
    raw = json.loads(corrections_file.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert len(raw) == 1
    assert raw[0]["model_id"] == "model_b"
    assert raw[0]["metric"] == "accuracy"
    assert abs(raw[0]["divergence"]) > 0.15


def test_load_corrections_reads_file(tmp_path: Path) -> None:
    """Write corrections JSON directly, then load_corrections returns matching CorrectionRecords."""
    corrections_file = tmp_path / "corrections.json"
    now = "2026-01-01T00:00:00+00:00"
    data = [
        {
            "model_id": "model_x",
            "metric": "latency",
            "old_value": 0.8,
            "new_value": 0.5,
            "divergence": -0.3,
            "timestamp": now,
        }
    ]
    corrections_file.write_text(json.dumps(data), encoding="utf-8")

    validator = KnowledgeValidator(corrections_path=corrections_file)
    records = validator.load_corrections()

    assert len(records) == 1
    rec = records[0]
    assert rec.model_id == "model_x"
    assert rec.metric == "latency"
    assert rec.old_value == pytest.approx(0.8)
    assert rec.new_value == pytest.approx(0.5)
    assert rec.divergence == pytest.approx(-0.3)
    assert rec.timestamp == now


def test_multiple_models_validated(tmp_path: Path) -> None:
    """3 models validated, only 1 diverges — 1 correction produced, checked_models=3."""
    corrections_file = tmp_path / "corrections.json"
    validator = KnowledgeValidator(corrections_path=corrections_file)

    knowledge = {
        "model_a": {"quality": 0.9},
        "model_b": {"quality": 0.85},
        "model_c": {"quality": 0.7},
    }
    actual = {
        "model_a": {"quality": 0.9},  # no divergence
        "model_b": {"quality": 0.82},  # 0.03 < 0.15, no correction
        "model_c": {"quality": 0.45},  # 0.25 > 0.15, triggers correction
    }

    report = validator.validate(knowledge, actual)

    assert report.checked_models == 3
    assert len(report.corrections) == 1
    assert report.corrections[0].model_id == "model_c"


def test_validation_report_structure(tmp_path: Path) -> None:
    """ValidationReport has the expected fields and types."""
    corrections_file = tmp_path / "corrections.json"
    validator = KnowledgeValidator(corrections_path=corrections_file)

    knowledge = _make_knowledge("model_z", score=0.6)
    actual = _make_actual("model_z", score=0.6)

    report = validator.validate(knowledge, actual)

    assert isinstance(report, ValidationReport)
    assert isinstance(report.corrections, list)
    assert isinstance(report.checked_models, int)
    assert isinstance(report.timestamp, str)
    assert len(report.timestamp) > 10  # non-empty ISO-8601 string


def test_load_corrections_missing_file(tmp_path: Path) -> None:
    """load_corrections returns empty list gracefully when the file does not exist."""
    validator = KnowledgeValidator(corrections_path=tmp_path / "nonexistent.json")
    assert validator.load_corrections() == []


def test_models_absent_from_actual_not_checked(tmp_path: Path) -> None:
    """Models present only in knowledge_data (not in actual_data) are skipped silently."""
    corrections_file = tmp_path / "corrections.json"
    validator = KnowledgeValidator(corrections_path=corrections_file)

    knowledge = {
        "model_a": {"quality": 0.9},
        "model_b": {"quality": 0.9},  # not in actual
    }
    actual = {"model_a": {"quality": 0.6}}  # divergence triggers for model_a only

    report = validator.validate(knowledge, actual)

    assert report.checked_models == 1
    assert len(report.corrections) == 1
    assert report.corrections[0].model_id == "model_a"
