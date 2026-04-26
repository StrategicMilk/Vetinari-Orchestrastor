"""Tests for vetinari.validation.wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.factories import make_gate_check_result, make_validation_verification_result
from vetinari.validation.quality_gates import GateResult
from vetinari.validation.verification import VerificationStatus
from vetinari.validation.wiring import (
    StageGateResult,
    VerificationSummary,
    run_stage_gate,
    verify_worker_output,
    wire_validation_subsystem,
)

# ---------------------------------------------------------------------------
# StageGateResult dataclass
# ---------------------------------------------------------------------------


def test_stage_gate_result_fields() -> None:
    result = StageGateResult(
        passed=True,
        should_rework=False,
        gate_results=[],
        summary="ok",
    )
    assert result.passed is True
    assert result.should_rework is False
    assert result.gate_results == []
    assert result.summary == "ok"


# ---------------------------------------------------------------------------
# VerificationSummary dataclass
# ---------------------------------------------------------------------------


def test_verification_summary_fields() -> None:
    summary = VerificationSummary(
        passed=True,
        error_count=0,
        warning_count=1,
        results={},
        recommendation="rework",
    )
    assert summary.passed is True
    assert summary.recommendation == "rework"


# ---------------------------------------------------------------------------
# run_stage_gate
# ---------------------------------------------------------------------------


def test_run_stage_gate_all_pass() -> None:
    passing = make_gate_check_result(result=GateResult.PASSED, score=0.9)
    with patch(
        "vetinari.validation.wiring.QualityGateRunner.run_gate",
        return_value=[passing],
    ):
        result = run_stage_gate("post_execution", {"code": "x = 1"})

    assert isinstance(result, StageGateResult)
    assert result.passed is True
    assert result.should_rework is False
    assert len(result.gate_results) == 1
    assert "passed" in result.summary.lower()


def test_run_stage_gate_one_failure() -> None:
    failing = make_gate_check_result(result=GateResult.FAILED, score=0.2)
    with patch(
        "vetinari.validation.wiring.QualityGateRunner.run_gate",
        return_value=[failing],
    ):
        result = run_stage_gate("post_execution", {})

    assert result.passed is False
    assert result.should_rework is True
    assert "failed" in result.summary.lower()
    assert "test_gate" in result.summary


def test_run_stage_gate_no_gates_returns_passed() -> None:
    """A stage with no configured gates should not trigger rework."""
    with patch(
        "vetinari.validation.wiring.QualityGateRunner.run_gate",
        return_value=[],
    ):
        result = run_stage_gate("unknown_stage", {})

    assert result.passed is True
    assert result.should_rework is False


def test_run_stage_gate_emits_event() -> None:
    """Event is published when the gate passes."""
    passing = make_gate_check_result(result=GateResult.PASSED)
    mock_bus = MagicMock()

    import vetinari.events as _events_mod

    with (
        patch("vetinari.validation.wiring.QualityGateRunner.run_gate", return_value=[passing]),
        patch.object(_events_mod, "get_event_bus", return_value=mock_bus),
    ):
        run_stage_gate("post_planning", {})

    mock_bus.publish.assert_called_once()
    event = mock_bus.publish.call_args[0][0]
    assert event.task_id == "post_planning"
    assert event.passed is True


def test_run_stage_gate_event_bus_unavailable_does_not_raise() -> None:
    """EventBus raising must log a warning and still return a valid StageGateResult."""
    passing = make_gate_check_result(result=GateResult.PASSED)

    import vetinari.events as _events_mod

    with (
        patch("vetinari.validation.wiring.QualityGateRunner.run_gate", return_value=[passing]),
        patch.object(_events_mod, "get_event_bus", side_effect=RuntimeError("no bus")),
    ):
        result = run_stage_gate("post_execution", {})

    assert result.passed is True  # normal result still returned


# ---------------------------------------------------------------------------
# verify_worker_output
# ---------------------------------------------------------------------------


def test_verify_worker_output_accept() -> None:
    clean = make_validation_verification_result(VerificationStatus.PASSED)
    with patch("vetinari.validation.wiring.get_verifier_pipeline") as mock_get:
        mock_pipeline = MagicMock()
        mock_pipeline.verify.return_value = {"syntax": clean}
        mock_get.return_value = mock_pipeline

        summary = verify_worker_output("x = 1")

    assert summary.passed is True
    assert summary.recommendation == "accept"
    assert summary.error_count == 0
    assert summary.warning_count == 0


def test_verify_worker_output_reject_on_error() -> None:
    failed = make_validation_verification_result(VerificationStatus.FAILED, errors=2)
    with patch("vetinari.validation.wiring.get_verifier_pipeline") as mock_get:
        mock_pipeline = MagicMock()
        mock_pipeline.verify.return_value = {"syntax": failed}
        mock_get.return_value = mock_pipeline

        summary = verify_worker_output("bad code!!!!")

    assert summary.passed is False
    assert summary.recommendation == "reject"
    assert summary.error_count == 2


def test_verify_worker_output_rework_on_warnings_only() -> None:
    warned = make_validation_verification_result(VerificationStatus.WARNING, warnings=1)
    with patch("vetinari.validation.wiring.get_verifier_pipeline") as mock_get:
        mock_pipeline = MagicMock()
        mock_pipeline.verify.return_value = {"imports": warned}
        mock_get.return_value = mock_pipeline

        summary = verify_worker_output("import ctypes")

    assert summary.passed is True
    assert summary.recommendation == "rework"
    assert summary.warning_count == 1


def test_verify_worker_output_uses_singleton() -> None:
    """verify_worker_output must reuse the singleton, not create a new pipeline."""
    with patch("vetinari.validation.wiring.get_verifier_pipeline") as mock_get:
        mock_pipeline = MagicMock()
        mock_pipeline.verify.return_value = {}
        mock_get.return_value = mock_pipeline

        verify_worker_output("hello")
        verify_worker_output("world")

    assert mock_get.call_count == 2  # called twice, once per invocation
    assert mock_pipeline.verify.call_count == 2


# ---------------------------------------------------------------------------
# wire_validation_subsystem
# ---------------------------------------------------------------------------


def test_wire_validation_subsystem_calls_singleton() -> None:
    with patch("vetinari.validation.wiring.get_verifier_pipeline") as mock_get:
        wire_validation_subsystem()

    mock_get.assert_called_once_with()


def test_wire_validation_subsystem_does_not_raise(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="vetinari.validation.wiring"):
        wire_validation_subsystem()
    assert any("validation" in r.message.lower() for r in caplog.records)
