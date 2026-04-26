"""Tests for vetinari.drift.wiring — startup and scheduled drift integration hooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.drift.monitor import DriftMonitorReport


@pytest.fixture(autouse=True)
def _reset_singletons():
    from vetinari.drift.contract_registry import reset_contract_registry
    from vetinari.drift.monitor import reset_drift_monitor
    from vetinari.drift.schema_validator import reset_schema_validator

    reset_drift_monitor()
    reset_schema_validator()
    reset_contract_registry()
    yield
    reset_drift_monitor()
    reset_schema_validator()
    reset_contract_registry()


class TestStartupDriftValidation:
    """Tests for startup_drift_validation()."""

    def test_startup_drift_validation_succeeds(self):
        """Returns True when all registered schemas validate cleanly."""
        from vetinari.drift.wiring import startup_drift_validation

        mock_monitor = MagicMock()
        mock_validator = MagicMock()
        mock_validator.list_schemas.return_value = ["plan_schema", "task_schema"]
        # validate() returning empty list means no errors
        mock_validator.validate.return_value = []

        with (
            patch("vetinari.drift.wiring.get_drift_monitor", return_value=mock_monitor),
            patch("vetinari.drift.wiring.get_schema_validator", return_value=mock_validator),
        ):
            result = startup_drift_validation()

        assert result is True
        mock_monitor.bootstrap.assert_called_once()
        assert mock_validator.validate.call_count == 2

    def test_startup_drift_validation_detects_invalid(self):
        """Returns False when SchemaValidator reports errors for a schema."""
        from vetinari.drift.wiring import startup_drift_validation

        mock_monitor = MagicMock()
        mock_validator = MagicMock()
        mock_validator.list_schemas.return_value = ["bad_schema"]
        mock_validator.validate.return_value = ["required field 'id' missing"]

        with (
            patch("vetinari.drift.wiring.get_drift_monitor", return_value=mock_monitor),
            patch("vetinari.drift.wiring.get_schema_validator", return_value=mock_validator),
        ):
            result = startup_drift_validation()

        assert result is False

    def test_startup_drift_validation_no_schemas(self):
        """Returns True (vacuously valid) when no schemas are registered."""
        from vetinari.drift.wiring import startup_drift_validation

        mock_monitor = MagicMock()
        mock_validator = MagicMock()
        mock_validator.list_schemas.return_value = []

        with (
            patch("vetinari.drift.wiring.get_drift_monitor", return_value=mock_monitor),
            patch("vetinari.drift.wiring.get_schema_validator", return_value=mock_validator),
        ):
            result = startup_drift_validation()

        assert result is True
        mock_validator.validate.assert_not_called()


class TestScheduleDriftAudit:
    """Tests for schedule_drift_audit()."""

    def test_schedule_drift_audit_clean(self):
        """Returns a clean DriftReport when the monitor finds no drift."""
        from vetinari.drift.wiring import schedule_drift_audit

        clean_report = DriftMonitorReport()
        mock_monitor = MagicMock()
        mock_monitor.run_full_audit.return_value = clean_report

        with patch("vetinari.drift.wiring.get_drift_monitor", return_value=mock_monitor):
            report = schedule_drift_audit()

        assert report.is_clean is True
        mock_monitor.run_full_audit.assert_called_once()

    def test_schedule_drift_audit_with_drift(self):
        """Returns a report with contract drifts and emits a warning."""
        from vetinari.drift.wiring import schedule_drift_audit

        drifted_report = DriftMonitorReport(
            contract_drifts={"AgentSpec": {"previous": "abc123", "current": "def456"}},
            issues=["AgentSpec hash changed"],
        )
        mock_monitor = MagicMock()
        mock_monitor.run_full_audit.return_value = drifted_report

        with patch("vetinari.drift.wiring.get_drift_monitor", return_value=mock_monitor):
            with patch("vetinari.drift.wiring.logger") as mock_logger:
                report = schedule_drift_audit()

        assert report.is_clean is False
        assert len(report.contract_drifts) == 1
        mock_logger.warning.assert_called()


class TestScheduleContractCheck:
    """Tests for schedule_contract_check()."""

    def test_schedule_contract_check_no_drift(self):
        """Calls snapshot() when all contracts are clean."""
        from vetinari.drift.wiring import schedule_contract_check

        mock_registry = MagicMock()
        mock_registry.check_drift.return_value = {}
        mock_registry.list_contracts.return_value = ["Task", "Plan", "AgentSpec"]

        with patch("vetinari.drift.wiring.get_contract_registry", return_value=mock_registry):
            result = schedule_contract_check()

        assert result == {}
        mock_registry.load_snapshot.assert_called_once()
        mock_registry.snapshot.assert_called_once()

    def test_schedule_contract_check_with_drift(self):
        """Returns drift dict and does NOT call snapshot() when drift detected."""
        from vetinari.drift.wiring import schedule_contract_check

        drift_result = {"Task": {"previous": "aaa", "current": "bbb"}}
        mock_registry = MagicMock()
        mock_registry.check_drift.return_value = drift_result

        with patch("vetinari.drift.wiring.get_contract_registry", return_value=mock_registry):
            result = schedule_contract_check()

        assert result == drift_result
        mock_registry.snapshot.assert_not_called()


class TestCheckGoalAdherence:
    """Tests for check_goal_adherence()."""

    def test_check_goal_adherence_high(self):
        """Returns high score when goal and output share relevant keywords."""
        from vetinari.drift.wiring import check_goal_adherence

        # Use closely related terms so keyword matching produces a decent score
        result = check_goal_adherence(
            original_goal="generate a Python report with cost analysis",
            task_output="The Python report includes a full cost analysis section with charts.",
            task_description="generate Python cost report",
        )

        # Score should be above the low-adherence threshold (0.4)
        assert result.score >= 0.4
        assert 0.0 <= result.score <= 1.0

    def test_check_goal_adherence_low(self):
        """Returns a low score and emits a warning for unrelated goal/output."""
        from vetinari.drift.wiring import check_goal_adherence

        with patch("vetinari.drift.wiring.logger") as mock_logger:
            result = check_goal_adherence(
                original_goal="zzz quantum chromodynamics zzz",
                task_output="breakfast cereal recipe with milk",
                task_description="breakfast task",
            )

        # Score should be at or near zero — no keyword overlap at all
        assert result.score < 0.4
        mock_logger.warning.assert_called()

    def test_check_goal_adherence_returns_adherence_result(self):
        """Return value has the expected fields from AdherenceResult."""
        from vetinari.drift.goal_tracker import AdherenceResult
        from vetinari.drift.wiring import check_goal_adherence

        result = check_goal_adherence(
            original_goal="write tests for the agent module",
            task_output="wrote unit tests for agent module",
            task_description="test writing task",
        )

        assert isinstance(result, AdherenceResult)
        assert hasattr(result, "score")
        assert hasattr(result, "keywords_matched")
        assert hasattr(result, "keywords_total")


class TestWireDriftSubsystem:
    """Tests for wire_drift_subsystem()."""

    def test_wire_drift_subsystem_calls_startup_validation(self, caplog):
        """wire_drift_subsystem() calls startup_drift_validation()."""
        import logging

        from vetinari.drift.wiring import wire_drift_subsystem

        with (
            patch("vetinari.drift.wiring.startup_drift_validation", return_value=True) as mock_validate,
            caplog.at_level(logging.INFO, logger="vetinari.drift.wiring"),
        ):
            wire_drift_subsystem()

        mock_validate.assert_called_once()
        assert any("wired successfully" in record.message.lower() for record in caplog.records)

    def test_wire_drift_subsystem_handles_invalid_schemas(self, caplog):
        """wire_drift_subsystem() logs a warning (not error) when validation returns False."""
        import logging

        from vetinari.drift.wiring import wire_drift_subsystem

        with (
            patch("vetinari.drift.wiring.startup_drift_validation", return_value=False),
            caplog.at_level(logging.WARNING, logger="vetinari.drift.wiring"),
        ):
            wire_drift_subsystem()
        # Should have logged a warning about schema issues, not raised
        assert any("warning" in r.message.lower() or "warn" in r.levelname.lower() for r in caplog.records)
