"""Integration tests verifying all Session 4B wiring entry points are callable."""

from __future__ import annotations

import pytest


class TestSession4BWiringEntryPoints:
    """Verify every wire_*_subsystem() function runs without error."""

    def test_wire_drift_subsystem(self, caplog):
        import logging

        from vetinari.drift.wiring import wire_drift_subsystem

        with caplog.at_level(logging.INFO, logger="vetinari.drift.wiring"):
            wire_drift_subsystem()
        assert len(caplog.records) >= 1

    def test_wire_validation_subsystem(self, caplog):
        import logging

        from vetinari.validation.wiring import wire_validation_subsystem

        with caplog.at_level(logging.INFO, logger="vetinari.validation.wiring"):
            wire_validation_subsystem()
        assert any("validation" in r.message.lower() for r in caplog.records)

    def test_wire_resilience_subsystem(self, caplog):
        import logging

        from vetinari.resilience.wiring import wire_resilience_subsystem

        with caplog.at_level(logging.INFO, logger="vetinari.resilience.wiring"):
            wire_resilience_subsystem()
        assert any("ready" in r.message for r in caplog.records)

    def test_wire_workflow_subsystem(self, caplog):
        import logging

        from vetinari.workflow.wiring import wire_workflow_subsystem

        with caplog.at_level(logging.INFO, logger="vetinari.workflow.wiring"):
            wire_workflow_subsystem()
        assert any("ready" in r.message for r in caplog.records)

    def test_wire_kaizen_subsystem(self, caplog):
        import logging

        from vetinari.kaizen.wiring import wire_kaizen_subsystem

        with caplog.at_level(logging.INFO, logger="vetinari.kaizen.wiring"):
            wire_kaizen_subsystem()
        assert any("ready" in r.message for r in caplog.records)


class TestSession4BImports:
    """Verify all new modules are importable from their packages."""

    def test_drift_package_exports(self):
        from vetinari.drift import (
            check_goal_adherence,
            schedule_contract_check,
            schedule_drift_audit,
            startup_drift_validation,
            wire_drift_subsystem,
        )

        assert callable(startup_drift_validation)

    def test_resilience_package_exports(self):
        from vetinari.resilience import (
            call_with_breaker,
            check_breaker_health,
            get_all_breaker_health,
            get_inference_breaker,
            wire_resilience_subsystem,
        )

        assert callable(call_with_breaker)

    def test_workflow_package_exports(self):
        from vetinari.workflow import (
            check_andon_before_dispatch,
            complete_and_pull,
            dispatch_or_queue,
            get_dispatch_status,
            raise_quality_andon,
            wire_workflow_subsystem,
        )

        assert callable(dispatch_or_queue)

    def test_kaizen_package_exports(self):
        from vetinari.kaizen import (
            scheduled_pdca_check,
            scheduled_regression_check,
            scheduled_trend_analysis,
            wire_kaizen_subsystem,
        )

        assert callable(scheduled_pdca_check)

    def test_validation_package_exports(self):
        from vetinari.validation import (
            StageGateResult,
            VerificationSummary,
        )

        assert isinstance(StageGateResult, type)
        assert isinstance(VerificationSummary, type)

    def test_remediation_engine_importable(self):
        from vetinari.system.remediation import (
            FailureMode,
            RemediationEngine,
            RemediationTier,
            get_remediation_engine,
        )

        assert callable(get_remediation_engine)
