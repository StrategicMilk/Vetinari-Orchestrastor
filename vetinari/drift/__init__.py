"""Vetinari Drift Control Package — Phase 7.

Provides three complementary drift-detection mechanisms:

    contract_registry   Hash-based fingerprinting of dataclass contracts.
    capability_auditor  Live vs. documented agent capability comparison.
    schema_validator    Structural validation of contract instances.
    monitor             Orchestrates all three; produces DriftReport.

Quick-start
-----------
    from vetinari.drift import get_drift_monitor

    monitor = get_drift_monitor()
    monitor.bootstrap()                 # seed baselines from live code
    report = monitor.run_full_audit()

    if not report.is_clean:
        for issue in report.issues:
            logger.debug(issue)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from .capability_auditor import (  # noqa: E402 - late import is required after bootstrap setup
    CapabilityAuditor,
    CapabilityFinding,
    get_capability_auditor,
    reset_capability_auditor,
)
from .contract_registry import (  # noqa: E402 - late import is required after bootstrap setup
    ContractDriftError,
    ContractRegistry,
    get_contract_registry,
    reset_contract_registry,
)
from .goal_tracker import (  # noqa: E402, VET123 - late import is required after bootstrap setup; barrel export preserves public import compatibility
    AdherenceResult,
    GoalTracker,
)
from .monitor import (  # noqa: E402 - late import is required after bootstrap setup
    DriftMonitor,
    DriftMonitorReport,
    get_drift_monitor,
    reset_drift_monitor,
)
from .schema_validator import (  # noqa: E402 - late import is required after bootstrap setup
    SchemaValidator,
    get_schema_validator,
    reset_schema_validator,
)
from .wiring import (  # noqa: E402 - late import is required after bootstrap setup
    check_goal_adherence,
    schedule_contract_check,
    schedule_drift_audit,
    startup_drift_validation,
    wire_drift_subsystem,
)

__all__ = [
    "AdherenceResult",
    "CapabilityAuditor",
    "CapabilityFinding",
    "ContractDriftError",
    "ContractRegistry",
    "DriftMonitor",
    "DriftMonitorReport",
    "DriftReport",
    "GoalTracker",
    "SchemaValidator",
    "check_goal_adherence",
    "get_capability_auditor",
    "get_contract_registry",
    "get_drift_monitor",
    "get_schema_validator",
    "reset_capability_auditor",
    "reset_contract_registry",
    "reset_drift_monitor",
    "reset_schema_validator",
    "schedule_contract_check",
    "schedule_drift_audit",
    "startup_drift_validation",
    "wire_drift_subsystem",
]
