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

from .capability_auditor import (  # noqa: E402
    CapabilityAuditor,
    CapabilityFinding,
    get_capability_auditor,
    reset_capability_auditor,
)
from .contract_registry import (  # noqa: E402
    ContractDriftError,
    ContractRegistry,
    get_contract_registry,
    reset_contract_registry,
)
from .goal_tracker import (  # noqa: E402
    AdherenceResult,
    GoalTracker,
    ScopeCreepItem,
    create_goal_tracker,
)
from .monitor import (  # noqa: E402
    DriftMonitor,
    DriftReport,
    get_drift_monitor,
    reset_drift_monitor,
)
from .schema_validator import (  # noqa: E402
    SchemaValidator,
    get_schema_validator,
    reset_schema_validator,
)

__all__ = [
    "AdherenceResult",
    # capability auditor
    "CapabilityAuditor",
    "CapabilityFinding",
    "ContractDriftError",
    # contract registry
    "ContractRegistry",
    # monitor
    "DriftMonitor",
    "DriftReport",
    # goal tracker
    "GoalTracker",
    # schema validator
    "SchemaValidator",
    "ScopeCreepItem",
    "create_goal_tracker",
    "get_capability_auditor",
    "get_contract_registry",
    "get_drift_monitor",
    "get_schema_validator",
    "reset_capability_auditor",
    "reset_contract_registry",
    "reset_drift_monitor",
    "reset_schema_validator",
]
