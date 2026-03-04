"""
Vetinari Drift Control Package — Phase 7

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
            print(issue)
"""

from .contract_registry import (
    ContractRegistry, ContractDriftError,
    get_contract_registry, reset_contract_registry,
)

from .capability_auditor import (
    CapabilityAuditor, CapabilityFinding,
    get_capability_auditor, reset_capability_auditor,
)

from .schema_validator import (
    SchemaValidator,
    get_schema_validator, reset_schema_validator,
)

from .monitor import (
    DriftMonitor, DriftReport,
    get_drift_monitor, reset_drift_monitor,
)

__all__ = [
    # contract registry
    "ContractRegistry", "ContractDriftError",
    "get_contract_registry", "reset_contract_registry",
    # capability auditor
    "CapabilityAuditor", "CapabilityFinding",
    "get_capability_auditor", "reset_capability_auditor",
    # schema validator
    "SchemaValidator",
    "get_schema_validator", "reset_schema_validator",
    # monitor
    "DriftMonitor", "DriftReport",
    "get_drift_monitor", "reset_drift_monitor",
]
