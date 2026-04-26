"""Vetinari system-level utilities — resource monitoring, hardware detection, health checks, and remediation."""

from __future__ import annotations

from vetinari.system.hardware_detect import (
    GpuInfo,
    GpuVendor,
    HardwareProfile,
    detect_hardware,
)
from vetinari.system.remediation import (
    FailureMode,
    RemediationAction,
    RemediationEngine,
    RemediationPlan,
    RemediationResult,
    RemediationTier,
    get_remediation_engine,
)
from vetinari.system.resource_monitor import (
    DiskStatus,
    DiskThreshold,
    check_disk_space,
    get_resource_monitor,
)

__all__ = [
    "DiskStatus",
    "DiskThreshold",
    "FailureMode",
    "GpuInfo",
    "GpuVendor",
    "HardwareProfile",
    "RemediationAction",
    "RemediationEngine",
    "RemediationPlan",
    "RemediationResult",
    "RemediationTier",
    "check_disk_space",
    "detect_hardware",
    "get_remediation_engine",
    "get_resource_monitor",
]
