"""Vetinari Workflow -- manufacturing-inspired workflow controls.

Provides formal quality gates, statistical process control (SPC),
Andon stop-the-line signals, and WIP limits for the orchestration pipeline.
"""

from __future__ import annotations

from vetinari.workflow.quality_gates import (
    WORKFLOW_GATES,
    GateAction,
    WorkflowGate,
    WorkflowGateRunner,
)
from vetinari.workflow.spc import (
    AndonSignal,
    AndonSystem,
    ControlChart,
    SPCAlert,
    SPCMonitor,
    WIPConfig,
    WIPTracker,
)

__all__ = [
    "WORKFLOW_GATES",
    # spc
    "AndonSignal",
    "AndonSystem",
    "ControlChart",
    # quality gates
    "GateAction",
    "SPCAlert",
    "SPCMonitor",
    "WIPConfig",
    "WIPTracker",
    "WorkflowGate",
    "WorkflowGateRunner",
]

__version__ = "1.0.0"
