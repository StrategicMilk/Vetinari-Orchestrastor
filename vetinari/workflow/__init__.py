"""Vetinari Workflow -- manufacturing-inspired workflow controls.

Provides formal quality gates, statistical process control (SPC),
Andon stop-the-line signals, and WIP limits for the orchestration pipeline.
"""

from vetinari.workflow.quality_gates import (
    GateAction,
    WorkflowGate,
    WorkflowGateRunner,
    WORKFLOW_GATES,
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
    # quality gates
    "GateAction",
    "WorkflowGate",
    "WorkflowGateRunner",
    "WORKFLOW_GATES",
    # spc
    "AndonSignal",
    "AndonSystem",
    "ControlChart",
    "SPCAlert",
    "SPCMonitor",
    "WIPConfig",
    "WIPTracker",
]

__version__ = "1.0.0"
