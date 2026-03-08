"""
Workflow Quality Gates
======================
Formal quality gates between workflow stages, inspired by manufacturing
stage-gate processes.  Each gate defines criteria that must be satisfied
before work proceeds to the next stage, along with a prescribed failure
action (re-plan, retry, escalate, or block).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class GateAction(Enum):
    """Action to take when a quality gate fails."""

    RE_PLAN = "re_plan"
    RETRY = "retry"
    ESCALATE = "escalate"
    BLOCK = "block"
    CONTINUE = "continue"


@dataclass
class WorkflowGate:
    """A formal quality gate between workflow stages.

    Parameters
    ----------
    name:
        Human-readable identifier for the gate.
    stage:
        Pipeline stage this gate guards
        (``post_decomposition``, ``post_execution``,
        ``post_verification``, ``pre_assembly``).
    criteria:
        Key/value pairs that incoming metrics must satisfy.
        Numeric values are treated as *minimum* thresholds;
        boolean values must match exactly.
    failure_action:
        What to do when the gate is not passed.
    """

    name: str
    stage: str
    criteria: Dict[str, Any]
    failure_action: GateAction


# ---------------------------------------------------------------------------
# Default gate definitions
# ---------------------------------------------------------------------------

WORKFLOW_GATES: Dict[str, WorkflowGate] = {
    "post_decomposition": WorkflowGate(
        name="decomposition_gate",
        stage="post_decomposition",
        criteria={
            "min_tasks": 1,
            "max_tasks": 50,
            "dag_valid": True,
            "goal_adherence": 0.7,
        },
        failure_action=GateAction.RE_PLAN,
    ),
    "post_execution": WorkflowGate(
        name="execution_gate",
        stage="post_execution",
        criteria={
            "quality_score": 0.6,
            "no_critical_security": True,
        },
        failure_action=GateAction.RETRY,
    ),
    "post_verification": WorkflowGate(
        name="verification_gate",
        stage="post_verification",
        criteria={
            "verification_passed": True,
            "drift_score_max": 0.5,
        },
        failure_action=GateAction.ESCALATE,
    ),
    "pre_assembly": WorkflowGate(
        name="assembly_gate",
        stage="pre_assembly",
        criteria={
            "all_dependencies_met": True,
            "no_blocked_tasks": True,
        },
        failure_action=GateAction.BLOCK,
    ),
}


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------

class WorkflowGateRunner:
    """Evaluates workflow gates and determines actions.

    Usage::

        runner = WorkflowGateRunner()
        passed, action, violations = runner.evaluate(
            "post_execution",
            {"quality_score": 0.8, "no_critical_security": True},
        )
    """

    def __init__(self, gates: Optional[Dict[str, WorkflowGate]] = None):
        self._gates: Dict[str, WorkflowGate] = gates if gates is not None else dict(WORKFLOW_GATES)
        self._history: List[Dict[str, Any]] = []

    # -- public API ---------------------------------------------------------

    def evaluate(
        self,
        stage: str,
        metrics: Dict[str, Any],
    ) -> Tuple[bool, GateAction, List[str]]:
        """Evaluate the gate for *stage* against *metrics*.

        Returns
        -------
        passed : bool
            ``True`` when every criterion is satisfied.
        action : GateAction
            ``CONTINUE`` on success, otherwise the gate's ``failure_action``.
        violations : list[str]
            Human-readable descriptions of each failed criterion.
        """
        gate = self._gates.get(stage)
        if gate is None:
            logger.warning("No gate defined for stage %r -- passing by default", stage)
            return True, GateAction.CONTINUE, []

        violations: List[str] = []

        for key, threshold in gate.criteria.items():
            value = metrics.get(key)

            if value is None:
                violations.append(f"Missing metric: {key}")
                continue

            if isinstance(threshold, bool):
                if value != threshold:
                    violations.append(
                        f"{key}: expected {threshold}, got {value}"
                    )
            elif isinstance(threshold, (int, float)):
                # For keys containing "max" the threshold is an upper bound;
                # otherwise it is a lower bound.
                if "max" in key:
                    if value > threshold:
                        violations.append(
                            f"{key}: {value} exceeds maximum {threshold}"
                        )
                else:
                    if value < threshold:
                        violations.append(
                            f"{key}: {value} below minimum {threshold}"
                        )

        passed = not violations
        action = GateAction.CONTINUE if passed else gate.failure_action

        # Record history entry
        entry = {
            "stage": stage,
            "gate": gate.name,
            "passed": passed,
            "action": action.value,
            "violations": list(violations),
            "metrics": dict(metrics),
            "timestamp": datetime.now().isoformat(),
        }
        self._history.append(entry)

        if passed:
            logger.info("Gate %s PASSED for stage %s", gate.name, stage)
        else:
            logger.warning(
                "Gate %s FAILED for stage %s: %s -> %s",
                gate.name, stage, violations, action.value,
            )

        return passed, action, violations

    def get_history(self) -> List[Dict[str, Any]]:
        """Return an ordered list of all gate evaluation results."""
        return list(self._history)

    # -- customisation helpers ----------------------------------------------

    def add_gate(self, stage: str, gate: WorkflowGate) -> None:
        """Register (or replace) a gate for *stage*."""
        self._gates[stage] = gate

    def remove_gate(self, stage: str) -> Optional[WorkflowGate]:
        """Remove and return the gate for *stage*, or ``None``."""
        return self._gates.pop(stage, None)

    @property
    def gates(self) -> Dict[str, WorkflowGate]:
        """Read-only view of the current gate map."""
        return dict(self._gates)
