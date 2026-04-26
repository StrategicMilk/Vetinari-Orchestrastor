"""Milestone Checkpoint System — structured approval gates during execution.

Provides MilestonePolicy configuration and MilestoneReached/MilestoneApproval
events for pausing execution at feature boundaries.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)


class MilestoneGranularity(Enum):
    """How often to pause for user approval."""

    ALL = "all"  # Checkpoint after every task
    FEATURES = "features"  # Checkpoint after each feature/component (DEFAULT)
    PHASES = "phases"  # Checkpoint only at phase transitions
    CRITICAL = "critical"  # Checkpoint only on high-risk/destructive tasks
    NONE = "none"  # No checkpoints (fully autonomous)


class MilestoneAction(Enum):
    """User response to a milestone checkpoint."""

    APPROVE = "approve"
    REVISE = "revise"
    SKIP_REMAINING = "skip_remaining"
    ABORT = "abort"


@dataclass
class MilestoneReached:
    """Event emitted when a milestone is reached during execution."""

    milestone_name: str
    task_id: str
    task_description: str
    completed_tasks: list[str]
    test_results: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    next_steps: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"MilestoneReached(milestone_name={self.milestone_name!r}, task_id={self.task_id!r}, "
            f"quality_score={self.quality_score!r}, completed_tasks={len(self.completed_tasks)})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the milestone event to a plain dictionary suitable for JSON export."""
        return dataclass_to_dict(self)

    def summary(self) -> str:
        """One-line summary for CLI display."""
        return (
            f"Milestone '{self.milestone_name}': {self.task_description} "
            f"(quality={self.quality_score:.1f}, {len(self.completed_tasks)} tasks done)"
        )


@dataclass
class MilestoneApproval:
    """User's response to a milestone checkpoint."""

    action: MilestoneAction
    feedback: str = ""
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"MilestoneApproval(action={self.action.value!r})"

    @classmethod
    def approve(cls) -> MilestoneApproval:
        """Create an approval response that allows execution to continue.

        Returns:
            MilestoneApproval with APPROVE action.
        """
        return cls(action=MilestoneAction.APPROVE)

    @classmethod
    def revise(cls, feedback: str) -> MilestoneApproval:
        """Create a revision response with feedback for the execution engine.

        Args:
            feedback: User feedback describing what should be revised.

        Returns:
            MilestoneApproval with REVISE action and attached feedback.
        """
        return cls(action=MilestoneAction.REVISE, feedback=feedback)

    @classmethod
    def skip_remaining(cls) -> MilestoneApproval:
        """Create a response that skips all remaining milestone checkpoints.

        Returns:
            MilestoneApproval with SKIP_REMAINING action.
        """
        return cls(action=MilestoneAction.SKIP_REMAINING)

    @classmethod
    def abort(cls) -> MilestoneApproval:
        """Create an abort response that stops plan execution entirely.

        Returns:
            MilestoneApproval with ABORT action.
        """
        return cls(action=MilestoneAction.ABORT)


@dataclass
class MilestonePolicy:
    """Configuration for milestone checkpoints.

    Controls when and how execution pauses for user approval.
    """

    granularity: MilestoneGranularity = MilestoneGranularity.FEATURES
    auto_approve_on_success: bool = True
    auto_approve_quality_threshold: float = 0.8
    timeout_seconds: float = 300.0  # 5 minutes, then auto-approve
    skip_all: bool = False  # Set by SKIP_REMAINING action

    def __repr__(self) -> str:
        return (
            f"MilestonePolicy(granularity={self.granularity.value!r}, "
            f"auto_approve_on_success={self.auto_approve_on_success!r}, "
            f"quality_threshold={self.auto_approve_quality_threshold!r})"
        )

    def should_checkpoint(self, task: Any) -> bool:
        """Determine if a task warrants a milestone checkpoint.

        Args:
            task: Task object with is_milestone, requires_approval fields.

        Returns:
            True if execution should pause after this task.
        """
        if self.skip_all or self.granularity == MilestoneGranularity.NONE:
            return False

        is_milestone = getattr(task, "is_milestone", False)
        requires_approval = getattr(task, "requires_approval", False)
        depth = getattr(task, "depth", 0)

        if self.granularity == MilestoneGranularity.ALL:
            return True
        if self.granularity == MilestoneGranularity.FEATURES:
            return is_milestone or requires_approval
        if self.granularity == MilestoneGranularity.PHASES:
            return depth == 0 and is_milestone
        if self.granularity == MilestoneGranularity.CRITICAL:
            return requires_approval
        return False

    def should_auto_approve(self, milestone: MilestoneReached) -> bool:
        """Check if a milestone can be auto-approved.

        Auto-approves when: task succeeded + quality above threshold + no issues.

        Returns:
            True if successful, False otherwise.
        """
        if not self.auto_approve_on_success:
            return False
        return milestone.quality_score >= self.auto_approve_quality_threshold

    def to_dict(self) -> dict[str, Any]:
        """Converts the milestone policy to a plain dictionary for serialization."""
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MilestonePolicy:
        """Deserialize a MilestonePolicy from a dictionary.

        Args:
            data: Dictionary with policy fields as produced by ``to_dict()``.

        Returns:
            A new MilestonePolicy instance.
        """
        return cls(
            granularity=MilestoneGranularity(data.get("granularity", "features")),
            auto_approve_on_success=data.get("auto_approve_on_success", True),
            auto_approve_quality_threshold=data.get("auto_approve_quality_threshold", 0.8),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            skip_all=data.get("skip_all", False),
        )


class MilestoneManager:
    """Manages milestone checkpoints during plan execution.

    Coordinates between the execution engine and the user interface
    (CLI or web) to pause, collect approval, and resume execution.
    """

    def __init__(self, policy: MilestonePolicy | None = None):
        self.policy = policy or MilestonePolicy()
        self._pending: MilestoneReached | None = None
        self._approval_event = threading.Event()
        self._approval: MilestoneApproval | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=1000)
        self._approval_callback: Callable | None = None

    def set_approval_callback(self, callback: Callable[[MilestoneReached], MilestoneApproval]) -> None:
        """Set a callback for milestone approval (used by CLI/web UI).

        The callback receives a MilestoneReached event and should return
        a MilestoneApproval. If no callback is set, milestones auto-approve
        after the timeout.
        """
        self._approval_callback = callback

    def check_and_wait(
        self,
        task: Any,
        result: Any,
        completed_tasks: list[str],
    ) -> MilestoneApproval:
        """Check if task is a milestone and wait for approval if needed.

        Args:
            task: The completed task.
            result: The AgentResult from execution.
            completed_tasks: List of task IDs completed so far.

        Returns:
            MilestoneApproval (APPROVE if not a milestone or auto-approved).
        """
        if not self.policy.should_checkpoint(task):
            return MilestoneApproval.approve()

        # Build milestone event
        quality = 0.0
        if hasattr(result, "metadata"):
            quality = result.metadata.get("quality_score", 0.0)

        milestone = MilestoneReached(
            milestone_name=getattr(task, "milestone_name", "") or f"Task {getattr(task, 'id', '?')}",
            task_id=getattr(task, "id", ""),
            task_description=getattr(task, "description", ""),
            completed_tasks=completed_tasks,
            quality_score=quality,
        )

        # Check auto-approve
        if self.policy.should_auto_approve(milestone):
            logger.info("[Milestone] Auto-approved: %s", milestone.milestone_name)
            self._record(milestone, MilestoneApproval.approve(), auto=True)
            return MilestoneApproval.approve()

        # Use callback if available
        if self._approval_callback:
            try:
                approval = self._approval_callback(milestone)
                self._record(milestone, approval, auto=False)
                self._handle_action(approval)
                return approval
            except Exception as e:
                logger.warning("[Milestone] Callback failed: %s, auto-approving", e)
                return MilestoneApproval.approve()

        # No callback — wait for a web UI submission via submit_approval().
        # Clear any stale event/approval from a previous milestone, record the
        # pending milestone so the status API can expose it, then block until
        # either the event fires (web UI called submit_approval) or the timeout
        # elapses (fall back to auto-approve so execution is never stuck forever).
        self._approval_event.clear()
        self._approval = None
        self._pending = milestone
        logger.info(
            "[Milestone] Waiting up to %.0fs for web UI approval of '%s'",
            self.policy.timeout_seconds,
            milestone.milestone_name,
        )
        received = self._approval_event.wait(timeout=self.policy.timeout_seconds)
        self._pending = None

        if received and self._approval is not None:
            approval = self._approval
            self._approval = None
            self._record(milestone, approval, auto=False)
            self._handle_action(approval)
            return approval

        # Timed out with no response — auto-approve and warn
        logger.warning(
            "[Milestone] No approval received within %.0fs for '%s' — auto-approving",
            self.policy.timeout_seconds,
            milestone.milestone_name,
        )
        self._record(milestone, MilestoneApproval.approve(), auto=True)
        return MilestoneApproval.approve()

    def submit_approval(self, approval: MilestoneApproval) -> None:
        """Submit an approval for the pending milestone (used by web UI async flow)."""
        self._approval = approval
        self._approval_event.set()

    def _handle_action(self, approval: MilestoneApproval) -> None:
        """Handle special actions like skip_remaining."""
        if approval.action == MilestoneAction.SKIP_REMAINING:
            self.policy.skip_all = True
            logger.info("[Milestone] Skip remaining enabled — no more checkpoints")

    def _record(self, milestone: MilestoneReached, approval: MilestoneApproval, auto: bool) -> None:
        """Record a milestone checkpoint and its approval in the history log.

        Args:
            milestone: The milestone event that was reached.
            approval: The approval response (user or auto-generated).
            auto: Whether the approval was automatic (True) or user-provided (False).
        """
        self._history.append({
            "milestone": milestone.to_dict(),
            "action": approval.action.value,
            "feedback": approval.feedback,
            "auto_approved": auto,
            "timestamp": time.time(),
        })

    def get_history(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded milestone checkpoint events.

        Returns:
            List of dictionaries, each containing milestone, action, and timestamp.
        """
        return list(self._history)
