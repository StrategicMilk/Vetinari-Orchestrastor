"""
Human-in-the-Loop with State Checkpointing
============================================

Provides full execution state serialization to disk, configurable risk
thresholds for auto-pause, audit trail of human approvals/rejections,
and seamless resume from checkpoint without replaying prior work.

Usage:
    from vetinari.orchestration.checkpointing import CheckpointManager

    mgr = CheckpointManager()
    cp_id = mgr.save_checkpoint(execution_state)
    state = mgr.load_checkpoint(cp_id)
    checkpoints = mgr.list_checkpoints()
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default checkpoint storage directory
DEFAULT_CHECKPOINT_DIR = ".vetinari/checkpoints"


@dataclass
class RiskThresholds:
    """Configurable risk thresholds that trigger auto-pause for human review."""
    token_cost_usd: float = 5.0          # Pause if estimated cost exceeds this
    failure_rate: float = 0.3             # Pause if failure rate exceeds 30%
    destructive_action: bool = True       # Always pause on destructive actions
    unknown_tool_call: bool = True        # Pause on unrecognized tool invocations
    quality_score_min: float = 0.4        # Pause if quality drops below this

    def should_pause(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Evaluate metrics against thresholds. Returns reason string or None."""
        cost = metrics.get("estimated_cost_usd", 0.0)
        if cost > self.token_cost_usd:
            return f"Estimated cost ${cost:.2f} exceeds threshold ${self.token_cost_usd:.2f}"

        fail_rate = metrics.get("failure_rate", 0.0)
        if fail_rate > self.failure_rate:
            return f"Failure rate {fail_rate:.1%} exceeds threshold {self.failure_rate:.1%}"

        if self.destructive_action and metrics.get("has_destructive_action", False):
            return "Destructive action detected, requires human approval"

        if self.unknown_tool_call and metrics.get("has_unknown_tool", False):
            return "Unknown tool call detected, requires human approval"

        quality = metrics.get("quality_score", 1.0)
        if quality < self.quality_score_min:
            return f"Quality score {quality:.2f} below minimum {self.quality_score_min:.2f}"

        return None


@dataclass
class ApprovalRecord:
    """Audit trail entry for a human approval or rejection decision."""
    record_id: str = field(default_factory=lambda: f"apr_{uuid.uuid4().hex[:8]}")
    checkpoint_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decision: str = ""          # "approved", "rejected", "modified"
    approver: str = ""          # Human identifier
    reason: str = ""            # Why they approved/rejected
    pause_trigger: str = ""     # What triggered the pause
    modifications: Dict[str, Any] = field(default_factory=dict)  # Any changes made

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExecutionCheckpoint:
    """Serializable snapshot of full execution state."""
    checkpoint_id: str = field(default_factory=lambda: f"cp_{uuid.uuid4().hex[:12]}")
    plan_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""

    # Task graph state
    completed_task_ids: List[str] = field(default_factory=list)
    running_task_ids: List[str] = field(default_factory=list)
    pending_task_ids: List[str] = field(default_factory=list)
    failed_task_ids: List[str] = field(default_factory=list)

    # Task outputs keyed by task_id
    task_outputs: Dict[str, Any] = field(default_factory=dict)

    # Execution metrics at checkpoint time
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Whether execution was paused for human review
    paused_for_review: bool = False
    pause_reason: str = ""

    # Approval trail for this checkpoint
    approvals: List[Dict[str, Any]] = field(default_factory=list)

    # Arbitrary metadata for resume context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionCheckpoint":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """
    Manages execution state checkpointing for human-in-the-loop workflows.

    Features:
    - Serialize full execution state to disk as JSON
    - Configurable risk thresholds that trigger auto-pause
    - Audit trail of all human approval/rejection decisions
    - Resume from any checkpoint without replaying completed work
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        risk_thresholds: Optional[RiskThresholds] = None,
    ):
        self._dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._thresholds = risk_thresholds or RiskThresholds()
        self._lock = threading.Lock()
        self._approval_log: List[ApprovalRecord] = []
        self._load_approval_log()
        logger.info("CheckpointManager initialized (dir=%s)", self._dir)

    # ------------------------------------------------------------------
    # Core checkpoint operations
    # ------------------------------------------------------------------

    def save_checkpoint(self, execution_state: Dict[str, Any]) -> str:
        """
        Serialize the current execution state to disk.

        Args:
            execution_state: Dictionary containing at minimum:
                - plan_id: str
                - completed_tasks: List[str]
                - running_tasks: List[str]
                - pending_tasks: List[str]
                - task_outputs: Dict[str, Any]
                - metrics: Dict[str, Any] (optional)
                - metadata: Dict[str, Any] (optional)

        Returns:
            checkpoint_id: Unique identifier for the saved checkpoint.
        """
        with self._lock:
            checkpoint = ExecutionCheckpoint(
                plan_id=execution_state.get("plan_id", ""),
                description=execution_state.get("description", ""),
                completed_task_ids=execution_state.get("completed_tasks", []),
                running_task_ids=execution_state.get("running_tasks", []),
                pending_task_ids=execution_state.get("pending_tasks", []),
                failed_task_ids=execution_state.get("failed_tasks", []),
                task_outputs=execution_state.get("task_outputs", {}),
                metrics=execution_state.get("metrics", {}),
                metadata=execution_state.get("metadata", {}),
            )

            # Check risk thresholds to decide if we need to pause
            pause_reason = self._thresholds.should_pause(checkpoint.metrics)
            if pause_reason:
                checkpoint.paused_for_review = True
                checkpoint.pause_reason = pause_reason
                logger.warning(
                    "Checkpoint %s paused for review: %s",
                    checkpoint.checkpoint_id, pause_reason,
                )

            # Persist to disk
            cp_path = self._checkpoint_path(checkpoint.checkpoint_id)
            try:
                with open(cp_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint.to_dict(), f, indent=2, default=str)
                logger.info(
                    "Checkpoint saved: %s (tasks completed=%d, pending=%d)",
                    checkpoint.checkpoint_id,
                    len(checkpoint.completed_task_ids),
                    len(checkpoint.pending_task_ids),
                )
            except Exception as exc:
                logger.error("Failed to save checkpoint %s: %s", checkpoint.checkpoint_id, exc)
                raise

            return checkpoint.checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> ExecutionCheckpoint:
        """
        Load a previously saved checkpoint from disk.

        Args:
            checkpoint_id: The checkpoint identifier returned by save_checkpoint.

        Returns:
            ExecutionCheckpoint with the full saved state.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            json.JSONDecodeError: If the checkpoint file is corrupted.
        """
        cp_path = self._checkpoint_path(checkpoint_id)
        if not cp_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            checkpoint = ExecutionCheckpoint.from_dict(data)
            logger.info("Checkpoint loaded: %s", checkpoint_id)
            return checkpoint
        except json.JSONDecodeError as exc:
            logger.error("Corrupted checkpoint file %s: %s", checkpoint_id, exc)
            raise

    def list_checkpoints(self, plan_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available checkpoints, optionally filtered by plan_id.

        Returns:
            List of checkpoint summaries with id, plan_id, created_at,
            task counts, and pause status.
        """
        results = []
        try:
            for cp_file in sorted(self._dir.glob("cp_*.json")):
                try:
                    with open(cp_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if plan_id and data.get("plan_id") != plan_id:
                        continue

                    results.append({
                        "checkpoint_id": data.get("checkpoint_id", cp_file.stem),
                        "plan_id": data.get("plan_id", ""),
                        "created_at": data.get("created_at", ""),
                        "description": data.get("description", ""),
                        "completed_tasks": len(data.get("completed_task_ids", [])),
                        "pending_tasks": len(data.get("pending_task_ids", [])),
                        "failed_tasks": len(data.get("failed_task_ids", [])),
                        "paused_for_review": data.get("paused_for_review", False),
                        "pause_reason": data.get("pause_reason", ""),
                    })
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping corrupt checkpoint %s: %s", cp_file.name, exc)
        except OSError as exc:
            logger.error("Error listing checkpoints: %s", exc)

        return results

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint from disk."""
        cp_path = self._checkpoint_path(checkpoint_id)
        try:
            if cp_path.exists():
                cp_path.unlink()
                logger.info("Checkpoint deleted: %s", checkpoint_id)
                return True
            return False
        except OSError as exc:
            logger.error("Failed to delete checkpoint %s: %s", checkpoint_id, exc)
            return False

    # ------------------------------------------------------------------
    # Human-in-the-loop approval workflow
    # ------------------------------------------------------------------

    def record_approval(
        self,
        checkpoint_id: str,
        decision: str,
        approver: str,
        reason: str = "",
        modifications: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRecord:
        """
        Record a human approval or rejection for a paused checkpoint.

        Args:
            checkpoint_id: The checkpoint being reviewed.
            decision: One of "approved", "rejected", "modified".
            approver: Human identifier (e.g. username or email).
            reason: Explanation for the decision.
            modifications: Any modifications the human made to the plan.

        Returns:
            The ApprovalRecord persisted to the audit trail.
        """
        with self._lock:
            # Load checkpoint to attach approval
            try:
                checkpoint = self.load_checkpoint(checkpoint_id)
            except FileNotFoundError:
                logger.error("Cannot record approval: checkpoint %s not found", checkpoint_id)
                raise

            record = ApprovalRecord(
                checkpoint_id=checkpoint_id,
                decision=decision,
                approver=approver,
                reason=reason,
                pause_trigger=checkpoint.pause_reason,
                modifications=modifications or {},
            )

            # Add to checkpoint
            checkpoint.approvals.append(record.to_dict())
            if decision == "approved":
                checkpoint.paused_for_review = False
                checkpoint.pause_reason = ""

            # Re-save checkpoint with approval
            cp_path = self._checkpoint_path(checkpoint_id)
            with open(cp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)

            # Add to global audit log
            self._approval_log.append(record)
            self._save_approval_log()

            logger.info(
                "Approval recorded: checkpoint=%s decision=%s approver=%s",
                checkpoint_id, decision, approver,
            )
            return record

    def get_approval_history(
        self,
        checkpoint_id: Optional[str] = None,
        approver: Optional[str] = None,
    ) -> List[ApprovalRecord]:
        """
        Retrieve the approval audit trail, optionally filtered.

        Args:
            checkpoint_id: Filter by checkpoint.
            approver: Filter by approver identity.

        Returns:
            List of ApprovalRecord entries matching the filters.
        """
        results = self._approval_log
        if checkpoint_id:
            results = [r for r in results if r.checkpoint_id == checkpoint_id]
        if approver:
            results = [r for r in results if r.approver == approver]
        return results

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def resume_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Prepare an execution state for resumption from a checkpoint.

        Loads the checkpoint and returns a state dict containing only
        the remaining (non-completed) work, so no prior tasks are replayed.

        Args:
            checkpoint_id: The checkpoint to resume from.

        Returns:
            Dictionary with keys: plan_id, completed_tasks, pending_tasks,
            task_outputs (from completed work), metadata, resume_from.
        """
        checkpoint = self.load_checkpoint(checkpoint_id)

        if checkpoint.paused_for_review:
            logger.warning(
                "Resuming from checkpoint %s that is paused for review: %s",
                checkpoint_id, checkpoint.pause_reason,
            )

        resume_state = {
            "plan_id": checkpoint.plan_id,
            "completed_tasks": checkpoint.completed_task_ids,
            "pending_tasks": checkpoint.pending_task_ids,
            "failed_tasks": checkpoint.failed_task_ids,
            "task_outputs": checkpoint.task_outputs,
            "metrics": checkpoint.metrics,
            "metadata": checkpoint.metadata,
            "resume_from": checkpoint_id,
            "resumed_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Resume state prepared from checkpoint %s: %d completed, %d pending",
            checkpoint_id,
            len(checkpoint.completed_task_ids),
            len(checkpoint.pending_task_ids),
        )
        return resume_state

    # ------------------------------------------------------------------
    # Risk threshold management
    # ------------------------------------------------------------------

    def update_thresholds(self, **kwargs) -> RiskThresholds:
        """
        Update risk thresholds dynamically.

        Accepts keyword arguments matching RiskThresholds fields.

        Returns:
            Updated RiskThresholds instance.
        """
        for key, value in kwargs.items():
            if hasattr(self._thresholds, key):
                setattr(self._thresholds, key, value)
                logger.info("Risk threshold updated: %s = %s", key, value)
            else:
                logger.warning("Unknown threshold field: %s", key)
        return self._thresholds

    def evaluate_risk(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Evaluate current execution metrics against risk thresholds.

        Returns:
            Reason string if thresholds are exceeded, None otherwise.
        """
        return self._thresholds.should_pause(metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        """Return the file path for a checkpoint."""
        safe_id = checkpoint_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe_id}.json"

    def _approval_log_path(self) -> Path:
        """Return the path to the global approval audit log."""
        return self._dir / "_approval_audit_log.json"

    def _load_approval_log(self) -> None:
        """Load the persisted approval audit log from disk."""
        log_path = self._approval_log_path()
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                self._approval_log = [ApprovalRecord.from_dict(e) for e in entries]
                logger.debug("Loaded %d approval records from audit log", len(self._approval_log))
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("Could not load approval log: %s", exc)
                self._approval_log = []

    def _save_approval_log(self) -> None:
        """Persist the approval audit log to disk."""
        log_path = self._approval_log_path()
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._approval_log], f, indent=2)
        except Exception as exc:
            logger.error("Failed to save approval log: %s", exc)
