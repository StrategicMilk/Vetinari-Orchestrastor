"""Helpers that build and append WorkReceipts at well-known emission points.

Three emission flavours:

- ``record_agent_completion`` is called from ``base_agent_completion.complete_task``
  after every Foreman / Worker / Inspector task. The agent_type maps to the
  receipt kind (FOREMAN -> PLAN_ROUND, WORKER -> WORKER_TASK,
  INSPECTOR -> INSPECTOR_PASS).
- ``record_training_step`` is called from ``vetinari.training.pipeline.TrainingPipeline.run``
  after a training run completes. Fallback runs (``_is_fallback=True``) are
  skipped to avoid inflating training counts (anti-pattern: Fallback as success).
- ``record_release_step`` is called by the release doctor after each step
  records a ``ReleaseClaimRecord`` so the Control Center reflects release
  progress without polling the proof file.

All three return the appended ``WorkReceipt`` for caller visibility.  None
of them raise on bus or store failure: a failing receipt emission must
never crash agent execution; the fallback is a structured WARNING log.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vetinari.agents.contracts import OutcomeSignal, Provenance
from vetinari.receipts.record import WorkReceipt, WorkReceiptKind
from vetinari.receipts.store import WorkReceiptStore
from vetinari.types import AgentType, EvidenceBasis

if TYPE_CHECKING:
    from vetinari.agents.base_agent import BaseAgent
    from vetinari.agents.contracts import AgentResult, AgentTask

logger = logging.getLogger(__name__)

# AgentType -> WorkReceiptKind mapping.  The factory pipeline produces
# exactly one receipt kind per agent role.
_AGENT_KIND_MAP: dict[AgentType, WorkReceiptKind] = {
    AgentType.FOREMAN: WorkReceiptKind.PLAN_ROUND,
    AgentType.WORKER: WorkReceiptKind.WORKER_TASK,
    AgentType.INSPECTOR: WorkReceiptKind.INSPECTOR_PASS,
}

_DEFAULT_PROJECT_ID = "default"


def _record_emission_failure(*, kind: str) -> None:
    """Increment a counter when a receipt emission fails silently.

    Receipt emission helpers swallow exceptions so they never crash
    agent execution, but a silent failure leaves the Control Center
    blind. The counter (vetinari.receipts.emission_failures) makes
    those silent failures observable in metrics dashboards.

    The metrics subsystem must never crash receipt emission either,
    and a WARNING here would spam logs whenever the metrics backend
    hiccups; ``contextlib.suppress`` is the right idiom — the counter
    increment is fire-and-forget. Operators who need to debug the
    metrics path will see a WARNING from inside ``get_metrics()``
    itself, not from this helper.
    """
    import contextlib

    with contextlib.suppress(Exception):
        from vetinari.metrics import get_metrics

        get_metrics().increment("vetinari.receipts.emission_failures", kind=kind)


def _summary(text: str | None, limit: int = 200) -> str:
    """Trim *text* to *limit* characters with an ellipsis if truncated.

    Accepts ``None`` defensively because callers occasionally pass an
    optional task field; coerces to an empty string in that case.
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _coerce_project_id(task: AgentTask | None, fallback: str = _DEFAULT_PROJECT_ID) -> str:
    """Pull the project_id from an AgentTask context, with safe fallback.

    Args:
        task: The AgentTask whose context may carry a ``project_id``.
        fallback: Value to use when no project_id is present.

    Returns:
        A non-empty project_id string.
    """
    if task is None:
        return fallback
    ctx = getattr(task, "context", None) or {}
    pid = ctx.get("project_id") if isinstance(ctx, dict) else None
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    return fallback


def _build_outcome_from_score(
    *,
    success: bool,
    score: float,
    source: str,
    scoring_available: bool = True,
    issues: Iterable[str] = (),
    suggestions: Iterable[str] = (),
) -> OutcomeSignal:
    """Build an OutcomeSignal that reflects a quality-scored agent outcome.

    The basis is ``LLM_JUDGMENT`` when scoring ran and the work
    succeeded; ``UNSUPPORTED`` is used for execution failures and for
    the case where scoring itself crashed (``scoring_available=False``)
    so consumers fail-closed on ambiguous outputs and never mistake a
    score-of-zero-because-scorer-died for a score-of-zero-because-it-was-bad.
    """
    if not success:
        return OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=tuple(issues) or ("agent execution did not succeed",),
            suggestions=tuple(suggestions),
            provenance=Provenance(
                source=source,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
    if not scoring_available:
        # Work succeeded but the quality scorer could not produce a
        # signal. Mark UNSUPPORTED rather than LLM_JUDGMENT so the
        # downstream consumer cannot treat score=0.0 as a real verdict.
        return OutcomeSignal(
            passed=True,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=tuple(issues) or ("quality scoring unavailable for this task",),
            suggestions=tuple(suggestions),
            provenance=Provenance(
                source=source,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
    return OutcomeSignal(
        passed=True,
        score=float(score),
        basis=EvidenceBasis.LLM_JUDGMENT,
        issues=tuple(issues),
        suggestions=tuple(suggestions),
        provenance=Provenance(
            source=source,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )


def record_agent_completion(
    *,
    agent: BaseAgent,
    task: AgentTask,
    result: AgentResult,
    score: float,
    scoring_available: bool = True,
    store: WorkReceiptStore | None = None,
) -> WorkReceipt | None:
    """Emit a WorkReceipt for a completed agent task.

    The receipt kind is derived from ``agent.agent_type``. Failures of the
    receipt subsystem (store I/O, bus publish) are logged as WARNING and
    return ``None`` so the calling pipeline (``complete_task``) is never
    crashed by an observability fault. Silent failures are also counted
    via ``vetinari.receipts.emission_failures`` so an operator can
    observe them in metrics.

    Args:
        agent: The BaseAgent that finished the task.
        task: The completed AgentTask.
        result: The AgentResult returned by execution.
        score: Quality score from the scorer (already computed in
            ``complete_task``); ``0.0`` for failures.
        scoring_available: True when the quality scorer produced
            ``score``; False when an early-return path bypassed the
            scorer (e.g. quality scoring crashed). Drives the
            ``OutcomeSignal.basis`` so consumers can distinguish
            "scored zero" from "scoring unavailable".
        store: Optional WorkReceiptStore override; the default uses the
            current repo root.

    Returns:
        The appended WorkReceipt, or ``None`` if the emission failed.
    """
    try:
        kind = _AGENT_KIND_MAP.get(agent.agent_type)
        if kind is None:
            logger.warning(
                "No receipt kind mapping for agent_type=%s — skipping receipt emission",
                agent.agent_type,
            )
            _record_emission_failure(kind="unknown_agent_type")
            return None

        # awaiting_user / awaiting_reason are set by Foreman/Inspector when
        # they explicitly mark a project blocked on the user. Worker NEVER
        # sets these directly; the WorkReceipt invariant is enforced in
        # WorkReceipt.__post_init__.
        meta: dict[str, Any] = result.metadata or {}
        awaiting_user = bool(meta.get("awaiting_user"))
        awaiting_reason: str | None = meta.get("awaiting_reason")
        if awaiting_user and agent.agent_type is AgentType.WORKER:
            # Worker MUST NOT set awaiting flags (shard 02, task 2.2). Drop
            # the flag rather than failing the receipt.
            logger.warning(
                "Worker attempted to set awaiting_user=True; suppressed (only Foreman/Inspector may block on user)",
            )
            awaiting_user = False
            awaiting_reason = None

        linked_claim_ids: tuple[str, ...] = tuple(meta.get("linked_claim_ids", ()))

        outcome = _build_outcome_from_score(
            success=result.success,
            score=score,
            scoring_available=scoring_available,
            source=f"vetinari.agents.{agent.agent_type.value.lower()}",
            issues=tuple(result.errors or ()),
        )

        output_str = result.output if isinstance(result.output, str) else str(result.output)

        receipt = WorkReceipt(
            project_id=_coerce_project_id(task),
            agent_id=getattr(agent, "name", agent.agent_type.value),
            agent_type=agent.agent_type,
            kind=kind,
            outcome=outcome,
            inputs_summary=_summary(task.description or task.prompt or ""),
            outputs_summary=_summary(output_str),
            awaiting_user=awaiting_user,
            awaiting_reason=awaiting_reason,
            linked_claim_ids=linked_claim_ids,
        )

        (store or WorkReceiptStore()).append(receipt)
        return receipt
    except Exception:
        logger.warning(
            "Failed to emit WorkReceipt for agent=%s task=%s — continuing",
            getattr(agent, "agent_type", "<unknown>"),
            getattr(task, "task_id", "<unknown>"),
            exc_info=True,
        )
        _record_emission_failure(kind="agent_completion")
        return None


def record_training_step(
    *,
    project_id: str,
    run_id: str,
    base_model: str,
    algorithm: str,
    epochs: int,
    training_examples: int,
    success: bool,
    eval_score: float = 0.0,
    error: str = "",
    is_fallback: bool = False,
    store: WorkReceiptStore | None = None,
) -> WorkReceipt | None:
    """Emit a TRAINING_STEP receipt for one training-run completion.

    Fallback runs (``is_fallback=True``) are NEVER recorded as completed
    training receipts (anti-pattern: Fallback as success).

    Args:
        project_id: Project this training run belongs to.
        run_id: Unique identifier of the training run.
        base_model: Model the run started from.
        algorithm: Which algorithm actually ran (``"sft"``, ``"dpo"``,
            ``"simpo"``, etc.). Required for post-SESSION-02 naming truth.
        epochs: Total epochs requested for the run.
        training_examples: Number of training examples in the dataset.
        success: Whether the training run completed successfully.
        eval_score: Final evaluation score; ``0.0`` when not available.
        error: Error message if the run failed; ``""`` otherwise.
        is_fallback: If True, the run was a fallback / skipped path and
            no receipt is emitted.
        store: Optional WorkReceiptStore override.

    Returns:
        The appended WorkReceipt, or ``None`` if the run was a fallback
        or emission failed.
    """
    if is_fallback:
        logger.info(
            "Skipping TRAINING_STEP receipt for fallback run %s (algorithm=%s) — "
            "fallbacks are not recorded as completed training",
            run_id,
            algorithm,
        )
        return None

    try:
        outcome = OutcomeSignal(
            passed=success,
            score=float(eval_score) if success else 0.0,
            basis=EvidenceBasis.TOOL_EVIDENCE if success else EvidenceBasis.UNSUPPORTED,
            issues=(error,) if error else (),
            provenance=Provenance(
                source="vetinari.training.pipeline",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                tool_name="qlora_trainer",
            ),
        )

        inputs = f"base_model={base_model} | algo={algorithm} | epochs={epochs} | examples={training_examples}"
        outputs = f"run_id={run_id} | success={success}" + (f" | eval_score={eval_score:.3f}" if success else "")

        receipt = WorkReceipt(
            project_id=project_id,
            agent_id=f"training-runner:{run_id}",
            # Training pipeline is an auxiliary runner, not a factory-pipeline
            # agent — labeled with AgentType.TRAINING per ADR-0103 so the
            # receipt's actor field is honest.
            agent_type=AgentType.TRAINING,
            kind=WorkReceiptKind.TRAINING_STEP,
            outcome=outcome,
            inputs_summary=_summary(inputs),
            outputs_summary=_summary(outputs),
        )

        (store or WorkReceiptStore()).append(receipt)
        return receipt
    except Exception:
        logger.warning(
            "Failed to emit TRAINING_STEP receipt for run %s — continuing",
            run_id,
            exc_info=True,
        )
        _record_emission_failure(kind="training_step")
        return None


def record_release_step(
    *,
    project_id: str,
    version: str,
    step_name: str,
    success: bool,
    proof_path: Path | str | None = None,
    linked_claim_ids: Iterable[str] = (),
    error: str = "",
    store: WorkReceiptStore | None = None,
) -> WorkReceipt | None:
    """Emit a RELEASE_STEP receipt for one release-doctor stage.

    The release doctor calls this after each pipeline step (build, install,
    doctor, smoke, sign) so the Control Center can show release progress
    without polling the proof file.

    Args:
        project_id: Project owning this release.
        version: Release version string (e.g. ``"0.9.0"``).
        step_name: Name of the release step (e.g. ``"build"``,
            ``"smoke"``, ``"sign"``).
        success: Whether the step succeeded.
        proof_path: Optional path to the proof artifact for the
            outputs_summary.
        linked_claim_ids: Identifiers of ClaimsLedger records emitted
            during this step.
        error: Error message if the step failed; ``""`` otherwise.
        store: Optional WorkReceiptStore override.

    Returns:
        The appended WorkReceipt, or ``None`` if emission failed.
    """
    try:
        outcome = OutcomeSignal(
            passed=success,
            score=1.0 if success else 0.0,
            basis=EvidenceBasis.TOOL_EVIDENCE if success else EvidenceBasis.UNSUPPORTED,
            issues=(error,) if error else (),
            provenance=Provenance(
                source="scripts.release_doctor",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                tool_name="release_doctor",
            ),
        )

        outputs = f"version={version} | step={step_name} | success={success}"
        if proof_path is not None:
            outputs += f" | proof={proof_path}"

        receipt = WorkReceipt(
            project_id=project_id,
            agent_id=f"release-doctor:{version}",
            # Release doctor is an auxiliary runner, not a factory-pipeline
            # agent — labeled with AgentType.RELEASE per ADR-0103.
            agent_type=AgentType.RELEASE,
            kind=WorkReceiptKind.RELEASE_STEP,
            outcome=outcome,
            inputs_summary=_summary(f"release pipeline step: {step_name}"),
            outputs_summary=_summary(outputs),
            linked_claim_ids=tuple(linked_claim_ids),
        )

        (store or WorkReceiptStore()).append(receipt)
        return receipt
    except Exception:
        logger.warning(
            "Failed to emit RELEASE_STEP receipt for version=%s step=%s — continuing",
            version,
            step_name,
            exc_info=True,
        )
        _record_emission_failure(kind="release_step")
        return None


__all__ = [
    "record_agent_completion",
    "record_release_step",
    "record_training_step",
]
