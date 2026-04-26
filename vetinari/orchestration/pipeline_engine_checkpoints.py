"""Best-effort checkpoint persistence helpers for pipeline engine stages.

Pipeline checkpoints let replay and cost-analysis tools inspect completed
stage boundaries without making checkpoint storage part of the critical path.
"""

from __future__ import annotations

import contextlib
from typing import Any


def _save_pipeline_checkpoint(
    *,
    trace_id: str,
    execution_id: str,
    step_name: str,
    step_index: int,
    status: str,
    output_snapshot: dict[str, Any],
) -> None:
    """Persist a stage checkpoint without interrupting pipeline execution.

    Args:
        trace_id: Trace identifier for the pipeline run.
        execution_id: Queue execution identifier for the run.
        step_name: Stable pipeline step name.
        step_index: Ordered step index used by replay tooling.
        status: Step completion status to record.
        output_snapshot: Small serializable snapshot of stage output.
    """
    with contextlib.suppress(Exception):
        from vetinari.observability.checkpoints import PipelineCheckpoint, get_checkpoint_store

        get_checkpoint_store().save_checkpoint(
            PipelineCheckpoint(
                trace_id=trace_id,
                execution_id=execution_id,
                step_name=step_name,
                step_index=step_index,
                status=status,
                output_snapshot=output_snapshot,
            )
        )
