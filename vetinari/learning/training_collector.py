"""TrainingDataCollector — thread-safe training data recorder with background I/O.

Records every agent execution to a JSONL file in a fire-and-forget manner via
a background writer thread, imposing zero latency on the inference hot path.

Export methods (SFT, DPO, prompt-variant, HuggingFace, few-shot, ranking) and
trace storage methods live in ``training_exports._TrainingExportsMixin`` to keep
this file under the 550-line ceiling.

The module-level ``get_training_collector()`` returns the process-wide singleton
protected by a double-checked lock.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import (
    _PROJECT_ROOT,
    QUEUE_TIMEOUT,
    THREAD_JOIN_TIMEOUT,
    TRAINING_COLLECTOR_QUEUE_SIZE,
    TRUNCATE_OUTPUT_PREVIEW,
    TRUNCATE_PROMPT_TRAINING,
    TRUNCATE_RESPONSE_TRAINING,
)
from vetinari.types import EvidenceBasis

from .training_exports import _TrainingExportsMixin
from .training_record import TrainingRecord

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.environ.get(
    "VETINARI_TRAINING_DATA_PATH",
    str(_PROJECT_ROOT / "training_data.jsonl"),
)

# Known fallback response patterns — records matching these are rejected to
# prevent training data poisoning from mock/fallback outputs.
# Primary check is _is_fallback metadata field; this is the secondary blocklist.
_FALLBACK_RESPONSE_PATTERNS: frozenset[str] = frozenset({
    "",
    "{}",
    '{"content":"","sections":[]}',
    '{"content": "", "sections": []}',
})


class TrainingDataCollector(_TrainingExportsMixin):
    """Thread-safe training data recorder with background I/O.

    Records agent executions to a JSONL file via a background writer thread,
    keeping the inference hot-path non-blocking.  Enforces data quality gates
    at record time (rejects fallback/mock outputs, zero-latency records, and
    secrets-containing prompts).

    Use ``get_training_collector()`` (module-level) rather than instantiating
    directly.

    Side effects:
      - On construction (non-sync mode): starts a background daemon thread
        named ``TrainingDataWriter`` that writes queued records to disk.
    """

    _instance: TrainingDataCollector | None = None
    _cls_lock = threading.Lock()

    def __init__(self, output_path: str = _DEFAULT_PATH, sync: bool = False) -> None:
        self._output_path = Path(output_path)
        self._queue: queue.Queue = queue.Queue(maxsize=TRAINING_COLLECTOR_QUEUE_SIZE)
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._sync = sync  # When True, writes happen inline (no background thread)
        self._record_count = 0

        if not sync:
            self._worker = threading.Thread(
                target=self._write_worker,
                name="TrainingDataWriter",
                daemon=True,
            )
            self._worker.start()
        else:
            self._worker = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, output_path: str = _DEFAULT_PATH) -> TrainingDataCollector:
        """Return the class-level singleton, creating it if needed.

        Starts the background writer thread on first creation. Subsequent
        calls with a different output_path are ignored — the first caller
        sets the path.

        Args:
            output_path: Path to the JSONL file where records are written.

        Returns:
            The shared TrainingDataCollector instance for this process.
        """
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls(output_path=output_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        task: str,
        prompt: str,
        response: str,
        score: float,
        model_id: str,
        task_type: str = "general",
        prompt_variant_id: str = "",
        agent_type: str = "",
        latency_ms: int = 0,
        tokens_used: int = 0,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
        benchmark_suite: str = "",
        benchmark_pass: bool = False,
        benchmark_score: float = 0.0,
        rejection_reason: str = "",
        rejection_category: str = "",
        inspector_feedback: str = "",
        trace_id: str = "",
        evidence_basis: EvidenceBasis | None = None,
        target_stream: str | None = None,
    ) -> None:
        """Record a task execution (non-blocking; queued for background write).

        Rejects records that indicate fallback/mock outputs to prevent training
        data poisoning.  Rejected records are logged at warning level.

        Records with ``tokens_used=0`` or ``latency_ms=0`` are silently
        discarded because those values indicate a fallback or mock response
        that must not influence model selection or quality scoring.

        The basis-aware filter rejects LLM-judgment signals aimed at tool-evidence
        training streams.  The typed ``evidence_basis`` and ``target_stream``
        parameters take precedence over the legacy ``metadata`` string keys;
        if both are supplied and disagree, the typed param wins and a WARNING
        is logged.

        Args:
            task: Short human-readable description of the task that was executed.
            prompt: The full prompt text sent to the model.
            response: The model's raw response text.
            score: Quality score in the range 0.0-1.0.
            model_id: Identifier of the model that produced the response
                (must be the *actual* model used, not the requested one).
            task_type: Broad category of the task (e.g. ``"general"``,
                ``"coding"``, ``"analysis"``). Defaults to ``"general"``.
            prompt_variant_id: Identifier of the prompt variant used for
                A/B comparison. Empty string when not applicable.
            agent_type: String identifier of the agent that ran the task.
                Empty string when not applicable.
            latency_ms: Wall-clock inference latency in milliseconds.
                Records with ``latency_ms=0`` are rejected.
            tokens_used: Total tokens consumed (prompt + completion).
                Records with ``tokens_used=0`` are rejected.
            success: Whether the task completed without an error.
            metadata: Optional free-form dict of additional context
                (e.g. temperature, strategy).
            benchmark_suite: Name of the benchmark suite that validated
                this output. Empty string when no benchmark was run.
            benchmark_pass: Whether the output passed benchmark validation.
            benchmark_score: Benchmark score in the range 0.0-1.0.
            rejection_reason: Why the Inspector rejected this output; empty
                string when the output was accepted.
            rejection_category: Short failure category label from the Inspector
                verdict (e.g. ``"quality_rejection"``); empty string when
                accepted.
            inspector_feedback: Full Inspector feedback summary text; empty
                string when no feedback was recorded.
            trace_id: Pipeline trace identifier for end-to-end observability.
                Propagated from CorrelationContext through the pipeline.
            evidence_basis: Typed EvidenceBasis enum value for the basis-aware
                filter.  Takes precedence over ``metadata["evidence_basis"]``
                when both are present.  Use this instead of the metadata key
                for new callers.
            target_stream: Target training stream name (e.g. ``"tool_evidence"``).
                Takes precedence over ``metadata["training_stream"]`` when both
                are present.
        """
        # -- Data quality gates --
        if tokens_used == 0:
            logger.warning(
                "[TrainingDataCollector] Rejected record: tokens_used=0 (likely fallback/mock), model=%s",
                model_id,
            )
            return
        if latency_ms == 0:
            logger.warning(
                "[TrainingDataCollector] Rejected record: latency_ms=0 (likely fallback/mock), model=%s",
                model_id,
            )
            return
        # Check _is_fallback field first (primary), then blocklist (secondary)
        if metadata and metadata.get("_is_fallback"):
            logger.warning(
                "[TrainingDataCollector] Rejected record: _is_fallback=True, model=%s",
                model_id,
            )
            return
        # Basis-aware filter: LLM_JUDGMENT signals must not enter tool-evidence
        # training streams.
        #
        # Source-of-truth ordering (typed params beat metadata strings):
        # 1. Resolve basis: typed param > metadata["evidence_basis"] fallback.
        #    If both are set and disagree, log a WARNING and use the typed param.
        # 2. Resolve stream: typed param > metadata["training_stream"] fallback.
        # 3. Reject when basis is LLM_JUDGMENT and stream is "tool_evidence".
        _meta_basis_str: str = (metadata or {}).get("evidence_basis", "") if metadata else ""
        _meta_stream_str: str = (metadata or {}).get("training_stream", "") if metadata else ""

        # Resolve basis value
        if evidence_basis is not None:
            if _meta_basis_str and _meta_basis_str != evidence_basis.value:
                logger.warning(
                    "[TrainingDataCollector] evidence_basis mismatch: typed param=%s, "
                    "metadata=%s — using typed param (model=%s)",
                    evidence_basis.value,
                    _meta_basis_str,
                    model_id,
                )
            _resolved_basis_str: str = evidence_basis.value
        else:
            _resolved_basis_str = _meta_basis_str

        # Resolve stream value
        _resolved_stream: str = target_stream if target_stream is not None else _meta_stream_str

        if _resolved_basis_str == EvidenceBasis.LLM_JUDGMENT.value and _resolved_stream == "tool_evidence":
            logger.warning(
                "[TrainingDataCollector] Rejected record: LLM-judgment signal (basis=%s) cannot enter "
                "tool-evidence training stream (stream=%s, model=%s, task_type=%s)",
                _resolved_basis_str,
                _resolved_stream,
                model_id,
                task_type,
            )
            return
        _stripped = response.strip() if response else ""
        if _stripped in _FALLBACK_RESPONSE_PATTERNS:
            logger.warning(
                "[TrainingDataCollector] Rejected record: response matches fallback pattern, model=%s",
                model_id,
            )
            return
        # Secrets filter: block records containing credentials or API keys
        try:
            from vetinari.learning.secrets_filter import filter_training_record

            is_safe, detections = filter_training_record(prompt, response)
            if not is_safe:
                logger.warning(
                    "[TrainingDataCollector] Rejected record: secrets detected (%d findings), model=%s",
                    len(detections),
                    model_id,
                )
                return
        except Exception:
            logger.warning(
                "[TrainingDataCollector] Secrets filter unavailable — proceeding without filter for model=%s",
                model_id,
            )

        # Snapshot VRAM usage if available
        vram_used = 0.0
        try:
            from vetinari.models.vram_manager import get_vram_manager

            vram_used = get_vram_manager().get_used_vram_gb()
        except Exception:
            logger.warning("Failed to snapshot VRAM usage for training record", exc_info=True)

        import uuid

        rec = TrainingRecord(
            record_id=f"tr_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            task=task[:TRUNCATE_OUTPUT_PREVIEW],
            prompt=prompt[:TRUNCATE_PROMPT_TRAINING],
            response=response[:TRUNCATE_RESPONSE_TRAINING],
            score=round(score, 4),
            model_id=model_id,
            task_type=task_type,
            prompt_variant_id=prompt_variant_id,
            agent_type=agent_type,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            success=success,
            vram_used_gb=round(vram_used, 2),
            benchmark_suite=benchmark_suite,
            benchmark_pass=benchmark_pass,
            benchmark_score=round(benchmark_score, 4),
            rejection_reason=rejection_reason,
            rejection_category=rejection_category,
            inspector_feedback=inspector_feedback,
            trace_id=trace_id,
            metadata=metadata or {},  # noqa: VET112 - empty fallback preserves optional request metadata contract
        )
        if self._sync:
            # Synchronous mode: write directly (used in tests)
            self._append(rec)
            return

        try:
            self._queue.put_nowait(rec)
        except queue.Full:
            logger.warning(
                "[TrainingDataCollector] Queue full — training record dropped. "
                "Consider increasing TRAINING_COLLECTOR_QUEUE_SIZE"
            )

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------

    def _write_worker(self) -> None:
        """Background thread: drain the record queue and write to JSONL."""
        while not self._shutdown.is_set():
            try:
                rec = self._queue.get(timeout=QUEUE_TIMEOUT)
                if rec is None:
                    # Sentinel placed by shutdown() — exit immediately.
                    return
                self._append(rec)
                self._queue.task_done()
            except queue.Empty:
                logger.warning("Training data queue poll timed out — no pending records, will retry")
                continue  # Normal timeout — no pending records
            except Exception as e:
                logger.warning("[TrainingDataCollector] Write error: %s", e)

    def _append(self, rec: TrainingRecord) -> None:
        """Append a record to the JSONL file (thread-safe).

        Args:
            rec: The TrainingRecord to persist.
        """
        with self._lock:
            try:
                self._output_path.parent.mkdir(parents=True, exist_ok=True)
                with Path(self._output_path).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                self._record_count += 1
            except Exception as e:
                logger.warning("[TrainingDataCollector] Append failed: %s", e)

    def flush(self, timeout: float = THREAD_JOIN_TIMEOUT) -> None:
        """Wait for the queue to drain, with timeout to avoid deadlocks.

        Args:
            timeout: Maximum seconds to wait for the queue to drain.
        """
        done = threading.Event()

        def _wait_join() -> None:
            self._queue.join()
            done.set()

        t = threading.Thread(target=_wait_join, daemon=True)
        t.start()
        done.wait(timeout=timeout)

    def shutdown(self, timeout: float = THREAD_JOIN_TIMEOUT) -> None:
        """Flush remaining records and stop the background worker.

        Signals the writer thread via the shutdown event and a sentinel queue
        entry so it unblocks immediately, then waits up to ``timeout`` seconds
        for it to exit.

        Args:
            timeout: Maximum seconds to wait for the worker thread to finish.
        """
        self.flush()
        self._shutdown.set()
        # Sentinel wakes the worker if it is blocked in queue.get().
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal data access
    # ------------------------------------------------------------------

    def _load_all(self) -> list[TrainingRecord]:
        """Load all records from the JSONL file.

        Returns:
            List of TrainingRecord objects; malformed lines are skipped with
            a warning.
        """
        if not self._output_path.exists():
            return []
        records: list[TrainingRecord] = []
        with Path(self._output_path).open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    records.append(TrainingRecord(**d))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    logger.warning(
                        "Skipping malformed training record in JSONL: %s",
                        line[:100] if isinstance(line, str) else repr(line)[:100],
                    )
                    continue
        return records

    # ------------------------------------------------------------------
    # Stats and counts
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return a summary of collected training data.

        Returns:
            Dict with keys: total, queued, sft_eligible (records scoring >=
            0.8), avg_score, by_task_type (per-type count and avg_score),
            and output_path. Returns ``{"total": 0, "queued": N}`` when no
            records have been written yet.
        """
        self.flush()
        all_records = self._load_all()
        if not all_records:
            return {"total": 0, "queued": self._queue.qsize()}

        by_type: dict[str, list[float]] = defaultdict(list)
        for r in all_records:
            by_type[r.task_type].append(r.score)

        return {
            "total": len(all_records),
            "queued": self._queue.qsize(),
            "sft_eligible": sum(1 for r in all_records if r.score >= 0.8),
            "avg_score": round(sum(r.score for r in all_records) / len(all_records), 3),
            "by_task_type": {
                tt: {
                    "count": len(scores),
                    "avg_score": round(sum(scores) / len(scores), 3),
                }
                for tt, scores in by_type.items()
            },
            "output_path": str(self._output_path),
        }

    def count_records(self) -> int:
        """Return the total number of recorded training examples.

        Returns:
            Count of records written to the backing JSONL file.
        """
        return self.get_stats().get("total", 0)

    def count_reasoning_episodes(self) -> int:
        """Count records with reasoning-related task types.

        Counts records where ``task_type`` is one of planning, reasoning,
        decomposition, or analysis — used by the curriculum to decide when
        self-play reasoning training is worthwhile.

        Returns:
            Number of reasoning-related records.
        """
        reasoning_types = {"planning", "reasoning", "decomposition", "analysis"}
        count = 0
        try:
            with self._lock:
                if self._output_path.exists():
                    with open(self._output_path, encoding="utf-8") as fh:
                        for line in fh:
                            with contextlib.suppress(json.JSONDecodeError):
                                rec = json.loads(line)
                                if rec.get("task_type", "") in reasoning_types:
                                    count += 1
        except OSError:
            logger.warning("Could not read training data for reasoning episode count — returning 0")
        return count

    def count_execution_traces(self) -> int:
        """Count records with code-execution task types.

        Counts records where ``task_type`` is one of coding, implementation,
        bug_fix, or refactoring — used by the curriculum to decide when RLEF
        (Reinforcement Learning from Execution Feedback) training is worthwhile.

        Returns:
            Number of code-execution-related records.
        """
        execution_types = {"coding", "implementation", "bug_fix", "refactoring"}
        count = 0
        try:
            with self._lock:
                if self._output_path.exists():
                    with open(self._output_path, encoding="utf-8") as fh:
                        for line in fh:
                            with contextlib.suppress(json.JSONDecodeError):
                                rec = json.loads(line)
                                if rec.get("task_type", "") in execution_types:
                                    count += 1
        except OSError:
            logger.warning("Could not read training data for execution trace count — returning 0")
        return count


# ---------------------------------------------------------------------------
# Module-level accessor (double-checked locking)
# ---------------------------------------------------------------------------

# Module-level singleton state.
# Written by: get_training_collector() on first call.
# Read by: all callers that need the shared instance.
# Lock: _collector_lock guards the check-then-create sequence.
_collector: TrainingDataCollector | None = None
_collector_lock = threading.Lock()


def get_training_collector(
    output_path: str = _DEFAULT_PATH,
) -> TrainingDataCollector:
    """Return the global TrainingDataCollector singleton.

    Thread-safe: uses double-checked locking to ensure a single instance and
    background writer thread are created even under concurrent startup.

    Args:
        output_path: Path to the JSONL file where records are written.
            Only the first call's value takes effect.

    Returns:
        The shared TrainingDataCollector instance for this process.
    """
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = TrainingDataCollector.get_instance(output_path)
    elif output_path != _DEFAULT_PATH and str(_collector._output_path) != output_path:
        logger.warning(
            "get_training_collector() called with output_path=%r but singleton already "
            "created with output_path=%r — ignoring new path; data will continue to be "
            "written to the existing path",
            output_path,
            str(_collector._output_path),
        )
    return _collector
