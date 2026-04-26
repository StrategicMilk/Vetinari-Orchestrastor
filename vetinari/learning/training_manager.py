"""Vetinari Training Manager.

=========================

Unified interface for managing fine-tuning workflows:

- Prepares training datasets from collected execution records
- Provides recommended hyperparameter configurations
- Submits local (Unsloth/LoRA) and cloud (HuggingFace) training jobs
- Tracks job status and recommends retraining when quality degrades

All heavy optional dependencies (unsloth, huggingface_hub) are handled
gracefully — if they are not installed, methods return structured error
results with installation instructions rather than raising ImportError.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.types import AgentType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TrainingDataset:
    """Prepared dataset ready for a training run."""

    records: list[dict[str, Any]]
    format: str  # "sft", "dpo", "hf", "ranking"
    stats: dict[str, Any]  # count, avg_score, task_type_breakdown

    def __repr__(self) -> str:
        return f"TrainingDataset(format={self.format!r}, records={len(self.records)})"


@dataclass
class TrainingResult:
    """Result of a completed (or failed) training run."""

    success: bool
    model_path: str | None
    metrics: dict[str, Any]  # loss, eval_loss, etc.
    duration_seconds: float
    error: str | None = None

    def __repr__(self) -> str:
        return (
            f"TrainingResult(success={self.success!r}, model_path={self.model_path!r}, "
            f"duration_seconds={self.duration_seconds!r})"
        )


@dataclass
class TrainingJob:
    """A tracked training job (local or cloud)."""

    job_id: str
    status: str  # StatusEnum: pending, running, completed, failed
    provider: str
    model_id: str
    created_at: str
    progress: float = 0.0
    result: TrainingResult | None = None

    def __repr__(self) -> str:
        return (
            f"TrainingJob(job_id={self.job_id!r}, status={self.status!r}, "
            f"provider={self.provider!r}, model_id={self.model_id!r}, "
            f"progress={self.progress!r})"
        )


@dataclass
class RetrainingRecommendation:
    """Whether a model/task combination warrants retraining."""

    model_id: str
    task_type: str
    current_avg_quality: float
    baseline_quality: float
    degradation: float  # fractional — 0.15 means 15 %
    recommended: bool
    reason: str
    recommended_method: str = "qlora"
    recommended_min_score: float = 0.85

    def __repr__(self) -> str:
        return (
            f"RetrainingRecommendation(model_id={self.model_id!r}, "
            f"task_type={self.task_type!r}, recommended={self.recommended!r}, "
            f"degradation={self.degradation!r})"
        )


# ---------------------------------------------------------------------------
# TrainingManager
# ---------------------------------------------------------------------------

# Degradation threshold that triggers a retraining recommendation
_RETRAIN_DEGRADATION_THRESHOLD = 0.15
# Minimum baseline quality to compare against
_BASELINE_QUALITY = 0.80
# Minimum records required for a valid local LoRA run
_MIN_QLORA_RECORDS = 100


class TrainingManager:
    """Unified training workflow coordinator.

    Uses :func:`get_training_collector` internally to access accumulated
    execution records without requiring a path argument on every call.
    """

    def __init__(self, data_path: str | None = None):
        self._data_path = data_path
        self._jobs: dict[str, TrainingJob] = {}

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _get_collector(self):
        from vetinari.learning.training_data import TrainingDataCollector, get_training_collector

        if self._data_path:
            # Use a dedicated instance so different data_paths don't share state
            if not hasattr(self, "_collector_instance"):
                self._collector_instance = TrainingDataCollector(output_path=self._data_path)
            return self._collector_instance
        return get_training_collector()

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_training_data(
        self,
        min_score: float = 0.8,
        format: str = "sft",
        task_type: str | None = None,
    ) -> TrainingDataset:
        """Prepare a training dataset using the requested format.

        Parameters
        ----------
        min_score:
            Minimum quality score for records to include.
        format:
            One of ``"sft"``, ``"dpo"``, ``"hf"``, or ``"ranking"``.
        task_type:
            Optional filter by task type (e.g. ``"coding"``).

        Returns:
        -------
        TrainingDataset
            Populated with records, format tag, and summary statistics.

        Args:
            min_score: The min score.
            format: The format.
            task_type: The task type.
        """
        collector = self._get_collector()

        if format == "hf":
            records = collector.export_hf_dataset(min_score=min_score, task_type=task_type)
        elif format == "dpo":
            records = collector.export_dpo_dataset(task_type=task_type)
        elif format == "ranking":
            records = collector.export_ranking_dataset()
        else:
            # default: sft
            records = collector.export_sft_dataset(min_score=min_score, task_type=task_type)

        # Build stats
        count = len(records)
        avg_score = 0.0
        task_type_breakdown: dict[str, int] = {}

        if format == "sft":
            scores = [r.get("score", 0.0) for r in records if isinstance(r.get("score"), (int, float))]
            avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
            for r in records:
                tt = r.get("task_type", "unknown")
                task_type_breakdown[tt] = task_type_breakdown.get(tt, 0) + 1
        elif format == "hf":
            # HF format has no score field; use collector stats for approximation
            try:
                raw_stats = collector.get_stats()
                avg_score = raw_stats.get("avg_score", 0.0)
            except Exception:
                logger.warning("Could not retrieve HF-format collector stats; avg_score remains 0.0", exc_info=True)

        stats: dict[str, Any] = {
            "count": count,
            "avg_score": avg_score,
            "task_type_breakdown": task_type_breakdown,
        }
        return TrainingDataset(records=records, format=format, stats=stats)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_training_config(self, method: str = "qlora") -> dict[str, Any]:
        """Return recommended hyperparameters for a training method.

        Parameters
        ----------
        method:
            ``"qlora"`` for QLoRA/Unsloth or ``"full"`` for full fine-tuning.

        Returns:
        -------
        dict
            Hyperparameter dictionary ready to pass to a training framework.
        """
        if method == "full":
            return {
                "method": "full",
                "lr": 1e-5,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
            }
        # qlora with DoRA (default) — DoRA outperforms standard LoRA at small ranks
        return {
            "method": "qlora",
            "lr": 2e-4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0,
            "target_modules": "all_linear",
            "use_dora": True,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "load_in_4bit": True,
        }

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def train_local(
        self,
        model_id: str,
        dataset: TrainingDataset,
        method: str = "qlora",
        config: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Attempt a local fine-tuning run using Unsloth/LoRA.

        If ``unsloth`` is not installed, returns a failed result with
        installation instructions — it does NOT raise an ImportError.

        Parameters
        ----------
        model_id:
            HuggingFace model identifier or local path.
        dataset:
            Prepared dataset from :meth:`prepare_training_data`.
        method:
            ``"qlora"`` (default) or ``"full"``.
        config:
            Override hyperparameters (merged with :meth:`get_training_config`).

        Args:
            model_id: The model id.
            dataset: The dataset.
            method: The method.
            config: The config.

        Returns:
            The TrainingResult result.
        """
        start = time.monotonic()
        job_id = f"local-{int(start * 1000)}"

        def _record_job(status: str, progress: float, result: TrainingResult | None = None) -> None:
            self._jobs[job_id] = TrainingJob(
                job_id=job_id,
                status=status,
                provider="local",
                model_id=model_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                progress=progress,
                result=result,
            )

        _record_job("running", 0.0)

        # Validate dataset size
        min_records = _MIN_QLORA_RECORDS if method == "qlora" else 50
        if len(dataset.records) < min_records:
            duration = round(time.monotonic() - start, 2)
            result = TrainingResult(
                success=False,
                model_path=None,
                metrics={},
                duration_seconds=duration,
                error=(
                    f"Dataset too small: {len(dataset.records)} records "
                    f"(minimum {min_records} required for {method}). "
                    "Collect more execution data or lower min_score."
                ),
            )
            _record_job("failed", 1.0, result)
            return result

        # Merge config
        hparams = self.get_training_config(method)
        if config:
            hparams.update(config)

        # Delegate to the actual training pipeline
        try:
            from vetinari.training.pipeline import TrainingPipeline

            pipeline = TrainingPipeline()
            reqs = pipeline.check_requirements()
            if not reqs.get("ready_for_training", False):
                missing = [lib for lib, avail in reqs.get("libraries", {}).items() if not avail]
                duration = round(time.monotonic() - start, 2)
                result = TrainingResult(
                    success=False,
                    model_path=None,
                    metrics={},
                    duration_seconds=duration,
                    error=(
                        "Training libraries not installed. Missing: "
                        + ", ".join(missing)
                        + ". Install with: pip install trl peft bitsandbytes transformers"  # noqa: VET301 — user guidance string
                    ),
                )
                _record_job("failed", 1.0, result)
                return result
        except (ImportError, ValueError):
            # ValueError: torch.__spec__ is None when torch is partially installed
            duration = round(time.monotonic() - start, 2)
            logger.warning(
                "Training pipeline module not importable — trl/peft/transformers may be missing; "
                "training aborted after %.2fs",
                duration,
            )
            result = TrainingResult(
                success=False,
                model_path=None,
                metrics={},
                duration_seconds=duration,
                error="Training pipeline module not available.",
            )
            _record_job("failed", 1.0, result)
            return result

        # Derive the dominant task type from the dataset's breakdown stats so
        # the pipeline can label the run correctly even when curation is skipped.
        task_type_from_dataset: str | None = None
        if dataset.stats.get("task_type_breakdown"):
            breakdown = dataset.stats["task_type_breakdown"]
            if isinstance(breakdown, dict) and breakdown:
                task_type_from_dataset = max(breakdown, key=lambda k: breakdown[k])

        # Serialize the validated dataset to a JSONL file so the pipeline uses
        # exactly the records that were validated above rather than re-curating
        # from the collector (which may include new, un-validated data).
        _ds_dir = tempfile.mkdtemp(prefix="vetinari_train_")
        _ds_path = str(Path(_ds_dir) / "dataset.jsonl")
        try:
            with Path(_ds_path).open("w", encoding="utf-8") as _f:
                for rec in dataset.records:
                    _rec_dict = rec.to_dict() if hasattr(rec, "to_dict") else (
                        rec if isinstance(rec, dict) else vars(rec)
                    )
                    _f.write(json.dumps(_rec_dict, ensure_ascii=False) + "\n")
            logger.info(
                "[TrainingManager] Serialized %d validated records to %s",
                len(dataset.records),
                _ds_path,
            )
        except Exception as _exc:
            logger.warning(
                "[TrainingManager] Could not serialize dataset to JSONL (%s) — pipeline will re-curate",
                _exc,
            )
            _ds_path = None  # type: ignore[assignment]

        logger.info("[TrainingManager] Starting local %s training for %s", method, model_id)
        try:
            run = pipeline.run(
                base_model=model_id,
                task_type=task_type_from_dataset,
                min_score=0.8,
                epochs=hparams.get("num_train_epochs", 3),
                dataset_path=_ds_path,
            )
        except Exception as exc:
            duration = round(time.monotonic() - start, 2)
            logger.exception("[TrainingManager] Training pipeline failed for %s", model_id)
            result = TrainingResult(
                success=False,
                model_path=None,
                metrics={},
                duration_seconds=duration,
                error=str(exc),
            )
            _record_job("failed", 1.0, result)
            return result

        duration = round(time.monotonic() - start, 2)

        # Record training outcome for agent-level tracking
        try:
            from vetinari.training.agent_trainer import AgentTrainer

            trainer = AgentTrainer()
            trainer.record_training(
                agent_type=AgentType.WORKER.value,
                model_path=model_id,
                metrics={
                    "method": method,
                    "success": run.success,
                    "training_examples": run.training_examples,
                    "eval_score": run.eval_score,
                },
            )
        except Exception:
            logger.warning("[TrainingManager] Could not record training to agent history", exc_info=True)

        result = TrainingResult(
            success=run.success,
            model_path=run.output_model_path,
            metrics={
                "training_examples": run.training_examples,
                "eval_score": run.eval_score,
            },
            duration_seconds=duration,
            error=None if run.success else "Training pipeline reported failure",
        )
        _record_job("completed" if result.success else "failed", 1.0, result)
        return result

    # ------------------------------------------------------------------
    # Cloud training
    # ------------------------------------------------------------------

    def train_cloud(
        self,
        model_id: str,
        dataset: TrainingDataset,
        provider: str = "huggingface",
        config: dict[str, Any] | None = None,
    ) -> TrainingJob:
        """Submit a cloud fine-tuning job.

        Returns a :class:`TrainingJob` in ``"pending"`` status.  Full API
        integration is an optional feature — if the provider SDK is not
        configured, the job stub includes setup instructions in the job ID.

        Parameters
        ----------
        model_id:
            Base model identifier.
        dataset:
            Prepared dataset from :meth:`prepare_training_data`.
        provider:
            ``"huggingface"`` (default) or another provider name.
        config:
            Optional provider-specific configuration overrides.

        Args:
            model_id: The model id.
            dataset: The dataset.
            provider: The provider.
            config: The config.

        Returns:
            The TrainingJob result.

        Raises:
            RuntimeError: Always — cloud training is not yet implemented.
        """
        raise RuntimeError(
            f"Cloud training via '{provider}' is not yet implemented. "
            "Use train_local() for on-device fine-tuning with QLoRA/DoRA, "
            "or contribute a provider integration."
        )

    # ------------------------------------------------------------------
    # Job registry
    # ------------------------------------------------------------------

    def get_training_status(self, job_id: str) -> TrainingJob | None:
        """Look up a training job by its ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[TrainingJob]:
        """Return all tracked training jobs."""
        return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Retraining recommendations
    # ------------------------------------------------------------------

    def should_retrain(self, model_id: str, task_type: str) -> RetrainingRecommendation:
        """Evaluate whether a model warrants retraining for a task type.

        Compares the rolling average quality of recent records against
        :data:`_BASELINE_QUALITY`.  Recommends retraining when degradation
        exceeds :data:`_RETRAIN_DEGRADATION_THRESHOLD` (15 %).

        Parameters
        ----------
        model_id:
            Model to evaluate.
        task_type:
            Task category to filter records by.

        Args:
            model_id: The model id.
            task_type: The task type.

        Returns:
            The RetrainingRecommendation result.
        """
        collector = self._get_collector()
        try:
            all_records = collector._load_all()
        except Exception:
            all_records = []

        # Filter to this model + task. Support wildcard patterns in model_id so
        # callers can match a family of models (e.g., "qwen/*") without enumerating
        # every variant. fnmatch.fnmatch() treats non-wildcard strings as exact matches,
        # so "qwen-7b" only matches records for that exact model.
        relevant = [
            r
            for r in all_records
            if fnmatch.fnmatch(r.model_id, model_id) and fnmatch.fnmatch(r.task_type, task_type)
        ]

        if not relevant:
            return RetrainingRecommendation(
                model_id=model_id,
                task_type=task_type,
                current_avg_quality=0.0,
                baseline_quality=_BASELINE_QUALITY,
                degradation=0.0,
                recommended=False,
                reason="No records found for this model/task combination.",
            )

        current_avg = round(sum(r.score for r in relevant) / len(relevant), 4)
        degradation = round(max(0.0, (_BASELINE_QUALITY - current_avg) / _BASELINE_QUALITY), 4)
        recommended = degradation >= _RETRAIN_DEGRADATION_THRESHOLD

        if recommended:
            reason = (
                f"Quality degraded {degradation * 100:.1f}% below baseline "
                f"({current_avg:.3f} vs {_BASELINE_QUALITY:.3f} baseline). "
                f"Retraining on {len(relevant)} records recommended."
            )
        else:
            reason = (
                f"Quality acceptable: {current_avg:.3f} "
                f"(baseline {_BASELINE_QUALITY:.3f}, "
                f"degradation {degradation * 100:.1f}%)."
            )

        return RetrainingRecommendation(
            model_id=model_id,
            task_type=task_type,
            current_avg_quality=current_avg,
            baseline_quality=_BASELINE_QUALITY,
            degradation=degradation,
            recommended=recommended,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: TrainingManager | None = None
_manager_lock = threading.Lock()


def get_training_manager(data_path: str | None = None) -> TrainingManager:
    """Return the global TrainingManager singleton.

    When an explicit ``data_path`` is provided and the singleton has not yet
    been created, the new instance uses that path.  Once the singleton exists,
    passing a different explicit ``data_path`` raises ``ValueError`` — callers
    that genuinely need a different path must instantiate ``TrainingManager``
    directly.  Passing ``data_path=None`` always returns the existing singleton
    without raising.

    Args:
        data_path: Optional path to the training data directory.  Only the
            first non-``None`` call's value takes effect; a later call with a
            *different* non-``None`` value raises ``ValueError``.

    Returns:
        The shared TrainingManager instance for this process.

    Raises:
        ValueError: If the singleton was already created with a different
            ``data_path`` and the caller passes a new non-``None`` value.
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = TrainingManager(data_path=data_path)
    elif data_path is not None and _manager._data_path != data_path:
        raise ValueError(
            f"get_training_manager: singleton already initialized with data_path={_manager._data_path!r}; "
            f"refusing to re-init with data_path={data_path!r}. "
            "Instantiate TrainingManager directly if a different path is required."
        )
    return _manager
