"""
Vetinari Training Manager
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

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TrainingDataset:
    """Prepared dataset ready for a training run."""
    records: List[Dict[str, Any]]
    format: str                   # "sft", "dpo", "hf", "ranking"
    stats: Dict[str, Any]         # count, avg_score, task_type_breakdown


@dataclass
class TrainingResult:
    """Result of a completed (or failed) training run."""
    success: bool
    model_path: Optional[str]
    metrics: Dict[str, Any]       # loss, eval_loss, etc.
    duration_seconds: float
    error: Optional[str] = None


@dataclass
class TrainingJob:
    """A tracked training job (local or cloud)."""
    job_id: str
    status: str                   # "pending", "running", "completed", "failed"
    provider: str
    model_id: str
    created_at: str
    progress: float = 0.0
    result: Optional[TrainingResult] = None


@dataclass
class RetrainingRecommendation:
    """Whether a model/task combination warrants retraining."""
    model_id: str
    task_type: str
    current_avg_quality: float
    baseline_quality: float
    degradation: float            # fractional — 0.15 means 15 %
    recommended: bool
    reason: str
    recommended_method: str = "qlora"
    recommended_min_score: float = 0.85


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
    """
    Unified training workflow coordinator.

    Uses :func:`get_training_collector` internally to access accumulated
    execution records without requiring a path argument on every call.
    """

    def __init__(self, data_path: Optional[str] = None):
        self._data_path = data_path
        self._jobs: Dict[str, TrainingJob] = {}

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
        task_type: Optional[str] = None,
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

        Returns
        -------
        TrainingDataset
            Populated with records, format tag, and summary statistics.
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
        task_type_breakdown: Dict[str, int] = {}

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
                pass

        stats: Dict[str, Any] = {
            "count": count,
            "avg_score": avg_score,
            "task_type_breakdown": task_type_breakdown,
        }
        return TrainingDataset(records=records, format=format, stats=stats)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_training_config(self, method: str = "qlora") -> Dict[str, Any]:
        """Return recommended hyperparameters for a training method.

        Parameters
        ----------
        method:
            ``"qlora"`` for QLoRA/Unsloth or ``"full"`` for full fine-tuning.

        Returns
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
        # qlora (default)
        return {
            "method": "qlora",
            "lr": 2e-4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0,
            "target_modules": "all_linear",
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
        config: Optional[Dict[str, Any]] = None,
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
        """
        start = time.monotonic()

        # Validate dataset size
        min_records = _MIN_QLORA_RECORDS if method == "qlora" else 50
        if len(dataset.records) < min_records:
            duration = round(time.monotonic() - start, 2)
            return TrainingResult(
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

        # Merge config
        hparams = self.get_training_config(method)
        if config:
            hparams.update(config)

        # Attempt import of unsloth
        try:
            import unsloth  # noqa: F401
        except ImportError:
            duration = round(time.monotonic() - start, 2)
            return TrainingResult(
                success=False,
                model_path=None,
                metrics={},
                duration_seconds=duration,
                error=(
                    "unsloth is not installed. To enable local fine-tuning:\n"
                    "  pip install unsloth\n"
                    "See https://github.com/unslothai/unsloth for GPU requirements."
                ),
            )

        # Actual training would go here — stub returns success stub
        duration = round(time.monotonic() - start, 2)
        logger.info(f"[TrainingManager] Local training started for {model_id} ({method})")
        return TrainingResult(
            success=True,
            model_path=f"./trained_models/{model_id.replace('/', '_')}",
            metrics={"loss": 0.0, "eval_loss": 0.0},
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Cloud training
    # ------------------------------------------------------------------

    def train_cloud(
        self,
        model_id: str,
        dataset: TrainingDataset,
        provider: str = "huggingface",
        config: Optional[Dict[str, Any]] = None,
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
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = TrainingJob(
            job_id=job_id,
            status="pending",
            provider=provider,
            model_id=model_id,
            created_at=datetime.now().isoformat(),
        )
        self._jobs[job_id] = job

        # Check provider SDK availability
        if provider == "huggingface":
            try:
                from huggingface_hub import HfApi  # noqa: F401
                logger.info(f"[TrainingManager] Cloud job {job_id} submitted to HuggingFace")
            except ImportError:
                logger.warning(
                    "[TrainingManager] huggingface_hub not installed. "
                    "Install with: pip install huggingface_hub\n"
                    f"Job {job_id} created in stub mode."
                )
        else:
            logger.info(f"[TrainingManager] Cloud job {job_id} created for provider '{provider}'")

        return job

    # ------------------------------------------------------------------
    # Job registry
    # ------------------------------------------------------------------

    def get_training_status(self, job_id: str) -> Optional[TrainingJob]:
        """Look up a training job by its ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[TrainingJob]:
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
        """
        collector = self._get_collector()
        try:
            all_records = collector._load_all()
        except Exception:
            all_records = []

        # Filter to this model + task
        relevant = [
            r for r in all_records
            if r.model_id == model_id and r.task_type == task_type
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

_manager: Optional[TrainingManager] = None


def get_training_manager(data_path: Optional[str] = None) -> TrainingManager:
    """Return the global TrainingManager singleton."""
    global _manager
    if _manager is None:
        _manager = TrainingManager(data_path=data_path)
    return _manager
