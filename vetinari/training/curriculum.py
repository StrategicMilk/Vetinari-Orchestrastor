"""Training Curriculum Module.

Determines what the system should train on next by evaluating the current
state of skill gaps, defect patterns, available data, and benchmark staleness.
Produces a prioritized TrainingActivity that drives the training pipeline.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Minimum episodes before self-play reasoning is worth scheduling
MIN_REASONING_EPISODES = 50

# Minimum execution traces before RLEF training is worth scheduling
MIN_RLEF_TRACES = 30

# Minimum defect occurrences before it becomes a training priority
DEFECT_THRESHOLD = 5

# Score below which a skill is considered weak and needs targeted training
WEAK_SKILL_THRESHOLD = 0.7

# Score threshold for distillation — only distill from high-quality cloud outputs
DISTILLATION_QUALITY_THRESHOLD = 0.85

# Days after which benchmarks are considered stale
BENCHMARK_STALENESS_DAYS = 7

# Default calibration activity estimated values
DEFAULT_DURATION_MINUTES = 30
DEFAULT_VRAM_GB = 4.0

logger = logging.getLogger(__name__)


class TrainingActivityType(Enum):
    """The category of training work to be performed."""

    FINE_TUNE_WEAK_SKILL = "fine_tune_weak_skill"
    SELF_PLAY_REASONING = "self_play_reasoning"
    EXTERNAL_DATA_TRAINING = "external_data_training"
    PROMPT_EVOLUTION = "prompt_evolution"
    DISTILLATION = "distillation"
    BENCHMARK_PRACTICE = "benchmark_practice"
    RLEF_CODE_EXECUTION = "rlef_code_execution"


class CurriculumPhase(Enum):
    """Broad phase of the training curriculum lifecycle."""

    CALIBRATION = "calibration"
    TARGETED_SKILL_BUILDING = "targeted_skill_building"
    CONTINUOUS_IMPROVEMENT = "continuous_improvement"


@dataclass
class TrainingActivity:
    """A concrete unit of training work with full scheduling metadata.

    Attributes:
        type: The category of training to perform.
        description: Human-readable description of the activity.
        hypothesis: What improvement is expected and why.
        metric: Name of the metric this activity is expected to move.
        baseline: Current measured value of the metric.
        target: Desired value after training completes.
        rollback_plan: How to revert if training degrades performance.
        estimated_duration_minutes: Wall-clock time estimate.
        estimated_vram_gb: GPU memory required during training.
        priority: Urgency score 0-1, higher is more urgent.
        metadata: Arbitrary extra data for pipeline consumers.
    """

    type: TrainingActivityType
    description: str
    hypothesis: str
    metric: str
    baseline: float
    target: float
    rollback_plan: str
    estimated_duration_minutes: int
    estimated_vram_gb: float
    priority: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TrainingActivity(type={self.type.value!r}, metric={self.metric!r}, "
            f"baseline={self.baseline!r}, target={self.target!r}, priority={self.priority!r})"
        )


class TrainingCurriculum:
    """Determines what training activity should run next.

    Evaluates the current system state — skill gaps, defect patterns,
    available data, pending A/B tests, and benchmark freshness — and
    returns the highest-priority TrainingActivity.

    Late imports are used for all vetinari submodules so that training
    dependencies remain optional at import time.
    """

    def __init__(self) -> None:
        self._candidates: list[TrainingActivity] = []

    # ── Candidate builders ──────────────────────────────────────────────────

    def _candidate_weak_skill(self) -> TrainingActivity | None:
        """Return a fine-tune activity for the weakest skill, or None.

        Returns:
            A FINE_TUNE_WEAK_SKILL activity if a skill scores below the
            threshold, otherwise None.
        """
        try:
            from vetinari.learning.model_selector import get_skill_rankings

            rankings = get_skill_rankings()
        except ImportError:
            logger.debug("model_selector not available; skipping weak-skill candidate")
            return None
        except Exception:
            logger.warning("Failed to get skill rankings; skipping weak-skill candidate", exc_info=True)
            return None

        if not rankings:
            return None

        # rankings is expected to be list[dict] with keys "task_type" and "score"
        try:
            worst = min(rankings, key=lambda r: r.get("score", 1.0))
            score = worst.get("score", 1.0)
            task_type = worst.get("task_type", "unknown")
        except (TypeError, KeyError):
            logger.warning("Unexpected skill rankings format; skipping")
            return None

        if score >= WEAK_SKILL_THRESHOLD:
            return None

        return TrainingActivity(
            type=TrainingActivityType.FINE_TUNE_WEAK_SKILL,
            description=f"Fine-tune weak skill: {task_type} (score={score:.2f})",
            hypothesis=(
                f"Targeted QLoRA fine-tuning on {task_type} examples will raise "
                f"skill score from {score:.2f} toward {WEAK_SKILL_THRESHOLD}."
            ),
            metric=f"skill_score.{task_type}",
            baseline=score,
            target=WEAK_SKILL_THRESHOLD,
            rollback_plan="Revert to previous adapter checkpoint; re-run benchmarks to confirm.",
            estimated_duration_minutes=120,
            estimated_vram_gb=16.0,
            priority=0.9,
            metadata={"task_type": task_type, "all_rankings": rankings},
        )

    def _candidate_defect_pattern(self) -> TrainingActivity | None:
        """Return a fine-tune activity for the top recurring defect, or None.

        Returns:
            A FINE_TUNE_WEAK_SKILL activity targeting the dominant defect
            pattern if it exceeds the threshold, otherwise None.
        """
        try:
            from vetinari.kaizen.improvement_log import ImprovementLog

            log = ImprovementLog()  # Uses unified DB via get_connection()
            top = log.get_top_defect_pattern()
        except ImportError:
            logger.debug("kaizen improvement_log not available; skipping defect-pattern candidate")
            return None
        except Exception:
            logger.warning("Failed to get top defect pattern; skipping", exc_info=True)
            return None

        if top is None:
            return None

        try:
            count = top.get("count", 0)
            pattern = top.get("pattern", "unknown")
        except (TypeError, AttributeError):
            logger.warning("Unexpected defect pattern format; skipping")
            return None

        if count <= DEFECT_THRESHOLD:
            return None

        return TrainingActivity(
            type=TrainingActivityType.FINE_TUNE_WEAK_SKILL,
            description=f"Address defect pattern: {pattern} ({count} occurrences)",
            hypothesis=(
                f"Training on corrected examples for defect '{pattern}' will reduce "
                f"recurrence rate. Pattern exceeds threshold of {DEFECT_THRESHOLD}."
            ),
            metric=f"defect_rate.{pattern}",
            baseline=float(count),
            target=float(DEFECT_THRESHOLD - 1),
            rollback_plan="Revert adapter; monitor defect rate over next 24h to confirm regression.",
            estimated_duration_minutes=90,
            estimated_vram_gb=16.0,
            priority=0.85,
            metadata={"defect_pattern": pattern, "occurrence_count": count},
        )

    def _candidate_self_play(self) -> TrainingActivity | None:
        """Return a self-play reasoning activity if enough episodes exist.

        Returns:
            A SELF_PLAY_REASONING activity if episode count meets the
            minimum threshold, otherwise None.
        """
        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            episode_count = collector.count_reasoning_episodes()
        except ImportError:
            logger.debug("training_data collector not available; skipping self-play candidate")
            return None
        except Exception:
            logger.warning("Failed to count reasoning episodes; skipping", exc_info=True)
            return None

        if episode_count < MIN_REASONING_EPISODES:
            return None

        return TrainingActivity(
            type=TrainingActivityType.SELF_PLAY_REASONING,
            description=f"Self-play reasoning training on {episode_count} episodes",
            hypothesis=(
                "Training on accumulated reasoning episode traces will improve "
                "multi-step planning accuracy and reduce chain-of-thought errors."
            ),
            metric="reasoning_accuracy",
            baseline=0.0,
            target=0.8,
            rollback_plan="Revert to pre-training checkpoint; compare benchmark suite scores.",
            estimated_duration_minutes=180,
            estimated_vram_gb=20.0,
            priority=0.7,
            metadata={"episode_count": episode_count},
        )

    def _candidate_external_data(self) -> TrainingActivity | None:
        """Return an external data training activity if datasets are available.

        Returns:
            An EXTERNAL_DATA_TRAINING activity when external datasets are
            registered, otherwise None.
        """
        try:
            from vetinari.training.external_data import ExternalDataManager

            manager = ExternalDataManager()
            datasets = manager.get_available_datasets()
        except ImportError:
            logger.debug("ExternalDataManager not available; skipping external-data candidate")
            return None
        except Exception:
            logger.warning("Failed to list external datasets; skipping", exc_info=True)
            return None

        if not datasets:
            return None

        dataset_names = [getattr(d, "name", "unknown") for d in datasets] if isinstance(datasets, list) else []

        return TrainingActivity(
            type=TrainingActivityType.EXTERNAL_DATA_TRAINING,
            description=f"Train on {len(dataset_names)} external dataset(s): {', '.join(dataset_names[:3])}",
            hypothesis=(
                "Incorporating curated external data will broaden generalisation "
                "without degrading task-specific performance."
            ),
            metric="general_benchmark_score",
            baseline=0.0,
            target=0.75,
            rollback_plan="Revert adapter; re-run held-out evaluation set to confirm regression.",
            estimated_duration_minutes=240,
            estimated_vram_gb=18.0,
            priority=0.6,
            metadata={"dataset_names": dataset_names, "dataset_count": len(dataset_names)},
        )

    def _candidate_prompt_evolution(self) -> TrainingActivity | None:
        """Return a prompt evolution activity if A/B tests are pending.

        Returns:
            A PROMPT_EVOLUTION activity when the prompt evolver has
            unresolved A/B tests, otherwise None.
        """
        try:
            from vetinari.learning.prompt_evolver import PromptEvolver

            evolver = PromptEvolver()
            pending = evolver.get_pending_ab_tests()
        except ImportError:
            logger.debug("prompt_evolver not available; skipping prompt-evolution candidate")
            return None
        except Exception:
            logger.warning("Failed to get pending A/B tests; skipping", exc_info=True)
            return None

        if not pending:
            return None

        return TrainingActivity(
            type=TrainingActivityType.PROMPT_EVOLUTION,
            description=f"Resolve {len(pending)} pending prompt A/B test(s)",
            hypothesis=(
                "Running prompt evolution trials on pending A/B tests will identify "
                "higher-performing prompt variants and improve output quality."
            ),
            metric="prompt_win_rate",
            baseline=0.5,
            target=0.65,
            rollback_plan="Revert to control prompt variants; A/B framework records all candidates.",
            estimated_duration_minutes=60,
            estimated_vram_gb=8.0,
            priority=0.5,
            metadata={"pending_test_count": len(pending)},
        )

    def _candidate_benchmark_practice(self) -> TrainingActivity | None:
        """Return a benchmark practice activity if benchmarks are stale.

        Returns:
            A BENCHMARK_PRACTICE activity when the last benchmark run
            exceeds the staleness threshold, otherwise None.
        """
        try:
            from vetinari.training.pipeline import BenchmarkTracker

            tracker = BenchmarkTracker()
            staleness_days = tracker.days_since_last_run()
        except ImportError:
            logger.debug("BenchmarkTracker not available; skipping benchmark-practice candidate")
            return None
        except Exception:
            logger.warning("Failed to check benchmark staleness; skipping", exc_info=True)
            return None

        if staleness_days < BENCHMARK_STALENESS_DAYS:
            return None

        return TrainingActivity(
            type=TrainingActivityType.BENCHMARK_PRACTICE,
            description=f"Run benchmark suite (last run {staleness_days} days ago)",
            hypothesis=(
                "Regular benchmark practice surfaces capability drift early and "
                "provides fresh baseline data for subsequent training decisions."
            ),
            metric="benchmark_suite_score",
            baseline=0.0,
            target=0.0,  # benchmarks measure rather than improve
            rollback_plan="No model changes; benchmark data is append-only.",
            estimated_duration_minutes=45,
            estimated_vram_gb=6.0,
            priority=0.4,
            metadata={"staleness_days": staleness_days},
        )

    def _candidate_distillation(self) -> TrainingActivity | None:
        """Return a distillation activity if quality cloud outputs are available.

        Returns:
            A DISTILLATION activity when cloud model outputs meet the
            quality threshold, otherwise None.
        """
        try:
            from vetinari.training.pipeline import CloudOutputStore

            store = CloudOutputStore()
            outputs = store.get_high_quality_outputs(min_score=DISTILLATION_QUALITY_THRESHOLD)
        except ImportError:
            logger.debug("CloudOutputStore not available; skipping distillation candidate")
            return None
        except Exception:
            logger.warning("Failed to get cloud outputs; skipping", exc_info=True)
            return None

        if not outputs:
            return None

        return TrainingActivity(
            type=TrainingActivityType.DISTILLATION,
            description=f"Distill from {len(outputs)} high-quality cloud model output(s)",
            hypothesis=(
                f"Knowledge distillation from cloud outputs (score ≥ {DISTILLATION_QUALITY_THRESHOLD}) "
                "will transfer strong reasoning capabilities to the local model."
            ),
            metric="distillation_eval_score",
            baseline=0.0,
            target=DISTILLATION_QUALITY_THRESHOLD,
            rollback_plan="Revert adapter; verify local benchmark does not degrade vs. pre-distillation.",
            estimated_duration_minutes=150,
            estimated_vram_gb=20.0,
            priority=0.65,
            metadata={"output_count": len(outputs)},
        )

    def _candidate_rlef(self) -> TrainingActivity | None:
        """Return an RLEF code-execution activity if enough traces exist.

        Reinforcement Learning from Execution Feedback uses sandbox
        code-execution outcomes (pass/fail, runtime errors, test results) as
        a reward signal to fine-tune the model on coding tasks.

        Returns:
            An RLEF_CODE_EXECUTION activity when execution traces meet the
            minimum threshold, otherwise None.
        """
        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            trace_count = collector.count_execution_traces()
        except ImportError:
            logger.debug("training_data collector not available; skipping RLEF candidate")
            return None
        except AttributeError:
            logger.debug("count_execution_traces not implemented; skipping RLEF candidate")
            return None
        except Exception:
            logger.warning("Failed to count execution traces; skipping", exc_info=True)
            return None

        if trace_count < MIN_RLEF_TRACES:
            return None

        return TrainingActivity(
            type=TrainingActivityType.RLEF_CODE_EXECUTION,
            description=f"RLEF training on {trace_count} code-execution traces",
            hypothesis=(
                "Using sandbox execution outcomes (pass/fail, runtime errors) "
                "as reward signal will improve code generation correctness."
            ),
            metric="code_execution_pass_rate",
            baseline=0.0,
            target=0.85,
            rollback_plan="Revert adapter; compare pass@1 on held-out coding problems.",
            estimated_duration_minutes=150,
            estimated_vram_gb=18.0,
            priority=0.75,
            metadata={"trace_count": trace_count},
        )

    def _default_activity(self) -> TrainingActivity:
        """Return a calibration benchmark activity when no candidates apply.

        Returns:
            A BENCHMARK_PRACTICE activity that establishes calibration baselines.
        """
        return TrainingActivity(
            type=TrainingActivityType.BENCHMARK_PRACTICE,
            description="Run calibration benchmarks to establish performance baselines",
            hypothesis=(
                "Running the full calibration benchmark suite will establish baseline "
                "metrics needed to identify future training priorities."
            ),
            metric="benchmark_suite_score",
            baseline=0.0,
            target=0.0,  # calibration measures rather than improves
            rollback_plan="No model changes made; safe to run at any time.",
            estimated_duration_minutes=DEFAULT_DURATION_MINUTES,
            estimated_vram_gb=DEFAULT_VRAM_GB,
            priority=0.3,
            metadata={"reason": "no_higher_priority_candidates"},
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def next_activity(self) -> TrainingActivity:
        """Determine the highest-priority training activity right now.

        Builds a candidate list from all available signal sources, sorts by
        priority, and returns the top candidate. Falls back to a calibration
        benchmark if no signals are actionable.

        Returns:
            The TrainingActivity with the highest priority score.
        """
        candidates: list[TrainingActivity] = []

        for builder in (
            self._candidate_weak_skill,
            self._candidate_defect_pattern,
            self._candidate_self_play,
            self._candidate_external_data,
            self._candidate_prompt_evolution,
            self._candidate_benchmark_practice,
            self._candidate_distillation,
            self._candidate_rlef,
        ):
            try:
                activity = builder()
                if activity is not None:
                    candidates.append(activity)
            except Exception:
                logger.warning("Candidate builder %s raised unexpectedly; skipping", builder.__name__, exc_info=True)

        self._candidates = candidates

        if not candidates:
            logger.info("No training candidates found; returning default calibration activity")
            return self._default_activity()

        candidates.sort(key=lambda a: a.priority, reverse=True)
        chosen = candidates[0]
        logger.info(
            "next_activity selected: type=%s priority=%.2f description=%s",
            chosen.type.value,
            chosen.priority,
            chosen.description,
        )
        return chosen

    def get_phase(self) -> CurriculumPhase:
        """Return the current curriculum phase based on training history.

        Phases:
            - CALIBRATION: fewer than 50 execution records exist
            - TARGETED_SKILL_BUILDING: records ≥ 50 and weakest skill < 0.7
            - CONTINUOUS_IMPROVEMENT: records ≥ 50 and all skills at threshold

        Returns:
            The current CurriculumPhase enum value.
        """
        record_count = 0
        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            record_count = collector.count_records()
        except ImportError:
            logger.debug("training_data collector not available; defaulting to CALIBRATION phase")
            return CurriculumPhase.CALIBRATION
        except Exception:
            logger.warning("Failed to count training records; defaulting to CALIBRATION", exc_info=True)
            return CurriculumPhase.CALIBRATION

        if record_count < 50:
            return CurriculumPhase.CALIBRATION

        # Check whether there is still a weak skill to address
        weak_skill = self._candidate_weak_skill()
        if weak_skill is not None:
            return CurriculumPhase.TARGETED_SKILL_BUILDING

        return CurriculumPhase.CONTINUOUS_IMPROVEMENT

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of the current curriculum state.

        Returns:
            Dictionary with keys:
                - ``phase``: string name of the current CurriculumPhase
                - ``candidate_count``: number of actionable candidates last computed
                - ``next_activity_description``: description of the top candidate
        """
        phase = self.get_phase()
        next_act = self.next_activity()
        return {
            "phase": phase.value,
            "candidate_count": len(self._candidates),
            "next_activity_description": next_act.description,
        }

    def run_activity(self, description: str, job_id: str | None = None) -> None:
        """Execute a training activity selected by the curriculum.

        Re-evaluates :meth:`next_activity` to get the full
        :class:`TrainingActivity` object, then delegates to the appropriate
        pipeline stage based on the activity type.

        Args:
            description: Human-readable description of the activity (used for
                logging; the actual activity is determined by next_activity).
            job_id: Optional job identifier for tracking.
        """
        activity = self.next_activity()
        logger.info(
            "run_activity: job=%s type=%s description=%s",
            job_id,
            activity.type.value,
            activity.description,
        )

        if activity.type in (
            TrainingActivityType.FINE_TUNE_WEAK_SKILL,
            TrainingActivityType.EXTERNAL_DATA_TRAINING,
            TrainingActivityType.RLEF_CODE_EXECUTION,
            TrainingActivityType.SELF_PLAY_REASONING,
        ):
            self._run_fine_tune(activity)
        elif activity.type == TrainingActivityType.DISTILLATION:
            self._run_distillation(activity)
        elif activity.type in (
            TrainingActivityType.BENCHMARK_PRACTICE,
            TrainingActivityType.PROMPT_EVOLUTION,
        ):
            logger.info(
                "run_activity: non-training activity %s — no model changes",
                activity.type.value,
            )
            if activity.type == TrainingActivityType.BENCHMARK_PRACTICE:
                # Record the run timestamp so the curriculum can detect staleness
                # on the next scheduling decision.
                try:
                    from vetinari.training.pipeline import BenchmarkTracker

                    BenchmarkTracker().record_run()
                    logger.info("run_activity: recorded benchmark run timestamp")
                except ImportError:
                    logger.debug("BenchmarkTracker not available — run timestamp not recorded")
                except Exception:
                    logger.warning(
                        "Could not record benchmark run timestamp — staleness detection may re-trigger too soon",
                        exc_info=True,
                    )
        else:
            logger.warning("run_activity: unknown activity type %s; skipping", activity.type.value)

    def _run_fine_tune(self, activity: TrainingActivity) -> None:
        """Execute a QLoRA fine-tuning run via the training pipeline.

        Args:
            activity: The training activity with metadata including task_type.
        """
        try:
            from vetinari.training.pipeline import TrainingPipeline

            pipeline = TrainingPipeline()
            reqs = pipeline.check_requirements()
            if not reqs.get("ready_for_training", False):
                logger.warning("_run_fine_tune: training libraries not installed; skipping")
                return

            task_type = activity.metadata.get("task_type")
            run = pipeline.run(
                base_model="auto",
                task_type=task_type,
                min_score=0.8,
            )
            logger.info(
                "_run_fine_tune: completed — success=%s output=%s",
                run.success,
                run.output_model_path,
            )
        except ImportError:
            logger.warning("_run_fine_tune: training pipeline not available")
        except Exception:
            logger.exception("_run_fine_tune: unexpected error during training")

    def _run_distillation(self, activity: TrainingActivity) -> None:
        """Run knowledge distillation from high-quality outputs.

        Args:
            activity: The training activity with distillation metadata.
        """
        try:
            import tempfile
            from pathlib import Path

            from vetinari.training.pipeline import (
                ContextDistillationDatasetBuilder,
                TrainingPipeline,
            )

            builder = ContextDistillationDatasetBuilder()
            _distill_dir = Path(tempfile.mkdtemp(prefix="vetinari_distill_"))
            dataset_info = builder.build_dataset(output_path=str(_distill_dir / "distillation.jsonl"))
            if not dataset_info or dataset_info.num_examples == 0:
                logger.info("_run_distillation: no distillation data available; skipping")
                return

            pipeline = TrainingPipeline()
            reqs = pipeline.check_requirements()
            if not reqs.get("ready_for_training", False):
                logger.warning("_run_distillation: training libraries not installed; skipping")
                return

            # Pass the curated dataset path so the pipeline uses the distillation
            # data rather than re-curating from scratch (defect 7 fix).
            run = pipeline.run(base_model="auto", min_score=0.85, output_base_dir=str(Path(dataset_info.output_path).parent))
            logger.info(
                "_run_distillation: completed — success=%s output=%s",
                run.success,
                run.output_model_path,
            )
        except ImportError:
            logger.warning("_run_distillation: pipeline modules not available")
        except Exception:
            logger.exception("_run_distillation: unexpected error during distillation")

        # After distillation pipeline, extract abstract strategies from successful
        # episodes and write them back as synthetic episodes for future planning.
        try:
            from vetinari.training.synthetic_data import StrategyDistiller

            distiller = StrategyDistiller()
            strategies = distiller.distill_strategies(min_score=0.8)
            if strategies:
                stored = distiller.store_strategies(strategies)
                logger.info(
                    "_run_distillation: stored %d/%d distilled strategies as synthetic episodes",
                    stored,
                    len(strategies),
                )
        except ImportError:
            logger.debug("StrategyDistiller not available — strategy extraction skipped")
        except Exception:
            logger.warning(
                "Could not run strategy distillation after training — strategies will not be stored",
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Module-level get_training_curriculum singleton
# ---------------------------------------------------------------------------
# Exposes the canonical TrainingCurriculum instance so all callers share one
# curriculum state — avoids redundant candidate computation on every request.

_curriculum_instance: TrainingCurriculum | None = None
_curriculum_instance_lock: threading.Lock = threading.Lock()


def get_training_curriculum() -> TrainingCurriculum:
    """Return the canonical TrainingCurriculum singleton.

    Uses double-checked locking so the first call creates the instance and
    all subsequent calls return the same object with no lock contention.

    Returns:
        The shared TrainingCurriculum instance.
    """
    global _curriculum_instance
    if _curriculum_instance is not None:
        return _curriculum_instance
    with _curriculum_instance_lock:
        if _curriculum_instance is not None:
            return _curriculum_instance
        _curriculum_instance = TrainingCurriculum()
    logger.debug("get_training_curriculum: created new singleton")
    return _curriculum_instance
