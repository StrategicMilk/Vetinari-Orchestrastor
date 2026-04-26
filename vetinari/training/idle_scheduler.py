"""Idle detection and training scheduler for Vetinari.

Responsible for detecting when the system has no active user requests and
orchestrating idle-time training activities such as model fine-tuning and
curriculum-driven self-improvement cycles.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from vetinari.types import StatusEnum

logger = logging.getLogger(__name__)

# Polling interval for the background scheduler loop
POLL_INTERVAL_SECONDS = 60

# Minimum free VRAM required before starting a training cycle (in GB)
MIN_FREE_VRAM_GB = 8.0

# Minimum number of training records required before starting a training cycle
MIN_TRAINING_RECORDS = 100


@dataclass
class IdleTrainingJob:
    """Represents a single idle-time training job.

    Attributes:
        job_id: Unique identifier for the job.
        status: Current status — one of "pending", "running", "paused", "completed", "failed".
        activity_description: Human-readable description of the training activity.
        started_at: ISO-8601 UTC timestamp when the job started.
        progress: Fraction complete in [0.0, 1.0].
    """

    job_id: str
    status: str
    activity_description: str
    started_at: str
    progress: float = 0.0

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"TrainingJob(job_id={self.job_id!r}, status={self.status!r}, progress={self.progress!r})"


# Alias for backward compatibility
TrainingJob = IdleTrainingJob


class IdleDetector:
    """Detects when Vetinari has no active user requests.

    Records activity timestamps and exposes an ``idle`` property that
    returns ``True`` once no activity has been observed for at least
    ``min_idle_minutes``.  All mutable state is protected by a
    :class:`threading.Lock` so the detector is safe to call from multiple
    threads simultaneously.
    """

    def __init__(self, min_idle_minutes: int = 5) -> None:
        """Initialise the detector.

        Args:
            min_idle_minutes: Number of minutes of inactivity before the
                system is considered idle.  Must be a positive integer.
        """
        self._min_idle_minutes: int = min_idle_minutes
        self._last_activity: datetime = datetime.now(timezone.utc)
        self._was_idle: bool = False
        self._lock: threading.Lock = threading.Lock()

    def record_activity(self) -> None:
        """Record that user or agent activity just occurred.

        Updates the internal timestamp.  If the system was previously idle,
        logs the transition back to active so operators can observe the
        lifecycle.
        """
        with self._lock:
            was_idle = self._was_idle
            self._last_activity = datetime.now(timezone.utc)
            self._was_idle = False

        if was_idle:
            logger.info("IdleDetector: system transitioned from idle to active")

    @property
    def idle(self) -> bool:
        """Whether the system is currently idle.

        Returns:
            ``True`` if no activity has been recorded for at least
            ``min_idle_minutes``, ``False`` otherwise.
        """
        with self._lock:
            elapsed = (datetime.now(timezone.utc) - self._last_activity).total_seconds()
            is_idle = elapsed >= self._min_idle_minutes * 60
            if is_idle and not self._was_idle:
                self._was_idle = True
                logger.info(
                    "IdleDetector: system became idle after %.1f minutes of inactivity",
                    elapsed / 60,
                )
            return is_idle

    @property
    def idle_duration_minutes(self) -> float:
        """How long the system has been idle, in minutes.

        Returns:
            Minutes since the last recorded activity.  Returns ``0.0`` if
            the system is not currently idle.
        """
        with self._lock:
            elapsed_seconds = (datetime.now(timezone.utc) - self._last_activity).total_seconds()
            is_idle = elapsed_seconds >= self._min_idle_minutes * 60
            if not is_idle:
                return 0.0
            idle_seconds = max(0.0, elapsed_seconds - self._min_idle_minutes * 60)
            return idle_seconds / 60


class TrainingScheduler:
    """Orchestrates idle-time training activities.

    Polls the system every :data:`POLL_INTERVAL_SECONDS` seconds.  When
    the :class:`IdleDetector` reports the system is idle and all
    preconditions pass, a training cycle is started.  User requests
    pre-empt training: call :meth:`pause_for_user_request` on incoming
    requests and :meth:`resume_after_user_request` when they finish.

    Example::

        detector = IdleDetector(min_idle_minutes=5)
        scheduler = TrainingScheduler(idle_detector=detector)
        scheduler.start()
        # …later…
        scheduler.stop()
    """

    def __init__(
        self,
        idle_detector: IdleDetector,
        vram_manager: object | None = None,
    ) -> None:
        """Initialise the scheduler.

        Args:
            idle_detector: Shared :class:`IdleDetector` instance used to
                decide when training is permitted.
            vram_manager: Optional VRAM manager object.  When provided, its
                ``free_vram_gb`` attribute (or ``get_free_vram_gb()`` method)
                is queried before starting a cycle.  When ``None`` the VRAM
                check is skipped.
        """
        self._idle_detector: IdleDetector = idle_detector
        self._vram_manager: object | None = vram_manager

        self._shutdown_event: threading.Event = threading.Event()
        self._paused: bool = False
        self._lock: threading.Lock = threading.Lock()

        self._current_job: IdleTrainingJob | None = None
        self._thread: threading.Thread | None = None

        # History of all manually triggered and idle-time training jobs.
        # Each entry is a dict with job_id, activity_description, started_at.
        self._history: list[dict] = []

    # ── Public control API ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler daemon thread.

        The thread polls every :data:`POLL_INTERVAL_SECONDS` seconds and
        triggers training cycles when conditions are met.  Safe to call
        once; subsequent calls are no-ops if already running.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("TrainingScheduler.start() called but thread is already running")
            return

        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="training-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("TrainingScheduler started (poll interval=%ds)", POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        """Signal the scheduler to shut down and wait for it to finish.

        Blocks until the background thread exits.  Safe to call even if
        the scheduler was never started.
        """
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join()
            logger.info("TrainingScheduler stopped")
        self._thread = None

    def pause_for_user_request(self) -> None:
        """Gracefully pause any current training job for an incoming request.

        Sets the internal paused flag and transitions a running job to
        "paused" status so the run loop can skip new training cycles while
        the user request is active.
        """
        with self._lock:
            self._paused = True
            if self._current_job is not None and self._current_job.status == StatusEnum.RUNNING.value:
                self._current_job.status = "paused"
                logger.info(
                    "TrainingScheduler: paused job %s for user request",
                    self._current_job.job_id,
                )

    def resume_after_user_request(self) -> None:
        """Resume a paused training job if the system is still idle.

        Only resumes if the :class:`IdleDetector` still reports idle;
        otherwise the paused job remains paused until the next poll cycle.
        """
        with self._lock:
            self._paused = False

        if not self._idle_detector.idle:
            logger.debug(
                "TrainingScheduler.resume_after_user_request: system no longer idle, not resuming",
            )
            return

        with self._lock:
            if self._current_job is not None and self._current_job.status == "paused":
                self._current_job.status = "running"
                logger.info(
                    "TrainingScheduler: resumed job %s after user request",
                    self._current_job.job_id,
                )

    def start_manual_cycle(
        self,
        activity_description: str = "Manual training cycle",
    ) -> str:
        """Trigger a manual training cycle immediately, bypassing idle detection.

        Creates a :class:`TrainingJob` with a ``"manual-"`` prefix, records it
        in :attr:`_history`, and runs :meth:`_execute_training_cycle` in a
        daemon thread so the caller is not blocked.

        If a training job is already running, returns the sentinel string
        ``"already_running"`` and does NOT append to history.

        Args:
            activity_description: Human-readable description of the activity to
                run.  Defaults to ``"Manual training cycle"`` when omitted.

        Returns:
            A ``"manual-"``-prefixed hex job ID on success, or
            ``"already_running"`` when a job is already in flight.
        """
        with self._lock:
            if self._current_job is not None:
                logger.info(
                    "start_manual_cycle: job %s already running — ignoring request",
                    self._current_job.job_id,
                )
                return "already_running"

        job_id = "manual-" + uuid.uuid4().hex
        job = IdleTrainingJob(
            job_id=job_id,
            status="running",
            activity_description=activity_description,
            started_at=datetime.now(timezone.utc).isoformat(),
            progress=0.0,
        )

        with self._lock:
            self._current_job = job
            self._history.append({
                "job_id": job_id,
                "activity_description": activity_description,
                "started_at": job.started_at,
            })

        logger.info(
            "start_manual_cycle: started job=%s activity=%r",
            job_id,
            activity_description,
        )

        thread = threading.Thread(
            target=self._execute_training_cycle,
            name=f"manual-training-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return job_id

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def current_job(self) -> IdleTrainingJob | None:
        """The currently active :class:`TrainingJob`, or ``None``.

        Returns:
            The active job dataclass or ``None`` if no job is in flight.
        """
        with self._lock:
            return self._current_job

    @property
    def is_training(self) -> bool:
        """Whether a training job is currently running.

        Returns:
            ``True`` if a job exists and its status is ``"running"``.
        """
        with self._lock:
            return self._current_job is not None and self._current_job.status == StatusEnum.RUNNING.value

    # ── Internal loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main background loop that drives training cycles.

        Polls every :data:`POLL_INTERVAL_SECONDS` seconds.  On each tick:

        1. Check whether the system is idle.
        2. Check whether the scheduler is paused.
        3. Check whether a job is already running.
        4. Verify all preconditions via :meth:`_can_train`.
        5. Execute a training cycle via :meth:`_execute_training_cycle`.
        """
        while not self._shutdown_event.is_set():
            try:
                self._shutdown_event.wait(timeout=POLL_INTERVAL_SECONDS)
                if self._shutdown_event.is_set():
                    break

                with self._lock:
                    paused = self._paused
                    already_running = (
                        self._current_job is not None and self._current_job.status == StatusEnum.RUNNING.value
                    )

                if not self._idle_detector.idle:
                    logger.debug("TrainingScheduler: system is not idle, skipping cycle")
                    continue

                if paused:
                    logger.debug("TrainingScheduler: scheduler is paused, skipping cycle")
                    continue

                if already_running:
                    logger.debug("TrainingScheduler: job already running, skipping cycle")
                    continue

                if not self._can_train():
                    logger.info("TrainingScheduler: preconditions not met, skipping cycle")
                    continue

                self._execute_training_cycle()

            except Exception:
                logger.exception("TrainingScheduler._run_loop: unexpected error during cycle")

    def _can_train(self) -> bool:
        """Check whether all preconditions for a training cycle are satisfied.

        Evaluates:
        - Sufficient free VRAM (≥ :data:`MIN_FREE_VRAM_GB` GB) if a
          VRAMManager is present.
        - No training job already running.
        - Sufficient training records (≥ :data:`MIN_TRAINING_RECORDS`).

        Logs the specific reason for each failing check so operators can
        diagnose why training is not starting.

        Returns:
            ``True`` if all checks pass, ``False`` otherwise.
        """
        can = True

        # ── VRAM check ──────────────────────────────────────────────────────
        if self._vram_manager is not None:
            try:
                free_gb: float
                if hasattr(self._vram_manager, "get_free_vram_gb"):
                    free_gb = float(self._vram_manager.get_free_vram_gb())
                elif hasattr(self._vram_manager, "free_vram_gb"):
                    free_gb = float(self._vram_manager.free_vram_gb)
                else:
                    logger.warning(
                        "_can_train: VRAMManager has no known VRAM attribute, skipping VRAM check",
                    )
                    free_gb = MIN_FREE_VRAM_GB  # treat as sufficient

                if free_gb < MIN_FREE_VRAM_GB:
                    logger.info(
                        "_can_train: insufficient free VRAM (%.1f GB available, need %.1f GB)",
                        free_gb,
                        MIN_FREE_VRAM_GB,
                    )
                    can = False
            except Exception:
                logger.exception("_can_train: error querying VRAMManager, skipping VRAM check")

        # ── Active job check ────────────────────────────────────────────────
        with self._lock:
            if self._current_job is not None and self._current_job.status == StatusEnum.RUNNING.value:
                logger.info("_can_train: a training job is already running (%s)", self._current_job.job_id)
                can = False

        # ── Training data check ─────────────────────────────────────────────
        record_count = self._count_training_records()
        if record_count < MIN_TRAINING_RECORDS:
            logger.info(
                "_can_train: insufficient training records (%d available, need %d)",
                record_count,
                MIN_TRAINING_RECORDS,
            )
            can = False

        # ── Model availability check ──────────────────────────────────────────
        try:
            from vetinari.training.pipeline import TrainingPipeline

            pipeline = TrainingPipeline()
            resolved = pipeline._resolve_base_model("auto")
            if not resolved or resolved == "auto":
                logger.info("_can_train: no model available for training")
                can = False
        except Exception:
            logger.warning("_can_train: model availability check skipped")

        return can

    def _count_training_records(self) -> int:
        """Return the number of available training records.

        Attempts to query the training pipeline.  Falls back to 0 on any
        error so that a missing or broken pipeline never prevents the
        scheduler from making a decision (it will simply skip the cycle).

        Returns:
            Count of available training records, or 0 on failure.
        """
        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            stats = collector.get_stats()
            return stats.get("total", 0)
        except ImportError:
            logger.debug("_count_training_records: training_data module not available")
            return 0
        except Exception:
            logger.warning("_count_training_records: could not count records", exc_info=True)
            return 0

    def _execute_training_cycle(self) -> None:
        """Execute one idle-time training cycle.

        Retrieves the next activity from the curriculum (if the curriculum
        module is available) and runs it.  Creates a :class:`TrainingJob`
        to track progress.  Logs a warning and returns gracefully if the
        curriculum module is not yet available.
        """
        activity_description = self._get_next_curriculum_activity()
        if activity_description is None:
            # Check if the MetaOptimizer detected collapse risk before simply skipping.
            # COLLAPSE_RISK means training is actively hurting quality — halt the scheduler
            # to prevent further degradation until a human intervenes.
            try:
                from vetinari.learning.meta_optimizer import LearningPhase, MetaOptimizer

                phase = MetaOptimizer().detect_phase()
                if phase == LearningPhase.COLLAPSE_RISK:
                    logger.warning(
                        "_execute_training_cycle: MetaOptimizer detected COLLAPSE_RISK"
                        " — halting scheduler to prevent further quality degradation"
                    )
                    self._shutdown_event.set()
                    return
            except Exception:
                logger.warning(
                    "_execute_training_cycle: could not check MetaOptimizer phase"
                    " — proceeding with normal skip (scheduler continues)"
                )
            logger.info("_execute_training_cycle: no curriculum activity available, skipping")
            return

        job = IdleTrainingJob(
            job_id=uuid.uuid4().hex,
            status="running",
            activity_description=activity_description,
            started_at=datetime.now(timezone.utc).isoformat(),
            progress=0.0,
        )

        with self._lock:
            self._current_job = job

        logger.info(
            "TrainingScheduler: starting training cycle job=%s activity=%r",
            job.job_id,
            job.activity_description,
        )

        try:
            self._run_activity(job)
        except Exception:
            logger.exception(
                "TrainingScheduler: training cycle failed for job=%s",
                job.job_id,
            )
            with self._lock:
                if self._current_job is not None and self._current_job.job_id == job.job_id:
                    self._current_job.status = "failed"
        else:
            with self._lock:
                if (
                    self._current_job is not None
                    and self._current_job.job_id == job.job_id
                    and self._current_job.status == StatusEnum.RUNNING.value
                ):
                    self._current_job.status = "completed"
                    self._current_job.progress = 1.0
            logger.info("TrainingScheduler: completed training cycle job=%s", job.job_id)

            # Notify the meta-optimizer that a training cycle completed so it
            # can track per-strategy ROI and detect saturation/collapse phases.
            try:
                from vetinari.learning.meta_optimizer import MetaOptimizer

                MetaOptimizer().record_cycle(
                    strategy_name="training",
                    quality_gain=0.0,  # actual gain unknown at this level; collector provides detail
                    success=True,
                )
            except ImportError:
                logger.debug("MetaOptimizer not available — skipping cycle record")
            except Exception:
                logger.warning(
                    "Could not record training cycle in MetaOptimizer — ROI tracking will be incomplete",
                    exc_info=True,
                )

            # Reinforce the current best config with a positive quality signal now
            # that a training cycle completed successfully.  Uses a conservative
            # baseline score (0.7) because actual quality gain is unknown at this
            # level — the archive's EMA smooths out this uncertainty over time.
            try:
                from vetinari.learning.improvement_archive import get_improvement_archive
                from vetinari.types import AgentType

                _archive = get_improvement_archive()
                for _agent_type in AgentType:
                    _top = _archive.get_best_configs(_agent_type.value, limit=1)
                    if _top:
                        _archive.update_score(_top[0].config_id, 0.7)
            except ImportError:
                logger.debug("ImprovementArchive not available — config score update skipped")
            except Exception:
                logger.warning(
                    "Could not update config scores in ImprovementArchive — archive quality signals will be stale",
                    exc_info=True,
                )

        # Run memory consolidation as housekeeping during idle time
        self._consolidate_memory()

    def _consolidate_memory(self) -> None:
        """Run memory consolidation during idle time.

        Performs four housekeeping operations in order:
        1. Session-to-long-term promotion (existing ``consolidate()``)
        2. Ebbinghaus decay pruning — remove memories below strength threshold
        3. Episode-to-semantic promotion — extract patterns from recurring episodes
        4. Contradiction flagging — log entries with conflicting relationship types

        Also prunes stale improvement archive stepping-stones and old plan
        records to keep storage bounded during long-running sessions.
        """
        try:
            from vetinari.memory.unified import get_unified_store

            store = get_unified_store()

            # 1. Session-to-long-term promotion
            promoted = store.consolidate()
            if promoted > 0:
                logger.info(
                    "TrainingScheduler: memory consolidation promoted %d entries to long-term storage",
                    promoted,
                )

            # 2. Ebbinghaus decay pruning — remove weak memories
            self._prune_weak_memories(store)

            # 3. Episode-to-semantic promotion
            try:
                pattern_count = store.promote_episodes_to_semantic()
                if pattern_count > 0:
                    logger.info(
                        "TrainingScheduler: promoted %d episode groups to semantic patterns",
                        pattern_count,
                    )
            except Exception:
                logger.warning(
                    "Episode-to-semantic promotion failed — patterns will not be extracted this cycle",
                    exc_info=True,
                )

            # 4. Flag contradictions
            self._flag_contradictions(store)

        except ImportError:
            logger.debug("Memory consolidation skipped — unified memory store not available")
        except Exception:
            logger.warning("Memory consolidation failed during idle cycle — will retry next cycle", exc_info=True)

        # Prune improvement archive stepping-stones for all known agent types
        # so the archive stays bounded and stale configs don't accumulate.
        try:
            from vetinari.learning.improvement_archive import ImprovementArchive
            from vetinari.types import AgentType

            archive = ImprovementArchive()
            for agent_type in AgentType:
                pruned = archive.prune_stepping_stones(agent_type.value)
                if pruned > 0:
                    logger.info(
                        "TrainingScheduler: pruned %d stepping-stone configs for agent=%s",
                        pruned,
                        agent_type.value,
                    )
        except ImportError:
            logger.debug("ImprovementArchive not available — stepping-stone pruning skipped")
        except Exception:
            logger.warning(
                "Could not prune improvement archive stepping-stones — archive may grow unbounded",
                exc_info=True,
            )

        # Prune plan records older than the default retention window.
        try:
            from vetinari.memory.plan_tracking import MemoryStore

            deleted = MemoryStore().prune_old_plans()
            if deleted > 0:
                logger.info("TrainingScheduler: pruned %d old plan records from memory store", deleted)
        except ImportError:
            logger.debug("MemoryStore not available — plan pruning skipped")
        except Exception:
            logger.warning(
                "Could not prune old plan records from memory store — storage may grow unbounded",
                exc_info=True,
            )

    def _prune_weak_memories(self, store: Any) -> None:
        """Remove memories whose Ebbinghaus retention strength has decayed below threshold.

        Uses the decay model from memory_storage.ebbinghaus_strength to
        identify entries that are no longer worth retaining.
        """
        try:
            from vetinari.memory.memory_storage import PRUNE_THRESHOLD, ebbinghaus_strength

            with store._lock:
                rows = store._conn.execute(
                    "SELECT id, importance, timestamp, recall_count FROM memories WHERE forgotten = 0"
                ).fetchall()

            weak_ids: list[str] = []
            for row in rows:
                # timestamp is ISO or epoch-ms — normalise to ms
                ts = row["timestamp"]
                if isinstance(ts, str):
                    continue  # ISO timestamps handled by SQL eviction; skip here
                strength = ebbinghaus_strength(
                    importance=float(row["importance"] or 0.5),
                    created_ts_ms=int(ts),
                    recall_count=int(row["recall_count"] or 0),
                )
                if strength < PRUNE_THRESHOLD:
                    weak_ids.append(row["id"])

            for entry_id in weak_ids:
                store.forget(entry_id, reason="Ebbinghaus strength below prune threshold")

            if weak_ids:
                logger.info(
                    "TrainingScheduler: pruned %d weak memories (Ebbinghaus decay)",
                    len(weak_ids),
                )
        except ImportError:
            logger.debug("Ebbinghaus pruning skipped — memory_storage not available")
        except Exception:
            logger.warning(
                "Ebbinghaus decay pruning failed — weak memories will persist until next cycle",
                exc_info=True,
            )

    def _flag_contradictions(self, store: Any) -> None:
        """Log memories linked by CONTRADICTS relationships for human review.

        Scans the relationship graph for contradiction edges and logs both
        sides so operators can resolve conflicting knowledge.
        """
        try:
            with store._lock:
                rows = store._conn.execute(
                    "SELECT id, supersedes_id, content FROM memories "
                    "WHERE relationship_type = 'contradicts' AND forgotten = 0"
                ).fetchall()

            for row in rows:
                logger.info(
                    "TrainingScheduler: contradiction detected — memory %s contradicts %s: %.100s",
                    row["id"],
                    row["supersedes_id"],
                    row["content"],
                )
        except Exception:
            logger.warning("Contradiction flagging skipped — column may not exist yet", exc_info=True)

    def _get_next_curriculum_activity(self) -> str | None:
        """Retrieve the next activity description from the curriculum module.

        Performs a late import of the curriculum module so that the
        scheduler degrades gracefully when the curriculum has not yet been
        implemented.

        Returns:
            Activity description string, or ``None`` if none is available.
        """
        try:
            from vetinari.training.curriculum import TrainingCurriculum  # late import

            curriculum = TrainingCurriculum()
            activity = curriculum.next_activity()
            if activity is None:
                logger.debug("_get_next_curriculum_activity: curriculum returned no activity")
                return self._fallback_activity_from_meta_optimizer()
            # Accept either a string or an object with a ``description`` attribute
            if isinstance(activity, str):
                return activity
            return str(getattr(activity, "description", activity))
        except ImportError:
            logger.warning(
                "_get_next_curriculum_activity: vetinari.training.curriculum not available yet",
            )
            return self._fallback_activity_from_meta_optimizer()
        except Exception:
            logger.exception("_get_next_curriculum_activity: error fetching curriculum activity")
            return None

    def _fallback_activity_from_meta_optimizer(self) -> str | None:
        """Ask the MetaOptimizer for the highest-ROI strategy when curriculum is unavailable.

        Returns:
            Activity description derived from the strategy suggestion, or None if
            the MetaOptimizer is unavailable or recommends halting.
        """
        try:
            from vetinari.learning.meta_optimizer import MetaOptimizer

            suggestion = MetaOptimizer().suggest_next_strategy()
            if suggestion is None:
                logger.warning(
                    "_fallback_activity_from_meta_optimizer: MetaOptimizer recommends halting — collapse risk detected"
                )
                return None
            logger.info("_fallback_activity_from_meta_optimizer: MetaOptimizer suggests strategy=%s", suggestion)
            return f"MetaOptimizer-suggested activity: {suggestion}"
        except ImportError:
            logger.debug("MetaOptimizer not available — no fallback activity")
            return None
        except Exception:
            logger.warning(
                "Could not get MetaOptimizer strategy suggestion — idle cycle will be skipped",
                exc_info=True,
            )
            return None

    def _run_activity(self, job: IdleTrainingJob) -> None:
        """Execute the training activity for the given job.

        Attempts to delegate to the curriculum module's ``run_activity``
        method.  Falls back to logging the activity description when the
        curriculum module is unavailable, ensuring the scheduler never
        crashes on a missing dependency.

        Args:
            job: The :class:`TrainingJob` describing the work to perform.
        """
        try:
            from vetinari.training.curriculum import TrainingCurriculum  # late import

            curriculum = TrainingCurriculum()
            if hasattr(curriculum, "run_activity"):
                curriculum.run_activity(job.activity_description, job_id=job.job_id)
                with self._lock:
                    if self._current_job is not None and self._current_job.job_id == job.job_id:
                        self._current_job.progress = 1.0
                return
        except ImportError:
            logger.warning(
                "_run_activity: vetinari.training.curriculum not available, logging activity only",
            )
        except Exception:
            logger.exception("_run_activity: error running curriculum activity, logging only")

        # Fallback: log the activity so the cycle is not silently lost
        logger.info(
            "_run_activity: [TRAINING CYCLE] job=%s activity=%r (curriculum module pending)",
            job.job_id,
            job.activity_description,
        )
        with self._lock:
            if self._current_job is not None and self._current_job.job_id == job.job_id:
                self._current_job.progress = 1.0


# ---------------------------------------------------------------------------
# Module-level scheduler registry
# ---------------------------------------------------------------------------
# The web layer (training_api.py) registers the scheduler singleton here once
# it is created, keeping the dependency arrow pointing inward:
#   web -> training  (correct)
# rather than the circular:
#   training -> web  (forbidden)

_registered_scheduler: TrainingScheduler | None = None
_registry_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# Module-level get_training_scheduler singleton
# ---------------------------------------------------------------------------
# Exposes the canonical TrainingScheduler instance for non-web callers that
# need to inspect or control the scheduler without going through the web layer.

_scheduler_instance: TrainingScheduler | None = None
_scheduler_instance_lock: threading.Lock = threading.Lock()


def get_training_scheduler() -> TrainingScheduler:
    """Return the canonical TrainingScheduler singleton.

    Creates an :class:`IdleDetector` and a :class:`TrainingScheduler` on the
    first call using double-checked locking.  Subsequent calls return the same
    instance so all callers share one scheduler.

    Returns:
        The shared :class:`TrainingScheduler` instance.
    """
    global _scheduler_instance
    if _scheduler_instance is not None:
        return _scheduler_instance
    with _scheduler_instance_lock:
        if _scheduler_instance is not None:
            return _scheduler_instance
        detector = IdleDetector()
        _scheduler_instance = TrainingScheduler(idle_detector=detector)
        register_scheduler(_scheduler_instance)
    logger.debug("get_training_scheduler: created new singleton")
    return _scheduler_instance


def register_scheduler(scheduler: TrainingScheduler) -> None:
    """Register the application's TrainingScheduler singleton.

    Called once by the web layer (vetinari.web.training_api) after it creates
    the shared scheduler.  Subsequent calls with the same instance are no-ops;
    calls with a different instance replace the registration and log a warning.

    Args:
        scheduler: The TrainingScheduler instance to register.
    """
    global _registered_scheduler
    with _registry_lock:
        if _registered_scheduler is scheduler:
            return
        if _registered_scheduler is not None:
            logger.warning("register_scheduler: replacing existing scheduler registration")
        _registered_scheduler = scheduler
    logger.debug("register_scheduler: scheduler registered")


def get_idle_detector() -> IdleDetector | None:
    """Return the IdleDetector from the registered TrainingScheduler.

    Returns ``None`` if no scheduler has been registered yet.  This is used
    by the Flask ``before_request`` hook to notify the detector of user
    activity so training does not fire while the user is actively working.

    Returns:
        The shared IdleDetector instance, or None.
    """
    with _registry_lock:
        scheduler = _registered_scheduler
    if scheduler is None:
        logger.debug("get_idle_detector: no scheduler registered yet")
        return None
    return scheduler._idle_detector
