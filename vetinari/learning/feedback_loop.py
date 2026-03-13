"""Feedback Loop - Vetinari Self-Improvement Subsystem.

Closes the learning loop: execution outcomes → model performance updates.
Feeds quality scores back into ModelPerformance table and DynamicModelRouter.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Closes the feedback loop between execution outcomes and model selection.

    After each task completes, this component:
    1. Fetches the quality score for the task
    2. Updates ModelPerformance in the memory store (running EMA)
    3. Updates the DynamicModelRouter performance cache
    4. Optionally updates the Ponder scoring system
    """

    EMA_ALPHA = 0.3  # Exponential moving average weight for recent performance

    def __init__(self):
        self._memory = None
        self._router = None

    def _get_memory(self):
        if self._memory is None:
            try:
                from vetinari.memory import get_memory_store

                self._memory = get_memory_store()
            except Exception:
                logger.debug("Failed to initialize memory store for feedback loop", exc_info=True)
        return self._memory

    def _get_router(self):
        if self._router is None:
            try:
                from vetinari.dynamic_model_router import get_model_router

                self._router = get_model_router()
            except Exception:
                logger.debug("Failed to initialize model router for feedback loop", exc_info=True)
        return self._router

    def record_outcome(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        quality_score: float,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        success: bool = True,
        benchmark_result: dict | None = None,
    ) -> None:
        """Record a task outcome and propagate it to all performance tracking systems.

        Args:
            task_id: The task identifier.
            model_id: The model used.
            task_type: Type of task (coding, research, etc.).
            quality_score: Overall quality score 0.0-1.0.
            latency_ms: Execution latency in milliseconds.
            cost_usd: Estimated cost in USD.
            success: Whether the task succeeded.
            benchmark_result: Optional benchmark data dict with keys:
                - pass_rate (float): 0.0-1.0 fraction of benchmark cases passed
                - task_type (str): benchmark task category
                - suite_name (str): benchmark suite identifier
                - n_trials (int): number of benchmark trials run
                - avg_score (float): average benchmark score
        """
        logger.debug(
            f"[FeedbackLoop] Recording outcome: task={task_id} model={model_id} "
            f"type={task_type} quality={quality_score:.2f} success={success}"
        )

        # 1. Update memory store ModelPerformance
        self._update_memory_performance(model_id, task_type, quality_score, latency_ms, success)

        # 2. Update DynamicModelRouter performance cache
        self._update_router_cache(model_id, task_type, quality_score, latency_ms, success)

        # 3. Update SubtaskMemory outcome with quality score
        self._update_subtask_quality(task_id, quality_score, success)

        # 4. Update Thompson Sampling arms
        self._update_thompson_arms(model_id, task_type, quality_score, success)

        # 5. If benchmark_result provided, feed it into model selection and learning
        if benchmark_result:
            self._process_benchmark_result(model_id, task_type, benchmark_result)

    def record_benchmark_outcome(
        self,
        model_id: str,
        benchmark_result: dict,
    ) -> None:
        """Record a standalone benchmark outcome (not tied to a specific task).

        Extracts pass_rate and task_type from the benchmark result and updates
        all performance tracking systems with benchmark-weighted signals.

        Args:
            model_id: The model that was benchmarked.
            benchmark_result: Dict with keys:
                - pass_rate (float): 0.0-1.0 fraction of cases passed
                - task_type (str): benchmark task category
                - suite_name (str): benchmark suite identifier
                - n_trials (int): number of benchmark trials run
                - avg_score (float): average benchmark score
        """
        task_type = benchmark_result.get("task_type", "general")
        pass_rate = benchmark_result.get("pass_rate", 0.0)
        avg_score = benchmark_result.get("avg_score", pass_rate)

        logger.debug(
            f"[FeedbackLoop] Recording benchmark outcome: model={model_id} "
            f"type={task_type} pass_rate={pass_rate:.2f} avg_score={avg_score:.2f}"
        )

        self._process_benchmark_result(model_id, task_type, benchmark_result)

    def _process_benchmark_result(self, model_id: str, task_type: str, benchmark_result: dict) -> None:
        """Process a benchmark result and propagate to all learning subsystems."""
        pass_rate = float(benchmark_result.get("pass_rate", 0.0))
        n_trials = int(benchmark_result.get("n_trials", 1))
        float(benchmark_result.get("avg_score", pass_rate))
        suite_name = benchmark_result.get("suite_name", "unknown")
        bench_task_type = benchmark_result.get("task_type", task_type)

        # Update Thompson Sampling with 3x weight (benchmarks are more reliable)
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            get_thompson_selector().update_from_benchmark(
                model_id=model_id,
                pass_rate=pass_rate,
                n_trials=n_trials,
                task_type=bench_task_type,
            )
        except Exception as e:
            logger.debug(f"Thompson benchmark update failed: {e}")

        # Update memory store with benchmark-informed performance
        try:
            mem = self._get_memory()
            if mem:
                existing = mem.get_model_performance(model_id, bench_task_type) or {}
                old_rate = existing.get("success_rate", 0.7)
                # Benchmark signals get higher EMA weight
                benchmark_ema = 0.5
                new_rate = (1 - benchmark_ema) * old_rate + benchmark_ema * pass_rate
                mem.update_model_performance(
                    model_id,
                    bench_task_type,
                    {
                        "success_rate": round(new_rate, 4),
                        "benchmark_pass_rate": round(pass_rate, 4),
                        "benchmark_suite": suite_name,
                        "total_uses": existing.get("total_uses", 0),
                    },
                )
        except Exception as e:
            logger.debug(f"Memory benchmark update failed: {e}")

    def _update_memory_performance(
        self, model_id: str, task_type: str, quality: float, latency_ms: int, success: bool
    ) -> None:
        """Update the ModelPerformance running averages in the memory store."""
        mem = self._get_memory()
        if mem is None:
            return
        try:
            existing = mem.get_model_performance(model_id, task_type) or {}
            old_rate = existing.get("success_rate", 0.7)
            old_latency = existing.get("avg_latency", latency_ms or 1000)
            old_uses = existing.get("total_uses", 0)

            # Exponential moving average update — blend success and quality into
            # a single signal to avoid double-counting
            signal = 0.5 * (1.0 if success else 0.0) + 0.5 * quality
            new_rate = (1 - self.EMA_ALPHA) * old_rate + self.EMA_ALPHA * signal
            new_latency = (1 - self.EMA_ALPHA) * old_latency + self.EMA_ALPHA * (latency_ms or old_latency)

            # Pass dict-form update (new signature)
            mem.update_model_performance(
                model_id,
                task_type,
                {
                    "success_rate": round(new_rate, 4),
                    "avg_latency": int(new_latency),
                    "total_uses": old_uses + 1,
                },
            )
        except Exception as e:
            logger.debug("Memory performance update failed: %s", e)

    def _update_router_cache(
        self, model_id: str, task_type: str, quality: float, latency_ms: int, success: bool
    ) -> None:
        """Update the DynamicModelRouter's in-memory performance cache."""
        router = self._get_router()
        if router is None:
            return
        try:
            cache_key = f"{model_id}:{task_type}"
            existing = router.get_performance_cache(cache_key)
            old_rate = existing.get("success_rate", 0.7)
            old_latency = existing.get("avg_latency_ms", latency_ms or 1000)

            signal = 0.5 * float(success) + 0.5 * quality
            new_rate = (1 - self.EMA_ALPHA) * old_rate + self.EMA_ALPHA * signal
            new_latency = (1 - self.EMA_ALPHA) * old_latency + self.EMA_ALPHA * (latency_ms or old_latency)

            router.update_performance_cache(
                cache_key,
                {
                    "success_rate": round(new_rate, 4),
                    "avg_latency_ms": int(new_latency),
                    "quality_score": round(quality, 4),
                    "last_updated": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.debug("Router cache update failed: %s", e)

    def _update_subtask_quality(self, task_id: str, quality: float, success: bool) -> None:
        """Annotate the SubtaskMemory record with quality score."""
        mem = self._get_memory()
        if mem is None:
            return
        try:
            mem.update_subtask_quality(task_id, quality_score=quality, succeeded=success)
        except Exception as e:
            logger.debug("Subtask quality update failed: %s", e)

    def _update_thompson_arms(self, model_id: str, task_type: str, quality: float, success: bool) -> None:
        """Update Thompson Sampling arm for this model+task_type pair."""
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            get_thompson_selector().update(model_id, task_type, quality, success)
        except Exception as e:
            logger.debug(f"Thompson arm update failed: {e}")


# Singleton
_feedback_loop: FeedbackLoop | None = None


def get_feedback_loop() -> FeedbackLoop:
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop
