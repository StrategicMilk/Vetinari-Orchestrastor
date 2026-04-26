"""Learning pipeline Litestar handlers for Thompson Sampling, quality, and training data.

Native Litestar equivalents of the routes previously registered by
``learning_api._register(bp)``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

Endpoints
---------
    GET /api/v1/learning/thompson          — Thompson Sampling arm states
    GET /api/v1/learning/quality-history   — Quality score time series
    GET /api/v1/learning/training-stats    — Training data statistics
    GET /api/v1/learning/workflow-patterns — Discovered workflow patterns
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get

    from vetinari.web.responses import litestar_error_response

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def create_learning_api_handlers() -> list[Any]:
    """Create Litestar handlers for the learning pipeline API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler objects, or empty list when Litestar
        is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — learning API handlers not registered")
        return []

    # -- GET /api/v1/learning/thompson -------------------------------------------

    @get("/api/v1/learning/thompson", media_type=MediaType.JSON)
    async def get_thompson_arms() -> dict[str, Any]:
        """Return Thompson Sampling arm states for all registered model/task-type pairs.

        Reads alpha, beta, mean, and pull counts from the module-level
        ThompsonSelector singleton so the UI can display current exploitation
        confidence for each arm without triggering a sample.

        Returns:
            JSON object with an ``arms`` dict keyed by arm identifier,
            or a 503 response when the Thompson selector is unavailable.
        """
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            selector = get_thompson_selector()
            arms = {}
            for key, arm in selector._arms.items():
                arms[key] = {
                    "model_id": arm.model_id,
                    "task_type": arm.task_type,
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "mean": arm.mean,
                    "total_pulls": arm.total_pulls,
                    "last_updated": arm.last_updated,
                }
            return {"arms": arms}
        except Exception:
            logger.warning("Thompson selector unavailable — cannot serve arm states, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Thompson sampling subsystem unavailable", 503
            )

    # -- GET /api/v1/learning/quality-history ------------------------------------

    @get("/api/v1/learning/quality-history", media_type=MediaType.JSON)
    async def get_quality_history() -> dict[str, Any]:
        """Return quality score time series for all scored tasks.

        Serialises each QualityScore record from the module-level scorer's
        history so the UI can plot score trends over time per model and task type.

        Returns:
            JSON object with a ``history`` list of scored task records,
            or a 503 response when the quality scorer is unavailable.
        """
        try:
            from vetinari.learning.quality_scorer import get_quality_scorer

            scorer = get_quality_scorer()
            history = scorer.get_history()
            serialised = [
                {
                    "task_id": s.task_id,
                    "model_id": s.model_id,
                    "task_type": s.task_type,
                    "overall_score": s.overall_score,
                    "method": s.method,
                    "timestamp": s.timestamp,
                }
                for s in history
            ]
            return {"history": serialised}
        except Exception:
            logger.warning("Quality scorer unavailable — cannot serve quality history, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Quality history subsystem unavailable", 503
            )

    # -- GET /api/v1/learning/training-stats -------------------------------------

    @get("/api/v1/learning/training-stats", media_type=MediaType.JSON)
    async def get_training_stats() -> dict[str, Any]:
        """Return aggregate statistics from the training data collector.

        Calls the module-level TrainingDataCollector singleton to summarise
        example counts, coverage per task type, and collection timestamps so
        the UI can monitor training data health.

        Returns:
            JSON object with a ``stats`` dict,
            or a 503 response when the training collector is unavailable.
        """
        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            stats = collector.get_stats()
            return {"stats": stats}
        except Exception:
            logger.warning("Training data collector unavailable — cannot serve training stats, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Training stats subsystem unavailable", 503
            )

    # -- GET /api/v1/learning/workflow-patterns ----------------------------------

    @get("/api/v1/learning/workflow-patterns", media_type=MediaType.JSON)
    async def get_workflow_patterns() -> dict[str, Any]:
        """Return all discovered workflow patterns from the WorkflowLearner.

        Calls ``WorkflowLearner().get_all_patterns()`` to retrieve every pattern
        the learner has extracted from completed pipeline runs so the dashboard
        can display learning progress and pattern coverage.

        Returns:
            JSON object with a ``patterns`` list of pattern records,
            or a 503 response when the workflow learner is unavailable.
        """
        try:
            from vetinari.learning.workflow_learner import WorkflowLearner

            learner = WorkflowLearner()
            patterns = learner.get_all_patterns()
            return {"patterns": patterns}
        except Exception:
            logger.warning("Workflow learner unavailable — cannot serve workflow patterns, returning 503")
            return litestar_error_response(  # type: ignore[return-value]
                "Workflow patterns subsystem unavailable", 503
            )

    return [
        get_thompson_arms,
        get_quality_history,
        get_training_stats,
        get_workflow_patterns,
    ]
