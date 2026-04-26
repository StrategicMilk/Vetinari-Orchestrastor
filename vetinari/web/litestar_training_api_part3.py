"""Training API part 3 — additional training pipeline endpoints.

Provides handlers for training dry-runs, rule management, data sync,
synthetic data generation, and idle training statistics.

Endpoints
---------
    POST /api/v1/training/dry-run           — validate training config without executing
    POST /api/v1/training/rules             — set training constraint rules
    POST /api/v1/training/sync-data         — trigger training data synchronization
    POST /api/v1/training/generate-synthetic — generate synthetic training data
    GET  /api/v1/training/idle-stats        — training statistics during idle periods
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, get, post

    from vetinari.web.litestar_guards import admin_guard

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


def _create_training_api_handlers_part3() -> list[Any]:
    """Create the third batch of training API route handlers.

    Covers dry-run validation, rule management, data sync, synthetic data
    generation, and idle training statistics.  Called by
    ``create_training_api_handlers()`` in ``litestar_training_api_part2``.

    Returns:
        List of 5 Litestar route handler functions, or an empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.litestar_training_api import _get_scheduler

    @post("/api/v1/training/dry-run", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_dry_run(data: dict[str, Any]) -> Response | dict[str, Any]:
        """Validate a training configuration without starting a training cycle.

        Accepts a training config dict and checks it against scheduler
        constraints.  Returns immediately with a validity verdict so the
        caller can surface errors to the user before committing.

        Args:
            data: Training configuration to validate.  Recognised keys depend
                on the scheduler's ``validate_config`` implementation, but
                typically include ``max_epochs``, ``learning_rate``, and
                ``target_agent``.

        Returns:
            JSON with ``status`` set to ``"valid"`` and the echoed ``config``,
            or ``status`` set to ``"invalid"`` and an ``errors`` list.  Returns
            503 when the scheduler is unavailable.
        """
        if not data:
            return Response(
                content={"status": "error", "message": "Request body must not be empty — provide a training config"},
                status_code=422,
                media_type=MediaType.JSON,
            )

        _DRY_RUN_KEYS = frozenset({"max_epochs", "learning_rate", "target_agent", "batch_size", "warmup_steps"})
        if not _DRY_RUN_KEYS.intersection(data):
            return Response(
                content={"status": "error", "message": "Request body contains no recognised training config keys"},
                status_code=422,
                media_type=MediaType.JSON,
            )

        scheduler = _get_scheduler()
        if scheduler is None:
            return Response(
                content={"status": "error", "message": "Training scheduler not available"},
                status_code=503,
                media_type=MediaType.JSON,
            )

        errors: list[str] = []
        try:
            if hasattr(scheduler, "validate_config"):
                result = scheduler.validate_config(data)
                # validate_config may return (is_valid, errors) or just bool
                if isinstance(result, tuple):
                    is_valid, errors = result
                else:
                    is_valid = bool(result)
            else:
                # Scheduler exists but has no validation API — treat as valid
                is_valid = True
        except Exception as exc:
            logger.warning(
                "training/dry-run: config validation raised an error — returning invalid: %s",
                exc,
            )
            is_valid = False
            errors = ["Validation failed unexpectedly. Check server logs for details."]

        if is_valid:
            return {"status": "valid", "config": data}
        return {"status": "invalid", "errors": errors}

    @post("/api/v1/training/rules", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_rules_set(data: dict[str, Any]) -> Response | dict[str, Any]:
        """Store training constraint rules for the pipeline.

        Accepts a ``rules`` list in the request body and persists it via the
        training config module when available.  Rules constrain which agents
        are eligible for training, minimum data thresholds, and quality gates.

        Args:
            data: Request body.  Expected key: ``rules`` (list of rule dicts).

        Returns:
            JSON with ``status`` and ``rules_count`` on success, or an error
            response if the rules store is unavailable.
        """
        if "rules" not in data:
            return Response(
                content={"status": "error", "error": "'rules' is required"},
                status_code=422,
                media_type=MediaType.JSON,
            )
        rules: list[Any] = data.get("rules", [])
        if not isinstance(rules, list):
            return Response(
                content={"status": "error", "error": "'rules' must be a list"},
                status_code=400,
                media_type=MediaType.JSON,
            )

        stored = False
        try:
            from vetinari.training.training_config import TrainingConfig

            config = TrainingConfig()
            if hasattr(config, "set_constraint_rules"):
                config.set_constraint_rules(rules)
                stored = True
        except ImportError:
            logger.debug("training/rules: training_config module not available — rules not persisted")
        except Exception as exc:
            logger.warning(
                "training/rules: failed to persist %d rules — rules not stored: %s",
                len(rules),
                exc,
            )

        if not stored:
            # All persistence attempts failed — signal service unavailability so
            # the caller knows the rules were NOT applied, not silently dropped.
            return Response(
                content={
                    "status": "error",
                    "error": "Rules could not be persisted — training config module unavailable",
                    "rules_count": len(rules),
                    "persisted": False,
                },
                status_code=503,
                media_type=MediaType.JSON,
            )
        return {"status": "ok", "rules_count": len(rules), "persisted": stored}

    @post("/api/v1/training/sync-data", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_sync_data(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Trigger synchronization of training data from all collection sources.

        Initiates a background sync that pulls training records from the
        collector, seeder, and any registered data providers into the active
        training dataset.  The sync runs asynchronously; use
        ``GET /api/v1/training/data/stats`` to poll progress.  This endpoint
        accepts no body parameters; any non-empty body is rejected with 422.

        Args:
            data: Request body — must be empty (``{}``).

        Returns:
            JSON with ``status`` and a descriptive ``message``.
        """
        from vetinari.web.responses import litestar_error_response

        if data is not None:
            return litestar_error_response("This endpoint takes no request body parameters", code=422)
        synced_sources: list[str] = []
        errors: list[str] = []

        try:
            from vetinari.learning.training_data import TrainingDataCollector

            collector = TrainingDataCollector()
            if hasattr(collector, "sync"):
                collector.sync()
                synced_sources.append("collector")
        except ImportError:
            logger.debug("training/sync-data: TrainingDataCollector not available")
        except Exception as exc:
            logger.warning(
                "training/sync-data: collector sync failed — continuing with other sources: %s",
                exc,
            )
            errors.append("Training data collector sync failed. Check server logs for details.")

        try:
            from vetinari.training.data_seeder import TrainingDataSeeder

            seeder = TrainingDataSeeder()
            if hasattr(seeder, "sync"):
                seeder.sync()
                synced_sources.append("seeder")
        except ImportError:
            logger.debug("training/sync-data: TrainingDataSeeder not available")
        except Exception as exc:
            logger.warning(
                "training/sync-data: seeder sync failed — continuing with other sources: %s",
                exc,
            )
            errors.append("Training data seeder sync failed. Check server logs for details.")

        if not synced_sources:
            logger.warning(
                "training/sync-data: no sources responded to sync request — errors: %s",
                errors,
            )
            return {
                "status": "error",
                "message": "Training data sync failed — no sources responded",
                "sources_synced": [],
                "errors": errors,
            }

        message = f"Training data sync initiated for sources: {', '.join(synced_sources)}"
        return {
            "status": "started",
            "message": message,
            "sources_synced": synced_sources,
            "errors": errors,
        }

    @post("/api/v1/training/generate-synthetic", media_type=MediaType.JSON, guards=[admin_guard])
    async def training_generate_synthetic(data: dict[str, Any]) -> Response | dict[str, Any]:
        """Initiate synthetic training data generation.

        Accepts a generation config and starts the synthetic data pipeline in
        the background.  Synthetic data augments real training records when
        real-world volume is insufficient for stable fine-tuning.

        Args:
            data: Generation config.  Recognised keys: ``agent_type`` (str),
                ``count`` (int, number of samples to generate), ``domain``
                (str, task domain hint), ``temperature`` (float, via
                InferenceConfigManager — do not hardcode here).

        Returns:
            JSON with ``status``, ``message``, and echoed generation params.
            Returns 400 if ``count`` is provided but non-positive.
        """
        if not data:
            return Response(
                content={
                    "status": "error",
                    "message": "Request body must not be empty — provide generation parameters",
                },
                status_code=422,
                media_type=MediaType.JSON,
            )

        _SYNTHETIC_KEYS = frozenset({"agent_type", "count", "domain", "temperature"})
        if not _SYNTHETIC_KEYS.intersection(data):
            return Response(
                content={"status": "error", "message": "Request body contains no recognised generation parameter keys"},
                status_code=422,
                media_type=MediaType.JSON,
            )

        # Validate 'count' only when explicitly provided — absent means "use default".
        if "count" in data:
            count_raw = data["count"]
            # bool is a subclass of int — reject it explicitly. Also reject None and non-int.
            if count_raw is None or isinstance(count_raw, bool) or not isinstance(count_raw, int):
                return Response(
                    content={"status": "error", "error": "'count' must be an integer"},
                    status_code=400,
                    media_type=MediaType.JSON,
                )
            if count_raw <= 0:
                return Response(
                    content={"status": "error", "error": "'count' must be a positive integer"},
                    status_code=400,
                    media_type=MediaType.JSON,
                )
            if count_raw > 10000:  # sanity cap — no legitimate run needs more than 10k samples
                return Response(
                    content={"status": "error", "error": "'count' must not exceed 10000"},
                    status_code=400,
                    media_type=MediaType.JSON,
                )
        generation_started = False
        try:
            from vetinari.training.synthetic_data import SyntheticDataGenerator

            generator = SyntheticDataGenerator()
            if hasattr(generator, "generate_async"):
                generator.generate_async(data)
                generation_started = True
            elif hasattr(generator, "generate"):
                generator.generate(data)
                generation_started = True
        except ImportError:
            logger.debug("training/generate-synthetic: SyntheticDataGenerator not available")
        except Exception as exc:
            logger.warning(
                "training/generate-synthetic: generation pipeline failed to start: %s",
                exc,
            )

        if not generation_started:
            logger.warning(
                "training/generate-synthetic: no generator backend responded — SyntheticDataGenerator unavailable"
            )
            return Response(
                content={
                    "status": "error",
                    "message": "Synthetic data generation failed — no generation backend available",
                    "generation_started": False,
                    "params": data,
                },
                status_code=503,
                media_type=MediaType.JSON,
            )

        return {
            "status": "started",
            "message": "Synthetic data generation initiated",
            "generation_started": True,
            "params": data,
        }

    @get("/api/v1/training/idle-stats", media_type=MediaType.JSON)
    async def training_idle_stats() -> dict[str, Any]:
        """Return training statistics gathered during idle-time sessions.

        Queries the TrainingScheduler for metrics accumulated across all
        previous idle training windows — total sessions, cumulative training
        duration, last session timestamp, and per-agent run counts.

        Returns:
            JSON with ``status``, ``idle_sessions`` count, and available
            scheduler metrics.  Returns a zero-count payload when the
            scheduler is unavailable rather than an error, because this is a
            read-only diagnostic endpoint.
        """
        scheduler = _get_scheduler()
        if scheduler is None:
            return {
                "status": "unavailable",
                "idle_sessions": 0,
            }

        stats: dict[str, Any] = {
            "status": "ok",
            "idle_sessions": 0,
        }

        try:
            if hasattr(scheduler, "get_idle_stats"):
                idle_metrics = scheduler.get_idle_stats()
                stats.update(idle_metrics)
            elif hasattr(scheduler, "get_stats"):
                raw = scheduler.get_stats()
                stats["idle_sessions"] = raw.get("idle_sessions", 0)
                stats["scheduler_stats"] = raw
        except Exception as exc:
            logger.warning(
                "training/idle-stats: could not read scheduler stats — returning zero counts: %s",
                exc,
            )

        return stats

    return [
        training_dry_run,
        training_rules_set,
        training_sync_data,
        training_generate_synthetic,
        training_idle_stats,
    ]
