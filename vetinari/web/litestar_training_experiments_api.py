"""Training experiments API — native Litestar handlers. Native Litestar equivalents (ADR-0066). URL paths identical to Flask.

Endpoints:
    GET  /api/v1/training/experiments/{experiment_id}  — experiment detail
    POST /api/v1/training/experiments/compare          — compare experiments
    GET  /api/v1/training/data/browse                  — paginated data browser
    POST /api/v1/training/data/upload                  — upload JSONL training data
    GET  /api/v1/training/automation/rules             — list automation rules
    POST /api/v1/training/automation/rules             — create/update a rule
    GET  /api/v1/training/progress/stream              — SSE live training metrics
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, get, post
    from litestar.datastructures import UploadFile
    from litestar.params import Parameter
    from litestar.response import ServerSentEvent

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# Path for persisting automation rules between process restarts.
_RULES_PATH = Path(__file__).parents[2] / "config" / "training_automation_rules.yaml"

# Upload directory for JSONL training data files.
_UPLOAD_DIR = Path(__file__).parents[2] / "outputs" / "training_uploads"

# Maximum upload size for training data (50 MB).
_MAX_TRAINING_UPLOAD_BYTES = 50 * 1024 * 1024

# Lock protecting _automation_rules list mutations and lazy initialization.
_automation_rules_lock = threading.Lock()

# Module-level rule store — loaded lazily on first access, not at import time.
_automation_rules: list[dict[str, Any]] | None = None

# Permitted file extensions for training data uploads.
_ALLOWED_EXTENSIONS = frozenset({".jsonl", ".json"})


def _load_automation_rules() -> list[dict[str, Any]]:
    """Load automation rules from the YAML persistence file.

    Returns an empty list when the file does not exist yet or when it contains
    no rules.  Logs a warning and returns an empty list on parse errors so that
    the API endpoint remains callable even after a corrupted write.

    Returns:
        The list of rule dicts stored in the YAML file, or an empty list.
    """
    if not _RULES_PATH.exists():
        return []
    try:
        with Path(_RULES_PATH).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, list):
            return data
        logger.warning("_load_automation_rules: unexpected YAML structure, resetting rules")
        return []
    except yaml.YAMLError as exc:
        logger.warning("_load_automation_rules: YAML parse error: %s", exc)
        return []


def _save_automation_rules(rules: list[dict[str, Any]]) -> None:
    """Persist automation rules to the YAML file on disk.

    Creates the ``config/`` directory if it does not already exist.

    Args:
        rules: The full list of rule dicts to write.

    Raises:
        OSError: If the file cannot be written due to permission or I/O errors.
    """
    _RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with Path(_RULES_PATH).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(rules, fh, default_flow_style=False, allow_unicode=True)


def _get_automation_rules() -> list[dict[str, Any]]:
    """Return the automation rules list, loading from disk on first call.

    Uses double-checked locking so that disk I/O happens at most once per
    process lifetime even under concurrent requests.

    Returns:
        The list of automation rule dicts, possibly empty if no rules exist.
    """
    global _automation_rules
    if _automation_rules is None:
        with _automation_rules_lock:
            if _automation_rules is None:
                _automation_rules = _load_automation_rules()
    return _automation_rules


async def _generate_training_progress_events() -> AsyncGenerator[dict[str, Any], None]:
    """Yield SSE frames for the live training-progress stream.

    Emits ``training_status`` events for status transitions and
    ``training_progress`` events containing per-step metrics while a training
    run is active.  A keepalive comment is emitted every 25 seconds to prevent
    proxies from closing an idle connection.

    Yields:
        Dicts with ``event`` + ``data`` keys for status/progress frames, or
        ``comment`` key for keepalives — both understood by Litestar's SSE
        encoder.
    """
    import asyncio

    from vetinari.web.litestar_training_api import _get_scheduler, _is_scheduler_training

    if not _is_scheduler_training():
        yield {"event": "training_status", "data": json.dumps({"status": "idle"})}
        return

    last_heartbeat = time.monotonic()
    from vetinari.types import StatusEnum

    last_status: str = StatusEnum.RUNNING.value
    yield {"event": "training_status", "data": json.dumps({"status": last_status})}

    while _is_scheduler_training():
        now = time.monotonic()

        # Emit current progress snapshot from the real scheduler job if available.
        scheduler = _get_scheduler()
        if scheduler is not None:
            job = scheduler.current_job
            if job is not None:
                progress_snapshot = {
                    "job_id": job.job_id,
                    "activity_description": job.activity_description,
                    "progress": job.progress,
                }
                yield {"event": "training_progress", "data": json.dumps(progress_snapshot)}

        # Heartbeat every 25 seconds to keep the connection alive.
        if now - last_heartbeat >= 25:
            yield {"comment": "keepalive"}
            last_heartbeat = now

        await asyncio.sleep(2)

    # Training just finished — emit final status.
    yield {"event": "training_status", "data": json.dumps({"status": "finished"})}


def create_training_experiments_handlers() -> list[Any]:
    """Create all Litestar route handlers for the training experiments API.

    Called by ``vetinari.web.litestar_app.create_app()`` to register these
    handlers in the main Litestar application.

    Returns:
        List of Litestar route handler functions, or empty list when
        Litestar is not installed.
    """
    if not _LITESTAR_AVAILABLE:
        logger.debug("Litestar not available — training experiments API handlers not registered")
        return []

    from vetinari.web.litestar_guards import admin_guard

    # -- GET /api/v1/training/experiments/{experiment_id} ----------------------

    @get("/api/v1/training/experiments/{experiment_id:str}", media_type=MediaType.JSON)
    async def get_experiment(experiment_id: str) -> Any:
        """Return detailed data for a single training experiment.

        Searches the AgentTrainer stats for an entry whose key matches
        ``experiment_id``.  Returns 503 when the training subsystem itself is
        unavailable (distinct from experiment-not-found which is 404).

        Args:
            experiment_id: The unique identifier of the experiment to retrieve.

        Returns:
            JSON object with the full experiment data on success.
            503 when the AgentTrainer module cannot be loaded.
            404 when the trainer is available but the experiment is not found.
        """
        from litestar import Response

        from vetinari.web.responses import litestar_error_response

        trainer_available = False
        agent_stats: dict[str, Any] = {}
        trainer_load_error: Exception | None = None

        try:
            from vetinari.training.agent_trainer import get_agent_trainer

            trainer = get_agent_trainer()
            agent_stats = trainer.get_stats()
            trainer_available = True
        except Exception as exc:
            trainer_load_error = exc
            logger.warning("get_experiment: agent trainer unavailable: %s", exc)

        if not trainer_available:
            # Trainer module itself is missing — service unavailable, not just missing data
            logger.warning(
                "get_experiment: trainer not loaded for experiment %r, returning 503: %s",
                experiment_id,
                trainer_load_error,
            )
            return litestar_error_response("Training subsystem unavailable — cannot retrieve experiment data", 503)

        if experiment_id in agent_stats:
            return {
                "status": "ok",
                "experiment_id": experiment_id,
                "data": agent_stats[experiment_id],
            }

        # Also check experiment_id as a numeric index if stats is list-shaped.
        try:
            idx = int(experiment_id)
            ranked: list[Any] = []
            try:
                from vetinari.training.agent_trainer import get_agent_trainer

                trainer2 = get_agent_trainer()
                ranked = trainer2.get_training_priority()
            except Exception as exc:
                logger.warning("get_experiment: priority lookup failed: %s", exc)
            if 0 <= idx < len(ranked):
                name, score = ranked[idx]
                return {
                    "status": "ok",
                    "experiment_id": experiment_id,
                    "data": {"agent": name, "priority_score": score},
                }
        except ValueError:
            logger.warning("get_experiment: experiment_id %r is not a numeric index", experiment_id)

        # Trainer was available but experiment was not found → 404
        return Response(
            content={"status": "error", "error": f"Experiment '{experiment_id}' not found"},
            status_code=404,
            media_type="application/json",
        )

    # -- POST /api/v1/training/experiments/compare -----------------------------

    @post("/api/v1/training/experiments/compare", media_type=MediaType.JSON)
    async def compare_experiments(data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compare two or more training experiments side-by-side.

        Accepts a JSON body with ``experiment_ids`` (list of strings).  Returns
        config differences and a metrics comparison table for each requested
        experiment.  At least two IDs are required.

        Args:
            data: JSON request body containing ``experiment_ids`` list.

        Returns:
            JSON with ``experiments`` mapping each ID to its data, and a
            ``comparison`` summary of common and differing fields.

        Raises:
            ValueError: If fewer than two experiment IDs are provided.
        """
        from vetinari.web.request_validation import json_object_body
        from vetinari.web.responses import litestar_error_response

        body = json_object_body(data)
        if body is None:
            return litestar_error_response(  # type: ignore[return-value]
                "Request body must be a JSON object", 400
            )

        experiment_ids_raw = body.get("experiment_ids", [])
        if not isinstance(experiment_ids_raw, list):
            return litestar_error_response(  # type: ignore[return-value]
                "'experiment_ids' must be a list", 422
            )
        if any(not isinstance(eid, str) for eid in experiment_ids_raw):
            return litestar_error_response(  # type: ignore[return-value]
                "'experiment_ids' must be a list of strings", 422
            )
        experiment_ids: list[str] = experiment_ids_raw

        if len(experiment_ids) < 2:
            return litestar_error_response(  # type: ignore[return-value]
                "At least 2 experiment_ids are required", 400
            )

        agent_stats: dict[str, Any] = {}
        try:
            from vetinari.training.agent_trainer import get_agent_trainer

            trainer = get_agent_trainer()
            agent_stats = trainer.get_stats()
        except Exception as exc:
            logger.warning("compare_experiments: agent trainer unavailable: %s", exc)

        experiments: dict[str, Any] = {}
        missing: list[str] = []
        for eid in experiment_ids:
            if eid in agent_stats:
                experiments[eid] = agent_stats[eid]
            else:
                missing.append(eid)
                experiments[eid] = None

        # Build a simple field-level diff across the found experiments.
        found = {k: v for k, v in experiments.items() if v is not None}
        comparison: dict[str, Any] = {"config_differences": {}, "metrics": {}}
        if len(found) >= 2:
            all_keys: set[str] = set()
            for item_data in found.values():
                if isinstance(item_data, dict):
                    all_keys.update(item_data.keys())
            for key in all_keys:
                values = {eid: item_data.get(key) for eid, item_data in found.items() if isinstance(item_data, dict)}
                unique_vals = {str(v) for v in values.values()}
                if len(unique_vals) > 1:
                    comparison["config_differences"][key] = values
                else:
                    comparison["metrics"][key] = next(iter(values.values()))

        return {
            "status": "ok",
            "experiments": experiments,
            "missing": missing,
            "comparison": comparison,
        }

    # -- GET /api/v1/training/data/browse --------------------------------------

    @get("/api/v1/training/data/browse", media_type=MediaType.JSON)
    async def browse_training_data(
        page: int = Parameter(query="page", default=1),
        per_page: int = Parameter(query="per_page", default=20),
        type_filter: str | None = Parameter(query="type", default=None),
        min_quality: float | None = Parameter(query="min_quality", default=None),
    ) -> Any:
        """Return a paginated list of training data items.

        Queries the ``TrainingDataCollector`` when available.  Returns 503 when
        the collector module cannot be loaded — callers should not treat an empty
        list as "no data" because the subsystem may simply be unavailable.

        Args:
            page: 1-based page number (default 1).
            per_page: Items per page (default 20, max 100).
            type_filter: Optional filter on item task_type field.
            min_quality: Optional minimum quality score filter (float).

        Returns:
            JSON with ``items``, ``total``, ``page``, ``per_page`` keys on success.
            503 when the TrainingDataCollector module cannot be loaded.
        """
        from vetinari.web.responses import litestar_error_response

        if page < 1:
            page = 1
        per_page = max(1, min(per_page, 100))

        collector_available = False
        raw_items: list[dict[str, Any]] = []
        try:
            from vetinari.learning.training_collector import get_training_collector

            collector = get_training_collector()
            collector_available = True
            try:
                sft_records = collector.export_sft_dataset(min_score=min_quality or 0.0)  # noqa: VET112 - empty fallback preserves optional request metadata contract
                if isinstance(sft_records, list):
                    raw_items.extend(sft_records)
            except Exception:
                logger.warning("Could not load training records for browse", exc_info=True)
        except Exception as exc:
            logger.warning("browse_training_data: collector unavailable — returning 503: %s", exc)

        if not collector_available:
            return litestar_error_response("Training data subsystem unavailable — collector could not be loaded", 503)

        # Apply filters.
        if type_filter:
            raw_items = [i for i in raw_items if isinstance(i, dict) and i.get("task_type") == type_filter]
        if min_quality is not None:
            raw_items = [i for i in raw_items if isinstance(i, dict) and float(i.get("quality", 0.0)) >= min_quality]

        total = len(raw_items)
        start = (page - 1) * per_page
        page_items = raw_items[start : start + per_page]

        return {"items": page_items, "total": total, "page": page, "per_page": per_page}

    # -- GET /api/v1/training/automation/rules ---------------------------------

    @get("/api/v1/training/automation/rules", media_type=MediaType.JSON)
    async def list_automation_rules() -> Any:
        """Return the list of stored training automation rule definitions.

        Rules are loaded from ``config/training_automation_rules.yaml`` on first
        access and kept in the module-level ``_automation_rules`` list.  These are
        persisted rule definitions only — no scheduler executes them automatically.
        Triggering automation requires a manual API call or an external scheduler.

        Returns:
            JSON object with ``rules`` list, ``count`` integer, and ``scheduling``
            string (always ``"manual"`` — rules are stored but not auto-scheduled).
            503 when the automation rules config cannot be loaded.
        """
        try:
            rules = _get_automation_rules()
            return {"rules": list(rules), "count": len(rules), "scheduling": "manual"}
        except Exception as exc:
            logger.warning(
                "list_automation_rules: automation rules config unavailable — returning 503: %s",
                exc,
            )
            from vetinari.web.responses import litestar_error_response

            return litestar_error_response(
                "Training automation rules subsystem unavailable — config could not be loaded", 503
            )

    # -- POST /api/v1/training/automation/rules --------------------------------

    @post(
        "/api/v1/training/automation/rules",
        media_type=MediaType.JSON,
        guards=[admin_guard],
        status_code=200,
    )
    async def create_automation_rule(
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create or update a training automation rule.

        Accepts a JSON body with an arbitrary rule definition.  If the body
        contains an ``id`` field and a rule with that ID already exists, the
        existing entry is replaced.  Otherwise the rule is appended (201).

        Args:
            data: JSON request body containing the rule definition.

        Returns:
            JSON object with ``status``, ``rule``, and ``count`` fields.
        """
        from vetinari.web.responses import litestar_error_response

        rule = data
        if not rule:
            return litestar_error_response(  # type: ignore[return-value]
                "Request body must be a non-empty JSON object", 400
            )

        rule_id: str | None = rule.get("id") or rule.get("name")
        # Ensure the rule list is initialised BEFORE acquiring the write lock.
        # _get_automation_rules() uses its own double-checked locking internally;
        # calling it inside _automation_rules_lock would deadlock on the first
        # write when _automation_rules is still None.
        _get_automation_rules()  # prime singleton if not already loaded
        with _automation_rules_lock:
            # _automation_rules is guaranteed non-None after the call above.
            rules = _automation_rules  # type: ignore[assignment]
            assert rules is not None
            if rule_id:
                for i, existing in enumerate(rules):
                    if (existing.get("id") or existing.get("name")) == rule_id:
                        rules[i] = rule
                        _save_automation_rules(rules)
                        logger.info("create_automation_rule: updated rule id=%s", rule_id)
                        return {"status": "updated", "rule": rule, "count": len(rules)}

            rules.append(rule)
            _save_automation_rules(rules)
            logger.info("create_automation_rule: created rule (total=%d)", len(rules))
            return {"status": "created", "rule": rule, "count": len(rules)}

    # -- GET /api/v1/training/progress/stream (SSE) ----------------------------

    @get("/api/v1/training/progress/stream", sync_to_thread=False)
    def stream_training_progress() -> ServerSentEvent:
        """SSE endpoint that streams live training metrics to the browser.

        Probes the scheduler before committing SSE response headers so that a
        scheduler failure can be returned as a proper 503 rather than a
        mid-stream crash.  Once headers are committed, Litestar cannot change
        the status code, so the pre-flight check must happen here.

        Emits ``training_status`` events for status transitions and
        ``training_progress`` events containing per-step metrics while a
        training run is active.  A keepalive comment is sent every 25 seconds
        to prevent proxies from closing an idle connection.

        Returns:
            A ``ServerSentEvent`` streaming response.

        Raises:
            HTTPException: 503 if the training scheduler is unavailable.
        """
        from litestar.exceptions import HTTPException

        from vetinari.web.litestar_training_api import _is_scheduler_training

        try:
            _is_scheduler_training()
        except Exception as err:
            logger.warning(
                "stream_training_progress: scheduler probe failed — returning 503 before SSE headers committed"
            )
            raise HTTPException(status_code=503, detail="Training scheduler unavailable") from err
        return ServerSentEvent(
            _generate_training_progress_events(),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- POST /api/v1/training/data/upload -------------------------------------

    @post(
        "/api/v1/training/data/upload",
        media_type=MediaType.JSON,
        guards=[admin_guard],
    )
    async def upload_training_data(data: UploadFile) -> dict[str, Any]:
        """Accept a JSONL file upload and persist valid entries to disk.

        Expects a ``multipart/form-data`` request with a ``file`` field
        containing a ``.jsonl`` or ``.json`` file.  Each line is parsed as a
        JSON object; valid entries are written to
        ``outputs/training_uploads/<filename>``.  Invalid lines (empty or
        non-JSON) are counted and reported but do not cause the request to fail.

        Args:
            data: The uploaded file from the ``file`` multipart field.

        Returns:
            JSON object with ``ok``, ``imported``, ``errors``, ``filename``.
        """
        from vetinari.web.responses import litestar_error_response

        original_name: str = data.filename or "upload.jsonl"
        suffix = Path(original_name).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            return litestar_error_response(  # type: ignore[return-value]
                f"Unsupported file type '{suffix}'. Allowed: .jsonl, .json", 400
            )

        try:
            raw_bytes = await data.read()
            if len(raw_bytes) > _MAX_TRAINING_UPLOAD_BYTES:
                return litestar_error_response(  # type: ignore[return-value]
                    f"File size {len(raw_bytes)} bytes exceeds the 50 MB limit", 400
                )
            raw_content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Uploaded training file is not valid UTF-8 — rejecting with 400")
            return litestar_error_response("File must be valid UTF-8 text", 400)  # type: ignore[return-value]

        valid_entries: list[dict[str, Any]] = []
        error_count = 0
        for line in raw_content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
                if isinstance(entry, dict):
                    valid_entries.append(entry)
                else:
                    error_count += 1
            except json.JSONDecodeError:
                error_count += 1

        # Sanitise filename to prevent path traversal without werkzeug dependency.
        import re

        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(original_name).name) or "upload.jsonl"

        try:
            _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = _UPLOAD_DIR / safe_name
            with Path(dest).open("w", encoding="utf-8") as fh:
                for entry in valid_entries:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("upload_training_data: I/O error writing %s", original_name)
            return litestar_error_response("Internal server error", 500)  # type: ignore[return-value]

        logger.info(
            "upload_training_data: stored %d entries (%d errors) as %s",
            len(valid_entries),
            error_count,
            safe_name,
        )
        return {
            "ok": True,
            "imported": len(valid_entries),
            "errors": error_count,
            "filename": safe_name,
        }

    return [
        get_experiment,
        compare_experiments,
        browse_training_data,
        list_automation_rules,
        create_automation_rule,
        stream_training_progress,
        upload_training_data,
    ]
