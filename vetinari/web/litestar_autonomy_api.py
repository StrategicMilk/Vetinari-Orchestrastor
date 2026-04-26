"""Autonomy policy API handlers — manages per-action-type autonomy levels.

Native Litestar equivalents of the routes previously registered by
``autonomy_api``. Part of Flask->Litestar migration (ADR-0066).
URL paths identical to Flask originals.

Provides CRUD endpoints for autonomy policies that control how much
freedom agents have for different action types. Each action type has
a level: suggest, propose, confirm, or auto.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Request, Response, get, post, put

    from vetinari.web.litestar_guards import admin_guard

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False

# -- Constants ----------------------------------------------------------------

VALID_LEVELS = frozenset({"suggest", "propose", "confirm", "auto"})  # low→high looseness

# Resolved at import time; path construction is safe (no disk I/O)
_CONFIG_PATH = Path("config") / "autonomy_policies.yaml"

# Protects _policies_cache and file writes
_lock = threading.Lock()

# In-memory cache; populated lazily, invalidated on each write.
# Writer: _save_policies_unlocked()   Reader: _load_policies()
_policies_cache: list[AutonomyPolicy] | None = None

# -- Data model ---------------------------------------------------------------


@dataclass
class AutonomyPolicy:
    """A single autonomy-level assignment for one action type.

    Attributes:
        action_type: Machine-readable action identifier (e.g. ``"code_generation"``).
        level: Autonomy level — one of ``suggest``, ``propose``, ``confirm``, or ``auto``.
        description: Human-readable explanation of what this action type covers.
        updated_at: ISO-8601 timestamp of the last level change (empty when default).
    """

    action_type: str
    level: str  # one of VALID_LEVELS
    description: str = ""
    updated_at: str = ""  # ISO-8601; empty for unchanged defaults

    def __repr__(self) -> str:
        return "AutonomyPolicy(...)"


# Default policies applied when the YAML file does not yet exist.
DEFAULT_POLICIES: list[AutonomyPolicy] = [
    AutonomyPolicy("code_generation", "confirm", "Generate or modify source code"),
    AutonomyPolicy("code_review", "auto", "Review code quality and suggest improvements"),
    AutonomyPolicy("test_generation", "confirm", "Create or modify test files"),
    AutonomyPolicy("file_modification", "propose", "Modify existing project files"),
    AutonomyPolicy("deployment", "suggest", "Deploy or release changes"),
    AutonomyPolicy("dependency_update", "propose", "Add or update project dependencies"),
    AutonomyPolicy("documentation", "auto", "Create or update documentation"),
    AutonomyPolicy("research", "auto", "Research and information gathering"),
]

# -- Storage helpers ----------------------------------------------------------


def _policy_from_dict(raw: dict[str, Any]) -> AutonomyPolicy:
    """Construct an :class:`AutonomyPolicy` from a raw YAML dict.

    Unknown keys are silently ignored so the file can be hand-edited with
    comments without breaking the loader.

    Args:
        raw: Mapping loaded from one entry in ``autonomy_policies.yaml``.

    Returns:
        A fully populated :class:`AutonomyPolicy` instance.

    Raises:
        ValueError: If ``action_type`` or ``level`` are missing or blank.
    """
    action_type = str(raw.get("action_type") or "").strip()
    level = str(raw.get("level") or "").strip()
    if not action_type:
        raise ValueError("Policy entry missing 'action_type'")
    if not level:
        raise ValueError(f"Policy '{action_type}' missing 'level'")
    return AutonomyPolicy(
        action_type=action_type,
        level=level,
        description=str(raw.get("description") or ""),
        updated_at=str(raw.get("updated_at") or ""),
    )


def _load_policies() -> list[AutonomyPolicy]:
    """Load policies from the YAML config file, using the in-memory cache.

    Creates the config file with defaults when it does not exist.
    Must be called with ``_lock`` held by the caller.

    Returns:
        List of :class:`AutonomyPolicy` objects from the config file.
    """
    global _policies_cache

    if _policies_cache is not None:
        return _policies_cache

    config_path = _CONFIG_PATH
    if not config_path.exists():
        logger.info("autonomy_policies.yaml not found — creating with defaults at %s", config_path)
        _policies_cache = list(DEFAULT_POLICIES)
        _save_policies_unlocked(_policies_cache)
        return _policies_cache

    try:
        raw_text = config_path.read_text(encoding="utf-8")
        raw_data = yaml.safe_load(raw_text)
        if raw_data is None:
            raw_data = {}
        if not isinstance(raw_data, dict):
            raise ValueError("top-level YAML document must be an object")
        entries = raw_data.get("policies") or []
        if not isinstance(entries, list):
            raise ValueError("'policies' must be a list")
        _policies_cache = [_policy_from_dict(e) for e in entries if isinstance(e, dict)]
        if len(_policies_cache) != len(entries):
            raise ValueError("every policy entry must be an object")
        logger.debug("Loaded %d autonomy policies from %s", len(_policies_cache), config_path)
    except Exception as exc:
        logger.warning("Could not load autonomy_policies.yaml; refusing fallback defaults: %s", exc)
        _policies_cache = None
        raise RuntimeError(f"Could not load autonomy policies from {config_path}") from exc

    return _policies_cache


def _save_policies_unlocked(policies: list[AutonomyPolicy]) -> None:
    """Write policies to the YAML config file.

    Must be called with ``_lock`` held by the caller. Does NOT update
    ``_policies_cache`` — the caller is responsible for that.

    Args:
        policies: The full list of policies to persist.

    Raises:
        OSError: If the config file cannot be written.
    """
    config_path = _CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"policies": [asdict(p) for p in policies]}
    tmp_path = config_path.with_name(f".{config_path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, default_flow_style=False, allow_unicode=True)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(config_path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
    logger.debug("Saved %d autonomy policies to %s", len(policies), config_path)


def _policy_to_dict(p: AutonomyPolicy) -> dict[str, str]:
    """Convert a policy dataclass to a JSON-safe dict.

    Args:
        p: The :class:`AutonomyPolicy` to convert.

    Returns:
        Dict with string values suitable for JSON serialization.
    """
    return asdict(p)


# -- Handler factory ----------------------------------------------------------


def create_autonomy_api_handlers() -> list[Any]:
    """Return Litestar route handler instances for the autonomy policy API.

    Returns an empty list when Litestar is not installed so the caller can
    safely call this in environments that only have Flask.

    Returns:
        List of Litestar route handler objects covering all autonomy endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.responses import litestar_error_response

    @get("/api/v1/autonomy/policies", guards=[admin_guard])
    async def list_policies() -> Any:
        """Return all autonomy policies as a JSON list.

        When the YAML config is malformed and defaults are used, the response
        includes a ``"warning"`` field indicating the fallback is active.

        Returns:
            Dict with ``policies`` list (one entry per action type),
            ``count`` integer, and optional ``warning`` when defaults are used.
        """
        try:
            with _lock:
                policies = _load_policies()
                result = [_policy_to_dict(p) for p in policies]
        except Exception:
            logger.warning(
                "Failed to list autonomy policies — autonomy system unavailable, returning 500",
                exc_info=True,
            )
            return litestar_error_response("Autonomy system unavailable", code=500)
        return {"count": len(result), "policies": result}

    @put("/api/v1/autonomy/policies/{action_type:str}", guards=[admin_guard])
    async def update_policy(action_type: str, data: dict[str, Any]) -> Response:
        """Update the autonomy level for one action type.

        Expects a JSON body with a ``level`` field (one of: suggest, propose,
        confirm, auto). An optional ``description`` field, if provided,
        also replaces the stored description.

        Args:
            action_type: URL-captured action type identifier.
            data: Request body with ``level`` and optional ``description``.

        Returns:
            Dict with the updated policy and ``status: "updated"``.
            Returns 400 on validation failure, 404 when the action type
            does not exist, 500 when the autonomy system is unavailable.
        """
        level = str(data.get("level") or "").strip().lower()

        if not level:
            return litestar_error_response("Request body must include a 'level' field", code=400)
        if level not in VALID_LEVELS:
            return litestar_error_response(
                f"Invalid level '{level}' — must be one of: {', '.join(sorted(VALID_LEVELS))}",
                code=400,
            )

        new_description_raw = data.get("description")
        if new_description_raw is not None and not isinstance(new_description_raw, str):
            return litestar_error_response("'description' must be a string", code=422)
        new_description: str | None = new_description_raw

        try:
            with _lock:
                policies = _load_policies()

                updated_policy: AutonomyPolicy | None = None
                updated_list: list[AutonomyPolicy] = []
                for p in policies:
                    if p.action_type == action_type:
                        new_p = AutonomyPolicy(
                            action_type=p.action_type,
                            level=level,
                            description=(new_description if new_description is not None else p.description),
                            updated_at=datetime.now(timezone.utc).isoformat(),
                        )
                        updated_list.append(new_p)
                        updated_policy = new_p
                    else:
                        updated_list.append(p)

                if updated_policy is None:
                    return litestar_error_response(f"Action type '{action_type}' not found in policies", code=404)

                _save_policies_unlocked(updated_list)
                global _policies_cache
                _policies_cache = updated_list

                logger.info("Autonomy policy updated: action_type=%s level=%s", action_type, level)
        except Exception:
            logger.warning(
                "Failed to update autonomy policy for %r — autonomy system unavailable, returning 500",
                action_type,
                exc_info=True,
            )
            return litestar_error_response("Autonomy system unavailable", code=500)

        return Response(
            content={"status": "updated", "policy": _policy_to_dict(updated_policy)},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/v1/autonomy/status", guards=[admin_guard])
    async def autonomy_status() -> Any:
        """Return a summary of how many policies are set to each level.

        Returns:
            Dict with ``summary`` dict (level → count) and ``total`` integer.
            Returns 500 when the autonomy system is unavailable.
        """
        try:
            with _lock:
                policies = _load_policies()
        except Exception:
            logger.warning(
                "Failed to get autonomy status — autonomy system unavailable, returning 500",
                exc_info=True,
            )
            return litestar_error_response("Autonomy system unavailable", code=500)

        summary: dict[str, int] = dict.fromkeys(sorted(VALID_LEVELS), 0)
        for p in policies:
            if p.level in summary:
                summary[p.level] += 1
            else:
                # Defensive: handle any unexpected level stored in the file
                summary[p.level] = 1

        return {"total": len(policies), "summary": summary}

    @get("/api/v1/autonomy/promotions/pending", guards=[admin_guard])
    async def list_pending_promotions() -> Any:
        """Return all pending auto-promotions together with their veto deadlines.

        Each entry represents an auto-promotion proposed by the trust engine but
        not yet vetoed or expired.

        Returns:
            Dict with ``count`` and ``promotions`` list. Each promotion entry
            contains ``action_type``, ``current_level``, ``new_level``,
            ``proposed_at``, and ``veto_deadline``.
            Returns 500 when the governor is unavailable.
        """
        try:
            from vetinari.autonomy.governor import get_governor

            governor = get_governor()
            pending = governor.get_pending_promotions()
            promotions = [
                {
                    "action_type": p.action_type,
                    "current_level": p.current_level.value,
                    "new_level": p.new_level.value,
                    "proposed_at": p.proposed_at,
                    "veto_deadline": p.veto_deadline,
                }
                for p in pending.values()
            ]
        except Exception:
            logger.warning(
                "Failed to list pending promotions — autonomy system unavailable, returning 500",
                exc_info=True,
            )
            return litestar_error_response("Autonomy system unavailable", code=500)

        return {"count": len(promotions), "promotions": promotions}

    @post("/api/v1/autonomy/promotions/{action_type:str}/veto", guards=[admin_guard])
    async def veto_promotion(action_type: str, request: Request) -> Response:
        """Veto a pending auto-promotion before its deadline expires.

        Delegates to the governor's ``veto_promotion()`` method which removes
        the pending entry and resets trust counters.

        Args:
            action_type: URL-captured action type whose promotion to veto.
            request: Raw Litestar request — used to detect and reject any
                unexpected request body (this endpoint takes no input body).

        Returns:
            Dict with ``status: "vetoed"`` and ``action_type`` on success.
            Returns 400 if any request body is present (including null/empty JSON).
            Returns 404 if no pending promotion exists for the action type.
            Returns 500 when the governor is unavailable.
        """
        import re as _re

        if not _re.fullmatch(r"[A-Za-z0-9_-]{1,128}", action_type):
            return litestar_error_response(
                "action_type must contain only letters, digits, underscores, or hyphens (max 128 chars)",
                code=400,
            )

        # This endpoint acts on the URL path parameter only — no body is expected.
        # Check raw bytes so that null, empty JSON, and truncated payloads are all
        # rejected consistently, regardless of how Litestar parses the Content-Type.
        raw_body = await request.body()
        if raw_body.strip():
            return litestar_error_response(
                "This endpoint does not accept a request body",
                code=400,
            )
        try:
            from vetinari.autonomy.governor import get_governor

            governor = get_governor()
            vetoed = governor.veto_promotion(action_type)
        except Exception:
            logger.warning(
                "Failed to veto promotion for %r — autonomy system unavailable, returning 500",
                action_type,
                exc_info=True,
            )
            return litestar_error_response("Autonomy system unavailable", code=500)

        if not vetoed:
            return litestar_error_response(f"No pending promotion found for action type '{action_type}'", code=404)
        logger.info("Promotion vetoed via API: action_type=%s", action_type)
        return Response(
            content={"status": "vetoed", "action_type": action_type},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @post("/api/v1/undo/{action_id:str}", guards=[admin_guard])
    async def undo_action(action_id: str) -> Response:
        """Manually undo (rollback) a specific autonomous action by its ID.

        Looks up the action in the rollback registry and marks it as rolled back.
        An action can only be undone once — subsequent calls return 404.

        Args:
            action_id: URL-captured unique action identifier
                (``undo_<12-hex>`` format).

        Returns:
            Dict with ``status: "rolled_back"`` and ``action_id`` on success.
            Returns 404 if the action is not found in the registry.
            Returns 500 when the rollback registry is unavailable.
        """
        try:
            from vetinari.autonomy.rollback import get_rollback_registry

            registry = get_rollback_registry()
            result = registry.undo_action(action_id)
        except Exception:
            logger.warning(
                "Failed to undo action %r — autonomy system unavailable, returning 500",
                action_id,
                exc_info=True,
            )
            return litestar_error_response("Autonomy system unavailable", code=500)

        if result is None:
            return litestar_error_response(f"No rollback entry found for action_id '{action_id}'", code=404)
        logger.info("Action rolled back via API: action_id=%s", action_id)
        return Response(
            content={"status": "rolled_back", "action_id": action_id},
            status_code=200,
            media_type=MediaType.JSON,
        )

    @get("/api/v1/autonomy/rollback/history", guards=[admin_guard])
    async def rollback_history() -> Any:
        """Return the rollback history for the last 24 hours.

        Lists autonomous actions that were manually or automatically rolled back,
        ordered most-recent first.

        Returns:
            Dict with ``count`` and ``rollbacks`` list of action records with
            ``action_id``, ``action_type``, ``timestamp``, and ``quality_before``.
            Returns 500 when the rollback registry is unavailable.
        """
        try:
            from vetinari.autonomy.rollback import get_rollback_registry

            registry = get_rollback_registry()
            recent = registry.get_recent_actions(hours=24)
            # Build explicit dicts excluding reversible_data — that field may contain
            # sensitive state (pre-rollback config, credentials, internal IDs) that
            # must never be surfaced through the public API.
            rolled_back = [
                {
                    "action_id": a.action_id,
                    "action_type": a.action_type,
                    "timestamp": a.timestamp,
                    "quality_before": a.quality_before,
                    "quality_after": a.quality_after,
                    "rolled_back": a.rolled_back,
                }
                for a in recent
                if a.rolled_back
            ]
            rolled_back.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        except Exception:
            logger.warning(
                "Failed to get rollback history — autonomy system unavailable, returning 500",
                exc_info=True,
            )
            return litestar_error_response("Autonomy system unavailable", code=500)

        return {"count": len(rolled_back), "rollbacks": rolled_back}

    @get("/api/v1/autonomy/trust", guards=[admin_guard])
    async def trust_status() -> Any:
        """Return trust engine status including pending promotions and global mode.

        Combines per-action-type trust metrics, pending-promotion count, and
        the current global autonomy mode into one response.

        Returns:
            Dict with ``trust`` (per-action-type metrics), ``pending_promotions``
            (count), and ``autonomy_mode`` (string).
            Returns 500 when the governor is unavailable.
        """
        try:
            from vetinari.autonomy.governor import get_governor

            governor = get_governor()
            trust_data = governor.get_trust_status()
            pending_count = len(governor.get_pending_promotions())
            mode = governor.get_autonomy_mode()
        except Exception:
            logger.warning(
                "Failed to get trust status — autonomy system unavailable, returning 500",
                exc_info=True,
            )
            return litestar_error_response("Autonomy system unavailable", code=500)

        return {
            "trust": trust_data,
            "pending_promotions": pending_count,
            "autonomy_mode": mode.value,
        }

    return [
        list_policies,
        update_policy,
        autonomy_status,
        list_pending_promotions,
        veto_promotion,
        undo_action,
        rollback_history,
        trust_status,
    ]
