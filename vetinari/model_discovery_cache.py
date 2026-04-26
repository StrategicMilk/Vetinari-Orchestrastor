"""Model discovery cache helpers for search results and download state.

These functions keep transient search caches and persisted download job state
small, typed, and reusable behind the ``vetinari.model_discovery`` facade.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from vetinari.model_discovery_types import ModelCandidate

logger = logging.getLogger(__name__)

_CACHE_TTL_DAYS = 7  # Days before cached search results expire.


def _load_from_cache(cache_file: Path) -> list[ModelCandidate] | None:
    """Return cached candidates if file exists and is fresh, else None."""
    if not cache_file.exists():
        return None
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
    if age >= timedelta(days=_CACHE_TTL_DAYS):
        return None
    try:
        with Path(cache_file).open(encoding="utf-8") as f:
            data = json.load(f)
        return [ModelCandidate(**c) for c in data]
    except Exception as e:
        logger.warning("Cache load failed for %s: %s", cache_file, e)
        return None


def _save_to_cache(cache_file: Path, candidates: list[ModelCandidate]) -> None:
    try:
        with Path(cache_file).open("w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in candidates], f)
    except Exception as e:
        logger.warning("Cache save failed for %s: %s", cache_file, e)


def _public_download_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if not key.startswith("_")}


def _load_download_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Download state file %s is unreadable; treating it as empty", path)
        return {}
    if not isinstance(data, dict):
        return {}
    jobs = data.get("jobs", data)
    if not isinstance(jobs, dict):
        return {}
    return {
        str(job_id): dict(job)
        for job_id, job in jobs.items()
        if isinstance(job_id, str) and isinstance(job, dict)
    }


def _write_download_state(path: Path, jobs: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {"version": 1, "jobs": jobs}
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)
