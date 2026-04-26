"""Replay recording hooks — bridges live inference/tool calls into ReplayRecorder.

This module provides thin integration hooks that capture inference calls and
tool invocations as structured JSONL events when replay recording is enabled.
Recording is off by default and activated only when the environment variable
``VETINARI_REPLAY_RECORD=1`` is set, so production and test runs are not
affected unless explicitly opted in.

Session lifecycle
-----------------
A recording session is started automatically on the first recorded call if no
session is already active. The session name is derived from the current UTC
date so recordings from the same day are grouped together.  Callers that want
explicit control can call ``start_session()`` / ``stop_session()`` directly.

Thread safety
-------------
``ReplayRecorder`` is already thread-safe (internal lock). The module-level
session state (``_recorder``, ``_session_active``) is guarded by
``_session_lock``.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.replay.recorder import ReplayRecorder

logger = logging.getLogger(__name__)

# -- Feature flag ---------------------------------------------------------------
# Recording is disabled by default. Set VETINARI_REPLAY_RECORD=1 to enable.
# Checked once at import time so the hot path never calls os.environ.get().
REPLAY_RECORDING_ENABLED: bool = os.environ.get("VETINARI_REPLAY_RECORD", "0").strip() in ("1", "true", "yes")

# -- Session state (guarded by _session_lock) -----------------------------------
# Written by: start_session(), _ensure_session_active()
# Read by:    record_inference(), record_tool_call(), stop_session()
_recorder: ReplayRecorder | None = None
_session_active: bool = False
_session_lock = threading.Lock()

# Default output directory for session JSONL files
_RECORDINGS_DIR = Path(__file__).parent / "recordings"


# -- Public API ----------------------------------------------------------------


def start_session(scenario_name: str | None = None, output_dir: Path | None = None) -> None:
    """Begin a named replay recording session.

    Safe to call when ``REPLAY_RECORDING_ENABLED`` is False — returns
    immediately without side effects so callers need no guard.

    Args:
        scenario_name: Human-readable label for this session. Defaults to
            ``session_<UTC-date>_<UTC-time>`` if not provided.
        output_dir: Directory for the output JSONL file. Defaults to
            ``tests/replay/recordings/``.

    Raises:
        RuntimeError: If a session is already active.
    """
    if not REPLAY_RECORDING_ENABLED:
        return

    global _recorder, _session_active
    _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _name = scenario_name or f"session_{_ts}"
    _dir = output_dir or _RECORDINGS_DIR
    _dir.mkdir(parents=True, exist_ok=True)

    with _session_lock:
        if _session_active:
            raise RuntimeError("Replay session already active. Call stop_session() before starting a new one.")
        _recorder = ReplayRecorder(output_dir=_dir)
        _recorder.start_recording(_name)
        _session_active = True

    logger.debug("Replay session started: %s -> %s", _name, _dir)


def stop_session() -> Path | None:
    """Flush the active recording session to a JSONL file.

    Safe to call when ``REPLAY_RECORDING_ENABLED`` is False or when no
    session is active — returns ``None`` in both cases.

    Returns:
        Absolute path to the written JSONL file, or ``None`` if recording
        was not active.
    """
    if not REPLAY_RECORDING_ENABLED:
        return None

    global _recorder, _session_active
    with _session_lock:
        if not _session_active or _recorder is None:
            return None
        rec = _recorder
        _session_active = False
        _recorder = None

    path = rec.stop_recording()
    logger.debug("Replay session stopped; JSONL written to %s", path)
    return path


def record_inference(
    request: Any,
    response: Any,
    latency_ms: int = 0,
) -> None:
    """Record a single LLM inference call for replay.

    Extracts prompt, response text, and model ID from the adapter request
    and response objects. No-ops when recording is disabled or no session
    is active.

    Args:
        request: An ``InferenceRequest``-compatible object with ``prompt``
            and ``model_id`` attributes.
        response: An ``InferenceResponse``-compatible object with ``output``
            and ``model_id`` attributes.
        latency_ms: Measured inference latency in milliseconds.
    """
    if not REPLAY_RECORDING_ENABLED:
        return

    _ensure_session_active()

    with _session_lock:
        rec = _recorder

    if rec is None:
        return

    try:
        prompt = getattr(request, "prompt", "") or ""
        model_id = getattr(response, "model_id", None) or getattr(request, "model_id", "unknown")
        output = getattr(response, "output", "") or ""
        rec.record_llm_call(
            prompt=prompt,
            response=output,
            model_id=str(model_id),
            latency_ms=latency_ms,
            metadata={
                "agent": getattr(request, "metadata", {}).get("agent", "unknown"),
                "tokens_used": getattr(response, "tokens_used", 0),
            },
        )
    except Exception as exc:
        logger.warning(
            "Replay hooks: failed to record inference call for model %s — non-fatal: %s",
            getattr(request, "model_id", "unknown"),
            exc,
        )


def record_tool_call(
    tool_name: str,
    input_data: Any,
    output_data: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a single tool invocation for replay.

    No-ops when recording is disabled or no session is active.

    Args:
        tool_name: Name of the tool that was called.
        input_data: Input passed to the tool (should be JSON-serializable).
        output_data: Output returned by the tool (should be JSON-serializable).
        metadata: Optional extra key-value pairs to store with the event.
    """
    if not REPLAY_RECORDING_ENABLED:
        return

    _ensure_session_active()

    with _session_lock:
        rec = _recorder

    if rec is None:
        return

    try:
        rec.record_tool_call(
            tool_name=tool_name,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning(
            "Replay hooks: failed to record tool call %r — non-fatal: %s",
            tool_name,
            exc,
        )


def is_recording() -> bool:
    """Return True if a replay session is currently active.

    Returns:
        True when ``REPLAY_RECORDING_ENABLED`` is set and a session has
        been started but not yet stopped.
    """
    if not REPLAY_RECORDING_ENABLED:
        return False
    with _session_lock:
        return _session_active


# -- Internal helpers ----------------------------------------------------------


def _ensure_session_active() -> None:
    """Start a default session if recording is enabled but no session is active.

    Called at the start of every ``record_*`` function so callers that set
    ``VETINARI_REPLAY_RECORD=1`` without explicitly calling ``start_session()``
    still get their calls captured.
    """
    global _recorder, _session_active
    with _session_lock:
        if _session_active:
            return
        _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        _name = f"auto_session_{_ts}"
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        _recorder = ReplayRecorder(output_dir=_RECORDINGS_DIR)
        _recorder.start_recording(_name)
        _session_active = True
        logger.debug("Replay hooks: auto-started session %r", _name)
