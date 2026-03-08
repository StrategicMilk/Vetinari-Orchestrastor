"""User preferences API for Vetinari UI customization."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

preferences_bp = Blueprint("preferences", __name__)


@dataclass
class UserPreferences:
    """User preferences for UI customization."""

    agent_nicknames: Dict[str, str] = field(
        default_factory=lambda: {
            "PLANNER": "Vetinari",
            "RESEARCHER": "Ponder",
            "ARCHITECT": "Ridcully",
            "BUILDER": "Igor",
            "TESTER": "Vimes",
            "DOCUMENTER": "Carrot",
            "RESILIENCE": "Rincewind",
            "META": "Death",
        }
    )
    agent_icons: Dict[str, str] = field(default_factory=dict)
    theme: str = "dark"
    compact_mode: bool = False
    suggestion_frequency: str = "normal"  # "off", "low", "normal", "high"
    variant_level: str = "medium"  # "low", "medium", "high"
    show_learning_dashboard: bool = True
    show_agent_status: bool = True
    auto_approve_milestones: bool = False
    # ── Sampling parameter overrides ────────────────────────────────────────
    # Users can override the learned/default sampling profiles per task_type
    # or per model_id. Keys are "task:<task_type>" or "model:<model_id>".
    # Values are dicts of sampling params: temperature, top_p, top_k, min_p,
    # repeat_penalty, presence_penalty, frequency_penalty.
    # Example: {"task:coding": {"temperature": 0.2, "top_p": 0.9},
    #           "model:qwen3-30b-a3b": {"temperature": 0.7}}
    # These overrides take highest priority, above learned and default profiles.
    sampling_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class PreferencesManager:
    """Manages user preferences persistence."""

    DEFAULT_PATH = ".vetinari/user_preferences.json"

    def __init__(self, path: str = None):
        self._path = Path(path or self.DEFAULT_PATH)
        self._preferences: Optional[UserPreferences] = None

    def load(self) -> UserPreferences:
        """Load preferences from disk, or return defaults."""
        if self._preferences is not None:
            return self._preferences
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._preferences = UserPreferences(
                    **{
                        k: v
                        for k, v in data.items()
                        if k in UserPreferences.__dataclass_fields__
                    }
                )
            except Exception as e:
                logger.warning("Failed to load preferences: %s", e)
                self._preferences = UserPreferences()
        else:
            self._preferences = UserPreferences()
        return self._preferences

    def save(self, prefs: UserPreferences) -> None:
        """Save preferences to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(asdict(prefs), indent=2))
        self._preferences = prefs

    def update(self, updates: dict) -> UserPreferences:
        """Merge updates into current preferences."""
        prefs = self.load()
        for key, value in updates.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        self.save(prefs)
        return prefs

    def get_nickname(self, agent_type: str) -> str:
        """Get display name for an agent type."""
        prefs = self.load()
        return prefs.agent_nicknames.get(agent_type, agent_type)

    def get_sampling_override(
        self, task_type: str = None, model_id: str = None
    ) -> Dict[str, Any]:
        """Look up user sampling overrides for a task type and/or model.

        Resolution order (later wins):
          1. task:<task_type>  overrides
          2. model:<model_id> overrides

        Returns a merged dict of sampling parameters, or empty dict.
        """
        prefs = self.load()
        merged: Dict[str, Any] = {}
        if task_type:
            merged.update(prefs.sampling_overrides.get(f"task:{task_type}", {}))
        if model_id:
            merged.update(prefs.sampling_overrides.get(f"model:{model_id}", {}))
        return merged

    def reset(self) -> UserPreferences:
        """Reset to defaults."""
        self._preferences = UserPreferences()
        self.save(self._preferences)
        return self._preferences


# Singleton
_manager: Optional[PreferencesManager] = None


def get_preferences_manager(path: str = None) -> PreferencesManager:
    """Get or create the singleton PreferencesManager."""
    global _manager
    if _manager is None:
        _manager = PreferencesManager(path)
    return _manager


def reset_preferences_manager() -> None:
    """Reset the singleton (useful for testing)."""
    global _manager
    _manager = None


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@preferences_bp.route("/api/preferences", methods=["GET"])
def get_preferences():
    """Return current user preferences."""
    mgr = get_preferences_manager()
    return jsonify(asdict(mgr.load()))


@preferences_bp.route("/api/preferences", methods=["POST"])
def update_preferences():
    """Update user preferences with partial data."""
    mgr = get_preferences_manager()
    updates = request.get_json(silent=True) or {}
    if not isinstance(updates, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    # Only accept keys that exist in the UserPreferences dataclass
    allowed_keys = set(UserPreferences.__dataclass_fields__.keys())
    filtered = {k: v for k, v in updates.items() if k in allowed_keys}
    prefs = mgr.update(filtered)
    return jsonify(asdict(prefs))


@preferences_bp.route("/api/preferences/reset", methods=["POST"])
def reset_preferences():
    """Reset preferences to defaults."""
    mgr = get_preferences_manager()
    prefs = mgr.reset()
    return jsonify(asdict(prefs))


@preferences_bp.route("/api/preferences/nickname/<agent_type>", methods=["GET"])
def get_nickname(agent_type):
    """Get the display nickname for an agent type."""
    mgr = get_preferences_manager()
    return jsonify(
        {"agent_type": agent_type, "nickname": mgr.get_nickname(agent_type)}
    )


@preferences_bp.route("/api/preferences/sampling", methods=["GET"])
def get_sampling_overrides():
    """Return current sampling parameter overrides.

    Overrides are keyed by 'task:<task_type>' or 'model:<model_id>'.
    Values are dicts of sampling parameters (temperature, top_p, etc.).
    These take highest priority over learned and default profiles.
    """
    mgr = get_preferences_manager()
    prefs = mgr.load()
    return jsonify(prefs.sampling_overrides)


@preferences_bp.route("/api/preferences/sampling", methods=["POST"])
def update_sampling_overrides():
    """Update sampling parameter overrides (merge).

    Accepts JSON body: {"task:coding": {"temperature": 0.2}, ...}
    Merges into existing overrides. Send null/None value to delete a key.
    """
    mgr = get_preferences_manager()
    prefs = mgr.load()
    updates = request.get_json(silent=True) or {}
    if not isinstance(updates, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    # Validate sampling override keys must be prefixed with task: or model:
    allowed_params = {"temperature", "top_p", "top_k", "min_p", "repeat_penalty",
                      "presence_penalty", "frequency_penalty", "seed", "max_tokens"}
    for key, value in updates.items():
        if not (key.startswith("task:") or key.startswith("model:")):
            return jsonify({"error": f"Invalid key '{key}': must start with 'task:' or 'model:'"}), 400
        if value is not None and not isinstance(value, dict):
            return jsonify({"error": f"Value for '{key}' must be a JSON object or null"}), 400
        if isinstance(value, dict):
            invalid_params = set(value.keys()) - allowed_params
            if invalid_params:
                return jsonify({"error": f"Invalid sampling params: {invalid_params}"}), 400
    for key, value in updates.items():
        if value is None:
            prefs.sampling_overrides.pop(key, None)
        else:
            prefs.sampling_overrides[key] = value
    mgr.save(prefs)
    return jsonify(prefs.sampling_overrides)


@preferences_bp.route("/api/preferences/sampling/<path:override_key>", methods=["DELETE"])
def delete_sampling_override(override_key):
    """Delete a specific sampling override by key (e.g. 'task:coding')."""
    mgr = get_preferences_manager()
    prefs = mgr.load()
    removed = prefs.sampling_overrides.pop(override_key, None)
    mgr.save(prefs)
    return jsonify({"deleted": override_key, "existed": removed is not None})
