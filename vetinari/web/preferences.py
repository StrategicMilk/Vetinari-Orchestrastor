"""User preferences API for Vetinari UI customization."""

import json
import logging
import os
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
    prefs = mgr.update(updates)
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
