"""Server-side user preferences persistence.

Replaces client-only localStorage with a JSON file store that syncs
to the browser on load and accepts updates via REST API.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from vetinari.constants import _PROJECT_ROOT

logger = logging.getLogger(__name__)

_PREFS_PATH = _PROJECT_ROOT / ".vetinari" / "user_preferences.json"

# Whitelist of allowed preference keys (prevents arbitrary data injection)
ALLOWED_KEYS = frozenset({
    # Existing UI state
    "sidebarCollapsed",
    "reducedMotion",
    "compactMode",
    "theme",
    # Setup wizard
    "setupComplete",
    # Permissions
    "autonomyLevel",  # supervised | assisted | autonomous
    "allowModelDownload",  # ask | auto | deny
    "allowTrainingStart",  # ask | auto | deny
    "allowProjectExecute",  # ask | auto | deny
    "allowFileWrite",  # ask | auto | deny
    "allowDataCollection",  # ask | auto | deny
    # Notifications
    "notificationPreferences",
    "notificationDuration",  # int (ms), default 8000
    "notificationSound",  # bool
    # Appearance
    "interfaceMode",  # simple | standard | expert
    "accentColor",  # hex string
    "fontSize",  # int (px)
    "chatBubbleStyle",  # bubbles | flat | cards
    # Model paths
    "vetinari_image_models_dir",
    # User content
    "customInstructions",  # user-supplied global system prompt / instructions
    "defaultSystemPrompt",  # selected system prompt name
})

# Default values matching current localStorage defaults
DEFAULTS: dict[str, Any] = {
    "sidebarCollapsed": False,
    "reducedMotion": False,
    "compactMode": False,
    "theme": "dark",
    "setupComplete": False,
    "autonomyLevel": "assisted",
    "notificationPreferences": "all",
    "allowModelDownload": "ask",
    "allowTrainingStart": "ask",
    "allowProjectExecute": "auto",
    "allowFileWrite": "ask",
    "allowDataCollection": "auto",
    "notificationDuration": 8000,
    "notificationSound": False,
    "interfaceMode": "standard",
    "accentColor": "#4e9af9",
    "fontSize": 14,
    "chatBubbleStyle": "flat",
    "vetinari_image_models_dir": "",
}


class PreferencesManager:
    """Manages server-side user preferences with JSON file persistence."""

    def __init__(self, path: Path | None = None):
        """Load preferences from disk, initialising from defaults when the file is absent.

        Args:
            path: Path to the JSON preferences file.  Defaults to
                ``_PREFS_PATH`` (``.vetinari/user_preferences.json``) when
                ``None``.
        """
        self._path = path or _PREFS_PATH
        self._prefs: dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load preferences from disk, falling back to defaults."""
        try:
            if self._path.exists():
                with Path(self._path).open(encoding="utf-8") as f:
                    saved = json.load(f)
                # Only load allowed keys
                self._prefs = {k: saved[k] for k in saved if k in ALLOWED_KEYS}
                logger.debug("Loaded %d preferences from %s", len(self._prefs), self._path)
            else:
                self._prefs = {}
        except Exception as e:
            logger.warning("Failed to load preferences: %s", e)
            self._prefs = {}

    def _save(self):
        """Persist preferences to disk atomically.

        Raises:
            OSError: If the preference file cannot be written and replaced.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_name(f".{self._path.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(self._prefs, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self._path)
        except Exception:
            logger.warning("Failed to save preferences to %s", self._path, exc_info=True)
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise

    def get_all(self) -> dict[str, Any]:
        """Get all preferences with defaults applied.

        Returns:
            Dictionary mapping preference keys to their current values,
            with defaults filled in for any keys not explicitly set.
        """
        result = dict(DEFAULTS)
        result.update(self._prefs)
        return result

    def get(self, key: str) -> Any:
        """Get a single preference value."""
        return self._prefs.get(key, DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> bool:
        """Set a single preference, persisting immediately to disk.

        Args:
            key: Preference key (must be in ALLOWED_KEYS).
            value: New value for the preference.

        Returns:
            True if the key is allowed and the value was saved,
            False if the key is not in the whitelist.
        """
        if key not in ALLOWED_KEYS:
            return False
        self._prefs[key] = value
        self._save()
        return True

    def set_many(self, updates: dict[str, Any]) -> dict[str, bool]:
        """Set multiple preferences at once, persisting all changes in a single write.

        Args:
            updates: Dictionary mapping preference keys to their new values.

        Returns:
            Dictionary mapping each key to True if it was accepted (in ALLOWED_KEYS)
            or False if it was rejected.
        """
        results = {}
        changed = False
        for key, value in updates.items():
            if key in ALLOWED_KEYS:
                self._prefs[key] = value
                results[key] = True
                changed = True
            else:
                results[key] = False
        if changed:
            self._save()
        return results

    def reset(self, key: str | None = None) -> None:
        """Reset one or all preferences to defaults."""
        if key:
            self._prefs.pop(key, None)
        else:
            self._prefs.clear()
        self._save()


_manager: PreferencesManager | None = None
_manager_lock = threading.Lock()


def get_preferences_manager() -> PreferencesManager:
    """Get or create the global singleton preferences manager.

    Returns:
        The shared PreferencesManager instance, creating one on first call.
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = PreferencesManager()
    return _manager


def reset_preferences_manager() -> None:
    """Reset the global preferences manager (for testing)."""
    global _manager
    _manager = None
