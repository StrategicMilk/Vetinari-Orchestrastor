"""Server-side user preferences persistence.

Replaces client-only localStorage with a JSON file store that syncs
to the browser on load and accepts updates via REST API.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PREFS_PATH = Path(".vetinari/user_preferences.json")

# Whitelist of allowed preference keys (prevents arbitrary data injection)
ALLOWED_KEYS = frozenset({
    "sidebarCollapsed",
    "reducedMotion",
    "compactMode",
    "theme",
    "onboardingComplete",
    "vetinari_sd_host",
})

# Default values matching current localStorage defaults
DEFAULTS: Dict[str, Any] = {
    "sidebarCollapsed": False,
    "reducedMotion": False,
    "compactMode": False,
    "theme": "dark",
    "onboardingComplete": False,
    "vetinari_sd_host": "",
}


class PreferencesManager:
    """Manages server-side user preferences with JSON file persistence."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path or _PREFS_PATH
        self._prefs: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load preferences from disk, falling back to defaults."""
        try:
            if self._path.exists():
                with open(self._path) as f:
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
        """Persist preferences to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._prefs, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save preferences: %s", e)

    def get_all(self) -> Dict[str, Any]:
        """Get all preferences with defaults applied."""
        result = dict(DEFAULTS)
        result.update(self._prefs)
        return result

    def get(self, key: str) -> Any:
        """Get a single preference value."""
        return self._prefs.get(key, DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> bool:
        """Set a preference. Returns False if key not allowed."""
        if key not in ALLOWED_KEYS:
            return False
        self._prefs[key] = value
        self._save()
        return True

    def set_many(self, updates: Dict[str, Any]) -> Dict[str, bool]:
        """Set multiple preferences. Returns per-key success map."""
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

    def reset(self, key: Optional[str] = None):
        """Reset one or all preferences to defaults."""
        if key:
            self._prefs.pop(key, None)
        else:
            self._prefs.clear()
        self._save()


_manager: Optional[PreferencesManager] = None


def get_preferences_manager() -> PreferencesManager:
    """Get or create the global preferences manager."""
    global _manager
    if _manager is None:
        _manager = PreferencesManager()
    return _manager


def reset_preferences_manager() -> None:
    """Reset the global preferences manager (for testing)."""
    global _manager
    _manager = None
