"""Module with config cached at module level — clean for VET115."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Loaded once at startup — route handlers read _CONFIG, never the file directly.
_CONFIG: dict = json.loads(Path("config.json").read_text(encoding="utf-8")) if Path("config.json").exists() else {}


class DataApi:
    """Data access endpoints."""

    @app.route("/api/config")
    def get_config(self) -> dict:
        """Return the cached configuration.

        Config is loaded at module import time, not per-request, so repeated
        calls have zero disk I/O cost.

        Returns:
            Cached configuration dictionary.
        """
        return _CONFIG
