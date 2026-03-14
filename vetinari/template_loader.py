"""Template loader for versioned agent prompt templates.

Loads JSON template files from the templates/ directory, organized by version
and agent type. Supports the 6 consolidated agents plus legacy agent names
for backward compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent / "templates"

# Maps current consolidated agent names to their template files.
# Legacy names are mapped to their consolidated replacements.
_AGENT_FILE_MAP: dict[str, str] = {
    # 6 consolidated agents (current architecture)
    "planner": "planner.json",
    "researcher": "researcher.json",
    "oracle": "oracle.json",
    "builder": "builder.json",
    "quality": "quality.json",
    "operations": "operations.json",
    # Legacy agent names → consolidated equivalents
    "explorer": "researcher.json",
    "librarian": "researcher.json",
    "evaluator": "quality.json",
    "synthesizer": "operations.json",
    "ui_planner": "planner.json",
}

# The 6 canonical agent types used when loading all templates.
_CONSOLIDATED_AGENTS = ("planner", "researcher", "oracle", "builder", "quality", "operations")


class TemplateLoader:
    """Load versioned prompt templates for Vetinari agents.

    Templates are stored as JSON files under templates/{version}/{agent}.json.
    A versions.json manifest lists available versions.
    """

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or BASE

    def list_versions(self) -> list[str]:
        """Return available template versions from the manifest.

        Returns:
            The result string.
        """
        manifest = self.base_path / "versions.json"
        if not manifest.exists():
            return ["v1"]
        try:
            with open(manifest, encoding="utf-8") as f:
                data = json.load(f)
                return data.get("versions", ["v1"])
        except Exception:
            logger.warning("Failed to read template versions manifest")
            return ["v1"]

    def load_templates_for_agent(self, version: str, agent_type: str) -> list[dict]:
        """Load templates for a specific agent type and version.

        Args:
            version: Template version string (e.g. "v1").
            agent_type: Agent name, supports both consolidated and legacy names.

        Returns:
            List of template dicts, or empty list if not found.
        """
        filename = _AGENT_FILE_MAP.get(agent_type)
        if not filename:
            return []
        path = self.base_path / version / filename
        if not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to load templates from %s", path)
            return []

    def load_templates(self, version: str | None = None, agent_type: str | None = None) -> list[dict]:
        """Load templates, optionally filtered by agent type.

        Args:
            version: Template version. Uses default if not specified.
            agent_type: If provided, load only this agent's templates.

        Returns:
            List of template dicts from matching agent files.
        """
        ver = version or self.default_version()
        if agent_type:
            return self.load_templates_for_agent(ver, agent_type)
        templates: list[dict] = []
        for atype in _CONSOLIDATED_AGENTS:
            templates.extend(self.load_templates_for_agent(ver, atype))
        return templates

    def default_version(self) -> str:
        """Return the first available version, defaulting to 'v1'.

        Returns:
            The result string.
        """
        versions = self.list_versions()
        return versions[0] if versions else "v1"


template_loader = TemplateLoader()
