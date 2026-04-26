"""Skill Catalog Loader.

Discovers and parses all SKILL.md files from the catalog directory,
extracting YAML frontmatter into typed CatalogEntry dataclasses.
Provides lookup by agent, mode, capability, and tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vetinari.utils.frontmatter import parse_frontmatter
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)

# Catalog directory relative to this file
_CATALOG_ROOT = Path(__file__).parent / "catalog"

# Module-level lazy cache — populated on first access
_CATALOG: dict[str, CatalogEntry] | None = None


@dataclass(frozen=True)
class CatalogEntry:
    """A single skill catalog entry parsed from a SKILL.md file."""

    skill_id: str  # "{agent}/{skill-name}" e.g. "worker/feature-implementation"
    name: str
    description: str
    mode: str
    agent: str  # foreman, worker, inspector
    version: str
    capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    file_path: str = ""

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"CatalogEntry(skill_id={self.skill_id!r}, agent={self.agent!r}, mode={self.mode!r})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary for API responses and JSON export."""
        return dataclass_to_dict(self)


def load_catalog(catalog_root: Path | None = None) -> dict[str, CatalogEntry]:
    """Walk the catalog directory and load all SKILL.md files.

    Each SKILL.md file's YAML frontmatter is extracted and converted into a
    :class:`CatalogEntry`. The ``skill_id`` is derived from the path as
    ``"{agent}/{skill-name}"``.

    Args:
        catalog_root: Root directory to search. Defaults to the bundled
            ``catalog/`` directory next to this module.

    Returns:
        Mapping from ``skill_id`` to :class:`CatalogEntry` for every
        successfully parsed SKILL.md file.
    """
    root = catalog_root if catalog_root is not None else _CATALOG_ROOT
    catalog: dict[str, CatalogEntry] = {}

    if not root.is_dir():
        logger.warning("Catalog root does not exist: %s", root)
        return catalog

    for skill_md in root.rglob("SKILL.md"):
        # Derive skill_id from the two path components above SKILL.md:
        # catalog/{agent}/{skill-name}/SKILL.md  →  "{agent}/{skill-name}"
        parts = skill_md.relative_to(root).parts
        if len(parts) < 3:
            logger.warning("Unexpected SKILL.md path structure: %s; skipping", skill_md)
            continue

        agent_dir, skill_dir = parts[0], parts[1]
        skill_id = f"{agent_dir}/{skill_dir}"

        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s: %s", skill_md, exc)
            continue

        frontmatter, _ = parse_frontmatter(content)
        if not frontmatter:
            logger.warning("Skipping %s — empty or unparseable frontmatter", skill_md)
            continue

        entry = CatalogEntry(
            skill_id=skill_id,
            name=frontmatter.get("name", skill_dir),
            description=frontmatter.get("description", ""),
            mode=frontmatter.get("mode", ""),
            agent=frontmatter.get("agent", agent_dir),
            version=str(frontmatter.get("version", "1.0.0")),
            capabilities=list(frontmatter.get("capabilities") or []),
            tags=list(frontmatter.get("tags") or []),
            file_path=str(skill_md),
        )
        catalog[skill_id] = entry

    logger.info("Loaded %d skill catalog entries from %s", len(catalog), root)
    return catalog


def _ensure_loaded() -> dict[str, CatalogEntry]:
    """Return the module-level catalog, loading it on first access.

    Returns:
        The populated catalog mapping.
    """
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = load_catalog()
    return _CATALOG


def get_catalog_by_agent(
    agent: str,
    catalog: dict[str, CatalogEntry] | None = None,
) -> list[CatalogEntry]:
    """Return all catalog entries for a specific agent.

    Args:
        agent: Agent name to filter by (e.g. ``"worker"``, ``"foreman"``,
            ``"inspector"``).
        catalog: Optional pre-loaded catalog mapping. When ``None`` the
            module-level cache is used, loading it if necessary.

    Returns:
        List of :class:`CatalogEntry` objects whose ``agent`` field matches
        the given name. Empty list if none match.
    """
    entries = catalog if catalog is not None else _ensure_loaded()
    return [e for e in entries.values() if e.agent == agent]


def get_catalog_by_capability(
    capability: str,
    catalog: dict[str, CatalogEntry] | None = None,
) -> list[CatalogEntry]:
    """Return all catalog entries that declare a specific capability.

    Args:
        capability: Capability string to search for (e.g.
            ``"feature_implementation"``).
        catalog: Optional pre-loaded catalog mapping. When ``None`` the
            module-level cache is used, loading it if necessary.

    Returns:
        List of :class:`CatalogEntry` objects whose ``capabilities`` list
        contains the given capability. Empty list if none match.
    """
    entries = catalog if catalog is not None else _ensure_loaded()
    return [e for e in entries.values() if capability in e.capabilities]


def get_catalog_by_tag(
    tag: str,
    catalog: dict[str, CatalogEntry] | None = None,
) -> list[CatalogEntry]:
    """Return all catalog entries that carry a specific tag.

    Args:
        tag: Tag string to search for (e.g. ``"build"``).
        catalog: Optional pre-loaded catalog mapping. When ``None`` the
            module-level cache is used, loading it if necessary.

    Returns:
        List of :class:`CatalogEntry` objects whose ``tags`` list contains
        the given tag. Empty list if none match.
    """
    entries = catalog if catalog is not None else _ensure_loaded()
    return [e for e in entries.values() if tag in e.tags]
