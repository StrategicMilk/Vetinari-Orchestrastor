"""Architecture Decision Record (ADR) management system.

Provides a structured approach to documenting and tracking architecture
decisions across the Vetinari project.  Each ADR is stored as a JSON file
in the configured storage directory and loaded into memory at startup.

Usage::

    from vetinari.adr import ADRSystem

    system = ADRSystem.get_instance()
    adr = system.create_adr(
        title="Use six-agent pipeline",
        category="architecture",
        context="Need a structured agent hierarchy ...",
        decision="Adopt a six-agent pipeline: Planner -> Researcher -> ...",
    )
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from vetinari.security.sandbox import enforce_blocked_paths
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)


class ADRStatus(Enum):
    """Lifecycle status of an Architecture Decision Record."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class ADRCategory(Enum):
    """Classification categories for ADRs."""

    ARCHITECTURE = "architecture"
    SECURITY = "security"
    DATA_FLOW = "data_flow"
    API_DESIGN = "api_design"
    AGENT_DESIGN = "agent_design"
    DECOMPOSITION = "decomposition"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"


HIGH_STAKES_CATEGORIES: frozenset[ADRCategory] = frozenset({
    ADRCategory.ARCHITECTURE,
    ADRCategory.SECURITY,
    ADRCategory.DATA_FLOW,
})


@dataclass
class ADR:
    """A single Architecture Decision Record.

    Args:
        adr_id: Unique identifier (e.g. ``ADR-0001``).
        title: Short descriptive title for the decision.
        category: One of :class:`ADRCategory` values.
        context: Problem statement or background motivating the decision.
        decision: The decision that was made.
        status: Lifecycle status (default ``proposed``).
        consequences: Known positive and negative consequences.
        related_adrs: IDs of related ADRs.
        superseded_by: When ``status`` is ``superseded``, the ADR ID that
            replaces this one (e.g. ``ADR-0065``). ``None`` otherwise. Added
            2026-04-24 per supersession protocol in ``.claude/rules/agents.md``
            (@skip-validation: additive-only field per safety.md Safe Modification).
        created_at: ISO-8601 creation timestamp.
        updated_at: ISO-8601 last-update timestamp.
        created_by: Author identifier.
        notes: Free-form notes or discussion points.
    """

    adr_id: str
    title: str
    category: str
    context: str
    decision: str
    status: str = ADRStatus.PROPOSED.value
    consequences: str = ""
    related_adrs: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    created_at: str = ""
    updated_at: str = ""
    created_by: str = "system"
    notes: str = ""

    def __repr__(self) -> str:
        return f"ADR(adr_id={self.adr_id!r}, title={self.title!r}, category={self.category!r}, status={self.status!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ADR:
        """Deserialize an ADR from a dictionary.

        Args:
            data: Dictionary with ADR fields.

        Returns:
            Populated ADR instance with defaults for missing fields.
        """
        # @skip-validation: additive read of existing JSON key; no behavior change for missing key
        # Normalize status to canonical lowercase (ADRStatus enum values are lowercase).
        raw_status = data.get("status", ADRStatus.PROPOSED.value)
        status = raw_status.lower() if isinstance(raw_status, str) else ADRStatus.PROPOSED.value
        return cls(
            adr_id=data.get("adr_id", ""),
            title=data.get("title", ""),
            category=data.get("category", "architecture"),
            context=data.get("context", ""),
            decision=data.get("decision", ""),
            status=status,
            consequences=data.get("consequences", ""),
            related_adrs=data.get("related_adrs", []),
            superseded_by=data.get("superseded_by"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            created_by=data.get("created_by", "system"),
            notes=data.get("notes", ""),
        )


@dataclass
class ADRProposal:
    """A proposal for an architecture decision with multiple options.

    Args:
        question: The question or problem being addressed.
        options: List of option dicts with ``id``, ``description``, ``pros``, ``cons``.
        recommended: Zero-based index of the recommended option.
        rationale: Explanation for the recommendation.
    """

    question: str
    options: list[dict[str, Any]]
    recommended: int = 0
    rationale: str = ""

    def __repr__(self) -> str:
        return f"ADRProposal(options={len(self.options)}, recommended={self.recommended!r})"


class ADRSystem:
    """Manages Architecture Decision Records with JSON file persistence.

    ADRs are stored as individual JSON files in ``storage_path`` and loaded
    into memory at initialization.  Provides CRUD operations, filtering,
    statistics, and a proposal workflow.

    Args:
        storage_path: Directory for ADR JSON files.  Defaults to
            ``~/.vetinari/adr``.
    """

    _instance: ADRSystem | None = None
    _instance_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls, storage_path: str | None = None) -> ADRSystem:
        """Return the singleton ADRSystem, creating it on first call.

        Args:
            storage_path: Override the default storage directory (only used on
                first call).

        Returns:
            The shared ADRSystem instance.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(storage_path)
        return cls._instance

    def __init__(self, storage_path: str | None = None) -> None:
        if storage_path is None:
            # Default to adr/ in the project root (beside vetinari/ package)
            project_root = Path(__file__).resolve().parent.parent
            storage_path = str(project_root / "adr")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.adrs: dict[str, ADR] = {}
        self._load_adrs()

    def _load_adrs(self) -> None:
        """Load all ADR JSON files from the storage directory."""
        for file in self.storage_path.glob("*.json"):
            try:
                with Path(file).open(encoding="utf-8") as f:
                    data = json.load(f)
                    adr = ADR.from_dict(data)
                    self.adrs[adr.adr_id] = adr
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                logger.exception("Error loading ADR from %s", file)

    def _save_adr(self, adr: ADR) -> None:
        """Persist a single ADR to its JSON file.

        Args:
            adr: The ADR to save.

        Raises:
            ValueError: If the ADR ID contains path traversal sequences that
                would place the file outside the configured storage directory.
            SandboxPolicyViolation: If the resolved target lies inside a path
                that the sandbox policy blocks from writes.
        """
        target = (self.storage_path / f"{adr.adr_id}.json").resolve()
        if not target.is_relative_to(self.storage_path.resolve()):
            raise ValueError(f"ADR ID contains path traversal: {adr.adr_id}")
        # Sandbox gate: fail closed if the resolved target lands in a blocked
        # directory (e.g., /etc, ~/.ssh) — defence in depth beyond the
        # path-traversal check above, which only rejects escapes from the
        # configured storage_path. Storage paths themselves are attacker-
        # controllable in test/dev configs, so the sandbox policy has the
        # final say.
        enforce_blocked_paths(target)
        with target.open("w", encoding="utf-8") as f:
            json.dump(adr.to_dict(), f, indent=2)

    def _next_adr_id(self) -> str:
        """Generate the next sequential ADR ID.

        Returns:
            An ID like ``ADR-0001`` that does not collide with existing IDs.
        """
        existing_nums: list[int] = []
        for adr_id in self.adrs:
            # Parse "ADR-NNNN" format
            parts = adr_id.split("-", 1)
            if len(parts) == 2:
                with contextlib.suppress(ValueError):
                    existing_nums.append(int(parts[1]))
        next_num = max(existing_nums, default=0) + 1
        return f"ADR-{next_num:04d}"

    def create_adr(
        self,
        title: str,
        category: str,
        context: str,
        decision: str,
        consequences: str = "",
        created_by: str = "user",
        adr_id: str | None = None,
        status: str = ADRStatus.PROPOSED.value,
        related_adrs: list[str] | None = None,
        notes: str = "",
    ) -> ADR:
        """Create and persist a new ADR.

        Args:
            title: Short descriptive title.
            category: One of :class:`ADRCategory` values.
            context: Problem statement or background.
            decision: The decision made.
            consequences: Known consequences of the decision.
            created_by: Author identifier.
            adr_id: Explicit ID; auto-generated if ``None``.
            status: Initial status (default ``accepted``).
            related_adrs: IDs of related ADRs.
            notes: Free-form notes.

        Returns:
            The newly created ADR.
        """
        if adr_id is None:
            adr_id = self._next_adr_id()
        now = datetime.now(timezone.utc).isoformat()

        adr = ADR(
            adr_id=adr_id,
            title=title,
            category=category,
            context=context,
            decision=decision,
            consequences=consequences,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            status=status,
            related_adrs=related_adrs or [],  # noqa: VET112 - empty fallback preserves optional request metadata contract
            notes=notes,
        )

        self.adrs[adr_id] = adr
        self._save_adr(adr)
        logger.info("Created ADR %s: %s", adr_id, title)
        return adr

    def get_adr(self, adr_id: str) -> ADR | None:
        """Retrieve an ADR by ID.

        Args:
            adr_id: The ADR identifier (e.g. ``ADR-0001``).

        Returns:
            The ADR if found, otherwise ``None``.
        """
        return self.adrs.get(adr_id)

    def list_adrs(
        self,
        status: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[ADR]:
        """List ADRs with optional filtering.

        Args:
            status: Filter by status value (e.g. ``accepted``).
            category: Filter by category value.
            limit: Maximum number of results (default 100).

        Returns:
            List of matching ADRs sorted by creation date (newest first).
        """
        results = list(self.adrs.values())

        if status:
            results = [a for a in results if a.status == status]
        if category:
            results = [a for a in results if a.category == category]

        results.sort(key=lambda a: a.created_at, reverse=True)
        return results[:limit]

    def get_recent_decisions(self, n: int = 5) -> list[ADR]:
        """Return the *n* most recently created ADRs for contextual reference.

        Useful for injecting recent architectural decisions into agent prompts
        so they are aware of recent choices and constraints.

        Args:
            n: Maximum number of ADRs to return (default 5).

        Returns:
            List of up to *n* ADRs sorted newest-first.
        """
        all_adrs = list(self.adrs.values())
        all_adrs.sort(key=lambda a: a.created_at, reverse=True)
        return all_adrs[:n]

    def update_adr(self, adr_id: str, updates: dict[str, Any]) -> ADR | None:
        """Update fields on an existing ADR.

        Args:
            adr_id: The ADR to update.
            updates: Dictionary of field names to new values.

        Returns:
            The updated ADR, or ``None`` if not found.
        """
        adr = self.adrs.get(adr_id)
        if not adr:
            return None

        for key, value in updates.items():
            if hasattr(adr, key):
                setattr(adr, key, value)

        adr.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_adr(adr)
        return adr

    def deprecate_adr(self, adr_id: str, replacement_id: str | None = None) -> ADR | None:
        """Mark an ADR as deprecated, optionally linking a replacement.

        Args:
            adr_id: The ADR to deprecate.
            replacement_id: ID of the superseding ADR.

        Returns:
            The deprecated ADR, or ``None`` if not found.
        """
        adr = self.adrs.get(adr_id)
        if not adr:
            return None

        adr.status = ADRStatus.DEPRECATED.value
        if replacement_id:
            adr.related_adrs.append(replacement_id)
            replacement_adr = self.adrs.get(replacement_id)
            if replacement_adr:
                replacement_adr.related_adrs.append(adr_id)

        adr.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_adr(adr)
        return adr

    def supersede_adr(self, adr_id: str, replacement_id: str) -> ADR | None:
        """Mark an ADR as superseded by a replacement and link both sides.

        Use this when a newer ADR replaces an older one (per the supersession
        protocol in ``.claude/rules/agents.md``). Sets the superseded ADR's
        status to ``superseded`` and populates its ``superseded_by`` field.
        Adds a bidirectional entry to both ADRs' ``related_adrs``. Persists
        both sides in one call.

        Differs from :meth:`deprecate_adr`: deprecation says "do not use"
        without implying a replacement is mandatory; supersession says "this
        has been replaced by X" and requires the replacement to exist.

        Args:
            adr_id: The ADR being superseded.
            replacement_id: The ADR that replaces it; must already exist in
                the registry.

        Returns:
            The superseded ADR on success, or ``None`` if either ID is not
            found in the registry.
        """
        adr = self.adrs.get(adr_id)
        replacement = self.adrs.get(replacement_id)
        if adr is None or replacement is None:
            logger.warning(
                "supersede_adr: could not link %s -> %s (adr_found=%s, replacement_found=%s)",
                adr_id,
                replacement_id,
                adr is not None,
                replacement is not None,
            )
            return None

        adr.status = ADRStatus.SUPERSEDED.value
        adr.superseded_by = replacement_id
        if replacement_id not in adr.related_adrs:
            adr.related_adrs.append(replacement_id)
        if adr_id not in replacement.related_adrs:
            replacement.related_adrs.append(adr_id)

        now = datetime.now(timezone.utc).isoformat()
        adr.updated_at = now
        replacement.updated_at = now
        self._save_adr(adr)
        self._save_adr(replacement)
        logger.info("Superseded ADR %s by %s", adr_id, replacement_id)
        return adr

    def is_high_stakes(self, category: str) -> bool:
        """Check whether a category is considered high-stakes.

        Args:
            category: Category value string.

        Returns:
            ``True`` if the category requires extra review.
        """
        try:
            cat = ADRCategory(category)
            return cat in HIGH_STAKES_CATEGORIES
        except ValueError:
            logger.warning("Unknown ADR category %r — treating as non-high-stakes", category)
            return False

    def generate_proposal(self, context: str, num_options: int = 3) -> ADRProposal:
        """Generate a proposal template for a new architecture decision.

        Creates a proposal with placeholder options that can be customized
        before being accepted via :meth:`accept_proposal`.

        Args:
            context: Problem statement to address.
            num_options: Number of option templates to include.

        Returns:
            An ADRProposal with template options.
        """
        example_options = [
            {
                "id": "option_1",
                "description": "Use centralized architecture with a single orchestrator",
                "pros": ["Simple to understand", "Easy to coordinate"],
                "cons": ["Single point of failure", "Harder to scale"],
            },
            {
                "id": "option_2",
                "description": "Use distributed agent mesh with peer-to-peer communication",
                "pros": ["More resilient", "Better scaling"],
                "cons": ["More complex coordination", "Harder to debug"],
            },
            {
                "id": "option_3",
                "description": "Use hierarchical decomposition with manager agents",
                "pros": ["Balanced complexity", "Good for large tasks"],
                "cons": ["Requires careful hierarchy design", "Latency in deep trees"],
            },
        ]

        return ADRProposal(
            question=context,
            options=example_options[:num_options],
            recommended=0,
            rationale="Option 1 provides the best balance of simplicity and functionality for initial implementation.",
        )

    def accept_proposal(self, proposal: ADRProposal, title: str, category: str) -> ADR:
        """Accept a proposal and create an ADR from it.

        Args:
            proposal: The proposal to accept.
            title: Title for the resulting ADR.
            category: Category for the resulting ADR.

        Returns:
            The newly created ADR.
        """
        decision = "; ".join([f"{o['id']}: {o['description']}" for o in proposal.options])
        consequences = "\n".join([f"Pros: {', '.join(o.get('pros', []))}" for o in proposal.options])

        return self.create_adr(
            title=title,
            category=category,
            context=proposal.question,
            decision=decision,
            consequences=consequences,
            created_by="system",
        )

    def render_markdown(self, adr_id: str) -> str:
        """Render a single ADR as a human-readable markdown string.

        Args:
            adr_id: The ADR identifier (e.g. ``ADR-0001``).

        Returns:
            A markdown-formatted string representing the ADR.

        Raises:
            KeyError: If no ADR with the given ID exists.
        """
        adr = self.adrs.get(adr_id)
        if adr is None:
            raise KeyError(f"ADR not found: {adr_id}")

        related = ", ".join(adr.related_adrs) if adr.related_adrs else "None"
        date = adr.created_at[:10] if adr.created_at else "unknown"

        return (
            f"# {adr.adr_id}: {adr.title}\n\n"
            f"**Status:** {adr.status}  \n"
            f"**Category:** {adr.category}  \n"
            f"**Date:** {date}  \n"
            f"**Stakeholders:** {adr.created_by}\n\n"
            f"## Context\n\n{adr.context}\n\n"
            f"## Decision\n\n{adr.decision}\n\n"
            f"## Consequences\n\n{adr.consequences}\n\n"
            f"## Related ADRs\n\n{related}\n"
        )

    def export_all_markdown(self, output_dir: Path | str) -> list[str]:
        """Write all ADRs as individual markdown files to a directory.

        Files are named ``{adr_id}.md`` (e.g. ``ADR-0001.md``).  The
        directory is created if it does not exist.

        Args:
            output_dir: Directory path to write markdown files into.

        Returns:
            List of absolute file paths that were written, as strings.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        for adr_id, adr in self.adrs.items():
            content = self.render_markdown(adr_id)
            file_path = out / f"{adr.adr_id}.md"
            # Sandbox gate: defence in depth against a malicious output_dir
            # (tests/dev configs can pass user-controllable paths).
            enforce_blocked_paths(file_path)
            Path(file_path).write_text(content, encoding="utf-8")
            written.append(str(file_path.resolve()))
            logger.info("Exported ADR %s to %s", adr_id, file_path)

        return written

    def get_statistics(self) -> dict[str, Any]:
        """Compute summary statistics across all ADRs.

        Returns:
            Dictionary with ``total``, ``by_status``, ``by_category``,
            and ``high_stakes_count`` keys.
        """
        stats: dict[str, Any] = {
            "total": len(self.adrs),
            "by_status": {},
            "by_category": {},
            "high_stakes_count": 0,
        }

        for adr in self.adrs.values():
            stats["by_status"][adr.status] = stats["by_status"].get(adr.status, 0) + 1
            stats["by_category"][adr.category] = stats["by_category"].get(adr.category, 0) + 1

            if self.is_high_stakes(adr.category):
                stats["high_stakes_count"] += 1

        return stats


def get_adr_system() -> ADRSystem:
    """Lazy singleton accessor — avoids module-level I/O on import.

    Returns:
        The shared ADRSystem instance.
    """
    return ADRSystem.get_instance()
