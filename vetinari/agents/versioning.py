"""
Agent Versioning & Rollback
=============================

Tracks versions of agent configurations (prompts, sampling profiles,
tool sets) and supports rollback to any prior version.

Usage:
    from vetinari.agents.versioning import AgentVersionManager

    mgr = AgentVersionManager()
    v_id = mgr.save_version("BUILDER", config)
    mgr.rollback("BUILDER", v_id)
    diff = mgr.compare_versions("BUILDER", v1_id, v2_id)
"""

import copy
import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_VERSIONS_DIR = ".vetinari/agent_versions"


@dataclass
class AgentVersion:
    """A snapshot of an agent's configuration at a point in time."""
    version_id: str = field(default_factory=lambda: f"av_{uuid.uuid4().hex[:8]}")
    agent_type: str = ""
    version_number: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""

    # Configuration snapshot
    system_prompt: str = ""
    sampling_profile: Dict[str, Any] = field(default_factory=dict)
    tool_set: List[str] = field(default_factory=list)
    model_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    author: str = ""                # Who made this version
    reason: str = ""                # Why it was created
    parent_version_id: str = ""     # Previous version
    is_active: bool = False         # Whether this is the currently active version
    content_hash: str = ""          # Hash for deduplication

    # Performance metrics at time of versioning
    quality_score: Optional[float] = None
    success_rate: Optional[float] = None
    avg_latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentVersion":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class VersionDiff:
    """Comparison between two agent versions."""
    agent_type: str = ""
    version_a_id: str = ""
    version_b_id: str = ""
    version_a_number: int = 0
    version_b_number: int = 0

    prompt_changed: bool = False
    prompt_diff_summary: str = ""

    sampling_changed: bool = False
    sampling_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)  # param -> (old, new)

    tools_added: List[str] = field(default_factory=list)
    tools_removed: List[str] = field(default_factory=list)

    model_changed: bool = False
    model_diff: str = ""

    parameters_changed: bool = False
    parameter_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    quality_a: Optional[float] = None
    quality_b: Optional[float] = None
    quality_delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "agent_type": self.agent_type,
            "version_a_id": self.version_a_id,
            "version_b_id": self.version_b_id,
            "version_a_number": self.version_a_number,
            "version_b_number": self.version_b_number,
            "prompt_changed": self.prompt_changed,
            "prompt_diff_summary": self.prompt_diff_summary,
            "sampling_changed": self.sampling_changed,
            "sampling_changes": {k: list(v) for k, v in self.sampling_changes.items()},
            "tools_added": self.tools_added,
            "tools_removed": self.tools_removed,
            "model_changed": self.model_changed,
            "model_diff": self.model_diff,
            "parameters_changed": self.parameters_changed,
            "parameter_changes": {k: list(v) for k, v in self.parameter_changes.items()},
            "quality_a": self.quality_a,
            "quality_b": self.quality_b,
            "quality_delta": self.quality_delta,
        }
        return d

    @property
    def has_changes(self) -> bool:
        return (
            self.prompt_changed or self.sampling_changed or
            bool(self.tools_added) or bool(self.tools_removed) or
            self.model_changed or self.parameters_changed
        )


class AgentVersionManager:
    """
    Manages version history for agent configurations.

    Features:
    - Save snapshots of agent config (prompts, sampling, tools)
    - Rollback to any prior version
    - Compare two versions with detailed diff
    - Track performance metrics per version
    - Auto-versioning on config changes
    - Persistent storage to disk
    """

    def __init__(self, versions_dir: Optional[str] = None):
        """
        Args:
            versions_dir: Directory for version storage.
        """
        self._dir = Path(versions_dir or DEFAULT_VERSIONS_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # In-memory index: agent_type -> list of version_ids (ordered)
        self._version_index: Dict[str, List[str]] = {}
        # version_id -> AgentVersion
        self._versions: Dict[str, AgentVersion] = {}
        # agent_type -> currently active version_id
        self._active_versions: Dict[str, str] = {}

        self._load_index()
        logger.info("AgentVersionManager initialized: %d agents tracked (dir=%s)",
                     len(self._version_index), self._dir)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def save_version(
        self,
        agent_type: str,
        config: Dict[str, Any],
        description: str = "",
        author: str = "system",
        reason: str = "",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a new version of an agent's configuration.

        Args:
            agent_type: The agent type (e.g., "BUILDER", "RESEARCHER").
            config: Configuration dictionary with keys:
                - system_prompt: str
                - sampling_profile: Dict
                - tool_set: List[str]
                - model_id: str
                - parameters: Dict (any extra params)
            description: Description of what changed.
            author: Who made the change.
            reason: Why the change was made.
            metrics: Current performance metrics (quality_score, success_rate, etc.).

        Returns:
            version_id: Unique identifier for the saved version.
        """
        with self._lock:
            # Determine version number
            existing = self._version_index.get(agent_type, [])
            version_number = len(existing) + 1

            # Compute content hash for dedup
            content_str = json.dumps(config, sort_keys=True, default=str)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

            # Check for duplicate: if last version has same hash, skip
            if existing:
                last_version = self._versions.get(existing[-1])
                if last_version and last_version.content_hash == content_hash:
                    logger.debug("Version for %s unchanged (hash match), skipping", agent_type)
                    return existing[-1]

            parent_id = existing[-1] if existing else ""

            version = AgentVersion(
                agent_type=agent_type,
                version_number=version_number,
                description=description,
                system_prompt=config.get("system_prompt", ""),
                sampling_profile=config.get("sampling_profile", {}),
                tool_set=config.get("tool_set", []),
                model_id=config.get("model_id", ""),
                parameters=config.get("parameters", {}),
                author=author,
                reason=reason,
                parent_version_id=parent_id,
                is_active=True,
                content_hash=content_hash,
                quality_score=(metrics or {}).get("quality_score"),
                success_rate=(metrics or {}).get("success_rate"),
                avg_latency_ms=(metrics or {}).get("avg_latency_ms"),
            )

            # Deactivate previous active version
            if agent_type in self._active_versions:
                prev_id = self._active_versions[agent_type]
                prev = self._versions.get(prev_id)
                if prev:
                    prev.is_active = False

            # Store
            self._versions[version.version_id] = version
            if agent_type not in self._version_index:
                self._version_index[agent_type] = []
            self._version_index[agent_type].append(version.version_id)
            self._active_versions[agent_type] = version.version_id

            # Persist
            self._save_version_file(version)
            self._save_index()

            logger.info(
                "Agent version saved: %s v%d (id=%s, author=%s)",
                agent_type, version_number, version.version_id, author,
            )
            return version.version_id

    def rollback(self, agent_type: str, version_id: str) -> Optional[AgentVersion]:
        """
        Rollback an agent to a specific previous version.

        Marks the target version as active and returns it so the caller
        can apply the configuration.

        Args:
            agent_type: The agent type.
            version_id: The version to rollback to.

        Returns:
            The rolled-back AgentVersion, or None if not found.
        """
        with self._lock:
            version = self._versions.get(version_id)
            if version is None:
                # Try loading from disk
                version = self._load_version_file(version_id)
                if version is None:
                    logger.error("Version %s not found for rollback", version_id)
                    return None

            if version.agent_type != agent_type:
                logger.error(
                    "Version %s belongs to %s, not %s",
                    version_id, version.agent_type, agent_type,
                )
                return None

            # Deactivate current
            if agent_type in self._active_versions:
                prev_id = self._active_versions[agent_type]
                prev = self._versions.get(prev_id)
                if prev:
                    prev.is_active = False

            # Activate target
            version.is_active = True
            self._active_versions[agent_type] = version_id
            self._save_index()

            logger.info(
                "Agent %s rolled back to version %s (v%d)",
                agent_type, version_id, version.version_number,
            )
            return version

    def compare_versions(self, v1_id: str, v2_id: str) -> Optional[VersionDiff]:
        """
        Compare two versions and produce a detailed diff.

        Args:
            v1_id: First version ID (typically older).
            v2_id: Second version ID (typically newer).

        Returns:
            VersionDiff object, or None if either version is not found.
        """
        with self._lock:
            v1 = self._versions.get(v1_id) or self._load_version_file(v1_id)
            v2 = self._versions.get(v2_id) or self._load_version_file(v2_id)

            if v1 is None or v2 is None:
                logger.error("Cannot compare: version not found (v1=%s, v2=%s)", v1_id, v2_id)
                return None

            diff = VersionDiff(
                agent_type=v1.agent_type,
                version_a_id=v1_id,
                version_b_id=v2_id,
                version_a_number=v1.version_number,
                version_b_number=v2.version_number,
            )

            # Compare prompts
            if v1.system_prompt != v2.system_prompt:
                diff.prompt_changed = True
                diff.prompt_diff_summary = self._summarize_text_diff(
                    v1.system_prompt, v2.system_prompt
                )

            # Compare sampling profiles
            sp1 = v1.sampling_profile or {}
            sp2 = v2.sampling_profile or {}
            all_keys = set(sp1.keys()) | set(sp2.keys())
            for key in all_keys:
                val1 = sp1.get(key)
                val2 = sp2.get(key)
                if val1 != val2:
                    diff.sampling_changed = True
                    diff.sampling_changes[key] = (val1, val2)

            # Compare tool sets
            tools1 = set(v1.tool_set)
            tools2 = set(v2.tool_set)
            diff.tools_added = sorted(tools2 - tools1)
            diff.tools_removed = sorted(tools1 - tools2)

            # Compare model
            if v1.model_id != v2.model_id:
                diff.model_changed = True
                diff.model_diff = f"{v1.model_id} -> {v2.model_id}"

            # Compare parameters
            p1 = v1.parameters or {}
            p2 = v2.parameters or {}
            all_params = set(p1.keys()) | set(p2.keys())
            for key in all_params:
                val1 = p1.get(key)
                val2 = p2.get(key)
                if val1 != val2:
                    diff.parameters_changed = True
                    diff.parameter_changes[key] = (val1, val2)

            # Compare quality metrics
            diff.quality_a = v1.quality_score
            diff.quality_b = v2.quality_score
            if v1.quality_score is not None and v2.quality_score is not None:
                diff.quality_delta = v2.quality_score - v1.quality_score

            return diff

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_version(self, version_id: str) -> Optional[AgentVersion]:
        """Get a specific version by ID."""
        version = self._versions.get(version_id)
        if version is None:
            version = self._load_version_file(version_id)
        return version

    def get_active_version(self, agent_type: str) -> Optional[AgentVersion]:
        """Get the currently active version for an agent type."""
        vid = self._active_versions.get(agent_type)
        if vid:
            return self._versions.get(vid)
        return None

    def list_versions(self, agent_type: str) -> List[AgentVersion]:
        """List all versions for an agent type, ordered chronologically."""
        version_ids = self._version_index.get(agent_type, [])
        versions = []
        for vid in version_ids:
            v = self._versions.get(vid)
            if v is None:
                v = self._load_version_file(vid)
            if v:
                versions.append(v)
        return versions

    def list_agents(self) -> List[str]:
        """List all agent types that have version history."""
        return list(self._version_index.keys())

    def get_version_count(self, agent_type: str) -> int:
        """Get the number of versions for an agent type."""
        return len(self._version_index.get(agent_type, []))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_version_file(self, version: AgentVersion) -> None:
        """Save a version to its own JSON file."""
        try:
            agent_dir = self._dir / version.agent_type
            agent_dir.mkdir(parents=True, exist_ok=True)
            path = agent_dir / f"{version.version_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(version.to_dict(), f, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to save version file %s: %s", version.version_id, exc)

    def _load_version_file(self, version_id: str) -> Optional[AgentVersion]:
        """Load a version from disk by ID, scanning agent directories."""
        try:
            for agent_dir in self._dir.iterdir():
                if agent_dir.is_dir():
                    path = agent_dir / f"{version_id}.json"
                    if path.exists():
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        version = AgentVersion.from_dict(data)
                        self._versions[version_id] = version
                        return version
        except Exception as exc:
            logger.error("Failed to load version %s: %s", version_id, exc)
        return None

    def _save_index(self) -> None:
        """Save the version index and active versions to disk."""
        try:
            index_data = {
                "version_index": self._version_index,
                "active_versions": self._active_versions,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            index_path = self._dir / "_index.json"
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index_data, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save version index: %s", exc)

    def _load_index(self) -> None:
        """Load the version index from disk."""
        index_path = self._dir / "_index.json"
        if not index_path.exists():
            return
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._version_index = data.get("version_index", {})
            self._active_versions = data.get("active_versions", {})

            # Preload active versions into memory
            for agent_type, vid in self._active_versions.items():
                if vid not in self._versions:
                    self._load_version_file(vid)

            logger.debug("Version index loaded: %d agents", len(self._version_index))
        except Exception as exc:
            logger.warning("Failed to load version index: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_text_diff(text_a: str, text_b: str) -> str:
        """Create a human-readable summary of text differences."""
        if not text_a and text_b:
            return f"Added ({len(text_b)} chars)"
        if text_a and not text_b:
            return f"Removed ({len(text_a)} chars)"

        lines_a = text_a.strip().splitlines()
        lines_b = text_b.strip().splitlines()

        added = len(lines_b) - len(lines_a)
        chars_delta = len(text_b) - len(text_a)

        # Count changed lines
        changed = 0
        for i in range(min(len(lines_a), len(lines_b))):
            if lines_a[i] != lines_b[i]:
                changed += 1

        parts = []
        if added > 0:
            parts.append(f"+{added} lines")
        elif added < 0:
            parts.append(f"{added} lines")
        if changed > 0:
            parts.append(f"{changed} lines modified")
        if chars_delta != 0:
            parts.append(f"{chars_delta:+d} chars")

        return ", ".join(parts) if parts else "Minor changes"
