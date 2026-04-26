"""Failure registry — append-only JSONL store for pipeline failures and prevention rules.

Closes the kaizen feedback loop: every pipeline failure (Inspector rejection,
model timeout, anomaly detection hit) is logged here with structured root-cause
data.  After repeated failures in the same category, prevention rules are
auto-generated and fed back to the Inspector to catch known patterns early.

This is step 5 of the analytics pipeline:
Inference → Orchestration → Quality Gate → Failure Classification → **Failure Registry** → Prevention.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

# Lazy-initialized to avoid Path.home() at import time
_REGISTRY_DIR: Path | None = None
_RULES_DIR: Path | None = None

# Minimum failures in the same category before auto-generating a prevention rule
_RULE_GENERATION_THRESHOLD = 3


def _get_registry_dir() -> Path:
    """Return the failure registry data directory, resolving lazily via get_user_dir().

    Uses get_user_dir() so tests can override the location via VETINARI_USER_DIR
    without the directory being pinned at import time.
    """
    global _REGISTRY_DIR
    if _REGISTRY_DIR is None:
        from vetinari.constants import get_user_dir

        _REGISTRY_DIR = get_user_dir()
    return _REGISTRY_DIR


def _registry_path() -> Path:
    """Return the path to the failure registry JSONL file."""
    return _get_registry_dir() / "failure-registry.jsonl"


def _rules_path() -> Path:
    """Return the path to the prevention rules JSONL file."""
    return _get_registry_dir() / "prevention-rules.jsonl"


# ── Enums ────────────────────────────────────────────────────────────────────


class PreventionRuleType(Enum):
    """Types of prevention rules generated from failure patterns."""

    PATTERN = "pattern"  # Regex-based pattern match on output
    SEMANTIC = "semantic"  # Structural check (e.g., missing error handling)
    EXTRACTED = "extracted"  # Distilled from failure history analysis


class FailureStatus(Enum):
    """Lifecycle status of a failure registry entry."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    DEPRECATED = "deprecated"


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FailureRegistryEntry:
    """Immutable record of a single pipeline failure.

    Attributes:
        failure_id: Unique identifier prefixed with ``fail_``.
        timestamp: ISO-8601 UTC timestamp of the failure.
        category: Failure category (e.g., ``"inspector_rejection"``).
        severity: One of ``"warning"``, ``"error"``, ``"critical"``.
        description: Human-readable description of what failed.
        root_cause: Root cause analysis text.
        affected_components: List of component names impacted by the failure.
        prevention_rule: Prevention rule text if one was generated.
        status: Lifecycle status (active, resolved, deprecated).
    """

    failure_id: str
    timestamp: str
    category: str
    severity: str
    description: str
    root_cause: str = ""
    affected_components: list[str] = field(default_factory=list)
    prevention_rule: str = ""
    status: str = "active"

    def __repr__(self) -> str:
        return (
            f"FailureRegistryEntry(id={self.failure_id!r}, category={self.category!r}, "
            f"severity={self.severity!r}, status={self.status!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Plain dict with all entry fields as JSON-serializable values.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreventionRule:
    """A rule generated from repeated failure patterns to prevent recurrence.

    Attributes:
        rule_id: Unique identifier prefixed with ``prev_``.
        rule_type: Type of rule (pattern, semantic, or extracted).
        category: Failure category this rule guards against.
        pattern: Regex pattern, structural check description, or extracted heuristic.
        description: Human-readable explanation of what the rule catches.
        created_from_failures: List of failure_ids that triggered this rule's creation.
        created_at: ISO-8601 UTC timestamp of rule creation.
    """

    rule_id: str
    rule_type: str
    category: str
    pattern: str
    description: str
    created_from_failures: list[str] = field(default_factory=list)
    created_at: str = ""

    def __repr__(self) -> str:
        return f"PreventionRule(id={self.rule_id!r}, type={self.rule_type!r}, category={self.category!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Plain dict with all rule fields as JSON-serializable values.
        """
        return asdict(self)

    def matches(self, text: str) -> bool:
        """Check whether this rule's pattern matches the given text.

        For PATTERN rules, applies regex matching. For SEMANTIC and EXTRACTED
        rules, checks whether the pattern string appears as a substring
        (case-insensitive).

        Args:
            text: The output text to check against the rule.

        Returns:
            True if the rule matches.
        """
        if self.rule_type == PreventionRuleType.PATTERN.value:
            try:
                return bool(re.search(self.pattern, text, re.IGNORECASE))
            except re.error:
                logger.warning(
                    "Invalid regex in prevention rule %s — skipping: %s",
                    self.rule_id,
                    self.pattern,
                )
                return False
        # Semantic and extracted rules: substring match
        return self.pattern.lower() in text.lower()


# ── FailureRegistry ─────────────────────────────────────────────────────────


class FailureRegistry:
    """Append-only registry of pipeline failures with prevention rule generation.

    Thread-safe. Writes each failure as a single JSONL line. Prevention rules
    are auto-generated when ``_RULE_GENERATION_THRESHOLD`` failures accumulate
    in the same category.

    Side effects:
        - Writes to ``~/.vetinari/failure-registry.jsonl`` on every ``log_failure()`` call
        - Writes to ``~/.vetinari/prevention-rules.jsonl`` when rules are generated
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # In-memory cache of generated rules for fast lookup during inspection
        self._rules_cache: list[PreventionRule] | None = None
        self._rules_cache_lock = threading.Lock()

    # -- Write operations ------------------------------------------------------

    def log_failure(
        self,
        category: str,
        severity: str,
        description: str,
        root_cause: str = "",
        affected_components: list[str] | None = None,
        prevention_rule: str = "",
    ) -> FailureRegistryEntry:
        """Log a pipeline failure to the append-only registry.

        Creates a new entry with a unique ID and appends it as a single JSON
        line to the registry file. After logging, checks whether repeated
        failures in the same category should trigger prevention rule generation.

        Args:
            category: Failure category (e.g., ``"inspector_rejection"``,
                ``"model_timeout"``, ``"anomaly_detected"``).
            severity: One of ``"warning"``, ``"error"``, ``"critical"``.
            description: Human-readable description of the failure.
            root_cause: Root cause analysis text.
            affected_components: List of component names impacted.
            prevention_rule: Prevention rule text if already known.

        Returns:
            The newly created FailureRegistryEntry.
        """
        components = list(affected_components) if affected_components is not None else []
        entry = FailureRegistryEntry(
            failure_id=f"fail_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            severity=severity,
            description=description,
            root_cause=root_cause,
            affected_components=components,
            prevention_rule=prevention_rule,
            status=FailureStatus.ACTIVE.value,
        )

        with self._lock:
            self._append_entry(entry)

        logger.info(
            "Failure logged — id=%s category=%s severity=%s: %s",
            entry.failure_id,
            category,
            severity,
            description[:120],
        )

        # Check if this category has enough failures to generate a prevention rule
        self.check_and_generate_prevention_rules(category)

        return entry

    def resolve_failure(self, failure_id: str) -> bool:
        """Mark a failure as resolved by rewriting its status in the registry.

        Loads all entries, updates the matching entry's status to ``"resolved"``,
        and rewrites the file. This is an infrequent operation so the full
        rewrite is acceptable.

        Args:
            failure_id: The failure_id to resolve.

        Returns:
            True if the failure was found and resolved, False if not found.
        """
        with self._lock:
            entries = self._load_all_entries()
            found = False
            updated: list[dict[str, Any]] = []

            for entry_dict in entries:
                if entry_dict.get("failure_id") == failure_id:
                    entry_dict["status"] = FailureStatus.RESOLVED.value
                    found = True
                updated.append(entry_dict)

            if found:
                self._rewrite_entries(updated)
                logger.info("Failure %s resolved", failure_id)

            return found

    # -- Read operations -------------------------------------------------------

    def get_failures(
        self,
        category: str | None = None,
        since: float | None = None,
    ) -> list[FailureRegistryEntry]:
        """Load and filter failure entries from the registry file.

        Args:
            category: When set, only entries with this category are returned.
            since: When set, only entries after this Unix timestamp are returned.

        Returns:
            List of matching FailureRegistryEntry instances.
        """
        raw_entries = self._load_all_entries()
        results: list[FailureRegistryEntry] = []

        for entry_dict in raw_entries:
            if category and entry_dict.get("category") != category:
                continue
            if since:
                try:
                    entry_ts = datetime.fromisoformat(entry_dict.get("timestamp", "")).timestamp()
                    if entry_ts < since:
                        continue
                except (ValueError, TypeError):
                    # Unparseable timestamp — include the entry rather than silently dropping it
                    logger.debug("Could not parse timestamp for entry — including unfiltered")

            try:
                results.append(
                    FailureRegistryEntry(**{
                        k: v for k, v in entry_dict.items() if k in FailureRegistryEntry.__dataclass_fields__
                    })
                )
            except TypeError:
                logger.warning(
                    "Skipping malformed failure entry: %s",
                    entry_dict.get("failure_id", "unknown"),
                )

        return results

    def get_prevention_rules(self) -> list[PreventionRule]:
        """Load all prevention rules from the dedicated rules file.

        Returns cached rules if available, otherwise loads from disk.

        Returns:
            List of PreventionRule instances.
        """
        with self._rules_cache_lock:
            if self._rules_cache is not None:
                return list(self._rules_cache)

        rules = self._load_rules_from_disk()
        with self._rules_cache_lock:
            self._rules_cache = rules
        return list(rules)

    # -- Prevention rule generation --------------------------------------------

    def check_and_generate_prevention_rules(self, category: str) -> PreventionRule | None:
        """Check if a category has enough failures to generate a prevention rule.

        Groups active failures by category and generates a rule when the count
        reaches ``_RULE_GENERATION_THRESHOLD``. Skips categories that already
        have a prevention rule.

        Args:
            category: The failure category to check.

        Returns:
            The newly generated PreventionRule, or None if threshold not met.
        """
        entries = self.get_failures(category=category)
        active = [e for e in entries if e.status == FailureStatus.ACTIVE.value]

        if len(active) < _RULE_GENERATION_THRESHOLD:
            return None

        # Check if we already have a rule for this category
        existing_rules = self.get_prevention_rules()
        if any(r.category == category for r in existing_rules):
            return None

        # Generate rule from failure descriptions
        rule = self._generate_rule(category, active)
        if rule:
            self._save_rule(rule)
            # Invalidate cache
            with self._rules_cache_lock:
                self._rules_cache = None
            logger.info(
                "Prevention rule generated — id=%s category=%s from %d failures",
                rule.rule_id,
                category,
                len(active),
            )
        return rule

    # -- Internal helpers ------------------------------------------------------

    def _append_entry(self, entry: FailureRegistryEntry) -> None:
        """Append a single entry as JSONL. Caller must hold self._lock."""
        path = _registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except OSError as exc:
            logger.error(
                "Could not write failure entry %s — entry lost: %s",
                entry.failure_id,
                exc,
            )

    def _load_all_entries(self) -> list[dict[str, Any]]:
        """Load all entries from the JSONL file as raw dicts."""
        path = _registry_path()
        if not path.exists():
            return []

        entries: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping malformed JSON at line %d in %s",
                            line_no,
                            path,
                        )
        except OSError as exc:
            logger.warning(
                "Could not read failure registry — returning empty: %s",
                exc,
            )
        return entries

    def _rewrite_entries(self, entries: list[dict[str, Any]]) -> None:
        """Rewrite all entries to the JSONL file. Caller must hold self._lock."""
        path = _registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            logger.error("Could not rewrite failure registry — data may be stale: %s", exc)

    def _generate_rule(
        self,
        category: str,
        failures: list[FailureRegistryEntry],
    ) -> PreventionRule | None:
        """Generate a prevention rule from a set of related failures.

        Extracts common patterns from failure descriptions. Uses PATTERN type
        for failures with identifiable regex patterns, SEMANTIC for structural
        issues, and EXTRACTED for general failure history analysis.

        Args:
            category: The failure category.
            failures: List of related failure entries.

        Returns:
            A PreventionRule, or None if no meaningful rule can be extracted.
        """
        descriptions = [f.description for f in failures]
        failure_ids = [f.failure_id for f in failures]

        # Find common words across descriptions (excluding stop words)
        _stop_words = frozenset({
            "the",
            "a",
            "an",
            "is",
            "was",
            "were",
            "are",
            "be",
            "been",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "and",
            "or",
            "not",
            "no",
            "but",
            "this",
            "that",
            "it",
        })
        word_counts: dict[str, int] = {}
        for desc in descriptions:
            words = set(desc.lower().split()) - _stop_words
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Words present in all failure descriptions are the pattern
        common = [w for w, c in word_counts.items() if c >= len(failures)]

        if not common:
            # Extracted rule: summarize the category
            rule_type = PreventionRuleType.EXTRACTED.value
            pattern = f"Repeated {category} failures detected"
            description = (
                f"Auto-generated from {len(failures)} failures in category '{category}'. "
                f"Descriptions: {'; '.join(d[:80] for d in descriptions[:3])}"
            )
        elif any(kw in common for kw in ("missing", "lacking", "absent", "without", "no")):
            # Semantic rule: structural absence
            rule_type = PreventionRuleType.SEMANTIC.value
            pattern = " ".join(sorted(common)[:5])
            description = f"Output must not have: {pattern} (from {len(failures)} failures)"
        else:
            # Pattern rule: regex from common words
            rule_type = PreventionRuleType.PATTERN.value
            pattern = "|".join(re.escape(w) for w in sorted(common)[:5])
            description = f"Pattern match for known failure: {pattern} (from {len(failures)} failures)"

        return PreventionRule(
            rule_id=f"prev_{uuid.uuid4().hex[:12]}",
            rule_type=rule_type,
            category=category,
            pattern=pattern,
            description=description,
            created_from_failures=failure_ids,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _save_rule(self, rule: PreventionRule) -> None:
        """Append a prevention rule to the rules JSONL file."""
        path = _rules_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rule.to_dict(), default=str) + "\n")
        except OSError as exc:
            logger.error(
                "Could not save prevention rule %s — rule lost: %s",
                rule.rule_id,
                exc,
            )

    def _load_rules_from_disk(self) -> list[PreventionRule]:
        """Load all prevention rules from the JSONL file."""
        path = _rules_path()
        if not path.exists():
            return []

        rules: list[PreventionRule] = []
        try:
            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        rules.append(
                            PreventionRule(**{
                                k: v for k, v in data.items() if k in PreventionRule.__dataclass_fields__
                            })
                        )
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "Skipping malformed prevention rule at line %d in %s",
                            line_no,
                            path,
                        )
        except OSError as exc:
            logger.warning(
                "Could not read prevention rules — returning empty: %s",
                exc,
            )
        return rules

    # -- Remediation outcome tracking (14.1) ------------------------------------

    def log_remediation_outcome(
        self,
        failure_mode: str,
        action_description: str,
        success: bool,
    ) -> None:
        """Log the outcome of a remediation action for trend tracking.

        Appends a record to ``~/.vetinari/remediation-outcomes.jsonl``
        (separate from the main failure registry) so that success rates
        per (failure_mode, action) pair can be computed.

        Args:
            failure_mode: The failure mode that was remediated (e.g. ``"oom"``).
            action_description: Description of the remediation action taken.
            success: Whether the remediation resolved the failure.
        """
        import uuid
        from datetime import datetime, timezone

        record = {
            "outcome_id": f"rem_{uuid.uuid4().hex[:12]}",
            "failure_mode": failure_mode,
            "action_description": action_description,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
        }

        path = _get_registry_dir() / "remediation-outcomes.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.error(
                "Could not write remediation outcome — record lost: %s",
                exc,
            )

        logger.info(
            "Remediation outcome logged — mode=%s action=%s success=%s",
            failure_mode,
            action_description[:80],
            success,
        )

    def get_remediation_stats(self) -> dict[tuple[str, str], dict[str, int]]:
        """Return per-(failure_mode, action) success/failure counts.

        Reads from ``~/.vetinari/remediation-outcomes.jsonl`` and aggregates
        counts.

        Returns:
            Dict mapping ``(failure_mode, action_description)`` to
            ``{"success": int, "failure": int}``.
        """
        path = _get_registry_dir() / "remediation-outcomes.jsonl"
        if not path.exists():
            return {}

        stats: dict[tuple[str, str], dict[str, int]] = {}
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Skipping malformed remediation outcome record in %s: %s",
                            path,
                            exc,
                        )
                        continue
                    key = (record.get("failure_mode", ""), record.get("action_description", ""))
                    if key not in stats:
                        stats[key] = {"success": 0, "failure": 0}
                    if record.get("success"):
                        stats[key]["success"] += 1
                    else:
                        stats[key]["failure"] += 1
        except OSError as exc:
            logger.warning(
                "Could not read remediation outcomes — returning empty stats: %s",
                exc,
            )
        return stats

    def get_remediation_confidence(
        self,
        failure_mode: str,
        action_description: str,
    ) -> float:
        """Return confidence score for a (failure_mode, action) pair.

        Confidence is the success rate: successes / total.  Returns 0.0 if
        no outcomes have been recorded for this pair.

        Args:
            failure_mode: The failure mode string.
            action_description: The remediation action description.

        Returns:
            Float between 0.0 and 1.0.
        """
        stats = self.get_remediation_stats()
        counts = stats.get((failure_mode, action_description))
        if not counts:
            return 0.0
        total = counts["success"] + counts["failure"]
        if total == 0:
            return 0.0
        return counts["success"] / total

    def reset(self) -> None:
        """Clear all in-memory state. Intended for test isolation only."""
        with self._rules_cache_lock:
            self._rules_cache = None


# ── Singleton management ─────────────────────────────────────────────────────

_registry: FailureRegistry | None = None
_registry_lock = threading.Lock()


def get_failure_registry() -> FailureRegistry:
    """Return the process-wide FailureRegistry singleton (thread-safe).

    Uses double-checked locking so the lock is only acquired during
    first initialization.

    Returns:
        The shared FailureRegistry instance.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = FailureRegistry()
    return _registry


def reset_failure_registry() -> None:
    """Destroy the singleton so the next call creates a fresh instance.

    Resets both the registry singleton and the cached registry directory so
    that tests can change VETINARI_USER_DIR between calls and get a clean slate.

    Intended for test isolation only.
    """
    global _registry, _REGISTRY_DIR
    with _registry_lock:
        if _registry is not None:
            _registry.reset()
        _registry = None
        _REGISTRY_DIR = None
