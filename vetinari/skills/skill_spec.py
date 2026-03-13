"""Skill Specification Contract.

=============================
Defines the ``SkillSpec`` dataclass — the canonical contract for all Vetinari
skills, parallel to ``AgentSpec`` for agents.

Every skill MUST have a corresponding ``SkillSpec`` entry in the programmatic
registry (``skill_registry.py``).

The spec enforces three pillars:
  1. **Standards** — coding/output quality rules the skill must uphold
  2. **Guidelines** — best-practice advice for skill consumers
  3. **Constraints** — hard limits on resources, scope, and safety
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass
class SkillStandard:
    """A single quality standard that a skill must enforce.

    Standards are *mandatory* — a skill execution that violates any standard
    is considered non-conformant and should be flagged by the verifier.
    """

    id: str  # e.g. "STD-BLD-001"
    category: str  # "code_quality", "security", "testing", "documentation", "performance"
    rule: str  # Human-readable rule statement
    severity: str = "error"  # "error" (must fix) | "warning" (should fix)
    check_hint: str = ""  # Machine-actionable hint for automated validation

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "rule": self.rule,
            "severity": self.severity,
            "check_hint": self.check_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillStandard:
        return cls(
            id=data.get("id", ""),
            category=data.get("category", ""),
            rule=data.get("rule", ""),
            severity=data.get("severity", "error"),
            check_hint=data.get("check_hint", ""),
        )


@dataclass
class SkillGuideline:
    """A best-practice guideline for skill consumers.

    Guidelines are *advisory* — they improve quality but are not enforced
    as hard failures.
    """

    id: str  # e.g. "GDL-BLD-001"
    category: str  # "usage", "integration", "performance", "output_format"
    recommendation: str  # What to do
    rationale: str = ""  # Why it matters

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillGuideline:
        return cls(
            id=data.get("id", ""),
            category=data.get("category", ""),
            recommendation=data.get("recommendation", ""),
            rationale=data.get("rationale", ""),
        )


@dataclass
class SkillConstraint:
    """A hard constraint on skill execution.

    Constraints are *immutable limits* — they cannot be overridden by the
    caller and are enforced at the framework level.
    """

    id: str  # e.g. "CON-BLD-001"
    category: str  # "resource", "scope", "safety", "concurrency"
    description: str  # What is constrained
    limit: str  # The actual limit value (human-readable)
    enforcement: str = "hard"  # "hard" (runtime enforced) | "soft" (logged warning)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "limit": self.limit,
            "enforcement": self.enforcement,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillConstraint:
        return cls(
            id=data.get("id", ""),
            category=data.get("category", ""),
            description=data.get("description", ""),
            limit=data.get("limit", ""),
            enforcement=data.get("enforcement", "hard"),
        )


# ---------------------------------------------------------------------------
# Main SkillSpec
# ---------------------------------------------------------------------------


@dataclass
class SkillSpec:
    """Specification contract for all Vetinari skills — parallel to AgentSpec for agents.

    Three pillars of quality governance:
      - ``standards``: Mandatory rules — violations are errors.
      - ``guidelines``: Best-practice advice — violations are warnings.
      - ``constraints``: Hard limits — enforced by the runtime.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    skill_id: str  # Unique kebab-case identifier
    name: str  # Human-readable name
    description: str  # One-line purpose
    version: str = "1.0.0"  # Semantic version
    agent_type: str | None = None  # Owning agent type (AgentType.value)
    modes: list[str] = field(default_factory=list)  # Compatible agent modes

    # ── Capability declaration ────────────────────────────────────────────
    capabilities: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)  # JSON schema for inputs
    output_schema: dict[str, Any] = field(default_factory=dict)  # JSON schema for outputs

    # ── Resource constraints (aligned with Phase 8 ResourceConstraint) ───
    max_tokens: int = 4096
    max_retries: int = 3
    timeout_seconds: int = 120
    max_cost_usd: float = 0.50
    requires_tools: list[str] = field(default_factory=list)  # Tool dependencies

    # ── Quality standards ─────────────────────────────────────────────────
    min_verification_score: float = 0.5
    require_schema_validation: bool = True
    forbidden_patterns: list[str] = field(default_factory=list)

    # ── Standards / Guidelines / Constraints (the three pillars) ──────────
    standards: list[SkillStandard] = field(default_factory=list)
    guidelines: list[SkillGuideline] = field(default_factory=list)
    constraints: list[SkillConstraint] = field(default_factory=list)

    # ── Metadata ──────────────────────────────────────────────────────────
    author: str = "vetinari"
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    deprecated: bool = False
    deprecated_by: str = ""  # Replacement skill_id if deprecated

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate this spec meets contract requirements. Returns list of errors."""
        errors = []
        if not self.skill_id:
            errors.append("skill_id is required")
        if not self.name:
            errors.append("name is required")
        if not self.description:
            errors.append("description is required")
        if not self.modes:
            errors.append(f"Skill '{self.skill_id}' has no modes defined")
        if not self.input_schema:
            errors.append(f"Skill '{self.skill_id}' missing input_schema")
        if not self.output_schema:
            errors.append(f"Skill '{self.skill_id}' missing output_schema")
        if self.max_tokens < 1:
            errors.append(f"Skill '{self.skill_id}' max_tokens must be positive")
        if self.timeout_seconds < 1:
            errors.append(f"Skill '{self.skill_id}' timeout_seconds must be positive")
        if not 0.0 <= self.min_verification_score <= 1.0:
            errors.append(f"Skill '{self.skill_id}' min_verification_score must be 0-1")
        # Validate standards have unique IDs
        std_ids = [s.id for s in self.standards]
        if len(std_ids) != len(set(std_ids)):
            errors.append(f"Skill '{self.skill_id}' has duplicate standard IDs")
        # Validate guidelines have unique IDs
        gdl_ids = [g.id for g in self.guidelines]
        if len(gdl_ids) != len(set(gdl_ids)):
            errors.append(f"Skill '{self.skill_id}' has duplicate guideline IDs")
        # Validate constraints have unique IDs
        con_ids = [c.id for c in self.constraints]
        if len(con_ids) != len(set(con_ids)):
            errors.append(f"Skill '{self.skill_id}' has duplicate constraint IDs")
        return errors

    def get_standards_by_category(self, category: str) -> list[SkillStandard]:
        """Return all standards in a given category."""
        return [s for s in self.standards if s.category == category]

    def get_error_standards(self) -> list[SkillStandard]:
        """Return all mandatory (severity=error) standards."""
        return [s for s in self.standards if s.severity == "error"]

    def get_constraints_by_category(self, category: str) -> list[SkillConstraint]:
        """Return all constraints in a given category."""
        return [c for c in self.constraints if c.category == category]

    def get_hard_constraints(self) -> list[SkillConstraint]:
        """Return all hard-enforced constraints."""
        return [c for c in self.constraints if c.enforcement == "hard"]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "id": self.skill_id,  # Backward compat with disk-based registry
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "agent_type": self.agent_type,
            "modes": self.modes,
            "capabilities": self.capabilities,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "max_cost_usd": self.max_cost_usd,
            "requires_tools": self.requires_tools,
            "min_verification_score": self.min_verification_score,
            "require_schema_validation": self.require_schema_validation,
            "forbidden_patterns": self.forbidden_patterns,
            "standards": [s.to_dict() for s in self.standards],
            "guidelines": [g.to_dict() for g in self.guidelines],
            "constraints": [c.to_dict() for c in self.constraints],
            "author": self.author,
            "tags": self.tags,
            "enabled": self.enabled,
            "deprecated": self.deprecated,
            "deprecated_by": self.deprecated_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillSpec:
        """Deserialize from dictionary."""
        return cls(
            skill_id=data.get("skill_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            agent_type=data.get("agent_type"),
            modes=data.get("modes", []),
            capabilities=data.get("capabilities", []),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            max_tokens=data.get("max_tokens", 4096),
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 120),
            max_cost_usd=data.get("max_cost_usd", 0.50),
            requires_tools=data.get("requires_tools", []),
            min_verification_score=data.get("min_verification_score", 0.5),
            require_schema_validation=data.get("require_schema_validation", True),
            forbidden_patterns=data.get("forbidden_patterns", []),
            standards=[SkillStandard.from_dict(s) for s in data.get("standards", [])],
            guidelines=[SkillGuideline.from_dict(g) for g in data.get("guidelines", [])],
            constraints=[SkillConstraint.from_dict(c) for c in data.get("constraints", [])],
            author=data.get("author", "vetinari"),
            tags=data.get("tags", []),
            enabled=data.get("enabled", True),
            deprecated=data.get("deprecated", False),
            deprecated_by=data.get("deprecated_by", ""),
        )
