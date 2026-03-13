"""Vetinari Constraints Package.

==============================
Unified constraint architecture for agent delegation rules, resource limits,
quality gates, and output validation.

Usage::

    from vetinari.constraints import get_constraint_registry

    registry = get_constraint_registry()
    constraints = registry.get_constraints_for_agent("BUILDER", mode="implement")
"""

from __future__ import annotations

from vetinari.constraints.architecture import (
    ARCHITECTURE_CONSTRAINTS,
    ArchitectureConstraint,
    get_constraint,
    validate_delegation,
)
from vetinari.constraints.quality_gates import (
    QUALITY_GATES,
    QualityGate,
    check_quality_gate,
    get_quality_gate,
)
from vetinari.constraints.registry import (
    AgentConstraints,
    ConstraintRegistry,
    get_constraint_registry,
    reset_constraint_registry,
)
from vetinari.constraints.resources import (
    AGENT_RESOURCE_LIMITS,
    ResourceConstraint,
    get_resource_constraint,
)
from vetinari.constraints.style import (
    STYLE_CONSTRAINTS,
    StyleConstraint,
    StyleRule,
    get_style_domain,
    get_style_rules,
    validate_output_style,
)

__all__ = [
    # Resources
    "AGENT_RESOURCE_LIMITS",
    # Architecture
    "ARCHITECTURE_CONSTRAINTS",
    # Quality gates
    "QUALITY_GATES",
    # Style
    "STYLE_CONSTRAINTS",
    # Registry
    "AgentConstraints",
    "ArchitectureConstraint",
    "ConstraintRegistry",
    "QualityGate",
    "ResourceConstraint",
    "StyleConstraint",
    "StyleRule",
    "check_quality_gate",
    "get_constraint",
    "get_constraint_registry",
    "get_quality_gate",
    "get_resource_constraint",
    "get_style_domain",
    "get_style_rules",
    "reset_constraint_registry",
    "validate_delegation",
    "validate_output_style",
]
