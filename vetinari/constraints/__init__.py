"""
Vetinari Constraints Package
==============================
Unified constraint architecture for agent delegation rules, resource limits,
quality gates, and output validation.

Usage::

    from vetinari.constraints import get_constraint_registry

    registry = get_constraint_registry()
    constraints = registry.get_constraints_for_agent("BUILDER", mode="implement")
"""

from vetinari.constraints.architecture import (
    ARCHITECTURE_CONSTRAINTS,
    ArchitectureConstraint,
    validate_delegation,
    get_constraint,
)
from vetinari.constraints.resources import (
    AGENT_RESOURCE_LIMITS,
    ResourceConstraint,
    get_resource_constraint,
)
from vetinari.constraints.quality_gates import (
    QUALITY_GATES,
    QualityGate,
    check_quality_gate,
    get_quality_gate,
)
from vetinari.constraints.style import (
    STYLE_CONSTRAINTS,
    StyleConstraint,
    StyleRule,
    get_style_domain,
    get_style_rules,
    validate_output_style,
)
from vetinari.constraints.registry import (
    AgentConstraints,
    ConstraintRegistry,
    get_constraint_registry,
    reset_constraint_registry,
)

__all__ = [
    # Architecture
    "ARCHITECTURE_CONSTRAINTS",
    "ArchitectureConstraint",
    "validate_delegation",
    "get_constraint",
    # Resources
    "AGENT_RESOURCE_LIMITS",
    "ResourceConstraint",
    "get_resource_constraint",
    # Quality gates
    "QUALITY_GATES",
    "QualityGate",
    "check_quality_gate",
    "get_quality_gate",
    # Style
    "STYLE_CONSTRAINTS",
    "StyleConstraint",
    "StyleRule",
    "get_style_domain",
    "get_style_rules",
    "validate_output_style",
    # Registry
    "AgentConstraints",
    "ConstraintRegistry",
    "get_constraint_registry",
    "reset_constraint_registry",
]
