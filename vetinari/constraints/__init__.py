"""Vetinari Constraints Package.

==============================
Unified constraint architecture for agent delegation rules, resource limits,
quality gates, and output validation.

Usage::

    from vetinari.constraints import get_constraint_registry

    registry = get_constraint_registry()
    constraints = registry.get_constraints_for_agent("WORKER", mode="build")
"""

from __future__ import annotations

from vetinari.constraints.architecture import (  # noqa: VET123 — get_constraint has no external callers but removing causes VET120
    get_constraint,
    validate_delegation,
)
from vetinari.constraints.registry import (  # noqa: VET123 — reset_constraint_registry has no external callers but removing causes VET120
    get_constraint_registry,
    reset_constraint_registry,
)
from vetinari.constraints.resources import (  # noqa: VET123 - barrel export preserves public import compatibility
    ResourceConstraint,
    get_resource_constraint,
)
from vetinari.constraints.style import (  # noqa: VET123 — STYLE_CONSTRAINTS imported from vetinari.constraints in tests
    STYLE_CONSTRAINTS,
    StyleConstraint,
    StyleRule,
    get_style_domain,
    get_style_rules,
    validate_output_style,
)
from vetinari.validation.quality_gates import (  # noqa: VET123 - barrel export preserves public import compatibility
    QualityGate,
    check_quality_gate,
    get_quality_gate,
)

__all__ = [
    "STYLE_CONSTRAINTS",
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
