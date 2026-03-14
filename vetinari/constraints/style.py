"""Style Constraints.

==================
Defines document and code style rules that agents must follow when producing
output.  Style constraints are *advisory* — violations are logged and
returned in results but do not block execution.

Usage::

    from vetinari.constraints.style import get_style_rules, validate_output_style

    rules = get_style_rules("documentation")
    issues = validate_output_style(output_text, "code")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StyleRule:
    """A single style rule."""

    rule_id: str
    description: str
    pattern: str = ""  # Regex pattern to detect violation
    severity: str = "warning"  # "warning" or "info"
    applies_to: str = "all"  # "code", "documentation", "creative", "all"


@dataclass
class StyleConstraint:
    """Style constraints for a specific output domain."""

    domain: str  # "code", "documentation", "creative", "all"
    rules: list[StyleRule] = field(default_factory=list)
    max_line_length: int = 120
    require_headings: bool = False
    require_sections: bool = False
    forbidden_phrases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default style rules
# ---------------------------------------------------------------------------

_CODE_RULES: list[StyleRule] = [
    StyleRule(
        rule_id="code-no-todo",
        description="Avoid TODO/FIXME comments in final output",
        pattern=r"(?i)\b(TODO|FIXME|HACK|XXX)\b",
        severity="warning",
        applies_to="code",
    ),
    StyleRule(
        rule_id="code-no-print-debug",
        description="Avoid debug print statements",
        pattern=r"^\s*print\s*\(\s*['\"]debug",
        severity="warning",
        applies_to="code",
    ),
    StyleRule(
        rule_id="code-no-bare-except",
        description="Avoid bare except clauses",
        pattern=r"except\s*:",
        severity="warning",
        applies_to="code",
    ),
    StyleRule(
        rule_id="code-no-hardcoded-secrets",
        description="No hardcoded API keys or passwords",
        pattern=r"""(?i)(api_key|password|secret|token)\s*=\s*['"][^'"]{8,}['"]""",
        severity="warning",
        applies_to="code",
    ),
]

_DOC_RULES: list[StyleRule] = [
    StyleRule(
        rule_id="doc-no-placeholder",  # noqa: VET034
        description="No placeholder text in documentation",  # noqa: VET034
        pattern=r"(?i)\b(lorem ipsum|TBD|placeholder|TODO)\b",  # noqa: VET034
        severity="warning",
        applies_to="documentation",
    ),
    StyleRule(
        rule_id="doc-consistent-headings",
        description="Use consistent heading styles (# not ===)",
        pattern=r"^={3,}\s*$",
        severity="info",
        applies_to="documentation",
    ),
]

_CREATIVE_RULES: list[StyleRule] = [
    StyleRule(
        rule_id="creative-no-cliche-opener",
        description="Avoid cliched openers in creative writing",
        pattern=r"(?i)^(once upon a time|it was a dark and stormy|in a world where)",
        severity="info",
        applies_to="creative",
    ),
]


# ---------------------------------------------------------------------------
# Style constraint sets per domain
# ---------------------------------------------------------------------------

STYLE_CONSTRAINTS: dict[str, StyleConstraint] = {
    "code": StyleConstraint(
        domain="code",
        rules=_CODE_RULES,
        max_line_length=120,
        forbidden_phrases=["FIXME", "HACK"],
    ),
    "documentation": StyleConstraint(
        domain="documentation",
        rules=_DOC_RULES,
        max_line_length=100,
        require_headings=True,
        require_sections=True,
        forbidden_phrases=["lorem ipsum", "TBD"],  # noqa: VET034
    ),
    "creative": StyleConstraint(
        domain="creative",
        rules=_CREATIVE_RULES,
        max_line_length=0,  # No limit for creative
    ),
}


# ---------------------------------------------------------------------------
# Agent-type to style domain mapping
# ---------------------------------------------------------------------------

AGENT_STYLE_DOMAIN: dict[str, str] = {
    "BUILDER": "code",
    "QUALITY": "code",
    "ARCHITECT": "code",
    "PLANNER": "documentation",
    "OPERATIONS": "documentation",  # default; creative mode overrides
    "CONSOLIDATED_RESEARCHER": "documentation",
    "ORACLE": "documentation",
    "CONSOLIDATED_ORACLE": "documentation",
    "ORCHESTRATOR": "documentation",
}

# Mode-specific overrides
_MODE_STYLE_OVERRIDES: dict[str, str] = {
    "creative_writing": "creative",
    "image_generation": "creative",
    "code_review": "code",
    "security_audit": "code",
    "test_generation": "code",
    "documentation": "documentation",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_style_domain(agent_type: str, mode: str | None = None) -> str:
    """Determine the style domain for an agent/mode combination.

    Args:
        agent_type: The agent type.
        mode: The mode.

    Returns:
        The result string.
    """
    if mode and mode in _MODE_STYLE_OVERRIDES:
        return _MODE_STYLE_OVERRIDES[mode]
    return AGENT_STYLE_DOMAIN.get(agent_type, "documentation")


def get_style_rules(domain: str) -> StyleConstraint:
    """Get style constraints for a domain."""
    return STYLE_CONSTRAINTS.get(domain, STYLE_CONSTRAINTS.get("documentation"))


def validate_output_style(
    text: str,
    domain: str,
    agent_type: str | None = None,
    mode: str | None = None,
) -> list[dict[str, str]]:
    """Validate text against style rules for the given domain.

    Returns list of style issues: [{"rule_id": ..., "description": ..., "severity": ...}]

    Args:
        text: The text.
        domain: The domain.
        agent_type: The agent type.
        mode: The mode.

    Returns:
        The result string.
    """
    if not text:
        return []

    effective_domain = domain
    if agent_type:
        effective_domain = get_style_domain(agent_type, mode)

    constraint = get_style_rules(effective_domain)
    if constraint is None:
        return []

    issues: list[dict[str, str]] = []

    # Check rules
    for rule in constraint.rules:
        if rule.applies_to not in (effective_domain, "all"):
            continue
        if rule.pattern:
            try:
                if re.search(rule.pattern, text, re.MULTILINE):
                    issues.append(
                        {
                            "rule_id": rule.rule_id,
                            "description": rule.description,
                            "severity": rule.severity,
                        }
                    )
            except re.error:  # noqa: VET022
                pass

    # Check line length (only if max_line_length > 0)
    if constraint.max_line_length > 0:
        for i, line in enumerate(text.split("\n"), 1):
            if len(line) > constraint.max_line_length:
                issues.append(
                    {
                        "rule_id": "line-too-long",
                        "description": f"Line {i} exceeds {constraint.max_line_length} chars ({len(line)})",
                        "severity": "info",
                    }
                )
                break  # Only report first occurrence

    # Check forbidden phrases
    for phrase in constraint.forbidden_phrases:
        if phrase.lower() in text.lower():
            issues.append(
                {
                    "rule_id": "forbidden-phrase",
                    "description": f"Contains forbidden phrase: '{phrase}'",
                    "severity": "warning",
                }
            )

    return issues
