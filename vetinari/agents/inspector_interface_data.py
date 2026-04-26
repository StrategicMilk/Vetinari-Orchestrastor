"""Inspector agent interface contract — quality review, security, and testing.

Imported by vetinari.agents.interfaces — do not use directly.
"""

from __future__ import annotations

from vetinari.agents.interface_types import AgentInterface, Capability, CapabilityType
from vetinari.types import AgentType

# ===== INSPECTOR INTERFACE =====
INSPECTOR_INTERFACE = AgentInterface(
    agent_name="Inspector",
    agent_type=AgentType.INSPECTOR.value,
    version="1.0.0",
    capabilities=[
        Capability(
            name="code_review",
            type=CapabilityType.VERIFICATION,
            description="Review code quality, logic, and maintainability",
            input_schema={
                "type": "object",
                "properties": {
                    "artifacts": {"type": "array"},
                    "criteria": {"type": "array"},
                },
                "required": ["artifacts"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "verdict": {"type": "string"},
                    "quality_score": {"type": "number"},
                    "improvements": {"type": "array"},
                },
            },
        ),
        Capability(
            name="security_audit",
            type=CapabilityType.GOVERNANCE,
            description="Audit code for security vulnerabilities and policy violations",
            input_schema={
                "type": "object",
                "properties": {
                    "artifacts": {"type": "array"},
                    "threat_model": {"type": "object"},
                },
                "required": ["artifacts"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "vulnerabilities": {"type": "array"},
                    "severity_scores": {"type": "object"},
                    "remediation": {"type": "array"},
                },
            },
        ),
        Capability(
            name="test_generation",
            type=CapabilityType.TESTING,
            description="Generate unit, integration, and e2e test suites",
            input_schema={
                "type": "object",
                "properties": {
                    "module": {"type": "string"},
                    "coverage_target": {"type": "number"},
                    "test_types": {"type": "array"},
                },
                "required": ["module"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "tests": {"type": "array"},
                    "coverage_estimate": {"type": "number"},
                },
            },
        ),
        Capability(
            name="code_simplification",
            type=CapabilityType.OPTIMIZATION,
            description="Simplify and refactor code while preserving behaviour",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "simplification_goals": {"type": "array"},
                },
                "required": ["code"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "simplified_code": {"type": "string"},
                    "changes_summary": {"type": "string"},
                },
            },
        ),
    ],
    required_context=["quality_standards", "analysis_tools"],
    error_codes={
        "NO_ARTIFACTS": "No code artifacts provided for review",
        "UNSUPPORTED_LANGUAGE": "The code language is not supported",
    },
)
