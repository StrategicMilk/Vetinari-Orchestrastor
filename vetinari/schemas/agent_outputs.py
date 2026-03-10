"""
Typed Output Schemas (C6)
==========================
Pydantic BaseModel schemas for every agent mode output. Used for
validation after _infer_json() calls with auto-repair on failure.

Each schema defines the expected structure of agent mode outputs,
enabling structured validation and IDE autocompletion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback: use dataclasses to mimic Pydantic when not installed
    from dataclasses import dataclass, field as dc_field

    class _BaseMeta(type):
        """Metaclass that adds model_validate and model_json_schema."""
        def __new__(mcs, name, bases, namespace):
            cls = super().__new__(mcs, name, bases, namespace)
            return cls

    class BaseModel(metaclass=_BaseMeta):  # type: ignore[no-redef]
        """Minimal Pydantic-like base when pydantic is not installed."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @classmethod
        def model_validate(cls, data: dict) -> "BaseModel":
            return cls(**data)

        @classmethod
        def model_json_schema(cls) -> dict:
            return {"type": "object", "properties": {}}

        def model_dump(self) -> dict:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def Field(default=None, **kwargs):  # type: ignore[no-redef]
        return default


# ── Planner Agent Schemas ─────────────────────────────────────────────

class PlanOutput(BaseModel):
    """Output schema for PlannerAgent plan mode."""
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")
    estimated_steps: int = Field(default=0)
    complexity: str = Field(default="medium")
    dependencies: List[str] = Field(default_factory=list)


class ClarifyOutput(BaseModel):
    """Output schema for PlannerAgent clarify mode."""
    questions: List[str] = Field(default_factory=list)
    ambiguities: List[Dict[str, str]] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5)


class SummariseOutput(BaseModel):
    """Output schema for PlannerAgent summarise mode."""
    summary: str = Field(default="")
    key_points: List[str] = Field(default_factory=list)
    decisions_made: List[str] = Field(default_factory=list)
    open_items: List[str] = Field(default_factory=list)


class ConsolidateOutput(BaseModel):
    """Output schema for PlannerAgent consolidate mode."""
    consolidated: str = Field(default="")
    entries_processed: int = Field(default=0)
    themes: List[str] = Field(default_factory=list)


# ── Researcher Agent Schemas ─────────────────────────────────────────

class CodeDiscoveryOutput(BaseModel):
    """Output schema for ResearcherAgent code_discovery mode."""
    files: List[Dict[str, Any]] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    architecture: str = Field(default="")
    dependencies: List[str] = Field(default_factory=list)


class DomainResearchOutput(BaseModel):
    """Output schema for ResearcherAgent domain_research mode."""
    findings: List[Dict[str, str]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    confidence: float = Field(default=0.5)
    recommendations: List[str] = Field(default_factory=list)


class APILookupOutput(BaseModel):
    """Output schema for ResearcherAgent api_lookup mode."""
    endpoints: List[Dict[str, Any]] = Field(default_factory=list)
    documentation: str = Field(default="")
    examples: List[str] = Field(default_factory=list)
    version: str = Field(default="")


# ── Oracle Agent Schemas ──────────────────────────────────────────────

class ArchitectureOutput(BaseModel):
    """Output schema for OracleAgent architecture mode."""
    analysis: str = Field(default="")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    risks: List[Dict[str, str]] = Field(default_factory=list)
    score: float = Field(default=0.5)


class RiskAssessmentOutput(BaseModel):
    """Output schema for OracleAgent risk_assessment mode."""
    risks: List[Dict[str, Any]] = Field(default_factory=list)
    overall_risk: str = Field(default="medium")
    mitigations: List[Dict[str, str]] = Field(default_factory=list)
    risk_matrix: Dict[str, Any] = Field(default_factory=dict)


class ContrarianReviewOutput(BaseModel):
    """Output schema for OracleAgent contrarian_review mode."""
    challenges: List[Dict[str, str]] = Field(default_factory=list)
    blind_spots: List[str] = Field(default_factory=list)
    alternative_approaches: List[Dict[str, str]] = Field(default_factory=list)
    verdict: str = Field(default="")


# ── Builder Agent Schemas ─────────────────────────────────────────────

class BuildOutput(BaseModel):
    """Output schema for BuilderAgent build mode."""
    scaffold_code: str = Field(default="")
    tests: List[Dict[str, str]] = Field(default_factory=list)
    artifacts: List[Dict[str, str]] = Field(default_factory=list)
    implementation_notes: List[str] = Field(default_factory=list)
    summary: str = Field(default="")


class ImageGenerationOutput(BaseModel):
    """Output schema for BuilderAgent image_generation mode."""
    images: List[Dict[str, Any]] = Field(default_factory=list)
    spec: Dict[str, Any] = Field(default_factory=dict)
    sd_available: bool = Field(default=False)
    count: int = Field(default=0)


# ── Quality Agent Schemas ─────────────────────────────────────────────

class CodeReviewOutput(BaseModel):
    """Output schema for QualityAgent code_review mode."""
    score: float = Field(default=0.5)
    summary: str = Field(default="")
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SecurityAuditOutput(BaseModel):
    """Output schema for QualityAgent security_audit mode."""
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")
    overall_risk: str = Field(default="medium")
    score: float = Field(default=0.5)
    heuristic_count: int = Field(default=0)


class TestGenerationOutput(BaseModel):
    """Output schema for QualityAgent test_generation mode."""
    tests: str = Field(default="")
    test_count: int = Field(default=0)
    coverage_estimate: float = Field(default=0.0)
    fixtures: List[str] = Field(default_factory=list)
    edge_cases_covered: List[str] = Field(default_factory=list)


class SimplificationOutput(BaseModel):
    """Output schema for QualityAgent simplification mode."""
    score: float = Field(default=0.5)
    complexity_issues: List[Dict[str, str]] = Field(default_factory=list)
    overall_recommendations: List[str] = Field(default_factory=list)
    estimated_line_reduction: int = Field(default=0)


# ── Operations Agent Schemas ──────────────────────────────────────────

class DocumentationOutput(BaseModel):
    """Output schema for OperationsAgent documentation mode."""
    content: str = Field(default="")
    type: str = Field(default="api_reference")
    sections: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CostAnalysisOutput(BaseModel):
    """Output schema for OperationsAgent cost_analysis mode."""
    comparisons: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: str = Field(default="")
    estimated_tokens: int = Field(default=0)
    analysis: str = Field(default="")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)


class ExperimentOutput(BaseModel):
    """Output schema for OperationsAgent experiment mode."""
    experiment: Dict[str, Any] = Field(default_factory=dict)
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    variants: List[Dict[str, str]] = Field(default_factory=list)
    sample_size: int = Field(default=0)
    duration_days: int = Field(default=7)


class ErrorRecoveryOutput(BaseModel):
    """Output schema for OperationsAgent error_recovery mode."""
    root_cause: str = Field(default="")
    category: str = Field(default="")
    severity: str = Field(default="medium")
    recovery_strategy: Dict[str, Any] = Field(default_factory=dict)
    matched_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    prevention: List[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    """Output schema for OperationsAgent synthesis mode."""
    synthesis: str = Field(default="")
    sources_used: List[str] = Field(default_factory=list)
    conflicts_resolved: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5)


class MonitorOutput(BaseModel):
    """Output schema for OperationsAgent monitor mode."""
    status: str = Field(default="healthy")
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metrics_summary: Dict[str, Any] = Field(default_factory=dict)


# ── Schema Registry ──────────────────────────────────────────────────

OUTPUT_SCHEMAS: Dict[str, type] = {
    # Planner
    "plan": PlanOutput,
    "clarify": ClarifyOutput,
    "summarise": SummariseOutput,
    "consolidate": ConsolidateOutput,
    # Researcher
    "code_discovery": CodeDiscoveryOutput,
    "domain_research": DomainResearchOutput,
    "api_lookup": APILookupOutput,
    # Oracle
    "architecture": ArchitectureOutput,
    "risk_assessment": RiskAssessmentOutput,
    "contrarian_review": ContrarianReviewOutput,
    # Builder
    "build": BuildOutput,
    "image_generation": ImageGenerationOutput,
    # Quality
    "code_review": CodeReviewOutput,
    "security_audit": SecurityAuditOutput,
    "test_generation": TestGenerationOutput,
    "simplification": SimplificationOutput,
    # Operations
    "documentation": DocumentationOutput,
    "cost_analysis": CostAnalysisOutput,
    "experiment": ExperimentOutput,
    "error_recovery": ErrorRecoveryOutput,
    "synthesis": SynthesisOutput,
    "monitor": MonitorOutput,
}


def validate_output(mode: str, data: Any) -> Optional[BaseModel]:
    """Validate agent output against the schema for the given mode.

    Returns a validated model instance, or None if no schema exists
    or validation fails.
    """
    schema_cls = OUTPUT_SCHEMAS.get(mode)
    if schema_cls is None:
        return None

    if not isinstance(data, dict):
        return None

    try:
        return schema_cls.model_validate(data)
    except Exception:
        # Auto-repair: try to construct with available fields
        try:
            return schema_cls(**{
                k: v for k, v in data.items()
                if k in schema_cls.__annotations__
            } if hasattr(schema_cls, "__annotations__") else data)
        except Exception:
            return None


def get_schema_for_mode(mode: str) -> Optional[type]:
    """Get the output schema class for a mode."""
    return OUTPUT_SCHEMAS.get(mode)
