"""Universal skill output contract for all Vetinari agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Verdict(Enum):
    PASS = "pass"  # noqa: S105
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"


class DataProvenance(Enum):
    MEASURED = "measured"
    INFERRED = "inferred"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class ArtifactType(Enum):
    CODE = "code"
    TEST = "test"
    CONFIG = "config"
    DOCS = "docs"
    DATA = "data"
    REPORT = "report"


@dataclass
class Finding:
    id: str  # e.g., "SEC-001", "TEST-003"
    severity: Severity
    category: str  # Agent-specific category
    title: str  # One-line summary
    location: str  # File:line or component name
    evidence: str  # ACTUAL code/data — NOT invented
    recommendation: str  # SPECIFIC fix — NOT generic advice
    confidence: float = 0.8  # 0.0-1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "category": self.category,
            "title": self.title,
            "location": self.location,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }


@dataclass
class Artifact:
    filename: str
    content: str
    artifact_type: ArtifactType
    language: str = "text"  # e.g., "python", "yaml", "markdown"
    validated: bool = False

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "content": self.content[:500],  # truncate for serialization
            "artifact_type": self.artifact_type.value,
            "language": self.language,
            "validated": self.validated,
        }


@dataclass
class SkillOutput:
    """Universal structured output for all Vetinari agents and skills."""

    agent_type: str
    task_summary: str
    verdict: Verdict
    confidence: float
    findings: list[Finding] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    artifacts: list[Artifact] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    data_provenance: DataProvenance = DataProvenance.UNKNOWN
    self_check_passed: bool = False
    self_check_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "task_summary": self.task_summary,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "findings": [f.to_dict() for f in self.findings],
            "scores": self.scores,
            "overall_score": self.overall_score,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "sources": self.sources,
            "data_provenance": self.data_provenance.value,
            "self_check_passed": self.self_check_passed,
            "self_check_issues": self.self_check_issues,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SkillOutput:
        return cls(
            agent_type=d.get("agent_type", "unknown"),
            task_summary=d.get("task_summary", ""),
            verdict=Verdict(d.get("verdict", "needs_review")),
            confidence=d.get("confidence", 0.0),
            findings=[
                Finding(
                    id=f.get("id", ""),
                    severity=Severity(f.get("severity", "info")),
                    category=f.get("category", ""),
                    title=f.get("title", ""),
                    location=f.get("location", ""),
                    evidence=f.get("evidence", ""),
                    recommendation=f.get("recommendation", ""),
                    confidence=f.get("confidence", 0.8),
                )
                for f in d.get("findings", [])
            ],
            scores=d.get("scores", {}),
            overall_score=d.get("overall_score", 0.0),
            artifacts=[
                Artifact(
                    filename=a.get("filename", ""),
                    content=a.get("content", ""),
                    artifact_type=ArtifactType(a.get("artifact_type", "report")),
                    language=a.get("language", "text"),
                    validated=a.get("validated", False),
                )
                for a in d.get("artifacts", [])
            ],
            sources=d.get("sources", []),
            data_provenance=DataProvenance(d.get("data_provenance", "unknown")),
            self_check_passed=d.get("self_check_passed", False),
            self_check_issues=d.get("self_check_issues", []),
        )


# Per-agent scoring rubric definitions
SCORING_RUBRICS = {
    "RESEARCHER": {
        "source_quality": {"weight": 0.30, "description": "Quality and authority of cited sources"},
        "completeness": {"weight": 0.25, "description": "Coverage of all relevant facets"},
        "accuracy": {"weight": 0.25, "description": "Claims backed by evidence"},
        "actionability": {"weight": 0.20, "description": "Concrete, specific recommendations"},
    },
    "BUILDER": {
        "syntax_validity": {"weight": 0.25, "description": "Code parses/compiles correctly"},
        "completeness": {"weight": 0.25, "description": "All functions fully implemented"},
        "test_coverage": {"weight": 0.20, "description": "Edge cases and error paths tested"},
        "style_compliance": {"weight": 0.15, "description": "Docstrings, PEP 8, type hints"},
        "error_handling": {"weight": 0.15, "description": "Specific exceptions, recovery logic"},
    },
    "TESTER": {
        "test_validity": {"weight": 0.30, "description": "Assertions test meaningful conditions"},
        "coverage_breadth": {"weight": 0.25, "description": "Percentage of functions covered"},
        "edge_cases": {"weight": 0.20, "description": "Boundary, null, overflow tested"},
        "fixture_quality": {"weight": 0.15, "description": "Parameterized fixtures, factories"},
        "independence": {"weight": 0.10, "description": "Tests are fully isolated"},
    },
    "ARCHITECT": {
        "risk_identification": {"weight": 0.30, "description": "Specific risks with likelihood+impact"},
        "tradeoff_analysis": {"weight": 0.25, "description": "Pros/cons with quantified comparison"},
        "feasibility": {"weight": 0.25, "description": "Technical/resource constraints addressed"},
        "actionability": {"weight": 0.20, "description": "Step-by-step implementation roadmap"},
    },
    "DOCUMENTER": {
        "accuracy": {"weight": 0.30, "description": "Docs match actual code"},
        "completeness": {"weight": 0.25, "description": "All public APIs documented"},
        "clarity": {"weight": 0.25, "description": "Clear prose with examples"},
        "structure": {"weight": 0.20, "description": "Proper TOC, cross-refs, headings"},
    },
    "RESILIENCE": {
        "error_classification": {"weight": 0.30, "description": "Errors categorized with root cause taxonomy"},
        "recovery_strategy": {"weight": 0.30, "description": "Specific fix with tested recovery path"},
        "root_cause_depth": {"weight": 0.20, "description": "Full causal chain to root cause"},
        "prevention": {"weight": 0.20, "description": "Concrete code changes to prevent recurrence"},
    },
    "META": {
        "data_backing": {"weight": 0.35, "description": "Claims backed by measured telemetry"},
        "impact_estimation": {"weight": 0.25, "description": "Quantified impact with confidence intervals"},
        "actionability": {"weight": 0.25, "description": "Concrete config/code changes"},
        "risk_assessment": {"weight": 0.15, "description": "Risk/reward ratio per recommendation"},
    },
    "PLANNER": {
        "decomposition_quality": {"weight": 0.30, "description": "Proper DAG with dependencies, parallelism"},
        "agent_assignment": {"weight": 0.25, "description": "Optimal agent-to-task matching"},
        "completeness": {"weight": 0.25, "description": "All paths including error handling"},
        "feasibility": {"weight": 0.20, "description": "Tasks achievable within resource limits"},
    },
}


def compute_overall_score(scores: dict[str, float], agent_type: str) -> float:
    """Compute weighted overall score using the agent's rubric."""
    rubric = SCORING_RUBRICS.get(agent_type, {})
    if not rubric or not scores:
        return sum(scores.values()) / max(len(scores), 1)

    total_weight = 0.0
    weighted_sum = 0.0
    for dim, config in rubric.items():
        if dim in scores:
            weighted_sum += scores[dim] * config["weight"]
            total_weight += config["weight"]

    return round(weighted_sum / max(total_weight, 0.01), 3)


# VAGUE_PATTERNS used by self-check
VAGUE_PATTERNS = [
    "consider",
    "might want to",
    "could potentially",
    "it may be helpful",
    "you should think about",
]


def self_check(output: SkillOutput) -> SkillOutput:
    """Verify output quality. Mutates and returns the output."""
    issues = []

    for finding in output.findings:
        if not finding.evidence or finding.evidence.strip() == "":
            issues.append(f"Finding {finding.id} has no evidence")
        if finding.location == "" or finding.location == "N/A":
            issues.append(f"Finding {finding.id} has no location")

    for finding in output.findings:
        for pattern in VAGUE_PATTERNS:
            if pattern in finding.recommendation.lower():
                issues.append(f"Finding {finding.id} has vague recommendation: '{pattern}'")

    for dim, score in output.scores.items():
        if score == 0.0 or score == 1.0:
            issues.append(f"Score '{dim}' is suspiciously extreme ({score})")

    if not output.task_summary:
        issues.append("Missing task_summary")
    if output.confidence == 0.0:
        issues.append("Confidence is 0.0 — was it actually computed?")

    output.self_check_passed = len(issues) == 0
    output.self_check_issues = issues
    return output
