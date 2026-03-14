"""Explain Agent module."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def is_explainability_enabled() -> bool:
    """Check if explainability is currently enabled."""
    return os.environ.get("EXPLAINABILITY_ENABLED", "true").lower() in ("1", "true", "yes")


# Keep for backwards compatibility
EXPLAINABILITY_ENABLED = is_explainability_enabled()


@dataclass
class ExplanationBlock:
    """A single explanation block for a plan or subtask."""

    id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    target_id: str = ""  # plan_id or subtask_id
    domain: str = "general"  # planning, coding, data_processing, infra, docs, ai_experiments, research
    content: str = ""
    confidence: float = 0.5  # 0.0 - 1.0
    sources: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sanitized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplanationBlock:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PlanExplanation:
    """Explanation for an entire plan."""

    plan_id: str = ""
    plan_version: int = 1
    blocks: list[ExplanationBlock] = field(default_factory=list)
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict.

        Returns:
            The result string.
        """
        data = asdict(self)
        data["blocks"] = [b.to_dict() for b in self.blocks]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanExplanation:
        """Create from dict.

        Returns:
            The PlanExplanation result.
        """
        if "blocks" in data and isinstance(data["blocks"], list):
            data["blocks"] = [ExplanationBlock.from_dict(b) if isinstance(b, dict) else b for b in data["blocks"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SubtaskExplanation:
    """Explanation for a specific subtask."""

    subtask_id: str = ""
    subtask_description: str = ""
    blocks: list[ExplanationBlock] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict.

        Returns:
            The result string.
        """
        data = asdict(self)
        data["blocks"] = [b.to_dict() for b in self.blocks]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtaskExplanation:
        """Create from dict.

        Returns:
            The SubtaskExplanation result.
        """
        if "blocks" in data and isinstance(data["blocks"], list):
            data["blocks"] = [ExplanationBlock.from_dict(b) if isinstance(b, dict) else b for b in data["blocks"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ExplainAgent:
    """Agent responsible for generating explanations for plans and subtasks.

    This agent produces explainability artifacts that help users understand
    why certain plans or subtasks were chosen, what factors influenced
    decisions, and what confidence the system has in its recommendations.
    """

    def __init__(self):
        # Read at instance creation time, not module load time
        self.enabled = os.environ.get("EXPLAINABILITY_ENABLED", "true").lower() in ("1", "true", "yes")
        self._domain_templates = self._load_domain_templates()

    def _load_domain_templates(self) -> dict[str, dict[str, str]]:
        """Load domain-specific explanation templates."""
        return {
            "coding": {
                "plan_summary": "This plan was selected because it minimizes complexity while ensuring maintainability. "
                "The approach favors local models for core components to reduce latency and cost, "
                "with cloud augmentation reserved for high-uncertainty tasks.",
                "key_factors": [
                    "API surface area kept minimal for easier maintenance",
                    "Local models prioritized for predictable performance",
                    "Cloud models used only for complex reasoning tasks",
                ],
            },
            "data_processing": {
                "plan_summary": "This plan prioritizes data quality and pipeline reliability. "
                "Each stage includes validation checkpoints to prevent downstream errors.",
                "key_factors": [
                    "Schema validation at ingestion stage",
                    "Data quality checks before transformation",
                    "Pipeline monitoring at each step",
                ],
            },
            "infra": {
                "plan_summary": "This infrastructure plan emphasizes observability and reliability. "
                "Health checks and metrics are configured to detect issues early.",
                "key_factors": [
                    "Comprehensive health endpoints",
                    "Metrics collection at critical points",
                    "Alerting configured for SLAs",
                ],
            },
            "docs": {
                "plan_summary": "This documentation plan ensures comprehensive API coverage "
                "with practical examples for common use cases.",
                "key_factors": [
                    "All endpoints documented with examples",
                    "Usage guides for common scenarios",
                    "Validation against actual code",
                ],
            },
            "ai_experiments": {
                "plan_summary": "This experiment plan is designed to compare model performance "
                "across multiple dimensions including accuracy, latency, and cost.",
                "key_factors": [
                    "Controlled experiment conditions",
                    "Multiple evaluation metrics",
                    "Statistical significance consideration",
                ],
            },
            "research": {
                "plan_summary": "This research plan follows a systematic approach to gather "
                "relevant sources and synthesize actionable insights.",
                "key_factors": [
                    "Multiple source types consulted",
                    "Trade-offs explicitly analyzed",
                    "Recommendations tied to evidence",
                ],
            },
            "general": {
                "plan_summary": "This plan was selected based on a balance of risk, cost, and complexity. "
                "The approach provides adequate coverage while maintaining flexibility.",
                "key_factors": [
                    "Risk score within acceptable threshold",
                    "Cost-effective resource allocation",
                    "Clear success criteria defined",
                ],
            },
        }

    def explain_plan(self, plan) -> PlanExplanation:
        """Generate an explanation for a plan.

        Args:
            plan: Plan object to explain

        Returns:
            PlanExplanation containing blocks and summary
        """
        if not self.enabled:
            logger.debug("Explainability disabled, returning empty explanation")
            return PlanExplanation(plan_id=plan.plan_id)

        logger.info("Generating explanation for plan: %s", plan.plan_id)

        explanation = PlanExplanation(
            plan_id=plan.plan_id,
            plan_version=plan.plan_version,
            blocks=[],
            summary="",
            sources=[],
            created_at=datetime.now().isoformat(),
        )

        # Determine domain from plan
        domain = self._infer_domain(plan.goal)
        template = self._domain_templates.get(domain, self._domain_templates["general"])

        # Add risk assessment block
        risk_block = ExplanationBlock(
            target_id=plan.plan_id,
            domain="planning",
            content=f"Risk assessment: {plan.risk_level.value if hasattr(plan.risk_level, 'value') else str(plan.risk_level)} risk (score: {plan.risk_score:.2f}). "
            f"This plan has {len(plan.subtasks)} subtasks across {max([s.depth for s in plan.subtasks], default=0) + 1} depth levels.",
            confidence=0.85,
            sources=["PlanMode risk scoring"],
            sanitized=True,
        )
        explanation.blocks.append(risk_block)

        # Add domain-specific justification block
        justification_block = ExplanationBlock(
            target_id=plan.plan_id,
            domain=domain,
            content=template["plan_summary"],
            confidence=0.75,
            sources=["Domain templates", "Plan metadata"],
            sanitized=True,
        )
        explanation.blocks.append(justification_block)

        # Add key factors block
        factors_content = "Key factors considered:\n" + "\n".join([f"- {f}" for f in template["key_factors"]])
        factors_block = ExplanationBlock(
            target_id=plan.plan_id,
            domain=domain,
            content=factors_content,
            confidence=0.70,
            sources=["Plan analysis"],
            sanitized=True,
        )
        explanation.blocks.append(factors_block)

        # Add model selection rationale if available
        if hasattr(plan, "chosen_plan_id") and plan.chosen_plan_id:
            model_block = ExplanationBlock(
                target_id=plan.plan_id,
                domain="model_selection",
                content=f"Selected plan variant: {plan.chosen_plan_id}. "
                f"Justification: {plan.plan_justification or 'Auto-selected based on risk score'}",
                confidence=0.80,
                sources=["Ponder scoring", "Plan candidate evaluation"],
                sanitized=True,
            )
            explanation.blocks.append(model_block)

        # Add dependency analysis if there are subtasks
        if plan.subtasks:
            dep_count = sum(len(deps) for deps in plan.dependencies.values()) if plan.dependencies else 0
            dep_block = ExplanationBlock(
                target_id=plan.plan_id,
                domain="planning",
                content=f"Dependency analysis: {len(plan.subtasks)} subtasks with {dep_count} dependencies. "
                f"Execution will proceed in dependency-order to maximize parallelization.",
                confidence=0.90,
                sources=["Scheduler dependency analysis"],
                sanitized=True,
            )
            explanation.blocks.append(dep_block)

        # Generate human-readable summary
        explanation.summary = self._generate_summary(plan, domain, template)

        return explanation

    def explain_subtask(self, subtask, parent_plan=None) -> SubtaskExplanation:
        """Generate an explanation for a specific subtask.

        Args:
            subtask: Subtask object to explain
            parent_plan: Optional parent plan for context

        Returns:
            SubtaskExplanation containing blocks and notes
        """
        if not self.enabled:
            logger.debug("Explainability disabled, returning empty subtask explanation")
            return SubtaskExplanation(subtask_id=subtask.subtask_id)

        logger.info("Generating explanation for subtask: %s", subtask.subtask_id)

        explanation = SubtaskExplanation(
            subtask_id=subtask.subtask_id,
            subtask_description=subtask.description,
            blocks=[],
            notes="",
            created_at=datetime.now().isoformat(),
        )

        domain = subtask.domain.value if hasattr(subtask.domain, "value") else str(subtask.domain)

        # Add depth/context block
        depth_block = ExplanationBlock(
            target_id=subtask.subtask_id,
            domain="planning",
            content=f"This subtask is at depth {subtask.depth} within the plan hierarchy. "
            f"It {'has' if subtask.dependencies else 'has no'} dependencies on other subtasks.",
            confidence=0.95,
            sources=["Plan hierarchy"],
            sanitized=True,
        )
        explanation.blocks.append(depth_block)

        # Add domain-specific rationale
        template = self._domain_templates.get(domain, self._domain_templates["general"])
        domain_block = ExplanationBlock(
            target_id=subtask.subtask_id,
            domain=domain,
            content=f"Domain: {domain}. This subtask aligns with the overall {domain} strategy: "
            f"{template['plan_summary'][:100]}...",
            confidence=0.65,
            sources=["Domain templates"],
            sanitized=True,
        )
        explanation.blocks.append(domain_block)

        # Add model assignment rationale if available
        if subtask.assigned_model_id:
            model_block = ExplanationBlock(
                target_id=subtask.subtask_id,
                domain="model_selection",
                content=f"Assigned model: {subtask.assigned_model_id}. "
                f"This model was selected based on capability match and resource availability.",
                confidence=0.75,
                sources=["Ponder scoring", "Model pool"],
                sanitized=True,
            )
            explanation.blocks.append(model_block)

        # Add DoD context if available
        if subtask.definition_of_done and hasattr(subtask.definition_of_done, "criteria"):
            dod_content = "Definition of Done:\n" + "\n".join([f"- {c}" for c in subtask.definition_of_done.criteria])
            dod_block = ExplanationBlock(
                target_id=subtask.subtask_id,
                domain="planning",
                content=dod_content,
                confidence=0.90,
                sources=["Subtask metadata"],
                sanitized=True,
            )
            explanation.blocks.append(dod_block)

        # Generate notes
        explanation.notes = f"Subtask '{subtask.description}' at depth {subtask.depth} in {domain} domain."

        return explanation

    def _infer_domain(self, goal: str) -> str:
        """Infer the domain from a goal string."""
        goal_lower = goal.lower()

        # Order matters - check more specific domains first
        if any(kw in goal_lower for kw in ["ci/cd", "kubernetes", "helm", "docker", "terraform", "ansible"]):
            return "infra"
        elif any(
            kw in goal_lower
            for kw in ["etl", "data pipeline", "ingest", "data transformation", "spark", "process data"]
        ):
            return "data_processing"
        elif any(
            kw in goal_lower
            for kw in [
                "benchmark",
                "model experiment",
                "compare models",
                "evaluate models",
                "experiment",
                "comparison",
                "evaluate",
                "compare model",
            ]
        ):
            return "ai_experiments"
        elif any(kw in goal_lower for kw in ["research", "literature", "study analysis", "investigation"]):
            return "research"
        elif any(kw in goal_lower for kw in ["api documentation", "user guide", "readme", "technical docs"]):
            return "docs"
        elif (
            any(kw in goal_lower for kw in ["unit test", "integration test", "test suite", "pytest"])
            or any(kw in goal_lower for kw in ["implement", "create feature", "build module", "develop", "build"])
            or any(
                kw in goal_lower
                for kw in [
                    "web app",
                    "rest api",
                    "python",
                    "javascript",
                    "java",
                    "code",
                    "function",
                    "module",
                    "class",
                    "api endpoint",
                ]
            )
        ):
            return "coding"
        else:
            return "general"

    def _generate_summary(self, plan, domain: str, template: dict[str, str]) -> str:
        """Generate a human-readable summary for the plan."""
        risk_label = plan.risk_level.value if hasattr(plan.risk_level, "value") else str(plan.risk_level)

        summary_parts = [
            f"Plan: {plan.goal[:80]}{'...' if len(plan.goal) > 80 else ''}",
            f"Domain: {domain}",
            f"Risk: {risk_label} ({plan.risk_score:.2f})",
            f"Subtasks: {len(plan.subtasks)}",
            f"Auto-approved: {'Yes' if plan.auto_approved else 'No'}",
        ]

        if plan.plan_justification:
            summary_parts.append(f"Rationale: {plan.plan_justification[:100]}...")

        return " | ".join(summary_parts)

    def sanitize_explanation(self, explanation: PlanExplanation) -> PlanExplanation:
        """Remove sensitive information from an explanation for public exposure.

        Args:
            explanation: The explanation to sanitize

        Returns:
            A sanitized copy of the explanation
        """
        sanitized = PlanExplanation(
            plan_id=explanation.plan_id,
            plan_version=explanation.plan_version,
            blocks=[],
            summary=explanation.summary,  # Summary is pre-sanitized
            sources=explanation.sources,
            created_at=explanation.created_at,
        )

        for block in explanation.blocks:
            if block.sanitized:
                sanitized.blocks.append(block)
            else:
                # Create sanitized version
                sanitized_block = ExplanationBlock(
                    id=block.id,
                    target_id=block.target_id,
                    domain=block.domain,
                    content="[Content sanitized for public exposure]",
                    confidence=block.confidence,
                    sources=["[Sources sanitized]"],
                    timestamp=block.timestamp,
                    sanitized=True,
                )
                sanitized.blocks.append(sanitized_block)

        return sanitized


_explain_agent: ExplainAgent | None = None


def get_explain_agent() -> ExplainAgent:
    """Get or create the global ExplainAgent instance.

    Returns:
        The ExplainAgent result.
    """
    global _explain_agent
    if _explain_agent is None:
        _explain_agent = ExplainAgent()
    return _explain_agent
