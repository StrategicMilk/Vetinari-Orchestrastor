"""
Consolidated Oracle Agent (Phase 3)
====================================
Replaces: ORACLE + PONDER

Modes:
- architecture: Architecture decisions and design guidance
- risk_assessment: Risk analysis with likelihood/impact scoring
- ontological_analysis: Deep conceptual analysis
- contrarian_review: Challenge assumptions, find blind spots
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

logger = logging.getLogger(__name__)


class ConsolidatedOracleAgent(MultiModeAgent):
    """Unified oracle agent for architecture decisions, risk assessment, and deep deliberation."""

    MODES = {
        "architecture": "_execute_architecture",
        "risk_assessment": "_execute_risk_assessment",
        "ontological_analysis": "_execute_ontological",
        "contrarian_review": "_execute_contrarian",
    }
    DEFAULT_MODE = "architecture"
    MODE_KEYWORDS = {
        "architecture": ["architect", "design", "structure", "pattern", "component", "module"],
        "risk_assessment": ["risk", "vulnerab", "threat", "impact", "likelihood", "mitigat"],
        "ontological_analysis": ["ontolog", "concept", "fundament", "deep analy", "ponder", "deliberat", "reflect"],
        "contrarian_review": ["contrarian", "challenge", "assumption", "blind spot", "devil", "critique"],
    }
    LEGACY_TYPE_TO_MODE = {
        "ORACLE": "architecture",
        "PONDER": "ontological_analysis",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.CONSOLIDATED_ORACLE, config)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Oracle — the system's strategic advisor. "
            "You provide architecture guidance, risk assessment, deep analysis, "
            "and contrarian perspectives to improve decision quality."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "architecture": (
                "You are Vetinari's Architecture Oracle. Your role is to:\n"
                "- Evaluate architectural decisions and tradeoffs\n"
                "- Recommend design patterns and system structures\n"
                "- Identify coupling, cohesion, and scalability concerns\n"
                "- Provide concrete design guidelines with rationale\n\n"
                "Be specific and cite established patterns (SOLID, DDD, etc.)."
            ),
            "risk_assessment": (
                "You are Vetinari's Risk Analyst. Your role is to:\n"
                "- Identify risks with likelihood and impact scores (1-5)\n"
                "- Propose mitigation strategies for each risk\n"
                "- Assess overall risk posture\n"
                "- Prioritize risks by composite score\n\n"
                "Use a structured risk matrix approach."
            ),
            "ontological_analysis": (
                "You are Vetinari's Deep Thinker (Ponder). Your role is to:\n"
                "- Conduct deep conceptual analysis of complex problems\n"
                "- Examine foundational assumptions and first principles\n"
                "- Explore philosophical and theoretical implications\n"
                "- Synthesize insights from multiple domains\n\n"
                "Take time to think deeply. Quality over speed."
            ),
            "contrarian_review": (
                "You are Vetinari's Contrarian Reviewer. Your role is to:\n"
                "- Challenge assumptions and conventional wisdom\n"
                "- Identify blind spots and unstated dependencies\n"
                "- Propose alternative approaches others may have missed\n"
                "- Play devil's advocate constructively\n\n"
                "Be rigorous but fair. Critique ideas, not people."
            ),
        }
        return prompts.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        if output is None:
            return VerificationResult(passed=False, issues=[{"message": "No output"}], score=0.0)
        if isinstance(output, dict):
            has_substance = bool(
                output.get("analysis") or output.get("recommendations")
                or output.get("risks") or output.get("insights")
            )
            return VerificationResult(passed=has_substance, score=0.8 if has_substance else 0.3)
        return VerificationResult(passed=True, score=0.6)

    def _execute_architecture(self, task: AgentTask) -> AgentResult:
        subject = task.context.get("subject", task.description)
        prompt = (
            f"Provide architectural analysis and guidance for:\n{subject}\n\n"
            "Respond as JSON:\n"
            '{"analysis": {"summary": "...", "key_decisions": [...]}, '
            '"recommendations": [{"recommendation": "...", "rationale": "...", "priority": "high|medium|low"}], '
            '"design_patterns": [...], "tradeoffs": [{"option": "...", "pros": [...], "cons": [...]}]}'
        )
        result = self._infer_json(prompt, fallback={"analysis": {"summary": subject}, "recommendations": []})
        return AgentResult(success=True, output=result, metadata={"mode": "architecture"})

    def _execute_risk_assessment(self, task: AgentTask) -> AgentResult:
        subject = task.context.get("subject", task.description)
        prompt = (
            f"Assess risks for:\n{subject}\n\n"
            "Respond as JSON:\n"
            '{"risks": [{"risk": "...", "likelihood": 3, "impact": 4, '
            '"composite_score": 12, "mitigation": "...", "category": "technical|operational|security"}], '
            '"overall_risk_level": "high|medium|low", '
            '"recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"risks": [], "overall_risk_level": "unknown"})
        return AgentResult(success=True, output=result, metadata={"mode": "risk_assessment"})

    def _execute_ontological(self, task: AgentTask) -> AgentResult:
        subject = task.context.get("subject", task.description)
        prompt = (
            f"Conduct deep ontological analysis of:\n{subject}\n\n"
            "Examine first principles, foundational assumptions, and conceptual foundations.\n\n"
            "Respond as JSON:\n"
            '{"insights": [{"insight": "...", "depth": "surface|structural|foundational", '
            '"confidence": 0.8}], "assumptions_challenged": [...], '
            '"first_principles": [...], "synthesis": "..."}'
        )
        result = self._infer_json(prompt, fallback={"insights": [], "synthesis": ""})
        return AgentResult(success=True, output=result, metadata={"mode": "ontological_analysis"})

    def _execute_contrarian(self, task: AgentTask) -> AgentResult:
        subject = task.context.get("subject", task.description)
        prompt = (
            f"Provide a contrarian review of:\n{subject}\n\n"
            "Challenge assumptions, identify blind spots, propose alternatives.\n\n"
            "Respond as JSON:\n"
            '{"challenges": [{"assumption": "...", "challenge": "...", "severity": "high|medium|low"}], '
            '"blind_spots": [...], "alternatives": [{"approach": "...", "rationale": "...", "feasibility": 0.7}], '
            '"recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"challenges": [], "blind_spots": []})
        return AgentResult(success=True, output=result, metadata={"mode": "contrarian_review"})

    def get_capabilities(self) -> List[str]:
        return [
            "architecture_decision", "design_pattern_recommendation",
            "risk_assessment", "risk_mitigation",
            "ontological_analysis", "first_principles_thinking",
            "contrarian_review", "assumption_challenging",
        ]


# Singleton
_consolidated_oracle_agent: Optional[ConsolidatedOracleAgent] = None


def get_consolidated_oracle_agent(config: Optional[Dict[str, Any]] = None) -> ConsolidatedOracleAgent:
    global _consolidated_oracle_agent
    if _consolidated_oracle_agent is None:
        _consolidated_oracle_agent = ConsolidatedOracleAgent(config)
    return _consolidated_oracle_agent
