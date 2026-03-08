"""
Consolidated Researcher Agent (Phase 3)
========================================
Replaces: EXPLORER + RESEARCHER + LIBRARIAN

Modes:
- code_discovery: Fast code/document/pattern extraction (from Explorer)
- domain_research: Feasibility analysis, competitive analysis (from Researcher)
- api_lookup: API/docs lookup, library discovery (from Librarian)
- lateral_thinking: Creative problem-solving and alternative approaches
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

logger = logging.getLogger(__name__)


class ConsolidatedResearcherAgent(MultiModeAgent):
    """Unified research agent for code discovery, domain research, and API lookup."""

    MODES = {
        "code_discovery": "_execute_code_discovery",
        "domain_research": "_execute_domain_research",
        "api_lookup": "_execute_api_lookup",
        "lateral_thinking": "_execute_lateral_thinking",
    }
    DEFAULT_MODE = "code_discovery"
    MODE_KEYWORDS = {
        "code_discovery": ["code", "file", "class", "function", "pattern", "codebase", "discover", "explore", "search code"],
        "domain_research": ["research", "feasib", "competit", "market", "domain", "analys"],
        "api_lookup": ["api", "library", "framework", "package", "documentation", "docs", "license", "dependency"],
        "lateral_thinking": ["lateral", "creative", "alternative", "novel", "brainstorm", "unconventional"],
    }
    LEGACY_TYPE_TO_MODE = {
        "EXPLORER": "code_discovery",
        "RESEARCHER": "domain_research",
        "LIBRARIAN": "api_lookup",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.CONSOLIDATED_RESEARCHER, config)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Research Agent. You handle code discovery, "
            "domain research, API/library lookup, and creative problem-solving."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "code_discovery": (
                "You are Vetinari's Code Explorer. Your role is to:\n"
                "- Search codebases for relevant files, classes, functions, and patterns\n"
                "- Map project structure and dependencies\n"
                "- Extract reusable patterns and identify code smells\n"
                "- Discover entry points, test files, and configuration\n\n"
                "Be thorough but focused. Return structured findings."
            ),
            "domain_research": (
                "You are Vetinari's Domain Researcher. Your role is to:\n"
                "- Conduct feasibility analysis for technical approaches\n"
                "- Research domain-specific best practices\n"
                "- Analyze competing solutions and their tradeoffs\n"
                "- Assess technical risks and opportunities\n\n"
                "Provide evidence-based recommendations with confidence scores."
            ),
            "api_lookup": (
                "You are Vetinari's API & Library Specialist. Your role is to:\n"
                "- Research APIs, libraries, and frameworks\n"
                "- Evaluate library fitness for specific use cases\n"
                "- Check compatibility, licensing, and maintenance status\n"
                "- Provide integration examples and best practices\n\n"
                "Include version info, license type, and fit score (0-1)."
            ),
            "lateral_thinking": (
                "You are Vetinari's Lateral Thinking Specialist. Your role is to:\n"
                "- Generate unconventional approaches to problems\n"
                "- Challenge assumptions and propose alternatives\n"
                "- Draw inspiration from other domains\n"
                "- Evaluate creative solutions for feasibility\n\n"
                "Think outside the box while remaining practical."
            ),
        }
        return prompts.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        if output is None:
            return VerificationResult(passed=False, issues=[{"message": "No output"}], score=0.0)
        if isinstance(output, dict):
            has_findings = bool(output.get("findings") or output.get("results") or output.get("recommendations"))
            return VerificationResult(passed=has_findings, score=0.8 if has_findings else 0.3)
        return VerificationResult(passed=True, score=0.6)

    # ------------------------------------------------------------------
    # Code Discovery (from ExplorerAgent)
    # ------------------------------------------------------------------

    def _execute_code_discovery(self, task: AgentTask) -> AgentResult:
        query = task.context.get("query", task.description)
        scope = task.context.get("scope", "code")

        prompt = (
            f"Discover and analyze code/patterns related to:\n{query}\n\n"
            f"Scope: {scope}\n\n"
            "Respond as JSON:\n"
            '{"findings": [{"file": "...", "type": "class|function|pattern", '
            '"name": "...", "description": "...", "relevance": 0.9}], '
            '"patterns": [...], "recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"findings": [], "patterns": []})
        return AgentResult(
            success=True, output=result or {"findings": []},
            metadata={"mode": "code_discovery", "scope": scope},
        )

    # ------------------------------------------------------------------
    # Domain Research (from ResearcherAgent)
    # ------------------------------------------------------------------

    def _execute_domain_research(self, task: AgentTask) -> AgentResult:
        query = task.context.get("query", task.description)
        scope = task.context.get("scope", "general")

        # Try web search if available
        search_results = self._search(query, max_results=5)

        search_context = ""
        if search_results:
            search_context = "\n\nWeb search results:\n" + "\n".join(
                f"- {r['title']}: {r['snippet']}" for r in search_results[:3]
            )

        prompt = (
            f"Research the following topic:\n{query}\n\n"
            f"Scope: {scope}\n{search_context}\n\n"
            "Respond as JSON:\n"
            '{"findings": [{"topic": "...", "summary": "...", "confidence": 0.9}], '
            '"recommendations": [...], "sources": [...], '
            '"feasibility_score": 0.8}'
        )
        result = self._infer_json(prompt, fallback={"findings": [], "recommendations": []})
        return AgentResult(
            success=True, output=result or {"findings": []},
            metadata={"mode": "domain_research", "search_results": len(search_results)},
        )

    # ------------------------------------------------------------------
    # API Lookup (from LibrarianAgent)
    # ------------------------------------------------------------------

    def _execute_api_lookup(self, task: AgentTask) -> AgentResult:
        query = task.context.get("query", task.description)

        search_results = self._search(f"{query} API documentation library", max_results=5)
        search_context = ""
        if search_results:
            search_context = "\n\nSearch results:\n" + "\n".join(
                f"- {r['title']}: {r['snippet']}" for r in search_results[:3]
            )

        prompt = (
            f"Research APIs, libraries, and frameworks for:\n{query}\n"
            f"{search_context}\n\n"
            "Respond as JSON:\n"
            '{"findings": [{"name": "...", "type": "library|api|framework", '
            '"version": "...", "license": "...", "fit_score": 0.8, '
            '"pros": [...], "cons": [...], "integration_notes": "..."}], '
            '"recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"findings": [], "recommendations": []})
        return AgentResult(
            success=True, output=result or {"findings": []},
            metadata={"mode": "api_lookup"},
        )

    # ------------------------------------------------------------------
    # Lateral Thinking
    # ------------------------------------------------------------------

    def _execute_lateral_thinking(self, task: AgentTask) -> AgentResult:
        problem = task.context.get("problem", task.description)

        prompt = (
            f"Apply lateral thinking to this problem:\n{problem}\n\n"
            "Generate at least 3 unconventional approaches. For each:\n"
            "- Describe the approach\n"
            "- Explain why it might work\n"
            "- Rate feasibility (0-1)\n\n"
            "Respond as JSON:\n"
            '{"approaches": [{"description": "...", "rationale": "...", '
            '"feasibility": 0.7, "inspiration": "..."}], '
            '"recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"approaches": [], "recommendations": []})
        return AgentResult(
            success=True, output=result or {"approaches": []},
            metadata={"mode": "lateral_thinking"},
        )

    def get_capabilities(self) -> List[str]:
        return [
            "code_discovery", "pattern_extraction", "project_mapping",
            "domain_research", "feasibility_analysis", "competitive_analysis",
            "api_lookup", "library_evaluation", "license_assessment",
            "lateral_thinking", "creative_problem_solving",
        ]


# Singleton
_consolidated_researcher_agent: Optional[ConsolidatedResearcherAgent] = None


def get_consolidated_researcher_agent(config: Optional[Dict[str, Any]] = None) -> ConsolidatedResearcherAgent:
    global _consolidated_researcher_agent
    if _consolidated_researcher_agent is None:
        _consolidated_researcher_agent = ConsolidatedResearcherAgent(config)
    return _consolidated_researcher_agent
