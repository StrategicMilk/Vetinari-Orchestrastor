"""Vetinari Researcher Agent — consolidated from Explorer + Librarian + Synthesizer.

The Researcher agent is responsible for domain research, feasibility analysis,
competitive analysis, code exploration, library lookup, and multi-source synthesis.
"""

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Researcher agent — domain research, code exploration, library lookup, and synthesis.

    Absorbs:
        - ExplorerAgent: code/document/class discovery, pattern extraction, symbol lookup
        - LibrarianAgent: literature/library research, API/docs lookup, best practices
        - SynthesizerAgent: multi-source synthesis, artifact fusion, report generation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.RESEARCHER, config)
        self._max_competitors = self._config.get("max_competitors", 5)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Researcher. You combine domain research, codebase exploration,
library/API documentation lookup, and multi-source synthesis into a single unified capability.

Your responsibilities:
1. Analyze domains comprehensively with feasibility assessments
2. Explore codebases — locate files, symbols, patterns, and dependencies
3. Look up library documentation, API references, and best practices
4. Synthesize findings from multiple sources into coherent reports
5. Identify and compare competitors and alternatives
6. Provide evidence-based recommendations with confidence levels

Output format must include findings by area, feasibility score, competitor analysis, and recommendations."""

    def get_capabilities(self) -> List[str]:
        return [
            "domain_analysis",
            "feasibility_assessment",
            "competitor_analysis",
            "market_research",
            "technology_scouting",
            "risk_identification",
            # From ExplorerAgent
            "code_navigation",
            "symbol_lookup",
            "pattern_matching",
            "project_mapping",
            # From LibrarianAgent
            "api_reference",
            "library_research",
            "best_practices",
            # From SynthesizerAgent
            "multi_source_synthesis",
            "report_generation",
            "conflict_resolution",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute task, delegating to explorer, librarian, or synthesizer based on keywords."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("explore", "codebase", "symbol", "navigate", "find file", "class discovery", "pattern", "grep", "search code", "project map")):
                result = self._delegate_to_explorer(task)
            elif any(kw in desc for kw in ("library", "api doc", "reference", "best practice", "documentation lookup", "package", "import")):
                result = self._delegate_to_librarian(task)
            elif any(kw in desc for kw in ("synthesize", "combine", "fuse", "merge results", "consolidate", "summarize multiple", "report from")):
                result = self._delegate_to_synthesizer(task)
            else:
                result = self._execute_research(task)

            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Research failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )

    def _delegate_to_explorer(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.explorer_agent import ExplorerAgent
        agent = ExplorerAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_librarian(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.librarian_agent import LibrarianAgent
        agent = LibrarianAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_synthesizer(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.synthesizer_agent import SynthesizerAgent
        agent = SynthesizerAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _execute_research(self, task: AgentTask) -> AgentResult:
        """Execute domain research (original ResearcherAgent logic)."""
        domain_topic = task.context.get("domain_topic", task.description)
        questions = task.context.get("questions", [])

        findings = self._perform_research(domain_topic, questions)

        return AgentResult(
            success=True,
            output=findings,
            metadata={
                "domain": domain_topic,
                "questions_answered": len(questions),
                "areas_covered": len(findings.get("findings", []))
            }
        )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the research findings meet quality standards.
        
        Args:
            output: The findings to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for findings
        findings = output.get("findings", [])
        if not findings:
            issues.append({"type": "no_findings", "message": "No research findings"})
            score -= 0.3
        
        # Check for competitor analysis
        if not output.get("competitor_analysis"):
            issues.append({"type": "no_competitors", "message": "Competitor analysis missing"})
            score -= 0.2
        
        # Check for feasibility assessment
        if "feasibility_score" not in output:
            issues.append({"type": "no_feasibility", "message": "Feasibility score missing"})
            score -= 0.2
        
        # Check for recommendations
        if not output.get("recommendations"):
            issues.append({"type": "no_recommendations", "message": "Recommendations missing"})
            score -= 0.1
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _perform_research(self, domain_topic: str, questions: List[str]) -> Dict[str, Any]:
        """Perform domain research using real web search + LLM synthesis.

        Pipeline:
        1. Multi-query web search across domain aspects
        2. LLM synthesis of real search results
        3. Graceful fallback to structured template if LLM unavailable
        """
        import json

        # ------------------------------------------------------------------
        # Step 1: Web search across multiple aspects
        # ------------------------------------------------------------------
        search_queries = [domain_topic] + (questions[:3] if questions else [])
        search_queries.extend([
            f"{domain_topic} alternatives comparison",
            f"{domain_topic} best practices {__import__('datetime').datetime.now().year}",
            f"{domain_topic} technical challenges",
        ])

        all_results: List[Dict] = []
        seen_urls: set = set()
        citations: List[str] = []

        for query in search_queries[:5]:  # Limit to 5 queries
            results = self._search(query, max_results=3)
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
                    citations.append(f"[{len(citations)+1}] {r['title']}. {url}")

        # ------------------------------------------------------------------
        # Step 2: LLM synthesis of search results
        # ------------------------------------------------------------------
        synthesis_prompt = f"""You are a research analyst. Analyze these search results about '{domain_topic}':

SEARCH RESULTS:
{json.dumps(all_results[:10], indent=2)}

QUESTIONS TO ANSWER: {questions}

Produce a comprehensive research report as JSON with this exact structure:
{{
  "findings": [
    {{"area": "market", "summary": "...", "findings": ["insight1", "insight2"], "evidence": [{{"source": "...", "link": "..."}}]}},
    {{"area": "technology", "summary": "...", "findings": [...], "evidence": [...]}},
    {{"area": "feasibility", "summary": "...", "findings": [...], "evidence": [...]}},
    {{"area": "risks", "summary": "...", "findings": [...], "evidence": [...]}}
  ],
  "feasibility_score": 0.75,
  "feasibility_assessment": "...",
  "competitor_analysis": [{{"name": "...", "strengths": [...], "weaknesses": [...], "market_position": "..."}}],
  "recommendations": ["recommendation 1", ...],
  "risks": [{{"risk": "...", "likelihood": 0.5, "impact": 0.5}}]
}}

Base all findings strictly on the search results. Do not hallucinate. If information is not in the search results, indicate uncertainty."""

        llm_result = self._infer_json(synthesis_prompt)

        if llm_result and isinstance(llm_result, dict) and llm_result.get("findings"):
            llm_result["domain"] = domain_topic
            llm_result["provenance"] = citations
            llm_result["sources_searched"] = len(all_results)
            return llm_result

        # ------------------------------------------------------------------
        # Step 3: Fallback – structured output from search results alone
        # ------------------------------------------------------------------
        self._log("info", f"LLM synthesis unavailable, returning raw search results for '{domain_topic}'")
        
        findings = []
        for i, area in enumerate(["market", "technology", "feasibility", "risks"]):
            area_results = all_results[i*2:(i+1)*2] if all_results else []
            findings.append({
                "area": area,
                "summary": f"{area.title()} analysis for {domain_topic}",
                "findings": [r["snippet"][:150] for r in area_results] or [f"See {domain_topic} documentation"],
                "evidence": [{"source": r["title"], "link": r["url"]} for r in area_results]
            })

        competitors = []
        for r in all_results[:self._max_competitors]:
            if r.get("title"):
                competitors.append({
                    "name": r["title"][:60],
                    "strengths": [r["snippet"][:100]],
                    "weaknesses": [],
                    "market_position": "Identified"
                })

        return {
            "domain": domain_topic,
            "findings": findings,
            "feasibility_score": min(0.9, 0.3 + 0.1 * len(all_results)),
            "feasibility_assessment": f"Based on {len(all_results)} sources found for {domain_topic}",
            "competitor_analysis": competitors,
            "recommendations": [
                f"Review: {r['url']}" for r in all_results[:3]
            ] or ["Further research required"],
            "risks": [
                {"risk": "Information may be incomplete", "likelihood": 0.5 if all_results else 0.9, "impact": 0.4}
            ],
            "provenance": citations,
            "sources_searched": len(all_results),
        }


# Singleton instance
_researcher_agent: Optional[ResearcherAgent] = None


def get_researcher_agent(config: Optional[Dict[str, Any]] = None) -> ResearcherAgent:
    """Get the singleton Researcher agent instance."""
    global _researcher_agent
    if _researcher_agent is None:
        _researcher_agent = ResearcherAgent(config)
    return _researcher_agent
