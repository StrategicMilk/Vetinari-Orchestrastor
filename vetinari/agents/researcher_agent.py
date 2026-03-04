"""
Vetinari Researcher Agent

The Researcher agent is responsible for domain research, feasibility analysis,
and competitive analysis.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class ResearcherAgent(BaseAgent):
    """Researcher agent - domain research, feasibility analysis, competitive analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.RESEARCHER, config)
        self._max_competitors = self._config.get("max_competitors", 5)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Researcher. Perform domain analysis, feasibility assessments, 
and competitor comparisons. Deliver structured findings and recommendations that inform plan tasks.

You must:
1. Analyze the domain comprehensively
2. Assess technical and business feasibility
3. Identify and compare competitors
4. Provide evidence-based recommendations
5. Highlight potential risks and opportunities
6. Organize findings by research area

Output format must include findings by area, feasibility score, competitor analysis, and recommendations."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "domain_analysis",
            "feasibility_assessment",
            "competitor_analysis",
            "market_research",
            "technology_scouting",
            "risk_identification"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the research task.
        
        Args:
            task: The task containing the research topic
            
        Returns:
            AgentResult containing the research findings
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            domain_topic = task.context.get("domain_topic", task.description)
            questions = task.context.get("questions", [])
            
            # Perform domain research (simulated - in production would use actual research)
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
            
        except Exception as e:
            self._log("error", f"Research failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
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
            f"{domain_topic} best practices 2025",
            f"{domain_topic} technical challenges",
        ])

        all_results: List[Dict] = []
        citations: List[str] = []

        for query in search_queries[:5]:  # Limit to 5 queries
            results = self._search(query, max_results=3)
            for r in results:
                if r["url"] not in [x.get("url") for x in all_results]:
                    all_results.append(r)
                    citations.append(f"[{len(citations)+1}] {r['title']}. {r['url']}")

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
            "feasibility_score": 0.7 if all_results else 0.5,
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
