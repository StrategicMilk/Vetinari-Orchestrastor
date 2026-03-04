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
        """Perform domain research and analysis.
        
        Args:
            domain_topic: The domain to research
            questions: Specific questions to answer
            
        Returns:
            Dictionary containing research findings
        """
        # This is a simplified implementation
        # In production, this would use actual research data sources
        
        findings = []
        
        # Create research areas
        research_areas = ["market", "technology", "feasibility", "risks"]
        
        for area in research_areas:
            findings.append({
                "area": area,
                "summary": f"Analysis of {area} for {domain_topic}",
                "findings": [
                    f"Key insight 1 about {area}",
                    f"Key insight 2 about {area}"
                ],
                "evidence": [
                    {"source": "industry_report", "link": f"https://reports.example.com/{area}"},
                    {"source": "technical_paper", "link": f"https://papers.example.com/{area}"}
                ]
            })
        
        # Competitor analysis
        competitors = []
        for i in range(min(3, self._max_competitors)):
            competitors.append({
                "name": f"Competitor {i+1}",
                "strengths": ["Feature A", "Feature B"],
                "weaknesses": ["Limitation A", "Limitation B"],
                "market_position": "Growing" if i == 0 else "Established"
            })
        
        return {
            "domain": domain_topic,
            "findings": findings,
            "feasibility_score": 0.75,
            "feasibility_assessment": f"{domain_topic} is technically and commercially feasible",
            "competitor_analysis": competitors,
            "recommendations": [
                "Recommend approach A for implementation",
                "Consider timing for market entry",
                "Invest in differentiation strategy"
            ],
            "risks": [
                {"risk": "Market saturation", "likelihood": 0.6, "impact": 0.7},
                {"risk": "Technical complexity", "likelihood": 0.4, "impact": 0.5}
            ]
        }


# Singleton instance
_researcher_agent: Optional[ResearcherAgent] = None


def get_researcher_agent(config: Optional[Dict[str, Any]] = None) -> ResearcherAgent:
    """Get the singleton Researcher agent instance."""
    global _researcher_agent
    if _researcher_agent is None:
        _researcher_agent = ResearcherAgent(config)
    return _researcher_agent
