"""
Vetinari Librarian Agent

The Librarian agent is responsible for literature/library/API/docs lookup and 
provides authoritative references and design patterns.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class LibrarianAgent(BaseAgent):
    """Librarian agent - literature/library/API/docs lookup and research."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.LIBRARIAN, config)
        self._sources = self._config.get("sources", ["docs", "apis", "libraries", "patterns"])
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Librarian. Search for authoritative references, API docs, 
libraries, and patterns relevant to the current plan. Provide summarized findings with direct 
citations and potential fit assessments for the tasks.

You must:
1. Search authoritative sources (documentation, APIs, libraries, patterns)
2. Provide direct citations with URLs
3. Include code snippets where relevant
4. Give a fit assessment for the current tasks
5. Highlight licenses and compatibility concerns
6. Organize findings by source type

Output format must include summary, sources, fit_assessment, and recommendations."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "api_documentation_lookup",
            "library_discovery",
            "pattern_research",
            "license_analysis",
            "fit_assessment",
            "citation_generation"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the library research task.
        
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
            topic = task.context.get("topic", task.description)
            sources = task.context.get("sources", self._sources)
            
            # Perform library research (simulated - in production would use actual lookups)
            findings = self._research_libraries(topic, sources)
            
            return AgentResult(
                success=True,
                output=findings,
                metadata={
                    "topic": topic,
                    "sources_searched": len(sources),
                    "results_count": len(findings.get("sources", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"Library research failed: {str(e)}")
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
        
        # Check for summary
        if not output.get("summary"):
            issues.append({"type": "missing_summary", "message": "Summary is missing"})
            score -= 0.2
        
        # Check for sources
        sources = output.get("sources", [])
        if not sources:
            issues.append({"type": "no_sources", "message": "No sources found"})
            score -= 0.3
        
        # Check for fit assessment
        if not output.get("fit_assessment"):
            issues.append({"type": "missing_assessment", "message": "Fit assessment is missing"})
            score -= 0.2
        
        # Check for recommendations
        if not output.get("recommendations"):
            issues.append({"type": "missing_recommendations", "message": "Recommendations missing"})
            score -= 0.1
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _research_libraries(self, topic: str, sources: List[str]) -> Dict[str, Any]:
        """Perform library and API research.
        
        Args:
            topic: The topic to research
            sources: The sources to search
            
        Returns:
            Dictionary containing research findings
        """
        # This is a simplified implementation
        # In production, this would use actual API/library lookups
        
        found_sources = []
        
        # Simulate finding relevant libraries and APIs
        relevant_keywords = ["api", "sdk", "library", "framework", "package", "module"]
        keywords_found = [kw for kw in relevant_keywords if kw in topic.lower()]
        
        if "api" in topic.lower() or "docs" in sources:
            found_sources.append({
                "title": f"Official API Documentation for {topic}",
                "url": f"https://docs.example.com/{topic.replace(' ', '_')}",
                "type": "api_docs",
                "summary": f"Comprehensive API reference for {topic}",
                "license": "CC-BY-4.0"
            })
        
        if "library" in topic.lower() or "libraries" in sources:
            found_sources.append({
                "title": f"Popular {topic} Libraries",
                "url": f"https://pypi.org/search/?q={topic.replace(' ', '+')}",
                "type": "libraries",
                "summary": f"Package registry for {topic} implementations",
                "license": "Varies"
            })
        
        if "pattern" in topic.lower() or "patterns" in sources:
            found_sources.append({
                "title": f"Design Patterns for {topic}",
                "url": f"https://patterns.example.com/{topic.replace(' ', '-')}",
                "type": "patterns",
                "summary": f"Best practices and design patterns for {topic}",
                "license": "CC0"
            })
        
        return {
            "summary": f"Research findings for topic: {topic}",
            "sources": found_sources,
            "fit_assessment": f"The identified libraries and APIs provide good coverage for {topic} implementation",
            "recommendations": [
                "Consider starting with the official API documentation",
                "Evaluate popular libraries for feature completeness",
                "Review design patterns for architectural guidance"
            ],
            "keywords": keywords_found or ["general"]
        }


# Singleton instance
_librarian_agent: Optional[LibrarianAgent] = None


def get_librarian_agent(config: Optional[Dict[str, Any]] = None) -> LibrarianAgent:
    """Get the singleton Librarian agent instance."""
    global _librarian_agent
    if _librarian_agent is None:
        _librarian_agent = LibrarianAgent(config)
    return _librarian_agent
