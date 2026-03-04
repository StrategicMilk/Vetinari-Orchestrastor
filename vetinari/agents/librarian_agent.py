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
        """Perform library and API research using real web search + LLM.

        Searches authoritative sources for libraries, APIs, and patterns,
        then uses the LLM to evaluate fit and produce recommendations.
        """
        import json as _json

        # Multi-query web search targeting authoritative sources
        queries = [
            f"{topic} library documentation site:pypi.org OR site:npmjs.com OR site:github.com",
            f"{topic} API reference official documentation",
            f"best {topic} libraries 2025",
        ]

        all_results = []
        seen_urls: set = set()
        for query in queries[:2]:
            for r in self._search(query, max_results=4):
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)

        search_text = _json.dumps(all_results[:8], indent=2)

        prompt = f"""You are a library research expert. Find and evaluate resources for: "{topic}"
Sources requested: {sources}

SEARCH RESULTS:
{search_text}

Produce a JSON research report:
{{
  "summary": "Brief summary of findings",
  "sources": [
    {{"title": "...", "url": "real URL from search results", "type": "api_docs|library|pattern|tutorial", "summary": "...", "license": "..."}}
  ],
  "fit_assessment": "How well do found resources cover {topic}?",
  "recommendations": ["specific recommendation 1", ...],
  "keywords": ["key", "terms", "found"]
}}

Only use real URLs from the search results. Assess licensing where visible."""

        result = self._infer_json(prompt)

        if result and isinstance(result, dict) and result.get("sources"):
            return result

        # Fallback: return formatted raw search results
        self._log("warning", "LLM library research unavailable, returning raw search results")
        found_sources = [
            {
                "title": r["title"],
                "url": r["url"],
                "type": "web",
                "summary": r["snippet"][:150],
                "license": "Unknown",
            }
            for r in all_results[:5]
        ]

        return {
            "summary": f"Web search results for: {topic}",
            "sources": found_sources,
            "fit_assessment": f"Found {len(found_sources)} resources for {topic} (LLM evaluation unavailable)",
            "recommendations": ["Review the sources above for relevance", "Check official documentation"],
            "keywords": topic.lower().split()[:5],
        }


# Singleton instance
_librarian_agent: Optional[LibrarianAgent] = None


def get_librarian_agent(config: Optional[Dict[str, Any]] = None) -> LibrarianAgent:
    """Get the singleton Librarian agent instance."""
    global _librarian_agent
    if _librarian_agent is None:
        _librarian_agent = LibrarianAgent(config)
    return _librarian_agent
