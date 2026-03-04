"""
Vetinari Explorer Agent

The Explorer agent is responsible for fast code/document/class discovery
and pattern extraction.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class ExplorerAgent(BaseAgent):
    """Explorer agent - fast code/document discovery and pattern extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.EXPLORER, config)
        self._max_results = self._config.get("max_results", 10)
        self._search_paths = self._config.get("search_paths", [])
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Explorer. Your mission is to rapidly discover code patterns, 
reference implementations, relevant docs, APIs, and examples that could satisfy 
the current task.

You must:
1. Search comprehensively for relevant code patterns
2. Provide direct references with links
3. Include code snippets where appropriate
4. Give a short rationale for each finding
5. NOT alter code - only discover and report

Scope options: code, docs, apis, patterns

Output format must be a findings object with references."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "code_search",
            "pattern_discovery",
            "api_lookup",
            "reference_extraction",
            "documentation_search"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the exploration task.
        
        Args:
            task: The task containing the exploration goal
            
        Returns:
            AgentResult containing the findings
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            goal_context = task.prompt or task.description
            scope = task.context.get("scope", "code")
            
            # Perform exploration (simulated - in production would use actual search)
            findings = self._explore(goal_context, scope)
            
            return AgentResult(
                success=True,
                output=findings,
                metadata={
                    "scope": scope,
                    "findings_count": len(findings.get("findings", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"Exploration failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the exploration findings meet quality standards.
        
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
            issues.append({"type": "no_findings", "message": "No findings returned"})
            score -= 0.3
        
        # Check for references
        references = output.get("references", [])
        if not references and findings:
            issues.append({"type": "no_references", "message": "No references provided"})
            score -= 0.2
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _explore(self, goal_context: str, scope: str) -> Dict[str, Any]:
        """Perform exploration based on the goal.
        
        Args:
            goal_context: The context to search for
            scope: The scope of search (code, docs, apis, patterns)
            
        Returns:
            Dictionary containing findings
        """
        # This is a simplified implementation
        # In production, this would use actual search tools
        
        findings = []
        references = []
        
        # Analyze the goal context to determine search terms
        search_terms = self._extract_search_terms(goal_context)
        
        # Generate findings based on search terms
        for i, term in enumerate(search_terms[:self._max_results]):
            finding = {
                "id": f"f{i+1}",
                "type": scope,
                "source": "explorer",
                "summary": f"Pattern for {term}",
                "relevance_score": 0.9 - (i * 0.1),
                "description": f"Found relevant patterns for '{term}' in {scope}"
            }
            findings.append(finding)
            references.append({
                "title": f"Reference for {term}",
                "url": f"https://example.com/{term}",
                "type": scope
            })
        
        return {
            "findings": findings,
            "references": references,
            "scope": scope,
            "search_terms": search_terms
        }
    
    def _extract_search_terms(self, goal_context: str) -> List[str]:
        """Extract search terms from the goal context.
        
        Args:
            goal_context: The context to extract from
            
        Returns:
            List of search terms
        """
        # Simple extraction - split on common delimiters
        terms = []
        
        # Extract key phrases
        important_words = [
            "api", "database", "frontend", "backend", "authentication",
            "testing", "deployment", "configuration", "models", "views",
            "controller", "service", "repository", "middleware"
        ]
        
        context_lower = goal_context.lower()
        for word in important_words:
            if word in context_lower:
                terms.append(word)
        
        # If no terms found, use the whole context
        if not terms:
            terms = goal_context.split()[:5]
        
        return terms


# Singleton instance
_explorer_agent: Optional[ExplorerAgent] = None


def get_explorer_agent(config: Optional[Dict[str, Any]] = None) -> ExplorerAgent:
    """Get the singleton Explorer agent instance."""
    global _explorer_agent
    if _explorer_agent is None:
        _explorer_agent = ExplorerAgent(config)
    return _explorer_agent
