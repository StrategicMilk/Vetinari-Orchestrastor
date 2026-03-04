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
            codebase_path = task.context.get("codebase_path", "")

            # Local codebase exploration via RepoMap
            repo_context = ""
            if codebase_path and scope in ("code", "patterns", "apis"):
                try:
                    from vetinari.repo_map import get_repo_map
                    mapper = get_repo_map()
                    repo_context = mapper.generate_for_task(
                        root_path=codebase_path,
                        task_description=goal_context,
                        max_tokens=1000,
                    )
                except Exception as _rm_err:
                    self._log("debug", f"RepoMap failed: {_rm_err}")

            findings = self._explore(goal_context, scope, repo_context=repo_context)

            result = AgentResult(
                success=True,
                output=findings,
                metadata={
                    "scope": scope,
                    "findings_count": len(findings.get("findings", [])),
                    "local_repo_mapped": bool(repo_context),
                },
            )
            self.complete_task(task, result)
            return result
            
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
    
    def _explore(self, goal_context: str, scope: str, repo_context: str = "") -> Dict[str, Any]:
        """Perform exploration using real web search + LLM synthesis.

        Searches the web for relevant patterns, references, and implementations
        for the goal context, then uses the LLM to rank and summarize findings.
        """
        import json as _json

        # If we have local repo context, include it in the synthesis prompt
        local_context_str = (
            f"\nLOCAL CODEBASE STRUCTURE:\n{repo_context}\n" if repo_context else ""
        )

        # Real web search for the goal context
        search_query = f"{goal_context} {scope} examples patterns"
        search_results = self._search(search_query, max_results=self._max_results)

        # Secondary search for code examples if scope is code-related
        if scope in ("code", "apis", "patterns"):
            code_results = self._search(f"{goal_context} code implementation", max_results=3)
            # Merge, deduplicating by URL
            seen = {r["url"] for r in search_results}
            for r in code_results:
                if r["url"] not in seen:
                    search_results.append(r)
                    seen.add(r["url"])

        if not search_results:
            # No search results -- use LLM knowledge directly
            prompt = (
                f"Explore and discover relevant {scope} patterns for: {goal_context}\n"
                f"{local_context_str}\n"
                f"Return JSON with: findings[], references[], search_terms[]"
            )
            result = self._infer_json(prompt)
            if result:
                result["scope"] = scope
                return result

        # Use LLM to synthesize real search results into structured findings
        search_text = _json.dumps(search_results[:8], indent=2)

        prompt = f"""You are an exploration expert. Analyze these search results about '{goal_context}' (scope: {scope}).{local_context_str}

SEARCH RESULTS:
{search_text}

Synthesize the findings into a structured exploration report as JSON:
{{
  "findings": [
    {{"id": "f1", "type": "{scope}", "summary": "...", "relevance_score": 0.9, "description": "..."}}
  ],
  "references": [
    {{"title": "...", "url": "actual URL from results", "type": "{scope}"}}
  ],
  "search_terms": ["extracted", "key", "terms"],
  "scope": "{scope}"
}}

Use only URLs from the actual search results. Rank by relevance to the goal."""

        result = self._infer_json(prompt)

        if result and isinstance(result, dict) and result.get("findings"):
            result["scope"] = scope
            return result

        # Fallback: convert raw search results to findings format
        findings = []
        references = []
        for i, r in enumerate(search_results[:self._max_results]):
            findings.append({
                "id": f"f{i+1}",
                "type": scope,
                "summary": r["title"],
                "relevance_score": r.get("source_reliability", 0.5),
                "description": r["snippet"][:200],
            })
            references.append({
                "title": r["title"],
                "url": r["url"],
                "type": scope,
            })

        return {
            "findings": findings,
            "references": references,
            "scope": scope,
            "search_terms": goal_context.split()[:5],
        }


# Singleton instance
_explorer_agent: Optional[ExplorerAgent] = None


def get_explorer_agent(config: Optional[Dict[str, Any]] = None) -> ExplorerAgent:
    """Get the singleton Explorer agent instance."""
    global _explorer_agent
    if _explorer_agent is None:
        _explorer_agent = ExplorerAgent(config)
    return _explorer_agent
