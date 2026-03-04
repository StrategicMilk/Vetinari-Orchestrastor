"""
Vetinari Oracle Agent

The Oracle agent is responsible for architectural decisions, risk assessment,
and high-level debugging strategies.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class OracleAgent(BaseAgent):
    """Oracle agent - architectural decisions, risk assessment, debugging."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.ORACLE, config)
        self._max_risks = self._config.get("max_risks", 5)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Oracle. Your role is to provide architectural guidance, 
assess risks, and propose robust designs for complex tasks.

You must:
1. Review architectural options and propose the best approach
2. Identify potential risks with likelihood and impact
3. Provide mitigation strategies for each risk
4. Outline trade-offs clearly
5. Give recommended guidelines for implementation

Output format must include architecture_vision, risks array, and recommended_guidelines."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "architecture_design",
            "risk_assessment",
            "tradeoff_analysis",
            "debugging_strategy",
            "technical_guidance"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the oracle task.
        
        Args:
            task: The task containing the architectural question
            
        Returns:
            AgentResult containing the architectural guidance
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            goal = task.prompt or task.description
            context = task.context or {}
            
            # Generate architectural guidance
            guidance = self._provide_guidance(goal, context)
            
            result = AgentResult(
                success=True,
                output=guidance,
                metadata={
                    "risks_identified": len(guidance.get("risks", [])),
                    "guidelines_count": len(guidance.get("recommended_guidelines", []))
                }
            )
            self.complete_task(task, result)
            return result
            
        except Exception as e:
            self._log("error", f"Oracle guidance failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the architectural guidance meets quality standards.
        
        Args:
            output: The guidance to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for required fields
        if "architecture_vision" not in output:
            issues.append({"type": "missing_vision", "message": "No architecture vision provided"})
            score -= 0.3
        
        risks = output.get("risks", [])
        if not risks:
            issues.append({"type": "no_risks", "message": "No risks identified"})
            score -= 0.2
        
        guidelines = output.get("recommended_guidelines", [])
        if not guidelines:
            issues.append({"type": "no_guidelines", "message": "No guidelines provided"})
            score -= 0.2
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _provide_guidance(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide architectural guidance using LLM reasoning.

        Replaces all keyword-based heuristics with genuine LLM analysis.
        Falls back to a generic guidance set if the LLM is unavailable.
        """
        import json as _json

        context_str = ""
        if context:
            context_str = f"\nContext: {_json.dumps(context, default=str)[:500]}"

        # Optionally search for architectural best practices
        search_results = self._search(f"{goal} architecture best practices", max_results=3)
        search_context = ""
        if search_results:
            search_context = "\nRelevant references:\n" + "\n".join(
                f"- {r['title']}: {r['snippet'][:100]}" for r in search_results
            )

        prompt = f"""You are an expert software architect. Provide comprehensive architectural guidance.

GOAL: {goal}{context_str}{search_context}

Produce a JSON architectural assessment:
{{
  "architecture_vision": "Detailed description of the recommended architecture",
  "architecture_type": "api_service|web_application|data_platform|microservice|cli_tool|ml_pipeline|general",
  "risks": [
    {{"risk": "...", "likelihood": 0.0-1.0, "impact": 0.0-1.0, "mitigation": "..."}}
  ],
  "recommended_guidelines": ["guideline 1", "guideline 2", ...],
  "trade_offs": "Description of key trade-offs in the chosen approach",
  "constraints": {{}}
}}

Provide {self._max_risks} specific, actionable risks with realistic probabilities.
Guidelines should be concrete and goal-specific."""

        result = self._infer_json(prompt)

        if result and isinstance(result, dict) and result.get("architecture_vision"):
            result["constraints"] = context.get("constraints", result.get("constraints", {}))
            return result

        # Fallback: minimal but generic guidance
        self._log("warning", "LLM oracle guidance unavailable, returning generic guidance")
        return {
            "architecture_vision": f"A modular, maintainable architecture for: {goal}",
            "architecture_type": "general",
            "risks": [
                {"risk": "Scope creep", "likelihood": 0.6, "impact": 0.7, "mitigation": "Define clear MVP requirements"},
                {"risk": "Technical debt", "likelihood": 0.5, "impact": 0.6, "mitigation": "Schedule regular refactoring"},
            ],
            "recommended_guidelines": [
                "Use version control from day one",
                "Write automated tests for all new code",
                "Use CI/CD pipelines",
                "Document architectural decisions",
            ],
            "trade_offs": "Evaluated without LLM — manual review recommended",
            "constraints": context.get("constraints", {}),
        }


# Singleton instance
_oracle_agent: Optional[OracleAgent] = None


def get_oracle_agent(config: Optional[Dict[str, Any]] = None) -> OracleAgent:
    """Get the singleton Oracle agent instance."""
    global _oracle_agent
    if _oracle_agent is None:
        _oracle_agent = OracleAgent(config)
    return _oracle_agent
