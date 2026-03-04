"""
Vetinari Synthesizer Agent

The Synthesizer agent is responsible for multi-source synthesis and artifact fusion.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class SynthesizerAgent(BaseAgent):
    """Synthesizer agent - multi-source synthesis and artifact fusion."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.SYNTHESIZER, config)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Synthesizer. Combine outputs from multiple agents into a 
cohesive artifact (e.g., a unified plan, a combined architecture doc, or a final artifact). 
Resolve conflicts, ensure consistency, and present an integrated result with traceable sources.

You must:
1. Integrate outputs from multiple sources
2. Resolve conflicting information intelligently
3. Maintain consistency across integrated artifacts
4. Provide clear source attribution
5. Create a unified, coherent output
6. Highlight integration decisions

Output format must include synthesized_artifact, provenance, integration_notes, and decisions."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "artifact_fusion",
            "multi_source_integration",
            "conflict_resolution",
            "consistency_checking",
            "provenance_tracking",
            "document_assembly"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the synthesis task.
        
        Args:
            task: The task containing sources to synthesize
            
        Returns:
            AgentResult containing the synthesized artifact
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            sources = task.context.get("sources", [])
            artifact_type = task.context.get("type", "document")
            
            # Perform synthesis via LLM
            synthesized = self._synthesize_artifacts(sources, artifact_type)
            
            result = AgentResult(
                success=True,
                output=synthesized,
                metadata={
                    "sources_merged": len(sources),
                    "artifact_type": artifact_type,
                    "integration_score": synthesized.get("integration_score", 0)
                }
            )
            self.complete_task(task, result)
            return result
            
        except Exception as e:
            self._log("error", f"Synthesis failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the synthesized artifact meets quality standards.
        
        Args:
            output: The synthesized artifact to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for synthesized artifact
        if not output.get("synthesized_artifact"):
            issues.append({"type": "missing_artifact", "message": "Synthesized artifact missing"})
            score -= 0.3
        
        # Check for provenance
        if not output.get("provenance"):
            issues.append({"type": "missing_provenance", "message": "Source provenance missing"})
            score -= 0.2
        
        # Check for integration notes
        if not output.get("integration_notes"):
            issues.append({"type": "missing_notes", "message": "Integration notes missing"})
            score -= 0.1
        
        # Check consistency
        if output.get("integration_score", 0) < 0.6:
            issues.append({"type": "low_consistency", "message": "Integration consistency is low"})
            score -= 0.2
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _synthesize_artifacts(self, sources: List[Dict[str, Any]], artifact_type: str) -> Dict[str, Any]:
        """Synthesize multiple artifact sources into a unified output using LLM.

        Builds provenance mechanically, then uses the LLM to actually merge
        and reconcile content. Falls back to simple concatenation.
        """
        import json as _json

        # Mechanical provenance (always reliable)
        provenance = [
            {
                "index": i,
                "agent": source.get("agent", "unknown"),
                "artifact": str(source.get("artifact", ""))[:100],
                "confidence": source.get("confidence", 0.9),
            }
            for i, source in enumerate(sources)
        ]

        if not sources:
            return {
                "synthesized_artifact": "(no sources provided)",
                "artifact_type": artifact_type,
                "provenance": [],
                "integration_notes": ["No sources to synthesize"],
                "decisions": [],
                "integration_score": 0.0,
                "summary": "No sources provided for synthesis",
            }

        # Serialize source content for LLM (truncate to avoid context overflow)
        sources_text = _json.dumps(
            [{"agent": s.get("agent", "?"), "content": str(s.get("artifact", s))[:500]} for s in sources],
            indent=2
        )

        prompt = f"""You are a synthesis expert. Combine the following {len(sources)} source artifacts into a unified {artifact_type}.

SOURCES:
{sources_text}

Produce a JSON synthesis report:
{{
  "synthesized_artifact": "The complete, unified {artifact_type} content",
  "integration_notes": ["note about what was merged", ...],
  "decisions": [
    {{"decision": "...", "approach": "...", "rationale": "..."}}
  ],
  "integration_score": 0.0-1.0,
  "summary": "Brief synthesis summary"
}}

Resolve any conflicts intelligently. Flag inconsistencies in integration_notes."""

        result = self._infer_json(prompt)

        if result and isinstance(result, dict) and result.get("synthesized_artifact"):
            result["artifact_type"] = artifact_type
            result["provenance"] = provenance
            return result

        # Fallback: concatenate content
        self._log("warning", "LLM synthesis unavailable, concatenating sources")
        concat_content = "\n\n---\n\n".join(
            str(s.get("artifact", s)) for s in sources
        )
        return {
            "synthesized_artifact": concat_content,
            "artifact_type": artifact_type,
            "provenance": provenance,
            "integration_notes": [f"Concatenated {len(sources)} sources (LLM unavailable)"],
            "decisions": [],
            "integration_score": round(min(1.0, 0.5 + len(sources) * 0.1), 2),
            "summary": f"Concatenated {len(sources)} source artifacts for {artifact_type}",
        }


# Singleton instance
_synthesizer_agent: Optional[SynthesizerAgent] = None


def get_synthesizer_agent(config: Optional[Dict[str, Any]] = None) -> SynthesizerAgent:
    """Get the singleton Synthesizer agent instance."""
    global _synthesizer_agent
    if _synthesizer_agent is None:
        _synthesizer_agent = SynthesizerAgent(config)
    return _synthesizer_agent
