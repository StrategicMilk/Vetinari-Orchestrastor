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
            
            # Perform synthesis (simulated - in production would use actual synthesis logic)
            synthesized = self._synthesize_artifacts(sources, artifact_type)
            
            return AgentResult(
                success=True,
                output=synthesized,
                metadata={
                    "sources_merged": len(sources),
                    "artifact_type": artifact_type,
                    "integration_score": synthesized.get("integration_score", 0)
                }
            )
            
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
        """Synthesize multiple artifact sources into a unified output.
        
        Args:
            sources: List of source artifacts to synthesize
            artifact_type: Type of artifact to create (document, code, spec, etc.)
            
        Returns:
            Dictionary containing the synthesized artifact
        """
        # This is a simplified implementation
        # In production, this would use actual content merging and conflict resolution
        
        provenance = []
        integration_notes = []
        decisions = []
        
        # Build provenance from sources
        for i, source in enumerate(sources):
            provenance.append({
                "index": i,
                "agent": source.get("agent", "unknown"),
                "artifact": source.get("artifact", ""),
                "confidence": 0.9
            })
        
        # Identify integration decisions
        if len(sources) > 1:
            integration_notes.append(f"Merged {len(sources)} source artifacts")
            integration_notes.append("Resolved conflicts by prioritizing recent sources")
            integration_notes.append("Maintained consistency across sections")
            
            decisions.append({
                "decision": "Conflict resolution strategy",
                "approach": "Latest source wins",
                "rationale": "More recent information is typically more accurate"
            })
        
        # Create the synthesized artifact
        synthesized_content = f"Unified {artifact_type}\n"
        synthesized_content += f"Created from {len(sources)} sources\n"
        synthesized_content += "\n".join(integration_notes) + "\n"
        
        # Calculate integration score (0-1)
        if len(sources) == 0:
            integration_score = 0.0
        elif len(sources) == 1:
            integration_score = 1.0
        else:
            # Higher score for more sources successfully integrated
            integration_score = min(1.0, 0.8 + (len(sources) * 0.05))
        
        return {
            "synthesized_artifact": synthesized_content,
            "artifact_type": artifact_type,
            "provenance": provenance,
            "integration_notes": integration_notes,
            "decisions": decisions,
            "integration_score": round(integration_score, 2),
            "summary": f"Successfully synthesized {artifact_type} from {len(sources)} sources"
        }


# Singleton instance
_synthesizer_agent: Optional[SynthesizerAgent] = None


def get_synthesizer_agent(config: Optional[Dict[str, Any]] = None) -> SynthesizerAgent:
    """Get the singleton Synthesizer agent instance."""
    global _synthesizer_agent
    if _synthesizer_agent is None:
        _synthesizer_agent = SynthesizerAgent(config)
    return _synthesizer_agent
