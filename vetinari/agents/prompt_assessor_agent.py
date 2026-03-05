"""
Prompt Assessor Agent — Stage 0 of the Vetinari pipeline.

Analyzes user intent, estimates scope/complexity, and produces structured
context that enriches planning and model selection for all subsequent stages.
"""

import json
import re
from typing import Any, Dict, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, VerificationResult
from vetinari.types import AgentType


class PromptAssessorAgent(BaseAgent):
    """
    Stage 0 agent: Intent classification, scope analysis, complexity estimation.

    Output JSON schema:
    {
        "intent": "implementation|research|investigation|fix|refactor|documentation",
        "scope": "brief description of scope",
        "tasks": ["explicit task 1", "implicit task 2", ...],
        "files_affected": ["file1.py", ...],
        "dependencies": ["lib1", "lib2", ...],
        "complexity": "simple|moderate|complex|massive",
        "suggested_depth": 3,
        "context_summary": "key context for planners"
    }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.PROMPT_ASSESSOR, config)

    def get_system_prompt(self) -> str:
        return (
            "You are Vetinari's Prompt Assessor — the first agent in the pipeline. "
            "Your role is to deeply analyse the user's request before any planning begins.\n\n"
            "You MUST output ONLY valid JSON with these fields:\n"
            '{"intent": "implementation|research|investigation|fix|refactor|documentation",\n'
            ' "scope": "one-sentence description",\n'
            ' "tasks": ["explicit task list derived from the prompt"],\n'
            ' "files_affected": ["files likely to change or be created"],\n'
            ' "dependencies": ["libraries/services/configs needed"],\n'
            ' "complexity": "simple|moderate|complex|massive",\n'
            ' "suggested_depth": 3,\n'
            ' "context_summary": "key context for downstream agents"}\n\n'
            "Complexity rules:\n"
            "- simple: 1-3 tasks, no external dependencies, < 1 hour\n"
            "- moderate: 4-10 tasks, some dependencies, 1-4 hours\n"
            "- complex: 11-30 tasks, multiple systems, 4-16 hours\n"
            "- massive: 30+ tasks, architectural changes, multi-day\n\n"
            "suggested_depth: simple→3, moderate→6, complex→10, massive→16"
        )

    def execute(self, task: AgentTask) -> AgentResult:
        prompt = task.prompt or task.description or ""
        context = task.context or {}

        user_msg = f"Analyse this user request:\n\n{prompt}"
        if context:
            user_msg += f"\n\nAdditional context: {json.dumps(context, default=str)[:500]}"

        try:
            raw = self._call_llm(self.get_system_prompt(), user_msg, max_tokens=1024)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                assessment = json.loads(match.group())
            else:
                # Fallback: heuristic assessment
                assessment = self._heuristic_assess(prompt)

            return AgentResult(
                task_id=task.task_id,
                success=True,
                output=assessment,
                agent_type=self._agent_type.value,
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                output=self._heuristic_assess(prompt),
                errors=[str(e)],
                agent_type=self._agent_type.value,
            )

    def verify(self, result: AgentResult) -> VerificationResult:
        output = result.output
        if not isinstance(output, dict):
            return VerificationResult(passed=False, score=0.0, feedback="Output is not a dict")
        required = {"intent", "scope", "complexity", "suggested_depth"}
        missing = required - output.keys()
        if missing:
            return VerificationResult(
                passed=False, score=0.5,
                feedback=f"Missing fields: {missing}"
            )
        valid_complexities = {"simple", "moderate", "complex", "massive"}
        if output.get("complexity") not in valid_complexities:
            return VerificationResult(passed=False, score=0.6,
                                      feedback=f"Invalid complexity: {output.get('complexity')}")
        return VerificationResult(passed=True, score=1.0, feedback="Assessment complete")

    def _heuristic_assess(self, prompt: str) -> Dict[str, Any]:
        """Lightweight fallback when LLM is unavailable."""
        words = prompt.lower().split()
        intent = "implementation"
        if any(w in words for w in ("fix", "bug", "error", "broken", "crash")):
            intent = "fix"
        elif any(w in words for w in ("research", "investigate", "find", "why")):
            intent = "research"
        elif any(w in words for w in ("document", "readme", "docs")):
            intent = "documentation"
        elif any(w in words for w in ("refactor", "clean", "reorganize")):
            intent = "refactor"

        word_count = len(words)
        if word_count < 20:
            complexity, depth = "simple", 3
        elif word_count < 60:
            complexity, depth = "moderate", 6
        elif word_count < 150:
            complexity, depth = "complex", 10
        else:
            complexity, depth = "massive", 16

        return {
            "intent": intent,
            "scope": prompt[:100],
            "tasks": [prompt[:80]],
            "files_affected": [],
            "dependencies": [],
            "complexity": complexity,
            "suggested_depth": depth,
            "context_summary": f"Heuristic assessment: {intent} task, {complexity} complexity",
        }


# Module-level singleton
_assessor_instance: Optional[PromptAssessorAgent] = None


def get_prompt_assessor_agent() -> PromptAssessorAgent:
    """Return the module-level PromptAssessorAgent singleton."""
    global _assessor_instance
    if _assessor_instance is None:
        _assessor_instance = PromptAssessorAgent()
    return _assessor_instance
