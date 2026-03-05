"""
Prompt Rewriter Agent — Stage 0.5 of the Vetinari pipeline.

Translates user natural language into structured, AI-optimised prompts using
the INSTRUCTIONS / CONTEXT / TASK / OUTPUT FORMAT four-block pattern. Also
produces agent-specific prompt variants (coding, research, planning, etc.) so
each downstream agent receives a prompt tuned to its role.
"""

import json
import re
from typing import Any, Dict, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, VerificationResult
from vetinari.types import AgentType


class PromptRewriterAgent(BaseAgent):
    """
    Stage 0.5 agent: disambiguates, restructures, and enriches user prompts.

    Input context (from PromptAssessorAgent):
        intent, scope, tasks, complexity, context_summary

    Output JSON schema:
    {
        "original_prompt": "...",
        "rewritten_prompt": "...",
        "instructions": "...",
        "context": "...",
        "task": "...",
        "output_format": "...",
        "variants": {
            "coding": "...",
            "research": "...",
            "planning": "..."
        },
        "do_not": ["do not add unrelated features", ...]
    }
    """

    _SYSTEM = (
        "You are Vetinari's Prompt Rewriter — Stage 0.5 of the pipeline. "
        "Your job is to transform a user's natural-language request into a "
        "structured, AI-optimised prompt.\n\n"
        "Apply the four-block pattern:\n"
        "  INSTRUCTIONS — role, persona, and behavioural constraints for the AI.\n"
        "  CONTEXT      — background information, existing codebase state, "
        "dependencies, constraints.\n"
        "  TASK         — numbered list of explicit, atomic action items using "
        "strong action verbs (implement, create, modify, return, validate).\n"
        "  OUTPUT FORMAT — precise spec of what the final output must contain.\n\n"
        "Also produce short variants for: coding, research, planning.\n\n"
        "Output ONLY valid JSON with these fields:\n"
        '{"original_prompt": "...", "rewritten_prompt": "full 4-block prompt", '
        '"instructions": "...", "context": "...", "task": "...", '
        '"output_format": "...", '
        '"variants": {"coding": "...", "research": "...", "planning": "..."}, '
        '"do_not": ["scope boundary 1", ...]}'
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.PROMPT_REWRITER, config)

    def get_system_prompt(self) -> str:
        return self._SYSTEM

    def execute(self, task: AgentTask) -> AgentResult:
        prompt = task.prompt or task.description or ""
        assessment: Dict[str, Any] = task.context or {}

        # Build rich user message incorporating assessor output
        parts = [f"Rewrite this user prompt:\n\n{prompt}"]
        if assessment:
            parts.append(
                f"\nAssessment context:\n"
                f"  Intent:     {assessment.get('intent', 'unknown')}\n"
                f"  Scope:      {assessment.get('scope', '')}\n"
                f"  Complexity: {assessment.get('complexity', 'moderate')}\n"
                f"  Tasks:      {', '.join(assessment.get('tasks', [])[:5])}\n"
                f"  Summary:    {assessment.get('context_summary', '')}"
            )
        user_msg = "\n".join(parts)

        try:
            raw = self._call_llm(self._SYSTEM, user_msg, max_tokens=2048)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = self._heuristic_rewrite(prompt, assessment)

            return AgentResult(
                task_id=task.task_id,
                success=True,
                output=result,
                agent_type=self._agent_type.value,
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                output=self._heuristic_rewrite(prompt, assessment),
                errors=[str(e)],
                agent_type=self._agent_type.value,
            )

    def verify(self, result: AgentResult) -> VerificationResult:
        output = result.output
        if not isinstance(output, dict):
            return VerificationResult(passed=False, score=0.0, feedback="Output is not a dict")
        required = {"rewritten_prompt", "task", "variants"}
        missing = required - output.keys()
        if missing:
            return VerificationResult(
                passed=False, score=0.5,
                feedback=f"Missing fields: {missing}",
            )
        variants = output.get("variants", {})
        if not isinstance(variants, dict) or not variants:
            return VerificationResult(
                passed=False, score=0.7,
                feedback="variants must be a non-empty dict",
            )
        return VerificationResult(passed=True, score=1.0, feedback="Rewrite complete")

    # ------------------------------------------------------------------
    # Heuristic fallback (no LLM required)
    # ------------------------------------------------------------------

    def _heuristic_rewrite(
        self, prompt: str, assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Lightweight fallback when LLM is unavailable."""
        intent = assessment.get("intent", "implementation")
        complexity = assessment.get("complexity", "moderate")
        scope = assessment.get("scope", prompt[:120])
        tasks_list = assessment.get("tasks", [prompt[:80]])

        # Build the four blocks
        role_map = {
            "implementation": "expert software engineer",
            "research": "thorough research analyst",
            "investigation": "skilled software investigator",
            "fix": "experienced debugger",
            "refactor": "clean-code advocate",
            "documentation": "technical writer",
        }
        role = role_map.get(intent, "senior software engineer")

        instructions = (
            f"You are a {role}. Produce complete, production-ready output. "
            "Do not use placeholders, stubs, or TODO comments."
        )
        context_block = (
            f"Scope: {scope}. Complexity: {complexity}. "
            f"Original request: {prompt[:200]}"
        )
        task_block = "\n".join(
            f"{i + 1}. {t}" for i, t in enumerate(tasks_list[:10])
        ) or f"1. {prompt[:120]}"
        output_format = (
            "Return the complete implementation with all files, "
            "no placeholders, no truncated sections."
        )

        rewritten = (
            f"## INSTRUCTIONS\n{instructions}\n\n"
            f"## CONTEXT\n{context_block}\n\n"
            f"## TASK\n{task_block}\n\n"
            f"## OUTPUT FORMAT\n{output_format}"
        )

        # Agent-specific variants
        coding_variant = (
            f"Implement the following using complete, tested code: {prompt[:200]}. "
            "All imports must resolve. No stubs."
        )
        research_variant = (
            f"Research and summarise: {prompt[:200]}. "
            "Return findings, sources, and key insights."
        )
        planning_variant = (
            f"Create a step-by-step plan for: {prompt[:200]}. "
            "Decompose into atomic tasks with clear dependencies."
        )

        do_not = [
            "Do not add unrelated features",
            "Do not over-engineer beyond the stated requirements",
            "Do not leave any placeholder or stub code",
        ]

        return {
            "original_prompt": prompt,
            "rewritten_prompt": rewritten,
            "instructions": instructions,
            "context": context_block,
            "task": task_block,
            "output_format": output_format,
            "variants": {
                "coding": coding_variant,
                "research": research_variant,
                "planning": planning_variant,
            },
            "do_not": do_not,
        }


# Module-level singleton
_rewriter_instance: Optional["PromptRewriterAgent"] = None


def get_prompt_rewriter_agent() -> "PromptRewriterAgent":
    """Return the module-level PromptRewriterAgent singleton."""
    global _rewriter_instance
    if _rewriter_instance is None:
        _rewriter_instance = PromptRewriterAgent()
    return _rewriter_instance
