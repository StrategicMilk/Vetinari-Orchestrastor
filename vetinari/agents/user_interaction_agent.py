"""
User Interaction Agent - Vetinari Phase 5

Handles ambiguity detection, clarifying question generation, and
user context gathering during pipeline execution.

Can pause the pipeline to gather information from the user via:
- CLI prompts (interactive mode)
- Web API callback endpoints (web UI mode)
- Pre-defined context (non-interactive mode)
"""

import logging
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class UserInteractionAgent(BaseAgent):
    """
    Handles user interaction for clarification and context gathering.

    Detects ambiguous or under-specified goals and generates targeted
    clarifying questions. Can operate in:
    - interactive: CLI prompts (sys.stdin)
    - callback: Custom callback function
    - non_interactive: Returns questions without blocking
    """

    INTERACTION_MODES = ("interactive", "callback", "non_interactive")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.USER_INTERACTION, config)
        self._mode = config.get("mode", "interactive") if config else "interactive"
        self._callback: Optional[Callable[[str, List[str]], str]] = None
        self._pending_questions: List[Dict[str, Any]] = []
        self._gathered_context: Dict[str, Any] = {}

    def set_interaction_mode(self, mode: str, callback: Callable = None) -> None:
        """Set the interaction mode."""
        if mode not in self.INTERACTION_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.INTERACTION_MODES}")
        self._mode = mode
        self._callback = callback

    def get_system_prompt(self) -> str:
        return """You are Vetinari's User Interaction Specialist. Your role is to:
1. Detect when user goals are ambiguous or under-specified
2. Generate clear, targeted clarifying questions
3. Prioritize the most important questions (max 3)
4. Incorporate user responses into the task context

Be concise and specific. Ask one focused question at a time when possible."""

    def get_capabilities(self) -> List[str]:
        return [
            "ambiguity_detection",
            "clarification_generation",
            "context_gathering",
            "user_prompt",
            "response_integration",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute user interaction for a task.

        Returns AgentResult with:
        - output: gathered context dict
        - metadata: questions_asked, responses_gathered
        - needs_user_input: True if non-interactive and questions pending
        """
        task = self.prepare_task(task)

        goal = task.context.get("goal", task.description)
        existing_context = task.context.get("existing_context", {})
        max_questions = task.context.get("max_questions", 3)

        # Check for ambiguity
        is_ambiguous, questions = self._detect_ambiguity(goal, existing_context)

        if not is_ambiguous or not questions:
            return AgentResult(
                success=True,
                output=existing_context,
                metadata={"questions_asked": 0, "responses_gathered": 0, "ambiguous": False},
            )

        # Limit questions
        questions = questions[:max_questions]
        self._pending_questions = [{"question": q, "answered": False} for q in questions]

        # Gather responses based on mode
        if self._mode == "interactive":
            responses = self._interactive_prompt(questions)
        elif self._mode == "callback" and self._callback:
            responses = self._callback_prompt(goal, questions)
        else:
            # Non-interactive: return questions for the caller to handle
            return AgentResult(
                success=True,
                output={
                    "pending_questions": questions,
                    "needs_user_input": True,
                    "existing_context": existing_context,
                },
                metadata={
                    "questions_asked": len(questions),
                    "responses_gathered": 0,
                    "ambiguous": True,
                    "needs_user_input": True,
                },
            )

        # Incorporate responses into context
        enriched_context = dict(existing_context)
        for q, r in zip(questions, responses):
            enriched_context[f"clarification_{len(enriched_context)}"] = {
                "question": q,
                "answer": r,
            }
        self._gathered_context = enriched_context

        return AgentResult(
            success=True,
            output=enriched_context,
            metadata={
                "questions_asked": len(questions),
                "responses_gathered": len(responses),
                "ambiguous": True,
            },
        )

    def verify(self, output: Any) -> VerificationResult:
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"type": "invalid_output"}], score=0.0)
        return VerificationResult(passed=True, issues=[], score=1.0)

    def _detect_ambiguity(self, goal: str, context: Dict[str, Any]) -> tuple:
        """
        Detect if a goal is ambiguous and generate clarifying questions.

        Returns:
            Tuple of (is_ambiguous, questions_list).
        """
        # First try LLM-based detection
        prompt = f"""Analyze this goal for ambiguity: "{goal}"

Context already available: {list(context.keys())}

Is this goal specific enough to execute? Generate clarifying questions if needed.

Respond as JSON:
{{
  "is_ambiguous": true/false,
  "ambiguity_reasons": ["reason1", ...],
  "questions": ["Specific question 1?", "Specific question 2?"],
  "missing_information": ["info1", ...]
}}

Only flag as ambiguous if critical information is missing. Don't ask about nice-to-have details."""

        result = self._infer_json(prompt)
        if result and isinstance(result, dict):
            is_ambiguous = result.get("is_ambiguous", False)
            questions = result.get("questions", [])
            return is_ambiguous, questions

        # Fallback: heuristic detection
        questions = []
        g = goal.lower()

        if len(goal.split()) < 5:
            questions.append("Could you provide more details about what you want to accomplish?")

        if any(w in g for w in ["something", "stuff", "things", "it"]):
            questions.append("Can you be more specific about what 'it' or 'this' refers to?")

        if "build" in g or "create" in g or "make" in g:
            if not any(w in g for w in ["python", "javascript", "web", "api", "cli", "app"]):
                questions.append("What technology stack or programming language should be used?")

        if "improve" in g or "fix" in g or "optimize" in g:
            if "vetinari" not in g and not context.get("target_file"):
                questions.append("What specific component or file should be improved?")

        return len(questions) > 0, questions

    def _interactive_prompt(self, questions: List[str]) -> List[str]:
        """Prompt the user interactively via CLI."""
        responses: List[str] = []
        print("\n[Vetinari] Additional context needed:")
        print("-" * 40)

        for i, question in enumerate(questions, 1):
            print(f"\n{i}. {question}")
            try:
                if sys.stdin.isatty():
                    response = input("   > ").strip()
                else:
                    response = sys.stdin.readline().strip()
                responses.append(response or "(no response)")
            except (EOFError, KeyboardInterrupt):
                responses.append("(skipped)")

        print("-" * 40)
        return responses

    def _callback_prompt(self, goal: str, questions: List[str]) -> List[str]:
        """Use callback function to gather responses."""
        if not self._callback:
            return ["(no callback)"] * len(questions)
        try:
            result = self._callback(goal, questions)
            if isinstance(result, list):
                return result
            if isinstance(result, str):
                return [result] * len(questions)
        except Exception as e:
            logger.warning(f"Interaction callback failed: {e}")
        return ["(callback error)"] * len(questions)

    def ask_for_more_context(self, goal: str, specific_need: str) -> str:
        """
        Ask the user for a specific piece of missing context.

        Args:
            goal: The current goal being worked on.
            specific_need: What specific information is needed.

        Returns:
            The user's response or empty string if non-interactive.
        """
        question = f"For goal '{goal[:60]}', I need: {specific_need}"

        if self._mode == "interactive":
            print(f"\n[Vetinari] {question}")
            try:
                return input("   > ").strip() if sys.stdin.isatty() else sys.stdin.readline().strip()
            except (EOFError, KeyboardInterrupt):
                return ""
        elif self._mode == "callback" and self._callback:
            responses = self._callback_prompt(goal, [question])
            return responses[0] if responses else ""
        return ""

    def get_pending_questions(self) -> List[Dict[str, Any]]:
        """Get any unanswered questions (for web UI polling)."""
        return [q for q in self._pending_questions if not q.get("answered")]

    def answer_question(self, question: str, answer: str) -> None:
        """Provide an answer to a pending question (web UI callback)."""
        for q in self._pending_questions:
            if q["question"] == question:
                q["answer"] = answer
                q["answered"] = True
                self._gathered_context[f"clarification_{question[:30]}"] = {
                    "question": question,
                    "answer": answer,
                }
                break

    def get_gathered_context(self) -> Dict[str, Any]:
        """Get all gathered context from user interactions."""
        return dict(self._gathered_context)


# Singleton
_user_interaction_agent: Optional[UserInteractionAgent] = None


def get_user_interaction_agent(config: Optional[Dict[str, Any]] = None) -> UserInteractionAgent:
    """Get the singleton User Interaction agent instance."""
    global _user_interaction_agent
    if _user_interaction_agent is None:
        _user_interaction_agent = UserInteractionAgent(config)
    return _user_interaction_agent
