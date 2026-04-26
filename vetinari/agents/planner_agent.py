"""Vetinari Foreman Agent (v0.5.0).

The Foreman is the central planning and user interaction agent in the
3-agent factory pipeline (Foreman -> Worker -> Inspector). It generates
dynamic plans from goals, coordinates Worker task assignment, and handles
user clarification and context management.

Modes: plan, clarify, consolidate, summarise, prune, extract

Mode prompts live in planner_prompts.py; decomposition helpers in planner_decompose.py.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    Plan,
    VerificationResult,
)
from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.planner_decompose import decompose_goal_keyword, decompose_goal_llm, is_vague_goal
from vetinari.agents.planner_prompts import FOREMAN_MODE_PROMPTS
from vetinari.constants import TRUNCATE_CONTENT_ANALYSIS, TRUNCATE_CONTEXT
from vetinari.exceptions import JurisdictionViolation
from vetinari.plan_cache import PlanCache, get_plan_cache
from vetinari.types import AgentType

logger = logging.getLogger(__name__)


class ForemanAgent(MultiModeAgent):
    """Foreman agent - planning, user interaction, and context management.

    The Foreman orchestrates the factory pipeline by decomposing goals into
    task DAGs, assigning work to the Worker, and managing user interaction.
    """

    MODES = {
        "plan": "_execute_plan",
        "clarify": "_execute_clarify",
        "consolidate": "_execute_consolidate",
        "summarise": "_execute_summarise",
        "prune": "_execute_prune",
        "extract": "_execute_extract",
    }
    DEFAULT_MODE = "plan"
    MODE_KEYWORDS = {
        "plan": ["plan", "decompose", "schedule", "specify", "goal", "task", "breakdown"],
        "clarify": ["ambiguous", "clarif", "question", "unclear", "vague", "user input"],
        "consolidate": ["consolidat", "memory", "merge", "context"],
        "summarise": ["summari", "summariz", "digest", "recap"],
        "prune": ["prune", "trim", "reduce", "budget", "token limit"],
        "extract": ["extract", "knowledge", "entities", "structured"],
    }
    _MAX_ENTRIES_FOR_CONSOLIDATION = 50
    # Foreman is a pure coordinator — inference is restricted to planning modes.
    # Task execution inference must go through Worker. See ADR-0093.
    _PLANNING_MODES: frozenset[str] = frozenset(MODES.keys())

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(AgentType.FOREMAN, config)
        self._max_depth = self._config.get("max_depth", 14)
        self._min_tasks = self._config.get("min_tasks", 5)
        self._max_tasks = self._config.get("max_tasks", 15)
        # Orchestrator state (absorbed)
        self._interaction_mode = (config or {}).get("mode", "interactive")  # noqa: VET112 - empty fallback preserves optional request metadata contract
        self._callback: Callable | None = None
        self._pending_questions: list[dict[str, Any]] = []
        self._gathered_context: dict[str, Any] = {}
        self._max_context_tokens = int(
            (config or {}).get("max_context_tokens", os.environ.get("VETINARI_MAX_CONTEXT_TOKENS", "4096")),  # noqa: VET112 - empty fallback preserves optional request metadata contract
        )
        self._plan_cache: PlanCache = get_plan_cache()

    def _infer(self, prompt: str, **kwargs: Any) -> str:
        """Guard that restricts Foreman inference to planning/coordination only.

        The Foreman decomposes goals and delegates execution to Workers.
        Direct task-execution inference MUST go through the Worker pipeline.
        All six Foreman modes (plan, clarify, consolidate, summarise, prune,
        extract) are planning modes — this guard catches calls made outside
        a recognized mode context or from future non-planning modes.

        Args:
            prompt: The user/task prompt forwarded to the LLM.
            **kwargs: Remaining inference parameters forwarded to super().

        Returns:
            LLM response text.

        Raises:
            JurisdictionViolation: If called outside a recognized planning mode.
        """
        if self._current_mode not in self._PLANNING_MODES:
            raise JurisdictionViolation(
                f"Foreman inference blocked — mode {self._current_mode!r} is not a "
                f"planning mode. Task execution must be delegated to Worker. "
                f"Allowed modes: {sorted(self._PLANNING_MODES)}"
            )
        return super()._infer(prompt, **kwargs)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Foreman — the factory pipeline orchestrator. "
            "You handle goal decomposition, task scheduling, Worker assignment, "
            "user interaction (ambiguity detection, clarifying questions), and "
            "context management (memory consolidation, session summarisation, "
            "knowledge extraction)."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        """Return the LLM system prompt for the given Foreman mode.

        Prompts are stored in planner_prompts.py to keep this file under
        the 550-line limit.

        Args:
            mode: One of plan, clarify, consolidate, summarise, prune, extract.

        Returns:
            System prompt string, or empty string for unknown modes.
        """
        return FOREMAN_MODE_PROMPTS.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        """Verify output — mode-aware.

        Returns:
            The VerificationResult result.
        """
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"message": "Output must be a dict"}], score=0.0)

        mode = self._current_mode or self.DEFAULT_MODE
        if mode == "plan":
            issues = []
            score = 1.0
            required_fields = ["plan_id", "goal", "tasks"]
            for f in required_fields:
                if f not in output:
                    issues.append({"type": "missing_field", "message": f"Missing: {f}"})
                    score -= 0.2
            tasks = output.get("tasks", [])
            if len(tasks) < self._min_tasks:
                issues.append({"type": "insufficient_tasks", "message": f"Too few tasks: {len(tasks)}"})
                score -= 0.1
            if not any(t.get("dependencies") for t in tasks):
                issues.append({"type": "no_dependencies", "message": "No task dependencies"})
                score -= 0.1
            return VerificationResult(passed=score >= 0.7, issues=issues, score=max(0, score))

        return VerificationResult(passed=True, score=0.8)

    def get_capabilities(self) -> list[str]:
        """Return capability strings describing this agent's supported modes and features.

        Returns:
            List of capability identifiers such as plan generation,
            task decomposition, and risk assessment.
        """
        return [
            "plan_generation",
            "task_decomposition",
            "dependency_mapping",
            "resource_estimation",
            "risk_assessment",
            "ambiguity_detection",
            "clarification_generation",
            "context_gathering",
            "memory_consolidation",
            "session_summarisation",
            "context_pruning",
            "knowledge_extraction",
        ]

    # ------------------------------------------------------------------
    # Plan mode
    # ------------------------------------------------------------------

    def _execute_plan(self, task: AgentTask) -> AgentResult:
        """Generate a plan from the goal, with cache lookup and storage.

        Checks the plan cache for a similar past plan before generating a new
        one. On a cache hit the cached plan data is returned directly. On a
        miss a new plan is generated and stored in the cache for future reuse.

        Args:
            task: The agent task carrying the goal prompt and context.

        Returns:
            An AgentResult whose output is the plan dict and whose metadata
            includes the plan_id, task_count, goal, and whether the plan was
            served from cache.
        """
        goal = task.prompt or task.description
        context = task.context

        # Check cache for a similar past plan before generating a new one.
        cached = self._plan_cache.find_similar(goal)
        if cached is not None:
            logger.info(
                "Plan cache hit for goal (hash=%s, hit_count=%d) — skipping LLM decomposition",
                cached.goal_hash,
                cached.hit_count,
            )
            return AgentResult(
                success=True,
                output=cached.plan_data,
                metadata={
                    "plan_id": cached.plan_data.get("plan_id"),
                    "task_count": len(cached.plan_data.get("tasks", [])),
                    "goal": goal,
                    "from_cache": True,
                    "cache_hit_count": cached.hit_count,
                },
            )

        plan = self._generate_plan(goal, context)

        # Store the new plan in the cache for future reuse.
        self._plan_cache.store(goal, plan.to_dict())
        logger.debug("Stored new plan in cache (goal_hash=%s)", self._plan_cache._goal_hash(goal))

        return AgentResult(
            success=True,
            output=plan.to_dict(),
            metadata={
                "plan_id": plan.plan_id,
                "task_count": len(plan.tasks),
                "goal": goal,
                "from_cache": False,
            },
        )

    def _decompose_goal_keyword(self, goal: str, context: dict[str, Any] | None = None) -> list:
        """Decompose a goal into tasks using keyword heuristics — thin wrapper around the module-level helper.

        Delegates to ``decompose_goal_keyword`` in ``planner_decompose``. The
        *context* parameter is accepted for API symmetry with ``_decompose_goal_llm``
        but is not used in the keyword-based path.

        Args:
            goal: The user goal string to decompose.
            context: Ignored; present for call-site symmetry only.

        Returns:
            List of Task objects built from keyword pattern matching.
        """
        return decompose_goal_keyword(goal)

    def _decompose_goal_llm(self, goal: str, context: dict[str, Any], max_tasks: int | None = None) -> list:
        """Decompose a goal into tasks via LLM — thin wrapper around the module-level helper.

        Delegates to ``decompose_goal_llm`` in ``planner_decompose``, which
        contains the full prompt-building and response-parsing logic.

        Args:
            goal: The user goal string to decompose.
            context: Optional context dict; may contain ``request_spec``.
            max_tasks: Maximum tasks to request. Defaults to ``self._max_tasks``.

        Returns:
            List of Task objects with dependencies and depth pre-computed.
            Returns an empty list if the LLM returns nothing useful.
        """
        return decompose_goal_llm(self, goal, context, max_tasks if max_tasks is not None else self._max_tasks)

    def _generate_plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Generate a plan from the goal using LLM-powered decomposition.

        Falls back to keyword-based decomposition if the LLM is unavailable.

        Args:
            goal: The user goal string.
            context: Context dict passed through to decomposition helpers.

        Returns:
            Plan with tasks and DAG structure populated.
        """
        plan = Plan.create_new(goal)

        if is_vague_goal(goal):
            plan.needs_context = True
            plan.follow_up_question = "Could you provide more details about what you want to build?"
            return plan

        tasks = decompose_goal_llm(self, goal, context, self._max_tasks)
        if not tasks:
            tasks = decompose_goal_keyword(goal)

        plan.tasks = tasks
        if len(tasks) > self._max_tasks:
            plan.warnings.append(f"Generated {len(tasks)} tasks - consider breaking into smaller goals")

        return plan

    # ------------------------------------------------------------------
    # Clarify mode
    # ------------------------------------------------------------------

    def _execute_clarify(self, task: AgentTask) -> AgentResult:
        goal = task.context.get("goal", task.description)
        existing_context = task.context.get("existing_context", {})
        max_questions = task.context.get("max_questions", 3)

        is_ambiguous, questions = self._detect_ambiguity(goal, existing_context)

        if not is_ambiguous or not questions:
            return AgentResult(
                success=True,
                output=existing_context,
                metadata={"questions_asked": 0, "ambiguous": False},
            )

        questions = questions[:max_questions]
        self._pending_questions = [{"question": q, "answered": False} for q in questions]

        if self._interaction_mode == "interactive":
            responses = self._interactive_prompt(questions)
        elif self._interaction_mode == "callback" and self._callback:
            responses = self._callback_prompt(goal, questions)
        else:
            return AgentResult(
                success=True,
                output={"pending_questions": questions, "needs_user_input": True, "existing_context": existing_context},
                metadata={"questions_asked": len(questions), "needs_user_input": True},
            )

        enriched = dict(existing_context)
        for q, r in zip(questions, responses):
            enriched[f"clarification_{len(enriched)}"] = {"question": q, "answer": r}
        self._gathered_context = enriched

        return AgentResult(
            success=True,
            output=enriched,
            metadata={"questions_asked": len(questions), "responses_gathered": len(responses)},
        )

    def _detect_ambiguity(self, goal: str, context: dict) -> tuple:
        prompt = (
            f'Analyze this goal for ambiguity: "{goal}"\n'
            f"Context available: {list(context.keys())}\n\n"
            "Respond as JSON:\n"
            '{"is_ambiguous": true/false, "questions": ["..."], "missing_information": ["..."]}\n\n'
            "Only flag as ambiguous if critical information is missing."
        )
        result = self._infer_json(prompt)
        if result and isinstance(result, dict):
            return result.get("is_ambiguous", False), result.get("questions", [])

        # Heuristic fallback
        questions = []
        g = goal.lower()
        if len(goal.split()) < 5:
            questions.append("Could you provide more details about what you want to accomplish?")
        if any(w in g for w in ["something", "stuff", "things", "it"]):
            questions.append("Can you be more specific about what 'it' refers to?")
        if any(w in g for w in ["build", "create", "make"]) and not any(
            w in g for w in ["python", "javascript", "web", "api", "cli"]
        ):
            questions.append("What technology stack should be used?")
        return len(questions) > 0, questions

    def _interactive_prompt(self, questions: list[str]) -> list[str]:
        responses = []
        logger.info("Additional context needed:")
        for i, q in enumerate(questions, 1):
            logger.info("%d. %s", i, q)
            try:
                r = input("   > ").strip() if sys.stdin.isatty() else sys.stdin.readline().strip()
                responses.append(r or "(no response)")
            except (EOFError, KeyboardInterrupt):
                responses.append("(skipped)")
        return responses

    def _callback_prompt(self, goal: str, questions: list[str]) -> list[str]:
        if not self._callback:
            return ["(no callback)"] * len(questions)
        try:
            result = self._callback(goal, questions)
            return result if isinstance(result, list) else [str(result)] * len(questions)
        except Exception:
            logger.warning(
                "Clarification callback raised an error for goal %r — substituting '(callback error)' for all %d questions",
                goal,
                len(questions),
            )
            return ["(callback error)"] * len(questions)

    def set_interaction_mode(self, mode: str, callback: Callable | None = None) -> None:
        """Configure how the Foreman collects answers during ``clarify`` operations.

        Args:
            mode: Interaction mode — ``"auto"`` generates synthetic answers,
                ``"callback"`` invokes *callback* for each question, or any other
                value falls back to returning empty strings.
            callback: Callable invoked with a list of question strings; must return
                a list of answer strings of the same length. Only used when
                *mode* is ``"callback"``.
        """
        self._interaction_mode = mode
        self._callback = callback

    # ------------------------------------------------------------------
    # Consolidate / Summarise / Prune / Extract modes
    # ------------------------------------------------------------------

    def _execute_consolidate(self, task: AgentTask) -> AgentResult:
        ctx = task.context
        session_id = ctx.get("session_id", "")
        project_id = ctx.get("project_id", "")
        entries = self._load_memory_entries(session_id, project_id)

        if not entries:
            return AgentResult(
                success=True,
                output=self._fallback_consolidation(task, []),
                metadata={"operation": "consolidate", "entries_processed": 0},
            )

        entries_text = json.dumps(entries[: self._MAX_ENTRIES_FOR_CONSOLIDATION], indent=2)[:TRUNCATE_CONTEXT]
        prompt = (
            f"Consolidate the following {len(entries)} memory entries. "
            f"Extract key knowledge, identify patterns, create concise summary.\n\n"
            f"## Entries\n{entries_text}\n\n"
            "## Output (JSON)\n"
            '{"consolidated_summary": "...", "key_knowledge": [{"fact": "...", "confidence": 0.9}], '
            '"patterns_identified": [...], "entries_processed": ' + str(len(entries)) + "}"
        )
        result = self._infer_json(prompt, fallback=self._fallback_consolidation(task, entries))
        if result and isinstance(result, dict):
            result.setdefault("entries_processed", len(entries))
            return AgentResult(
                success=True,
                output=result,
                metadata={"operation": "consolidate", "entries_processed": len(entries)},
            )
        fb = self._fallback_consolidation(task, entries)
        return AgentResult(success=True, output=fb, metadata={"operation": "consolidate"})

    def _execute_summarise(self, task: AgentTask) -> AgentResult:
        ctx = task.context
        history = ctx.get("history", []) or ctx.get("messages", [])
        if not history:
            history = self._load_memory_entries(ctx.get("session_id", ""), ctx.get("project_id", ""))

        prompt = (
            f"Summarise {len(history)} session entries for an AI orchestration system.\n\n"
            f"## History\n{json.dumps(history[:30], indent=2)[:TRUNCATE_CONTENT_ANALYSIS]}\n\n"
            "## Output (JSON)\n"
            '{"session_summary": "...", "goals_achieved": [...], "next_steps": [...], '
            '"entries_processed": ' + str(len(history)) + "}"
        )
        result = self._infer_json(prompt, fallback=self._fallback_consolidation(task, history))
        if result and isinstance(result, dict):
            result.setdefault("entries_processed", len(history))
            return AgentResult(success=True, output=result, metadata={"operation": "summarise"})
        return AgentResult(success=True, output=self._fallback_consolidation(task, history))

    def _execute_prune(self, task: AgentTask) -> AgentResult:
        ctx = task.context or {}
        entries = ctx.get("entries", [])
        max_tokens = ctx.get("max_tokens", self._max_context_tokens)

        if not entries:
            return AgentResult(
                success=True,
                output={
                    "consolidated_summary": "No entries to prune",
                    "pruned_count": 0,
                    "entries_processed": 0,
                },
            )

        prompt = (
            f"Prune context to fit within {max_tokens} tokens. "
            f"Keep highest relevance entries.\n\n"
            f"## Entries ({len(entries)})\n{json.dumps(entries[:40], indent=2)[:TRUNCATE_CONTENT_ANALYSIS]}\n\n"
            "## Output (JSON)\n"
            '{"entries_to_retain": [...], "stale_entries": [...], "pruned_count": 0}'
        )
        result = self._infer_json(prompt, fallback={"pruned_count": 0, "entries_processed": len(entries)})
        if result and isinstance(result, dict):
            return AgentResult(success=True, output=result, metadata={"operation": "prune"})
        return AgentResult(success=True, output={"pruned_count": 0})

    def _execute_extract(self, task: AgentTask) -> AgentResult:
        text = task.context.get("text", "") or task.description or ""
        prompt = (
            f"Extract structured knowledge from:\n{text[:TRUNCATE_CONTENT_ANALYSIS]}\n\n"
            "## Output (JSON)\n"
            '{"key_knowledge": [{"fact": "...", "confidence": 0.9}], '
            '"entities_discovered": [{"name": "...", "type": "..."}]}'
        )
        result = self._infer_json(prompt, fallback={"key_knowledge": [], "entities_discovered": []})
        if result and isinstance(result, dict):
            return AgentResult(success=True, output=result, metadata={"operation": "extract"})
        return AgentResult(success=True, output={"key_knowledge": []})

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _load_memory_entries(self, session_id: str, project_id: str) -> list[dict]:
        entries = []
        try:
            from vetinari.memory.unified import get_unified_memory_store

            store = get_unified_memory_store()
            if hasattr(store, "search"):
                results = store.search("", limit=50)
                entries.extend(r.to_dict() if hasattr(r, "to_dict") else r for r in results or [])  # noqa: VET112 - empty fallback preserves optional request metadata contract
        except Exception:
            logger.warning("Failed to load memory entries from unified memory store", exc_info=True)
        return entries

    def _fallback_consolidation(self, task: AgentTask, entries: list) -> dict[str, Any]:
        return {
            "consolidated_summary": f"Context consolidation for: {(task.description or 'session')[:100]}",
            "session_summary": f"Processed {len(entries)} entries. LLM unavailable.",
            "key_knowledge": [],
            "entries_processed": len(entries),
            "retrieval_recommendations": [{"query_type": "semantic", "strategy": "hybrid"}],
        }

    # ------------------------------------------------------------------
    # Mode verification requirements
    # ------------------------------------------------------------------

    def validate_agent_output(
        self,
        agent_type: str,
        mode: str,
        output: dict[str, Any] | None,
    ) -> tuple[bool, list[str]]:
        """Validate agent output against mode-specific verification requirements.

        Checks the output against ``MODE_VERIFICATION_REQUIREMENTS`` defined in
        ``vetinari.agents.practices``. Returns pass/fail and a list of unmet
        requirements for rework routing.

        Args:
            agent_type: The agent type value (e.g. "WORKER").
            mode: The mode name (e.g. "build").
            output: The agent's output dict to validate.

        Returns:
            Tuple of (passed: bool, unmet_requirements: list[str]).
        """
        try:
            from vetinari.agents.practices import get_verification_requirements
        except ImportError:
            logger.warning(
                "Could not import vetinari.agents.practices — cannot validate %s:%s output, treating as failed",
                agent_type,
                mode,
            )
            return False, ["agent practices module unavailable"]

        requirements = get_verification_requirements(agent_type, mode)
        if not requirements:
            return True, []

        if output is None:
            return False, requirements

        metadata = output if isinstance(output, dict) else {}
        verification = metadata.get("verification", {})

        unmet: list[str] = [req for req in requirements if not verification.get(req, False)]
        passed = len(unmet) == 0
        if not passed:
            logger.warning(
                "Agent %s:%s failed verification — unmet requirements: %s",
                agent_type,
                mode,
                unmet,
            )
        return passed, unmet


# Singleton instance
_foreman_agent: ForemanAgent | None = None
_foreman_agent_lock = threading.Lock()


def get_foreman_agent(config: dict[str, Any] | None = None) -> ForemanAgent:
    """Get the singleton Foreman agent instance.

    Args:
        config: Optional configuration dict.

    Returns:
        A configured ForemanAgent instance.
    """
    global _foreman_agent
    if _foreman_agent is None:
        with _foreman_agent_lock:
            if _foreman_agent is None:
                _foreman_agent = ForemanAgent(config)
    return _foreman_agent
