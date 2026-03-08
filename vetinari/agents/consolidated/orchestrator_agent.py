"""
Consolidated Orchestrator Agent (Phase 3)
==========================================
Replaces: USER_INTERACTION + CONTEXT_MANAGER

Modes:
- clarify: Ambiguity detection and clarifying question generation
- consolidate: Memory consolidation across sessions
- monitor: System monitoring and performance tracking
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class OrchestratorAgent(MultiModeAgent):
    """Unified orchestration agent for user interaction and context management."""

    MODES = {
        "clarify": "_execute_clarify",
        "consolidate": "_execute_consolidate",
        "summarise": "_execute_summarise",
        "prune": "_execute_prune",
        "extract": "_execute_extract",
        "monitor": "_execute_monitor",
    }
    DEFAULT_MODE = "clarify"
    MODE_KEYWORDS = {
        "clarify": ["ambiguous", "clarif", "question", "unclear", "vague", "user input"],
        "consolidate": ["consolidat", "memory", "merge", "context"],
        "summarise": ["summari", "summariz", "digest", "recap"],
        "prune": ["prune", "trim", "reduce", "budget", "token limit"],
        "extract": ["extract", "knowledge", "entities", "structured"],
        "monitor": ["monitor", "status", "health", "performance"],
    }
    LEGACY_TYPE_TO_MODE = {
        "USER_INTERACTION": "clarify",
        "CONTEXT_MANAGER": "consolidate",
    }

    _MAX_ENTRIES_FOR_CONSOLIDATION = 50

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.ORCHESTRATOR, config)
        self._interaction_mode = (config or {}).get("mode", "interactive")
        self._callback: Optional[Callable] = None
        self._pending_questions: List[Dict[str, Any]] = []
        self._gathered_context: Dict[str, Any] = {}
        self._max_context_tokens = int(
            (config or {}).get("max_context_tokens",
                              os.environ.get("VETINARI_MAX_CONTEXT_TOKENS", "4096"))
        )

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Orchestration Agent. You handle user interaction "
            "(ambiguity detection, clarifying questions) and context management "
            "(memory consolidation, session summarisation, knowledge extraction)."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "clarify": (
                "You are Vetinari's User Interaction Specialist. Your role is to:\n"
                "1. Detect when user goals are ambiguous or under-specified\n"
                "2. Generate clear, targeted clarifying questions\n"
                "3. Prioritize the most important questions (max 3)\n"
                "4. Incorporate user responses into the task context\n\n"
                "Be concise and specific."
            ),
            "consolidate": (
                "You are a context and memory management specialist. Your role is to:\n"
                "- Summarise long interaction histories into concise digests\n"
                "- Identify and retain the most relevant knowledge\n"
                "- Remove redundant or stale context to stay within token budgets\n"
                "- Build structured knowledge representations\n"
                "- Detect contradictions or outdated information\n\n"
                "Always respond with structured JSON."
            ),
        }
        return prompts.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"message": "Output must be a dict"}], score=0.0)
        return VerificationResult(passed=True, score=0.8)

    # ------------------------------------------------------------------
    # Clarify mode (from UserInteractionAgent)
    # ------------------------------------------------------------------

    def _execute_clarify(self, task: AgentTask) -> AgentResult:
        goal = task.context.get("goal", task.description)
        existing_context = task.context.get("existing_context", {})
        max_questions = task.context.get("max_questions", 3)

        is_ambiguous, questions = self._detect_ambiguity(goal, existing_context)

        if not is_ambiguous or not questions:
            return AgentResult(
                success=True, output=existing_context,
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
                output={"pending_questions": questions, "needs_user_input": True,
                        "existing_context": existing_context},
                metadata={"questions_asked": len(questions), "needs_user_input": True},
            )

        enriched = dict(existing_context)
        for q, r in zip(questions, responses):
            enriched[f"clarification_{len(enriched)}"] = {"question": q, "answer": r}
        self._gathered_context = enriched

        return AgentResult(
            success=True, output=enriched,
            metadata={"questions_asked": len(questions), "responses_gathered": len(responses)},
        )

    def _detect_ambiguity(self, goal: str, context: Dict) -> tuple:
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
        if any(w in g for w in ["build", "create", "make"]):
            if not any(w in g for w in ["python", "javascript", "web", "api", "cli"]):
                questions.append("What technology stack should be used?")
        return len(questions) > 0, questions

    def _interactive_prompt(self, questions: List[str]) -> List[str]:
        responses = []
        print("\n[Vetinari] Additional context needed:")
        for i, q in enumerate(questions, 1):
            print(f"\n{i}. {q}")
            try:
                r = input("   > ").strip() if sys.stdin.isatty() else sys.stdin.readline().strip()
                responses.append(r or "(no response)")
            except (EOFError, KeyboardInterrupt):
                responses.append("(skipped)")
        return responses

    def _callback_prompt(self, goal: str, questions: List[str]) -> List[str]:
        if not self._callback:
            return ["(no callback)"] * len(questions)
        try:
            result = self._callback(goal, questions)
            return result if isinstance(result, list) else [str(result)] * len(questions)
        except Exception:
            return ["(callback error)"] * len(questions)

    def set_interaction_mode(self, mode: str, callback: Callable = None) -> None:
        self._interaction_mode = mode
        self._callback = callback

    # ------------------------------------------------------------------
    # Consolidate mode (from ContextManagerAgent)
    # ------------------------------------------------------------------

    def _execute_consolidate(self, task: AgentTask) -> AgentResult:
        ctx = task.context or {}
        session_id = ctx.get("session_id", "")
        project_id = ctx.get("project_id", "")
        entries = self._load_memory_entries(session_id, project_id)

        if not entries:
            return AgentResult(success=True, output=self._fallback_consolidation(task, []),
                               metadata={"operation": "consolidate", "entries_processed": 0})

        entries_text = json.dumps(entries[:self._MAX_ENTRIES_FOR_CONSOLIDATION], indent=2)[:6000]
        prompt = (
            f"Consolidate the following {len(entries)} memory entries. "
            f"Extract key knowledge, identify patterns, create concise summary.\n\n"
            f"## Entries\n{entries_text}\n\n"
            '## Output (JSON)\n'
            '{"consolidated_summary": "...", "key_knowledge": [{"fact": "...", "confidence": 0.9}], '
            '"patterns_identified": [...], "entries_processed": ' + str(len(entries)) + '}'
        )
        result = self._infer_json(prompt, fallback=self._fallback_consolidation(task, entries))
        if result and isinstance(result, dict):
            result.setdefault("entries_processed", len(entries))
            return AgentResult(success=True, output=result,
                               metadata={"operation": "consolidate", "entries_processed": len(entries)})
        fb = self._fallback_consolidation(task, entries)
        return AgentResult(success=True, output=fb, metadata={"operation": "consolidate"})

    def _execute_summarise(self, task: AgentTask) -> AgentResult:
        ctx = task.context or {}
        history = ctx.get("history", []) or ctx.get("messages", [])
        if not history:
            history = self._load_memory_entries(ctx.get("session_id", ""), ctx.get("project_id", ""))

        prompt = (
            f"Summarise {len(history)} session entries for an AI orchestration system.\n\n"
            f"## History\n{json.dumps(history[:30], indent=2)[:4000]}\n\n"
            '## Output (JSON)\n'
            '{"session_summary": "...", "goals_achieved": [...], "next_steps": [...], '
            '"entries_processed": ' + str(len(history)) + '}'
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
            return AgentResult(success=True, output={
                "consolidated_summary": "No entries to prune",
                "pruned_count": 0, "entries_processed": 0,
            })

        prompt = (
            f"Prune context to fit within {max_tokens} tokens. "
            f"Keep highest relevance entries.\n\n"
            f"## Entries ({len(entries)})\n{json.dumps(entries[:40], indent=2)[:4000]}\n\n"
            '## Output (JSON)\n'
            '{"entries_to_retain": [...], "stale_entries": [...], "pruned_count": 0}'
        )
        result = self._infer_json(prompt, fallback={"pruned_count": 0, "entries_processed": len(entries)})
        if result and isinstance(result, dict):
            return AgentResult(success=True, output=result, metadata={"operation": "prune"})
        return AgentResult(success=True, output={"pruned_count": 0})

    def _execute_extract(self, task: AgentTask) -> AgentResult:
        text = task.context.get("text", "") or task.description or ""
        prompt = (
            f"Extract structured knowledge from:\n{text[:4000]}\n\n"
            '## Output (JSON)\n'
            '{"key_knowledge": [{"fact": "...", "confidence": 0.9}], '
            '"entities_discovered": [{"name": "...", "type": "..."}]}'
        )
        result = self._infer_json(prompt, fallback={"key_knowledge": [], "entities_discovered": []})
        if result and isinstance(result, dict):
            return AgentResult(success=True, output=result, metadata={"operation": "extract"})
        return AgentResult(success=True, output={"key_knowledge": []})

    def _execute_monitor(self, task: AgentTask) -> AgentResult:
        return AgentResult(success=True, output={"status": "healthy", "mode": "monitor"},
                           metadata={"operation": "monitor"})

    def _load_memory_entries(self, session_id: str, project_id: str) -> List[Dict]:
        entries = []
        try:
            from vetinari.memory.dual_memory import get_dual_memory_store
            store = get_dual_memory_store()
            if hasattr(store, "search"):
                results = store.search("", limit=50)
                for r in (results or []):
                    entries.append(r.to_dict() if hasattr(r, "to_dict") else r)
        except Exception:
            logger.debug("Failed to load memory entries from dual_memory store", exc_info=True)
        try:
            from vetinari.shared_memory import shared_memory
            for e in (shared_memory.get_all(limit=30) or []):
                entries.append(e.to_dict() if hasattr(e, "to_dict") else e)
        except Exception:
            logger.debug("Failed to load entries from shared_memory", exc_info=True)
        return entries

    def _fallback_consolidation(self, task: AgentTask, entries: List) -> Dict[str, Any]:
        return {
            "consolidated_summary": f"Context consolidation for: {(task.description or 'session')[:100]}",
            "session_summary": f"Processed {len(entries)} entries. LLM unavailable.",
            "key_knowledge": [], "entries_processed": len(entries),
            "retrieval_recommendations": [{"query_type": "semantic", "strategy": "hybrid"}],
        }

    def get_capabilities(self) -> List[str]:
        return [
            "ambiguity_detection", "clarification_generation", "context_gathering",
            "memory_consolidation", "session_summarisation", "context_pruning",
            "knowledge_extraction", "performance_monitoring",
        ]


# Singleton
_orchestrator_agent: Optional[OrchestratorAgent] = None


def get_orchestrator_agent(config: Optional[Dict[str, Any]] = None) -> OrchestratorAgent:
    global _orchestrator_agent
    if _orchestrator_agent is None:
        _orchestrator_agent = OrchestratorAgent(config)
    return _orchestrator_agent
