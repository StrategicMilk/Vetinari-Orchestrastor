"""
ContextManagerAgent - Long-term context management, memory consolidation, session summarisation.

Provides intelligent context management including:
- Memory consolidation across sessions
- Context window optimisation
- Session summarisation
- Relevance-based retrieval recommendations
- Knowledge graph construction from interaction history
- Context pruning to stay within token budgets
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentTask, AgentResult, AgentType, VerificationResult,
)

logger = logging.getLogger(__name__)


class ContextManagerAgent(BaseAgent):
    """Agent for managing long-term context, memory consolidation, and session summarisation."""

    # Maximum number of memory entries to include in a single LLM prompt
    _MAX_ENTRIES_FOR_CONSOLIDATION = 50

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.CONTEXT_MANAGER, config)
        self._max_context_tokens = int(
            (config or {}).get(
                "max_context_tokens",
                os.environ.get("VETINARI_MAX_CONTEXT_TOKENS", "4096")
            )
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return (
            "You are a context and memory management specialist. Your role is to:\n"
            "- Summarise long interaction histories into concise, information-dense digests\n"
            "- Identify and retain the most relevant knowledge from past sessions\n"
            "- Remove redundant, stale, or low-value context to stay within token budgets\n"
            "- Build structured knowledge representations (entities, relationships, facts)\n"
            "- Recommend retrieval strategies for future queries\n"
            "- Detect and surface contradictions or outdated information\n\n"
            "Always respond with structured JSON as specified in the task context."
        )

    def get_capabilities(self) -> List[str]:
        return [
            "memory_consolidation",
            "session_summarisation",
            "context_window_optimisation",
            "relevance_scoring",
            "knowledge_extraction",
            "context_pruning",
            "retrieval_strategy_design",
            "contradiction_detection",
            "cross_session_continuity",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        self.validate_task(task)
        self.prepare_task(task)
        try:
            result = self._manage_context(task)
            agent_result = AgentResult(
                task_id=task.task_id,
                agent_type=self._agent_type,
                success=True,
                output=result,
                metadata={
                    "operation": task.context.get("operation", "consolidate") if task.context else "consolidate",
                    "entries_processed": result.get("entries_processed", 0),
                },
            )
            self.complete_task(task, agent_result)
            return agent_result
        except Exception as exc:
            logger.error(f"[ContextManagerAgent] execute() failed: {exc}")
            return AgentResult(
                task_id=task.task_id,
                agent_type=self._agent_type,
                success=False,
                output={},
                error=str(exc),
            )

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(passed=False, score=0.0,
                                      issues=["Output must be a dict"])
        if not output.get("consolidated_summary") and not output.get("session_summary"):
            issues.append("No summary produced")
            score -= 0.4
        if not output.get("key_knowledge"):
            issues.append("No key knowledge extracted")
            score -= 0.2
        if not output.get("retrieval_recommendations"):
            score -= 0.1  # not critical
        passed = score >= 0.5
        return VerificationResult(passed=passed, score=round(score, 2), issues=issues)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _manage_context(self, task: AgentTask) -> Dict[str, Any]:
        """Route to the appropriate context management operation."""
        ctx = task.context or {}
        operation = ctx.get("operation", "consolidate").lower()

        if operation == "summarise" or operation == "summarize":
            return self._summarise_session(task)
        elif operation == "prune":
            return self._prune_context(task)
        elif operation == "extract":
            return self._extract_knowledge(task)
        else:
            return self._consolidate_memory(task)

    def _consolidate_memory(self, task: AgentTask) -> Dict[str, Any]:
        """Consolidate memory entries from the dual memory store."""
        ctx = task.context or {}
        session_id = ctx.get("session_id", "")
        project_id = ctx.get("project_id", "")

        # Retrieve recent memory entries
        memory_entries = self._load_memory_entries(session_id, project_id)

        if not memory_entries:
            return self._fallback_consolidation(task, [])

        # Build consolidation prompt
        entries_text = json.dumps(memory_entries[:self._MAX_ENTRIES_FOR_CONSOLIDATION], indent=2)

        prompt = f"""Consolidate and analyse the following memory entries from an AI orchestration session.
Extract the most important knowledge, identify patterns, and create a concise summary.

## Memory Entries ({len(memory_entries)} total, showing first {min(len(memory_entries), self._MAX_ENTRIES_FOR_CONSOLIDATION)})
{entries_text[:6000]}

## Task Context
Session ID: {session_id or 'N/A'}
Project ID: {project_id or 'N/A'}
Goal: {task.description or 'General memory consolidation'}

## Required Output (JSON)
{{
  "consolidated_summary": "Concise summary of all activity and outcomes",
  "session_summary": "What was accomplished in this session",
  "key_knowledge": [
    {{"fact": "...", "confidence": 0.9, "source": "...", "relevance": "high|medium|low"}}
  ],
  "entities_discovered": [
    {{"name": "...", "type": "...", "attributes": {{...}}}}
  ],
  "patterns_identified": ["...", ...],
  "contradictions": ["...", ...],
  "stale_entries": ["entry_id_1", ...],
  "entries_to_retain": ["entry_id_2", ...],
  "retrieval_recommendations": [
    {{"query_type": "...", "strategy": "semantic|keyword|hybrid", "index_fields": [...]}}
  ],
  "context_budget_analysis": {{
    "total_entries": {len(memory_entries)},
    "estimated_tokens": 0,
    "recommended_prune_count": 0,
    "priority_retention": ["..."]
  }},
  "next_session_context": "Key context to carry forward",
  "entries_processed": {len(memory_entries)}
}}

Return ONLY valid JSON."""

        result = self._infer_json(
            prompt, fallback=self._fallback_consolidation(task, memory_entries)
        )
        if result and isinstance(result, dict):
            result.setdefault("entries_processed", len(memory_entries))
            # Persist consolidated summary back to memory
            self._persist_consolidation(result, session_id, project_id)
            return result
        return self._fallback_consolidation(task, memory_entries)

    def _summarise_session(self, task: AgentTask) -> Dict[str, Any]:
        """Produce a concise session summary."""
        ctx = task.context or {}
        session_history = ctx.get("history", []) or ctx.get("messages", [])

        if not session_history:
            session_history = self._load_memory_entries(
                ctx.get("session_id", ""), ctx.get("project_id", "")
            )

        history_text = json.dumps(session_history[:30], indent=2)

        prompt = f"""Summarise the following session history for an AI orchestration system.
Create a concise summary suitable for use as context in a future session.

## Session History
{history_text[:4000]}

## Required Output (JSON)
{{
  "session_summary": "2-3 paragraph summary of what happened",
  "consolidated_summary": "1-paragraph executive summary",
  "goals_achieved": ["...", ...],
  "goals_pending": ["...", ...],
  "key_knowledge": [
    {{"fact": "...", "confidence": 0.9, "source": "...", "relevance": "high|medium|low"}}
  ],
  "decisions_made": ["...", ...],
  "artifacts_produced": ["...", ...],
  "next_steps": ["...", ...],
  "retrieval_recommendations": [],
  "entries_processed": {len(session_history)}
}}

Return ONLY valid JSON."""

        result = self._infer_json(
            prompt, fallback=self._fallback_consolidation(task, session_history)
        )
        if result and isinstance(result, dict):
            result.setdefault("entries_processed", len(session_history))
            return result
        return self._fallback_consolidation(task, session_history)

    def _prune_context(self, task: AgentTask) -> Dict[str, Any]:
        """Identify and remove low-value context entries."""
        ctx = task.context or {}
        entries = ctx.get("entries", [])
        max_tokens = ctx.get("max_tokens", self._max_context_tokens)

        if not entries:
            return {
                "consolidated_summary": "No entries to prune",
                "session_summary": "Context pruning skipped - no entries provided",
                "key_knowledge": [],
                "stale_entries": [],
                "entries_to_retain": [],
                "pruned_count": 0,
                "entries_processed": 0,
                "retrieval_recommendations": [],
            }

        prompt = f"""You are pruning a context store to fit within {max_tokens} tokens.
Identify which entries to keep (highest relevance, most recent, highest information density)
and which to discard (stale, redundant, low value).

## Entries ({len(entries)})
{json.dumps(entries[:40], indent=2)[:4000]}

## Required Output (JSON)
{{
  "consolidated_summary": "Summary of retained content",
  "session_summary": "What was retained and why",
  "entries_to_retain": ["id_1", "id_2", ...],
  "stale_entries": ["id_3", ...],
  "pruning_rationale": "...",
  "estimated_tokens_retained": 0,
  "key_knowledge": [],
  "retrieval_recommendations": [],
  "pruned_count": 0,
  "entries_processed": {len(entries)}
}}

Return ONLY valid JSON."""

        result = self._infer_json(
            prompt, fallback=self._fallback_consolidation(task, entries)
        )
        if result and isinstance(result, dict):
            result.setdefault("entries_processed", len(entries))
            return result
        return self._fallback_consolidation(task, entries)

    def _extract_knowledge(self, task: AgentTask) -> Dict[str, Any]:
        """Extract structured knowledge from unstructured context."""
        ctx = task.context or {}
        text = ctx.get("text", "") or task.description or ""

        prompt = f"""Extract structured knowledge from the following text for an AI orchestration system.

## Text
{text[:4000]}

## Required Output (JSON)
{{
  "consolidated_summary": "Summary of extracted knowledge",
  "session_summary": "What knowledge was extracted",
  "key_knowledge": [
    {{"fact": "...", "confidence": 0.9, "source": "...", "relevance": "high|medium|low"}}
  ],
  "entities_discovered": [
    {{"name": "...", "type": "person|system|concept|tool|file|...", "attributes": {{...}}}}
  ],
  "relationships": [
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ],
  "retrieval_recommendations": [
    {{"query_type": "...", "strategy": "semantic|keyword|hybrid", "index_fields": [...]}}
  ],
  "entries_processed": 1
}}

Return ONLY valid JSON."""

        result = self._infer_json(
            prompt, fallback=self._fallback_consolidation(task, [])
        )
        if result and isinstance(result, dict):
            return result
        return self._fallback_consolidation(task, [])

    def _load_memory_entries(self, session_id: str, project_id: str) -> List[Dict]:
        """Load memory entries from the memory subsystem."""
        entries = []
        try:
            from vetinari.memory.dual_memory import get_dual_memory_store
            store = get_dual_memory_store()
            # Try to get recent entries
            if hasattr(store, "search"):
                results = store.search("", limit=50)
                for r in (results or []):
                    if hasattr(r, "to_dict"):
                        entries.append(r.to_dict())
                    elif isinstance(r, dict):
                        entries.append(r)
        except Exception as e:
            logger.debug(f"[ContextManager] Could not load memory entries: {e}")

        # Also try shared memory
        try:
            from vetinari.shared_memory import shared_memory
            sm_entries = shared_memory.get_all(limit=30)
            for e in (sm_entries or []):
                if hasattr(e, "to_dict"):
                    entries.append(e.to_dict())
                elif isinstance(e, dict):
                    entries.append(e)
        except Exception:
            pass

        return entries

    def _persist_consolidation(
        self, result: Dict, session_id: str, project_id: str
    ) -> None:
        """Persist the consolidated summary back to memory."""
        try:
            from vetinari.memory.dual_memory import get_dual_memory_store
            store = get_dual_memory_store()
            if hasattr(store, "remember"):
                store.remember(
                    content=result.get("consolidated_summary", ""),
                    agent_name="context_manager",
                    memory_type="consolidation",
                    tags=["consolidated", "session_summary"],
                )
        except Exception as e:
            logger.debug(f"[ContextManager] Could not persist consolidation: {e}")

    def _fallback_consolidation(
        self, task: AgentTask, entries: List
    ) -> Dict[str, Any]:
        """Structured fallback when LLM is unavailable."""
        return {
            "consolidated_summary": (
                f"Context consolidation for: {task.description[:100] if task.description else 'session'}"
            ),
            "session_summary": (
                f"Processed {len(entries)} memory entries. "
                "LLM consolidation unavailable — manual review recommended."
            ),
            "key_knowledge": [],
            "entities_discovered": [],
            "patterns_identified": [
                "Context consolidation is running periodically",
                "Memory entries are being tracked",
            ],
            "contradictions": [],
            "stale_entries": [],
            "entries_to_retain": [e.get("id", str(i)) for i, e in enumerate(entries[:20])],
            "retrieval_recommendations": [
                {
                    "query_type": "semantic",
                    "strategy": "hybrid",
                    "index_fields": ["content", "summary", "tags"],
                }
            ],
            "context_budget_analysis": {
                "total_entries": len(entries),
                "estimated_tokens": len(entries) * 100,
                "recommended_prune_count": max(0, len(entries) - 20),
                "priority_retention": ["recent", "high_confidence", "decisions"],
            },
            "next_session_context": f"Continue from: {task.description[:80] if task.description else 'previous session'}",
            "entries_processed": len(entries),
        }


# Need os import at top
import os


# Singleton
_context_manager_agent: Optional[ContextManagerAgent] = None


def get_context_manager_agent(
    config: Optional[Dict[str, Any]] = None
) -> ContextManagerAgent:
    """Get the singleton ContextManagerAgent instance."""
    global _context_manager_agent
    if _context_manager_agent is None:
        _context_manager_agent = ContextManagerAgent(config)
    return _context_manager_agent
