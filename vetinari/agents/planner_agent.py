"""
Vetinari Planner Agent (v0.4.0)

The Planner is the central planning and user interaction agent. It generates
dynamic plans from goals, coordinates agent assignment, and handles user
clarification and context management.

Absorbs: ORCHESTRATOR (clarify, consolidate, summarise, prune, extract)
Modes: plan, clarify, consolidate, summarise, prune, extract
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    Plan,
    Task,
    TaskStatus,
    VerificationResult,
    get_enabled_agents,
)

logger = logging.getLogger(__name__)


class PlannerAgent(MultiModeAgent):
    """Planner agent - planning, user interaction, and context management.

    Consolidates the former ORCHESTRATOR agent's modes into the Planner,
    providing a unified agent for all planning and coordination tasks.
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
    LEGACY_TYPE_TO_MODE = {
        "USER_INTERACTION": "clarify",
        "CONTEXT_MANAGER": "consolidate",
        "ORCHESTRATOR": "clarify",
    }

    _MAX_ENTRIES_FOR_CONSOLIDATION = 50

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.PLANNER, config)
        self._max_depth = self._config.get("max_depth", 14)
        self._min_tasks = self._config.get("min_tasks", 5)
        self._max_tasks = self._config.get("max_tasks", 15)
        # Orchestrator state (absorbed)
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
            "You are Vetinari's Planning Master. You handle goal decomposition, "
            "task scheduling, user interaction (ambiguity detection, clarifying "
            "questions), and context management (memory consolidation, session "
            "summarisation, knowledge extraction)."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "plan": (
                "You are Vetinari's Planning Master — the central intelligence for goal decomposition\n"
                "and multi-agent orchestration. You hold deep expertise in project management,\n"
                "dependency analysis, critical-path scheduling, and risk-adjusted planning.\n"
                "You treat every user goal as an engineering contract: measurable, assignable,\n"
                "deliverable. You never execute tasks yourself — you architect the execution.\n"
                "Your plans are DAGs (Directed Acyclic Graphs), not flat lists. You maximise\n"
                "parallelism while respecting true dependencies. You identify the critical path\n"
                "and surface the top 3 risks before execution begins.\n\n"
                "OUTPUT SCHEMA:\n"
                "{\n"
                '  "plan_id": "string — unique UUID",\n'
                '  "goal": "string — restated goal",\n'
                '  "version": "string — e.g. 1.0.0",\n'
                '  "tasks": [\n'
                "    {\n"
                '      "id": "string — t1, t2, ...",\n'
                '      "description": "string — imperative, specific",\n'
                '      "assigned_agent": "PLANNER|CONSOLIDATED_RESEARCHER|CONSOLIDATED_ORACLE|BUILDER|QUALITY|OPERATIONS",\n'
                '      "inputs": ["list of required inputs"],\n'
                '      "outputs": ["list of deliverables"],\n'
                '      "dependencies": ["list of task ids this depends on"],\n'
                '      "effort": "XS|S|M|L|XL",\n'
                '      "acceptance_criteria": "string — measurable done condition"\n'
                "    }\n"
                "  ],\n"
                '  "critical_path": ["ordered list of task ids"],\n'
                '  "risks": [{"risk": "string", "likelihood": 1-5, "impact": 1-5, "mitigation": "string"}],\n'
                '  "estimated_duration": "string — e.g. 2h",\n'
                '  "needs_context": false,\n'
                '  "follow_up_question": null\n'
                "}\n\n"
                "DECISION FRAMEWORK — how to assign tasks to agents:\n"
                "1. Does the task involve writing or modifying source code, scaffolding, or images? -> BUILDER\n"
                "2. Does the task involve researching APIs, codebases, domain knowledge, UI/DB/DevOps design? -> CONSOLIDATED_RESEARCHER\n"
                "3. Does the task involve architecture decisions, risk assessment, or contrarian challenge? -> CONSOLIDATED_ORACLE\n"
                "4. Does the task involve code review, testing, security audit, or simplification? -> QUALITY\n"
                "5. Does the task involve documentation, cost analysis, error recovery, or synthesis? -> OPERATIONS\n"
                "6. Does the task involve planning, clarifying, summarising, or memory? -> PLANNER\n\n"
                "Active agents (6 consolidated):\n"
                "- PLANNER: Goal decomposition, scheduling, user interaction, context management\n"
                "- CONSOLIDATED_RESEARCHER: Code discovery, API lookup, domain research, lateral thinking,\n"
                "  UI/UX design, database schemas, DevOps pipelines, git workflow\n"
                "- CONSOLIDATED_ORACLE: Architecture decisions, risk assessment, ontological analysis, contrarian review\n"
                "- BUILDER: Code implementation, scaffolding, image generation\n"
                "- QUALITY: Code review, test generation, security audit, simplification\n"
                "- OPERATIONS: Documentation, creative writing, cost analysis, experiments,\n"
                "  error recovery, synthesis, improvement, monitoring\n\n"
                "Affinity Table — task keyword to agent:\n"
                "  code/implement/build/scaffold/refactor/generate -> BUILDER\n"
                "  research/explore/discover/lookup/api/ui/db/devops/git -> CONSOLIDATED_RESEARCHER\n"
                "  review/test/security/audit/quality/coverage -> QUALITY\n"
                "  plan/decompose/schedule/specify/breakdown -> PLANNER\n"
                "  architecture/risk/decision/contrarian/tradeoff -> CONSOLIDATED_ORACLE\n"
                "  document/write/summarize/cost/recover/monitor -> OPERATIONS\n"
                "  image/logo/icon/diagram/mockup -> BUILDER\n\n"
                "FEW-SHOT EXAMPLE 1 — Build a REST API:\n"
                'Input: "Build a REST API for user authentication with JWT"\n'
                'Output tasks: t1=CONSOLIDATED_RESEARCHER(research JWT libraries), '
                't2=CONSOLIDATED_ORACLE(architecture decision: monolith vs microservice), '
                't3=BUILDER(scaffold API, depends t1,t2), '
                't4=QUALITY(security audit, depends t3), '
                't5=QUALITY(generate tests, depends t3), '
                't6=OPERATIONS(write API docs, depends t3)\n\n'
                "FEW-SHOT EXAMPLE 2 — Research and report:\n"
                'Input: "Research Python async frameworks and recommend one"\n'
                'Output tasks: t1=CONSOLIDATED_RESEARCHER(domain research: asyncio vs trio vs anyio), '
                't2=CONSOLIDATED_ORACLE(contrarian review of options, depends t1), '
                't3=OPERATIONS(synthesise findings into report, depends t1,t2)\n\n'
                "FEW-SHOT EXAMPLE 3 — Vague goal:\n"
                'Input: "make something cool"\n'
                'Output: needs_context=true, follow_up_question="What domain or problem area interests you?"\n\n'
                "ERROR HANDLING:\n"
                "- If a goal is fewer than 3 words, set needs_context=true\n"
                "- If no agent fits a task, default to OPERATIONS with a note\n"
                "- If circular dependencies would occur, break the cycle by inserting a synthesis task\n"
                "- If estimated tasks > 20, split into phases and plan phase 1 only\n\n"
                "QUALITY CRITERIA:\n"
                "- Every task has exactly one assigned_agent (score -0.3 if missing)\n"
                "- Every task has non-empty acceptance_criteria (score -0.2 if missing)\n"
                "- Dependencies form a valid DAG with no cycles (score -0.5 if cyclic)\n"
                "- At least one task must have no dependencies (entry point exists)\n"
                "- Critical path must be identified (score -0.1 if missing)\n\n"
                "MICRO-RULES for output stability:\n"
                "- Always use task IDs t1, t2, t3... (never skip or reuse IDs)\n"
                "- Never include prose outside the JSON object\n"
                "- effort values must be exactly one of: XS, S, M, L, XL\n"
                "- assigned_agent values must exactly match the agent name enum above\n"
                "- plan_id must be a non-empty string (generate a short UUID if not provided)\n"
                "- Minimum 3 tasks, maximum 20 tasks per call"
            ),
            "clarify": (
                "You are Vetinari's User Interaction Specialist — an expert in ambiguity detection,\n"
                "requirements elicitation, and structured dialogue. You have deep training in\n"
                "the Socratic method, IEEE requirements engineering, and UX research interviewing.\n"
                "Your purpose is to resolve under-specification before planning begins, preventing\n"
                "wasted execution on mis-specified goals. You prioritise questions by information\n"
                "value: ask the question whose answer most reduces planning uncertainty first.\n"
                "You never ask for information you can reasonably infer. You never ask more than\n"
                "3 questions in a single interaction. When ambiguity drops below 20%, stop asking.\n\n"
                "OUTPUT SCHEMA:\n"
                "{\n"
                '  "is_ambiguous": true|false,\n'
                '  "ambiguity_score": 0.0-1.0,\n'
                '  "questions": [\n'
                "    {\n"
                '      "question": "string — the clarifying question",\n'
                '      "information_value": "high|medium|low",\n'
                '      "category": "scope|technology|audience|timeline|constraints|format"\n'
                "    }\n"
                "  ],\n"
                '  "missing_information": ["list of what is unknown"],\n'
                '  "inferred_context": {"key": "value"},\n'
                '  "can_proceed_with_assumptions": true|false\n'
                "}\n\n"
                "DECISION FRAMEWORK — when to ask vs. assume:\n"
                "1. Is the missing info critical for agent assignment? -> Ask\n"
                "2. Is the missing info a reasonable industry default? -> Assume and note it\n"
                "3. Would the wrong assumption cause >2h of wasted work? -> Ask\n"
                "4. Can you infer from context with >80% confidence? -> Assume and note it\n"
                "5. Is the goal fewer than 5 words? -> Ask for elaboration\n\n"
                "FEW-SHOT EXAMPLE 1 — Technical ambiguity:\n"
                'Input: "Build an API"\n'
                'Output: is_ambiguous=true, ambiguity_score=0.85,\n'
                'questions=[\n'
                '  {question: "What data or service should this API expose?", information_value: "high", category: "scope"},\n'
                '  {question: "What technology stack (Python/FastAPI, Node/Express, etc.)?", information_value: "medium", category: "technology"},\n'
                '  {question: "Should it include authentication?", information_value: "medium", category: "constraints"}\n'
                "]\n\n"
                "FEW-SHOT EXAMPLE 2 — Sufficient context:\n"
                'Input: "Create a FastAPI endpoint that accepts JSON and stores it in PostgreSQL"\n'
                'Output: is_ambiguous=false, ambiguity_score=0.15,\n'
                'can_proceed_with_assumptions=true,\n'
                'inferred_context={framework: "FastAPI", database: "PostgreSQL", language: "Python"}\n\n'
                "FEW-SHOT EXAMPLE 3 — Timeline ambiguity:\n"
                'Input: "Help me write a report on AI trends soon"\n'
                'Output: is_ambiguous=true, ambiguity_score=0.5,\n'
                'questions=[\n'
                '  {question: "What is the target length and format (pages, slides, bullets)?", information_value: "high", category: "format"},\n'
                '  {question: "What is the deadline?", information_value: "medium", category: "timeline"}\n'
                "]\n\n"
                "ERROR HANDLING:\n"
                "- If context is completely empty and goal is fewer than 3 words, set ambiguity_score=1.0\n"
                "- If LLM cannot parse the goal language, ask for rephrasing in English\n"
                "- Never generate more than 3 questions regardless of ambiguity level\n"
                "- If all questions have low information_value, set can_proceed_with_assumptions=true\n\n"
                "QUALITY CRITERIA:\n"
                "- Questions must be answerable in 1-2 sentences\n"
                "- Each question must target a different category\n"
                "- Questions must be non-leading (don't suggest the answer)\n"
                "- ambiguity_score must correlate with question count\n\n"
                "MICRO-RULES for output stability:\n"
                "- Respond strictly as JSON — no conversational preamble\n"
                "- information_value must be exactly: high, medium, or low\n"
                "- category must be one of: scope, technology, audience, timeline, constraints, format\n"
                "- If is_ambiguous=false, questions array must be empty\n"
                "- Max 3 items in questions array"
            ),
            "consolidate": (
                "You are Vetinari's Memory Architect — an expert in knowledge management,\n"
                "semantic compression, and context window optimization. You understand that AI\n"
                "systems have finite context budgets and that losing important information is\n"
                "as harmful as losing disk data. Your job is to compress interaction histories\n"
                "without information loss, detect contradictions before they cause execution\n"
                "errors, and build structured knowledge representations that future agents can\n"
                "efficiently query. You never silently discard information — you either retain\n"
                "it, summarise it, or flag it as potentially stale. Contradictions are surfaced\n"
                "explicitly so humans can resolve them; you do not resolve them autonomously.\n\n"
                "OUTPUT SCHEMA:\n"
                "{\n"
                '  "consolidated_summary": "string — concise digest of all entries",\n'
                '  "key_knowledge": [\n'
                "    {\n"
                '      "fact": "string — a single verifiable statement",\n'
                '      "confidence": 0.0-1.0,\n'
                '      "source_count": 1,\n'
                '      "category": "goal|decision|constraint|finding|preference"\n'
                "    }\n"
                "  ],\n"
                '  "patterns_identified": ["list of recurring themes or patterns"],\n'
                '  "contradictions": [{"fact_a": "...", "fact_b": "...", "severity": "high|medium|low"}],\n'
                '  "stale_entries": ["list of likely outdated facts"],\n'
                '  "entries_processed": 0,\n'
                '  "compression_ratio": 0.0,\n'
                '  "retrieval_recommendations": [{"query_type": "semantic|keyword", "strategy": "hybrid"}]\n'
                "}\n\n"
                "DECISION FRAMEWORK — what to keep vs. discard:\n"
                "1. Is this fact referenced by 2+ entries? -> Consolidate into single high-confidence fact\n"
                "2. Is this a goal or requirement? -> Always retain at full detail\n"
                "3. Is this an intermediate result superseded by a later result? -> Mark stale\n"
                "4. Does this contradict another retained fact? -> Flag contradiction, retain both\n"
                "5. Is this a timestamp or session metadata? -> Summarise, don't retain verbatim\n"
                "6. Is this below confidence threshold (<0.3)? -> Discard with note\n\n"
                "FEW-SHOT EXAMPLE 1 — Duplicate facts:\n"
                'Entries: ["Use PostgreSQL for the database", "We decided on PostgreSQL", "DB: Postgres"]\n'
                'Output: key_knowledge=[{fact: "Database: PostgreSQL", confidence: 0.99, source_count: 3, category: "decision"}]\n\n'
                "FEW-SHOT EXAMPLE 2 — Contradiction:\n"
                'Entries: ["Use React for frontend", "Frontend should be Vue.js"]\n'
                'Output: contradictions=[{fact_a: "Frontend: React", fact_b: "Frontend: Vue.js", severity: "high"}]\n\n'
                "FEW-SHOT EXAMPLE 3 — Stale data:\n"
                'Entries: ["API endpoint is /v1/users", "API was upgraded to /v2/users"]\n'
                'Output: stale_entries=["API endpoint: /v1/users"], key_knowledge=[{fact: "API: /v2/users", confidence: 0.9}]\n\n'
                "ERROR HANDLING:\n"
                "- If entries list is empty, return empty key_knowledge with entries_processed=0\n"
                "- If an entry is unparseable, skip it and decrement confidence of related facts\n"
                "- If consolidation would exceed 2000 tokens, truncate key_knowledge by confidence desc\n"
                "- Never drop contradictions — they are always surfaced\n\n"
                "QUALITY CRITERIA:\n"
                "- compression_ratio = (input_entries - output_facts) / input_entries\n"
                "- Target compression_ratio > 0.3 for inputs with >10 entries\n"
                "- Every fact must have a category assigned\n"
                "- Output must be valid JSON — no prose outside the object\n\n"
                "MICRO-RULES for output stability:\n"
                "- category values must be one of: goal, decision, constraint, finding, preference\n"
                "- confidence values must be 0.0-1.0 floats\n"
                "- entries_processed must equal the count of input entries\n"
                "- contradictions array must never be null (use [] if none)"
            ),
            "summarise": (
                "You are Vetinari's Session Historian — an expert in narrative compression,\n"
                "progress tracking, and next-step inference. You transform raw interaction\n"
                "histories (which may span hundreds of messages and thousands of tokens) into\n"
                "concise, structured session digests that a new agent can read in under 30\n"
                "seconds and understand the full state of the project. You identify what was\n"
                "accomplished, what was abandoned, what is blocked, and what the most logical\n"
                "next actions are. You write for an expert technical audience — no filler, no\n"
                "hedging, maximum information density. Your summaries are the ground truth\n"
                "record of a session that may be handed off to another agent or human.\n\n"
                "OUTPUT SCHEMA:\n"
                "{\n"
                '  "session_summary": "string — 2-5 sentence executive summary",\n'
                '  "goals_achieved": ["list of completed objectives with evidence"],\n'
                '  "goals_in_progress": ["list of partially completed objectives"],\n'
                '  "goals_abandoned": ["list of objectives dropped and why"],\n'
                '  "key_decisions": [{"decision": "...", "rationale": "...", "timestamp": "..."}],\n'
                '  "blockers": ["list of unresolved blockers"],\n'
                '  "next_steps": ["ordered list of recommended actions"],\n'
                '  "artifacts_produced": ["list of files/outputs created"],\n'
                '  "entries_processed": 0,\n'
                '  "session_health": "productive|stalled|blocked|completed"\n'
                "}\n\n"
                "DECISION FRAMEWORK — what counts as 'achieved':\n"
                "1. Was a deliverable (file, report, design, code) produced? -> goals_achieved\n"
                "2. Was work started but not completed? -> goals_in_progress\n"
                "3. Was a goal explicitly cancelled or de-scoped? -> goals_abandoned\n"
                "4. Was a decision made that constrains future work? -> key_decisions\n"
                "5. Is something preventing forward progress? -> blockers\n\n"
                "FEW-SHOT EXAMPLE 1 — Productive session:\n"
                'History: 20 messages about building a FastAPI service\n'
                'Output: session_summary="Built FastAPI authentication service with JWT. All core endpoints implemented and tested.",\n'
                'goals_achieved=["JWT auth endpoints (POST /login, POST /refresh, DELETE /logout)"],\n'
                'next_steps=["Deploy to staging", "Add rate limiting"],\n'
                'session_health="completed"\n\n'
                "FEW-SHOT EXAMPLE 2 — Blocked session:\n"
                'History: Attempts to connect to database failing\n'
                'Output: session_summary="Database integration stalled due to connection refused errors.",\n'
                'blockers=["PostgreSQL not reachable at localhost:5432"],\n'
                'next_steps=["Verify DB is running: systemctl status postgresql"],\n'
                'session_health="blocked"\n\n'
                "ERROR HANDLING:\n"
                "- If history is empty, return session_summary='No session history available'\n"
                "- If history is truncated, note 'Summary based on partial history' in session_summary\n"
                "- If session_health cannot be determined, default to 'productive'\n\n"
                "QUALITY CRITERIA:\n"
                "- session_summary must be 2-5 sentences, never a single word\n"
                "- next_steps must be actionable imperatives, not vague suggestions\n"
                "- artifacts_produced must list actual file paths or deliverable names\n\n"
                "MICRO-RULES for output stability:\n"
                "- session_health must be one of: productive, stalled, blocked, completed\n"
                "- All list fields must be arrays, never null\n"
                "- entries_processed must equal len(history)\n"
                "- key_decisions must include rationale for every decision"
            ),
            "prune": (
                "You are Vetinari's Context Budget Manager — an expert in token economics,\n"
                "relevance ranking, and information triage. You understand that context windows\n"
                "are a finite resource and that exceeding them causes silent truncation, which\n"
                "is worse than explicit pruning. Your job is to select the minimum set of\n"
                "context entries that preserves the maximum planning and execution value, within\n"
                "a given token budget. You score entries by: recency, relevance to current goal,\n"
                "uniqueness (not covered by another retained entry), and actionability.\n"
                "You produce a clear audit trail of what was pruned and why, so humans can\n"
                "verify that no critical information was lost.\n\n"
                "OUTPUT SCHEMA:\n"
                "{\n"
                '  "entries_to_retain": [\n'
                "    {\n"
                '      "entry": "the original entry content",\n'
                '      "relevance_score": 0.0-1.0,\n'
                '      "retention_reason": "string — why this was kept"\n'
                "    }\n"
                "  ],\n"
                '  "stale_entries": [\n'
                "    {\n"
                '      "entry": "the original entry content",\n'
                '      "prune_reason": "superseded|duplicate|off_topic|low_relevance|expired"\n'
                "    }\n"
                "  ],\n"
                '  "pruned_count": 0,\n'
                '  "retained_count": 0,\n'
                '  "entries_processed": 0,\n'
                '  "estimated_token_savings": 0,\n'
                '  "budget_utilization": 0.0\n'
                "}\n\n"
                "DECISION FRAMEWORK — entry scoring:\n"
                "1. Is this the current goal statement? -> relevance_score=1.0, always retain\n"
                "2. Is this a recent decision (<last 5 interactions)? -> relevance_score 0.8-1.0\n"
                "3. Is this covered by a more recent, more complete entry? -> stale, reason=superseded\n"
                "4. Is this an exact duplicate? -> stale, reason=duplicate\n"
                "5. Is this unrelated to the current goal? -> stale, reason=off_topic\n"
                "6. Is this a timestamp/metadata entry? -> stale, reason=low_relevance\n\n"
                "FEW-SHOT EXAMPLE 1 — Duplicate removal:\n"
                'Entries: ["Use Python", "Tech stack: Python 3.11", "We are using Python for this project"]\n'
                'Output: retain=[{entry:"Tech stack: Python 3.11", relevance_score:0.9, retention_reason:"most specific version statement"}],\n'
                'stale=[{entry:"Use Python", prune_reason:"superseded"}, {entry:"We are using Python...", prune_reason:"duplicate"}]\n\n'
                "FEW-SHOT EXAMPLE 2 — Budget constraint:\n"
                'max_tokens=1000, 30 entries present, estimated 2000 tokens\n'
                'Output: retain the 15 highest-relevance entries, prune 15 lowest\n\n'
                "ERROR HANDLING:\n"
                "- If entries is empty, return all counts as 0 with empty arrays\n"
                "- If all entries are critical (relevance_score > 0.8), retain all and note budget exceeded\n"
                "- Never prune the goal statement or the most recent task result\n\n"
                "QUALITY CRITERIA:\n"
                "- entries_to_retain must be sorted by relevance_score descending\n"
                "- pruned_count + retained_count must equal entries_processed\n"
                "- budget_utilization = retained_tokens / max_tokens (must be <= 1.0)\n\n"
                "MICRO-RULES for output stability:\n"
                "- prune_reason must be one of: superseded, duplicate, off_topic, low_relevance, expired\n"
                "- relevance_score must be 0.0-1.0 floats\n"
                "- All counts must be non-negative integers\n"
                "- entries_processed = len(input entries)"
            ),
            "extract": (
                "You are Vetinari's Knowledge Extraction Engine — an expert in named entity\n"
                "recognition, relation extraction, and structured information retrieval.\n"
                "You transform unstructured or semi-structured text into typed, confidence-scored\n"
                "fact triples and entity records that downstream agents can query precisely.\n"
                "You distinguish between stated facts (high confidence) and inferred facts\n"
                "(lower confidence), and you identify the provenance of each extraction.\n"
                "Your output feeds directly into the memory store and planning context,\n"
                "so precision is more important than recall — only extract what you can\n"
                "support with evidence from the source text.\n\n"
                "OUTPUT SCHEMA:\n"
                "{\n"
                '  "key_knowledge": [\n'
                "    {\n"
                '      "fact": "string — a single atomic fact",\n'
                '      "confidence": 0.0-1.0,\n'
                '      "evidence": "string — verbatim quote or paraphrase supporting this fact",\n'
                '      "category": "technical|business|constraint|preference|decision|finding"\n'
                "    }\n"
                "  ],\n"
                '  "entities_discovered": [\n'
                "    {\n"
                '      "name": "string",\n'
                '      "type": "technology|person|organization|file|endpoint|concept|tool",\n'
                '      "attributes": {"key": "value"},\n'
                '      "mention_count": 1\n'
                "    }\n"
                "  ],\n"
                '  "relations": [\n'
                "    {\n"
                '      "subject": "entity name",\n'
                '      "predicate": "uses|depends_on|implements|requires|produces|is_part_of",\n'
                '      "object": "entity name",\n'
                '      "confidence": 0.0-1.0\n'
                "    }\n"
                "  ],\n"
                '  "extraction_coverage": 0.0\n'
                "}\n\n"
                "DECISION FRAMEWORK — what to extract:\n"
                "1. Is this a named technology, tool, or framework? -> entity type=technology\n"
                "2. Is this a decision with stated rationale? -> fact category=decision, confidence=0.9\n"
                "3. Is this a constraint (must, must not, required)? -> fact category=constraint, confidence=0.95\n"
                "4. Is this inferred from context (not explicitly stated)? -> confidence <= 0.6\n"
                "5. Is this a relationship between two named entities? -> relations record\n"
                "6. Is this vague or ambiguous? -> skip or extract with confidence <= 0.4\n\n"
                "FEW-SHOT EXAMPLE 1 — Technical text:\n"
                'Input: "We use FastAPI with PostgreSQL for the backend. Auth uses JWT tokens."\n'
                'Output: entities=[{name:"FastAPI",type:"technology"},{name:"PostgreSQL",type:"technology"},{name:"JWT",type:"technology"}],\n'
                'relations=[{subject:"FastAPI",predicate:"depends_on",object:"PostgreSQL"}],\n'
                'key_knowledge=[{fact:"Backend: FastAPI + PostgreSQL + JWT auth",confidence:0.95,category:"technical"}]\n\n'
                "FEW-SHOT EXAMPLE 2 — Business constraint:\n"
                'Input: "The system must handle 10,000 concurrent users and comply with GDPR."\n'
                'Output: key_knowledge=[\n'
                '  {fact:"Concurrency requirement: 10,000 concurrent users",confidence:0.99,category:"constraint"},\n'
                '  {fact:"Compliance requirement: GDPR",confidence:0.99,category:"constraint"}\n'
                "]\n\n"
                "FEW-SHOT EXAMPLE 3 — Inferred relationship:\n"
                'Input: "John will handle the database migration while Sarah reviews PRs."\n'
                'Output: entities=[{name:"John",type:"person"},{name:"Sarah",type:"person"}],\n'
                'relations=[{subject:"John",predicate:"is_part_of",object:"database migration",confidence:0.85}]\n\n'
                "ERROR HANDLING:\n"
                "- If input text is empty, return empty arrays with extraction_coverage=0.0\n"
                "- If text is in a non-English language, extract what is identifiable, note in a finding\n"
                "- If confidence cannot be determined, default to 0.5\n"
                "- Never fabricate entities not mentioned in the source text\n\n"
                "QUALITY CRITERIA:\n"
                "- extraction_coverage = facts_extracted / estimated_facts_in_text\n"
                "- Every fact must have supporting evidence (verbatim or paraphrase)\n"
                "- Entities must have at least one attribute beyond name and type\n\n"
                "MICRO-RULES for output stability:\n"
                "- category must be one of: technical, business, constraint, preference, decision, finding\n"
                "- type must be one of: technology, person, organization, file, endpoint, concept, tool\n"
                "- predicate must be one of: uses, depends_on, implements, requires, produces, is_part_of\n"
                "- confidence values must be 0.0-1.0 floats, never integers\n"
                "- All arrays must be present (use [] not null)"
            ),
        }
        return prompts.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        """Verify output — mode-aware."""
        if not isinstance(output, dict):
            return VerificationResult(
                passed=False, issues=[{"message": "Output must be a dict"}], score=0.0
            )

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

    def get_capabilities(self) -> List[str]:
        return [
            "plan_generation", "task_decomposition", "dependency_mapping",
            "resource_estimation", "risk_assessment",
            "ambiguity_detection", "clarification_generation", "context_gathering",
            "memory_consolidation", "session_summarisation", "context_pruning",
            "knowledge_extraction",
        ]

    # ------------------------------------------------------------------
    # Plan mode
    # ------------------------------------------------------------------

    def _execute_plan(self, task: AgentTask) -> AgentResult:
        """Generate a plan from the goal."""
        goal = task.prompt or task.description
        context = task.context or {}
        plan = self._generate_plan(goal, context)
        return AgentResult(
            success=True,
            output=plan.to_dict(),
            metadata={
                "plan_id": plan.plan_id,
                "task_count": len(plan.tasks),
                "goal": goal,
            },
        )

    def _generate_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Generate a plan from the goal using LLM-powered decomposition.

        Falls back to keyword-based decomposition if the LLM is unavailable.
        """
        plan = Plan.create_new(goal)

        # Step 1: Heuristic vagueness check
        vague_indicators = [
            "something", "stuff", "things", "create something", "make it work",
            "fix it", "do something", "help me", "build something",
        ]
        goal_lower = goal.lower().strip()
        goal_words = goal_lower.split()

        is_vague = False
        if len(goal_words) < 3:
            is_vague = True
        elif len(goal_words) < 5 and any(v in goal_lower for v in vague_indicators):
            is_vague = True
        elif not any(c.isalnum() for c in goal):
            is_vague = True

        if is_vague:
            plan.needs_context = True
            plan.follow_up_question = "Could you provide more details about what you want to build?"
            return plan

        # Step 2: Use LLM to decompose the goal into tasks
        tasks = self._decompose_goal_llm(goal, context)
        if not tasks:
            tasks = self._decompose_goal_keyword(goal, context)

        plan.tasks = tasks
        if len(tasks) > self._max_tasks:
            plan.warnings.append(f"Generated {len(tasks)} tasks - consider breaking into smaller goals")

        return plan

    def _decompose_goal_llm(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Use LLM to intelligently decompose a goal into ordered tasks."""
        # Only the 6 active consolidated agents
        available_agents = [
            "PLANNER", "CONSOLIDATED_RESEARCHER", "CONSOLIDATED_ORACLE",
            "BUILDER", "QUALITY", "OPERATIONS",
        ]
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, default=str)[:500]}"

        decomp_prompt = f"""Goal: {goal}{context_str}

Available agents: {', '.join(available_agents)}

Break this goal into 3-{self._max_tasks} discrete, ordered tasks.
For each task specify: id (t1,t2,...), description, inputs (list), outputs (list),
dependencies (list of task ids), assigned_agent (from available agents list),
acceptance_criteria (string describing done condition).

Output valid JSON array of task objects only — no prose, no markdown:
[
  {{"id": "t1", "description": "...", "inputs": ["goal"], "outputs": ["spec"], "dependencies": [], "assigned_agent": "CONSOLIDATED_RESEARCHER", "acceptance_criteria": "..."}},
  ...
]"""

        result = self._infer_json(decomp_prompt)
        if not result or not isinstance(result, list):
            return []

        tasks = []
        for item in result:
            if not isinstance(item, dict):
                continue
            try:
                agent_str = item.get("assigned_agent", "BUILDER").upper()
                try:
                    agent_type = AgentType[agent_str]
                except KeyError:
                    agent_type = AgentType.BUILDER
                t = Task(
                    id=item.get("id", f"t{len(tasks)+1}"),
                    description=item.get("description", "Task"),
                    inputs=item.get("inputs", []),
                    outputs=item.get("outputs", []),
                    dependencies=item.get("dependencies", []),
                    assigned_agent=agent_type,
                    depth=0,
                )
                tasks.append(t)
            except Exception:
                continue

        # Recalculate actual DAG depths
        if tasks:
            id_to_task = {t.id: t for t in tasks}

            def get_depth(task_id: str, visited: set) -> int:
                if task_id in visited:
                    return 0
                visited.add(task_id)
                t = id_to_task.get(task_id)
                if not t or not t.dependencies:
                    return 0
                return 1 + max(get_depth(dep, visited) for dep in t.dependencies)

            for t in tasks:
                t.depth = get_depth(t.id, set())

        return tasks

    def _decompose_goal_keyword(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Keyword-based fallback decomposition when LLM is unavailable."""
        goal_lower = goal.lower()
        tasks = []
        task_counter = [1]

        def next_id(prefix='t'):
            tid = f"{prefix}{task_counter[0]}"
            task_counter[0] += 1
            return tid

        # Analysis task always first
        t1 = Task(
            id=next_id(), description="Analyze requirements and create detailed specification",
            inputs=["goal"], outputs=["requirements_spec", "architecture_doc"],
            dependencies=[], assigned_agent=AgentType.CONSOLIDATED_RESEARCHER, depth=0,
        )
        tasks.append(t1)

        is_code_heavy = any(kw in goal_lower for kw in [
            "code", "implement", "build", "create", "program", "agent",
            "script", "app", "web", "software",
        ])
        is_ui_needed = any(kw in goal_lower for kw in [
            "ui", "frontend", "interface", "web", "app", "dashboard", "website",
        ])
        is_research = any(kw in goal_lower for kw in [
            "research", "analyze", "investigate", "study", "review",
        ])
        is_data = any(kw in goal_lower for kw in [
            "data", "database", "sql", "query", "schema",
        ])

        t2 = Task(
            id=next_id(), description="Set up project structure and dependencies",
            inputs=["requirements_spec"], outputs=["project_structure", "package_files"],
            dependencies=[t1.id], assigned_agent=AgentType.BUILDER, depth=1,
        )
        tasks.append(t2)

        if is_research:
            tasks.append(Task(
                id=next_id(), description="Conduct domain research and competitor analysis",
                inputs=["goal"], outputs=["research_report"],
                dependencies=[t1.id], assigned_agent=AgentType.CONSOLIDATED_RESEARCHER, depth=1,
            ))

        if is_code_heavy:
            t_impl = Task(
                id=next_id(), description="Implement core business logic and data models",
                inputs=["requirements_spec", "project_structure"], outputs=["core_modules"],
                dependencies=[t2.id], assigned_agent=AgentType.BUILDER, depth=1,
            )
            tasks.append(t_impl)
            if is_ui_needed:
                tasks.append(Task(
                    id=next_id(), description="Implement user interface and interactions",
                    inputs=["core_modules"], outputs=["ui_components"],
                    dependencies=[t_impl.id], assigned_agent=AgentType.CONSOLIDATED_RESEARCHER, depth=2,
                ))
            tasks.append(Task(
                id=next_id(), description="Write unit tests and integration tests",
                inputs=["core_modules"], outputs=["test_files"],
                dependencies=[t_impl.id], assigned_agent=AgentType.QUALITY, depth=2,
            ))

        if is_data:
            tasks.append(Task(
                id=next_id(), description="Set up database schema and data layer",
                inputs=["requirements_spec"], outputs=["schema_files"],
                dependencies=[t1.id], assigned_agent=AgentType.CONSOLIDATED_RESEARCHER, depth=1,
            ))

        last = tasks[-1]
        tasks.append(Task(
            id=next_id(), description="Code quality review and refinement",
            inputs=[last.outputs[0] if last.outputs else "result"], outputs=["code_review"],
            dependencies=[last.id], assigned_agent=AgentType.QUALITY, depth=2,
        ))
        tasks.append(Task(
            id=next_id(), description="Generate documentation and final summary",
            inputs=["code_review"], outputs=["documentation"],
            dependencies=[tasks[-1].id], assigned_agent=AgentType.OPERATIONS, depth=3,
        ))
        tasks.append(Task(
            id=next_id(), description="Security review and compliance check",
            inputs=["documentation"], outputs=["security_report"],
            dependencies=[tasks[-1].id], assigned_agent=AgentType.QUALITY, depth=4,
        ))
        return tasks

    # ------------------------------------------------------------------
    # Clarify mode (absorbed from OrchestratorAgent)
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
        """Set the interaction mode for clarify operations."""
        self._interaction_mode = mode
        self._callback = callback

    # ------------------------------------------------------------------
    # Consolidate mode (absorbed from OrchestratorAgent)
    # ------------------------------------------------------------------

    def _execute_consolidate(self, task: AgentTask) -> AgentResult:
        ctx = task.context or {}
        session_id = ctx.get("session_id", "")
        project_id = ctx.get("project_id", "")
        entries = self._load_memory_entries(session_id, project_id)

        if not entries:
            return AgentResult(
                success=True, output=self._fallback_consolidation(task, []),
                metadata={"operation": "consolidate", "entries_processed": 0},
            )

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
            return AgentResult(
                success=True, output=result,
                metadata={"operation": "consolidate", "entries_processed": len(entries)},
            )
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

    # ------------------------------------------------------------------
    # Memory helpers (absorbed from OrchestratorAgent)
    # ------------------------------------------------------------------

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


# Singleton instance
_planner_agent: Optional[PlannerAgent] = None


def get_planner_agent(config: Optional[Dict[str, Any]] = None) -> PlannerAgent:
    """Get the singleton Planner agent instance."""
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = PlannerAgent(config)
    return _planner_agent
