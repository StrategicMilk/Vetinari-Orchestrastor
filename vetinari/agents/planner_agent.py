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
                "You are Vetinari's Planning Master. You receive a user goal and context.\n"
                "Your job is to produce a complete, versioned Plan (DAG) that assigns tasks to\n"
                "the appropriate agents, defines dependencies, estimates effort, and flags any\n"
                "context needs or follow-up questions.\n\n"
                "Rules:\n"
                "1. Output strictly valid JSON matching the Plan schema\n"
                "2. Every plan must include a path to final delivery\n"
                "3. Do NOT execute tasks — only plan and delegate\n"
                "4. If a subtask fails during execution, propose a re-plan for that subtask tree\n"
                "5. Include explicit acceptance criteria (Definition of Done) for each task\n"
                "6. Define a rollback trigger if critical dependencies fail\n"
                "7. Prefer parallelism: tasks that don't depend on each other should run concurrently\n"
                "8. Minimum viable plan: 3 tasks. Maximum: 20 tasks per top-level goal.\n\n"
                "Active agents (6 consolidated):\n"
                "- PLANNER: Goal decomposition, scheduling, user interaction, context management\n"
                "- RESEARCHER: Code discovery, API lookup, domain research, lateral thinking,\n"
                "  UI/UX design, database schemas, DevOps pipelines, git workflow\n"
                "- ORACLE: Architecture decisions, risk assessment, ontological analysis, contrarian review\n"
                "- BUILDER: Code implementation, scaffolding, image generation\n"
                "- QUALITY: Code review, test generation, security audit, simplification\n"
                "- OPERATIONS: Documentation, creative writing, cost analysis, experiments,\n"
                "  error recovery, synthesis, improvement, monitoring\n\n"
                "Affinity Table — task type to agent:\n"
                "  code/implement/build/scaffold/refactor   -> BUILDER\n"
                "  research/explore/discover/lookup/api     -> RESEARCHER\n"
                "  review/test/security/audit/quality       -> QUALITY\n"
                "  plan/decompose/schedule/specify          -> PLANNER\n"
                "  design/ui/database/devops/git            -> RESEARCHER\n"
                "  architecture/risk/decision/contrarian    -> ORACLE\n"
                "  document/write/summarize/cost/recover    -> OPERATIONS\n"
                "  clarify/interact/consolidate             -> PLANNER\n"
                "  image/logo/icon/diagram/mockup           -> BUILDER\n\n"
                "DECOMPOSITION QUALITY:\n"
                "- Every subtask must have measurable completion criteria\n"
                "- Prefer parallel execution: tasks without dependencies should run concurrently\n"
                "- Estimate effort: XS (<5min), S (<30min), M (<2h), L (<8h), XL (>8h)\n\n"
                "RISK ASSESSMENT:\n"
                "- Identify the top 3 risks with likelihood and impact\n"
                "- Include a rollback strategy for the highest-risk subtask\n\n"
                "SELF-CHECK:\n"
                "- Does every task have exactly one responsible agent?\n"
                "- Are all dependencies explicit?\n"
                "- Is there a critical path? What is the estimated total duration?\n"
                "- Would removing any task break the plan?"
            ),
            "clarify": (
                "You are Vetinari's User Interaction Specialist. Your role is to:\n"
                "1. Detect when user goals are ambiguous or under-specified\n"
                "2. Generate clear, targeted clarifying questions\n"
                "3. Prioritize the most important questions (max 3)\n"
                "4. Incorporate user responses into the task context\n\n"
                "Ask no more than 3 clarifying questions. Rank by information value. "
                "Stop when ambiguity drops below 20%.\n"
                "Be concise and specific."
            ),
            "consolidate": (
                "You are a context and memory management specialist. Your role is to:\n"
                "- Summarise long interaction histories into concise digests\n"
                "- Identify and retain the most relevant knowledge\n"
                "- Remove redundant or stale context to stay within token budgets\n"
                "- Build structured knowledge representations\n"
                "- Detect contradictions or outdated information\n\n"
                "Preserve all unique information. Flag contradictions rather than "
                "silently resolving them.\n"
                "Always respond with structured JSON."
            ),
            "summarise": (
                "You are a session summarisation specialist. Produce concise, "
                "structured digests of session histories, identifying goals achieved "
                "and recommended next steps."
            ),
            "prune": (
                "You are a context pruning specialist. Reduce context to fit within "
                "token budgets while retaining the highest-relevance entries."
            ),
            "extract": (
                "You are a knowledge extraction specialist. Extract structured "
                "facts and entities from text with confidence scores."
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
