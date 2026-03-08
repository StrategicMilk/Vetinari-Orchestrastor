"""Vetinari Planner Agent — consolidated from User Interaction + Context Manager.

The Planner is the central orchestration agent that generates dynamic plans
from goals, manages user interaction, and coordinates context across agents.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AGENT_REGISTRY,
    AgentResult,
    AgentSpec,
    AgentTask,
    AgentType,
    Plan,
    Task,
    TaskStatus,
    VerificationResult,
    get_enabled_agents,
)

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Planner agent — central orchestration, plan generation, user interaction, and context management.

    Absorbs:
        - UserInteractionAgent: clarification questions, user preference elicitation, feedback collection
        - ContextManagerAgent: context window tracking, compression, cross-agent context sharing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.PLANNER, config)
        self._max_depth = self._config.get("max_depth", 14)
        self._min_tasks = self._config.get("min_tasks", 5)
        self._max_tasks = self._config.get("max_tasks", 15)

    def get_system_prompt(self) -> str:
        agent_descriptions = self._get_dynamic_agent_list()
        return f"""You are Vetinari's Planning Master. You receive a user goal and a context.
Your job is to produce a complete, versioned Plan (DAG) that assigns tasks to the
appropriate agents, defines dependencies, estimates effort, and flags any context
needs or follow-up questions.

You also handle user interaction (clarification, preference gathering, feedback)
and context management (context window tracking, compression, cross-agent sharing).

Rules:
1. Output strictly valid JSON matching the Plan schema
2. Every plan must include a path to final delivery
3. Do NOT execute tasks — only plan and delegate
4. If a subtask fails during execution, propose a re-plan for that subtask tree
5. Include explicit acceptance criteria (Definition of Done) for each task
6. Define a rollback trigger if critical dependencies fail
7. Prefer parallelism: tasks that don't depend on each other should run in parallel
8. Minimum viable plan: 3 tasks. Maximum: 20 tasks per top-level goal.

{agent_descriptions}

Output format: valid JSON array of task objects."""

    def _get_dynamic_agent_list(self) -> str:
        """Build agent list dynamically from the registry."""
        try:
            specs = get_enabled_agents()
            lines = ["Available agents and their roles:"]
            for spec in specs:
                if spec.agent_type == AgentType.PLANNER:
                    continue  # Don't list self
                lines.append(f"- {spec.agent_type.value}: {spec.description}")
            return "\n".join(lines)
        except Exception:
            # Fallback: build from AGENT_REGISTRY directly
            agent_lines = []
            for spec in AGENT_REGISTRY.values():
                agent_lines.append(f"- {spec.agent_type.value}: {', '.join(spec.expertise_areas[:3])}")
            return "Available agents:\n" + "\n".join(agent_lines)
    
    def get_capabilities(self) -> List[str]:
        return [
            "plan_generation",
            "task_decomposition",
            "dependency_mapping",
            "resource_estimation",
            "risk_assessment",
            # From UserInteractionAgent
            "clarification_questions",
            "user_preference_elicitation",
            "feedback_collection",
            # From ContextManagerAgent
            "context_tracking",
            "context_compression",
            "cross_agent_context_sharing",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute task, delegating to user interaction or context manager based on keywords."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("clarify", "ask user", "user preference", "feedback", "user interaction", "elicit", "confirm with user")):
                result = self._delegate_to_user_interaction(task)
            elif any(kw in desc for kw in ("context window", "compress context", "context sharing", "context manager", "token budget", "context limit")):
                result = self._delegate_to_context_manager(task)
            else:
                result = self._execute_planning(task)

            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"PlannerAgent execution failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )

    def _delegate_to_user_interaction(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.user_interaction_agent import UserInteractionAgent
        agent = UserInteractionAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_context_manager(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.context_manager_agent import ContextManagerAgent
        agent = ContextManagerAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _execute_planning(self, task: AgentTask) -> AgentResult:
        """Execute plan generation (original PlannerAgent logic)."""
        goal = task.prompt or task.description
        context = task.context or {}

        plan = self._generate_plan(goal, context)

        return AgentResult(
            success=True,
            output=plan.to_dict(),
            metadata={
                "plan_id": plan.plan_id,
                "task_count": len(plan.tasks),
                "goal": goal
            }
        )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the plan meets quality standards.
        
        Args:
            output: The plan to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check required fields
        required_fields = ["plan_id", "goal", "tasks"]
        for field in required_fields:
            if field not in output:
                issues.append({"type": "missing_field", "message": f"Missing required field: {field}"})
                score -= 0.2
        
        # Check tasks
        tasks = output.get("tasks", [])
        if len(tasks) < self._min_tasks:
            issues.append({"type": "insufficient_tasks", "message": f"Too few tasks: {len(tasks)}"})
            score -= 0.1
        
        # Check for dependencies
        has_dependencies = any(t.get("dependencies") for t in tasks)
        if not has_dependencies:
            issues.append({"type": "no_dependencies", "message": "No task dependencies defined"})
            score -= 0.1
        
        # Pass if score threshold met; issues are warnings, not automatic failures
        passed = score >= 0.7
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _generate_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Generate a plan from the goal using LLM-powered decomposition.

        Falls back to keyword-based decomposition if the LLM is unavailable.
        """
        plan = Plan.create_new(goal)

        # Step 1: Check if the goal is too vague using LLM (or simple heuristics)
        vague_check_prompt = (
            f"User goal: \"{goal}\"\n\n"
            "Is this goal specific enough to begin work on, or does it need clarification?\n"
            "Reply with JSON: {\"is_clear\": true/false, \"clarification_needed\": \"question if not clear\"}"
        )
        vague_result = self._infer_json(vague_check_prompt, expect_json=True)
        if vague_result and not vague_result.get("is_clear", True):
            plan.needs_context = True
            plan.follow_up_question = vague_result.get(
                "clarification_needed",
                "Could you provide more details about what you want to build?"
            )
            return plan

        # Heuristic clarity assessment as fallback
        clarity = self._assess_goal_clarity(goal)
        if clarity < 0.4:
            plan.needs_context = True
            plan.follow_up_question = "Could you provide more details about what you want to build?"
            return plan

        # Step 1.5: Inject similar past plans for better decomposition
        past_plans_context = ""
        try:
            from vetinari.learning.episode_memory import get_episode_memory
            episodes = get_episode_memory().recall(goal, k=3, min_score=0.6)
            if episodes:
                past_plans_context = "\n\nSimilar past tasks that worked well:\n"
                for ep in episodes[:3]:
                    past_plans_context += (
                        f"- Task: {ep.get('task_description', '')[:100]} "
                        f"(agent={ep.get('agent_type', '?')}, "
                        f"quality={ep.get('quality_score', 0):.1f})\n"
                    )
        except Exception:
            pass

        # Step 2: Use LLM to decompose the goal into tasks
        tasks = self._decompose_goal_llm(goal, context, past_plans_context)
        if not tasks:
            # Fallback to keyword-based decomposition
            tasks = self._decompose_goal_keyword(goal, context)

        plan.tasks = tasks

        # Validate the DAG
        dag_issues = self.validate_dag(tasks)
        if dag_issues:
            plan.warnings.extend(dag_issues)
            self._log("warning", f"[Planner] DAG validation issues: {dag_issues}")

        if len(tasks) > self._max_tasks:
            plan.warnings.append(f"Generated {len(tasks)} tasks - consider breaking into smaller goals")

        return plan

    def _assess_goal_clarity(self, goal: str) -> float:
        """Assess how clear and actionable a goal is.

        Returns:
            Clarity score 0.0-1.0 (below 0.4 triggers clarification request).
        """
        score = 0.5  # Baseline
        words = goal.split()

        # Reward: longer, more specific goals
        if len(words) >= 8:
            score += 0.15
        elif len(words) >= 5:
            score += 0.1
        elif len(words) < 3:
            score -= 0.2

        # Reward: contains technical/action terms
        action_terms = {
            "create", "build", "implement", "add", "fix", "remove", "update",
            "refactor", "test", "deploy", "configure", "integrate", "migrate",
            "design", "optimize", "analyze", "generate", "convert", "extract",
        }
        if any(w.lower() in action_terms for w in words):
            score += 0.15

        # Penalty: vague terms
        vague_terms = {"something", "stuff", "things", "maybe", "possibly", "whatever", "idk"}
        vague_count = sum(1 for w in words if w.lower() in vague_terms)
        score -= vague_count * 0.15

        # Reward: mentions specific technologies
        tech_terms = {
            "python", "javascript", "react", "flask", "django", "api", "rest",
            "graphql", "database", "sql", "docker", "kubernetes", "aws", "gcp",
            "css", "html", "typescript", "rust", "go", "java",
        }
        if any(w.lower() in tech_terms for w in words):
            score += 0.1

        # Reward: contains numbers (specific requirements)
        if any(c.isdigit() for c in goal):
            score += 0.05

        return max(0.0, min(1.0, score))

    def _decompose_goal_llm(self, goal: str, context: Dict[str, Any], past_plans: str = "") -> List[Task]:
        """Use LLM to intelligently decompose a goal into ordered tasks."""
        try:
            available_agents = [
                spec.agent_type.value for spec in get_enabled_agents()
                if spec.agent_type != AgentType.PLANNER
            ]
        except Exception:
            available_agents = [
                "EXPLORER", "RESEARCHER", "BUILDER", "TESTER",
                "ARCHITECT", "DOCUMENTER", "RESILIENCE", "META",
            ]
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, default=str)[:500]}"
        past_str = past_plans if past_plans else ""

        decomp_prompt = f"""Goal: {goal}{context_str}{past_str}

Available agents: {', '.join(available_agents)}

Break this goal into 3-{self._max_tasks} discrete, ordered tasks.
For each task specify: id (t1,t2,...), description, inputs (list), outputs (list),
dependencies (list of task ids), assigned_agent (from available agents list),
acceptance_criteria (string describing done condition).

Output valid JSON array of task objects only — no prose, no markdown:
[
  {{"id": "t1", "description": "...", "inputs": ["goal"], "outputs": ["spec"], "dependencies": [], "assigned_agent": "EXPLORER", "acceptance_criteria": "..."}},
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
                # Calculate actual DAG depth rather than dependency count
                t = Task(
                    id=item.get("id", f"t{len(tasks)+1}"),
                    description=item.get("description", "Task"),
                    inputs=item.get("inputs", []),
                    outputs=item.get("outputs", []),
                    dependencies=item.get("dependencies", []),
                    assigned_agent=agent_type,
                    depth=0,  # Depth will be recalculated after all tasks are loaded
                )
                tasks.append(t)
            except Exception:
                continue

        # Recalculate actual DAG depths
        if tasks:
            id_to_task = {t.id: t for t in tasks}
            def get_depth(task_id: str, visited: set) -> int:
                if task_id in visited:
                    return 0  # Cycle guard
                visited.add(task_id)
                t = id_to_task.get(task_id)
                if not t or not t.dependencies:
                    return 0
                return 1 + max(get_depth(dep, visited) for dep in t.dependencies)
            for t in tasks:
                t.depth = get_depth(t.id, set())

        # Return whatever the LLM generated — don't discard valid small plans
        # If truly empty, caller falls back to keyword decomposition
        return tasks

    @staticmethod
    def validate_dag(tasks: List[Task]) -> List[str]:
        """Validate the task DAG for common issues.

        Returns:
            List of issue descriptions (empty if valid).
        """
        issues = []
        task_ids = {t.id for t in tasks}

        # Check for missing dependencies
        for t in tasks:
            for dep in t.dependencies:
                if dep not in task_ids:
                    issues.append(f"Task {t.id} depends on non-existent task {dep}")

        # Check for circular dependencies via DFS
        adj = {t.id: list(t.dependencies) for t in tasks}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in task_ids}

        def dfs(node):
            color[node] = GRAY
            for dep in adj.get(node, []):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    issues.append(f"Circular dependency detected involving task {node} and {dep}")
                    return
                if color[dep] == WHITE:
                    dfs(dep)
            color[node] = BLACK

        for tid in task_ids:
            if color[tid] == WHITE:
                dfs(tid)

        # Check for orphan tasks (no outputs consumed by anyone)
        all_deps = set()
        for t in tasks:
            all_deps.update(t.dependencies)
        leaf_tasks = task_ids - all_deps
        # Having leaf tasks is normal (they're the final outputs), but ALL tasks being leaves is suspicious
        if len(leaf_tasks) == len(tasks) and len(tasks) > 2:
            issues.append("No task dependencies defined — all tasks are independent (may indicate poor decomposition)")

        # Check for empty descriptions
        for t in tasks:
            if not t.description or len(t.description.strip()) < 5:
                issues.append(f"Task {t.id} has empty or too-short description")

        return issues

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
            dependencies=[], assigned_agent=AgentType.EXPLORER, depth=0
        )
        tasks.append(t1)

        is_code_heavy = any(kw in goal_lower for kw in [
            "code", "implement", "build", "create", "program", "agent",
            "script", "app", "web", "software"
        ])
        is_ui_needed = any(kw in goal_lower for kw in [
            "ui", "frontend", "interface", "web", "app", "dashboard", "website"
        ])
        is_research = any(kw in goal_lower for kw in [
            "research", "analyze", "investigate", "study", "review"
        ])
        is_data = any(kw in goal_lower for kw in [
            "data", "database", "sql", "query", "schema"
        ])

        t2 = Task(
            id=next_id(), description="Set up project structure and dependencies",
            inputs=["requirements_spec"], outputs=["project_structure", "package_files"],
            dependencies=[t1.id], assigned_agent=AgentType.BUILDER, depth=1
        )
        tasks.append(t2)

        if is_research:
            tasks.append(Task(
                id=next_id(), description="Conduct domain research and competitor analysis",
                inputs=["goal"], outputs=["research_report"],
                dependencies=[t1.id], assigned_agent=AgentType.RESEARCHER, depth=1
            ))

        if is_code_heavy:
            t_impl = Task(
                id=next_id(), description="Implement core business logic and data models",
                inputs=["requirements_spec", "project_structure"], outputs=["core_modules"],
                dependencies=[t2.id], assigned_agent=AgentType.BUILDER, depth=1
            )
            tasks.append(t_impl)
            if is_ui_needed:
                tasks.append(Task(
                    id=next_id(), description="Implement user interface and interactions",
                    inputs=["core_modules"], outputs=["ui_components"],
                    dependencies=[t_impl.id], assigned_agent=AgentType.UI_PLANNER, depth=2
                ))
            tasks.append(Task(
                id=next_id(), description="Write unit tests and integration tests",
                inputs=["core_modules"], outputs=["test_files"],
                dependencies=[t_impl.id], assigned_agent=AgentType.TEST_AUTOMATION, depth=2
            ))

        if is_data:
            tasks.append(Task(
                id=next_id(), description="Set up database schema and data layer",
                inputs=["requirements_spec"], outputs=["schema_files"],
                dependencies=[t1.id], assigned_agent=AgentType.DATA_ENGINEER, depth=1
            ))

        last = tasks[-1]
        tasks.append(Task(
            id=next_id(), description="Code quality review and refinement",
            inputs=[last.outputs[0] if last.outputs else "result"], outputs=["code_review"],
            dependencies=[last.id], assigned_agent=AgentType.EVALUATOR, depth=2
        ))
        tasks.append(Task(
            id=next_id(), description="Generate documentation and final summary",
            inputs=["code_review"], outputs=["documentation"],
            dependencies=[tasks[-1].id], assigned_agent=AgentType.DOCUMENTATION_AGENT, depth=3
        ))
        tasks.append(Task(
            id=next_id(), description="Security review and compliance check",
            inputs=["documentation"], outputs=["security_report"],
            dependencies=[tasks[-1].id], assigned_agent=AgentType.SECURITY_AUDITOR, depth=4
        ))
        return tasks


# Singleton instance
_planner_agent: Optional[PlannerAgent] = None


def get_planner_agent(config: Optional[Dict[str, Any]] = None) -> PlannerAgent:
    """Get the singleton Planner agent instance."""
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = PlannerAgent(config)
    return _planner_agent
