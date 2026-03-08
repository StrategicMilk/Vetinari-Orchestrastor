"""
Vetinari Planner Agent

The Planner is the central orchestration agent that generates dynamic plans
from goals and coordinates all other agents.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentSpec,
    AgentTask,
    AgentType,
    Plan,
    Task,
    TaskStatus,
    VerificationResult,
    get_enabled_agents
)


class PlannerAgent(BaseAgent):
    """Planner agent - central orchestration and dynamic plan generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.PLANNER, config)
        self._max_depth = self._config.get("max_depth", 14)
        self._min_tasks = self._config.get("min_tasks", 5)
        self._max_tasks = self._config.get("max_tasks", 15)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Planning Master. You receive a user goal and a context.
Your job is to produce a complete, versioned Plan (DAG) that assigns tasks to the
appropriate agents, defines dependencies, estimates effort, and flags any context
needs or follow-up questions.

Rules:
1. Output strictly valid JSON matching the Plan schema
2. Every plan must include a path to final delivery
3. Do NOT execute tasks — only plan and delegate
4. If a subtask fails during execution, propose a re-plan for that subtask tree
5. Include explicit acceptance criteria (Definition of Done) for each task
6. Define a rollback trigger if critical dependencies fail
7. Prefer parallelism: tasks that don't depend on each other should run in parallel
8. Minimum viable plan: 3 tasks. Maximum: 20 tasks per top-level goal.
9. Prefer consolidated agent types when available (see Affinity Table below)

Consolidated agents (preferred — use these when possible):
- PLANNER: Goal decomposition, task scheduling, dependency mapping, specification
- ORCHESTRATOR: User interaction, Socratic clarification, system monitoring, memory
- RESEARCHER: Code discovery, API/library lookup, domain research, lateral thinking
  (replaces EXPLORER, LIBRARIAN, legacy RESEARCHER)
- ORACLE: Architecture decisions, risk assessment, ontological analysis, contrarian review
  (replaces legacy ORACLE, PONDER)
- BUILDER: Code implementation, scaffolding, refactoring, test writing
- ARCHITECT: UI/UX design, database schemas, DevOps pipelines, git workflow
  (replaces UI_PLANNER, DATA_ENGINEER, DEVOPS, VERSION_CONTROL)
- QUALITY: Code review, test generation, security audit, simplification
  (replaces EVALUATOR, SECURITY_AUDITOR, TEST_AUTOMATION)
- OPERATIONS: Documentation, creative writing, cost analysis, experiments,
  error recovery, synthesis, image generation
  (replaces SYNTHESIZER, DOCUMENTATION_AGENT, COST_PLANNER,
  EXPERIMENTATION_MANAGER, IMPROVEMENT, ERROR_RECOVERY, IMAGE_GENERATOR)

Legacy agents (still available for backward compatibility):
- EXPLORER, LIBRARIAN, ORACLE, PONDER, UI_PLANNER, DATA_ENGINEER, DEVOPS,
  VERSION_CONTROL, EVALUATOR, SECURITY_AUDITOR, TEST_AUTOMATION, SYNTHESIZER,
  DOCUMENTATION_AGENT, COST_PLANNER, EXPERIMENTATION_MANAGER, IMPROVEMENT,
  ERROR_RECOVERY, CONTEXT_MANAGER, IMAGE_GENERATOR, USER_INTERACTION

Affinity Table — task type to preferred agent:
  code/implement/build/scaffold/refactor → BUILDER
  research/explore/discover/lookup/api   → RESEARCHER
  review/test/security/audit/quality     → QUALITY
  plan/decompose/schedule/specify        → PLANNER
  design/ui/database/devops/git          → ARCHITECT
  architecture/risk/decision/contrarian  → ORACLE
  document/write/summarize/cost/recover  → OPERATIONS
  clarify/interact/monitor/consolidate   → ORCHESTRATOR

Output format: valid JSON array of task objects."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "plan_generation",
            "task_decomposition",
            "dependency_mapping",
            "resource_estimation",
            "risk_assessment"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the planning task.
        
        Args:
            task: The task containing the goal to plan for
            
        Returns:
            AgentResult containing the generated Plan
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            goal = task.prompt or task.description
            context = task.context or {}
            
            # Create the plan
            plan = self._generate_plan(goal, context)
            
            # Complete the task
            task = self.complete_task(task, AgentResult(
                success=True,
                output=plan.to_dict(),
                metadata={"plan_id": plan.plan_id, "task_count": len(plan.tasks)}
            ))
            
            return AgentResult(
                success=True,
                output=plan.to_dict(),
                metadata={
                    "plan_id": plan.plan_id,
                    "task_count": len(plan.tasks),
                    "goal": goal
                }
            )
            
        except Exception as e:
            self._log("error", f"Planning failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
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

        # Step 1: Heuristic vagueness check (fast, no LLM call needed)
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
            # Fallback to keyword-based decomposition
            tasks = self._decompose_goal_keyword(goal, context)

        plan.tasks = tasks

        if len(tasks) > self._max_tasks:
            plan.warnings.append(f"Generated {len(tasks)} tasks - consider breaking into smaller goals")

        return plan

    def _decompose_goal_llm(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Use LLM to intelligently decompose a goal into ordered tasks."""
        # Consolidated agents (preferred) + legacy for backward compat
        available_agents = [
            # Consolidated (preferred)
            "PLANNER", "ORCHESTRATOR", "RESEARCHER", "ORACLE",
            "BUILDER", "ARCHITECT", "QUALITY", "OPERATIONS",
            # Legacy (still accepted)
            "EXPLORER", "LIBRARIAN", "EVALUATOR",
            "SYNTHESIZER", "UI_PLANNER", "SECURITY_AUDITOR",
            "DATA_ENGINEER", "DOCUMENTATION_AGENT", "COST_PLANNER",
            "TEST_AUTOMATION", "EXPERIMENTATION_MANAGER", "DEVOPS",
            "VERSION_CONTROL", "ERROR_RECOVERY", "IMAGE_GENERATOR",
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
