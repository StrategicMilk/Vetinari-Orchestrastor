"""
Vetinari Planner Agent

The Planner is the central orchestration agent that generates dynamic plans
from goals and coordinates all other agents.
"""

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
Your job is to produce a complete, versioned Plan (DAG) that assigns initial tasks 
to the appropriate agents, defines dependencies, estimates effort, and flags any 
context needs or follow-up questions.

You must:
1. Keep outputs strictly in the Plan schema
2. Include a path to final delivery
3. NOT execute tasks - only plan and delegate
4. Propose re-plans if any subtask fails or exceeds resource budgets
5. Include explicit success criteria for each phase
6. Define a rollback trigger if needed

Available agents: Explorer, Oracle, Librarian, Researcher, Evaluator, 
Synthesizer, Builder, UI Planner, Security Auditor, Data Engineer, 
Documentation Agent, Cost Planner, Test Automation, Experimentation Manager

Output format must be a valid Plan object."""
    
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
        
        passed = score >= 0.7 and len(issues) == 0
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _generate_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Generate a plan from the goal.
        
        Args:
            goal: The user's goal
            context: Optional context information
            
        Returns:
            A Plan object
        """
        # Check if goal is too vague
        vague_indicators = ["something", "stuff", "things", "make", "do", "create something"]
        goal_words = goal.lower().split()
        
        plan = Plan.create_new(goal)
        
        if len(goal_words) < 5 or any(vague in goal.lower() for vague in vague_indicators):
            plan.needs_context = True
            plan.follow_up_question = "Could you provide more details about what you want to build?"
            return plan
        
        # Generate tasks based on goal analysis
        tasks = self._decompose_goal(goal, context)
        plan.tasks = tasks
        
        # Add warnings if needed
        if len(tasks) > self._max_tasks:
            plan.warnings.append(f"Generated {len(tasks)} tasks - consider breaking into smaller goals")
        
        return plan
    
    def _decompose_goal(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Decompose a goal into tasks.
        
        Args:
            goal: The goal to decompose
            context: Context information
            
        Returns:
            List of Task objects
        """
        goal_lower = goal.lower()
        tasks = []
        task_counter = [1]
        
        def next_id(prefix='t'):
            tid = f"{prefix}{task_counter[0]}"
            task_counter[0] += 1
            return tid
        
        # Always start with analysis
        t1 = Task(
            id=next_id(),
            description="Analyze requirements and create detailed specification",
            inputs=["goal"],
            outputs=["requirements_spec", "architecture_doc"],
            dependencies=[],
            assigned_agent=AgentType.EXPLORER,
            depth=0
        )
        tasks.append(t1)
        
        # Determine task types based on goal
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
            "data", "database", "sql", "query", "analyze"
        ])
        
        # Setup task
        t2 = Task(
            id=next_id(),
            description="Set up project structure and dependencies",
            inputs=["requirements_spec"],
            outputs=["project_structure", "package_files"],
            dependencies=[t1.id],
            assigned_agent=AgentType.BUILDER,
            depth=1
        )
        tasks.append(t2)
        
        if is_code_heavy:
            # Core implementation
            t3 = Task(
                id=next_id(),
                description="Implement core business logic and data models",
                inputs=["requirements_spec", "project_structure"],
                outputs=["core_modules", "data_models"],
                dependencies=[t2.id],
                assigned_agent=AgentType.BUILDER,
                depth=1
            )
            tasks.append(t3)
            
            # UI if needed
            if is_ui_needed:
                t4 = Task(
                    id=next_id(),
                    description="Implement user interface and interactions",
                    inputs=["core_modules", "requirements_spec"],
                    outputs=["ui_components", "frontend_code"],
                    dependencies=[t3.id],
                    assigned_agent=AgentType.UI_PLANNER,
                    depth=1
                )
                tasks.append(t4)
            
            # Tests
            t5 = Task(
                id=next_id(),
                description="Write unit tests and integration tests",
                inputs=["core_modules"],
                outputs=["test_files", "test_results"],
                dependencies=[t3.id],
                assigned_agent=AgentType.TEST_AUTOMATION,
                depth=1
            )
            tasks.append(t5)
        
        if is_data:
            t6 = Task(
                id=next_id(),
                description="Set up database and data layer",
                inputs=["requirements_spec"],
                outputs=["schema_files", "migration_scripts"],
                dependencies=[t1.id],
                assigned_agent=AgentType.DATA_ENGINEER,
                depth=1
            )
            tasks.append(t6)
        
        if is_research:
            t7 = Task(
                id=next_id(),
                description="Conduct domain research and competitor analysis",
                inputs=["goal"],
                outputs=["research_report", "competitor_analysis"],
                dependencies=[t1.id],
                assigned_agent=AgentType.RESEARCHER,
                depth=1
            )
            tasks.append(t7)
        
        # Quality review
        last_task = tasks[-1]
        t8 = Task(
            id=next_id(),
            description="Review code quality and refine implementation",
            inputs=[last_task.outputs[0]] if last_task.outputs else ["goal"],
            outputs=["code_review", "refactoring_suggestions"],
            dependencies=[last_task.id],
            assigned_agent=AgentType.EVALUATOR,
            depth=1
        )
        tasks.append(t8)
        
        # Documentation
        t9 = Task(
            id=next_id(),
            description="Create documentation and final summary",
            inputs=["code_review"],
            outputs=["documentation", "final_summary"],
            dependencies=[t8.id],
            assigned_agent=AgentType.DOCUMENTATION_AGENT,
            depth=1
        )
        tasks.append(t9)
        
        # Security review
        t10 = Task(
            id=next_id(),
            description="Security review and policy compliance check",
            inputs=["documentation"],
            outputs=["security_report", "compliance_status"],
            dependencies=[t9.id],
            assigned_agent=AgentType.SECURITY_AUDITOR,
            depth=1
        )
        tasks.append(t10)
        
        return tasks


# Singleton instance
_planner_agent: Optional[PlannerAgent] = None


def get_planner_agent(config: Optional[Dict[str, Any]] = None) -> PlannerAgent:
    """Get the singleton Planner agent instance."""
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = PlannerAgent(config)
    return _planner_agent
