"""Planning Engine — LEGACY module for model-selection-aware task generation.

.. deprecated::
    This module will be merged into plan_mode.py in a future release.
    New code should use ``vetinari.plan_mode.PlanModeEngine`` where possible.
"""

from __future__ import annotations

import logging
import warnings

warnings.warn(
    "vetinari.planning_engine is deprecated and will be merged into "
    "vetinari.plan_mode in a future release. Use "
    "vetinari.plan_mode.PlanModeEngine instead.",
    DeprecationWarning,
    stacklevel=2,
)
from dataclasses import dataclass, field  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class Model:
    """Model configuration for the planning engine."""
    id: str
    name: str
    capabilities: list[str] = field(default_factory=list)
    context_len: int = 2048
    memory_gb: int = 2
    version: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "capabilities": self.capabilities,
            "context_len": self.context_len,
            "memory_gb": self.memory_gb,
            "version": self.version,
        }


@dataclass
class Task:
    """A unit of work within a plan or wave."""
    id: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    model_override: str = ""
    assigned_model_id: str = ""
    depth: int = 0
    parent_id: str = ""
    children: list[str] = field(default_factory=list)
    owner_id: str = ""
    status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
            "model_override": self.model_override,
            "assigned_model_id": self.assigned_model_id,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children": self.children,
            "owner_id": self.owner_id,
            "status": self.status,
        }


@dataclass
class Plan:
    """An execution plan containing ordered waves of tasks."""
    goal: str
    tasks: list[Task] = field(default_factory=list)
    model_scores: list[dict] = field(default_factory=list)
    notes: str = ""
    warnings: list[str] = field(default_factory=list)
    needs_context: bool = False
    follow_up_question: str = ""
    final_delivery_path: str = ""
    final_delivery_summary: str = ""

    def to_dict(self) -> dict:
        """To dict.

        Returns:
            Dictionary of results.
        """
        d = {
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "model_scores": self.model_scores,
            "notes": self.notes,
            "warnings": self.warnings,
            "needs_context": self.needs_context,
            "follow_up_question": self.follow_up_question,
        }
        if self.final_delivery_path:
            d["final_delivery_path"] = self.final_delivery_path
        if self.final_delivery_summary:
            d["final_delivery_summary"] = self.final_delivery_summary
        return d


class PlanningEngine:
    """Planning engine."""
    def __init__(
        self,
        default_models: list[str] | None = None,
        fallback_models: list[str] | None = None,
        uncensored_fallback_models: list[str] | None = None,
        memory_budget_gb: int = 48,
    ):
        self.default_models = default_models or []
        self.fallback_models = fallback_models or []
        self.uncensored_fallback_models = uncensored_fallback_models or []
        self.memory_budget_gb = memory_budget_gb

        # Prompt type detection keywords
        self.prompt_type_keywords = {
            "planning": ["plan", "strategy", "design", "workflow", "analyze", "architect", "organize", "structure"],
            "coding": [
                "code",
                "implement",
                "build",
                "create",
                "program",
                "script",
                "function",
                "class",
                "api",
                "software",
                "agent",
                "app",
                "web",
                "python",
                "javascript",
            ],
            "docs": ["document", "readme", "explain", "comment", "write", "description", "tutorial", "guide"],
            "reasoning": ["reason", "think", "analyze", "solve", "problem", "logic", "math", "calculate"],
            "creative": ["story", "poem", "creative", "write", "content", "article", "blog"],
            "financial": [
                "stock",
                "investment",
                "trading",
                "financial",
                "money",
                "profit",
                "crypto",
                "market",
                "economy",
            ],
            "data": ["data", "analyze", "process", "extract", "transform", "etl", "database", "query"],
        }

        # Keywords that indicate a model might refuse due to policy
        self.policy_sensitive_keywords = [
            "investment",
            "stock",
            "trading",
            "financial",
            "money",
            "profit",
            "hack",
            "exploit",
            "weapon",
            "illegal",
            "harmful",
            "adult",
        ]

        self.capability_keywords = {
            "code_gen": [
                "code",
                "implement",
                "build",
                "create",
                "python",
                "javascript",
                "script",
                "api",
                "web",
                "function",
                "class",
                "program",
                "software",
                "agent",
                "generate",
            ],
            "docs": ["document", "readme", "explain", "comment", "docs", "description", "writeup"],
            "chat": ["chat", "conversation", "message", "respond", "reply", "talk"],
            "reasoning": ["plan", "reason", "think", "analyze", "strategy", "design", "architect", "workflow"],
            "math": ["calculate", "math", "compute", "formula", "equation"],
            "creative": ["write", "story", "poem", "creative", "content"],
        }

    def plan(self, goal: str, system_prompt: str, models: list[dict], planning_model: str | None = None) -> Plan:
        """Create a plan for the given goal.

        Args:
            goal: The goal.
            system_prompt: The system prompt.
            models: The models.
            planning_model: The planning model.

        Returns:
            The Plan result.
        """
        plan = Plan(goal=goal)

        # Convert dict models to Model objects, filtering extra fields
        def to_model(m):
            return Model(
                id=m.get("id", m.get("name", "")),
                name=m.get("name", m.get("id", "")),
                capabilities=m.get("capabilities", []),
                context_len=m.get("context_len", 2048),
                memory_gb=m.get("memory_gb", 2),
                version=m.get("version", ""),
            )

        all_models = [to_model(m) for m in models]

        # Filter models by memory budget
        available_models = [m for m in all_models if m.memory_gb <= self.memory_budget_gb]

        # Also keep models that don't have memory info (assume they're small)
        available_models.extend([m for m in all_models if m.memory_gb == 2 and m not in available_models])

        if not available_models:
            plan.warnings.append(
                f"No models fit within memory budget of {self.memory_budget_gb}GB. Using smallest available."
            )
            available_models = sorted(all_models, key=lambda x: x.memory_gb)[:3]

        # Check if goal is policy-sensitive (might need uncensored fallback)
        goal_lower = goal.lower()
        is_policy_sensitive = any(kw in goal_lower for kw in self.policy_sensitive_keywords)

        # Detect prompt type
        prompt_type = self._detect_prompt_type(goal)
        plan.notes = f"Prompt type: {prompt_type}. "

        if is_policy_sensitive and self.uncensored_fallback_models:
            plan.warnings.append("Goal may be policy-sensitive. Using uncensored fallback models.")

        # Score all models for planning task
        planning_task = Task(
            id="planning",
            description=f"Analyze goal and create workflow plan: {goal[:100]}...",
            inputs=["goal"],
            outputs=["workflow_plan"],
            dependencies=[],
        )

        # Score models for planning
        planning_scores = self._score_models_for_task(planning_task, available_models)
        plan.model_scores = planning_scores

        # Select best model for planning
        best_planner = self._select_best_model(planning_scores, preferred_capabilities=["reasoning", "code_gen"])

        if not best_planner:
            plan.warnings.append("No suitable model found for planning. Using fallback.")
            best_planner = self._get_fallback_model(available_models, is_policy_sensitive)

        plan.notes = f"Planning model: {best_planner}"

        # Check if goal is too vague and needs more context
        vague_indicators = ["something", "stuff", "things", "make", "do", "create something", "build something"]
        goal_words = goal.lower().split()

        # If goal is very short or contains vague terms, ask for clarification
        if len(goal_words) < 5 or any(vague in goal.lower() for vague in vague_indicators):
            plan.needs_context = True
            plan.follow_up_question = "Could you provide more details about what you want to build? For example: What specific features should it have? What programming language or framework should it use? Who is the target user?"
            return plan

        # Use the best planning model to generate tasks
        # For now, we'll use a simple heuristic to generate tasks
        # In production, this would call the model to get the actual plan
        tasks = self._generate_tasks_from_goal(goal)

        # Score and assign models to each task
        for task in tasks:
            task_scores = self._score_models_for_task(task, available_models)

            # Check if task has explicit model override
            if task.model_override:
                task.assigned_model_id = task.model_override
            else:
                # Select best model for this task
                assigned = self._select_best_model(task_scores)

                if not assigned:
                    # Use fallback
                    plan.warnings.append(f"No suitable model for task {task.id}. Using fallback.")
                    assigned = self._get_fallback_model(available_models, is_policy_sensitive)

                task.assigned_model_id = assigned

            plan.tasks.append(task)

        # Check token limits
        self._check_token_limits(plan, available_models)

        return plan

    def _get_fallback_model(self, available_models: list[Model], is_policy_sensitive: bool = False) -> str:
        """Get the best fallback model, preferring uncensored if policy-sensitive."""
        # First try uncensored fallbacks if policy-sensitive
        if is_policy_sensitive and self.uncensored_fallback_models:
            for model_id in self.uncensored_fallback_models:
                for m in available_models:
                    if model_id.lower() in m.id.lower() or model_id.lower() in m.name.lower():
                        return m.id

        # Then try regular fallbacks
        if self.fallback_models:
            for model_id in self.fallback_models:
                for m in available_models:
                    if model_id.lower() in m.id.lower() or model_id.lower() in m.name.lower():
                        return m.id

        # Then try default models
        if self.default_models:
            for model_id in self.default_models:
                for m in available_models:
                    if model_id.lower() in m.id.lower() or model_id.lower() in m.name.lower():
                        return m.id

        # Finally, just return the smallest available model
        if available_models:
            smallest = min(available_models, key=lambda x: x.memory_gb)
            return smallest.id

        return ""

    def _check_for_policy_refusal(self, model_id: str, task_description: str) -> bool:
        """Check if a model might refuse based on task content."""
        # Models with certain keywords in name might be more likely to refuse
        refuse_indicators = ["safe", "guard", "policy", "claude", "gpt"]

        for indicator in refuse_indicators:
            if indicator in model_id.lower():
                # Check if task is policy-sensitive
                task_lower = task_description.lower()
                if any(kw in task_lower for kw in self.policy_sensitive_keywords):
                    return True
        return False

    def _detect_prompt_type(self, goal: str) -> str:
        """Detect the type of prompt based on keywords."""
        goal_lower = goal.lower()

        # Count matches for each type
        type_scores = {}
        for prompt_type, keywords in self.prompt_type_keywords.items():
            score = sum(1 for kw in keywords if kw in goal_lower)
            if score > 0:
                type_scores[prompt_type] = score

        if not type_scores:
            return "general"

        # Return the type with highest score
        return max(type_scores, key=type_scores.get)

    def _get_best_model_for_type(self, prompt_type: str, available_models: list[Model]) -> str:
        """Get the best model configuration for a specific prompt type."""
        # Define preferred capabilities for each prompt type
        type_preferences = {
            "planning": ["reasoning", "code_gen"],
            "coding": ["code_gen"],
            "docs": ["docs", "code_gen"],
            "reasoning": ["reasoning"],
            "creative": ["chat", "creative"],
            "financial": ["code_gen", "reasoning"],  # Use code models for financial tasks
            "data": ["code_gen"],
        }

        preferred_caps = type_preferences.get(prompt_type, ["code_gen", "chat"])

        # First try to find a model from defaults that matches
        for default_id in self.default_models:
            for m in available_models:
                if default_id.lower() in m.id.lower():
                    # Check if model has preferred capabilities
                    for cap in preferred_caps:
                        if cap in m.capabilities:
                            return m.id

        # Then try fallbacks
        for fallback_id in self.fallback_models:
            for m in available_models:
                if fallback_id.lower() in m.id.lower():
                    return m.id

        # Finally return the highest scoring model
        if available_models:
            return available_models[0].id

        return ""

    def _generate_tasks_from_goal(
        self, goal: str, parent_id: str = "", depth: int = 0, max_depth: int = 25
    ) -> list[Task]:
        """Generate granular tasks based on the goal with unlimited depth subtasks."""
        if depth >= max_depth:
            return []

        goal_lower = goal.lower()

        tasks = []
        task_counter = [1]

        def next_id(prefix="t"):
            """Next id for the current context.

            Returns:
                The result value.
            """
            tid = f"{prefix}{task_counter[0]}"
            task_counter[0] += 1
            return tid

        is_code_heavy = any(
            kw in goal_lower
            for kw in ["code", "implement", "build", "create", "program", "agent", "script", "app", "web", "software"]
        )
        any(kw in goal_lower for kw in ["plan", "strategy", "design", "workflow", "analyze"])
        any(kw in goal_lower for kw in ["document", "readme", "explain", "docs", "write"])
        is_data = any(kw in goal_lower for kw in ["data", "database", "sql", "query", "analyze"])
        is_api = any(kw in goal_lower for kw in ["api", "rest", "endpoint", "service"])
        is_research = any(kw in goal_lower for kw in ["research", "analyze", "investigate", "study", "review"])

        if depth == 0:
            t1 = Task(
                id=next_id(),
                description="Analyze requirements and create detailed specification",
                inputs=["goal"],
                outputs=["requirements_spec", "architecture_doc"],
                dependencies=[],
                model_override="",
                depth=depth,
                parent_id=parent_id,
            )
            tasks.append(t1)
            parent_id = t1.id

            t2 = Task(
                id=next_id(),
                description="Set up project structure and dependencies",
                inputs=["requirements_spec"],
                outputs=["project_structure", "package_files", "config_files"],
                dependencies=[t1.id],
                model_override="",
                depth=depth + 1,
                parent_id=parent_id,
            )
            tasks.append(t2)

            if is_code_heavy:
                t3 = Task(
                    id=next_id(),
                    description="Implement core business logic and data models",
                    inputs=["requirements_spec", "project_structure"],
                    outputs=["core_modules", "data_models"],
                    dependencies=[t2.id],
                    model_override="",
                    depth=depth + 1,
                    parent_id=parent_id,
                )
                tasks.append(t3)

                if is_api:
                    t3b = Task(
                        id=next_id(),
                        description="Implement API endpoints and service layer",
                        inputs=["core_modules", "requirements_spec"],
                        outputs=["api_endpoints", "service_layer"],
                        dependencies=[t3.id],
                        model_override="",
                        depth=depth + 2,
                        parent_id=t3.id,
                    )
                    tasks.append(t3b)

                t4 = Task(
                    id=next_id(),
                    description="Implement user interface and interactions",
                    inputs=["core_modules", "requirements_spec"],
                    outputs=["ui_components", "frontend_code"],
                    dependencies=[t3.id],
                    model_override="",
                    depth=depth + 1,
                    parent_id=parent_id,
                )
                tasks.append(t4)

                t5 = Task(
                    id=next_id(),
                    description="Write unit tests and integration tests",
                    inputs=["core_modules", "api_endpoints"] if is_api else ["core_modules"],
                    outputs=["test_files", "test_results"],
                    dependencies=[t4.id],
                    model_override="",
                    depth=depth + 1,
                    parent_id=parent_id,
                )
                tasks.append(t5)

            if is_data:
                t6 = Task(
                    id=next_id(),
                    description="Set up database and data layer",
                    inputs=["requirements_spec"],
                    outputs=["schema_files", "migration_scripts", "seed_data"],
                    dependencies=[t1.id],
                    model_override="",
                    depth=depth + 1,
                    parent_id=parent_id,
                )
                tasks.append(t6)

            t7 = Task(
                id=next_id(),
                description="Review code quality and refine implementation",
                inputs=[tasks[-1].outputs[0]] if tasks and tasks[-1].outputs else ["goal"],
                outputs=["code_review", "refactoring_suggestions"],
                dependencies=[tasks[-1].id] if len(tasks) > 1 else [],
                model_override="",
                depth=depth + 1,
                parent_id=parent_id,
            )
            tasks.append(t7)

            t8 = Task(
                id=next_id(),
                description="Run full test suite and validate functionality",
                inputs=["test_files", "code_review"],
                outputs=["validation_report", "test_coverage"],
                dependencies=[t7.id],
                model_override="",
                depth=depth + 1,
                parent_id=parent_id,
            )
            tasks.append(t8)

            if is_code_heavy:
                t9 = Task(
                    id=next_id(),
                    description="Build executable package or deployment artifacts",
                    inputs=["validated_code", "config_files"],
                    outputs=["build_artifacts", "deployment_package"],
                    dependencies=[t8.id],
                    model_override="",
                    depth=depth + 1,
                    parent_id=parent_id,
                )
                tasks.append(t9)

                t10 = Task(
                    id=next_id(),
                    description="Create deployment guide and final summary",
                    inputs=["build_artifacts"],
                    outputs=["deployment_guide", "final_summary"],
                    dependencies=[t9.id],
                    model_override="",
                    depth=depth + 1,
                    parent_id=parent_id,
                )
                tasks.append(t10)
            else:
                t10 = Task(
                    id=next_id(),
                    description="Create final documentation and summary",
                    inputs=["documentation"],
                    outputs=["final_summary", "user_guide"],
                    dependencies=[t8.id],
                    model_override="",
                    depth=depth + 1,
                    parent_id=parent_id,
                )
                tasks.append(t10)

        elif depth == 1 and is_code_heavy:
            subtasks = [
                ("Implement error handling and logging", ["core_modules"], ["error_handlers", "logging_config"]),
                ("Implement configuration management", ["project_structure"], ["config_module", "env_handlers"]),
                ("Implement utility functions and helpers", ["core_modules"], ["utilities", "helpers"]),
                ("Set up CI/CD pipeline", ["project_structure"], ["ci_config", "cd_scripts"]),
                ("Implement security measures", ["core_modules"], ["security_module", "auth_handlers"]),
            ]
            for desc, inp, out in subtasks:
                t = Task(
                    id=next_id(),
                    description=desc,
                    inputs=inp,
                    outputs=out,
                    dependencies=[parent_id],
                    model_override="",
                    depth=depth,
                    parent_id=parent_id,
                )
                tasks.append(t)

        elif depth == 1 and is_research:
            subtasks = [
                (
                    "Gather background information and existing solutions",
                    ["goal"],
                    ["background_docs", "existing_solutions"],
                ),
                (
                    "Analyze competitors and alternatives",
                    ["background_docs"],
                    ["competitor_analysis", "comparison_report"],
                ),
                ("Identify gaps and opportunities", ["competitor_analysis"], ["gap_analysis", "opportunities"]),
                ("Evaluate feasibility and risks", ["gap_analysis"], ["feasibility_report", "risk_assessment"]),
                ("Document findings and recommendations", ["feasibility_report"], ["final_report", "action_plan"]),
            ]
            for desc, inp, out in subtasks:
                t = Task(
                    id=next_id(),
                    description=desc,
                    inputs=inp,
                    outputs=out,
                    dependencies=[parent_id],
                    model_override="",
                    depth=depth,
                    parent_id=parent_id,
                )
                tasks.append(t)

        elif depth >= 2:
            subtasks = [
                ("Refine and optimize implementation", ["previous_output"], ["optimized_code", "performance_notes"]),
                ("Add edge case handling", ["previous_output"], ["edge_cases", "fallback_handlers"]),
                ("Document internal APIs", ["previous_output"], ["internal_docs", "api_reference"]),
                (
                    "Verify compliance with requirements",
                    ["previous_output", "requirements_spec"],
                    ["compliance_report"],
                ),
            ]
            for i, (desc, inp, out) in enumerate(subtasks):
                t = Task(
                    id=next_id(),
                    description=desc,
                    inputs=inp,
                    outputs=out,
                    dependencies=[parent_id] if i == 0 else [tasks[-1].id],
                    model_override="",
                    depth=depth,
                    parent_id=parent_id,
                )
                tasks.append(t)

        return tasks

    def _score_models_for_task(self, task: Task, models: list[Model]) -> list[dict]:
        """Score all models for a given task."""
        scores = []

        task_lower = task.description.lower()
        required_caps = self._infer_required_capabilities(task_lower)

        for model in models:
            score = 0
            cap_matches = []

            # Capability matching (40% weight)
            for req_cap in required_caps:
                if req_cap in model.capabilities:
                    score += 40
                    cap_matches.append(req_cap)
                # Partial match
                for model_cap in model.capabilities:
                    if req_cap in model_cap or model_cap in req_cap:
                        score += 20
                        cap_matches.append(f"partial:{model_cap}")

            # Context length fit (20% weight)
            # Prefer smaller context for simpler tasks
            if task.outputs:
                estimated_tokens = sum(len(o) for o in task.outputs) // 4
                if model.context_len >= estimated_tokens:
                    score += 20
                else:
                    score -= 10

            # Memory efficiency (20% weight)
            # Prefer smaller models for simpler tasks
            if len(task.inputs) <= 1 and len(task.outputs) <= 1:
                if model.memory_gb <= 4:
                    score += 20
                elif model.memory_gb <= 16:
                    score += 10
            else:
                # More complex task - allow larger models
                if model.memory_gb >= 16:
                    score += 15
                elif model.memory_gb >= 8:
                    score += 10

            # Task-specific heuristics (20% weight)
            if "reasoning" in required_caps or "plan" in task_lower:  # noqa: SIM102
                # High reasoning models preferred for planning
                if "reasoning" in model.capabilities or "30b" in model.id or "70b" in model.id:
                    score += 20

            if "code" in required_caps or "code_gen" in required_caps:  # noqa: SIM102
                # Code generation models preferred for implementation
                if "code" in model.capabilities or "coder" in model.id.lower():
                    score += 20

            scores.append(
                {
                    "model_id": model.id,
                    "model_name": model.name,
                    "score": score,
                    "capabilities": model.capabilities,
                    "capability_matches": cap_matches,
                    "memory_gb": model.memory_gb,
                    "context_len": model.context_len,
                }
            )

        # Sort by score descending
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

    def _select_best_model(self, scores: list[dict], preferred_capabilities: list[str] | None = None) -> str:
        """Select the best model from scored models."""
        if not scores:
            return ""

        # Filter by preferred capabilities if specified
        if preferred_capabilities:
            for pref_cap in preferred_capabilities:
                for s in scores:
                    if pref_cap in s.get("capability_matches", []):
                        return s["model_id"]

        # Return top scorer
        return scores[0]["model_id"]

    def _infer_required_capabilities(self, text: str) -> list[str]:
        """Infer required capabilities from text."""
        required = []

        for cap, keywords in self.capability_keywords.items():
            if any(kw in text for kw in keywords):
                required.append(cap)

        # Always add code_gen for implementation tasks
        if any(kw in text for kw in ["implement", "build", "create", "code"]) and "code_gen" not in required:
            required.append("code_gen")

        return required if required else ["code_gen", "chat"]

    def _check_token_limits(self, plan: Plan, models: list[Model]):
        """Check if plan fits within token limits."""
        model_map = {m.id: m for m in models}

        for task in plan.tasks:
            model = model_map.get(task.assigned_model_id)
            if not model:
                continue

            # Estimate input tokens
            input_text = " ".join([*task.inputs, task.description])
            estimated_input = len(input_text) // 4

            # Estimate output tokens (rough)
            output_text = " ".join(task.outputs)
            estimated_output = len(output_text) // 4

            total_estimate = estimated_input + estimated_output

            if total_estimate > model.context_len * 0.8:  # 80% of context
                plan.warnings.append(
                    f"Task {task.id} may exceed token limits for model {model.id}. "
                    f"Estimated: {total_estimate}, Context: {model.context_len}"
                )
