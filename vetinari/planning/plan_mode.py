"""Plan Mode — generates, evaluates, and approves agent execution plans.

This is the planning step of the request pipeline:
Intake → **Planning** → Execution → Quality Gate → Assembly.

``PlanModeEngine`` implements the plan-first orchestration pattern:
1. Generate plan candidates from goals (LLM-powered, falling back to templates).
2. Evaluate and rank candidates by risk score.
3. Auto-approve low-risk plans in dry-run mode.
4. Support manual approval for high-risk plans.
5. Execute approved plans with subtask tracking.

Template data (domain/agent subtask skeletons) lives in ``plan_templates``
to keep this module under the 550-line ceiling.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from vetinari.constants import MAX_TOKENS_PLAN_VARIANT
from vetinari.exceptions import PlanningError
from vetinari.memory import MemoryStore, get_memory_store
from vetinari.memory.unified import get_unified_memory_store  # noqa: F401 — re-exported for test patching
from vetinari.planning.plan_executor import _PlanExecutorMixin
from vetinari.planning.plan_templates import AGENT_TEMPLATES, DOMAIN_TEMPLATES
from vetinari.planning.plan_types import (
    DefinitionOfDone,
    DefinitionOfReady,
    Plan,
    PlanApprovalRequest,
    PlanCandidate,
    PlanGenerationRequest,
    PlanRiskLevel,
    PlanStatus,
    StatusEnum,
    Subtask,
    TaskDomain,
)

logger = logging.getLogger(__name__)

# -- Module-level config flags (read once at import time from environment) --
# Who reads: PlanModeEngine.__init__, get_plan_engine()
# Who writes: environment variables (before process start)
PLAN_MODE_DEFAULT = os.environ.get("PLAN_MODE_DEFAULT", "true").lower() in ("1", "true", "yes")
PLAN_MODE_ENABLE = os.environ.get("PLAN_MODE_ENABLE", "true").lower() in ("1", "true", "yes")
DRY_RUN_ENABLED = os.environ.get("DRY_RUN_ENABLED", "false").lower() in ("1", "true", "yes")
DRY_RUN_RISK_THRESHOLD = float(os.environ.get("DRY_RUN_RISK_THRESHOLD", "0.25"))
DEPTH_CAP = int(os.environ.get("PLAN_DEPTH_CAP", "16"))
MAX_CANDIDATES = int(os.environ.get("PLAN_MAX_CANDIDATES", "3"))


class PlanModeEngine(_PlanExecutorMixin):
    """Plan Mode Engine — generates, evaluates, and approves plans.

    This engine implements the plan-first orchestration pattern:
    1. Generate plan candidates from goals
    2. Evaluate and rank candidates
    3. Allow dry-run mode with auto-approval for low-risk plans
    4. Support manual approval for high-risk plans
    5. Execute approved plans with subtask tracking
    """

    def __init__(self, memory_store: MemoryStore | None = None):
        self.memory = memory_store or get_memory_store()
        self.plan_depth_cap = DEPTH_CAP
        self.max_candidates = MAX_CANDIDATES
        self.dry_run_risk_threshold = DRY_RUN_RISK_THRESHOLD

        self._domain_templates = self._load_domain_templates()
        self._agent_templates = self._load_agent_templates()

    def _load_domain_templates(self) -> dict[TaskDomain, list[dict]]:
        """Return domain-specific subtask template dicts from plan_templates module."""
        return dict(DOMAIN_TEMPLATES)

    def _load_agent_templates(self) -> dict[str, list[dict]]:
        """Return agent-specific subtask template dicts from plan_templates module."""
        return dict(AGENT_TEMPLATES)

    def generate_plan(self, request: PlanGenerationRequest) -> Plan:
        """Generate a plan from a goal.

        Creates multiple plan candidates, evaluates them, and returns a
        Plan object ready for approval or execution. Consults WorkflowLearner
        for domain hints when available.

        Args:
            request: PlanGenerationRequest containing the goal, constraints,
                     domain hint, and other planning parameters.

        Returns:
            A fully constructed Plan with ranked subtasks and risk metadata.
        """
        logger.info("Generating plan for goal: %s...", request.goal[:100])

        # Consult WorkflowLearner for recommendations before planning
        workflow_hints: dict[str, Any] = {}
        try:
            from vetinari.learning.workflow_learner import get_workflow_learner

            workflow_hints = get_workflow_learner().get_recommendations(request.goal)
            if workflow_hints.get("confidence", 0) > 0.5:
                logger.info(
                    "WorkflowLearner recommends domain=%s, depth=%s, agents=%s",
                    workflow_hints.get("domain"),
                    workflow_hints.get("recommended_depth"),
                    workflow_hints.get("preferred_agents"),
                )
        except Exception as e:
            logger.warning("WorkflowLearner not available: %s", e)

        plan = Plan(goal=request.goal, constraints=request.constraints, dry_run=request.dry_run, plan_candidates=[])

        domain = request.domain_hint or self._infer_domain(request.goal)

        candidates = self._generate_candidates(
            goal=request.goal,
            constraints=request.constraints,
            domain=domain,
            max_candidates=request.max_candidates,
            depth_cap=request.plan_depth_cap,
        )

        plan.plan_candidates = candidates

        if candidates:
            best_candidate = min(candidates, key=lambda c: c.risk_score)
            plan.chosen_plan_id = best_candidate.plan_id
            plan.plan_justification = best_candidate.justification
            plan.risk_score = best_candidate.risk_score
            plan.risk_level = best_candidate.risk_level
            plan.subtasks = self._create_subtasks_from_candidate(best_candidate, plan.plan_id)
            plan.dependencies = best_candidate.dependencies

        plan.calculate_risk_score()

        if request.dry_run:
            plan.status = PlanStatus.DRAFT
            if plan.risk_score <= self.dry_run_risk_threshold:
                plan.auto_approved = True
                plan.status = PlanStatus.APPROVED
                plan.approved_by = "system_auto"
                plan.approved_at = datetime.now(timezone.utc).isoformat()
        else:
            plan.status = PlanStatus.DRAFT

        # Plan explanation generation disabled (explain_agent removed in Phase 10)
        # Future: wire through Operations agent's creative_writing mode

        self._persist_plan(plan)

        logger.info(
            "Plan generated: %s, risk_score=%.2f, subtasks=%s, auto_approved=%s",
            plan.plan_id,
            plan.risk_score,
            len(plan.subtasks),
            plan.auto_approved,
        )

        return plan

    def _infer_domain(self, goal: str) -> TaskDomain:
        """Infer the domain from the goal text using keyword matching.

        Args:
            goal: Free-text goal description.

        Returns:
            The most likely TaskDomain for the goal.
        """
        goal_lower = goal.lower()

        if any(kw in goal_lower for kw in ["code", "implement", "build", "feature", "api", "function"]):
            return TaskDomain.CODING
        if any(kw in goal_lower for kw in ["etl", "data", "pipeline", "process", "transform"]):
            return TaskDomain.DATA_PROCESSING
        if any(kw in goal_lower for kw in ["infra", "deploy", "monitor", "logging", "ci/cd"]):
            return TaskDomain.INFRA
        if any(kw in goal_lower for kw in ["document", "docs", "write", "guide"]):
            return TaskDomain.DOCS
        if any(kw in goal_lower for kw in ["experiment", "model", "test", "benchmark", "evaluate"]):
            return TaskDomain.AI_EXPERIMENTS
        if any(kw in goal_lower for kw in ["research", "analyze", "study", "investigate"]):
            return TaskDomain.RESEARCH
        return TaskDomain.GENERAL

    def _generate_candidates(
        self,
        goal: str,
        constraints: str,
        domain: TaskDomain,
        max_candidates: int,
        depth_cap: int,
    ) -> list[PlanCandidate]:
        """Generate multiple plan candidates, using LLM when available.

        Args:
            goal: The goal description.
            constraints: Any constraints on the plan.
            domain: The inferred or specified task domain.
            max_candidates: Maximum number of candidates to generate.
            depth_cap: Maximum plan depth allowed.

        Returns:
            List of PlanCandidate objects sorted from lowest to highest risk.
        """
        templates = self._domain_templates.get(domain, self._domain_templates[TaskDomain.GENERAL])

        # Gather quality history for calibrated estimates
        quality_context = ""
        try:
            from vetinari.learning.quality_scorer import get_quality_scorer

            scorer = get_quality_scorer()
            for tpl in templates:
                task_type = tpl.get("task_type", domain.value) if isinstance(tpl, dict) else domain.value
                history = scorer.get_history(task_type=task_type)
                if history:
                    recent = history[:5]
                    avg_score = sum(h.overall_score for h in recent) / len(recent)
                    quality_context += f"\n- {task_type}: avg quality={avg_score:.2f} over {len(recent)} recent tasks"
        except Exception:  # Quality history is enrichment only; proceed without it
            logger.warning("Quality scorer history unavailable for domain %s", domain.value, exc_info=True)

        # Try LLM-powered candidate generation
        try:
            from vetinari.adapter_manager import get_adapter_manager
            from vetinari.adapters.base import InferenceRequest

            adapter = get_adapter_manager()
            quality_section = f"\n\nHistorical quality data:{quality_context}" if quality_context else ""
            prompt_text = (
                f"Generate {min(max_candidates, 3)} plan variants for this goal:\n"
                f"Goal: {goal}\n"
                f"Domain: {domain.value}\n"
                f"Constraints: {constraints or 'none'}\n"
                f"{quality_section}\n\n"
                f"For each variant, provide on a single line: summary|risk(0.0-1.0)|hours|cost_usd|subtask_count\n"
                f"Variant 1 should be conservative (low risk), variant 2 balanced, variant 3 aggressive (fast but riskier)."
            )
            request = InferenceRequest(
                model_id="",  # Let adapter pick first available
                prompt=prompt_text,
                system_prompt="You are a project planner. Output exactly the requested format, one variant per line.",
                max_tokens=MAX_TOKENS_PLAN_VARIANT,
            )
            response = adapter.infer(request)
            if hasattr(response, "status") and response.status == "error":
                logger.debug("LLM inference returned error: %s", getattr(response, "error", "unknown"))
                raise RuntimeError(getattr(response, "error", "inference error"))
            content = response.output.strip() if hasattr(response, "output") else ""
            if content:
                return self._parse_llm_candidates(content, goal, domain, depth_cap, templates, max_candidates)
        except Exception as e:
            logger.warning("LLM candidate generation unavailable, using fallback: %s", e)

        # Fallback: hardcoded candidate generation
        return self._generate_fallback_candidates(goal, domain, templates, max_candidates, depth_cap)

    def _parse_llm_candidates(
        self,
        content: str,
        goal: str,
        domain: TaskDomain,
        depth_cap: int,
        templates: list,
        max_candidates: int,
    ) -> list[PlanCandidate]:
        """Parse LLM output into PlanCandidate objects.

        Args:
            content: Raw LLM output with one pipe-delimited variant per line.
            goal: The original goal string.
            domain: The task domain.
            depth_cap: Maximum plan depth.
            templates: Domain subtask templates for fallback subtask count.
            max_candidates: Maximum candidates to produce.

        Returns:
            List of parsed PlanCandidate objects.
        """
        candidates = []
        for i, line in enumerate(content.strip().split("\n")):
            if i >= min(max_candidates, 3):
                break
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                try:
                    summary = parts[0]
                    risk = max(0.0, min(1.0, float(parts[1])))
                    hours = max(0.5, float(parts[2]))
                    cost = max(1.0, float(parts[3]))
                    subtasks = max(1, int(float(parts[4])))
                except (ValueError, IndexError):
                    logger.warning(
                        "Could not parse plan cost/subtask line — skipping malformed entry, variant may use fallback values"
                    )
                    continue
            else:
                # Couldn't parse — use fallback values for this variant
                summary = f"Plan variant {i + 1} for: {goal[:50]}..."
                risk = 0.15 + (i * 0.1)
                hours = 1.0 + i * 0.5
                cost = 10.0 * (1 + i * 0.3)
                subtasks = len(templates) + i * 2

            candidate = PlanCandidate(
                plan_id=f"plan_{uuid.uuid4().hex[:8]}",
                plan_version=1,
                summary=summary,
                description=f"Implementation plan for: {goal}",
                justification=f"LLM-analyzed {domain.value} plan variant",
                risk_score=risk,
                estimated_duration_seconds=hours * 3600.0,
                estimated_cost=cost,
                subtask_count=subtasks,
                max_depth=min(depth_cap, 3 + i),
                domains=[domain],
            )
            self._assign_risk_level(candidate)
            candidate.dependencies = self._generate_dependencies(subtasks)
            candidates.append(candidate)

        # If parsing failed entirely, fall back
        if not candidates:
            return self._generate_fallback_candidates(goal, domain, templates, max_candidates, depth_cap)
        return candidates

    def _generate_fallback_candidates(
        self,
        goal: str,
        domain: TaskDomain,
        templates: list,
        max_candidates: int,
        depth_cap: int,
    ) -> list[PlanCandidate]:
        """Generate candidates with hardcoded heuristics (no LLM required).

        Args:
            goal: The original goal string.
            domain: The task domain.
            templates: Domain subtask templates used for subtask count.
            max_candidates: Maximum candidates to generate.
            depth_cap: Maximum plan depth.

        Returns:
            List of PlanCandidate objects (up to min(max_candidates, 3)).
        """
        candidates = []
        for i in range(min(max_candidates, 3)):
            candidate = PlanCandidate(
                plan_id=f"plan_{uuid.uuid4().hex[:8]}",
                plan_version=1,
                summary=f"Plan variant {i + 1} for: {goal[:50]}...",
                description=f"Implementation plan for: {goal}",
                justification=f"Generated based on {domain.value} domain patterns",
                risk_score=0.15 + (i * 0.1),
                estimated_duration_seconds=3600.0 * (1 + i * 0.5),
                estimated_cost=10.0 * (1 + i * 0.3),
                subtask_count=len(templates) + i * 2,
                max_depth=min(depth_cap, 3 + i),
                domains=[domain],
            )
            self._assign_risk_level(candidate)
            candidate.dependencies = self._generate_dependencies(len(templates) + i * 2)
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _assign_risk_level(candidate: PlanCandidate) -> None:
        """Set risk_level based on risk_score thresholds.

        Args:
            candidate: PlanCandidate to update in place.
        """
        if candidate.risk_score >= 0.75:
            candidate.risk_level = PlanRiskLevel.CRITICAL
        elif candidate.risk_score >= 0.5:
            candidate.risk_level = PlanRiskLevel.HIGH
        elif candidate.risk_score >= 0.25:
            candidate.risk_level = PlanRiskLevel.MEDIUM
        else:
            candidate.risk_level = PlanRiskLevel.LOW

    def _generate_dependencies(self, subtask_count: int) -> dict[str, list[str]]:
        """Generate sequential dependency chains for a set of subtasks.

        Every third subtask depends on the one before it; others have no
        dependencies. This produces a branching plan structure suitable
        for parallel execution.

        Args:
            subtask_count: Number of subtasks to generate dependencies for.

        Returns:
            Dict mapping subtask IDs to their dependency ID lists.
        """
        deps = {}
        for i in range(subtask_count):
            task_id = f"subtask_{i:03d}"
            if i > 0 and i % 3 == 0:
                deps[task_id] = [f"subtask_{i - 1:03d}"]
            else:
                deps[task_id] = []
        return deps

    def _create_subtasks_from_candidate(self, candidate: PlanCandidate, plan_id: str) -> list[Subtask]:
        """Create Subtask objects from a plan candidate's domain templates.

        Args:
            candidate: The chosen plan candidate.
            plan_id: The parent plan ID.

        Returns:
            List of Subtask objects with time/cost estimates distributed
            evenly across the template steps.
        """
        subtasks = []

        domain = candidate.domains[0] if candidate.domains else TaskDomain.GENERAL
        templates = self._domain_templates.get(domain, [])

        n_templates = len(templates) if templates else 1
        for i, template in enumerate(templates):
            subtask = Subtask(
                subtask_id=f"subtask_{i:03d}",
                plan_id=plan_id,
                description=template.get("description", f"Task {i + 1}"),
                domain=template.get("domain", domain),
                depth=0,
                status=StatusEnum.PENDING,
                definition_of_done=template.get("definition_of_done", DefinitionOfDone()),
                definition_of_ready=template.get("definition_of_ready", DefinitionOfReady()),
                time_estimate_seconds=candidate.estimated_duration_seconds / n_templates,
                cost_estimate=candidate.estimated_cost / n_templates,
            )
            subtasks.append(subtask)

        return subtasks

    def _persist_plan(self, plan: Plan) -> bool:
        """Persist plan and all its subtasks to memory store.

        Args:
            plan: The plan to persist.

        Returns:
            True if the write succeeded, False otherwise.
        """
        plan_data = plan.to_dict()
        plan_data["plan_json"] = json.dumps(plan.to_dict())

        success = self.memory.write_plan_history(plan_data)
        if not success:
            return False

        for subtask in plan.subtasks:
            subtask_ok = self.memory.write_subtask_memory(subtask.to_dict())
            if not subtask_ok:
                logger.warning(
                    "Failed to persist subtask %s for plan %s — plan header written but subtask lost",
                    subtask.subtask_id,
                    plan.plan_id,
                )
                success = False

        return success

    def approve_plan(self, request: PlanApprovalRequest) -> Plan:
        """Record an approval or rejection decision for a plan.

        Updates the plan's status, approver, and timestamp in memory. For
        rejections the plan transitions to REJECTED with the optional
        rejection reason attached.

        Args:
            request: Approval request containing plan ID, decision, approver
                identity, and optional rejection reason.

        Returns:
            The updated Plan object reflecting the new status.

        Raises:
            PlanningError: If no plan with the given ID exists in history,
                or if the plan is already in a terminal state.
        """
        plan_data_list = self.memory.query_plan_history(plan_id=request.plan_id)

        if not plan_data_list:
            raise PlanningError(f"Plan not found: {request.plan_id}")

        plan = Plan.from_dict(plan_data_list[0])

        # Guard: cannot approve/reject plans in terminal states
        _terminal = {PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED, PlanStatus.REJECTED}
        if plan.status in _terminal:
            raise PlanningError(
                f"Cannot {'approve' if request.approved else 'reject'} plan {plan.plan_id}: "
                f"already in terminal state '{plan.status.value}'"
            )

        if request.approved:
            plan.status = PlanStatus.APPROVED
            plan.approved_by = request.approver
            plan.approved_at = datetime.now(timezone.utc).isoformat()
            plan.auto_approved = False
        else:
            plan.status = PlanStatus.REJECTED
            plan.plan_justification = request.reason

        plan.updated_at = datetime.now(timezone.utc).isoformat()

        self._persist_plan(plan)

        logger.info("Plan %s %s by %s", plan.plan_id, "approved" if request.approved else "rejected", request.approver)

        return plan

    def get_plan(self, plan_id: str) -> Plan | None:
        """Retrieve a plan by ID from memory.

        Args:
            plan_id: The plan identifier to look up.

        Returns:
            The Plan object, or None if not found.
        """
        plan_data_list = self.memory.query_plan_history(plan_id=plan_id)

        if not plan_data_list:
            return None

        plan = Plan.from_dict(plan_data_list[0])

        # PlanHistory only stores the header; subtasks live in SubtaskMemory.
        # Reload them if the header round-trip produced an empty subtask list.
        if not plan.subtasks:
            subtask_dicts = self.memory.query_subtasks(plan_id=plan_id)
            plan.subtasks = [Subtask.from_dict(s) for s in subtask_dicts]

        return plan

    def get_plan_history(self, goal_contains: str | None = None, limit: int = 10) -> list[dict]:
        """Get plan history from memory, optionally filtered by goal text.

        Args:
            goal_contains: Optional substring to filter by goal text.
            limit: Maximum number of plans to return.

        Returns:
            List of serialized plan dicts.
        """
        return self.memory.query_plan_history(goal_contains=goal_contains, limit=limit)

    def get_subtasks(self, plan_id: str) -> list[Subtask]:
        """Get all subtasks for a plan.

        Args:
            plan_id: The parent plan ID.

        Returns:
            List of Subtask objects for this plan.
        """
        subtask_data = self.memory.query_subtasks(plan_id=plan_id)
        return [Subtask.from_dict(s) for s in subtask_data]

    def update_subtask_status(
        self,
        plan_id: str,
        subtask_id: str,
        status: StatusEnum,
        outcome: str | None = None,
    ) -> bool:
        """Update a subtask's status and optionally record its outcome.

        Args:
            plan_id: The parent plan ID (used for context only; not filtered).
            subtask_id: The subtask ID to update.
            status: The new status to set.
            outcome: Optional outcome description to record.

        Returns:
            True if the update succeeded, False if the subtask was not found.
        """
        subtask_data_list = self.memory.query_subtasks(subtask_id=subtask_id)

        if not subtask_data_list:
            return False

        subtask_data = subtask_data_list[0]
        subtask_data["status"] = status.value
        if outcome:
            subtask_data["outcome"] = outcome

        subtask_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        return self.memory.write_subtask_memory(subtask_data)

    def calculate_plan_risk(self, plan: Plan) -> float:
        """Recalculate and return the risk score for a plan.

        Args:
            plan: The plan to score.

        Returns:
            Updated risk score (0.0 to 1.0).
        """
        return plan.calculate_risk_score()

    def is_low_risk(self, risk_score: float) -> bool:
        """Check if a risk score is below the threshold for auto-approval.

        Args:
            risk_score: The risk score to evaluate.

        Returns:
            True if the score is at or below the dry-run threshold.
        """
        return risk_score <= self.dry_run_risk_threshold


# -- Module-level singleton (double-checked locking pattern) --
# Who writes: get_plan_engine(), init_plan_engine()
# Who reads: get_plan_engine()
# Lock: _plan_engine_lock protects the check-then-assign in get_plan_engine()
_plan_engine: PlanModeEngine | None = None
_plan_engine_lock = threading.Lock()


def get_plan_engine() -> PlanModeEngine:
    """Get or create the global PlanModeEngine singleton.

    Uses double-checked locking so only one instance is created even
    under concurrent access.

    Resolves ``_plan_engine`` through ``sys.modules`` rather than the
    function's own ``__globals__`` dict so that test patches applied via
    ``unittest.mock.patch("vetinari.planning.plan_mode._plan_engine", ...)``
    are always visible regardless of which module-object snapshot the calling
    closure was captured from.

    Returns:
        The shared PlanModeEngine instance.
    """
    import sys as _sys

    _mod = _sys.modules[__name__]
    if _mod._plan_engine is None:
        with _plan_engine_lock:
            if _mod._plan_engine is None:
                _mod._plan_engine = PlanModeEngine()
    return _mod._plan_engine


def init_plan_engine(memory_store: MemoryStore | None = None) -> PlanModeEngine:
    """Replace the module-level PlanModeEngine singleton with a fresh instance.

    Intended for tests and startup code that needs a clean engine with a
    specific memory store.

    Args:
        memory_store: Optional MemoryStore for persisting plan data;
            uses the default if omitted.

    Returns:
        The newly created PlanModeEngine instance.
    """
    global _plan_engine
    _plan_engine = PlanModeEngine(memory_store=memory_store)  # noqa: VET111 - stateful fallback preserves legacy compatibility
    return _plan_engine
