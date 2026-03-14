"""Assignment Pass.

===============
Executes the model/agent assignment pass for all pending subtasks in a plan.
Uses the DynamicModelRouter to assign the best available model to each task.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def execute_assignment_pass(
    plan_id: str,
    auto_assign: bool = True,
) -> dict[str, Any]:
    """Execute the assignment pass for a plan.

    For each unassigned subtask in the plan, selects the best model+agent
    using DynamicModelRouter and updates the subtask record.

    Args:
        plan_id: The plan to process
        auto_assign: If True, automatically apply assignments. If False,
                     return recommendations without applying them.

    Returns:
        dict with assignment results per subtask
    """
    try:
        from vetinari.models.dynamic_model_router import get_dynamic_router
        from vetinari.planning.subtask_tree import subtask_tree

        subtasks = subtask_tree.get_all_subtasks(plan_id)
        router = get_dynamic_router()

        assignments = []
        errors = []

        for st in subtasks:
            if st.assigned_agent and st.assigned_agent != "unassigned":
                assignments.append(
                    {
                        "subtask_id": st.subtask_id,
                        "description": st.description,
                        "agent_type": st.agent_type,
                        "assigned_agent": st.assigned_agent,
                        "model": getattr(st, "assigned_model", None),
                        "action": "skipped",
                    }
                )
                continue

            try:
                task_type = st.agent_type or "general"
                model_id = router.select_model(
                    task_type=task_type.lower(),
                    task_description=st.description,
                )
                if auto_assign:
                    subtask_tree.update_subtask(
                        plan_id,
                        st.subtask_id,
                        {
                            "assigned_agent": task_type,
                            "assigned_model": model_id,
                            "status": "assigned",
                        },
                    )
                assignments.append(
                    {
                        "subtask_id": st.subtask_id,
                        "description": st.description,
                        "agent_type": task_type,
                        "assigned_agent": task_type,
                        "model": model_id,
                        "action": "assigned" if auto_assign else "recommended",
                    }
                )
            except Exception as e:
                logger.warning("Assignment failed for subtask %s: %s", st.subtask_id, e)
                errors.append({"subtask_id": st.subtask_id, "error": str(e)})

        return {
            "plan_id": plan_id,
            "total": len(subtasks),
            "assigned": sum(1 for a in assignments if a.get("action") == "assigned"),
            "skipped": sum(1 for a in assignments if a.get("action") == "skipped"),
            "errors": len(errors),
            "assignments": assignments,
            "error_details": errors,
            "auto_assign": auto_assign,
        }

    except Exception as e:
        logger.error("execute_assignment_pass failed: %s", e)
        return {"plan_id": plan_id, "error": str(e), "assignments": []}
