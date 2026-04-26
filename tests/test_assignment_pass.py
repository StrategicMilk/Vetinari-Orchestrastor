"""Behavioral tests for vetinari.planning.assignment_pass."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vetinari.planning.assignment_pass import execute_assignment_pass


class TestAssignmentPass:
    """Tests for the assignment pass module."""

    def test_auto_assign_updates_unassigned_subtasks(self):
        """Unassigned subtasks receive model+agent assignments and status updates."""
        subtasks = [
            SimpleNamespace(
                subtask_id="sub-1",
                description="Write code",
                agent_type="coding",
                assigned_agent="unassigned",
            ),
            SimpleNamespace(
                subtask_id="sub-2",
                description="Already handled",
                agent_type="analysis",
                assigned_agent="analysis",
                assigned_model_id="model-existing",
            ),
        ]
        router = MagicMock()
        router.select_model.return_value = "model-coder"
        subtask_tree = MagicMock()
        subtask_tree.get_all_subtasks.return_value = subtasks

        with (
            patch("vetinari.models.dynamic_model_router.get_dynamic_router", return_value=router),
            patch("vetinari.planning.subtask_tree.subtask_tree", subtask_tree),
        ):
            result = execute_assignment_pass("plan-1", auto_assign=True)

        subtask_tree.update_subtask.assert_called_once_with(
            "plan-1",
            "sub-1",
            {
                "assigned_agent": "coding",
                "assigned_model_id": "model-coder",
                "status": "assigned",
            },
        )
        assert result["assigned"] == 1
        assert result["skipped"] == 1
        assert result["errors"] == 0
        assert result["assignments"][0]["action"] == "assigned"
        assert result["assignments"][1]["action"] == "skipped"

    def test_recommendation_mode_does_not_mutate_subtasks(self):
        """auto_assign=False returns recommendations without writing updates."""
        subtasks = [
            SimpleNamespace(
                subtask_id="sub-1",
                description="Analyze logs",
                agent_type="analysis",
                assigned_agent="unassigned",
            ),
        ]
        router = MagicMock()
        router.select_model.return_value = "model-analyst"
        subtask_tree = MagicMock()
        subtask_tree.get_all_subtasks.return_value = subtasks

        with (
            patch("vetinari.models.dynamic_model_router.get_dynamic_router", return_value=router),
            patch("vetinari.planning.subtask_tree.subtask_tree", subtask_tree),
        ):
            result = execute_assignment_pass("plan-2", auto_assign=False)

        subtask_tree.update_subtask.assert_not_called()
        assert result["assigned"] == 0
        assert result["skipped"] == 0
        assert result["assignments"][0]["action"] == "recommended"
        assert result["assignments"][0]["model"] == "model-analyst"

    def test_assignment_failure_is_reported_without_aborting_plan(self):
        """A single router failure is captured in error_details and processing continues."""
        subtasks = [
            SimpleNamespace(
                subtask_id="sub-1",
                description="Bad route",
                agent_type="analysis",
                assigned_agent="unassigned",
            ),
            SimpleNamespace(
                subtask_id="sub-2",
                description="Good route",
                agent_type="coding",
                assigned_agent="unassigned",
            ),
        ]
        router = MagicMock()
        router.select_model.side_effect = [RuntimeError("router failed"), "model-ok"]
        subtask_tree = MagicMock()
        subtask_tree.get_all_subtasks.return_value = subtasks

        with (
            patch("vetinari.models.dynamic_model_router.get_dynamic_router", return_value=router),
            patch("vetinari.planning.subtask_tree.subtask_tree", subtask_tree),
        ):
            result = execute_assignment_pass("plan-3", auto_assign=True)

        assert result["errors"] == 1
        assert result["assigned"] == 1
        assert result["error_details"] == [{"subtask_id": "sub-1", "error": "router failed"}]
        assert result["assignments"][0]["subtask_id"] == "sub-2"
