"""
Tests for plan approval gating functionality.

These tests verify:
1. Coding tasks require approval in Plan mode
2. Non-coding tasks do not require approval
3. Auto-approval for low-risk dry-run tasks
4. Approval logging to memory
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestApprovalGating:
    """Tests for approval gating logic."""

    def test_coding_task_requires_approval_in_plan_mode(self):
        """Test that coding tasks require approval in Plan mode."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import StatusEnum, Subtask, TaskDomain

        engine = PlanModeEngine()

        coding_task = Subtask(
            subtask_id="test_001",
            description="Implement login API",
            domain=TaskDomain.CODING,
            status=StatusEnum.PENDING,
        )

        requires = engine.requires_approval(coding_task, plan_mode=True)
        assert requires is True

    def test_coding_task_no_approval_in_build_mode(self):
        """Test that coding tasks do NOT require approval in Build mode."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import StatusEnum, Subtask, TaskDomain

        engine = PlanModeEngine()

        coding_task = Subtask(
            subtask_id="test_001",
            description="Implement login API",
            domain=TaskDomain.CODING,
            status=StatusEnum.PENDING,
        )

        requires = engine.requires_approval(coding_task, plan_mode=False)
        assert requires is False

    def test_docs_task_no_approval(self):
        """Test that docs tasks do not require approval."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import StatusEnum, Subtask, TaskDomain

        engine = PlanModeEngine()

        docs_task = Subtask(
            subtask_id="test_002",
            description="Write API documentation",
            domain=TaskDomain.DOCS,
            status=StatusEnum.PENDING,
        )

        requires = engine.requires_approval(docs_task, plan_mode=True)
        assert requires is False

    def test_infra_task_no_approval(self):
        """Test that infrastructure tasks do not require approval."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import StatusEnum, Subtask, TaskDomain

        engine = PlanModeEngine()

        infra_task = Subtask(
            subtask_id="test_003",
            description="Setup CI/CD pipeline",
            domain=TaskDomain.INFRA,
            status=StatusEnum.PENDING,
        )

        requires = engine.requires_approval(infra_task, plan_mode=True)
        assert requires is False

    def test_check_subtask_approval_required(self):
        """Test the check_subtask_approval_required method."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import Plan, StatusEnum, Subtask, TaskDomain
        from vetinari.types import PlanStatus

        engine = PlanModeEngine()

        plan = Plan(plan_id="test_plan", goal="Build a web app", status=PlanStatus.DRAFT)

        coding_task = Subtask(
            subtask_id="test_001", description="Implement API", domain=TaskDomain.CODING, status=StatusEnum.PENDING
        )
        plan.subtasks.append(coding_task)

        result = engine.check_subtask_approval_required(plan, "test_001", plan_mode=True)

        assert result["requires_approval"] is True
        assert result["domain"] == "coding"
        assert result["plan_mode"] is True

    def test_check_subtask_approval_not_found(self):
        """Test approval check for non-existent subtask."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import Plan
        from vetinari.types import PlanStatus

        engine = PlanModeEngine()

        plan = Plan(plan_id="test_plan", goal="Build a web app", status=PlanStatus.DRAFT)

        result = engine.check_subtask_approval_required(plan, "nonexistent", plan_mode=True)

        assert result["requires_approval"] is False
        assert "error" in result

    def test_is_low_risk(self):
        """Test the is_low_risk method."""
        from vetinari.planning.plan_mode import PlanModeEngine

        engine = PlanModeEngine()

        assert engine.is_low_risk(0.1) is True
        assert engine.is_low_risk(0.25) is True
        assert engine.is_low_risk(0.26) is False
        assert engine.is_low_risk(0.5) is False

    def test_auto_approve_low_risk_dry_run(self):
        """Test auto-approval for low-risk dry-run tasks."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import Plan, StatusEnum, Subtask, TaskDomain
        from vetinari.types import PlanStatus

        engine = PlanModeEngine()

        plan = Plan(
            plan_id="test_plan", goal="Build a simple web app", status=PlanStatus.DRAFT, dry_run=True, risk_score=0.15
        )

        docs_task = Subtask(
            subtask_id="test_001", description="Write documentation", domain=TaskDomain.DOCS, status=StatusEnum.PENDING
        )

        auto_approved = engine.auto_approve_if_low_risk(plan, docs_task)
        assert auto_approved is True

    def test_no_auto_approve_high_risk(self):
        """Test that high-risk tasks are not auto-approved."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import Plan, StatusEnum, Subtask, TaskDomain
        from vetinari.types import PlanStatus

        engine = PlanModeEngine()

        plan = Plan(
            plan_id="test_plan", goal="Build a complex system", status=PlanStatus.DRAFT, dry_run=True, risk_score=0.5
        )

        coding_task = Subtask(
            subtask_id="test_001",
            description="Implement core functionality",
            domain=TaskDomain.CODING,
            status=StatusEnum.PENDING,
        )

        auto_approved = engine.auto_approve_if_low_risk(plan, coding_task)
        assert auto_approved is False

    def test_no_auto_approve_non_dry_run(self):
        """Test that non-dry-run tasks are not auto-approved."""
        from vetinari.planning.plan_mode import PlanModeEngine
        from vetinari.planning.plan_types import Plan, StatusEnum, Subtask, TaskDomain
        from vetinari.types import PlanStatus

        engine = PlanModeEngine()

        plan = Plan(
            plan_id="test_plan", goal="Build a web app", status=PlanStatus.DRAFT, dry_run=False, risk_score=0.15
        )

        docs_task = Subtask(
            subtask_id="test_001", description="Write documentation", domain=TaskDomain.DOCS, status=StatusEnum.PENDING
        )

        auto_approved = engine.auto_approve_if_low_risk(plan, docs_task)
        assert auto_approved is False


class TestApprovalLogging:
    """Tests for approval logging to memory."""

    @patch("vetinari.memory.unified.get_unified_memory_store")
    def test_log_approval_decision(self, mock_get_store):
        """Test logging an approval decision."""
        from vetinari.planning.plan_mode import PlanModeEngine

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        engine = PlanModeEngine()

        result = engine.log_approval_decision(
            plan_id="test_plan",
            subtask_id="test_001",
            approved=True,
            approver="test_user",
            reason="Looks good",
            risk_score=0.2,
        )

        assert result is True
        mock_store.remember.assert_called_once()

    @patch("vetinari.memory.unified.get_unified_memory_store")
    def test_log_rejection_decision(self, mock_get_store):
        """Test logging a rejection decision."""
        from vetinari.planning.plan_mode import PlanModeEngine

        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        engine = PlanModeEngine()

        result = engine.log_approval_decision(
            plan_id="test_plan",
            subtask_id="test_001",
            approved=False,
            approver="test_user",
            reason="Needs more work",
            risk_score=0.6,
        )

        assert result is True
        mock_store.remember.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
