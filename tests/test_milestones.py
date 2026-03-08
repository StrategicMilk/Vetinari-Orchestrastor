"""Tests for milestone checkpoint system."""

import pytest
from vetinari.orchestration.milestones import (
    MilestoneGranularity,
    MilestoneAction,
    MilestoneReached,
    MilestoneApproval,
    MilestonePolicy,
    MilestoneManager,
)
from vetinari.agents.contracts import Task, AgentType


class TestMilestonePolicy:
    def test_default_policy(self):
        policy = MilestonePolicy()
        assert policy.granularity == MilestoneGranularity.FEATURES

    def test_features_checks_milestone_flag(self):
        policy = MilestonePolicy(granularity=MilestoneGranularity.FEATURES)
        t1 = Task(id="t1", description="Normal task", is_milestone=False)
        t2 = Task(id="t2", description="Milestone task", is_milestone=True, milestone_name="Feature A")
        assert not policy.should_checkpoint(t1)
        assert policy.should_checkpoint(t2)

    def test_all_granularity(self):
        policy = MilestonePolicy(granularity=MilestoneGranularity.ALL)
        t = Task(id="t1", description="Any task")
        assert policy.should_checkpoint(t)

    def test_none_granularity(self):
        policy = MilestonePolicy(granularity=MilestoneGranularity.NONE)
        t = Task(id="t1", description="Milestone", is_milestone=True, requires_approval=True)
        assert not policy.should_checkpoint(t)

    def test_critical_only(self):
        policy = MilestonePolicy(granularity=MilestoneGranularity.CRITICAL)
        t1 = Task(id="t1", description="Normal milestone", is_milestone=True)
        t2 = Task(id="t2", description="Critical", requires_approval=True)
        assert not policy.should_checkpoint(t1)
        assert policy.should_checkpoint(t2)

    def test_skip_all(self):
        policy = MilestonePolicy(granularity=MilestoneGranularity.ALL, skip_all=True)
        t = Task(id="t1", description="Task")
        assert not policy.should_checkpoint(t)

    def test_auto_approve(self):
        policy = MilestonePolicy(auto_approve_quality_threshold=0.8)
        good = MilestoneReached(
            milestone_name="test", task_id="t1", task_description="done",
            completed_tasks=["t1"], quality_score=0.9,
        )
        bad = MilestoneReached(
            milestone_name="test", task_id="t1", task_description="done",
            completed_tasks=["t1"], quality_score=0.5,
        )
        assert policy.should_auto_approve(good)
        assert not policy.should_auto_approve(bad)

    def test_serialization(self):
        policy = MilestonePolicy(granularity=MilestoneGranularity.PHASES)
        d = policy.to_dict()
        restored = MilestonePolicy.from_dict(d)
        assert restored.granularity == MilestoneGranularity.PHASES


class TestMilestoneManager:
    def test_non_milestone_auto_approves(self):
        mgr = MilestoneManager()
        t = Task(id="t1", description="Regular task")
        result = type("R", (), {"metadata": {}, "success": True})()
        approval = mgr.check_and_wait(t, result, ["t1"])
        assert approval.action == MilestoneAction.APPROVE

    def test_milestone_with_callback(self):
        mgr = MilestoneManager(MilestonePolicy(auto_approve_on_success=False))
        mgr.set_approval_callback(lambda m: MilestoneApproval.approve())
        t = Task(id="t1", description="Feature", is_milestone=True, milestone_name="Auth")
        result = type("R", (), {"metadata": {}, "success": True})()
        approval = mgr.check_and_wait(t, result, ["t1"])
        assert approval.action == MilestoneAction.APPROVE

    def test_skip_remaining(self):
        mgr = MilestoneManager(MilestonePolicy(auto_approve_on_success=False))
        mgr.set_approval_callback(lambda m: MilestoneApproval.skip_remaining())
        t = Task(id="t1", description="Feature", is_milestone=True)
        result = type("R", (), {"metadata": {}, "success": True})()
        mgr.check_and_wait(t, result, ["t1"])
        assert mgr.policy.skip_all is True

    def test_history_recorded(self):
        mgr = MilestoneManager()
        t = Task(id="t1", description="Feature", is_milestone=True, milestone_name="Test")
        result = type("R", (), {"metadata": {"quality_score": 0.9}, "success": True})()
        mgr.check_and_wait(t, result, ["t1"])
        assert len(mgr.get_history()) == 1


class TestMilestoneReached:
    def test_summary(self):
        m = MilestoneReached(
            milestone_name="Auth Module",
            task_id="t5",
            task_description="Implement authentication",
            completed_tasks=["t1", "t2", "t3", "t4", "t5"],
            quality_score=0.85,
        )
        s = m.summary()
        assert "Auth Module" in s
        assert "0.8" in s  # quality shown

    def test_serialization(self):
        m = MilestoneReached(
            milestone_name="Test",
            task_id="t1",
            task_description="desc",
            completed_tasks=["t1"],
        )
        d = m.to_dict()
        assert d["milestone_name"] == "Test"


class TestMilestoneApproval:
    def test_factory_methods(self):
        assert MilestoneApproval.approve().action == MilestoneAction.APPROVE
        assert MilestoneApproval.revise("fix bugs").action == MilestoneAction.REVISE
        assert MilestoneApproval.revise("fix bugs").feedback == "fix bugs"
        assert MilestoneApproval.skip_remaining().action == MilestoneAction.SKIP_REMAINING
        assert MilestoneApproval.abort().action == MilestoneAction.ABORT
