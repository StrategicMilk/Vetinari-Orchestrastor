"""Tests for vetinari/plan_types.py — Phase 7D"""
import json
import unittest

from vetinari.plan_types import (
    DefinitionOfDone,
    DefinitionOfReady,
    Plan,
    PlanApprovalRequest,
    PlanCandidate,
    PlanGenerationRequest,
    PlanRiskLevel,
    PlanStatus,
    Subtask,
    SubtaskStatus,
    TaskDomain,
    TaskRationale,
)


class TestEnums(unittest.TestCase):
    def test_plan_status_values(self):
        assert PlanStatus.DRAFT.value == "draft"
        assert PlanStatus.APPROVED.value == "approved"
        assert PlanStatus.FAILED in list(PlanStatus)

    def test_subtask_status_values(self):
        assert SubtaskStatus.PENDING.value == "pending"
        assert SubtaskStatus.COMPLETED in list(SubtaskStatus)

    def test_task_domain_values(self):
        assert TaskDomain.CODING == "coding"
        assert TaskDomain.RESEARCH in list(TaskDomain)

    def test_plan_risk_level_values(self):
        assert PlanRiskLevel.LOW == "low"
        assert PlanRiskLevel.CRITICAL in list(PlanRiskLevel)


class TestDefinitionOfDone(unittest.TestCase):
    def test_defaults(self):
        dod = DefinitionOfDone()
        assert dod.criteria == []
        assert not dod.auto_verify

    def test_with_criteria(self):
        dod = DefinitionOfDone(criteria=["tests pass", "linted"], auto_verify=True)
        assert len(dod.criteria) == 2
        assert dod.auto_verify


class TestDefinitionOfReady(unittest.TestCase):
    def test_defaults(self):
        dor = DefinitionOfReady()
        assert dor.blockers_removed
        assert dor.dependencies_met


class TestTaskRationale(unittest.TestCase):
    def test_creation(self):
        r = TaskRationale(reason="best fit", capability_match=0.9, cost_estimate=0.02)
        assert r.reason == "best fit"
        self.assertAlmostEqual(r.capability_match, 0.9)
        assert r.alternatives_considered == []


class TestSubtask(unittest.TestCase):
    def test_auto_id(self):
        s1 = Subtask()
        s2 = Subtask()
        assert s1.subtask_id != s2.subtask_id
        assert s1.subtask_id.startswith("subtask_")

    def test_defaults(self):
        s = Subtask()
        assert s.status == SubtaskStatus.PENDING
        assert s.domain == TaskDomain.GENERAL
        assert s.depth == 0

    def test_custom_fields(self):
        s = Subtask(description="Build feature", domain=TaskDomain.CODING,
                    depth=2, status=SubtaskStatus.IN_PROGRESS)
        assert s.description == "Build feature"
        assert s.domain == TaskDomain.CODING

    def test_to_dict(self):
        s = Subtask(description="test")
        d = s.to_dict()
        assert "subtask_id" in d
        assert "description" in d
        assert "status" in d

    def test_to_dict_json_serialisable(self):
        s = Subtask(description="json test")
        d = s.to_dict()
        json.dumps(d)  # should not raise

    def test_from_dict(self):
        s = Subtask(description="roundtrip", domain=TaskDomain.CODING)
        d = s.to_dict()
        s2 = Subtask.from_dict(d)
        assert s2.subtask_id == s.subtask_id
        assert s2.domain == TaskDomain.CODING


class TestPlan(unittest.TestCase):
    def test_auto_id(self):
        p = Plan(goal="build something")
        assert p.plan_id.startswith("plan_")

    def test_defaults(self):
        p = Plan(goal="test goal")
        assert p.status == PlanStatus.DRAFT
        assert not p.dry_run
        assert p.subtasks == []

    def test_to_dict(self):
        p = Plan(goal="test")
        d = p.to_dict()
        assert "plan_id" in d
        assert "goal" in d
        assert "status" in d

    def test_from_dict(self):
        p = Plan(goal="original")
        d = p.to_dict()
        p2 = Plan.from_dict(d)
        assert p2.plan_id == p.plan_id
        assert p2.goal == "original"


class TestPlanCandidate(unittest.TestCase):
    def test_auto_plan_id(self):
        pc = PlanCandidate(justification="best approach", risk_score=0.3)
        assert pc.plan_id.startswith("plan_")
        self.assertAlmostEqual(pc.risk_score, 0.3)

    def test_to_dict(self):
        pc = PlanCandidate(justification="ok", risk_score=0.5)
        d = pc.to_dict()
        assert "plan_id" in d
        assert "justification" in d
        assert "risk_score" in d

    def test_from_dict(self):
        pc = PlanCandidate(justification="from dict test", risk_score=0.2)
        d = pc.to_dict()
        pc2 = PlanCandidate.from_dict(d)
        assert pc2.plan_id == pc.plan_id
        self.assertAlmostEqual(pc2.risk_score, 0.2)


class TestPlanGenerationRequest(unittest.TestCase):
    def test_defaults(self):
        req = PlanGenerationRequest(goal="build api")
        assert req.goal == "build api"
        assert req.plan_depth_cap == 16
        assert req.max_candidates == 3
        assert not req.dry_run

    def test_optional_constraints(self):
        req = PlanGenerationRequest(goal="g", constraints="No external APIs")
        assert req.constraints == "No external APIs"


class TestPlanApprovalRequest(unittest.TestCase):
    def test_approval(self):
        # PlanApprovalRequest has: plan_id, approved, approver, reason, ...
        req = PlanApprovalRequest(
            plan_id="p1",
            approved=True,
            approver="admin",
        )
        assert req.approved
        assert req.approver == "admin"

    def test_rejection(self):
        req = PlanApprovalRequest(
            plan_id="p1",
            approved=False,
            approver="user",
            reason="Too risky",
        )
        assert not req.approved
        assert req.reason == "Too risky"


if __name__ == "__main__":
    unittest.main()
