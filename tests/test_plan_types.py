"""Tests for vetinari/plan_types.py — Phase 7D"""
import json
import unittest
from vetinari.plan_types import (
    PlanStatus, SubtaskStatus, TaskDomain, PlanRiskLevel,
    DefinitionOfDone, DefinitionOfReady, TaskRationale,
    Subtask, Plan, PlanCandidate,
    PlanGenerationRequest, PlanApprovalRequest,
)


class TestEnums(unittest.TestCase):
    def test_plan_status_values(self):
        self.assertEqual(PlanStatus.DRAFT, "draft")
        self.assertEqual(PlanStatus.APPROVED, "approved")
        self.assertIn(PlanStatus.FAILED, list(PlanStatus))

    def test_subtask_status_values(self):
        self.assertEqual(SubtaskStatus.PENDING, "pending")
        self.assertIn(SubtaskStatus.COMPLETED, list(SubtaskStatus))

    def test_task_domain_values(self):
        self.assertEqual(TaskDomain.CODING, "coding")
        self.assertIn(TaskDomain.RESEARCH, list(TaskDomain))

    def test_plan_risk_level_values(self):
        self.assertEqual(PlanRiskLevel.LOW, "low")
        self.assertIn(PlanRiskLevel.CRITICAL, list(PlanRiskLevel))


class TestDefinitionOfDone(unittest.TestCase):
    def test_defaults(self):
        dod = DefinitionOfDone()
        self.assertEqual(dod.criteria, [])
        self.assertFalse(dod.auto_verify)

    def test_with_criteria(self):
        dod = DefinitionOfDone(criteria=["tests pass", "linted"], auto_verify=True)
        self.assertEqual(len(dod.criteria), 2)
        self.assertTrue(dod.auto_verify)


class TestDefinitionOfReady(unittest.TestCase):
    def test_defaults(self):
        dor = DefinitionOfReady()
        self.assertTrue(dor.blockers_removed)
        self.assertTrue(dor.dependencies_met)


class TestTaskRationale(unittest.TestCase):
    def test_creation(self):
        r = TaskRationale(reason="best fit", capability_match=0.9, cost_estimate=0.02)
        self.assertEqual(r.reason, "best fit")
        self.assertAlmostEqual(r.capability_match, 0.9)
        self.assertEqual(r.alternatives_considered, [])


class TestSubtask(unittest.TestCase):
    def test_auto_id(self):
        s1 = Subtask()
        s2 = Subtask()
        self.assertNotEqual(s1.subtask_id, s2.subtask_id)
        self.assertTrue(s1.subtask_id.startswith("subtask_"))

    def test_defaults(self):
        s = Subtask()
        self.assertEqual(s.status, SubtaskStatus.PENDING)
        self.assertEqual(s.domain, TaskDomain.GENERAL)
        self.assertEqual(s.depth, 0)

    def test_custom_fields(self):
        s = Subtask(description="Build feature", domain=TaskDomain.CODING,
                    depth=2, status=SubtaskStatus.IN_PROGRESS)
        self.assertEqual(s.description, "Build feature")
        self.assertEqual(s.domain, TaskDomain.CODING)

    def test_to_dict(self):
        s = Subtask(description="test")
        d = s.to_dict()
        self.assertIn("subtask_id", d)
        self.assertIn("description", d)
        self.assertIn("status", d)

    def test_to_dict_json_serialisable(self):
        s = Subtask(description="json test")
        d = s.to_dict()
        json.dumps(d)  # should not raise

    def test_from_dict(self):
        s = Subtask(description="roundtrip", domain=TaskDomain.CODING)
        d = s.to_dict()
        s2 = Subtask.from_dict(d)
        self.assertEqual(s2.subtask_id, s.subtask_id)
        self.assertEqual(s2.domain, TaskDomain.CODING)


class TestPlan(unittest.TestCase):
    def test_auto_id(self):
        p = Plan(goal="build something")
        self.assertTrue(p.plan_id.startswith("plan_"))

    def test_defaults(self):
        p = Plan(goal="test goal")
        self.assertEqual(p.status, PlanStatus.DRAFT)
        self.assertFalse(p.dry_run)
        self.assertEqual(p.subtasks, [])

    def test_to_dict(self):
        p = Plan(goal="test")
        d = p.to_dict()
        self.assertIn("plan_id", d)
        self.assertIn("goal", d)
        self.assertIn("status", d)

    def test_from_dict(self):
        p = Plan(goal="original")
        d = p.to_dict()
        p2 = Plan.from_dict(d)
        self.assertEqual(p2.plan_id, p.plan_id)
        self.assertEqual(p2.goal, "original")


class TestPlanCandidate(unittest.TestCase):
    def test_auto_plan_id(self):
        pc = PlanCandidate(justification="best approach", risk_score=0.3)
        self.assertTrue(pc.plan_id.startswith("plan_"))
        self.assertAlmostEqual(pc.risk_score, 0.3)

    def test_to_dict(self):
        pc = PlanCandidate(justification="ok", risk_score=0.5)
        d = pc.to_dict()
        self.assertIn("plan_id", d)
        self.assertIn("justification", d)
        self.assertIn("risk_score", d)

    def test_from_dict(self):
        pc = PlanCandidate(justification="from dict test", risk_score=0.2)
        d = pc.to_dict()
        pc2 = PlanCandidate.from_dict(d)
        self.assertEqual(pc2.plan_id, pc.plan_id)
        self.assertAlmostEqual(pc2.risk_score, 0.2)


class TestPlanGenerationRequest(unittest.TestCase):
    def test_defaults(self):
        req = PlanGenerationRequest(goal="build api")
        self.assertEqual(req.goal, "build api")
        self.assertEqual(req.plan_depth_cap, 16)
        self.assertEqual(req.max_candidates, 3)
        self.assertFalse(req.dry_run)

    def test_optional_constraints(self):
        req = PlanGenerationRequest(goal="g", constraints="No external APIs")
        self.assertEqual(req.constraints, "No external APIs")


class TestPlanApprovalRequest(unittest.TestCase):
    def test_approval(self):
        # PlanApprovalRequest has: plan_id, approved, approver, reason, ...
        req = PlanApprovalRequest(
            plan_id="p1",
            approved=True,
            approver="admin",
        )
        self.assertTrue(req.approved)
        self.assertEqual(req.approver, "admin")

    def test_rejection(self):
        req = PlanApprovalRequest(
            plan_id="p1",
            approved=False,
            approver="user",
            reason="Too risky",
        )
        self.assertFalse(req.approved)
        self.assertEqual(req.reason, "Too risky")


if __name__ == "__main__":
    unittest.main()
