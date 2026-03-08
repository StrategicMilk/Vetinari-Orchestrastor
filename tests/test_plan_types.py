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


class TestPlanEdgeCases(unittest.TestCase):
    def test_empty_goal_plan(self):
        """Plan with empty goal string is valid but goal is empty."""
        p = Plan(goal="")
        self.assertEqual(p.goal, "")
        self.assertEqual(p.status, PlanStatus.DRAFT)

    def test_plan_from_dict_invalid_status_raises(self):
        """from_dict with an unrecognised status string should raise ValueError."""
        p = Plan(goal="test")
        d = p.to_dict()
        d["status"] = "not_a_real_status"
        with self.assertRaises(ValueError):
            Plan.from_dict(d)

    def test_plan_from_dict_invalid_risk_level_raises(self):
        """from_dict with an unrecognised risk_level should raise ValueError."""
        p = Plan(goal="test")
        d = p.to_dict()
        d["risk_level"] = "extreme"
        with self.assertRaises(ValueError):
            Plan.from_dict(d)

    def test_plan_calculate_risk_score_no_subtasks(self):
        """Risk score on a plan with no subtasks should be 0.0."""
        p = Plan(goal="empty plan")
        score = p.calculate_risk_score()
        self.assertEqual(score, 0.0)

    def test_get_subtask_returns_none_for_missing_id(self):
        p = Plan(goal="goal")
        result = p.get_subtask("nonexistent_id")
        self.assertIsNone(result)


class TestSubtaskEdgeCases(unittest.TestCase):
    def test_from_dict_invalid_status_raises(self):
        s = Subtask(description="test")
        d = s.to_dict()
        d["status"] = "invalid_status"
        with self.assertRaises(ValueError):
            Subtask.from_dict(d)

    def test_from_dict_invalid_domain_raises(self):
        s = Subtask(description="test")
        d = s.to_dict()
        d["domain"] = "not_a_domain"
        with self.assertRaises(ValueError):
            Subtask.from_dict(d)

    def test_subtask_none_description_allowed(self):
        """Subtask description defaults to empty string, not None."""
        s = Subtask()
        self.assertEqual(s.description, "")

    def test_subtask_negative_depth(self):
        """Subtask allows negative depth (no validation guard)."""
        s = Subtask(depth=-1)
        self.assertEqual(s.depth, -1)


class TestPlanCandidateEdgeCases(unittest.TestCase):
    def test_from_dict_invalid_risk_level_raises(self):
        pc = PlanCandidate(justification="j", risk_score=0.1)
        d = pc.to_dict()
        d["risk_level"] = "unknown_level"
        with self.assertRaises(ValueError):
            PlanCandidate.from_dict(d)

    def test_zero_risk_score(self):
        pc = PlanCandidate(justification="safe", risk_score=0.0)
        self.assertEqual(pc.risk_score, 0.0)


class TestDefinitionOfDoneEdgeCases(unittest.TestCase):
    def test_empty_criteria_list(self):
        """DefinitionOfDone with explicit empty criteria list is valid."""
        dod = DefinitionOfDone(criteria=[])
        self.assertEqual(dod.criteria, [])

    def test_single_criterion(self):
        """DefinitionOfDone with one criterion stores it correctly."""
        dod = DefinitionOfDone(criteria=["all tests pass"])
        self.assertEqual(len(dod.criteria), 1)
        self.assertEqual(dod.criteria[0], "all tests pass")


if __name__ == "__main__":
    unittest.main()
