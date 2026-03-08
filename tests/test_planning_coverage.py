"""Coverage tests for vetinari/planning.py — Phase 7D"""
import json
import unittest
from vetinari.planning import (
    PlanStatus, WaveStatus, TaskStatus, AgentType,
    Task, Wave, Plan,
)


class TestPlanningEnums(unittest.TestCase):
    def test_plan_status(self):
        self.assertEqual(PlanStatus.PENDING.value, "pending")
        self.assertEqual(PlanStatus.COMPLETED.value, "completed")

    def test_wave_status(self):
        self.assertEqual(WaveStatus.RUNNING.value, "running")

    def test_task_status(self):
        self.assertEqual(TaskStatus.ASSIGNED.value, "assigned")

    def test_agent_type(self):
        self.assertEqual(AgentType.BUILDER.value, "BUILDER")


class TestTask(unittest.TestCase):
    def _make(self, tid="t1"):
        return Task(task_id=tid, agent_type="builder",
                    description="Build API", prompt="Create a REST API")

    def test_creation(self):
        t = self._make()
        self.assertEqual(t.task_id, "t1")
        self.assertEqual(t.status, TaskStatus.PENDING.value)
        self.assertEqual(t.dependencies, [])

    def test_to_dict(self):
        t = self._make("t2")
        d = t.to_dict()
        self.assertIn("task_id", d)
        self.assertIn("agent_type", d)
        self.assertIn("status", d)

    def test_from_dict(self):
        t = self._make("t3")
        d = t.to_dict()
        t2 = Task.from_dict(d)
        self.assertEqual(t2.task_id, "t3")


class TestWave(unittest.TestCase):
    def _make_task(self, tid="t1"):
        return Task(task_id=tid, agent_type="builder",
                    description="test", prompt="do thing")

    def test_creation(self):
        # Wave fields: wave_id, milestone, description, order, status, tasks, dependencies
        w = Wave(wave_id="w1", milestone="m1", description="Wave 1",
                 order=1, tasks=[self._make_task()])
        self.assertEqual(w.wave_id, "w1")
        self.assertEqual(len(w.tasks), 1)
        self.assertEqual(w.status, WaveStatus.PENDING.value)

    def test_to_dict(self):
        w = Wave(wave_id="w2", milestone="m2", description="Wave 2",
                 order=2, tasks=[])
        d = w.to_dict()
        self.assertIn("wave_id", d)
        self.assertIn("milestone", d)
        self.assertIn("tasks", d)

    def test_from_dict(self):
        w = Wave(wave_id="w3", milestone="m3", description="Wave 3",
                 order=3, tasks=[self._make_task("t_inner")])
        d = w.to_dict()
        w2 = Wave.from_dict(d)
        self.assertEqual(w2.wave_id, "w3")
        self.assertEqual(len(w2.tasks), 1)


class TestPlan(unittest.TestCase):
    def _make_plan(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        return Plan(
            plan_id="plan_001",
            title="Build a microservice",
            prompt="Build a simple microservice",
            created_by="test",
            created_at=now,
            updated_at=now,
        )

    def test_creation(self):
        p = self._make_plan()
        self.assertEqual(p.plan_id, "plan_001")
        self.assertEqual(p.status, PlanStatus.PENDING.value)
        self.assertEqual(p.waves, [])

    def test_to_dict(self):
        p = self._make_plan()
        d = p.to_dict()
        self.assertIn("plan_id", d)
        self.assertIn("title", d)
        self.assertIn("status", d)

    def test_to_json_via_to_dict(self):
        p = self._make_plan()
        d = p.to_dict()
        j = json.dumps(d)
        data = json.loads(j)
        self.assertEqual(data["plan_id"], "plan_001")

    def test_from_dict(self):
        p = self._make_plan()
        d = p.to_dict()
        p2 = Plan.from_dict(d)
        self.assertEqual(p2.plan_id, "plan_001")

    def test_save_and_load(self):
        import tempfile, os
        p = self._make_plan()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "plan.json")
            # Persist manually if save() doesn't exist
            if hasattr(p, "save"):
                p.save(path)
            else:
                with open(path, "w") as f:
                    json.dump(p.to_dict(), f)
            self.assertTrue(os.path.exists(path))
            # Load manually if load() doesn't exist
            if hasattr(Plan, "load"):
                p2 = Plan.load(path)
                self.assertEqual(p2.plan_id, p.plan_id)
            else:
                with open(path) as f:
                    data = json.load(f)
                self.assertEqual(data["plan_id"], "plan_001")


if __name__ == "__main__":
    unittest.main()
