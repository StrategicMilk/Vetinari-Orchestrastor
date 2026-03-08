"""Comprehensive tests for vetinari/planning.py — legacy wave-based plan management."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from vetinari.planning import (
    WaveStatus,
    Task,
    Wave,
    Plan,
    PlanManager,
    get_plan_manager,
    _LazyPlanManager,
    plan_manager,
)
from vetinari.types import PlanStatus, TaskStatus, AgentType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(tid="t1", agent_type="builder", description="desc", prompt="do it",
               status=None, dependencies=None):
    t = Task(
        task_id=tid,
        agent_type=agent_type,
        description=description,
        prompt=prompt,
        dependencies=dependencies or [],
    )
    if status is not None:
        t.status = status
    return t


def _make_wave(wave_id="w1", milestone="M1", description="Wave 1", order=1,
               tasks=None, status=None, dependencies=None):
    w = Wave(
        wave_id=wave_id,
        milestone=milestone,
        description=description,
        order=order,
        tasks=tasks or [],
        dependencies=dependencies or [],
    )
    if status is not None:
        w.status = status
    return w


def _make_plan(plan_id="plan_test", title="Test Plan", prompt="Build it",
               created_by="tester", waves=None):
    now = datetime.now().isoformat()
    return Plan(
        plan_id=plan_id,
        title=title,
        prompt=prompt,
        created_by=created_by,
        created_at=now,
        updated_at=now,
        waves=waves or [],
    )


def _waves_data():
    """Minimal waves_data payload for PlanManager.create_plan."""
    return [
        {
            "milestone": "Phase 1",
            "description": "First wave",
            "tasks": [
                {"agent_type": "builder", "description": "Task A", "prompt": "Do A"},
                {"agent_type": "explorer", "description": "Task B", "prompt": "Do B"},
            ],
        },
        {
            "milestone": "Phase 2",
            "description": "Second wave",
            "tasks": [
                {"agent_type": "oracle", "description": "Task C", "prompt": "Do C"},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# WaveStatus enum
# ---------------------------------------------------------------------------

class TestWaveStatus(unittest.TestCase):
    def test_pending_value(self):
        self.assertEqual(WaveStatus.PENDING.value, "pending")

    def test_running_value(self):
        self.assertEqual(WaveStatus.RUNNING.value, "running")

    def test_completed_value(self):
        self.assertEqual(WaveStatus.COMPLETED.value, "completed")

    def test_failed_value(self):
        self.assertEqual(WaveStatus.FAILED.value, "failed")

    def test_blocked_value(self):
        self.assertEqual(WaveStatus.BLOCKED.value, "blocked")

    def test_all_members(self):
        members = {s.value for s in WaveStatus}
        self.assertEqual(members, {"pending", "running", "completed", "failed", "blocked"})

    def test_from_value(self):
        self.assertEqual(WaveStatus("running"), WaveStatus.RUNNING)

    def test_inequality(self):
        self.assertNotEqual(WaveStatus.PENDING, WaveStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

class TestTaskCreation(unittest.TestCase):
    def test_required_fields(self):
        t = _make_task()
        self.assertEqual(t.task_id, "t1")
        self.assertEqual(t.agent_type, "builder")
        self.assertEqual(t.description, "desc")
        self.assertEqual(t.prompt, "do it")

    def test_default_status(self):
        t = _make_task()
        self.assertEqual(t.status, TaskStatus.PENDING.value)

    def test_default_dependencies(self):
        t = _make_task()
        self.assertEqual(t.dependencies, [])

    def test_default_retry_count(self):
        t = _make_task()
        self.assertEqual(t.retry_count, 0)

    def test_default_priority(self):
        t = _make_task()
        self.assertEqual(t.priority, 5)

    def test_default_estimated_effort(self):
        t = _make_task()
        self.assertAlmostEqual(t.estimated_effort, 1.0)

    def test_default_depth(self):
        t = _make_task()
        self.assertEqual(t.depth, 0)

    def test_default_max_depth(self):
        t = _make_task()
        self.assertEqual(t.max_depth, 14)

    def test_default_max_depth_override(self):
        t = _make_task()
        self.assertEqual(t.max_depth_override, 0)

    def test_default_subtasks_empty(self):
        t = _make_task()
        self.assertEqual(t.subtasks, [])

    def test_default_dod_level(self):
        t = _make_task()
        self.assertEqual(t.dod_level, "Standard")

    def test_default_dor_level(self):
        t = _make_task()
        self.assertEqual(t.dor_level, "Standard")

    def test_default_wave_id(self):
        t = _make_task()
        self.assertEqual(t.wave_id, "")

    def test_with_dependencies(self):
        t = _make_task(dependencies=["dep1", "dep2"])
        self.assertEqual(t.dependencies, ["dep1", "dep2"])

    def test_custom_priority(self):
        t = Task(task_id="x", agent_type="builder", description="d", prompt="p", priority=9)
        self.assertEqual(t.priority, 9)

    def test_with_subtasks(self):
        sub = _make_task("sub1")
        t = Task(task_id="parent", agent_type="builder", description="d", prompt="p",
                 subtasks=[sub])
        self.assertEqual(len(t.subtasks), 1)


class TestTaskToDict(unittest.TestCase):
    def setUp(self):
        self.task = _make_task("t1")

    def test_to_dict_has_task_id(self):
        d = self.task.to_dict()
        self.assertEqual(d["task_id"], "t1")

    def test_to_dict_has_agent_type(self):
        d = self.task.to_dict()
        self.assertEqual(d["agent_type"], "builder")

    def test_to_dict_has_status(self):
        d = self.task.to_dict()
        self.assertIn("status", d)

    def test_to_dict_has_dependencies(self):
        d = self.task.to_dict()
        self.assertIsInstance(d["dependencies"], list)

    def test_to_dict_has_subtasks(self):
        d = self.task.to_dict()
        self.assertIsInstance(d["subtasks"], list)

    def test_to_dict_subtask_nested(self):
        sub = _make_task("sub1")
        t = Task(task_id="parent", agent_type="builder", description="d", prompt="p",
                 subtasks=[sub])
        d = t.to_dict()
        self.assertEqual(len(d["subtasks"]), 1)
        self.assertEqual(d["subtasks"][0]["task_id"], "sub1")

    def test_to_dict_is_json_serializable(self):
        d = self.task.to_dict()
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)

    def test_to_dict_has_all_fields(self):
        d = self.task.to_dict()
        required = {"task_id", "agent_type", "description", "prompt", "status",
                    "dependencies", "assigned_agent", "result", "error",
                    "planned_start", "planned_end", "actual_start", "actual_end",
                    "retry_count", "priority", "estimated_effort", "parent_id",
                    "depth", "max_depth", "max_depth_override", "subtasks",
                    "decomposition_seed", "dod_level", "dor_level", "wave_id"}
        self.assertTrue(required.issubset(d.keys()))


class TestTaskFromDict(unittest.TestCase):
    def test_roundtrip(self):
        t = _make_task("t99")
        t.priority = 8
        t.retry_count = 2
        d = t.to_dict()
        t2 = Task.from_dict(d)
        self.assertEqual(t2.task_id, "t99")
        self.assertEqual(t2.priority, 8)
        self.assertEqual(t2.retry_count, 2)

    def test_from_empty_dict_defaults(self):
        t = Task.from_dict({})
        self.assertEqual(t.task_id, "")
        self.assertEqual(t.agent_type, "builder")
        self.assertEqual(t.status, TaskStatus.PENDING.value)

    def test_from_dict_with_subtasks(self):
        sub_dict = _make_task("sub1").to_dict()
        data = _make_task("parent").to_dict()
        data["subtasks"] = [sub_dict]
        t = Task.from_dict(data)
        self.assertEqual(len(t.subtasks), 1)
        self.assertIsInstance(t.subtasks[0], Task)

    def test_from_dict_preserves_dependencies(self):
        t = _make_task(dependencies=["dep_a", "dep_b"])
        t2 = Task.from_dict(t.to_dict())
        self.assertEqual(t2.dependencies, ["dep_a", "dep_b"])

    def test_from_dict_dod_level(self):
        data = _make_task().to_dict()
        data["dod_level"] = "High"
        t = Task.from_dict(data)
        self.assertEqual(t.dod_level, "High")


# ---------------------------------------------------------------------------
# Wave dataclass
# ---------------------------------------------------------------------------

class TestWaveCreation(unittest.TestCase):
    def test_basic_fields(self):
        w = _make_wave()
        self.assertEqual(w.wave_id, "w1")
        self.assertEqual(w.milestone, "M1")
        self.assertEqual(w.order, 1)

    def test_default_status(self):
        w = _make_wave()
        self.assertEqual(w.status, WaveStatus.PENDING.value)

    def test_default_tasks_empty(self):
        w = _make_wave()
        self.assertEqual(w.tasks, [])

    def test_with_tasks(self):
        t = _make_task()
        w = _make_wave(tasks=[t])
        self.assertEqual(len(w.tasks), 1)


class TestWaveProperties(unittest.TestCase):
    def test_total_count_empty(self):
        w = _make_wave()
        self.assertEqual(w.total_count, 0)

    def test_total_count_with_tasks(self):
        tasks = [_make_task(f"t{i}") for i in range(5)]
        w = _make_wave(tasks=tasks)
        self.assertEqual(w.total_count, 5)

    def test_completed_count_none_complete(self):
        tasks = [_make_task(f"t{i}") for i in range(3)]
        w = _make_wave(tasks=tasks)
        self.assertEqual(w.completed_count, 0)

    def test_completed_count_all_complete(self):
        tasks = [_make_task(f"t{i}", status=TaskStatus.COMPLETED.value) for i in range(3)]
        w = _make_wave(tasks=tasks)
        self.assertEqual(w.completed_count, 3)

    def test_completed_count_partial(self):
        tasks = [
            _make_task("t1", status=TaskStatus.COMPLETED.value),
            _make_task("t2", status=TaskStatus.PENDING.value),
            _make_task("t3", status=TaskStatus.COMPLETED.value),
        ]
        w = _make_wave(tasks=tasks)
        self.assertEqual(w.completed_count, 2)


class TestWaveToDict(unittest.TestCase):
    def test_to_dict_has_wave_id(self):
        w = _make_wave()
        d = w.to_dict()
        self.assertEqual(d["wave_id"], "w1")

    def test_to_dict_has_tasks_list(self):
        t = _make_task()
        w = _make_wave(tasks=[t])
        d = w.to_dict()
        self.assertIsInstance(d["tasks"], list)
        self.assertEqual(len(d["tasks"]), 1)

    def test_to_dict_json_serializable(self):
        w = _make_wave(tasks=[_make_task()])
        json.dumps(w.to_dict())

    def test_from_dict_roundtrip(self):
        t = _make_task("inner")
        w = _make_wave("w99", tasks=[t])
        d = w.to_dict()
        w2 = Wave.from_dict(d)
        self.assertEqual(w2.wave_id, "w99")
        self.assertEqual(len(w2.tasks), 1)

    def test_from_dict_empty(self):
        w = Wave.from_dict({})
        self.assertEqual(w.wave_id, "")
        self.assertEqual(w.order, 1)
        self.assertEqual(w.status, WaveStatus.PENDING.value)


# ---------------------------------------------------------------------------
# Plan dataclass
# ---------------------------------------------------------------------------

class TestPlanCreation(unittest.TestCase):
    def test_basic_fields(self):
        p = _make_plan()
        self.assertEqual(p.plan_id, "plan_test")
        self.assertEqual(p.title, "Test Plan")
        self.assertEqual(p.created_by, "tester")

    def test_default_status(self):
        p = _make_plan()
        self.assertEqual(p.status, PlanStatus.PENDING.value)

    def test_default_waves_empty(self):
        p = _make_plan()
        self.assertEqual(p.waves, [])

    def test_default_template_version(self):
        p = _make_plan()
        self.assertEqual(p.template_version, "v1")

    def test_default_seed_rate(self):
        p = _make_plan()
        self.assertEqual(p.seed_rate, 2)

    def test_default_adr_history(self):
        p = _make_plan()
        self.assertEqual(p.adr_history, [])


class TestPlanProperties(unittest.TestCase):
    def _plan_with_waves(self):
        t1 = _make_task("t1")
        t2 = _make_task("t2", status=TaskStatus.COMPLETED.value)
        w1 = _make_wave("w1", tasks=[t1, t2])
        t3 = _make_task("t3", status=TaskStatus.COMPLETED.value)
        w2 = _make_wave("w2", tasks=[t3])
        return _make_plan(waves=[w1, w2])

    def test_total_tasks(self):
        p = self._plan_with_waves()
        self.assertEqual(p.total_tasks, 3)

    def test_total_tasks_no_waves(self):
        p = _make_plan()
        self.assertEqual(p.total_tasks, 0)

    def test_completed_tasks(self):
        p = self._plan_with_waves()
        self.assertEqual(p.completed_tasks, 2)

    def test_progress_percent_zero(self):
        p = _make_plan()
        self.assertEqual(p.progress_percent, 0.0)

    def test_progress_percent_partial(self):
        p = self._plan_with_waves()
        # 2 of 3 completed
        self.assertAlmostEqual(p.progress_percent, 66.7)

    def test_progress_percent_full(self):
        t1 = _make_task("t1", status=TaskStatus.COMPLETED.value)
        w1 = _make_wave("w1", tasks=[t1])
        p = _make_plan(waves=[w1])
        self.assertAlmostEqual(p.progress_percent, 100.0)

    def test_current_wave_none(self):
        p = _make_plan()
        self.assertIsNone(p.current_wave)

    def test_current_wave_running(self):
        w1 = _make_wave("w1", status=WaveStatus.COMPLETED.value)
        w2 = _make_wave("w2", status=WaveStatus.RUNNING.value)
        p = _make_plan(waves=[w1, w2])
        self.assertEqual(p.current_wave.wave_id, "w2")

    def test_effective_max_depth_default(self):
        p = _make_plan()
        self.assertEqual(p.effective_max_depth, 14)

    def test_effective_max_depth_override_low(self):
        p = _make_plan()
        p.max_depth_override = 10  # below 12 — clamps to 12
        self.assertEqual(p.effective_max_depth, 12)

    def test_effective_max_depth_override_high(self):
        p = _make_plan()
        p.max_depth_override = 20  # above 16 — clamps to 16
        self.assertEqual(p.effective_max_depth, 16)

    def test_effective_max_depth_override_valid(self):
        p = _make_plan()
        p.max_depth_override = 13
        self.assertEqual(p.effective_max_depth, 13)

    def test_effective_max_depth_override_zero(self):
        p = _make_plan()
        p.max_depth_override = 0  # 0 means no override
        self.assertEqual(p.effective_max_depth, 14)


class TestPlanAddAdr(unittest.TestCase):
    def test_add_adr_appends(self):
        p = _make_plan()
        p.add_adr("adr-001", "Use SQLite", "Need persistence", "SQLite chosen")
        self.assertEqual(len(p.adr_history), 1)

    def test_add_adr_returns_dict(self):
        p = _make_plan()
        result = p.add_adr("adr-001", "Use SQLite", "Need persistence", "SQLite chosen")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["adr_id"], "adr-001")
        self.assertEqual(result["title"], "Use SQLite")

    def test_add_adr_default_status_proposed(self):
        p = _make_plan()
        result = p.add_adr("adr-002", "title", "ctx", "decision")
        self.assertEqual(result["status"], "proposed")

    def test_add_adr_custom_status(self):
        p = _make_plan()
        result = p.add_adr("adr-003", "title", "ctx", "decision", status="accepted")
        self.assertEqual(result["status"], "accepted")

    def test_add_adr_has_created_at(self):
        p = _make_plan()
        result = p.add_adr("adr-004", "title", "ctx", "decision")
        self.assertIn("created_at", result)

    def test_add_multiple_adrs(self):
        p = _make_plan()
        p.add_adr("adr-1", "T1", "C1", "D1")
        p.add_adr("adr-2", "T2", "C2", "D2")
        self.assertEqual(len(p.adr_history), 2)


class TestPlanToFromDict(unittest.TestCase):
    def test_to_dict_has_plan_id(self):
        p = _make_plan()
        self.assertEqual(p.to_dict()["plan_id"], "plan_test")

    def test_to_dict_has_waves(self):
        t = _make_task()
        w = _make_wave(tasks=[t])
        p = _make_plan(waves=[w])
        d = p.to_dict()
        self.assertIsInstance(d["waves"], list)
        self.assertEqual(len(d["waves"]), 1)

    def test_to_dict_includes_computed_props(self):
        p = _make_plan()
        d = p.to_dict()
        self.assertIn("total_tasks", d)
        self.assertIn("completed_tasks", d)
        self.assertIn("progress_percent", d)

    def test_to_dict_json_serializable(self):
        p = _make_plan(waves=[_make_wave(tasks=[_make_task()])])
        json.dumps(p.to_dict())

    def test_from_dict_roundtrip(self):
        p = _make_plan()
        p.add_adr("adr-1", "T", "C", "D")
        d = p.to_dict()
        p2 = Plan.from_dict(d)
        self.assertEqual(p2.plan_id, "plan_test")
        self.assertEqual(len(p2.adr_history), 1)

    def test_from_dict_with_waves(self):
        t = _make_task("inner_t")
        w = _make_wave("inner_w", tasks=[t])
        p = _make_plan(waves=[w])
        p2 = Plan.from_dict(p.to_dict())
        self.assertEqual(len(p2.waves), 1)
        self.assertEqual(p2.waves[0].wave_id, "inner_w")

    def test_from_dict_empty(self):
        p = Plan.from_dict({})
        self.assertEqual(p.plan_id, "")
        self.assertEqual(p.status, PlanStatus.PENDING.value)


# ---------------------------------------------------------------------------
# PlanManager
# ---------------------------------------------------------------------------

class TestPlanManagerBase(unittest.TestCase):
    def setUp(self):
        PlanManager._instance = None
        self.tmp = tempfile.mkdtemp()
        self.pm = PlanManager(storage_path=self.tmp)

    def tearDown(self):
        PlanManager._instance = None


class TestPlanManagerInit(TestPlanManagerBase):
    def test_storage_path_created(self):
        self.assertTrue(Path(self.tmp).exists())

    def test_plans_dict_starts_empty(self):
        self.assertEqual(len(self.pm.plans), 0)

    def test_storage_path_attribute(self):
        self.assertEqual(str(self.pm.storage_path), self.tmp)


class TestPlanManagerSingleton(unittest.TestCase):
    def setUp(self):
        PlanManager._instance = None
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        PlanManager._instance = None

    def test_get_instance_creates(self):
        pm = PlanManager.get_instance(self.tmp)
        self.assertIsNotNone(pm)

    def test_get_instance_same_object(self):
        pm1 = PlanManager.get_instance(self.tmp)
        pm2 = PlanManager.get_instance(self.tmp)
        self.assertIs(pm1, pm2)

    def test_reset_creates_new(self):
        pm1 = PlanManager.get_instance(self.tmp)
        PlanManager._instance = None
        pm2 = PlanManager.get_instance(self.tmp)
        self.assertIsNot(pm1, pm2)


class TestCreatePlan(TestPlanManagerBase):
    def test_returns_plan(self):
        p = self.pm.create_plan("My Plan", "Build it")
        self.assertIsInstance(p, Plan)

    def test_plan_id_set(self):
        p = self.pm.create_plan("My Plan", "Build it")
        self.assertTrue(p.plan_id.startswith("plan_"))

    def test_plan_stored_in_dict(self):
        p = self.pm.create_plan("My Plan", "Build it")
        self.assertIn(p.plan_id, self.pm.plans)

    def test_plan_file_written(self):
        p = self.pm.create_plan("My Plan", "Build it")
        file_path = Path(self.tmp) / f"{p.plan_id}.json"
        self.assertTrue(file_path.exists())

    def test_plan_file_is_valid_json(self):
        p = self.pm.create_plan("My Plan", "Build it")
        file_path = Path(self.tmp) / f"{p.plan_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        self.assertEqual(data["plan_id"], p.plan_id)

    def test_create_with_created_by(self):
        p = self.pm.create_plan("Plan", "Prompt", created_by="alice")
        self.assertEqual(p.created_by, "alice")

    def test_create_default_created_by(self):
        p = self.pm.create_plan("Plan", "Prompt")
        self.assertEqual(p.created_by, "user")

    def test_create_with_waves_data(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.assertEqual(len(p.waves), 2)

    def test_create_with_waves_tasks(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.assertEqual(len(p.waves[0].tasks), 2)
        self.assertEqual(len(p.waves[1].tasks), 1)

    def test_create_wave_ids(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.assertEqual(p.waves[0].wave_id, "wave_1")
        self.assertEqual(p.waves[1].wave_id, "wave_2")

    def test_create_task_ids(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.assertEqual(p.waves[0].tasks[0].task_id, "task_1_1")
        self.assertEqual(p.waves[0].tasks[1].task_id, "task_1_2")

    def test_create_no_waves_data(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=None)
        self.assertEqual(p.waves, [])

    def test_task_priority_from_waves_data(self):
        wd = [{"milestone": "M", "tasks": [{"agent_type": "builder",
                                             "description": "d", "prompt": "p",
                                             "priority": 9}]}]
        p = self.pm.create_plan("Plan", "Prompt", waves_data=wd)
        self.assertEqual(p.waves[0].tasks[0].priority, 9)

    def test_task_default_priority(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.assertEqual(p.waves[0].tasks[0].priority, 5)

    def test_multiple_plans_stored(self):
        p1 = self.pm.create_plan("Plan 1", "Prompt 1")
        p2 = self.pm.create_plan("Plan 2", "Prompt 2")
        self.assertIn(p1.plan_id, self.pm.plans)
        self.assertIn(p2.plan_id, self.pm.plans)

    def test_wave_milestone_from_data(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.assertEqual(p.waves[0].milestone, "Phase 1")

    def test_wave_milestone_default(self):
        wd = [{"tasks": [{"agent_type": "builder", "description": "d", "prompt": "p"}]}]
        p = self.pm.create_plan("Plan", "Prompt", waves_data=wd)
        self.assertEqual(p.waves[0].milestone, "Wave 1")


class TestGetPlan(TestPlanManagerBase):
    def test_get_existing(self):
        p = self.pm.create_plan("Plan", "Prompt")
        result = self.pm.get_plan(p.plan_id)
        self.assertEqual(result.plan_id, p.plan_id)

    def test_get_nonexistent_returns_none(self):
        result = self.pm.get_plan("nonexistent")
        self.assertIsNone(result)

    def test_get_after_multiple_creates(self):
        p1 = self.pm.create_plan("Plan 1", "Prompt 1")
        p2 = self.pm.create_plan("Plan 2", "Prompt 2")
        self.assertEqual(self.pm.get_plan(p1.plan_id).title, "Plan 1")
        self.assertEqual(self.pm.get_plan(p2.plan_id).title, "Plan 2")


class TestListPlans(TestPlanManagerBase):
    def setUp(self):
        super().setUp()
        self.p1 = self.pm.create_plan("Plan A", "Prompt A")
        self.p2 = self.pm.create_plan("Plan B", "Prompt B")
        self.p3 = self.pm.create_plan("Plan C", "Prompt C")
        self.pm.update_plan(self.p2.plan_id, {"status": PlanStatus.ACTIVE.value})

    def test_list_all(self):
        plans = self.pm.list_plans()
        self.assertEqual(len(plans), 3)

    def test_list_by_status(self):
        plans = self.pm.list_plans(status=PlanStatus.PENDING.value)
        self.assertEqual(len(plans), 2)

    def test_list_active(self):
        plans = self.pm.list_plans(status=PlanStatus.ACTIVE.value)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].plan_id, self.p2.plan_id)

    def test_list_limit(self):
        plans = self.pm.list_plans(limit=2)
        self.assertEqual(len(plans), 2)

    def test_list_offset(self):
        all_plans = self.pm.list_plans()
        offset_plans = self.pm.list_plans(offset=1)
        self.assertEqual(len(offset_plans), 2)
        self.assertNotEqual(offset_plans[0].plan_id, all_plans[0].plan_id)

    def test_list_sorted_by_created_at_desc(self):
        plans = self.pm.list_plans()
        # Most recently created first
        for i in range(len(plans) - 1):
            self.assertGreaterEqual(plans[i].created_at, plans[i+1].created_at)

    def test_list_empty_filter(self):
        plans = self.pm.list_plans(status="cancelled")
        self.assertEqual(len(plans), 0)


class TestUpdatePlan(TestPlanManagerBase):
    def test_update_title(self):
        p = self.pm.create_plan("Old Title", "Prompt")
        self.pm.update_plan(p.plan_id, {"title": "New Title"})
        self.assertEqual(self.pm.get_plan(p.plan_id).title, "New Title")

    def test_update_status(self):
        p = self.pm.create_plan("Plan", "Prompt")
        self.pm.update_plan(p.plan_id, {"status": PlanStatus.ACTIVE.value})
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.ACTIVE.value)

    def test_update_persists_to_file(self):
        p = self.pm.create_plan("Plan", "Prompt")
        self.pm.update_plan(p.plan_id, {"title": "Updated"})
        file_path = Path(self.tmp) / f"{p.plan_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        self.assertEqual(data["title"], "Updated")

    def test_update_updates_updated_at(self):
        p = self.pm.create_plan("Plan", "Prompt")
        old_ts = p.updated_at
        import time; time.sleep(0.01)
        self.pm.update_plan(p.plan_id, {"title": "New"})
        updated_p = self.pm.get_plan(p.plan_id)
        self.assertGreaterEqual(updated_p.updated_at, old_ts)

    def test_update_nonexistent_returns_none(self):
        result = self.pm.update_plan("no_such_plan", {"title": "X"})
        self.assertIsNone(result)

    def test_update_returns_plan(self):
        p = self.pm.create_plan("Plan", "Prompt")
        result = self.pm.update_plan(p.plan_id, {"title": "New"})
        self.assertIsInstance(result, Plan)


class TestDeletePlan(TestPlanManagerBase):
    def test_delete_returns_true(self):
        p = self.pm.create_plan("Plan", "Prompt")
        self.assertTrue(self.pm.delete_plan(p.plan_id))

    def test_delete_removes_from_dict(self):
        p = self.pm.create_plan("Plan", "Prompt")
        self.pm.delete_plan(p.plan_id)
        self.assertNotIn(p.plan_id, self.pm.plans)

    def test_delete_removes_file(self):
        p = self.pm.create_plan("Plan", "Prompt")
        file_path = Path(self.tmp) / f"{p.plan_id}.json"
        self.assertTrue(file_path.exists())
        self.pm.delete_plan(p.plan_id)
        self.assertFalse(file_path.exists())

    def test_delete_nonexistent_returns_false(self):
        self.assertFalse(self.pm.delete_plan("ghost"))

    def test_get_after_delete_returns_none(self):
        p = self.pm.create_plan("Plan", "Prompt")
        self.pm.delete_plan(p.plan_id)
        self.assertIsNone(self.pm.get_plan(p.plan_id))


class TestStartPlan(TestPlanManagerBase):
    def test_start_sets_active(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.ACTIVE.value)

    def test_start_sets_first_wave_running(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).waves[0].status, WaveStatus.RUNNING.value)

    def test_start_second_wave_stays_pending(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).waves[1].status, WaveStatus.PENDING.value)

    def test_start_no_waves(self):
        p = self.pm.create_plan("Plan", "Prompt")
        result = self.pm.start_plan(p.plan_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, PlanStatus.ACTIVE.value)

    def test_start_nonexistent_returns_none(self):
        result = self.pm.start_plan("ghost")
        self.assertIsNone(result)

    def test_start_persists(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        file_path = Path(self.tmp) / f"{p.plan_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        self.assertEqual(data["status"], PlanStatus.ACTIVE.value)


class TestPausePlan(TestPlanManagerBase):
    def test_pause_active_plan(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.pause_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.PAUSED.value)

    def test_pause_sets_running_wave_blocked(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.pause_plan(p.plan_id)
        first_wave = self.pm.get_plan(p.plan_id).waves[0]
        self.assertEqual(first_wave.status, WaveStatus.BLOCKED.value)

    def test_pause_non_active_returns_none(self):
        p = self.pm.create_plan("Plan", "Prompt")
        result = self.pm.pause_plan(p.plan_id)
        self.assertIsNone(result)

    def test_pause_nonexistent_returns_none(self):
        self.assertIsNone(self.pm.pause_plan("ghost"))

    def test_pause_persists(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.pause_plan(p.plan_id)
        file_path = Path(self.tmp) / f"{p.plan_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        self.assertEqual(data["status"], PlanStatus.PAUSED.value)


class TestResumePlan(TestPlanManagerBase):
    def test_resume_paused_plan(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.pause_plan(p.plan_id)
        self.pm.resume_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.ACTIVE.value)

    def test_resume_restores_blocked_wave(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.pause_plan(p.plan_id)
        self.pm.resume_plan(p.plan_id)
        first_wave = self.pm.get_plan(p.plan_id).waves[0]
        self.assertEqual(first_wave.status, WaveStatus.RUNNING.value)

    def test_resume_non_paused_returns_none(self):
        p = self.pm.create_plan("Plan", "Prompt")
        result = self.pm.resume_plan(p.plan_id)
        self.assertIsNone(result)

    def test_resume_nonexistent_returns_none(self):
        self.assertIsNone(self.pm.resume_plan("ghost"))

    def test_resume_persists(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.pause_plan(p.plan_id)
        self.pm.resume_plan(p.plan_id)
        file_path = Path(self.tmp) / f"{p.plan_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        self.assertEqual(data["status"], PlanStatus.ACTIVE.value)


class TestCancelPlan(TestPlanManagerBase):
    def test_cancel_sets_cancelled(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.cancel_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.CANCELLED.value)

    def test_cancel_pending_waves_blocked(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.cancel_plan(p.plan_id)
        for w in self.pm.get_plan(p.plan_id).waves:
            if w.status not in (WaveStatus.COMPLETED.value,):
                self.assertEqual(w.status, WaveStatus.BLOCKED.value)

    def test_cancel_pending_tasks_blocked(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.cancel_plan(p.plan_id)
        for w in self.pm.get_plan(p.plan_id).waves:
            for t in w.tasks:
                self.assertEqual(t.status, TaskStatus.BLOCKED.value)

    def test_cancel_nonexistent_returns_none(self):
        self.assertIsNone(self.pm.cancel_plan("ghost"))

    def test_cancel_persists(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.cancel_plan(p.plan_id)
        file_path = Path(self.tmp) / f"{p.plan_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        self.assertEqual(data["status"], PlanStatus.CANCELLED.value)

    def test_cancel_already_cancelled_plan(self):
        p = self.pm.create_plan("Plan", "Prompt")
        self.pm.cancel_plan(p.plan_id)
        result = self.pm.cancel_plan(p.plan_id)
        self.assertIsNotNone(result)


class TestUpdateTaskStatus(TestPlanManagerBase):
    def setUp(self):
        super().setUp()
        self.plan = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(self.plan.plan_id)

    def test_update_task_to_running(self):
        p = self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_1", TaskStatus.RUNNING.value
        )
        task = p.waves[0].tasks[0]
        self.assertEqual(task.status, TaskStatus.RUNNING.value)
        self.assertNotEqual(task.actual_start, "")

    def test_update_task_to_completed(self):
        p = self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_1", TaskStatus.COMPLETED.value, result="done"
        )
        task = p.waves[0].tasks[0]
        self.assertEqual(task.status, TaskStatus.COMPLETED.value)
        self.assertEqual(task.result, "done")
        self.assertNotEqual(task.actual_end, "")

    def test_update_task_to_failed(self):
        p = self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_1", TaskStatus.FAILED.value, error="oops"
        )
        task = p.waves[0].tasks[0]
        self.assertEqual(task.status, TaskStatus.FAILED.value)
        self.assertEqual(task.error, "oops")
        self.assertNotEqual(task.actual_end, "")

    def test_update_nonexistent_plan_returns_none(self):
        result = self.pm.update_task_status(
            "ghost", "wave_1", "task_1_1", TaskStatus.COMPLETED.value
        )
        self.assertIsNone(result)

    def test_update_persists(self):
        self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_1", TaskStatus.COMPLETED.value
        )
        file_path = Path(self.tmp) / f"{self.plan.plan_id}.json"
        with open(file_path) as f:
            data = json.load(f)
        task_dict = data["waves"][0]["tasks"][0]
        self.assertEqual(task_dict["status"], TaskStatus.COMPLETED.value)

    def test_complete_all_tasks_advances_wave(self):
        # Complete both tasks in wave_1
        self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_1", TaskStatus.COMPLETED.value
        )
        self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_2", TaskStatus.COMPLETED.value
        )
        p = self.pm.get_plan(self.plan.plan_id)
        self.assertEqual(p.waves[0].status, WaveStatus.COMPLETED.value)
        self.assertEqual(p.waves[1].status, WaveStatus.RUNNING.value)

    def test_partial_complete_does_not_advance(self):
        self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_1", TaskStatus.COMPLETED.value
        )
        p = self.pm.get_plan(self.plan.plan_id)
        self.assertNotEqual(p.waves[0].status, WaveStatus.COMPLETED.value)

    def test_complete_last_wave_completes_plan(self):
        # Complete wave_1
        self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_1", TaskStatus.COMPLETED.value
        )
        self.pm.update_task_status(
            self.plan.plan_id, "wave_1", "task_1_2", TaskStatus.COMPLETED.value
        )
        # Complete wave_2
        self.pm.update_task_status(
            self.plan.plan_id, "wave_2", "task_2_1", TaskStatus.COMPLETED.value
        )
        p = self.pm.get_plan(self.plan.plan_id)
        self.assertEqual(p.status, PlanStatus.COMPLETED.value)


class TestLoadPlans(unittest.TestCase):
    """Tests that PlanManager re-loads persisted plans from disk."""

    def setUp(self):
        PlanManager._instance = None
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        PlanManager._instance = None

    def test_load_existing_plans(self):
        pm1 = PlanManager(storage_path=self.tmp)
        pm1.create_plan("Persisted Plan", "Prompt")

        PlanManager._instance = None
        pm2 = PlanManager(storage_path=self.tmp)
        self.assertEqual(len(pm2.plans), 1)

    def test_loaded_plan_correct_title(self):
        pm1 = PlanManager(storage_path=self.tmp)
        p = pm1.create_plan("Hello World", "Prompt")

        PlanManager._instance = None
        pm2 = PlanManager(storage_path=self.tmp)
        loaded = pm2.get_plan(p.plan_id)
        self.assertEqual(loaded.title, "Hello World")

    def test_bad_json_file_skipped(self):
        # Write a corrupt JSON file to the storage dir
        bad_file = Path(self.tmp) / "bad.json"
        bad_file.write_text("not valid json {{{")
        pm = PlanManager(storage_path=self.tmp)
        # Should not crash; bad file is skipped
        self.assertIsInstance(pm.plans, dict)


class TestGetPlanManagerFunction(unittest.TestCase):
    def setUp(self):
        PlanManager._instance = None
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        PlanManager._instance = None

    def test_get_plan_manager_returns_instance(self):
        # Need to prime the singleton with our temp dir first
        PlanManager._instance = PlanManager(storage_path=self.tmp)
        pm = get_plan_manager()
        self.assertIsInstance(pm, PlanManager)

    def test_get_plan_manager_same_instance(self):
        PlanManager._instance = PlanManager(storage_path=self.tmp)
        pm1 = get_plan_manager()
        pm2 = get_plan_manager()
        self.assertIs(pm1, pm2)


class TestLazyPlanManager(unittest.TestCase):
    def setUp(self):
        PlanManager._instance = None
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        PlanManager._instance = None

    def test_lazy_proxy_is_instance(self):
        self.assertIsInstance(plan_manager, _LazyPlanManager)

    def test_lazy_proxy_repr(self):
        PlanManager._instance = PlanManager(storage_path=self.tmp)
        r = repr(plan_manager)
        self.assertIsInstance(r, str)

    def test_lazy_proxy_delegates_attribute(self):
        PlanManager._instance = PlanManager(storage_path=self.tmp)
        # Accessing .plans should return the underlying dict
        self.assertIsInstance(plan_manager.plans, dict)


class TestPlanStatusTransitions(TestPlanManagerBase):
    """High-level state machine tests."""

    def test_pending_to_active(self):
        p = self.pm.create_plan("Plan", "Prompt")
        self.assertEqual(p.status, PlanStatus.PENDING.value)
        self.pm.start_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.ACTIVE.value)

    def test_active_to_paused_to_active(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.pause_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.PAUSED.value)
        self.pm.resume_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.ACTIVE.value)

    def test_active_to_cancelled(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        self.pm.cancel_plan(p.plan_id)
        self.assertEqual(self.pm.get_plan(p.plan_id).status, PlanStatus.CANCELLED.value)

    def test_full_wave_completion_chain(self):
        p = self.pm.create_plan("Plan", "Prompt", waves_data=_waves_data())
        self.pm.start_plan(p.plan_id)
        # Complete wave 1 tasks
        self.pm.update_task_status(p.plan_id, "wave_1", "task_1_1", TaskStatus.COMPLETED.value)
        self.pm.update_task_status(p.plan_id, "wave_1", "task_1_2", TaskStatus.COMPLETED.value)
        # Wave 1 done, wave 2 running
        refreshed = self.pm.get_plan(p.plan_id)
        self.assertEqual(refreshed.waves[0].status, WaveStatus.COMPLETED.value)
        self.assertEqual(refreshed.waves[1].status, WaveStatus.RUNNING.value)
        # Complete wave 2
        self.pm.update_task_status(p.plan_id, "wave_2", "task_2_1", TaskStatus.COMPLETED.value)
        final = self.pm.get_plan(p.plan_id)
        self.assertEqual(final.status, PlanStatus.COMPLETED.value)


if __name__ == "__main__":
    unittest.main()
