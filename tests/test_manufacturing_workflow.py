"""
Tests for vetinari.workflow -- Manufacturing Workflow Integration
================================================================
Covers quality gates, SPC, Andon signals, and WIP limits.
"""

import math
import pytest

from vetinari.workflow.quality_gates import (
    GateAction,
    WorkflowGate,
    WorkflowGateRunner,
    WORKFLOW_GATES,
)
from vetinari.workflow.spc import (
    AndonSignal,
    AndonSystem,
    ControlChart,
    SPCAlert,
    SPCMonitor,
    WIPConfig,
    WIPTracker,
)


# =========================================================================
# GateAction enum
# =========================================================================

class TestGateAction:
    """Verify the GateAction enum values."""

    def test_enum_values(self):
        assert GateAction.RE_PLAN.value == "re_plan"
        assert GateAction.RETRY.value == "retry"
        assert GateAction.ESCALATE.value == "escalate"
        assert GateAction.BLOCK.value == "block"
        assert GateAction.CONTINUE.value == "continue"

    def test_enum_count(self):
        assert len(GateAction) == 5


# =========================================================================
# WorkflowGate dataclass & WORKFLOW_GATES dict
# =========================================================================

class TestWorkflowGate:
    """Verify WorkflowGate creation and the default gate map."""

    def test_create_gate(self):
        gate = WorkflowGate(
            name="test_gate",
            stage="post_test",
            criteria={"score": 0.8},
            failure_action=GateAction.RETRY,
        )
        assert gate.name == "test_gate"
        assert gate.stage == "post_test"
        assert gate.criteria == {"score": 0.8}
        assert gate.failure_action is GateAction.RETRY

    def test_default_gates_keys(self):
        expected = {"post_decomposition", "post_execution", "post_verification", "pre_assembly"}
        assert set(WORKFLOW_GATES.keys()) == expected

    def test_decomposition_gate(self):
        g = WORKFLOW_GATES["post_decomposition"]
        assert g.name == "decomposition_gate"
        assert g.failure_action is GateAction.RE_PLAN
        assert g.criteria["min_tasks"] == 1
        assert g.criteria["max_tasks"] == 50
        assert g.criteria["dag_valid"] is True
        assert g.criteria["goal_adherence"] == 0.7

    def test_execution_gate(self):
        g = WORKFLOW_GATES["post_execution"]
        assert g.name == "execution_gate"
        assert g.failure_action is GateAction.RETRY
        assert g.criteria["quality_score"] == 0.6
        assert g.criteria["no_critical_security"] is True

    def test_verification_gate(self):
        g = WORKFLOW_GATES["post_verification"]
        assert g.name == "verification_gate"
        assert g.failure_action is GateAction.ESCALATE

    def test_assembly_gate(self):
        g = WORKFLOW_GATES["pre_assembly"]
        assert g.name == "assembly_gate"
        assert g.failure_action is GateAction.BLOCK


# =========================================================================
# WorkflowGateRunner
# =========================================================================

class TestWorkflowGateRunner:
    """Evaluate gates for each stage -- pass and fail cases."""

    def setup_method(self):
        self.runner = WorkflowGateRunner()

    # -- post_decomposition ------------------------------------------------

    def test_decomposition_pass(self):
        metrics = {
            "min_tasks": 5,
            "max_tasks": 10,
            "dag_valid": True,
            "goal_adherence": 0.9,
        }
        passed, action, violations = self.runner.evaluate("post_decomposition", metrics)
        assert passed is True
        assert action is GateAction.CONTINUE
        assert violations == []

    def test_decomposition_fail_dag_invalid(self):
        metrics = {
            "min_tasks": 5,
            "max_tasks": 10,
            "dag_valid": False,
            "goal_adherence": 0.9,
        }
        passed, action, violations = self.runner.evaluate("post_decomposition", metrics)
        assert passed is False
        assert action is GateAction.RE_PLAN
        assert any("dag_valid" in v for v in violations)

    def test_decomposition_fail_low_adherence(self):
        metrics = {
            "min_tasks": 5,
            "max_tasks": 10,
            "dag_valid": True,
            "goal_adherence": 0.3,
        }
        passed, action, violations = self.runner.evaluate("post_decomposition", metrics)
        assert passed is False
        assert action is GateAction.RE_PLAN
        assert any("goal_adherence" in v for v in violations)

    def test_decomposition_fail_too_many_tasks(self):
        """max_tasks criterion uses upper-bound semantics."""
        metrics = {
            "min_tasks": 5,
            "max_tasks": 60,   # exceeds threshold of 50
            "dag_valid": True,
            "goal_adherence": 0.8,
        }
        passed, action, violations = self.runner.evaluate("post_decomposition", metrics)
        assert passed is False
        assert any("max_tasks" in v for v in violations)

    def test_decomposition_fail_missing_metric(self):
        metrics = {"min_tasks": 5}  # missing several
        passed, action, violations = self.runner.evaluate("post_decomposition", metrics)
        assert passed is False
        assert any("Missing" in v for v in violations)

    # -- post_execution ----------------------------------------------------

    def test_execution_pass(self):
        metrics = {"quality_score": 0.85, "no_critical_security": True}
        passed, action, _ = self.runner.evaluate("post_execution", metrics)
        assert passed is True
        assert action is GateAction.CONTINUE

    def test_execution_fail_low_quality(self):
        metrics = {"quality_score": 0.3, "no_critical_security": True}
        passed, action, violations = self.runner.evaluate("post_execution", metrics)
        assert passed is False
        assert action is GateAction.RETRY
        assert any("quality_score" in v for v in violations)

    def test_execution_fail_security_issue(self):
        metrics = {"quality_score": 0.9, "no_critical_security": False}
        passed, action, violations = self.runner.evaluate("post_execution", metrics)
        assert passed is False
        assert action is GateAction.RETRY

    # -- post_verification -------------------------------------------------

    def test_verification_pass(self):
        metrics = {"verification_passed": True, "drift_score_max": 0.2}
        passed, action, _ = self.runner.evaluate("post_verification", metrics)
        assert passed is True
        assert action is GateAction.CONTINUE

    def test_verification_fail_drift(self):
        metrics = {"verification_passed": True, "drift_score_max": 0.8}
        passed, action, _ = self.runner.evaluate("post_verification", metrics)
        assert passed is False
        assert action is GateAction.ESCALATE

    def test_verification_fail_not_passed(self):
        metrics = {"verification_passed": False, "drift_score_max": 0.1}
        passed, action, _ = self.runner.evaluate("post_verification", metrics)
        assert passed is False
        assert action is GateAction.ESCALATE

    # -- pre_assembly ------------------------------------------------------

    def test_assembly_pass(self):
        metrics = {"all_dependencies_met": True, "no_blocked_tasks": True}
        passed, action, _ = self.runner.evaluate("pre_assembly", metrics)
        assert passed is True
        assert action is GateAction.CONTINUE

    def test_assembly_fail_dependencies(self):
        metrics = {"all_dependencies_met": False, "no_blocked_tasks": True}
        passed, action, _ = self.runner.evaluate("pre_assembly", metrics)
        assert passed is False
        assert action is GateAction.BLOCK

    # -- unknown stage passes by default -----------------------------------

    def test_unknown_stage_passes(self):
        passed, action, _ = self.runner.evaluate("nonexistent", {})
        assert passed is True
        assert action is GateAction.CONTINUE

    # -- history -----------------------------------------------------------

    def test_history_recorded(self):
        self.runner.evaluate("post_execution", {"quality_score": 0.9, "no_critical_security": True})
        self.runner.evaluate("post_execution", {"quality_score": 0.1, "no_critical_security": True})

        history = self.runner.get_history()
        assert len(history) == 2
        assert history[0]["passed"] is True
        assert history[1]["passed"] is False
        assert "timestamp" in history[0]

    # -- add / remove gates ------------------------------------------------

    def test_add_gate(self):
        custom = WorkflowGate(
            name="custom",
            stage="post_custom",
            criteria={"x": 10},
            failure_action=GateAction.BLOCK,
        )
        self.runner.add_gate("post_custom", custom)
        passed, action, _ = self.runner.evaluate("post_custom", {"x": 5})
        assert passed is False
        assert action is GateAction.BLOCK

    def test_remove_gate(self):
        removed = self.runner.remove_gate("pre_assembly")
        assert removed is not None
        assert removed.name == "assembly_gate"
        # Now that stage passes by default
        passed, _, _ = self.runner.evaluate("pre_assembly", {})
        assert passed is True

    def test_gates_property(self):
        gates = self.runner.gates
        assert "post_execution" in gates


# =========================================================================
# ControlChart
# =========================================================================

class TestControlChart:
    """Test statistical computations on the control chart."""

    def test_empty_chart(self):
        chart = ControlChart(metric_name="test")
        assert chart.mean == 0.0
        assert chart.sigma == 0.0
        assert chart.ucl == 0.0
        assert chart.lcl == 0.0

    def test_single_value(self):
        chart = ControlChart(metric_name="test")
        chart.add_value(5.0)
        assert chart.mean == 5.0
        assert chart.sigma == 0.0
        # sigma=0 => ucl == lcl == mean
        assert chart.ucl == 5.0
        assert chart.lcl == 5.0

    def test_mean_and_sigma(self):
        chart = ControlChart(metric_name="test")
        for v in [10.0, 20.0, 30.0]:
            chart.add_value(v)
        assert chart.mean == pytest.approx(20.0)
        expected_sigma = math.sqrt(((10 - 20)**2 + (20 - 20)**2 + (30 - 20)**2) / 3)
        assert chart.sigma == pytest.approx(expected_sigma)

    def test_ucl_lcl(self):
        chart = ControlChart(metric_name="test", sigma_multiplier=2.0)
        for v in [10.0, 20.0, 30.0]:
            chart.add_value(v)
        m = chart.mean
        s = chart.sigma
        assert chart.ucl == pytest.approx(m + 2.0 * s)
        assert chart.lcl == pytest.approx(m - 2.0 * s)

    def test_is_in_control(self):
        chart = ControlChart(metric_name="test", sigma_multiplier=3.0)
        for v in [50.0, 51.0, 49.0, 50.5, 50.2]:
            chart.add_value(v)
        # A value close to the mean should be in control
        assert chart.is_in_control(50.0) is True
        # A wildly different value should not
        assert chart.is_in_control(200.0) is False

    def test_window_size(self):
        chart = ControlChart(metric_name="test", window_size=3)
        for v in [100.0, 100.0, 100.0, 1.0, 2.0, 3.0]:
            chart.add_value(v)
        # Window only contains the last 3 values: 1, 2, 3
        assert chart.mean == pytest.approx(2.0)

    def test_cpk_both_specs(self):
        chart = ControlChart(metric_name="test")
        for v in [10.0, 11.0, 12.0, 10.5, 11.5]:
            chart.add_value(v)
        cpk = chart.get_cpk(spec_upper=15.0, spec_lower=5.0)
        assert cpk is not None
        assert cpk > 0

    def test_cpk_upper_only(self):
        chart = ControlChart(metric_name="test")
        for v in [10.0, 11.0, 12.0]:
            chart.add_value(v)
        cpk = chart.get_cpk(spec_upper=20.0)
        assert cpk is not None
        assert cpk > 0

    def test_cpk_no_specs(self):
        chart = ControlChart(metric_name="test")
        for v in [10.0, 11.0]:
            chart.add_value(v)
        assert chart.get_cpk() is None

    def test_cpk_zero_sigma(self):
        chart = ControlChart(metric_name="test")
        chart.add_value(5.0)
        assert chart.get_cpk(spec_upper=10.0) is None

    def test_trend_detection(self):
        chart = ControlChart(metric_name="test")
        # Monotonically increasing 7 consecutive values
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
            chart.add_value(v)
        assert chart._detect_trend(min_consecutive=7) is True

    def test_no_trend(self):
        chart = ControlChart(metric_name="test")
        for v in [1.0, 3.0, 2.0, 4.0, 3.0]:
            chart.add_value(v)
        assert chart._detect_trend() is False

    def test_run_detection(self):
        chart = ControlChart(metric_name="test")
        # Mean ~ 10.  Then 8 values all above mean.
        for v in [5.0, 5.0, 5.0, 15.0, 15.0, 15.0]:
            chart.add_value(v)
        # After these, mean is 10.  Now add 8 above mean:
        for v in [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]:
            chart.add_value(v)
        assert chart._detect_run(min_run=8) is True

    def test_no_run(self):
        chart = ControlChart(metric_name="test")
        for v in [10.0, 20.0, 10.0, 20.0]:
            chart.add_value(v)
        assert chart._detect_run() is False


# =========================================================================
# SPCMonitor
# =========================================================================

class TestSPCMonitor:
    """Test the multi-metric SPC monitor."""

    def test_update_no_alert_initially(self):
        mon = SPCMonitor()
        alert = mon.update("quality_score", 0.9)
        assert alert is None  # Not enough data

    def test_update_returns_alert_above_ucl(self):
        mon = SPCMonitor(window_size=20, sigma_multiplier=2.0)
        # Establish baseline with slight variation (non-zero sigma)
        import random
        rng = random.Random(42)
        for _ in range(20):
            mon.update("latency_ms", 100.0 + rng.uniform(-1, 1))
        # Spike well above — must exceed mean + 2*sigma even after inclusion
        alert = mon.update("latency_ms", 500.0)
        assert alert is not None
        assert alert.alert_type == "above_ucl"
        assert alert.metric_name == "latency_ms"

    def test_update_returns_alert_below_lcl(self):
        mon = SPCMonitor(window_size=20, sigma_multiplier=2.0)
        import random
        rng = random.Random(99)
        for _ in range(20):
            mon.update("score", 100.0 + rng.uniform(-1, 1))
        alert = mon.update("score", -300.0)
        assert alert is not None
        assert alert.alert_type == "below_lcl"

    def test_get_chart(self):
        mon = SPCMonitor()
        mon.update("x", 1.0)
        chart = mon.get_chart("x")
        assert chart is not None
        assert chart.metric_name == "x"
        assert len(chart.values) == 1

    def test_get_chart_missing(self):
        mon = SPCMonitor()
        assert mon.get_chart("nonexistent") is None

    def test_get_alerts_filtered(self):
        mon = SPCMonitor(window_size=20, sigma_multiplier=3.0)
        # Use very tight baseline to avoid spurious alerts
        for i in range(20):
            mon.update("a", 50.0)
            mon.update("b", 50.0)
        # Add slight variation so sigma > 0
        mon.update("a", 50.1)
        mon.update("b", 50.1)
        alerts_before_a = len(mon.get_alerts("a"))
        alerts_before_b = len(mon.get_alerts("b"))
        # Spike metric "a" well above UCL
        mon.update("a", 500.0)
        assert len(mon.get_alerts("a")) == alerts_before_a + 1
        assert len(mon.get_alerts("b")) == alerts_before_b
        assert len(mon.get_alerts("a")) > len(mon.get_alerts("b"))

    def test_get_summary(self):
        mon = SPCMonitor()
        for v in [1.0, 2.0, 3.0]:
            mon.update("x", v)
        summary = mon.get_summary()
        assert "x" in summary
        assert "mean" in summary["x"]
        assert "sigma" in summary["x"]
        assert "ucl" in summary["x"]
        assert "lcl" in summary["x"]
        assert summary["x"]["count"] == 3

    def test_default_metrics_attribute(self):
        assert "quality_score" in SPCMonitor.DEFAULT_METRICS
        assert "latency_ms" in SPCMonitor.DEFAULT_METRICS


# =========================================================================
# AndonSystem
# =========================================================================

class TestAndonSystem:
    """Test Andon stop-the-line alert management."""

    def test_raise_warning_no_pause(self):
        andon = AndonSystem()
        signal = andon.raise_signal("gate_1", "warning", "Minor issue")
        assert signal.severity == "warning"
        assert andon.is_paused() is False

    def test_raise_critical_pauses(self):
        andon = AndonSystem()
        andon.raise_signal("gate_2", "critical", "Build broken")
        assert andon.is_paused() is True

    def test_raise_emergency_pauses(self):
        andon = AndonSystem()
        andon.raise_signal("gate_3", "emergency", "Security breach")
        assert andon.is_paused() is True

    def test_acknowledge_resumes(self):
        andon = AndonSystem()
        andon.raise_signal("gate_2", "critical", "Build broken")
        assert andon.is_paused() is True

        success = andon.acknowledge(0)
        assert success is True
        assert andon.is_paused() is False

    def test_acknowledge_multiple_critical(self):
        andon = AndonSystem()
        andon.raise_signal("gate_a", "critical", "Failure A")
        andon.raise_signal("gate_b", "critical", "Failure B")
        assert andon.is_paused() is True

        andon.acknowledge(0)
        assert andon.is_paused() is True  # B still unacknowledged

        andon.acknowledge(1)
        assert andon.is_paused() is False

    def test_acknowledge_invalid_index(self):
        andon = AndonSystem()
        assert andon.acknowledge(999) is False
        assert andon.acknowledge(-1) is False

    def test_get_active_signals(self):
        andon = AndonSystem()
        andon.raise_signal("s1", "warning", "W1")
        andon.raise_signal("s2", "critical", "C1")

        active = andon.get_active_signals()
        assert len(active) == 2

        andon.acknowledge(0)
        active = andon.get_active_signals()
        assert len(active) == 1
        assert active[0].source == "s2"

    def test_get_all_signals(self):
        andon = AndonSystem()
        andon.raise_signal("s1", "warning", "W1")
        andon.acknowledge(0)
        andon.raise_signal("s2", "critical", "C1")

        all_sigs = andon.get_all_signals()
        assert len(all_sigs) == 2
        assert all_sigs[0].acknowledged is True
        assert all_sigs[1].acknowledged is False

    def test_signal_affected_tasks(self):
        andon = AndonSystem()
        signal = andon.raise_signal("gate", "warning", "Msg", affected_tasks=["t1", "t2"])
        assert signal.affected_tasks == ["t1", "t2"]

    def test_signal_default_affected_tasks(self):
        andon = AndonSystem()
        signal = andon.raise_signal("gate", "warning", "Msg")
        assert signal.affected_tasks == []

    def test_signal_has_timestamp(self):
        andon = AndonSystem()
        signal = andon.raise_signal("gate", "warning", "Msg")
        assert signal.timestamp is not None
        assert len(signal.timestamp) > 0


# =========================================================================
# WIPConfig
# =========================================================================

class TestWIPConfig:
    """Test WIP configuration and limit lookups."""

    def test_default_limits(self):
        cfg = WIPConfig()
        assert cfg.get_limit("BUILDER") == 2
        assert cfg.get_limit("TESTER") == 3
        assert cfg.get_limit("PLANNER") == 1
        assert cfg.get_limit("ARCHITECT") == 1

    def test_default_fallback(self):
        cfg = WIPConfig()
        assert cfg.get_limit("UNKNOWN_TYPE") == 4

    def test_custom_limits(self):
        cfg = WIPConfig(limits={"BUILDER": 10, "default": 1})
        assert cfg.get_limit("BUILDER") == 10
        assert cfg.get_limit("OTHER") == 1


# =========================================================================
# WIPTracker
# =========================================================================

class TestWIPTracker:
    """Test WIP limit enforcement and queuing."""

    def test_can_start_within_limit(self):
        tracker = WIPTracker()
        assert tracker.can_start("BUILDER") is True

    def test_start_task_success(self):
        tracker = WIPTracker()
        assert tracker.start_task("BUILDER", "t1") is True
        assert tracker.get_active_count("BUILDER") == 1

    def test_start_task_at_limit_fails(self):
        cfg = WIPConfig(limits={"BUILDER": 1, "default": 1})
        tracker = WIPTracker(config=cfg)
        assert tracker.start_task("BUILDER", "t1") is True
        assert tracker.start_task("BUILDER", "t2") is False
        assert tracker.get_active_count("BUILDER") == 1

    def test_complete_task_frees_slot(self):
        cfg = WIPConfig(limits={"BUILDER": 1, "default": 1})
        tracker = WIPTracker(config=cfg)
        tracker.start_task("BUILDER", "t1")
        tracker.complete_task("BUILDER", "t1")
        assert tracker.get_active_count("BUILDER") == 0
        assert tracker.can_start("BUILDER") is True

    def test_complete_pulls_from_queue(self):
        cfg = WIPConfig(limits={"BUILDER": 1, "default": 1})
        tracker = WIPTracker(config=cfg)
        tracker.start_task("BUILDER", "t1")
        tracker.enqueue("BUILDER", "t2")
        assert tracker.get_queue_depth() == 1

        pulled = tracker.complete_task("BUILDER", "t1")
        assert pulled is not None
        assert pulled["task_id"] == "t2"
        assert tracker.get_active_count("BUILDER") == 1
        assert tracker.get_queue_depth() == 0

    def test_complete_returns_none_when_empty_queue(self):
        tracker = WIPTracker()
        tracker.start_task("BUILDER", "t1")
        result = tracker.complete_task("BUILDER", "t1")
        assert result is None

    def test_enqueue(self):
        tracker = WIPTracker()
        tracker.enqueue("BUILDER", "t1")
        assert tracker.get_queue_depth() == 1

    def test_utilization(self):
        tracker = WIPTracker()
        tracker.start_task("BUILDER", "t1")

        util = tracker.get_utilization()
        assert "BUILDER" in util
        assert util["BUILDER"]["active"] == 1
        assert util["BUILDER"]["limit"] == 2
        assert util["BUILDER"]["utilization"] == pytest.approx(0.5)
        assert util["BUILDER"]["available"] == 1

    def test_utilization_includes_config_types(self):
        tracker = WIPTracker()
        util = tracker.get_utilization()
        # All configured types (except 'default') appear
        assert "PLANNER" in util
        assert "TESTER" in util
        assert "default" not in util

    def test_queue_respects_agent_type(self):
        """Completing a BUILDER task should not pull a TESTER task."""
        cfg = WIPConfig(limits={"BUILDER": 1, "TESTER": 1, "default": 1})
        tracker = WIPTracker(config=cfg)
        tracker.start_task("BUILDER", "b1")
        tracker.enqueue("TESTER", "te1")

        pulled = tracker.complete_task("BUILDER", "b1")
        assert pulled is None  # te1 is for TESTER, not BUILDER
        assert tracker.get_queue_depth() == 1


# =========================================================================
# Package-level imports
# =========================================================================

class TestPackageImports:
    """Verify that the workflow package __init__ exports everything."""

    def test_import_from_package(self):
        from vetinari.workflow import (
            GateAction,
            WorkflowGate,
            WorkflowGateRunner,
            WORKFLOW_GATES,
            AndonSignal,
            AndonSystem,
            ControlChart,
            SPCAlert,
            SPCMonitor,
            WIPConfig,
            WIPTracker,
        )
        # Just verifying the imports succeed and are the right types
        assert issubclass(GateAction, Enum)
        assert callable(WorkflowGateRunner)
        assert callable(SPCMonitor)
        assert callable(AndonSystem)
        assert callable(WIPTracker)


from enum import Enum  # noqa: E402 -- used in TestPackageImports
