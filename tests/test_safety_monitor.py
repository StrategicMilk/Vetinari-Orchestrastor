"""
Tests for vetinari.safety.agent_monitor and vetinari.safety.policy_enforcer.

Tests cover:
- Agent registration and heartbeat
- Step limit enforcement (StepLimitExceeded)
- Health check detecting stale agents
- Reset behaviour
- Stats reporting
- PolicyEnforcer file jurisdiction violations
- PolicyEnforcer delegation depth limits
- PolicyEnforcer irreversibility checks
- PolicyEnforcer resource budget checks
- Custom policy registration
- Singleton / reset behaviour for both classes
"""

from __future__ import annotations

import time
import unittest

from vetinari.safety.agent_monitor import (
    AgentMonitor,
    AgentTimeoutError,
    StepLimitExceeded,
    get_agent_monitor,
    reset_agent_monitor,
)
from vetinari.safety.policy_enforcer import (
    PolicyDecision,
    PolicyEnforcer,
    get_policy_enforcer,
    reset_policy_enforcer,
)
from vetinari.exceptions import AgentError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_monitor() -> AgentMonitor:
    reset_agent_monitor()
    return get_agent_monitor()


def _fresh_enforcer() -> PolicyEnforcer:
    reset_policy_enforcer()
    return get_policy_enforcer()


# ===========================================================================
# AgentMonitor tests
# ===========================================================================


class TestAgentMonitorSingleton(unittest.TestCase):
    def setUp(self):
        reset_agent_monitor()

    def tearDown(self):
        reset_agent_monitor()

    def test_singleton_same_instance(self):
        a = get_agent_monitor()
        b = get_agent_monitor()
        assert a is b

    def test_reset_yields_new_instance(self):
        a = get_agent_monitor()
        reset_agent_monitor()
        b = get_agent_monitor()
        assert a is not b


class TestAgentRegistration(unittest.TestCase):
    def setUp(self):
        reset_agent_monitor()
        self.monitor = get_agent_monitor()

    def tearDown(self):
        reset_agent_monitor()

    def test_register_agent_appears_in_stats(self):
        self.monitor.register_agent("agent-1")
        stats = self.monitor.get_stats()
        assert stats["registered_agents"] == 1

    def test_register_multiple_agents(self):
        self.monitor.register_agent("a1")
        self.monitor.register_agent("a2")
        self.monitor.register_agent("a3")
        assert self.monitor.get_stats()["registered_agents"] == 3

    def test_invalid_timeout_raises(self):
        with self.assertRaises(ValueError):
            self.monitor.register_agent("x", timeout_seconds=0)

    def test_invalid_max_steps_raises(self):
        with self.assertRaises(ValueError):
            self.monitor.register_agent("x", max_steps=0)


class TestHeartbeat(unittest.TestCase):
    def setUp(self):
        reset_agent_monitor()
        self.monitor = get_agent_monitor()

    def tearDown(self):
        reset_agent_monitor()

    def test_heartbeat_unregistered_raises(self):
        with self.assertRaises(KeyError):
            self.monitor.heartbeat("does-not-exist")

    def test_heartbeat_registered_succeeds(self):
        self.monitor.register_agent("worker-1")
        self.monitor.heartbeat("worker-1")  # should not raise

    def test_heartbeat_resets_timeout_clock(self):
        self.monitor.register_agent("worker-2", timeout_seconds=1.0)
        self.monitor.heartbeat("worker-2")
        # Immediately after heartbeat the agent should be healthy
        unhealthy = self.monitor.check_health()
        ids = [u["agent_id"] for u in unhealthy]
        assert "worker-2" not in ids


class TestStepLimit(unittest.TestCase):
    def setUp(self):
        reset_agent_monitor()
        self.monitor = get_agent_monitor()

    def tearDown(self):
        reset_agent_monitor()

    def test_step_limit_exceeded_raises(self):
        self.monitor.register_agent("limited", max_steps=3)
        self.monitor.record_step("limited")
        self.monitor.record_step("limited")
        self.monitor.record_step("limited")
        with self.assertRaises(StepLimitExceeded):
            self.monitor.record_step("limited")

    def test_step_limit_exception_is_agent_error(self):
        self.monitor.register_agent("sub", max_steps=1)
        self.monitor.record_step("sub")
        with self.assertRaises(AgentError):
            self.monitor.record_step("sub")

    def test_steps_within_limit_do_not_raise(self):
        self.monitor.register_agent("ok-agent", max_steps=10)
        for _ in range(10):
            self.monitor.record_step("ok-agent")  # should not raise

    def test_record_step_unregistered_raises(self):
        with self.assertRaises(KeyError):
            self.monitor.record_step("ghost")

    def test_total_steps_tracked_in_stats(self):
        self.monitor.register_agent("tracker", max_steps=20)
        self.monitor.record_step("tracker")
        self.monitor.record_step("tracker")
        stats = self.monitor.get_stats()
        assert stats["total_steps_recorded"] >= 2


class TestHealthCheck(unittest.TestCase):
    def setUp(self):
        reset_agent_monitor()
        self.monitor = get_agent_monitor()

    def tearDown(self):
        reset_agent_monitor()

    def test_healthy_agent_not_in_unhealthy_list(self):
        self.monitor.register_agent("healthy", timeout_seconds=300.0)
        self.monitor.heartbeat("healthy")
        unhealthy = self.monitor.check_health()
        assert all(u["agent_id"] != "healthy" for u in unhealthy)

    def test_stale_agent_detected(self):
        # Register with an extremely short timeout so it expires immediately
        self.monitor.register_agent("stale", timeout_seconds=0.0001)
        # Busy-wait long enough for the tiny timeout to expire
        time.sleep(0.01)
        unhealthy = self.monitor.check_health()
        ids = [u["agent_id"] for u in unhealthy]
        assert "stale" in ids

    def test_unhealthy_entry_has_expected_keys(self):
        self.monitor.register_agent("expired", timeout_seconds=0.0001)
        time.sleep(0.01)
        unhealthy = self.monitor.check_health()
        assert len(unhealthy) >= 1
        entry = next(u for u in unhealthy if u["agent_id"] == "expired")
        assert "reason" in entry
        assert "last_heartbeat_ago" in entry
        assert "timeout_seconds" in entry


class TestResetAgent(unittest.TestCase):
    def setUp(self):
        reset_agent_monitor()
        self.monitor = get_agent_monitor()

    def tearDown(self):
        reset_agent_monitor()

    def test_reset_clears_step_count(self):
        self.monitor.register_agent("r-agent", max_steps=2)
        self.monitor.record_step("r-agent")
        self.monitor.record_step("r-agent")
        # One more step would exceed limit — reset first
        self.monitor.reset_agent("r-agent")
        self.monitor.record_step("r-agent")  # should not raise

    def test_reset_unregistered_raises(self):
        with self.assertRaises(KeyError):
            self.monitor.reset_agent("nobody")


class TestMonitorStats(unittest.TestCase):
    def setUp(self):
        reset_agent_monitor()
        self.monitor = get_agent_monitor()

    def tearDown(self):
        reset_agent_monitor()

    def test_stats_keys_present(self):
        stats = self.monitor.get_stats()
        assert "registered_agents" in stats
        assert "total_steps_recorded" in stats
        assert "total_timeouts_detected" in stats
        assert "total_step_limit_violations" in stats
        assert "agents" in stats

    def test_step_limit_violation_counted(self):
        self.monitor.register_agent("v-agent", max_steps=1)
        self.monitor.record_step("v-agent")
        try:
            self.monitor.record_step("v-agent")
        except StepLimitExceeded:
            pass
        stats = self.monitor.get_stats()
        assert stats["total_step_limit_violations"] >= 1


# ===========================================================================
# PolicyEnforcer tests
# ===========================================================================


class TestPolicyEnforcerSingleton(unittest.TestCase):
    def setUp(self):
        reset_policy_enforcer()

    def tearDown(self):
        reset_policy_enforcer()

    def test_singleton_same_instance(self):
        a = get_policy_enforcer()
        b = get_policy_enforcer()
        assert a is b

    def test_reset_yields_new_instance(self):
        a = get_policy_enforcer()
        reset_policy_enforcer()
        b = get_policy_enforcer()
        assert a is not b


class TestFileJurisdiction(unittest.TestCase):
    def setUp(self):
        reset_policy_enforcer()
        self.enforcer = get_policy_enforcer()

    def tearDown(self):
        reset_policy_enforcer()

    def test_builder_allowed_in_vetinari(self):
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="write",
            target="vetinari/agents/foo.py",
            context={},
        )
        assert decision.allowed is True

    def test_builder_denied_outside_jurisdiction(self):
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="write",
            target="docs/architecture.md",
            context={},
        )
        assert decision.allowed is False
        assert "builder" in decision.reason

    def test_operations_allowed_in_docs(self):
        decision = self.enforcer.check_action(
            agent_type="operations",
            action="write",
            target="docs/README.md",
            context={},
        )
        assert decision.allowed is True

    def test_operations_denied_in_vetinari_source(self):
        decision = self.enforcer.check_action(
            agent_type="operations",
            action="write",
            target="vetinari/core.py",
            context={},
        )
        assert decision.allowed is False

    def test_read_action_skips_jurisdiction_check(self):
        # Researchers should be able to read any file
        decision = self.enforcer.check_action(
            agent_type="researcher",
            action="read",
            target="vetinari/types.py",
            context={},
        )
        assert decision.allowed is True


class TestDelegationDepth(unittest.TestCase):
    def setUp(self):
        reset_policy_enforcer()
        self.enforcer = get_policy_enforcer()

    def tearDown(self):
        reset_policy_enforcer()

    def test_within_depth_limit_allowed(self):
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="read",
            target="some/file.py",
            context={"delegation_depth": 2},
        )
        assert decision.allowed is True

    def test_exceeds_depth_limit_denied(self):
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="read",
            target="some/file.py",
            context={"delegation_depth": 4},
        )
        assert decision.allowed is False
        assert "depth" in decision.reason.lower()

    def test_custom_max_depth_respected(self):
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="read",
            target="some/file.py",
            context={"delegation_depth": 5, "max_delegation_depth": 10},
        )
        assert decision.allowed is True


class TestIrreversibility(unittest.TestCase):
    def setUp(self):
        reset_policy_enforcer()
        self.enforcer = get_policy_enforcer()

    def tearDown(self):
        reset_policy_enforcer()

    def test_delete_denied_by_default(self):
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="delete",
            target="vetinari/old_module.py",
            context={},
        )
        assert decision.allowed is False
        assert decision.risk_level == "high"

    def test_delete_allowed_with_explicit_flag(self):
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="delete",
            target="vetinari/old_module.py",
            context={"allow_destructive": True},
        )
        assert decision.allowed is True
        assert decision.risk_level == "medium"

    def test_drop_is_destructive(self):
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="drop",
            target="table_users",
            context={},
        )
        assert decision.allowed is False


class TestResourceBudget(unittest.TestCase):
    def setUp(self):
        reset_policy_enforcer()
        self.enforcer = get_policy_enforcer()

    def tearDown(self):
        reset_policy_enforcer()

    def test_over_token_budget_denied(self):
        decision = self.enforcer.check_action(
            agent_type="researcher",
            action="read",
            target="large_file.py",
            context={"tokens_used": 200_000, "max_tokens": 100_000},
        )
        assert decision.allowed is False
        assert "token" in decision.reason.lower()

    def test_over_time_budget_denied(self):
        decision = self.enforcer.check_action(
            agent_type="researcher",
            action="read",
            target="slow_op",
            context={"elapsed_seconds": 700.0, "max_time_seconds": 600.0},
        )
        assert decision.allowed is False
        assert "time" in decision.reason.lower()

    def test_within_budget_allowed(self):
        decision = self.enforcer.check_action(
            agent_type="researcher",
            action="read",
            target="file.py",
            context={"tokens_used": 1000, "elapsed_seconds": 5.0},
        )
        assert decision.allowed is True


class TestCustomPolicy(unittest.TestCase):
    def setUp(self):
        reset_policy_enforcer()
        self.enforcer = get_policy_enforcer()

    def tearDown(self):
        reset_policy_enforcer()

    def test_register_and_invoke_custom_policy(self):
        calls: list[tuple] = []

        def my_policy(agent_type, action, target, context):
            calls.append((agent_type, action, target))
            if target == "forbidden/path":
                return PolicyDecision(allowed=False, reason="custom block", risk_level="high")
            return None

        self.enforcer.register_policy("custom", my_policy)
        decision = self.enforcer.check_action(
            agent_type="builder",
            action="read",
            target="forbidden/path",
            context={},
        )
        assert decision.allowed is False
        assert decision.reason == "custom block"
        assert len(calls) >= 1


class TestPolicyDecisionDataclass(unittest.TestCase):
    def test_to_dict(self):
        d = PolicyDecision(allowed=True, reason="ok", risk_level="low").to_dict()
        assert d["allowed"] is True
        assert d["reason"] == "ok"
        assert d["risk_level"] == "low"


class TestEnforcerStats(unittest.TestCase):
    def setUp(self):
        reset_policy_enforcer()
        self.enforcer = get_policy_enforcer()

    def tearDown(self):
        reset_policy_enforcer()

    def test_stats_keys_present(self):
        stats = self.enforcer.get_stats()
        assert "registered_policies" in stats
        assert "total_checks" in stats
        assert "total_denied" in stats
        assert "total_flagged_irreversible" in stats

    def test_denied_count_increments(self):
        initial = self.enforcer.get_stats()["total_denied"]
        self.enforcer.check_action(
            agent_type="builder",
            action="write",
            target="docs/foo.md",
            context={},
        )
        assert self.enforcer.get_stats()["total_denied"] > initial

    def test_total_checks_increments(self):
        before = self.enforcer.get_stats()["total_checks"]
        self.enforcer.check_action("builder", "read", "any/file", {})
        self.enforcer.check_action("builder", "read", "any/file", {})
        after = self.enforcer.get_stats()["total_checks"]
        assert after == before + 2


# ===========================================================================
# Safety __init__ exports
# ===========================================================================


class TestSafetyModuleExports(unittest.TestCase):
    def test_imports_from_safety_package(self):
        from vetinari.safety import (  # noqa: F401
            get_agent_monitor,
            AgentMonitor,
            StepLimitExceeded,
            AgentTimeoutError,
            get_policy_enforcer,
            PolicyEnforcer,
            PolicyDecision,
            get_guardrails,
            GuardrailsManager,
            reset_guardrails,
        )


if __name__ == "__main__":
    unittest.main()
