"""Comprehensive tests for Session 1B security work.

Covers:
- SecretsFilter (vetinari.learning.secrets_filter)
- ResourceMonitor (vetinari.system.resource_monitor)
- PolicyEnforcer (vetinari.safety.policy_enforcer)
- UnifiedPermissionModel / enforce_all (vetinari.enforcement)
- RateLimiter (vetinari.web.rate_limiter)
- SSE improvements (vetinari.web.shared)
- RaceConditionFixes (vetinari.web.shared)
- FallbackDetection (vetinari.learning.training_data)
"""

from __future__ import annotations

import threading

import pytest

from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# 1. SecretsFilter
# ---------------------------------------------------------------------------


class TestSecretsFilter:
    """Tests for vetinari.learning.secrets_filter."""

    def test_detects_aws_access_key(self) -> None:
        """AWS AKIA-style key is detected by scan_text."""
        from vetinari.learning.secrets_filter import scan_text

        text = "Access key: AKIAIOSFODNN7EXAMPLE is in the config"
        detections = scan_text(text)
        labels = [d.pattern_label for d in detections]
        assert "aws_access_key" in labels

    def test_detects_github_pat(self) -> None:
        """GitHub classic PAT (ghp_ + 36 chars) is detected."""
        from vetinari.learning.secrets_filter import scan_text

        text = "found ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij in code"
        detections = scan_text(text)
        labels = [d.pattern_label for d in detections]
        assert "github_pat_classic" in labels

    def test_detects_pem_block(self) -> None:
        """PEM private key header is detected."""
        from vetinari.learning.secrets_filter import scan_text

        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQ..."
        detections = scan_text(text)
        labels = [d.pattern_label for d in detections]
        assert "pem_private_key" in labels

    def test_clean_text_passes(self) -> None:
        """Ordinary prose without secrets returns an empty detection list."""
        from vetinari.learning.secrets_filter import scan_text

        text = "The quick brown fox jumps over the lazy dog."
        detections = scan_text(text)
        assert detections == []

    def test_filter_training_record_blocks_secrets(self) -> None:
        """filter_training_record returns (False, detections) when secrets found."""
        from vetinari.learning.secrets_filter import filter_training_record

        prompt = "Use AKIAIOSFODNN7EXAMPLE to authenticate."
        response = "Here is the result."
        is_safe, detections = filter_training_record(prompt, response)
        assert is_safe is False
        assert len(detections) >= 1

    def test_filter_training_record_allows_clean_record(self) -> None:
        """filter_training_record returns (True, []) for clean content."""
        from vetinari.learning.secrets_filter import filter_training_record

        is_safe, detections = filter_training_record(
            "Explain Python list comprehensions.",
            "A list comprehension creates a new list by applying an expression.",
        )
        assert is_safe is True
        assert detections == []

    def test_is_blocked_file_docker_compose(self) -> None:
        """docker-compose.yml is always blocked."""
        from vetinari.learning.secrets_filter import is_blocked_file

        assert is_blocked_file("docker-compose.yml") is True

    def test_is_blocked_file_credentials_substring(self) -> None:
        """Files containing 'credentials' in the name are blocked."""
        from vetinari.learning.secrets_filter import is_blocked_file

        assert is_blocked_file("aws_credentials.cfg") is True

    def test_is_blocked_file_secret_substring(self) -> None:
        """Files containing 'secret' in the name are blocked."""
        from vetinari.learning.secrets_filter import is_blocked_file

        assert is_blocked_file("app_secret_config.json") is True

    def test_is_blocked_file_safe_file(self) -> None:
        """A normal Python module is not blocked."""
        from vetinari.learning.secrets_filter import is_blocked_file

        assert is_blocked_file("vetinari/agents/base_agent.py") is False

    def test_redact_secrets_replaces_aws_key(self) -> None:
        """redact_secrets substitutes AWS key with [REDACTED]."""
        from vetinari.learning.secrets_filter import redact_secrets

        text = "key=AKIAIOSFODNN7EXAMPLE rest of text"
        redacted = redact_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED]" in redacted

    def test_high_entropy_detection(self) -> None:
        """A 30-character random-looking token triggers high_entropy detection."""
        from vetinari.learning.secrets_filter import scan_text

        # 30 non-repeating characters — very high Shannon entropy
        text = "token=aB3dEfGhIjKlMnOpQrStUvWxYz12"
        detections = scan_text(text)
        labels = [d.pattern_label for d in detections]
        assert "high_entropy" in labels


# ---------------------------------------------------------------------------
# 2. ResourceMonitor
# ---------------------------------------------------------------------------


class TestResourceMonitor:
    """Tests for vetinari.system.resource_monitor."""

    def setup_method(self) -> None:
        """Reset singleton before each test to ensure clean state."""
        from vetinari.system.resource_monitor import reset_resource_monitor

        reset_resource_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test to avoid contaminating siblings."""
        from vetinari.system.resource_monitor import reset_resource_monitor

        reset_resource_monitor()

    def test_check_disk_space_returns_disk_status(self) -> None:
        """check_disk_space returns a DiskStatus with positive total_bytes."""
        from vetinari.system.resource_monitor import check_disk_space

        status = check_disk_space()
        assert status.total_bytes > 0
        assert status.usage_percent >= 0.0
        assert status.path != ""

    def test_threshold_ok_at_50_percent(self) -> None:
        """50% usage maps to DiskThreshold.OK."""
        from vetinari.system.resource_monitor import DiskThreshold, _classify_usage

        assert _classify_usage(50.0) == DiskThreshold.OK

    def test_threshold_warn_at_85_percent(self) -> None:
        """85% usage maps to DiskThreshold.WARN."""
        from vetinari.system.resource_monitor import DiskThreshold, _classify_usage

        assert _classify_usage(85.0) == DiskThreshold.WARN

    def test_threshold_pause_at_92_percent(self) -> None:
        """92% usage maps to DiskThreshold.PAUSE."""
        from vetinari.system.resource_monitor import DiskThreshold, _classify_usage

        assert _classify_usage(92.0) == DiskThreshold.PAUSE

    def test_threshold_read_only_at_97_percent(self) -> None:
        """97% usage maps to DiskThreshold.READ_ONLY."""
        from vetinari.system.resource_monitor import DiskThreshold, _classify_usage

        assert _classify_usage(97.0) == DiskThreshold.READ_ONLY

    def test_disk_status_is_ok_property(self) -> None:
        """DiskStatus.is_ok is True only when threshold is OK."""
        from vetinari.system.resource_monitor import DiskStatus, DiskThreshold

        ok_status = DiskStatus(
            total_bytes=100,
            used_bytes=50,
            free_bytes=50,
            usage_percent=50.0,
            threshold=DiskThreshold.OK,
            path="/",
        )
        assert ok_status.is_ok is True

        warn_status = DiskStatus(
            total_bytes=100,
            used_bytes=85,
            free_bytes=15,
            usage_percent=85.0,
            threshold=DiskThreshold.WARN,
            path="/",
        )
        assert warn_status.is_ok is False

    def test_disk_status_should_pause_writes(self) -> None:
        """should_pause_writes is True for PAUSE and READ_ONLY thresholds."""
        from vetinari.system.resource_monitor import DiskStatus, DiskThreshold

        for threshold in (DiskThreshold.PAUSE, DiskThreshold.READ_ONLY):
            status = DiskStatus(
                total_bytes=100,
                used_bytes=92,
                free_bytes=8,
                usage_percent=92.0,
                threshold=threshold,
                path="/",
            )
            assert status.should_pause_writes is True

        ok_status = DiskStatus(
            total_bytes=100,
            used_bytes=50,
            free_bytes=50,
            usage_percent=50.0,
            threshold=DiskThreshold.OK,
            path="/",
        )
        assert ok_status.should_pause_writes is False

    def test_disk_status_is_read_only(self) -> None:
        """is_read_only is True only for READ_ONLY threshold."""
        from vetinari.system.resource_monitor import DiskStatus, DiskThreshold

        ro_status = DiskStatus(
            total_bytes=100,
            used_bytes=97,
            free_bytes=3,
            usage_percent=97.0,
            threshold=DiskThreshold.READ_ONLY,
            path="/",
        )
        assert ro_status.is_read_only is True

        pause_status = DiskStatus(
            total_bytes=100,
            used_bytes=92,
            free_bytes=8,
            usage_percent=92.0,
            threshold=DiskThreshold.PAUSE,
            path="/",
        )
        assert pause_status.is_read_only is False

    def test_resource_monitor_caching(self) -> None:
        """ResourceMonitor.check() returns the same object on immediate second call."""
        from vetinari.system.resource_monitor import get_resource_monitor

        monitor = get_resource_monitor()
        first = monitor.check()
        second = monitor.check()
        # Cached result must be the same object within the TTL window
        assert first is second

    def test_resource_monitor_singleton(self) -> None:
        """get_resource_monitor() always returns the same instance."""
        from vetinari.system.resource_monitor import get_resource_monitor

        assert get_resource_monitor() is get_resource_monitor()


# ---------------------------------------------------------------------------
# 3. PolicyEnforcer
# ---------------------------------------------------------------------------


class TestPolicyEnforcer:
    """Tests for vetinari.safety.policy_enforcer."""

    def setup_method(self) -> None:
        from vetinari.safety.policy_enforcer import reset_policy_enforcer

        reset_policy_enforcer()

    def teardown_method(self) -> None:
        from vetinari.safety.policy_enforcer import reset_policy_enforcer

        reset_policy_enforcer()

    def test_jurisdiction_uses_canonical_agent_types(self) -> None:
        """The jurisdiction map uses canonical AgentType values, not legacy strings."""
        from vetinari.safety.policy_enforcer import _JURISDICTION

        assert AgentType.WORKER.value.lower() in _JURISDICTION
        assert "builder" not in _JURISDICTION
        assert "planner" not in _JURISDICTION

    def test_worker_can_write_to_vetinari(self) -> None:
        """Agent 'builder' is permitted to write to vetinari/ paths."""
        from vetinari.safety.policy_enforcer import get_policy_enforcer

        enforcer = get_policy_enforcer()
        decision = enforcer.check_action(
            agent_type=AgentType.WORKER,
            action="write",
            target="vetinari/agents/new_agent.py",
            context={},
        )
        assert decision.allowed is True

    def test_foreman_cannot_write_to_vetinari(self) -> None:
        """Agent 'planner' (Foreman's map key) is not permitted to write vetinari/ files."""
        from vetinari.safety.policy_enforcer import get_policy_enforcer

        enforcer = get_policy_enforcer()
        decision = enforcer.check_action(
            agent_type=AgentType.FOREMAN,
            action="write",
            target="vetinari/core/main.py",
            context={},
        )
        assert decision.allowed is False

    def test_check_action_accepts_agent_type_value_string(self) -> None:
        """check_action accepts the .value string of AgentType enum directly."""
        from vetinari.safety.policy_enforcer import get_policy_enforcer

        enforcer = get_policy_enforcer()
        # AgentType.WORKER.value is "WORKER" — lowercased by the enforcer
        decision = enforcer.check_action(
            agent_type=AgentType.WORKER.value,
            action="read",
            target="vetinari/agents/something.py",
            context={},
        )
        # Read is not subject to jurisdiction — all pass through
        assert decision.allowed is True

    def test_policy_decision_is_dataclass(self) -> None:
        """PolicyDecision exposes allowed, reason, and risk_level."""
        from vetinari.safety.policy_enforcer import PolicyDecision

        decision = PolicyDecision(allowed=True, reason="test", risk_level="low")
        assert decision.allowed is True
        assert "test" in decision.reason
        assert decision.risk_level == "low"

    def test_get_stats_returns_dict(self) -> None:
        """get_stats() returns a dictionary with the expected keys."""
        from vetinari.safety.policy_enforcer import get_policy_enforcer

        stats = get_policy_enforcer().get_stats()
        assert "total_checks" in stats
        assert "total_denied" in stats
        assert "registered_policies" in stats


# ---------------------------------------------------------------------------
# 4. UnifiedPermissionModel — enforce_all
# ---------------------------------------------------------------------------


class TestUnifiedPermissionModel:
    """Tests for vetinari.enforcement.enforce_all."""

    def test_both_allow_returns_none(self) -> None:
        """enforce_all with valid depth and quality returns None (no exception raised)."""
        from vetinari.enforcement import enforce_all

        result = enforce_all(
            agent_type=AgentType.WORKER,
            current_depth=1,
            quality_score=0.95,
        )
        assert result is None

    def test_agent_jurisdiction_deny_raises(self) -> None:
        """enforce_all raises JurisdictionViolation when file is outside jurisdiction."""
        from vetinari.enforcement import enforce_all
        from vetinari.exceptions import JurisdictionViolation

        with pytest.raises(JurisdictionViolation):
            enforce_all(
                agent_type=AgentType.INSPECTOR,
                file_path="vetinari/core/main.py",  # Inspector cannot modify core
            )

    def test_enforce_all_raises_security_error_on_deny(self) -> None:
        """enforce_all raises a subclass of VetinariError when checks fail."""
        from vetinari.enforcement import enforce_all
        from vetinari.exceptions import VetinariError

        with pytest.raises(VetinariError):
            enforce_all(
                agent_type=AgentType.INSPECTOR,
                file_path="vetinari/core/main.py",
            )

    def test_enforce_all_worker_valid_file(self) -> None:
        """Worker is permitted to access its own jurisdiction files — enforce_all returns None."""
        from vetinari.enforcement import enforce_all

        # Worker has jurisdiction over vetinari/agents/consolidated/
        result = enforce_all(
            agent_type=AgentType.WORKER,
            file_path="vetinari/agents/consolidated/worker_agent.py",
        )
        assert result is None


# ---------------------------------------------------------------------------
# 6. SSE improvements
# ---------------------------------------------------------------------------


class TestSSEImprovements:
    """Tests for sequence numbers and dropped-event tracking in vetinari.web.shared."""

    def _cleanup(self, project_id: str) -> None:
        """Remove SSE state created by a test to avoid cross-test contamination."""
        from vetinari.web import shared

        with shared._sse_streams_lock:
            shared._sse_streams.pop(project_id, None)
            shared._sse_sequence_counters.pop(project_id, None)
            shared._sse_dropped_counts.pop(project_id, None)

    def test_events_have_sequence_numbers(self) -> None:
        """Events pushed via _push_sse_event carry a monotonically increasing 'id'."""
        from vetinari.web import shared

        pid = "sse-test-seq-001"
        try:
            q = shared._get_sse_queue(pid)
            shared._push_sse_event(pid, "test_event", {"msg": "hello"})

            event = q.get_nowait()
            assert "id" in event
            assert int(event["id"]) >= 1
        finally:
            self._cleanup(pid)

    def test_sequence_numbers_increment(self) -> None:
        """Subsequent events for the same project get strictly increasing sequence IDs."""
        from vetinari.web import shared

        pid = "sse-test-seq-002"
        try:
            q = shared._get_sse_queue(pid)
            shared._push_sse_event(pid, "event_a", {"n": 1})
            shared._push_sse_event(pid, "event_b", {"n": 2})

            first = q.get_nowait()
            second = q.get_nowait()
            assert int(second["id"]) > int(first["id"])
        finally:
            self._cleanup(pid)

    def test_dropped_events_tracked_on_full_queue(self) -> None:
        """When the SSE queue is full, _sse_dropped_counts is incremented."""
        import queue as _queue

        from vetinari.web import shared

        pid = "sse-test-drop-003"
        try:
            # Install a tiny queue that fills immediately
            with shared._sse_streams_lock:
                shared._sse_streams[pid] = _queue.Queue(maxsize=1)
                shared._sse_sequence_counters[pid] = 0
                shared._sse_dropped_counts[pid] = 0

            # Fill the queue with one event
            shared._push_sse_event(pid, "fill", {"x": 1})
            # This one should be dropped
            shared._push_sse_event(pid, "overflow", {"x": 2})

            with shared._sse_streams_lock:
                dropped = shared._sse_dropped_counts.get(pid, 0)
            assert dropped >= 1
        finally:
            self._cleanup(pid)


# ---------------------------------------------------------------------------
# 7. RaceConditionFixes
# ---------------------------------------------------------------------------


class TestRaceConditionFixes:
    """Tests for lock-guarded shared state in vetinari.web.shared."""

    def test_cancel_sets_flag_under_lock(self) -> None:
        """_cancel_project_task atomically sets the flag returned by _register_project_task."""
        from vetinari.web.shared import _cancel_project_task, _register_project_task

        pid = "race-test-cancel-001"
        flag = _register_project_task(pid)
        assert not flag.is_set()
        result = _cancel_project_task(pid)
        assert result is True
        assert flag.is_set()

    def test_set_orchestrator_uses_lock(self) -> None:
        """set_orchestrator replaces the singleton safely under the module lock."""
        from vetinari.web import shared

        sentinel = object()
        shared.set_orchestrator(sentinel)
        assert shared.orchestrator is sentinel

        # Restore to None so subsequent tests are unaffected
        shared.set_orchestrator(None)

    def test_concurrent_cancel_is_safe(self) -> None:
        """Multiple threads calling _cancel_project_task on different projects never deadlock."""
        from vetinari.web.shared import _cancel_project_task, _register_project_task

        pids = [f"race-concurrent-{i}" for i in range(10)]
        flags = [_register_project_task(p) for p in pids]

        errors: list[Exception] = []

        def cancel_all() -> None:
            for p in pids:
                try:
                    _cancel_project_task(p)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=cancel_all) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert errors == []
        assert all(f.is_set() for f in flags)


# ---------------------------------------------------------------------------
# 8. FallbackDetection
# ---------------------------------------------------------------------------


class TestFallbackDetection:
    """Tests for _is_fallback rejection in TrainingDataCollector."""

    def test_records_with_is_fallback_metadata_are_rejected(self) -> None:
        """A record whose metadata contains _is_fallback=True must not be stored."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector(output_path="/dev/null", sync=True)
        initial_count = collector._record_count

        collector.record(
            task="Test fallback rejection",
            prompt="Some prompt text",
            response="Some response text",
            score=0.9,
            model_id="test-model",
            task_type="general",
            latency_ms=100,
            tokens_used=50,
            metadata={"_is_fallback": True},
        )

        # _record_count must not have changed
        assert collector._record_count == initial_count

    def test_records_with_zero_tokens_are_rejected(self) -> None:
        """Records with tokens_used=0 indicate a mock response and must be rejected."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector(output_path="/dev/null", sync=True)
        initial_count = collector._record_count

        collector.record(
            task="Zero tokens test",
            prompt="Prompt",
            response="Response",
            score=0.8,
            model_id="test-model",
            task_type="general",
            latency_ms=100,
            tokens_used=0,  # rejected
        )

        assert collector._record_count == initial_count

    def test_records_with_zero_latency_are_rejected(self) -> None:
        """Records with latency_ms=0 indicate a mock response and must be rejected."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector(output_path="/dev/null", sync=True)
        initial_count = collector._record_count

        collector.record(
            task="Zero latency test",
            prompt="Prompt",
            response="Response",
            score=0.8,
            model_id="test-model",
            task_type="general",
            latency_ms=0,  # rejected
            tokens_used=100,
        )

        assert collector._record_count == initial_count

    def test_valid_record_is_accepted(self, tmp_path) -> None:
        """A record with proper latency, token count, and no fallback flag is stored."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector(output_path=str(tmp_path / "training.jsonl"), sync=True)
        initial_count = collector._record_count

        collector.record(
            task="Valid record acceptance test",
            prompt="Explain what a unit test is.",
            response="A unit test validates a single unit of code in isolation.",
            score=0.9,
            model_id="test-model",
            task_type="general",
            latency_ms=250,
            tokens_used=80,
            metadata={},
        )

        # _record_count increments when _append succeeds against a real writable path.
        assert collector._record_count == initial_count + 1

    @pytest.mark.parametrize(
        "response",
        [
            "",
            "{}",
            '{"content":"","sections":[]}',
            '{"content": "", "sections": []}',
        ],
    )
    def test_known_fallback_response_patterns_rejected(self, response: str) -> None:
        """Records matching known fallback response strings are discarded."""
        from vetinari.learning.training_data import TrainingDataCollector

        collector = TrainingDataCollector(output_path="/dev/null", sync=True)
        initial_count = collector._record_count

        collector.record(
            task="Fallback pattern test",
            prompt="Some prompt",
            response=response,
            score=0.5,
            model_id="test-model",
            task_type="general",
            latency_ms=100,
            tokens_used=10,
        )

        assert collector._record_count == initial_count
