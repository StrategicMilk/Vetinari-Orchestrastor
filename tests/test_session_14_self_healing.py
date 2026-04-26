"""Tests for Session 14 — Self-Healing Pipeline.

Covers all 7 items: remediation wiring (14.1), retry intelligence (14.2),
pipeline state (14.3), config self-tuning (14.4), graceful degradation (14.5),
circuit breaker wiring (14.6), and schema evolution (14.7).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.analytics.failure_registry import (
    FailureRegistry,
    PreventionRule,
    PreventionRuleType,
    reset_failure_registry,
)
from vetinari.config.self_tuning import (
    _DIVERGENCE_THRESHOLD,
    ConfigSelfTuner,
    TuningResult,
    reset_config_self_tuner,
)
from vetinari.orchestration.pipeline_state import (
    PipelineStateStore,
    _get_state_dir,
    reset_pipeline_state_store,
)
from vetinari.persistence.schema_evolution import (
    MigrationStep,
    SchemaRegistry,
    reset_schema_registry,
)
from vetinari.resilience.circuit_breaker import _AGENT_BREAKER_CONFIGS
from vetinari.resilience.degradation import (
    DegradationLevel,
    DegradationManager,
    FallbackEntry,
    SubsystemFallback,
    get_degradation_manager,
    reset_degradation_manager,
)
from vetinari.resilience.retry_intelligence import (
    RetryAnalyzer,
    RetryStrategy,
    reset_retry_analyzer,
)
from vetinari.system.remediation import (
    FailureMode,
    RemediationEngine,
    RemediationPlan,
    RemediationResult,
    RemediationTier,
)
from vetinari.types import AgentType

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def schema_registry():
    """Fresh SchemaRegistry for test isolation."""
    registry = SchemaRegistry()
    return registry


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all session-14 singletons before each test."""
    yield
    reset_schema_registry()
    reset_degradation_manager()
    reset_retry_analyzer()
    reset_pipeline_state_store()
    reset_config_self_tuner()
    reset_failure_registry()


@pytest.fixture
def degradation_mgr():
    """Fresh DegradationManager with default chains."""
    return DegradationManager()


@pytest.fixture
def retry_analyzer():
    """Fresh RetryAnalyzer instance."""
    return RetryAnalyzer()


@pytest.fixture
def pipeline_store(tmp_path, monkeypatch):
    """PipelineStateStore wired to a temp directory."""
    import vetinari.orchestration.pipeline_state as _ps_mod

    monkeypatch.setattr(_ps_mod, "_STATE_DIR", tmp_path / "pipeline-state")
    return PipelineStateStore()


@pytest.fixture
def self_tuner():
    """ConfigSelfTuner with a low threshold for testing."""
    return ConfigSelfTuner(tuning_threshold=3)


@pytest.fixture
def failure_registry(tmp_path, monkeypatch):
    """FailureRegistry wired to a temp directory for file I/O tests."""
    import vetinari.analytics.failure_registry as _fr_mod

    monkeypatch.setattr(_fr_mod, "_REGISTRY_DIR", tmp_path)
    return FailureRegistry()


# ── 14.7 Schema Evolution ──────────────────────────────────────────────────


class TestSchemaEvolution:
    """Tests for schema versioning, migration, and unknown field preservation."""

    def test_register_format_sets_current_version(self, schema_registry):
        schema_registry.register_format("test_format", current_version=3)
        assert schema_registry.get_current_version("test_format") == 3

    def test_unregistered_format_defaults_to_version_1(self, schema_registry):
        assert schema_registry.get_current_version("nonexistent") == 1

    def test_register_migration_increments_by_one(self, schema_registry):
        schema_registry.register_format("fmt", current_version=2)
        schema_registry.register_migration(
            "fmt",
            from_version=1,
            to_version=2,
            migrate_fn=lambda r: r,
            description="add field",
        )
        chain = schema_registry.get_migration_chain("fmt")
        assert len(chain) == 1
        assert chain[0].from_version == 1
        assert chain[0].to_version == 2

    def test_register_migration_rejects_non_sequential(self, schema_registry):
        schema_registry.register_format("fmt", current_version=3)
        with pytest.raises(ValueError, match="increment by 1"):
            schema_registry.register_migration(
                "fmt",
                from_version=1,
                to_version=3,
                migrate_fn=lambda r: r,
            )

    def test_migrate_record_applies_chain(self, schema_registry):
        schema_registry.register_format("fmt", current_version=3)
        schema_registry.register_migration(
            "fmt",
            1,
            2,
            migrate_fn=lambda r: {**r, "new_field": "added_v2"},
            description="add new_field",
        )
        schema_registry.register_migration(
            "fmt",
            2,
            3,
            migrate_fn=lambda r: {**r, "another": "added_v3"},
            description="add another",
        )
        record = {"data": "original", "schema_version": 1}
        result = schema_registry.migrate_record("fmt", record)

        assert result["schema_version"] == 3
        assert result["new_field"] == "added_v2"
        assert result["another"] == "added_v3"
        assert result["data"] == "original"

    def test_migrate_preserves_unknown_fields(self, schema_registry):
        """Unknown fields must never be silently dropped (14.7 requirement)."""
        schema_registry.register_format("fmt", current_version=2)
        schema_registry.register_migration(
            "fmt",
            1,
            2,
            migrate_fn=lambda r: {**r, "migrated": True},
        )
        record = {"schema_version": 1, "unknown_field": "keep_me", "extra": 42}
        result = schema_registry.migrate_record("fmt", record)

        assert result["unknown_field"] == "keep_me"
        assert result["extra"] == 42
        assert result["migrated"] is True

    def test_migrate_noop_when_already_current(self, schema_registry):
        schema_registry.register_format("fmt", current_version=2)
        record = {"schema_version": 2, "data": "ok"}
        result = schema_registry.migrate_record("fmt", record)
        assert result == record

    def test_stamp_record_sets_current_version(self, schema_registry):
        schema_registry.register_format("fmt", current_version=5)
        record = {"data": "new"}
        stamped = schema_registry.stamp_record("fmt", record)
        assert stamped["schema_version"] == 5

    def test_migrate_does_not_mutate_input(self, schema_registry):
        schema_registry.register_format("fmt", current_version=2)
        schema_registry.register_migration(
            "fmt",
            1,
            2,
            migrate_fn=lambda r: {**r, "added": True},
        )
        original = {"schema_version": 1, "data": "x"}
        schema_registry.migrate_record("fmt", original)
        assert "added" not in original


# ── 14.5 Graceful Degradation ──────────────────────────────────────────────


class TestGracefulDegradation:
    """Tests for per-subsystem fallback chains and level transitions."""

    @pytest.mark.parametrize(
        "subsystem",
        [
            "inference",
            "model_selection",
            "learning",
            "persistence",
        ],
    )
    def test_default_subsystems_start_at_primary(self, degradation_mgr, subsystem):
        level = degradation_mgr.get_current_level(subsystem)
        assert level == DegradationLevel.PRIMARY

    def test_get_degradation_manager_returns_singleton(self):
        first = get_degradation_manager()
        second = get_degradation_manager()

        assert first is second

    def test_get_fallback_degrades_one_level(self, degradation_mgr):
        entry = degradation_mgr.get_fallback("inference")
        assert isinstance(entry, FallbackEntry)
        assert entry.level == DegradationLevel.REDUCED
        assert degradation_mgr.get_current_level("inference") == DegradationLevel.REDUCED

    def test_successive_fallbacks_walk_down_chain(self, degradation_mgr):
        expected = [
            DegradationLevel.REDUCED,
            DegradationLevel.MINIMAL,
            DegradationLevel.CACHED,
            DegradationLevel.UNAVAILABLE,
        ]
        for expected_level in expected:
            entry = degradation_mgr.get_fallback("inference")
            assert entry.level == expected_level

    def test_fallback_at_unavailable_stays_unavailable(self, degradation_mgr):
        # Drive all the way down
        for _ in range(10):
            degradation_mgr.get_fallback("inference")
        assert degradation_mgr.get_current_level("inference") == DegradationLevel.UNAVAILABLE

    def test_report_recovery_restores_to_primary(self, degradation_mgr):
        degradation_mgr.get_fallback("inference")  # Degrade to REDUCED
        assert degradation_mgr.get_current_level("inference") == DegradationLevel.REDUCED

        level = degradation_mgr.report_recovery("inference")
        assert level == DegradationLevel.PRIMARY
        assert degradation_mgr.get_current_level("inference") == DegradationLevel.PRIMARY

    def test_unknown_subsystem_returns_none(self, degradation_mgr):
        assert degradation_mgr.get_fallback("nonexistent") is None
        assert degradation_mgr.get_current_level("nonexistent") is None

    def test_set_availability_skips_unavailable_level(self, degradation_mgr):
        degradation_mgr.set_availability(
            "inference",
            DegradationLevel.REDUCED,
            is_available=False,
        )
        # First fallback should skip REDUCED and go to MINIMAL
        entry = degradation_mgr.get_fallback("inference")
        assert entry.level == DegradationLevel.MINIMAL

    def test_get_status_returns_all_subsystems(self, degradation_mgr):
        status = degradation_mgr.get_status()
        assert "inference" in status
        assert "model_selection" in status
        assert status["inference"]["current_level"] == "primary"

    def test_user_message_empty_at_primary(self, degradation_mgr):
        msg = degradation_mgr.get_current_user_message("inference")
        assert msg == ""

    def test_user_message_populated_after_degradation(self, degradation_mgr):
        degradation_mgr.get_fallback("inference")  # -> REDUCED
        msg = degradation_mgr.get_current_user_message("inference")
        assert "smaller model" in msg.lower()

    @pytest.mark.parametrize("level", list(DegradationLevel))
    def test_degradation_level_enum_values(self, level):
        """Verify all DegradationLevel members are stable string values."""
        assert isinstance(level.value, str)
        assert level.value == level.name.lower()


# ── 14.2 Retry Intelligence ───────────────────────────────────────────────


class TestRetryIntelligence:
    """Tests for failure classification, registry lookup, and retry strategy."""

    @pytest.mark.parametrize(
        "text,expected_mode",
        [
            ("CUDA out of memory", "oom"),
            ("memory allocation failed for 4GB", "oom"),
            ("operation timed out after 30s", "hang"),
            ("deadlock detected in thread pool", "hang"),
            ("quality below threshold: 0.3", "quality_drop"),
            ("verification failed: output incomplete", "quality_drop"),
            ("no space left on device (ENOSPC)", "disk_full"),
            ("thermal throttling engaged", "thermal"),
            ("something completely unknown", ""),
        ],
    )
    def test_classify_failure_mode(self, text, expected_mode):
        assert RetryAnalyzer._classify_failure_mode(text) == expected_mode

    def test_novel_failure_returns_llm_brief_needed(self, retry_analyzer):
        """Unknown failures should flag for LLM analysis."""
        strategy = retry_analyzer.analyze(
            failure_trace="some unique error nobody has seen",
            error_msg="unique error",
            task_type="coding",
        )
        assert strategy.known is False
        assert strategy.llm_brief_needed is True

    @patch("vetinari.analytics.failure_registry.get_failure_registry")
    def test_matching_prevention_rule_returns_known_fix(self, mock_get_registry, retry_analyzer):
        """When a prevention rule matches, strategy should be known with high confidence."""
        mock_rule = MagicMock(spec=PreventionRule)
        mock_rule.matches.return_value = True
        mock_rule.rule_id = "prev_test001"
        mock_rule.category = "oom"
        mock_rule.description = "Reduce context window on OOM"

        mock_registry = MagicMock()
        mock_registry.get_prevention_rules.return_value = [mock_rule]
        mock_get_registry.return_value = mock_registry

        strategy = retry_analyzer.analyze(
            failure_trace="CUDA out of memory",
            error_msg="OOM",
        )
        assert strategy.known is True
        assert strategy.confidence >= 0.7
        assert strategy.matching_rule_id == "prev_test001"

    @patch("vetinari.analytics.failure_registry.get_failure_registry")
    def test_high_confidence_remediation_stats(self, mock_get_registry, retry_analyzer):
        """When remediation stats show high success rate, return known fix."""
        mock_registry = MagicMock()
        mock_registry.get_prevention_rules.return_value = []  # No rule match
        mock_registry.get_remediation_stats.return_value = {
            ("oom", "reduce_context"): {"success": 9, "failure": 1},
        }
        mock_get_registry.return_value = mock_registry

        strategy = retry_analyzer.analyze(
            failure_trace="out of memory during inference",
            error_msg="OOM error",
        )
        assert strategy.known is True
        assert strategy.confidence == pytest.approx(0.9, abs=0.01)
        assert strategy.fix_action == "reduce_context"

    def test_retry_strategy_repr_known(self):
        s = RetryStrategy(known=True, fix_action="restart", confidence=0.85)
        assert "known=True" in repr(s)
        assert "restart" in repr(s)

    def test_retry_strategy_repr_unknown(self):
        s = RetryStrategy(known=False, llm_brief_needed=True)
        assert "known=False" in repr(s)
        assert "llm_brief_needed=True" in repr(s)


# ── 14.3 Pipeline State ───────────────────────────────────────────────────


class TestPipelineState:
    """Tests for stage checkpointing, resume, and crash recovery."""

    def test_mark_and_get_resume_point(self, pipeline_store):
        pipeline_store.mark_stage_complete("task_1", "intake", {"items": 5})
        resume = pipeline_store.get_resume_point("task_1")
        assert resume is not None
        stage_name, snapshot = resume
        assert stage_name == "intake"
        assert snapshot["items"] == 5

    def test_resume_returns_last_stage(self, pipeline_store):
        pipeline_store.mark_stage_complete("task_1", "intake")
        pipeline_store.mark_stage_complete("task_1", "plan_gen", {"plan_id": "p1"})
        pipeline_store.mark_stage_complete("task_1", "execution", {"completed": 3})

        stage_name, snapshot = pipeline_store.get_resume_point("task_1")
        assert stage_name == "execution"
        assert snapshot["completed"] == 3

    def test_get_completed_stages_returns_ordered_list(self, pipeline_store):
        pipeline_store.mark_stage_complete("task_1", "intake")
        pipeline_store.mark_stage_complete("task_1", "plan_gen")
        pipeline_store.mark_stage_complete("task_1", "execution")

        stages = pipeline_store.get_completed_stages("task_1")
        assert stages == ["intake", "plan_gen", "execution"]

    def test_no_state_returns_none(self, pipeline_store):
        assert pipeline_store.get_resume_point("nonexistent") is None

    def test_no_state_returns_empty_stages(self, pipeline_store):
        assert pipeline_store.get_completed_stages("nonexistent") == []

    def test_clear_state_removes_file(self, pipeline_store):
        pipeline_store.mark_stage_complete("task_1", "intake")
        assert pipeline_store.get_resume_point("task_1") is not None

        pipeline_store.clear_state("task_1")
        assert pipeline_store.get_resume_point("task_1") is None

    def test_mark_stage_is_idempotent(self, pipeline_store):
        """Re-marking the same stage updates it rather than duplicating."""
        pipeline_store.mark_stage_complete("task_1", "intake", {"v": 1})
        pipeline_store.mark_stage_complete("task_1", "intake", {"v": 2})

        stages = pipeline_store.get_completed_stages("task_1")
        assert stages == ["intake"]  # Not duplicated

        _, snapshot = pipeline_store.get_resume_point("task_1")
        assert snapshot["v"] == 2  # Updated to latest

    def test_state_persists_to_disk(self, pipeline_store, tmp_path):
        pipeline_store.mark_stage_complete("task_42", "intake", {"ok": True})

        state_file = tmp_path / "pipeline-state" / "task_42.json"
        assert state_file.exists()

        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["task_id"] == "task_42"
        assert data["schema_version"] == 1
        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage"] == "intake"

    def test_separate_tasks_have_separate_state(self, pipeline_store):
        pipeline_store.mark_stage_complete("task_a", "intake")
        pipeline_store.mark_stage_complete("task_b", "plan_gen")

        assert pipeline_store.get_completed_stages("task_a") == ["intake"]
        assert pipeline_store.get_completed_stages("task_b") == ["plan_gen"]


# ── 14.4 Config Self-Tuning ───────────────────────────────────────────────


class TestConfigSelfTuning:
    """Tests for task counting, divergence detection, and auto-tuning."""

    def test_record_below_threshold_returns_none(self, self_tuner):
        result = self_tuner.record_task_completion("coding")
        assert result is None

    def test_counter_increments(self, self_tuner):
        self_tuner.record_task_completion("coding")
        self_tuner.record_task_completion("coding")
        counts = self_tuner.get_task_counts()
        assert counts["coding"] == 2

    @patch.object(ConfigSelfTuner, "check_and_tune")
    def test_threshold_triggers_tuning(self, mock_tune, self_tuner):
        """After threshold tasks, check_and_tune should be called."""
        mock_tune.return_value = TuningResult(
            task_type="coding",
            tuned=False,
            changes={},
            task_count=3,
        )
        for _ in range(2):
            self_tuner.record_task_completion("coding")
        # Third call hits threshold=3
        result = self_tuner.record_task_completion("coding")
        assert result is not None
        mock_tune.assert_called_once_with("coding", task_count=3)

    def test_counter_resets_after_threshold(self, self_tuner):
        with patch.object(ConfigSelfTuner, "check_and_tune") as mock_tune:
            mock_tune.return_value = TuningResult(
                task_type="coding",
                tuned=False,
                changes={},
                task_count=3,
            )
            for _ in range(3):
                self_tuner.record_task_completion("coding")
        counts = self_tuner.get_task_counts()
        assert counts["coding"] == 0

    @pytest.mark.parametrize(
        "current,learned,expected_above",
        [
            (1.0, 1.0, False),  # Identical — no divergence
            (1.0, 1.3, True),  # 30% divergence — above 20% threshold
            (100, 115, False),  # 15% divergence — below threshold
            (0.5, 0.7, True),  # 40% divergence — above threshold
        ],
    )
    def test_compute_divergence(self, current, learned, expected_above):
        div = ConfigSelfTuner._compute_divergence(current, learned)
        if expected_above:
            assert div > _DIVERGENCE_THRESHOLD
        else:
            assert div <= _DIVERGENCE_THRESHOLD

    def test_tuning_result_repr(self):
        r = TuningResult(task_type="coding", tuned=True, changes={"temp": {}}, task_count=100)
        assert "coding" in repr(r)
        assert "tuned=True" in repr(r)

    def test_separate_task_types_track_independently(self, self_tuner):
        self_tuner.record_task_completion("coding")
        self_tuner.record_task_completion("coding")
        self_tuner.record_task_completion("analysis")

        counts = self_tuner.get_task_counts()
        assert counts["coding"] == 2
        assert counts["analysis"] == 1


# ── 14.1 Remediation Wiring ───────────────────────────────────────────────


class TestRemediationWiring:
    """Tests for remediation outcome logging to the failure registry (14.1)."""

    def test_log_remediation_outcome_writes_to_file(self, failure_registry, tmp_path):
        failure_registry.log_remediation_outcome(
            failure_mode="oom",
            action_description="reduce context window",
            success=True,
        )
        outcomes_file = tmp_path / "remediation-outcomes.jsonl"
        assert outcomes_file.exists()

        lines = outcomes_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["failure_mode"] == "oom"
        assert record["action_description"] == "reduce context window"
        assert record["success"] is True
        assert record["schema_version"] == 1

    def test_get_remediation_stats_aggregates_outcomes(self, failure_registry):
        failure_registry.log_remediation_outcome("oom", "reduce_ctx", success=True)
        failure_registry.log_remediation_outcome("oom", "reduce_ctx", success=True)
        failure_registry.log_remediation_outcome("oom", "reduce_ctx", success=False)
        failure_registry.log_remediation_outcome("hang", "restart", success=True)

        stats = failure_registry.get_remediation_stats()
        assert stats["oom", "reduce_ctx"] == {"success": 2, "failure": 1}
        assert stats["hang", "restart"] == {"success": 1, "failure": 0}

    def test_get_remediation_confidence_computes_rate(self, failure_registry):
        for _ in range(8):
            failure_registry.log_remediation_outcome("oom", "fix_a", success=True)
        for _ in range(2):
            failure_registry.log_remediation_outcome("oom", "fix_a", success=False)

        confidence = failure_registry.get_remediation_confidence("oom", "fix_a")
        assert confidence == pytest.approx(0.8, abs=0.01)

    def test_confidence_zero_when_no_outcomes(self, failure_registry):
        assert failure_registry.get_remediation_confidence("oom", "nonexistent") == 0.0

    def test_confidence_increases_after_10_successes(self, failure_registry):
        """14.1 requirement: after 10 successful fixes, confidence should be high."""
        for _ in range(10):
            failure_registry.log_remediation_outcome("oom", "reduce_ctx", success=True)
        failure_registry.log_remediation_outcome("oom", "reduce_ctx", success=False)

        confidence = failure_registry.get_remediation_confidence("oom", "reduce_ctx")
        assert confidence > 0.9  # 10/11 ~ 0.909

    def test_remediation_engine_log_outcome_calls_registry(self):
        """Verify _log_outcome_to_registry delegates to failure registry."""
        engine = RemediationEngine()
        plan = RemediationPlan(
            failure_mode=FailureMode.OOM,
            diagnosis="Test diagnosis",
            actions=[],
            max_tier=RemediationTier.AUTO_FIX,
        )
        result = RemediationResult(
            success=True,
            failure_mode=FailureMode.OOM,
            tier_reached=RemediationTier.AUTO_FIX,
            actions_taken=["action_a", "action_b"],
        )

        with patch("vetinari.analytics.failure_registry.get_failure_registry") as mock_get_reg:
            mock_registry = MagicMock()
            mock_get_reg.return_value = mock_registry

            engine._log_outcome_to_registry(plan, result)

            assert mock_registry.log_remediation_outcome.call_count == 2
            mock_registry.log_remediation_outcome.assert_any_call(
                failure_mode="oom",
                action_description="action_a",
                success=True,
            )
            mock_registry.log_remediation_outcome.assert_any_call(
                failure_mode="oom",
                action_description="action_b",
                success=True,
            )


# ── 14.6 Circuit Breaker Wiring ───────────────────────────────────────────


class TestCircuitBreakerConfigs:
    """Verify circuit breaker configs cover all required call sites."""

    @pytest.mark.parametrize(
        "agent_type",
        [
            AgentType.FOREMAN,
            AgentType.WORKER,
            AgentType.INSPECTOR,
        ],
    )
    def test_agent_type_has_breaker_config(self, agent_type):
        assert agent_type.value in _AGENT_BREAKER_CONFIGS, f"Missing circuit breaker config for {agent_type.value}"

    @pytest.mark.parametrize("key", ["model_scout", "external_api"])
    def test_non_agent_breaker_configs_exist(self, key):
        assert key in _AGENT_BREAKER_CONFIGS, f"Missing circuit breaker config for {key}"

    def test_all_configs_have_positive_thresholds(self):
        for name, config in _AGENT_BREAKER_CONFIGS.items():
            assert config.failure_threshold > 0, f"{name} has non-positive failure_threshold"
            assert config.recovery_timeout > 0, f"{name} has non-positive recovery_timeout"
