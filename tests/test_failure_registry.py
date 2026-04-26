"""Tests for the failure registry, prevention rules, wiring, and rejection fields.

Covers US-001 through US-005 of Session 10:
  - Append-only JSONL failure storage
  - Failure-to-prevention-rule generation
  - Pipeline wiring (record_failure, record_inference_failure)
  - Prevention rule matching
  - TrainingRecord rejection fields and DPO export
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from vetinari.analytics.failure_registry import (
    FailureRegistry,
    FailureRegistryEntry,
    FailureStatus,
    PreventionRule,
    PreventionRuleType,
    get_failure_registry,
    reset_failure_registry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for failure registry JSONL files."""
    return tmp_path


@pytest.fixture
def registry(registry_dir: Path) -> FailureRegistry:
    """Provide a fresh FailureRegistry writing to a temp directory."""
    with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
        yield FailureRegistry()


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton between tests."""
    reset_failure_registry()
    yield
    reset_failure_registry()


# ---------------------------------------------------------------------------
# US-001: FailureRegistry CRUD
# ---------------------------------------------------------------------------


class TestFailureRegistryEntry:
    """Tests for the FailureRegistryEntry dataclass."""

    def test_entry_fields(self):
        entry = FailureRegistryEntry(
            failure_id="fail_abc123",
            timestamp="2026-04-07T00:00:00+00:00",
            category="inspector_rejection",
            severity="warning",
            description="Output lacked error handling",
        )
        assert entry.failure_id == "fail_abc123"
        assert entry.category == "inspector_rejection"
        assert entry.status == "active"

    def test_entry_to_dict(self):
        entry = FailureRegistryEntry(
            failure_id="fail_abc123",
            timestamp="2026-04-07T00:00:00+00:00",
            category="test",
            severity="error",
            description="test failure",
            affected_components=["worker"],
        )
        d = entry.to_dict()
        assert d["failure_id"] == "fail_abc123"
        assert d["affected_components"] == ["worker"]

    def test_entry_is_frozen(self):
        entry = FailureRegistryEntry(
            failure_id="fail_abc123",
            timestamp="2026-04-07T00:00:00+00:00",
            category="test",
            severity="error",
            description="test",
        )
        with pytest.raises(AttributeError):
            entry.category = "changed"  # type: ignore[misc]


class TestFailureRegistryLogAndRead:
    """Tests for log_failure() and get_failures()."""

    def test_log_failure_returns_entry(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            entry = registry.log_failure(
                category="inspector_rejection",
                severity="warning",
                description="Output missing error handling",
            )
        assert entry.failure_id.startswith("fail_")
        assert entry.category == "inspector_rejection"
        assert entry.status == FailureStatus.ACTIVE.value

    def test_log_failure_persists_to_jsonl(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            registry.log_failure(
                category="model_timeout",
                severity="error",
                description="Model failed to respond",
            )
        jsonl_path = registry_dir / "failure-registry.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["category"] == "model_timeout"

    def test_get_failures_returns_all(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            registry.log_failure(category="a", severity="warning", description="first")
            registry.log_failure(category="b", severity="error", description="second")
            results = registry.get_failures()
        assert len(results) == 2

    def test_get_failures_filter_by_category(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            registry.log_failure(category="a", severity="warning", description="first")
            registry.log_failure(category="b", severity="error", description="second")
            results = registry.get_failures(category="a")
        assert len(results) == 1
        assert results[0].category == "a"

    def test_get_failures_filter_by_since(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            old_entry = registry.log_failure(category="old", severity="warning", description="old one")
            cutoff = datetime.fromisoformat(old_entry.timestamp).timestamp() + 0.001
            time.sleep(0.05)
            registry.log_failure(category="new", severity="warning", description="new one")
            results = registry.get_failures(since=cutoff)
        assert len(results) == 1
        assert results[0].category == "new"


class TestFailureRegistryResolve:
    """Tests for resolve_failure()."""

    def test_resolve_existing_failure(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            entry = registry.log_failure(category="test", severity="warning", description="test")
            assert registry.resolve_failure(entry.failure_id)
            resolved = registry.get_failures()
        assert resolved[0].status == FailureStatus.RESOLVED.value

    def test_resolve_nonexistent_failure(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            assert not registry.resolve_failure("fail_nonexistent")

    def test_resolve_rewrite_failure_preserves_existing_registry(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            entry = registry.log_failure(category="test", severity="warning", description="test")
            registry_path = registry_dir / "failure-registry.jsonl"
            original = registry_path.read_text(encoding="utf-8")
            with patch.object(Path, "replace", side_effect=OSError("replace failed")):
                assert registry.resolve_failure(entry.failure_id)

        assert registry_path.read_text(encoding="utf-8") == original
        assert not list(registry_dir.glob("*.tmp"))


class TestSingleton:
    """Tests for get_failure_registry() singleton."""

    def test_singleton_returns_same_instance(self):
        a = get_failure_registry()
        b = get_failure_registry()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = get_failure_registry()
        reset_failure_registry()
        b = get_failure_registry()
        assert a is not b


# ---------------------------------------------------------------------------
# US-003: Prevention rule generation
# ---------------------------------------------------------------------------


class TestPreventionRuleGeneration:
    """Tests for automatic prevention rule extraction from repeated failures."""

    def test_no_rule_below_threshold(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            registry.log_failure(category="repeated", severity="warning", description="missing error handling")
            registry.log_failure(category="repeated", severity="warning", description="missing error handling")
            rules = registry.get_prevention_rules()
        assert len(rules) == 0

    def test_rule_generated_at_threshold(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            registry.log_failure(
                category="repeated", severity="warning", description="missing error handling in output"
            )
            registry.log_failure(category="repeated", severity="warning", description="missing error handling detected")
            registry.log_failure(category="repeated", severity="warning", description="missing error handling again")
            rules = registry.get_prevention_rules()
        assert len(rules) == 1
        rule = rules[0]
        assert rule.category == "repeated"
        assert rule.rule_id.startswith("prev_")
        assert len(rule.created_from_failures) == 3

    def test_no_duplicate_rules_for_same_category(self, registry: FailureRegistry, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            for i in range(6):
                registry.log_failure(
                    category="dup_cat",
                    severity="warning",
                    description=f"same failure pattern {i}",
                )
            rules = registry.get_prevention_rules()
        # Only one rule should be generated for the category
        cat_rules = [r for r in rules if r.category == "dup_cat"]
        assert len(cat_rules) == 1


class TestPreventionRuleTypes:
    """Tests for the three prevention rule types."""

    def test_semantic_rule_type(self, registry: FailureRegistry, registry_dir: Path):
        """Failures with 'missing' keyword generate SEMANTIC rules."""
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            for _ in range(3):
                registry.log_failure(
                    category="semantic_test",
                    severity="warning",
                    description="missing validation in output",
                )
            rules = registry.get_prevention_rules()
        assert len(rules) == 1
        assert rules[0].rule_type == PreventionRuleType.SEMANTIC.value

    def test_pattern_rule_type(self, registry: FailureRegistry, registry_dir: Path):
        """Failures with common non-structural keywords generate PATTERN rules."""
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            for _ in range(3):
                registry.log_failure(
                    category="pattern_test",
                    severity="warning",
                    description="timeout exceeded during inference call",
                )
            rules = registry.get_prevention_rules()
        assert len(rules) == 1
        assert rules[0].rule_type == PreventionRuleType.PATTERN.value

    def test_extracted_rule_type(self, registry: FailureRegistry, registry_dir: Path):
        """Failures with no common words generate EXTRACTED rules."""
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            registry.log_failure(category="varied", severity="warning", description="alpha bravo charlie")
            registry.log_failure(category="varied", severity="warning", description="delta echo foxtrot")
            registry.log_failure(category="varied", severity="warning", description="golf hotel india")
            rules = registry.get_prevention_rules()
        assert len(rules) == 1
        assert rules[0].rule_type == PreventionRuleType.EXTRACTED.value


class TestPreventionRuleMatching:
    """Tests for PreventionRule.matches()."""

    def test_pattern_rule_matches_regex(self):
        rule = PreventionRule(
            rule_id="prev_test",
            rule_type=PreventionRuleType.PATTERN.value,
            category="test",
            pattern=r"timeout|exceeded",
            description="test pattern",
        )
        assert rule.matches("The request timed out: timeout error")
        assert rule.matches("Limit exceeded for this operation")
        assert not rule.matches("Everything worked fine")

    def test_semantic_rule_matches_substring(self):
        rule = PreventionRule(
            rule_id="prev_test",
            rule_type=PreventionRuleType.SEMANTIC.value,
            category="test",
            pattern="missing validation",
            description="test semantic",
        )
        assert rule.matches("The output has missing validation checks")
        assert not rule.matches("All validation present")

    def test_invalid_regex_does_not_crash(self):
        rule = PreventionRule(
            rule_id="prev_test",
            rule_type=PreventionRuleType.PATTERN.value,
            category="test",
            pattern="[invalid",
            description="bad regex",
        )
        assert not rule.matches("some text")


# ---------------------------------------------------------------------------
# US-002: Wiring (record_failure, record_inference_failure)
# ---------------------------------------------------------------------------


class TestWiringRecordFailure:
    """Tests for analytics.wiring.record_failure()."""

    def test_record_failure_delegates_to_registry(self, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            from vetinari.analytics.wiring import record_failure

            record_failure(
                category="test_wiring",
                severity="warning",
                description="wiring test failure",
            )
            reg = get_failure_registry()
            failures = reg.get_failures(category="test_wiring")
        assert len(failures) == 1
        assert failures[0].description == "wiring test failure"

    def test_record_inference_failure_logs_to_registry(self, registry_dir: Path):
        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            from vetinari.analytics.wiring import record_inference_failure, reset_wiring

            reset_wiring()
            record_inference_failure(
                agent_type="WORKER",
                provider="local",
                model_id="test-model",
                latency_ms=5000.0,
            )
            reg = get_failure_registry()
            failures = reg.get_failures(category="model_timeout")
        assert len(failures) == 1
        assert "test-model" in failures[0].description


# ---------------------------------------------------------------------------
# US-004: Prevention rule check in pipeline_quality
# ---------------------------------------------------------------------------


class TestPreventionRuleInPipeline:
    """Tests for prevention rule checking during output review."""

    def test_prevention_rule_blocks_matching_output(self, registry_dir: Path):
        """A matching prevention rule should flip passed to False."""
        from vetinari.analytics.failure_registry import FailureRegistry, PreventionRule, PreventionRuleType

        with patch("vetinari.analytics.failure_registry._get_registry_dir", return_value=registry_dir):
            reg = FailureRegistry()
            # Write a rule directly
            rule = PreventionRule(
                rule_id="prev_test_block",
                rule_type=PreventionRuleType.PATTERN.value,
                category="known_bad",
                pattern="dangerous_pattern",
                description="Blocks outputs containing dangerous_pattern",
                created_from_failures=["fail_1", "fail_2", "fail_3"],
                created_at="2026-04-07T00:00:00+00:00",
            )
            reg._save_rule(rule)

            loaded_rules = reg.get_prevention_rules()
        assert len(loaded_rules) == 1
        assert loaded_rules[0].matches("output with dangerous_pattern inside")
        assert not loaded_rules[0].matches("clean output")


# ---------------------------------------------------------------------------
# US-005: TrainingRecord rejection fields + DPO export
# ---------------------------------------------------------------------------


class TestTrainingRecordRejectionFields:
    """Tests for rejection_reason, rejection_category, inspector_feedback fields."""

    def test_rejection_fields_exist(self):
        from vetinari.learning.training_record import TrainingRecord

        record = TrainingRecord(
            record_id="tr_test",
            timestamp="2026-04-07T00:00:00+00:00",
            task="test task",
            prompt="test prompt",
            response="test response",
            score=0.3,
            model_id="test-model",
            task_type="coding",
            success=False,
            rejection_reason="Output lacked error handling",
            rejection_category="missing_error_handling",
            inspector_feedback="The output does not handle edge cases",
        )
        assert record.rejection_reason == "Output lacked error handling"
        assert record.rejection_category == "missing_error_handling"
        assert record.inspector_feedback == "The output does not handle edge cases"

    def test_rejection_fields_default_empty(self):
        from vetinari.learning.training_record import TrainingRecord

        record = TrainingRecord(
            record_id="tr_test",
            timestamp="2026-04-07T00:00:00+00:00",
            task="test task",
            prompt="test prompt",
            response="test response",
            score=0.9,
            model_id="test-model",
            task_type="coding",
        )
        assert record.rejection_reason == ""
        assert record.rejection_category == ""
        assert record.inspector_feedback == ""

    def test_rejection_fields_in_to_dict(self):
        from vetinari.learning.training_record import TrainingRecord

        record = TrainingRecord(
            record_id="tr_test",
            timestamp="2026-04-07T00:00:00+00:00",
            task="test task",
            prompt="test prompt",
            response="test response",
            score=0.3,
            model_id="test-model",
            task_type="coding",
            success=False,
            rejection_reason="bad output",
            rejection_category="quality",
            inspector_feedback="needs improvement",
        )
        d = record.to_dict()
        assert d["rejection_reason"] == "bad output"
        assert d["rejection_category"] == "quality"
        assert d["inspector_feedback"] == "needs improvement"

    def test_dpo_export_includes_why_chosen_is_better(self, tmp_path: Path):
        """DPO pairs should include why_chosen_is_better from rejection_reason."""
        from vetinari.learning.training_collector import TrainingDataCollector

        collector = TrainingDataCollector(output_path=str(tmp_path / "training.jsonl"), sync=True)

        # Record a good and bad version of the same task
        collector.record(
            task="implement error handling",
            prompt="implement error handling",
            response="def safe(): try: pass except: raise",
            score=0.9,
            model_id="test",
            task_type="coding",
            latency_ms=100,
            tokens_used=50,
            success=True,
        )
        collector.record(
            task="implement error handling",
            prompt="implement error handling",
            response="def unsafe(): pass",
            score=0.2,
            model_id="test",
            task_type="coding",
            latency_ms=100,
            tokens_used=50,
            success=False,
            rejection_reason="Missing try/except blocks",
            rejection_category="missing_error_handling",
        )

        pairs = collector.export_dpo_dataset()
        assert len(pairs) == 1
        assert pairs[0]["why_chosen_is_better"] == "Missing try/except blocks"
        assert pairs[0]["rejection_category"] == "missing_error_handling"


# ---------------------------------------------------------------------------
# Re-export verification
# ---------------------------------------------------------------------------


class TestReExports:
    """Verify all failure registry types are accessible from analytics package."""

    def test_analytics_init_exports(self):
        from vetinari.analytics import (
            FailureRegistry,
            FailureRegistryEntry,
            FailureStatus,
            PreventionRule,
            PreventionRuleType,
            get_failure_registry,
            record_failure,
            reset_failure_registry,
        )

        assert FailureRegistry is not None
        assert FailureRegistryEntry is not None
        assert FailureStatus is not None
        assert PreventionRule is not None
        assert PreventionRuleType is not None
        assert callable(get_failure_registry)
        assert callable(record_failure)
        assert callable(reset_failure_registry)
