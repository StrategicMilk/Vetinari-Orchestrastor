"""Tests for vetinari.context.acon — ACON failure-driven context compression."""

from __future__ import annotations

import pytest

from vetinari.context.acon import (
    ACONCompressor,
    CompressionOutcome,
    InfoCategory,
)


@pytest.fixture
def compressor(tmp_path):
    """Create a fresh compressor with temp storage."""
    rules_path = tmp_path / "acon_rules.json"
    return ACONCompressor(rules_path=rules_path)


class TestACONCompression:
    """Tests for context compression behavior."""

    def test_preserves_critical_categories(self, compressor):
        """Task description and constraints are always preserved by default."""
        context = {
            "task_description": "Build a REST API",
            "constraints": {"max_tokens": 2048},
            "system_state": {"cpu_usage": 0.5},
        }
        compressed = compressor.compress(context)
        assert "task_description" in compressed
        assert "constraints" in compressed

    def test_removes_non_essential_categories(self, compressor):
        """System state (low priority) is removed by default."""
        context = {
            "task_description": "Build a REST API",
            "system_state": {"cpu_usage": 0.5},
        }
        compressed = compressor.compress(context)
        assert "task_description" in compressed
        assert "system_state" not in compressed

    def test_tracks_removed_categories(self, compressor):
        """Compressed context records what was removed."""
        context = {
            "task_description": "test",
            "system_state": {"data": "ephemeral"},
        }
        compressed = compressor.compress(context)
        assert "_acon_removed" in compressed
        assert "system_state" in compressed["_acon_removed"]

    def test_unknown_keys_preserved_by_default(self, compressor):
        """Keys that are not known InfoCategory values are kept unchanged."""
        context = {
            "task_description": "test",
            "some_custom_key": "custom value",
        }
        compressed = compressor.compress(context)
        assert "some_custom_key" in compressed
        assert compressed["some_custom_key"] == "custom value"

    def test_no_removal_metadata_when_nothing_removed(self, compressor):
        """When all categories are preserved, no _acon_removed key is added."""
        context = {
            "task_description": "only critical stuff",
            "constraints": {"limit": 10},
        }
        compressed = compressor.compress(context)
        assert "_acon_removed" not in compressed

    def test_full_and_compressed_hashes_recorded(self, compressor):
        """Hash sentinels are present when at least one category is removed."""
        context = {
            "task_description": "test",
            "system_state": {"ephemeral": True},
        }
        compressed = compressor.compress(context)
        assert "_acon_full_hash" in compressed
        assert "_acon_compressed_hash" in compressed
        # Hashes must differ because content changed
        assert compressed["_acon_full_hash"] != compressed["_acon_compressed_hash"]

    def test_five_default_categories_preserved(self, compressor):
        """Default rules preserve exactly 5 of the 10 standard categories."""
        stats = compressor.get_compression_stats()
        assert stats["preserved_categories"] == 5
        assert stats["compressible_categories"] == 5
        assert stats["total_categories"] == 10


class TestACONFailureLearning:
    """Tests for the failure-driven learning mechanism."""

    def test_failure_promotes_category_to_preserved(self, compressor):
        """After a compression-caused failure, removed categories become preserved."""
        # Initially system_state is not preserved
        context = {"task_description": "test", "system_state": {"data": "needed"}}
        compressed = compressor.compress(context)
        assert "system_state" not in compressed

        # Record a failure caused by removing system_state
        outcome = CompressionOutcome(
            task_id="task-001",
            compressed_success=False,
            full_context_hash="abc",
            compressed_context_hash="def",
            removed_categories=[InfoCategory.SYSTEM_STATE],
            failure_reason="Missing system state caused incorrect routing",
        )
        compressor.record_outcome(outcome)

        # Now system_state should be preserved
        compressed2 = compressor.compress(context)
        assert "system_state" in compressed2

    def test_successful_outcome_does_not_change_rules(self, compressor):
        """Successful compressions don't modify preservation rules."""
        stats_before = compressor.get_compression_stats()
        outcome = CompressionOutcome(
            task_id="task-002",
            compressed_success=True,
            full_context_hash="abc",
            compressed_context_hash="def",
            removed_categories=[InfoCategory.MODEL_METADATA],
        )
        compressor.record_outcome(outcome)
        stats_after = compressor.get_compression_stats()
        assert stats_before["preserved_categories"] == stats_after["preserved_categories"]

    def test_rules_persist_to_disk(self, compressor, tmp_path):
        """Rules survive save/load cycle."""
        # Trigger a rule update
        outcome = CompressionOutcome(
            task_id="task-003",
            compressed_success=False,
            full_context_hash="abc",
            compressed_context_hash="def",
            removed_categories=[InfoCategory.SYSTEM_STATE],
            failure_reason="needed system state",
        )
        compressor.record_outcome(outcome)

        # Create a new compressor from the same path
        rules_path = tmp_path / "acon_rules.json"
        compressor2 = ACONCompressor(rules_path=rules_path)
        rule = compressor2._rules.get(InfoCategory.SYSTEM_STATE)
        assert rule is not None
        assert rule.preserve is True

    def test_priority_increases_on_failure(self, compressor):
        """Each failure raises the category's priority by 15, capped at 100."""
        # MODEL_METADATA starts at priority 30
        initial_rule = compressor._rules[InfoCategory.MODEL_METADATA]
        assert initial_rule.priority == 30
        assert initial_rule.preserve is False

        outcome = CompressionOutcome(
            task_id="task-004",
            compressed_success=False,
            full_context_hash="a",
            compressed_context_hash="b",
            removed_categories=[InfoCategory.MODEL_METADATA],
            failure_reason="needed model metadata",
        )
        compressor.record_outcome(outcome)

        updated = compressor._rules[InfoCategory.MODEL_METADATA]
        assert updated.priority == 45  # 30 + 15
        assert updated.preserve is True

    def test_priority_capped_at_100(self, compressor):
        """Priority never exceeds 100, even after repeated failures."""
        # Artificially set a near-ceiling priority
        from vetinari.context.acon import CompressionRule

        compressor._rules[InfoCategory.SYSTEM_STATE] = CompressionRule(
            category=InfoCategory.SYSTEM_STATE,
            preserve=False,
            priority=95,
            reason="test setup",
        )

        outcome = CompressionOutcome(
            task_id="task-005",
            compressed_success=False,
            full_context_hash="a",
            compressed_context_hash="b",
            removed_categories=[InfoCategory.SYSTEM_STATE],
            failure_reason="capping test",
        )
        compressor.record_outcome(outcome)

        updated = compressor._rules[InfoCategory.SYSTEM_STATE]
        assert updated.priority == 100

    def test_failure_without_removed_categories_does_not_update_rules(self, compressor):
        """A failure with no removed categories leaves rules unchanged."""
        stats_before = compressor.get_compression_stats()
        outcome = CompressionOutcome(
            task_id="task-006",
            compressed_success=False,
            full_context_hash="a",
            compressed_context_hash="b",
            removed_categories=[],
            failure_reason="unrelated error",
        )
        compressor.record_outcome(outcome)
        stats_after = compressor.get_compression_stats()
        assert stats_before["preserved_categories"] == stats_after["preserved_categories"]

    def test_multiple_failures_promote_multiple_categories(self, compressor):
        """A single failure outcome with multiple removed categories promotes all of them."""
        outcome = CompressionOutcome(
            task_id="task-007",
            compressed_success=False,
            full_context_hash="a",
            compressed_context_hash="b",
            removed_categories=[InfoCategory.SYSTEM_STATE, InfoCategory.MODEL_METADATA],
            failure_reason="needed both",
        )
        compressor.record_outcome(outcome)

        assert compressor._rules[InfoCategory.SYSTEM_STATE].preserve is True
        assert compressor._rules[InfoCategory.MODEL_METADATA].preserve is True


class TestACONStats:
    """Tests for compression statistics."""

    def test_stats_reflect_history(self, compressor):
        """Stats accurately count successes and failures in history."""
        for i in range(3):
            compressor.record_outcome(
                CompressionOutcome(
                    task_id=f"s-{i}",
                    compressed_success=True,
                    full_context_hash="a",
                    compressed_context_hash="b",
                )
            )
        compressor.record_outcome(
            CompressionOutcome(
                task_id="f-1",
                compressed_success=False,
                full_context_hash="a",
                compressed_context_hash="b",
                removed_categories=[InfoCategory.MODEL_METADATA],
                failure_reason="test",
            )
        )
        stats = compressor.get_compression_stats()
        assert stats["history_size"] == 4
        assert stats["failure_count"] == 1
        assert stats["success_rate"] == 0.75

    def test_initial_success_rate_is_1_0_with_empty_history(self, compressor):
        """Success rate defaults to 1.0 when no outcomes have been recorded."""
        stats = compressor.get_compression_stats()
        assert stats["history_size"] == 0
        assert stats["success_rate"] == 1.0

    def test_history_capped_at_max_history(self, tmp_path):
        """History never grows past max_history entries."""
        small_compressor = ACONCompressor(
            rules_path=tmp_path / "rules.json",
            max_history=3,
        )
        for i in range(5):
            small_compressor.record_outcome(
                CompressionOutcome(
                    task_id=f"t-{i}",
                    compressed_success=True,
                    full_context_hash="a",
                    compressed_context_hash="b",
                )
            )
        stats = small_compressor.get_compression_stats()
        assert stats["history_size"] == 3


class TestACONCompressorPartialRuleLoad:
    """Partial rule files are overlaid onto defaults, not used as the full set."""

    def test_partial_rule_file_merges_onto_defaults(self, tmp_path) -> None:
        """Loading a one-rule file leaves the other 9 default categories intact.

        A rules file that contains only one entry must NOT collapse the known
        category set to 1.  The single stored rule should override its
        matching default, and the remaining 9 categories must survive from
        defaults so the compressor still knows about them.
        """
        rules_path = tmp_path / "partial_rules.json"
        # Write a file that only persists one rule — MODEL_METADATA set to preserve=True
        partial = {
            "rules": [
                {
                    "category": InfoCategory.MODEL_METADATA.value,
                    "preserve": True,
                    "priority": 99,
                    "reason": "override for test",
                }
            ]
        }
        import json

        rules_path.write_text(json.dumps(partial), encoding="utf-8")

        compressor = ACONCompressor(rules_path=rules_path)
        stats = compressor.get_compression_stats()

        # All 10 categories must still be present — the partial file is an overlay
        assert stats["total_categories"] == 10

        # The overridden rule must reflect the loaded value, not the default
        with compressor._lock:
            rule = compressor._rules[InfoCategory.MODEL_METADATA]
        assert rule.preserve is True
        assert rule.priority == 99


class TestACONSingleton:
    """Tests for the module-level singleton."""

    def test_get_acon_compressor_returns_same_instance(self, tmp_path):
        """Calling get_acon_compressor twice returns the same object."""
        # Reset singleton so each test gets a fresh one
        import vetinari.context.acon as acon_mod
        from vetinari.context.acon import get_acon_compressor

        original = acon_mod._instance
        acon_mod._instance = None
        try:
            rules_path = tmp_path / "singleton_rules.json"
            inst1 = get_acon_compressor(rules_path=rules_path)
            inst2 = get_acon_compressor(rules_path=rules_path)
            assert inst1 is inst2
        finally:
            acon_mod._instance = original

    def test_default_rules_load_when_no_file_exists(self, tmp_path):
        """A compressor with a non-existent rules file initialises default rules."""
        rules_path = tmp_path / "nonexistent.json"
        c = ACONCompressor(rules_path=rules_path)
        stats = c.get_compression_stats()
        assert stats["total_categories"] == 10

    def test_corrupt_rules_file_falls_back_to_defaults(self, tmp_path):
        """A corrupt rules file is handled gracefully, falling back to defaults."""
        rules_path = tmp_path / "corrupt.json"
        rules_path.write_text("{{not valid json", encoding="utf-8")
        c = ACONCompressor(rules_path=rules_path)
        stats = c.get_compression_stats()
        assert stats["total_categories"] == 10
