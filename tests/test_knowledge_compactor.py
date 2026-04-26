"""Tests for vetinari.kaizen.knowledge_compactor — KnowledgeCompactor and CompactionReport."""

from __future__ import annotations

import pytest

from vetinari.kaizen.knowledge_compactor import (
    CompactionReport,
    KnowledgeCompactor,
    _cluster_entries,
)
from vetinari.kaizen.knowledge_lint import LintFinding
from vetinari.memory.interfaces import MemoryEntry
from vetinari.types import MemoryType

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_entry(
    content: str = "some content",
    entry_type: MemoryType = MemoryType.DISCOVERY,
    recall_count: int = 0,
    agent: str = "test-agent",
) -> MemoryEntry:
    """Create a MemoryEntry with only the fields under test specified."""
    return MemoryEntry(
        agent=agent,
        entry_type=entry_type,
        content=content,
        recall_count=recall_count,
    )


# ── CompactionReport ──────────────────────────────────────────────────────────


class TestCompactionReport:
    def test_defaults_produce_empty_lists(self) -> None:
        report = CompactionReport()
        assert report.patterns_created == []
        assert report.principles_created == []
        assert report.rules_created == []
        assert report.contradictions == []
        assert report.input_entries == 0

    def test_timestamp_is_iso8601(self) -> None:
        report = CompactionReport()
        # Should parse without raising
        from datetime import datetime

        parsed = datetime.fromisoformat(report.timestamp)
        assert parsed.isoformat() == report.timestamp

    def test_repr_shows_counts(self) -> None:
        report = CompactionReport(
            patterns_created=[make_entry()],
            principles_created=[make_entry(), make_entry()],
            rules_created=[],
            contradictions=[],
        )
        r = repr(report)
        assert "patterns=1" in r
        assert "principles=2" in r
        assert "rules=0" in r
        assert "contradictions=0" in r


# ── _cluster_entries helper ───────────────────────────────────────────────────


class TestClusterEntries:
    def test_empty_input(self) -> None:
        assert _cluster_entries([], 0.6) == []

    def test_single_entry_forms_own_cluster(self) -> None:
        entry = make_entry("hello world")
        clusters = _cluster_entries([entry], 0.6)
        assert len(clusters) == 1
        assert clusters[0] == [entry]

    def test_identical_entries_cluster_together(self) -> None:
        entries = [make_entry("same content") for _ in range(4)]
        clusters = _cluster_entries(entries, 0.6)
        # All four should land in a single cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 4

    def test_dissimilar_entries_form_separate_clusters(self) -> None:
        entries = [
            make_entry("apple banana cherry"),
            make_entry("database transaction rollback"),
            make_entry("neural network backpropagation"),
        ]
        clusters = _cluster_entries(entries, 0.6)
        # All dissimilar — each is its own cluster
        assert len(clusters) == 3

    def test_threshold_controls_grouping(self) -> None:
        # Slightly different but similar strings
        a = make_entry("the quick brown fox jumps")
        b = make_entry("the quick brown fox leaps")
        # At low threshold (0.3) they cluster; at high threshold (0.99) they don't
        low = _cluster_entries([a, b], 0.3)
        high = _cluster_entries([a, b], 0.99)
        assert sum(len(c) for c in low) == 2
        assert sum(len(c) for c in high) == 2
        # Low threshold: one cluster of size 2
        assert any(len(c) == 2 for c in low)
        # High threshold: two clusters of size 1
        assert all(len(c) == 1 for c in high)

    def test_each_entry_assigned_exactly_once(self) -> None:
        entries = [make_entry(f"entry number {i}") for i in range(10)]
        clusters = _cluster_entries(entries, 0.5)
        all_entries = [e for c in clusters for e in c]
        assert len(all_entries) == 10
        assert {id(e) for e in all_entries} == {id(e) for e in entries}


# ── extract_patterns ──────────────────────────────────────────────────────────


class TestExtractPatterns:
    def setup_method(self) -> None:
        self.compactor = KnowledgeCompactor()

    def test_empty_input_returns_empty(self) -> None:
        assert self.compactor.extract_patterns([]) == []

    def test_cluster_below_minimum_produces_no_pattern(self) -> None:
        # Only 2 episodes — below the 3-episode minimum
        entries = [make_entry("same content") for _ in range(2)]
        patterns = self.compactor.extract_patterns(entries)
        assert patterns == []

    def test_cluster_at_minimum_produces_one_pattern(self) -> None:
        entries = [make_entry("same content") for _ in range(3)]
        patterns = self.compactor.extract_patterns(entries)
        assert len(patterns) == 1

    def test_cluster_above_minimum_produces_one_pattern(self) -> None:
        entries = [make_entry("same content") for _ in range(7)]
        patterns = self.compactor.extract_patterns(entries)
        assert len(patterns) == 1

    def test_pattern_has_correct_type(self) -> None:
        entries = [make_entry("same content") for _ in range(3)]
        patterns = self.compactor.extract_patterns(entries)
        assert patterns[0].entry_type == MemoryType.PATTERN

    def test_pattern_supersedes_first_episode(self) -> None:
        entries = [make_entry("same content") for _ in range(3)]
        patterns = self.compactor.extract_patterns(entries)
        assert patterns[0].supersedes_id == entries[0].id

    def test_pattern_content_includes_count(self) -> None:
        entries = [make_entry("important observation") for _ in range(3)]
        patterns = self.compactor.extract_patterns(entries)
        assert "3 episodes" in patterns[0].content

    def test_pattern_id_uses_mem_prefix(self) -> None:
        entries = [make_entry("same content") for _ in range(3)]
        patterns = self.compactor.extract_patterns(entries)
        assert patterns[0].id.startswith("mem_")

    def test_pattern_inherits_agent_from_representative(self) -> None:
        entries = [make_entry("same content", agent="foreman") for _ in range(3)]
        patterns = self.compactor.extract_patterns(entries)
        assert patterns[0].agent == "foreman"

    def test_two_distinct_clusters_produce_two_patterns(self) -> None:
        cluster_a = [make_entry("apple banana cherry") for _ in range(3)]
        cluster_b = [make_entry("neural network weights") for _ in range(3)]
        patterns = self.compactor.extract_patterns(cluster_a + cluster_b)
        assert len(patterns) == 2

    def test_provenance_is_knowledge_compactor(self) -> None:
        entries = [make_entry("same content") for _ in range(3)]
        patterns = self.compactor.extract_patterns(entries)
        assert patterns[0].provenance == "knowledge_compactor"


# ── synthesize_principles ─────────────────────────────────────────────────────


class TestSynthesizePrinciples:
    def setup_method(self) -> None:
        self.compactor = KnowledgeCompactor()

    def test_empty_input_returns_empty(self) -> None:
        assert self.compactor.synthesize_principles([]) == []

    def test_single_pattern_produces_no_principle(self) -> None:
        patterns = [make_entry("a pattern", entry_type=MemoryType.PATTERN)]
        assert self.compactor.synthesize_principles(patterns) == []

    def test_two_similar_patterns_produce_one_principle(self) -> None:
        patterns = [make_entry("similar pattern content", entry_type=MemoryType.PATTERN) for _ in range(2)]
        principles = self.compactor.synthesize_principles(patterns)
        assert len(principles) == 1

    def test_principle_has_correct_type(self) -> None:
        patterns = [make_entry("similar content", entry_type=MemoryType.PATTERN) for _ in range(2)]
        principles = self.compactor.synthesize_principles(patterns)
        assert principles[0].entry_type == MemoryType.PRINCIPLE

    def test_principle_content_prefixed(self) -> None:
        patterns = [make_entry("some insight", entry_type=MemoryType.PATTERN) for _ in range(2)]
        principles = self.compactor.synthesize_principles(patterns)
        assert principles[0].content.startswith("Principle: ")

    def test_principle_supersedes_first_pattern(self) -> None:
        patterns = [make_entry("similar content", entry_type=MemoryType.PATTERN) for _ in range(2)]
        principles = self.compactor.synthesize_principles(patterns)
        assert principles[0].supersedes_id == patterns[0].id

    def test_principle_id_uses_mem_prefix(self) -> None:
        patterns = [make_entry("similar content", entry_type=MemoryType.PATTERN) for _ in range(2)]
        principles = self.compactor.synthesize_principles(patterns)
        assert principles[0].id.startswith("mem_")

    def test_provenance_is_knowledge_compactor(self) -> None:
        patterns = [make_entry("similar content", entry_type=MemoryType.PATTERN) for _ in range(2)]
        principles = self.compactor.synthesize_principles(patterns)
        assert principles[0].provenance == "knowledge_compactor"


# ── codify_rules ──────────────────────────────────────────────────────────────


class TestCodifyRules:
    def setup_method(self) -> None:
        self.compactor = KnowledgeCompactor()

    def test_empty_input_returns_empty(self) -> None:
        assert self.compactor.codify_rules([]) == []

    def test_principle_below_threshold_produces_no_rule(self) -> None:
        principle = make_entry(
            "an insight",
            entry_type=MemoryType.PRINCIPLE,
            recall_count=2,
        )
        assert self.compactor.codify_rules([principle]) == []

    def test_principle_at_threshold_produces_rule(self) -> None:
        principle = make_entry(
            "an insight",
            entry_type=MemoryType.PRINCIPLE,
            recall_count=3,
        )
        rules = self.compactor.codify_rules([principle])
        assert len(rules) == 1

    def test_principle_above_threshold_produces_rule(self) -> None:
        principle = make_entry(
            "an insight",
            entry_type=MemoryType.PRINCIPLE,
            recall_count=10,
        )
        rules = self.compactor.codify_rules([principle])
        assert len(rules) == 1

    def test_rule_has_correct_type(self) -> None:
        principle = make_entry(
            "an insight",
            entry_type=MemoryType.PRINCIPLE,
            recall_count=3,
        )
        rules = self.compactor.codify_rules([principle])
        assert rules[0].entry_type == MemoryType.RULE

    def test_rule_content_prefixed(self) -> None:
        principle = make_entry(
            "always validate inputs",
            entry_type=MemoryType.PRINCIPLE,
            recall_count=3,
        )
        rules = self.compactor.codify_rules([principle])
        assert rules[0].content.startswith("Rule: ")
        assert "always validate inputs" in rules[0].content

    def test_rule_supersedes_principle(self) -> None:
        principle = make_entry(
            "an insight",
            entry_type=MemoryType.PRINCIPLE,
            recall_count=3,
        )
        rules = self.compactor.codify_rules([principle])
        assert rules[0].supersedes_id == principle.id

    def test_rule_id_uses_mem_prefix(self) -> None:
        principle = make_entry(
            "an insight",
            entry_type=MemoryType.PRINCIPLE,
            recall_count=3,
        )
        rules = self.compactor.codify_rules([principle])
        assert rules[0].id.startswith("mem_")

    def test_mix_of_confirmed_and_unconfirmed_principles(self) -> None:
        confirmed = make_entry("confirmed insight", entry_type=MemoryType.PRINCIPLE, recall_count=5)
        unconfirmed = make_entry("new insight", entry_type=MemoryType.PRINCIPLE, recall_count=1)
        rules = self.compactor.codify_rules([confirmed, unconfirmed])
        assert len(rules) == 1
        assert rules[0].supersedes_id == confirmed.id

    def test_provenance_is_knowledge_compactor(self) -> None:
        principle = make_entry("insight", entry_type=MemoryType.PRINCIPLE, recall_count=3)
        rules = self.compactor.codify_rules([principle])
        assert rules[0].provenance == "knowledge_compactor"


# ── check_cross_level_contradictions ─────────────────────────────────────────


class TestCheckCrossLevelContradictions:
    def setup_method(self) -> None:
        self.compactor = KnowledgeCompactor()

    def test_empty_input_returns_empty(self) -> None:
        assert self.compactor.check_cross_level_contradictions([]) == []

    def test_no_higher_level_entries_returns_empty(self) -> None:
        entries = [make_entry("content") for _ in range(3)]
        assert self.compactor.check_cross_level_contradictions(entries) == []

    def test_no_lower_level_entries_returns_empty(self) -> None:
        entries = [
            make_entry("content", entry_type=MemoryType.PATTERN),
            make_entry("content", entry_type=MemoryType.PRINCIPLE),
        ]
        assert self.compactor.check_cross_level_contradictions(entries) == []

    def test_identical_content_not_flagged(self) -> None:
        # Identical content should not be a contradiction (content_a == content_b check)
        high = make_entry("exact same content", entry_type=MemoryType.RULE)
        low = make_entry("exact same content", entry_type=MemoryType.DISCOVERY)
        findings = self.compactor.check_cross_level_contradictions([high, low])
        assert findings == []

    def test_near_match_flagged_as_contradiction(self) -> None:
        # One word changed — should be > 0.85 similar but not identical
        high = make_entry(
            "always validate all user inputs before processing",
            entry_type=MemoryType.RULE,
        )
        low = make_entry(
            "always validate all user inputs before handling",
            entry_type=MemoryType.DISCOVERY,
        )
        findings = self.compactor.check_cross_level_contradictions([high, low])
        assert len(findings) == 1

    def test_finding_is_lint_finding_instance(self) -> None:
        high = make_entry(
            "always validate all user inputs before processing",
            entry_type=MemoryType.RULE,
        )
        low = make_entry(
            "always validate all user inputs before handling",
            entry_type=MemoryType.DISCOVERY,
        )
        findings = self.compactor.check_cross_level_contradictions([high, low])
        assert isinstance(findings[0], LintFinding)

    def test_finding_category_is_contradiction(self) -> None:
        high = make_entry(
            "always validate all user inputs before processing",
            entry_type=MemoryType.RULE,
        )
        low = make_entry(
            "always validate all user inputs before handling",
            entry_type=MemoryType.DISCOVERY,
        )
        findings = self.compactor.check_cross_level_contradictions([high, low])
        assert findings[0].category == "contradiction"

    def test_finding_entry_ids_contain_both_entries(self) -> None:
        high = make_entry(
            "always validate all user inputs before processing",
            entry_type=MemoryType.RULE,
        )
        low = make_entry(
            "always validate all user inputs before handling",
            entry_type=MemoryType.DISCOVERY,
        )
        findings = self.compactor.check_cross_level_contradictions([high, low])
        assert high.id in findings[0].entry_ids
        assert low.id in findings[0].entry_ids

    def test_dissimilar_entries_not_flagged(self) -> None:
        high = make_entry("apple banana cherry mango", entry_type=MemoryType.PATTERN)
        low = make_entry("database transaction rollback", entry_type=MemoryType.DISCOVERY)
        findings = self.compactor.check_cross_level_contradictions([high, low])
        assert findings == []

    def test_all_compaction_levels_checked_as_higher(self) -> None:
        # Each of PATTERN, PRINCIPLE, RULE should trigger contradiction checks
        low = make_entry(
            "always validate all user inputs before handling",
            entry_type=MemoryType.DISCOVERY,
        )
        for higher_type in (MemoryType.PATTERN, MemoryType.PRINCIPLE, MemoryType.RULE):
            high = make_entry(
                "always validate all user inputs before processing",
                entry_type=higher_type,
            )
            findings = self.compactor.check_cross_level_contradictions([high, low])
            assert len(findings) == 1, f"Expected contradiction for higher type {higher_type}"


# ── compact (full pipeline) ───────────────────────────────────────────────────


class TestCompact:
    def setup_method(self) -> None:
        self.compactor = KnowledgeCompactor()

    def test_empty_input_returns_empty_report(self) -> None:
        report = self.compactor.compact([])
        assert isinstance(report, CompactionReport)
        assert report.input_entries == 0
        assert report.patterns_created == []
        assert report.principles_created == []
        assert report.rules_created == []
        assert report.contradictions == []

    def test_input_entries_count_set_correctly(self) -> None:
        entries = [make_entry(f"entry {i}") for i in range(5)]
        report = self.compactor.compact(entries)
        assert report.input_entries == 5

    def test_returns_compaction_report(self) -> None:
        report = self.compactor.compact([make_entry("x")])
        assert isinstance(report, CompactionReport)

    def test_existing_pattern_entries_not_re_processed_as_episodes(self) -> None:
        # PATTERN entries should not feed extract_patterns
        existing_pattern = make_entry("some pattern", entry_type=MemoryType.PATTERN)
        report = self.compactor.compact([existing_pattern])
        # No episodes to cluster, so no new patterns created
        assert report.patterns_created == []

    def test_full_ladder_end_to_end(self) -> None:
        # 3 similar episodes → 1 pattern
        # 2 similar patterns → 1 principle (need to inject an existing one to hit threshold)
        # principle with recall_count >= 3 → 1 rule

        # Build 3 similar episodes
        base = "use dependency injection for all service objects"
        episodes = [make_entry(base) for _ in range(3)]

        report = self.compactor.compact(episodes)

        # At minimum, patterns should be created
        assert len(report.patterns_created) >= 1
        assert report.patterns_created[0].entry_type == MemoryType.PATTERN

    def test_compact_with_mix_of_episode_and_higher_entries(self) -> None:
        episodes = [make_entry("use typed dicts over bare dicts") for _ in range(3)]
        existing_rule = make_entry("use typed dicts over plain dicts", entry_type=MemoryType.RULE)
        entries = episodes + [existing_rule]
        report = self.compactor.compact(entries)
        # Input count includes all entries
        assert report.input_entries == 4
        # Patterns from the 3 episodes
        assert len(report.patterns_created) >= 1

    def test_report_repr_is_string(self) -> None:
        report = self.compactor.compact([])
        assert isinstance(repr(report), str)

    def test_timestamp_is_iso8601(self) -> None:
        from datetime import datetime

        report = self.compactor.compact([])
        parsed = datetime.fromisoformat(report.timestamp)
        assert parsed.isoformat() == report.timestamp
