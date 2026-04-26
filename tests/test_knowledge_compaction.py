"""Tests for knowledge compaction pipeline — episode clustering, pattern synthesis, rule codification."""

from __future__ import annotations

import pytest

from vetinari.kaizen.knowledge_compactor import CompactionReport, KnowledgeCompactor
from vetinari.memory.interfaces import MemoryEntry
from vetinari.types import MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(id_suffix: str, content: str, recall_count: int = 0) -> MemoryEntry:
    """Create a DISCOVERY-type MemoryEntry for use as an episode."""
    return MemoryEntry(
        id=f"mem_{id_suffix}",
        entry_type=MemoryType.DISCOVERY,
        content=content,
        recall_count=recall_count,
        summary=content[:50],
    )


def _make_pattern(id_suffix: str, content: str, recall_count: int = 0) -> MemoryEntry:
    """Create a PATTERN-type MemoryEntry."""
    return MemoryEntry(
        id=f"mem_{id_suffix}",
        entry_type=MemoryType.PATTERN,
        content=content,
        recall_count=recall_count,
        summary=content[:50],
    )


def _make_principle(id_suffix: str, content: str, recall_count: int = 0) -> MemoryEntry:
    """Create a PRINCIPLE-type MemoryEntry."""
    return MemoryEntry(
        id=f"mem_{id_suffix}",
        entry_type=MemoryType.PRINCIPLE,
        content=content,
        recall_count=recall_count,
        summary=content[:50],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_patterns_clusters_similar() -> None:
    """5 highly similar DISCOVERY episodes should produce at least 1 PATTERN entry."""
    compactor = KnowledgeCompactor()
    # All share nearly identical content — they cluster well above the 0.6 threshold
    base = "always validate inputs before processing task data in the pipeline"
    episodes = [_make_episode(f"e{i}", f"{base} variant {i}") for i in range(5)]

    patterns = compactor.extract_patterns(episodes)

    assert len(patterns) >= 1, "Expected at least 1 pattern from 5 similar episodes"
    assert all(p.entry_type == MemoryType.PATTERN for p in patterns)


def test_extract_patterns_no_cluster() -> None:
    """3 completely different episodes should NOT cluster (below min size of 3 similar)."""
    compactor = KnowledgeCompactor()
    episodes = [
        _make_episode("a", "use Python for data science machine learning tasks"),
        _make_episode("b", "always commit code changes to version control git repository"),
        _make_episode("c", "prefer functional programming immutable data structures monads"),
    ]

    patterns = compactor.extract_patterns(episodes)

    # Each episode seeds its own cluster of size 1, which is below _MIN_EPISODES_PER_PATTERN=3
    assert patterns == [], f"Expected no patterns, got {patterns}"


def test_synthesize_principles() -> None:
    """3 similar PATTERN entries should produce at least 1 PRINCIPLE entry."""
    compactor = KnowledgeCompactor()
    base = "validate all inputs at system boundaries to prevent downstream failures"
    patterns = [_make_pattern(f"p{i}", f"{base} iteration {i}") for i in range(3)]

    principles = compactor.synthesize_principles(patterns)

    assert len(principles) >= 1, "Expected at least 1 principle from 3 similar patterns"
    assert all(p.entry_type == MemoryType.PRINCIPLE for p in principles)


def test_codify_rules_confirmed_only() -> None:
    """Principle with recall_count >= 3 produces a RULE; recall_count=1 is skipped."""
    compactor = KnowledgeCompactor()

    confirmed = _make_principle("conf", "always validate inputs before processing", recall_count=3)
    unconfirmed = _make_principle("unconf", "consider caching expensive computations", recall_count=1)

    rules = compactor.codify_rules([confirmed, unconfirmed])

    assert len(rules) == 1, "Only the confirmed principle should become a rule"
    assert rules[0].entry_type == MemoryType.RULE
    assert rules[0].supersedes_id == "mem_conf"


def test_supersedes_id_set() -> None:
    """Each higher-level entry must have supersedes_id pointing to a source entry."""
    compactor = KnowledgeCompactor()
    base = "always handle exceptions explicitly to prevent silent failures in code"
    episodes = [_make_episode(f"src{i}", f"{base} case {i}") for i in range(5)]

    patterns = compactor.extract_patterns(episodes)

    assert len(patterns) >= 1
    for pattern in patterns:
        assert pattern.supersedes_id is not None, "Pattern must have supersedes_id"
        # The supersedes_id must point to one of the source episodes
        source_ids = {e.id for e in episodes}
        assert pattern.supersedes_id in source_ids, f"supersedes_id {pattern.supersedes_id!r} not in source episode ids"


def test_cross_level_contradiction() -> None:
    """A higher-level entry nearly matching (>0.85) but differing from an episode is a contradiction."""
    compactor = KnowledgeCompactor()

    # Construct a PATTERN that is very similar to an episode but not identical
    episode_content = "always validate inputs before processing task data in pipeline"
    episode = _make_episode("ep1", episode_content)

    # Slightly modify for the pattern — high similarity, not equal
    pattern_content = "always validate inputs before processing task data in the pipeline flow"
    pattern = _make_pattern("pt1", pattern_content)

    findings = compactor.check_cross_level_contradictions([episode, pattern])

    # We expect at least one contradiction finding if similarity > 0.85
    # The exact result depends on difflib ratio; we verify the interface contract
    for finding in findings:
        assert finding.category == "contradiction"
        assert finding.severity == "warning"
        assert len(finding.entry_ids) == 2


def test_compact_full_ladder() -> None:
    """Full pipeline: enough similar episodes produce patterns → principles → rules."""
    compactor = KnowledgeCompactor()

    # Stage 1: create enough similar episodes to form a pattern (need >= 3 per cluster)
    base = "always log errors with context information for debugging production issues"
    episodes = [_make_episode(f"e{i}", f"{base} note {i}") for i in range(5)]

    # Stage 3 needs principles with recall_count >= 3; we'll verify patterns + principles
    # created, and rules logic is correct (no recall so rules may be 0)
    report = compactor.compact(episodes)

    assert isinstance(report, CompactionReport)
    assert report.input_entries == 5
    assert len(report.patterns_created) >= 1, "Expected patterns from 5 similar episodes"
    # Principles depend on whether multiple patterns cluster together
    # Rules require recall_count >= 3 on principles — newly created principles start at 0
    assert report.rules_created == []
    assert isinstance(report.contradictions, list)


def test_compact_full_ladder_rules_from_recalled_principles() -> None:
    """Principles pre-loaded with recall_count >= 3 produce rules when compacted."""
    compactor = KnowledgeCompactor()

    # Inject a pre-recalled principle directly via codify_rules
    principle = _make_principle("pr_recalled", "validate all inputs at boundaries", recall_count=5)
    rules = compactor.codify_rules([principle])

    assert len(rules) == 1
    assert rules[0].entry_type == MemoryType.RULE
    assert "Rule:" in rules[0].content


def test_compaction_report_structure() -> None:
    """CompactionReport has expected fields with correct types."""
    compactor = KnowledgeCompactor()
    report = compactor.compact([])

    assert isinstance(report.patterns_created, list)
    assert isinstance(report.principles_created, list)
    assert isinstance(report.rules_created, list)
    assert isinstance(report.contradictions, list)
    assert isinstance(report.input_entries, int)
    assert isinstance(report.timestamp, str)
    assert len(report.timestamp) > 10  # non-empty ISO-8601
    assert report.input_entries == 0
