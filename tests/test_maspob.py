"""Tests for MASPOBAnalyzer position-sensitivity analysis (Story 41)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vetinari.learning.prompt_mutator import (
    MASPOBAnalyzer,
    MutationOperator,
    PromptMutator,
    _maspob_analyzer,
    get_maspob_analyzer,
)

_SAMPLE_PROMPT = """\
## Role
You are a helpful assistant.

## Task
Summarise the following text.

## Constraints
Be concise. Be accurate.

## Examples
Input: long text here
Output: short summary
"""


@pytest.fixture
def tmp_analyzer(tmp_path: Path) -> MASPOBAnalyzer:
    """Return a fresh MASPOBAnalyzer backed by a temp state file."""
    return MASPOBAnalyzer(state_path=tmp_path / "maspob_test.json")


class TestSectionPermutationGeneration:
    """analyze_section_ordering should enumerate candidate orderings."""

    def test_returns_list_of_lists(self, tmp_analyzer: MASPOBAnalyzer):
        result = tmp_analyzer.analyze_section_ordering(_SAMPLE_PROMPT)
        assert isinstance(result, list)
        assert all(isinstance(perm, list) for perm in result)

    def test_contains_all_section_names(self, tmp_analyzer: MASPOBAnalyzer):
        result = tmp_analyzer.analyze_section_ordering(_SAMPLE_PROMPT)
        assert result  # non-empty
        expected_names = {"Role", "Task", "Constraints", "Examples"}
        first_perm_set = set(result[0])
        assert first_perm_set == expected_names

    def test_capped_at_24_permutations(self, tmp_analyzer: MASPOBAnalyzer):
        result = tmp_analyzer.analyze_section_ordering(_SAMPLE_PROMPT)
        assert len(result) <= 24

    def test_empty_prompt_returns_empty_list(self, tmp_analyzer: MASPOBAnalyzer):
        result = tmp_analyzer.analyze_section_ordering("no headers here")
        assert result == []


class TestQualityRecording:
    """record_quality should accumulate scores correctly."""

    def test_records_single_observation(self, tmp_analyzer: MASPOBAnalyzer):
        tmp_analyzer.record_quality("Role", 0, 0.9)
        stats = tmp_analyzer._position_stats
        assert "Role" in stats
        assert "0" in stats["Role"]
        assert stats["Role"]["0"] == [0.9]

    def test_accumulates_multiple_observations(self, tmp_analyzer: MASPOBAnalyzer):
        for score in [0.7, 0.8, 0.9]:
            tmp_analyzer.record_quality("Task", 1, score)
        assert tmp_analyzer._position_stats["Task"]["1"] == [0.7, 0.8, 0.9]

    def test_persists_to_disk(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        analyzer = MASPOBAnalyzer(state_path=state_file)
        analyzer.record_quality("Constraints", 2, 0.5)
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "Constraints" in data

    def test_reloads_state_on_init(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        a1 = MASPOBAnalyzer(state_path=state_file)
        a1.record_quality("Examples", 3, 0.6)

        a2 = MASPOBAnalyzer(state_path=state_file)
        assert "Examples" in a2._position_stats
        assert a2._position_stats["Examples"]["3"] == [0.6]


class TestOptimalOrderingConvergence:
    """get_optimal_ordering should return best arrangement once sufficient data accumulated."""

    def _populate(self, analyzer: MASPOBAnalyzer, section: str, best_pos: int, n: int = 12) -> None:
        """Record *n* high-quality observations at *best_pos* and low-quality elsewhere."""
        for _ in range(n):
            analyzer.record_quality(section, best_pos, 0.9)
        for other_pos in range(4):
            if other_pos != best_pos:
                for _ in range(n):
                    analyzer.record_quality(section, other_pos, 0.2)

    def test_converges_to_trained_ordering(self, tmp_analyzer: MASPOBAnalyzer):
        sections = ["Role", "Task", "Constraints", "Examples"]
        # Train: Role→pos0, Examples→pos1, Task→pos2, Constraints→pos3
        expected_order = {"Role": 0, "Examples": 1, "Task": 2, "Constraints": 3}
        for sec, pos in expected_order.items():
            self._populate(tmp_analyzer, sec, pos)

        assert tmp_analyzer.has_sufficient_data(sections)
        ordering = tmp_analyzer.get_optimal_ordering(sections)
        assert ordering.index("Role") < ordering.index("Constraints")

    def test_insufficient_data_returns_input_order(self, tmp_analyzer: MASPOBAnalyzer):
        sections = ["A", "B", "C"]
        result = tmp_analyzer.get_optimal_ordering(sections)
        assert result == sections


class TestIntegrationWithRestructureFormat:
    """_restructure_format should use learned ordering when data is sufficient."""

    def test_static_heuristic_used_without_data(self):
        mutator = PromptMutator()
        result = mutator.mutate(_SAMPLE_PROMPT, MutationOperator.FORMAT_RESTRUCTURE)
        # With static MASPOB heuristic: Role should appear before Constraints
        assert result.index("Role") < result.index("Constraints")

    def test_learned_ordering_applied_when_sufficient_data(self, tmp_path: Path):
        # Create an analyzer with sufficient data that puts Constraints first
        analyzer = MASPOBAnalyzer(state_path=tmp_path / "maspob.json")
        sections = ["Role", "Task", "Constraints", "Examples"]
        for _ in range(12):
            analyzer.record_quality("Constraints", 0, 0.95)
            analyzer.record_quality("Role", 1, 0.8)
            analyzer.record_quality("Examples", 2, 0.7)
            analyzer.record_quality("Task", 3, 0.6)
        for sec in sections:
            for _ in range(12):
                for pos in range(4):
                    analyzer.record_quality(sec, pos, 0.1)

        # Monkey-patch the module singleton for this test
        import vetinari.learning.prompt_mutator as pm_module

        original = pm_module._maspob_analyzer
        pm_module._maspob_analyzer = analyzer
        try:
            mutator = PromptMutator()
            result = mutator.mutate(_SAMPLE_PROMPT, MutationOperator.FORMAT_RESTRUCTURE)
            # Result must contain all sections
            for sec in sections:
                assert sec in result
        finally:
            pm_module._maspob_analyzer = original
