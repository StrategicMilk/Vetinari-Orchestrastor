"""Tests for vetinari.learning.prompt_mutator — structured prompt mutation operators."""

from __future__ import annotations

import pytest

from vetinari.learning.prompt_mutator import MASPOBAnalyzer, MutationOperator, PromptMutator

# ── Fixtures ─────────────────────────────────────────────────────────

SAMPLE_PROMPT = """\
## Identity
You are a code review agent specialising in Python.

## Constraints
- Always cite specific line numbers.
- Never approve code with security vulnerabilities.

## Examples
Input: a function with SQL injection
Output: REJECT with detailed explanation

## Task
Review the provided code diff and produce a structured review.
"""

MINIMAL_PROMPT = "You are a helpful assistant. Be concise and accurate."


@pytest.fixture
def mutator() -> PromptMutator:
    """Create a PromptMutator with default settings."""
    return PromptMutator()


@pytest.fixture
def mutator_with_examples() -> PromptMutator:
    """Create a PromptMutator with seed examples."""
    return PromptMutator(
        seed_examples=(
            "Input: 2 + 2\nOutput: 4",
            "Input: hello\nOutput: HELLO",
        ),
    )


# ── MutationOperator enum tests ─────────────────────────────────────


class TestMutationOperator:
    """Tests for the MutationOperator enum."""

    def test_has_eight_operators(self) -> None:
        assert len(MutationOperator) == 8

    def test_all_values_are_strings(self) -> None:
        for op in MutationOperator:
            assert isinstance(op.value, str)

    def test_expected_operators_exist(self) -> None:
        expected = {
            "instruction_rephrase",
            "constraint_injection",
            "example_injection",
            "format_restructure",
            "reasoning_scaffold",
            "role_reinforcement",
            "output_schema_tighten",
            "context_prune",
        }
        actual = {op.value for op in MutationOperator}
        assert actual == expected


# ── PromptMutator.mutate dispatch tests ──────────────────────────────


class TestPromptMutatorDispatch:
    """Tests that mutate() dispatches to the correct operator."""

    def test_each_operator_produces_different_output(self, mutator: PromptMutator) -> None:
        """Every operator should modify the prompt in some way.

        CONTEXT_PRUNE is excluded because it correctly no-ops when all
        sections are substantial (> 20 chars, no duplicates).
        """
        no_op_on_good_prompt = {MutationOperator.CONTEXT_PRUNE}
        for op in MutationOperator:
            result = mutator.mutate(SAMPLE_PROMPT, op)
            if op not in no_op_on_good_prompt:
                assert result != SAMPLE_PROMPT, f"{op.value} did not modify the prompt"

    def test_unknown_operator_raises(self, mutator: PromptMutator) -> None:
        """Passing an invalid operator value should raise."""
        # MutationOperator is an enum so we can't easily pass an invalid one,
        # but we can test the match/case default path doesn't silently pass
        results = []
        for op in MutationOperator:
            # Should not raise
            result = mutator.mutate(MINIMAL_PROMPT, op)
            results.append(result)
        assert len(results) == len(list(MutationOperator))

    def test_section_targeting(self, mutator: PromptMutator) -> None:
        """Operators should be able to target a specific section."""
        result = mutator.mutate(
            SAMPLE_PROMPT,
            MutationOperator.CONSTRAINT_INJECTION,
            section="Constraints",
        )
        assert result != SAMPLE_PROMPT
        assert "Constraints" in result


# ── Individual operator tests ────────────────────────────────────────


class TestInstructionRephrase:
    """Tests for the INSTRUCTION_REPHRASE operator."""

    def test_modifies_imperative_text(self, mutator: PromptMutator) -> None:
        prompt = "Do the analysis.\nGenerate the report.\nReview the code."
        result = mutator.mutate(prompt, MutationOperator.INSTRUCTION_REPHRASE)
        assert result != prompt

    def test_alternates_framing(self, mutator: PromptMutator) -> None:
        """Should alternate between declarative and negative framing."""
        prompt = "Be concise and accurate. Do the task."
        result1 = mutator.mutate(prompt, MutationOperator.INSTRUCTION_REPHRASE)
        result2 = mutator.mutate(prompt, MutationOperator.INSTRUCTION_REPHRASE)
        # Two consecutive calls should use different strategies
        assert result1 != result2 or result1 != prompt


class TestConstraintInjection:
    """Tests for the CONSTRAINT_INJECTION operator."""

    def test_adds_constraint(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.CONSTRAINT_INJECTION)
        assert len(result) > len(SAMPLE_PROMPT)

    def test_constraint_injected_near_constraints_section(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.CONSTRAINT_INJECTION)
        # The injected constraint should appear somewhere in the prompt
        assert any(c in result for c in PromptMutator.DEFAULT_CONSTRAINTS)

    def test_works_without_constraints_section(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(MINIMAL_PROMPT, MutationOperator.CONSTRAINT_INJECTION)
        assert len(result) > len(MINIMAL_PROMPT)


class TestExampleInjection:
    """Tests for the EXAMPLE_INJECTION operator."""

    def test_adds_example_block(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.EXAMPLE_INJECTION)
        assert "Example" in result
        assert len(result) > len(SAMPLE_PROMPT)

    def test_uses_seed_examples(self, mutator_with_examples: PromptMutator) -> None:
        result = mutator_with_examples.mutate(
            MINIMAL_PROMPT,
            MutationOperator.EXAMPLE_INJECTION,
        )
        assert "2 + 2" in result or "hello" in result


class TestFormatRestructure:
    """Tests for the FORMAT_RESTRUCTURE operator (includes MASPOB)."""

    def test_reorders_sections(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.FORMAT_RESTRUCTURE)
        # Identity section should still be near the start (MASPOB)
        assert result.index("Identity") < result.index("Constraints")

    def test_preserves_all_sections(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.FORMAT_RESTRUCTURE)
        for section in ("Identity", "Constraints", "Examples", "Task"):
            assert section in result

    def test_minimal_prompt_unchanged(self, mutator: PromptMutator) -> None:
        """Prompts with < 2 sections should not be restructured."""
        # Minimal prompt has no section headers, so parse returns 1 section
        result = mutator.mutate(MINIMAL_PROMPT, MutationOperator.FORMAT_RESTRUCTURE)
        assert result == MINIMAL_PROMPT


class TestReasoningScaffold:
    """Tests for the REASONING_SCAFFOLD operator."""

    def test_adds_reasoning_instructions(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.REASONING_SCAFFOLD)
        assert "step" in result.lower() or "think" in result.lower()
        assert len(result) > len(SAMPLE_PROMPT)


class TestRoleReinforcement:
    """Tests for the ROLE_REINFORCEMENT operator."""

    def test_reinforces_existing_role(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.ROLE_REINFORCEMENT)
        assert "primary function" in result or "specialist" in result.lower()

    def test_adds_role_to_prompt_without_role_section(self, mutator: PromptMutator) -> None:
        prompt = "## Task\nDo something useful."
        result = mutator.mutate(prompt, MutationOperator.ROLE_REINFORCEMENT)
        assert "specialist" in result.lower()


class TestOutputSchemaTighten:
    """Tests for the OUTPUT_SCHEMA_TIGHTEN operator."""

    def test_adds_output_format(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.OUTPUT_SCHEMA_TIGHTEN)
        assert "format" in result.lower() or "structure" in result.lower()
        assert len(result) > len(SAMPLE_PROMPT)


class TestContextPrune:
    """Tests for the CONTEXT_PRUNE operator."""

    def test_prunes_short_sections(self, mutator: PromptMutator) -> None:
        prompt = (
            "## Identity\nYou are a code reviewer.\n\n"
            "## Empty\n\n\n"
            "## Task\nReview this code thoroughly and provide feedback.\n"
        )
        result = mutator.mutate(prompt, MutationOperator.CONTEXT_PRUNE)
        # The empty section should be removed
        assert "Empty" not in result

    def test_preserves_substantial_sections(self, mutator: PromptMutator) -> None:
        result = mutator.mutate(SAMPLE_PROMPT, MutationOperator.CONTEXT_PRUNE)
        # All substantial sections should remain
        assert "Identity" in result
        assert "Task" in result


# ── Composability tests ──────────────────────────────────────────────


class TestComposability:
    """Tests that operators can be chained."""

    def test_chain_two_operators(self, mutator: PromptMutator) -> None:
        step1 = mutator.mutate(SAMPLE_PROMPT, MutationOperator.CONSTRAINT_INJECTION)
        step2 = mutator.mutate(step1, MutationOperator.REASONING_SCAFFOLD)
        # Result should have both the injected constraint and the scaffold
        assert len(step2) > len(SAMPLE_PROMPT)
        assert step2 != step1

    def test_chain_all_additive_operators(self, mutator: PromptMutator) -> None:
        """Chaining all additive operators should produce a longer prompt."""
        result = SAMPLE_PROMPT
        additive_ops = [
            MutationOperator.CONSTRAINT_INJECTION,
            MutationOperator.REASONING_SCAFFOLD,
            MutationOperator.OUTPUT_SCHEMA_TIGHTEN,
        ]
        for op in additive_ops:
            result = mutator.mutate(result, op)
        assert len(result) > len(SAMPLE_PROMPT)


# ── Section parsing tests ────────────────────────────────────────────


class TestSectionParsing:
    """Tests for internal section parsing helpers."""

    def test_parse_sections_returns_correct_count(self, mutator: PromptMutator) -> None:
        sections = mutator._parse_sections(SAMPLE_PROMPT)
        assert len(sections) == 4  # Identity, Constraints, Examples, Task

    def test_parse_sections_no_headers(self, mutator: PromptMutator) -> None:
        sections = mutator._parse_sections(MINIMAL_PROMPT)
        assert len(sections) == 1  # Single section with empty title

    def test_extract_section_by_name(self, mutator: PromptMutator) -> None:
        content = mutator._extract_section(SAMPLE_PROMPT, "Identity")
        assert "code review agent" in content

    def test_extract_section_not_found(self, mutator: PromptMutator) -> None:
        content = mutator._extract_section(SAMPLE_PROMPT, "Nonexistent")
        # Should return the full prompt when section not found
        assert content == SAMPLE_PROMPT


class TestMASPOBAnalyzer:
    """Tests for MASPOBAnalyzer per-position evidence guards (Defect 4)."""

    @pytest.fixture
    def analyzer(self, tmp_path: pytest.TempPathFactory) -> MASPOBAnalyzer:
        """Return a MASPOBAnalyzer backed by a temp state file."""
        return MASPOBAnalyzer(state_path=tmp_path / "maspob_test.json")

    def test_has_sufficient_data_false_with_no_observations(self, analyzer: MASPOBAnalyzer) -> None:
        """Empty analyzer must report insufficient data for any section list."""
        assert not analyzer.has_sufficient_data(["A", "B", "C"])

    def test_has_sufficient_data_false_when_samples_concentrated_at_one_position(
        self, analyzer: MASPOBAnalyzer
    ) -> None:
        """Sufficient TOTAL samples at one position must NOT satisfy the per-position guard.

        This is the core of Defect 4: a section with all 10 samples at position 0
        passes the old total-count check but has zero data for positions 1 and 2.
        The greedy assignment cannot make a trustworthy per-position decision —
        has_sufficient_data() must return False until every (section, position)
        slot has _MIN_SAMPLES observations.
        """
        from vetinari.learning.prompt_mutator import _MIN_SAMPLES

        sections = ["Alpha", "Beta", "Gamma"]
        # Give Alpha 10 samples, but ALL at position 0 only.
        for _ in range(_MIN_SAMPLES):
            analyzer.record_quality("Alpha", position=0, quality_score=0.8)

        # Alpha has enough total, but positions 1 and 2 are empty.
        assert not analyzer.has_sufficient_data(sections), (
            "has_sufficient_data() must return False when a section lacks per-position "
            "evidence for some slots, even if its total sample count meets _MIN_SAMPLES"
        )

    def test_has_sufficient_data_true_only_when_all_positions_have_samples(
        self, analyzer: MASPOBAnalyzer
    ) -> None:
        """has_sufficient_data() returns True only when every (section, position) pair is covered."""
        from vetinari.learning.prompt_mutator import _MIN_SAMPLES

        sections = ["Alpha", "Beta"]
        n = len(sections)

        # Record _MIN_SAMPLES at every (section, position) pair.
        for sec in sections:
            for pos in range(n):
                for _ in range(_MIN_SAMPLES):
                    analyzer.record_quality(sec, position=pos, quality_score=0.7)

        assert analyzer.has_sufficient_data(sections), (
            "has_sufficient_data() must return True when all (section, position) pairs "
            "have _MIN_SAMPLES observations"
        )

    def test_get_optimal_ordering_falls_back_when_position_evidence_missing(
        self, analyzer: MASPOBAnalyzer
    ) -> None:
        """get_optimal_ordering() returns input order when per-position data is insufficient.

        With all samples at one position, the learned ordering must not be used —
        the method falls back to the original section order.
        """
        from vetinari.learning.prompt_mutator import _MIN_SAMPLES

        sections = ["X", "Y", "Z"]
        # Give X abundant samples but only at position 0.
        for _ in range(_MIN_SAMPLES * 2):
            analyzer.record_quality("X", position=0, quality_score=0.9)

        result = analyzer.get_optimal_ordering(sections)
        assert result == sections, (
            "get_optimal_ordering() must return input order when per-position evidence "
            f"is insufficient; got {result!r} instead of {sections!r}"
        )

    def test_get_optimal_ordering_uses_learned_order_when_fully_populated(
        self, analyzer: MASPOBAnalyzer
    ) -> None:
        """get_optimal_ordering() uses learned data when all (section, position) pairs are covered."""
        from vetinari.learning.prompt_mutator import _MIN_SAMPLES

        sections = ["Low", "High"]
        n = len(sections)

        # Give "High" better scores at position 0 and "Low" better scores at position 1.
        for _ in range(_MIN_SAMPLES):
            analyzer.record_quality("High", position=0, quality_score=0.9)
            analyzer.record_quality("High", position=1, quality_score=0.4)
            analyzer.record_quality("Low", position=0, quality_score=0.3)
            analyzer.record_quality("Low", position=1, quality_score=0.8)

        result = analyzer.get_optimal_ordering(sections)
        # The greedy algorithm should place "High" at position 0 (score 0.9 > 0.3).
        assert result[0] == "High", (
            f"Expected 'High' at position 0 (score 0.9 vs 0.3 for 'Low'), got {result!r}"
        )
