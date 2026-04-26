"""Tests for RulesManager correction extraction and rule proposal from corrections."""

from __future__ import annotations

import pytest

from vetinari.rules_manager import RulesManager
from vetinari.types import AgentType


@pytest.fixture
def rules_manager(tmp_path) -> RulesManager:
    """Fresh RulesManager backed by a temporary file."""
    return RulesManager(rules_file=tmp_path / "rules.yaml")


# ---------------------------------------------------------------------------
# extract_correction — happy path
# ---------------------------------------------------------------------------


class TestExtractCorrection:
    def test_incomplete_output_detected_when_corrected_is_much_longer(self, rules_manager: RulesManager) -> None:
        short = "Yes."
        long = "Yes. " + ("detailed content " * 20)
        result = rules_manager.extract_correction(short, long)

        assert result["correction_type"] == "incomplete"
        assert "expanded" in result["specific_change"].lower()
        assert "complete" in result["generalized_rule"].lower()

    def test_verbose_output_detected_when_corrected_is_much_shorter(self, rules_manager: RulesManager) -> None:
        long = "word " * 100
        short = "word " * 10
        result = rules_manager.extract_correction(long, short)

        assert result["correction_type"] == "verbose"
        assert "shortened" in result["specific_change"].lower()
        assert "concise" in result["generalized_rule"].lower()

    def test_content_correction_detected_when_length_similar_but_text_differs(
        self, rules_manager: RulesManager
    ) -> None:
        original = "The answer is 42 and the sky is green."
        corrected = "The answer is 42 and the sky is blue. "  # same length, different content
        result = rules_manager.extract_correction(original, corrected)

        assert result["correction_type"] == "content"
        assert "modified" in result["specific_change"].lower()

    def test_content_correction_uses_context_as_rule_when_provided(self, rules_manager: RulesManager) -> None:
        original = "The answer is 42 and the sky is green."
        corrected = "The answer is 42 and the sky is blue.  "
        context = "Always verify colour facts from the knowledge base"
        result = rules_manager.extract_correction(original, corrected, context=context)

        assert result["generalized_rule"] == context

    def test_unknown_type_when_outputs_are_identical(self, rules_manager: RulesManager) -> None:
        text = "Identical output"
        result = rules_manager.extract_correction(text, text)

        assert result["correction_type"] == "unknown"
        assert result["generalized_rule"] == ""

    def test_evidence_field_contains_specific_change_and_context(self, rules_manager: RulesManager) -> None:
        short = "Hi."
        long = "Hi. " + ("extra content " * 20)
        result = rules_manager.extract_correction(short, long, context="Agent was too brief")

        assert "Correction applied:" in result["evidence"]
        assert "Agent was too brief" in result["evidence"]

    def test_evidence_truncates_context_at_200_chars(self, rules_manager: RulesManager) -> None:
        short = "Hi."
        long = "Hi. " + ("x " * 20)
        # Use a context whose first 200 and last 100 chars are distinguishably different
        long_context = "A" * 200 + "B" * 100
        result = rules_manager.extract_correction(short, long, context=long_context)

        # The evidence field embeds context[:200] — the B-suffix must not appear
        assert "B" not in result["evidence"]
        assert "A" * 10 in result["evidence"]

    @pytest.mark.parametrize(
        ("original", "corrected", "expected_type"),
        [
            ("short", "short " + "word " * 30, "incomplete"),
            ("word " * 50, "word " * 5, "verbose"),
        ],
    )
    def test_parametrized_length_detection(
        self,
        rules_manager: RulesManager,
        original: str,
        corrected: str,
        expected_type: str,
    ) -> None:
        result = rules_manager.extract_correction(original, corrected)
        assert result["correction_type"] == expected_type


# ---------------------------------------------------------------------------
# propose_rule_from_correction — happy path
# ---------------------------------------------------------------------------


class TestProposeRuleFromCorrection:
    def test_returns_false_when_outputs_identical_and_no_rule_generalized(self, rules_manager: RulesManager) -> None:
        text = "Same output"
        accepted = rules_manager.propose_rule_from_correction(
            agent_type=AgentType.WORKER.value,
            mode="code",
            original_output=text,
            corrected_output=text,
        )
        assert accepted is False

    def test_returns_false_on_first_two_observations(self, rules_manager: RulesManager) -> None:
        short = "Brief."
        long = "Brief. " + ("more detail " * 20)

        for _ in range(2):
            accepted = rules_manager.propose_rule_from_correction(
                agent_type=AgentType.WORKER.value,
                mode="code",
                original_output=short,
                corrected_output=long,
            )
            assert accepted is False

    def test_returns_true_on_third_observation_and_rule_accepted(self, rules_manager: RulesManager) -> None:
        short = "Brief."
        long = "Brief. " + ("more detail " * 20)

        for _ in range(2):
            rules_manager.propose_rule_from_correction(
                agent_type=AgentType.WORKER.value,
                mode="code",
                original_output=short,
                corrected_output=long,
            )

        accepted = rules_manager.propose_rule_from_correction(
            agent_type=AgentType.WORKER.value,
            mode="code",
            original_output=short,
            corrected_output=long,
        )
        assert accepted is True

        # Rule should now appear in agent rules
        agent_rules = rules_manager.get_agent_rules(AgentType.WORKER.value)
        assert any("complete" in r.lower() for r in agent_rules)

    def test_model_specific_rule_stored_under_model_when_model_name_given(self, rules_manager: RulesManager) -> None:
        short = "OK"
        long = "OK. " + ("extended " * 30)

        for _ in range(3):
            rules_manager.propose_rule_from_correction(
                agent_type=AgentType.WORKER.value,
                mode="code",
                original_output=short,
                corrected_output=long,
                model_name="qwen2.5-7b",
            )

        model_rules = rules_manager.get_model_rules("qwen2.5-7b")
        assert any("complete" in r.lower() for r in model_rules)

    def test_no_rule_proposed_when_context_empty_and_content_correction(self, rules_manager: RulesManager) -> None:
        # Content correction with no context falls back to a default rule
        original = "The sky is green and clear today ok"
        corrected = "The sky is blue  and clear today ok"
        # Should not raise; returns a default generalized rule
        accepted = rules_manager.propose_rule_from_correction(
            agent_type=AgentType.INSPECTOR.value,
            mode="review",
            original_output=original,
            corrected_output=corrected,
            context="",
        )
        # First call is always False (1 of 3 observations)
        assert accepted is False
        proposed = rules_manager.get_proposed_rules()
        assert len(proposed) == 1
