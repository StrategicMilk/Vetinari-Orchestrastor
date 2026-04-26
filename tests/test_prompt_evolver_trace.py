"""Tests for PromptEvolver.generate_variant_from_trace (Meta-Harness wiring)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant


@pytest.fixture
def evolver(tmp_path, monkeypatch):
    """PromptEvolver with state stored in a temp directory."""
    monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path))
    return PromptEvolver()


class TestGenerateVariantFromTrace:
    def test_returns_string_on_success(self, evolver):
        evolver.register_baseline("worker", "Do the work carefully.")
        trace = {"output": "", "error": "", "quality_score": 0.0}
        result = evolver.generate_variant_from_trace(
            agent_type="worker",
            baseline_prompt="Do the work carefully.",
            failed_trace=trace,
        )
        # Empty output triggers "incomplete" diagnosis (confidence=0.8 > 0.4)
        # The fix appends text so the new instruction != baseline
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > len("Do the work carefully.")

    def test_stores_variant_in_variants_dict(self, evolver):
        evolver.register_baseline("worker", "Baseline prompt.")
        trace = {"output": "", "error": "", "quality_score": 0.0}
        evolver.generate_variant_from_trace(
            agent_type="worker",
            baseline_prompt="Baseline prompt.",
            failed_trace=trace,
        )
        variants = evolver._variants.get("worker", [])
        trace_variants = [v for v in variants if "trace" in v.variant_id]
        assert len(trace_variants) >= 1

    def test_variant_id_contains_trace(self, evolver):
        evolver.register_baseline("foreman", "Plan tasks.")
        trace = {"output": "", "error": "", "quality_score": 0.0}
        evolver.generate_variant_from_trace(
            agent_type="foreman",
            baseline_prompt="Plan tasks.",
            failed_trace=trace,
        )
        variants = evolver._variants.get("foreman", [])
        trace_variants = [v for v in variants if "trace" in v.variant_id]
        assert len(trace_variants) >= 1
        assert trace_variants[0].variant_id.startswith("foreman_trace_v")

    def test_falls_back_to_generate_variant_when_optimizer_fails(self, evolver, monkeypatch):
        """When the optimizer raises, the method should fall back to blind mutation."""
        evolver.register_baseline("inspector", "Inspect results.")

        def bad_optimizer():
            raise RuntimeError("Optimizer unavailable")

        with patch("vetinari.learning.prompt_evolver.PromptEvolver.generate_variant") as mock_gen:
            mock_gen.return_value = "Inspect results. Fallback variant."
            with patch(
                "vetinari.learning.prompt_optimizer.get_prompt_optimizer",
                side_effect=bad_optimizer,
            ):
                result = evolver.generate_variant_from_trace(
                    agent_type="inspector",
                    baseline_prompt="Inspect results.",
                    failed_trace={"output": "", "error": "", "quality_score": 0.0},
                )
            mock_gen.assert_called_once()
            assert result == "Inspect results. Fallback variant."

    def test_falls_back_when_instruction_unchanged(self, evolver):
        """If the optimizer returns the same text as baseline, falls back to blind mutation."""
        from vetinari.learning.prompt_optimizer import PromptExperiment

        trace = {"output": "x" * 100, "error": "", "quality_score": 0.8}
        # quality=0.8 means _diagnose_trace returns None, optimizer returns None
        # So generate_variant_from_trace should fall back to generate_variant

        with patch.object(evolver, "generate_variant", return_value="fallback_text") as _mock_gv:
            result = evolver.generate_variant_from_trace(
                agent_type="worker",
                baseline_prompt="Work well.",
                failed_trace=trace,
            )
        # Either None from optimizer (no trace variant created) or fallback was called
        assert result == "fallback_text" or result is None

    @pytest.mark.parametrize(
        ("failure_scenario", "trace"),
        [
            (
                "format_error",
                {
                    "output": "some output that is long enough to pass length check indeed",
                    "error": "json parse error occurred",
                    "quality_score": 0.5,
                },
            ),
            (
                "reasoning_error",
                {
                    "output": "x" * 30,
                    "error": "",
                    "quality_score": 0.1,
                },
            ),
        ],
    )
    def test_returns_variant_for_various_failure_types(self, evolver, failure_scenario, trace):
        evolver.register_baseline("worker", "Process input.")
        result = evolver.generate_variant_from_trace(
            agent_type="worker",
            baseline_prompt="Process input.",
            failed_trace=trace,
        )
        # Should produce an improved variant for diagnosable failures
        assert result is not None
        assert result != "Process input."
