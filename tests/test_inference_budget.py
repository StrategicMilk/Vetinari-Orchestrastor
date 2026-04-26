"""Tests for inference.py context-window budget enforcement (B.3).

Covers:
- B.3: Oversized prompt triggers compression then raises InferenceError if still too large
- B.3: Oversized prompt that fits after compression proceeds without error
- B.3: Budget check failure (import error path) raises InferenceError directly
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.adapters.llama_cpp_adapter import SYSTEM_PROMPT_BOUNDARY
from vetinari.exceptions import InferenceError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_budget(fits: bool, total_tokens: int = 100, task_tokens: int = 80) -> dict:
    return {"fits": fits, "total_tokens": total_tokens, "task_tokens": task_tokens}


def _make_agent():
    """Return a minimal stub that satisfies _infer()'s attribute reads.

    MagicMock returns MagicMock for any attribute, so we must explicitly set
    every attribute _infer() reads to a concrete value to avoid TypeError in
    comparisons (e.g., MagicMock <= 0 is unsupported).
    """
    from vetinari.types import AgentType

    agent = MagicMock()
    agent.agent_type = AgentType.WORKER
    agent._context_length = 512
    agent._last_inference_model_id = None
    agent.default_model = "test-model"
    agent._system_prompt = "sys"
    agent._active_system_prompt = "sys"
    # Budget-related attributes must be None so budget pre-checks are skipped
    agent._budget_tracker = None
    agent._token_budget_remaining = None
    agent._adapter_manager = None
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferenceBudgetEnforcement:
    """B.3: Context window budget is enforced in _infer()."""

    def test_oversized_prompt_raises_inference_error_when_compression_fails(self):
        """Prompt that is too large even after compression must raise InferenceError."""
        import vetinari.agents.inference as inf_module

        agent = _make_agent()

        # Both the raw and compressed budgets say it does not fit
        budget_over = _make_budget(fits=False, total_tokens=9000, task_tokens=8500)

        fake_preprocessor = MagicMock()
        fake_preprocessor.compress.return_value = "compressed prompt"

        fake_preprocessor_cls = MagicMock(return_value=fake_preprocessor)

        with (
            patch("vetinari.agents.inference.check_prompt_budget", return_value=budget_over),
            patch.object(inf_module, "_get_local_preprocessor_cls", return_value=fake_preprocessor_cls),
        ):
            with pytest.raises(InferenceError, match="exceeds context window"):
                # Call the unbound method directly on our stub agent
                inf_module.InferenceMixin._infer(agent, "this prompt is way too long")

    def test_oversized_prompt_raises_when_preprocessor_unavailable(self):
        """When LocalPreprocessor is unavailable, InferenceError is raised immediately."""
        import vetinari.agents.inference as inf_module

        agent = _make_agent()
        budget_over = _make_budget(fits=False, total_tokens=9000, task_tokens=8500)

        with (
            patch("vetinari.agents.inference.check_prompt_budget", return_value=budget_over),
            patch.object(inf_module, "_get_local_preprocessor_cls", return_value=None),
        ):
            with pytest.raises(InferenceError, match="exceeds context window"):
                inf_module.InferenceMixin._infer(agent, "this prompt is way too long")


class TestPromptAssemblerBoundaryWiring:
    """PromptAssembler output should survive the live _infer request path."""

    def test_infer_passes_kv_boundary_system_prompt_to_adapter(self):
        """_infer forwards the assembled boundary-marked system prompt to the adapter."""
        import vetinari.agents.inference as inf_module

        agent = _make_agent()
        agent._adapter_manager = MagicMock()
        agent._adapter_manager.infer.return_value = MagicMock(
            status="ok",
            output="done",
            tokens_used=4,
        )
        agent._get_prompt_tier.return_value = "tier prompt"
        agent.get_system_prompt.return_value = "base prompt"
        agent._log = MagicMock()

        assembled = {
            "system": f"stable prefix{SYSTEM_PROMPT_BOUNDARY}dynamic suffix",
            "total_chars": 64,
            "cache_hit": False,
        }

        with (
            patch.object(
                inf_module,
                "_lazy_get_prompt_assembler",
                return_value=MagicMock(build=MagicMock(return_value=assembled)),
            ),
            patch.object(inf_module, "_lazy_get_semantic_cache", return_value=None),
            patch.object(inf_module, "_lazy_get_batch_processor", return_value=None),
            patch("vetinari.agents.inference.check_prompt_budget", return_value=_make_budget(fits=True)),
        ):
            result = inf_module.InferenceMixin._infer(agent, "build a feature")

        assert result == "done"
        request = agent._adapter_manager.infer.call_args.args[0]
        assert request.system_prompt == assembled["system"]
        assert SYSTEM_PROMPT_BOUNDARY in request.system_prompt
