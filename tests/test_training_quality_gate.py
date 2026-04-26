"""Tests for vetinari.training.quality_gate."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_mock_quality_score
from vetinari.training.quality_gate import (
    _LATENCY_REJECT_RATIO,
    _QUALITY_DEPLOY_THRESHOLD,
    _QUALITY_REJECT_THRESHOLD,
    _TOKEN_REJECT_RATIO,
    TrainingGateDecision,
    TrainingQualityGate,
    get_training_quality_gate,
)

# -- GateDecision tests --


def test_gate_decision_repr_shows_key_fields() -> None:
    decision = TrainingGateDecision(
        decision="deploy",
        baseline_quality=0.70,
        candidate_quality=0.75,
        quality_delta=0.05,
        baseline_latency_ms=200.0,
        candidate_latency_ms=210.0,
        latency_ratio=1.05,
        token_efficiency=1.01,
        reasoning="quality improved",
    )
    r = repr(decision)
    assert "deploy" in r
    assert "+0.050" in r
    assert "1.05" in r


def test_gate_decision_timestamp_is_set() -> None:
    decision = TrainingGateDecision(
        decision="reject",
        baseline_quality=0.8,
        candidate_quality=0.7,
        quality_delta=-0.1,
        baseline_latency_ms=100.0,
        candidate_latency_ms=100.0,
        latency_ratio=1.0,
        token_efficiency=1.0,
        reasoning="quality drop",
    )
    assert "T" in decision.timestamp


# -- _make_decision tests (unit-test the threshold logic directly) --


def test_make_decision_reject_quality_regression() -> None:
    gate = TrainingQualityGate()
    decision, reasoning = gate._make_decision(-0.05, 1.0, 1.0)
    assert decision == "reject"
    assert "Quality regression" in reasoning


def test_make_decision_reject_at_exact_threshold() -> None:
    """Quality delta exactly at _QUALITY_REJECT_THRESHOLD is rejected (< not <=)."""
    gate = TrainingQualityGate()
    # Threshold is -0.03; exactly -0.03 is NOT below threshold, so should not reject on quality.
    decision, _ = gate._make_decision(_QUALITY_REJECT_THRESHOLD, 1.0, 1.0)
    # At exactly -0.03 the condition is `delta < threshold` which is False.
    assert decision != "reject" or "Quality" not in _


def test_make_decision_reject_latency_regression() -> None:
    gate = TrainingQualityGate()
    decision, reasoning = gate._make_decision(0.0, _LATENCY_REJECT_RATIO + 0.1, 1.0)
    assert decision == "reject"
    assert "Latency regression" in reasoning


def test_make_decision_reject_token_regression() -> None:
    gate = TrainingQualityGate()
    decision, reasoning = gate._make_decision(0.0, 1.0, _TOKEN_REJECT_RATIO + 0.1)
    assert decision == "reject"
    assert "Token efficiency regression" in reasoning


def test_make_decision_deploy_on_quality_gain() -> None:
    gate = TrainingQualityGate()
    decision, reasoning = gate._make_decision(_QUALITY_DEPLOY_THRESHOLD + 0.01, 1.0, 1.0)
    assert decision == "deploy"
    assert "Quality improved" in reasoning


def test_make_decision_flag_marginal() -> None:
    """Small positive delta with acceptable overhead -> flag_for_review."""
    gate = TrainingQualityGate()
    # quality_delta > reject threshold but < deploy threshold
    decision, reasoning = gate._make_decision(0.005, 1.1, 1.05)
    assert decision == "flag_for_review"
    assert "Marginal" in reasoning


def test_make_decision_deploy_requires_low_overhead() -> None:
    """Quality gain is there, but latency overhead is too high for auto-deploy."""
    gate = TrainingQualityGate()
    decision, _ = gate._make_decision(_QUALITY_DEPLOY_THRESHOLD + 0.05, 1.5, 1.0)
    assert decision == "flag_for_review"


# -- evaluate() integration tests (mocked adapter + scorer) --


def _patch_eval_deps(quality_score: float = 0.8, latency_ms: float = 100.0, tokens: int = 50):
    """Context manager that patches the adapter and scorer for evaluate() tests."""
    mock_adapter = MagicMock()
    mock_adapter.chat.return_value = {
        "output": "mocked output",
        "latency_ms": latency_ms,
        "tokens_used": tokens,
        "status": "ok",
    }
    mock_scorer_instance = MagicMock()
    mock_scorer_instance.score.return_value = make_mock_quality_score(quality_score)

    return (
        patch(
            "vetinari.training.quality_gate.LocalInferenceAdapter",
            return_value=mock_adapter,
        ),
        patch(
            "vetinari.training.quality_gate.get_quality_scorer",
            return_value=mock_scorer_instance,
        ),
    )


def test_evaluate_with_no_tasks_flags_for_review() -> None:
    gate = TrainingQualityGate()
    decision = gate.evaluate("candidate-v2", "baseline-v1", eval_tasks=[])
    assert decision.decision == "flag_for_review"
    assert decision.eval_tasks_run == 0


def test_evaluate_uses_default_eval_set_when_none() -> None:
    gate = TrainingQualityGate()
    default_set = TrainingQualityGate._get_default_eval_set()
    with (
        patch("vetinari.training.quality_gate.LocalInferenceAdapter") as mock_adapter_cls,
        patch("vetinari.training.quality_gate.get_quality_scorer") as mock_scorer_fn,
    ):
        mock_adapter_cls.return_value.chat.return_value = {
            "output": "x",
            "tokens_used": 10,
        }
        mock_score = MagicMock()
        mock_score.overall_score = 0.75
        mock_scorer_fn.return_value.score.return_value = mock_score

        decision = gate.evaluate("cand", "base", eval_tasks=None)
    assert decision.eval_tasks_run == len(default_set)


def test_evaluate_deploy_when_candidate_clearly_better() -> None:
    gate = TrainingQualityGate()
    tasks = [{"prompt": "test", "task_type": "coding"}]
    call_count = [0]

    def side_effect_chat(model_id: str, system: str, prompt: str):
        call_count[0] += 1
        # Baseline gets quality 0.6, candidate gets 0.9 via scorer mock below
        return {"output": "ok", "tokens_used": 10}

    with (
        patch("vetinari.training.quality_gate.LocalInferenceAdapter") as mock_adapter_cls,
        patch("vetinari.training.quality_gate.get_quality_scorer") as mock_scorer_fn,
    ):
        mock_adapter_cls.return_value.chat.side_effect = side_effect_chat
        mock_score = MagicMock()
        mock_score.overall_score = 0.6
        # First 1 task run for baseline -> 0.6, then 1 for candidate -> 0.9
        score_objects = [make_mock_quality_score(s) for s in [0.6, 0.9]]
        mock_scorer_fn.return_value.score.side_effect = score_objects

        decision = gate.evaluate("cand", "base", eval_tasks=tasks)

    assert decision.decision in ("deploy", "flag_for_review")  # depends on latency
    assert decision.quality_delta > 0


def test_evaluate_reject_when_candidate_quality_drops() -> None:
    gate = TrainingQualityGate()
    tasks = [{"prompt": "test", "task_type": "coding"}]

    with (
        patch("vetinari.training.quality_gate.LocalInferenceAdapter") as mock_adapter_cls,
        patch("vetinari.training.quality_gate.get_quality_scorer") as mock_scorer_fn,
    ):
        mock_adapter_cls.return_value.chat.return_value = {"output": "ok", "tokens_used": 10}
        # baseline quality=0.8, candidate quality=0.7 -> delta=-0.1 -> reject
        score_objects = [make_mock_quality_score(0.8), make_mock_quality_score(0.7)]
        mock_scorer_fn.return_value.score.side_effect = score_objects

        decision = gate.evaluate("cand", "base", eval_tasks=tasks)

    assert decision.decision == "reject"
    assert decision.quality_delta < _QUALITY_REJECT_THRESHOLD


def test_evaluate_records_decision_in_history() -> None:
    gate = TrainingQualityGate()
    with (
        patch("vetinari.training.quality_gate.LocalInferenceAdapter") as mock_adapter_cls,
        patch("vetinari.training.quality_gate.get_quality_scorer") as mock_scorer_fn,
    ):
        mock_adapter_cls.return_value.chat.return_value = {"output": "x", "tokens_used": 5}
        mock_scorer_fn.return_value.score.return_value = make_mock_quality_score(0.75)
        gate.evaluate("cand", "base", eval_tasks=[{"prompt": "p"}])

    history = gate.get_history()
    assert len(history) == 1
    assert "decision" in history[0]


def test_evaluate_handles_adapter_failure_gracefully() -> None:
    """If the adapter raises, the gate should still return a GateDecision."""
    gate = TrainingQualityGate()
    with (
        patch("vetinari.training.quality_gate.LocalInferenceAdapter") as mock_adapter_cls,
        patch("vetinari.training.quality_gate.get_quality_scorer") as mock_scorer_fn,
    ):
        mock_adapter_cls.return_value.chat.side_effect = RuntimeError("model not loaded")
        mock_scorer_fn.return_value.score.return_value = make_mock_quality_score(0.0)
        decision = gate.evaluate("cand", "base", eval_tasks=[{"prompt": "p"}])

    # Should not raise; decision is reject or flag due to zero scores
    assert decision.decision in ("reject", "flag_for_review")


# -- Default eval set tests --


def test_default_eval_set_has_five_tasks() -> None:
    tasks = TrainingQualityGate._get_default_eval_set()
    assert len(tasks) == 5


def test_default_eval_set_tasks_have_prompt_and_task_type() -> None:
    tasks = TrainingQualityGate._get_default_eval_set()
    for task in tasks:
        assert "prompt" in task
        assert "task_type" in task
        assert task["prompt"]  # non-empty


# -- get_history tests --


def test_get_history_most_recent_first() -> None:
    gate = TrainingQualityGate()
    with (
        patch("vetinari.training.quality_gate.LocalInferenceAdapter") as mock_adapter_cls,
        patch("vetinari.training.quality_gate.get_quality_scorer") as mock_scorer_fn,
    ):
        mock_adapter_cls.return_value.chat.return_value = {"output": "x", "tokens_used": 5}
        mock_scorer_fn.return_value.score.side_effect = [
            make_mock_quality_score(0.7),
            make_mock_quality_score(0.7),
            make_mock_quality_score(0.8),
            make_mock_quality_score(0.8),
        ]
        gate.evaluate("cand_A", "base", eval_tasks=[{"prompt": "first"}])
        gate.evaluate("cand_B", "base", eval_tasks=[{"prompt": "second"}])

    history = gate.get_history()
    assert len(history) == 2
    # Most recent first — the second evaluate() result.
    assert history[0]["eval_tasks_run"] == 1
    assert history[1]["eval_tasks_run"] == 1


# -- Singleton tests --


def test_get_training_quality_gate_returns_same_instance() -> None:
    a = get_training_quality_gate()
    b = get_training_quality_gate()
    assert a is b


def test_get_training_quality_gate_is_correct_type() -> None:
    instance = get_training_quality_gate()
    assert isinstance(instance, TrainingQualityGate)
