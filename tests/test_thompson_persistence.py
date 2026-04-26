"""Tests for Thompson persistence + cold-start (Dept 4.3 #43-44)."""

from __future__ import annotations

import atexit
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestThompsonColdStart:
    """Tests for cold-start seeding via BenchmarkSeeder (not static BENCHMARK_PRIORS).

    Decision: BENCHMARK_PRIORS was emptied in SESSION-2-M1 fix 14 because
    cloud API model names (claude-sonnet-4-20250514) never matched local
    GGUF model IDs, creating phantom arms. Arms are now seeded on first
    observation via _get_informed_prior() -> BenchmarkSeeder.
    """

    def test_benchmark_priors_empty(self):
        """BENCHMARK_PRIORS is intentionally empty — seeding uses BenchmarkSeeder."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        assert len(ThompsonSamplingSelector.BENCHMARK_PRIORS) == 0

    def test_cold_start_no_arms_seeded_from_empty_priors(self):
        """On cold start with empty BENCHMARK_PRIORS, no phantom arms are created."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector

        selector = ThompsonSamplingSelector.__new__(ThompsonSamplingSelector)
        selector._arms = {}
        selector._lock = __import__("threading").Lock()
        selector._seed_from_benchmarks()

        # Empty priors means zero arms seeded — arms are created on first observation
        assert len(selector._arms) == 0

    def test_cold_start_skipped_with_existing_state(self):
        """When arms already loaded, seeding is skipped."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector
        from vetinari.learning.thompson_arms import ThompsonBetaArm

        selector = ThompsonSamplingSelector.__new__(ThompsonSamplingSelector)
        selector._arms = {"existing:arm": ThompsonBetaArm(model_id="existing", task_type="arm")}
        selector._lock = __import__("threading").Lock()
        selector._seed_from_benchmarks()

        # Should still have just the 1 existing arm
        assert len(selector._arms) == 1
        assert "existing:arm" in selector._arms

    def test_real_observations_overwhelm_informed_prior(self):
        """After 20+ updates, arm alpha/beta drift far from informed prior values."""
        from vetinari.learning.thompson_arms import ThompsonBetaArm

        # Simulate an arm seeded by BenchmarkSeeder with informed prior (3.0, 1.5)
        arm = ThompsonBetaArm(model_id="qwen2.5-72b", task_type="coding", alpha=3.0, beta=1.5)
        initial_alpha = arm.alpha

        # Simulate 25 successful updates
        for _ in range(25):
            arm.update(0.9, True)

        # Alpha should have grown significantly beyond initial prior
        assert arm.alpha > initial_alpha + 10


class TestThompsonPersistence:
    """Tests for atexit/shutdown persistence."""

    def test_save_state_called_on_update(self):
        """_save_state() is called after every update()."""
        import inspect

        from vetinari.learning.model_selector import ThompsonSamplingSelector

        source = inspect.getsource(ThompsonSamplingSelector.update)
        assert "_save_state" in source

    def test_atexit_registered_on_singleton(self):
        """get_thompson_selector() registers atexit handler."""
        import inspect

        from vetinari.learning.model_selector import get_thompson_selector

        source = inspect.getsource(get_thompson_selector)
        assert "atexit.register" in source

    def test_shutdown_callback_registered(self):
        """get_thompson_selector() registers shutdown.py callback."""
        import inspect

        from vetinari.learning.model_selector import get_thompson_selector

        source = inspect.getsource(get_thompson_selector)
        assert "register_callback" in source

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load produces equivalent arm state including last_updated."""
        from vetinari.learning.model_selector import ThompsonSamplingSelector
        from vetinari.learning.thompson_arms import ThompsonBetaArm

        selector = ThompsonSamplingSelector.__new__(ThompsonSamplingSelector)
        selector._arms = {}
        selector._lock = __import__("threading").Lock()

        ts = "2025-01-15T12:00:00+00:00"
        arm = ThompsonBetaArm(
            model_id="test-model",
            task_type="coding",
            alpha=5.0,
            beta=3.0,
            last_updated=ts,
        )
        selector._arms["test-model:coding"] = arm

        # Patch _get_state_dir to use tmp_path
        with patch.object(selector, "_get_state_dir", return_value=str(tmp_path)):
            selector._save_state()

        # Create new selector and load via the DB-only path (bypassing prune_stale_arms
        # so we test serialization fidelity for arbitrary timestamps, including old ones).
        selector2 = ThompsonSamplingSelector.__new__(ThompsonSamplingSelector)
        selector2._arms = {}
        selector2._lock = __import__("threading").Lock()
        with patch.object(selector2, "_get_state_dir", return_value=str(tmp_path)):
            selector2._load_state_from_db()

        assert "test-model:coding" in selector2._arms
        loaded = selector2._arms["test-model:coding"]
        assert loaded.alpha == 5.0
        assert loaded.beta == 3.0
        # last_updated must survive the save/load round-trip so that
        # prune_stale_arms uses the correct freshness timestamp after restart.
        assert loaded.last_updated == ts, f"last_updated not preserved: expected {ts!r}, got {loaded.last_updated!r}"


class TestPruneStaleArms:
    """Tests for prune_stale_arms — stale arm removal from Thompson state."""

    def test_prune_removes_old_arm_not_in_known_models(self):
        """Arms older than cutoff and not in known_model_ids are removed."""
        from datetime import datetime, timedelta, timezone

        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.learning.thompson_persistence import prune_stale_arms

        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        arms = {
            "stale-model:coding": ThompsonBetaArm(
                model_id="stale-model", task_type="coding", last_updated=old_timestamp
            )
        }
        pruned = prune_stale_arms(arms, known_model_ids={"active-model"}, days=30)

        assert pruned == 1
        assert "stale-model:coding" not in arms

    def test_prune_preserves_active_model_regardless_of_age(self):
        """Arms for currently-known models are preserved even if old."""
        from datetime import datetime, timedelta, timezone

        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.learning.thompson_persistence import prune_stale_arms

        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        arms = {
            "active-model:coding": ThompsonBetaArm(
                model_id="active-model", task_type="coding", last_updated=old_timestamp
            )
        }
        pruned = prune_stale_arms(arms, known_model_ids={"active-model"}, days=30)

        assert pruned == 0
        assert "active-model:coding" in arms

    def test_prune_preserves_recent_arms_without_known_models(self):
        """Recent arms are not pruned even when known_model_ids is None."""
        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.learning.thompson_persistence import prune_stale_arms

        arms = {"recent-model:general": ThompsonBetaArm(model_id="recent-model", task_type="general")}
        pruned = prune_stale_arms(arms, known_model_ids=None, days=30)

        assert pruned == 0


# -- BetaArm.update_from_signal --


class TestBetaArmUpdateFromSignal:
    """Tests for BetaArm.update_from_signal — typed SuccessSignal update path."""

    def test_success_increments_alpha(self) -> None:
        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.ontology import SuccessSignal

        arm = ThompsonBetaArm(model_id="qwen2-7b", task_type="coding")
        alpha_before = arm.alpha
        sig = SuccessSignal.from_quality_score(0.9, True, model_id="qwen2-7b", task_type="coding")
        arm.update_from_signal(sig)
        assert arm.alpha > alpha_before

    def test_failure_increments_beta(self) -> None:
        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.ontology import SuccessSignal

        arm = ThompsonBetaArm(model_id="qwen2-7b", task_type="coding")
        beta_before = arm.beta
        sig = SuccessSignal.from_quality_score(0.3, False, model_id="qwen2-7b", task_type="coding")
        arm.update_from_signal(sig)
        assert arm.beta > beta_before

    def test_total_pulls_increments(self) -> None:
        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.ontology import SuccessSignal

        arm = ThompsonBetaArm(model_id="llama-3-8b", task_type="general")
        sig = SuccessSignal.from_quality_score(0.8, True)
        arm.update_from_signal(sig)
        assert arm.total_pulls == 1

    def test_equivalent_to_direct_update(self) -> None:
        """update_from_signal and update() with same inputs produce identical state."""
        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.ontology import SuccessSignal

        arm_a = ThompsonBetaArm(model_id="m", task_type="t")
        arm_b = ThompsonBetaArm(model_id="m", task_type="t")
        sig = SuccessSignal.from_quality_score(0.75, True)
        arm_a.update_from_signal(sig)
        arm_b.update(sig.quality_weight, sig.success)
        assert arm_a.alpha == arm_b.alpha
        assert arm_a.beta == arm_b.beta
        assert arm_a.total_pulls == arm_b.total_pulls

    def test_quality_weight_scales_alpha_increment(self) -> None:
        """Higher quality weight produces a larger alpha increment."""
        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.ontology import SuccessSignal

        arm_hi = ThompsonBetaArm(model_id="m", task_type="t")
        arm_lo = ThompsonBetaArm(model_id="m", task_type="t")
        arm_hi.update_from_signal(SuccessSignal.from_quality_score(0.9, True))
        arm_lo.update_from_signal(SuccessSignal.from_quality_score(0.4, True))
        assert arm_hi.alpha > arm_lo.alpha

    def test_prune_returns_count_of_removed_arms(self):
        """Return value equals number of arms removed."""
        from datetime import datetime, timedelta, timezone

        from vetinari.learning.thompson_arms import ThompsonBetaArm
        from vetinari.learning.thompson_persistence import prune_stale_arms

        old_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        arms = {
            "old-a:coding": ThompsonBetaArm(model_id="old-a", task_type="coding", last_updated=old_ts),
            "old-b:general": ThompsonBetaArm(model_id="old-b", task_type="general", last_updated=old_ts),
        }
        pruned = prune_stale_arms(arms, known_model_ids=set(), days=30)

        assert pruned == 2
        assert len(arms) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Basis-aware training filter (Task 2.2)
# ═══════════════════════════════════════════════════════════════════════════


class TestThompsonLLMJudgmentFilter:
    """Basis-aware filter in TrainingDataCollector must reject LLM-judgment signals
    that target a tool-evidence training stream.

    This protects the training pipeline from poisoning: LLM quality scores are
    signals about model behavior, not ground-truth tool results. Mixing them
    into a tool-evidence stream would silently degrade the stream's reliability.

    Uses sync=True mode so record() calls _append() inline, letting us spy on
    _append to detect whether the record reached the write path.

    record() positional args: task, prompt, response, score, model_id.
    latency_ms and tokens_used must be non-zero to pass the early data-quality
    guards that execute before the basis-aware filter.
    """

    _RECORD_KWARGS: dict = {
        "task": "test task",
        "prompt": "do stuff",
        "response": "def foo(): pass",
        "score": 0.8,
        "model_id": "test-model",
        "task_type": "coding",
        "latency_ms": 100,
        "tokens_used": 50,
    }

    def _make_collector(self, tmp_path):
        """Create a sync TrainingDataCollector writing to tmp_path."""
        from vetinari.learning.training_collector import TrainingDataCollector

        return TrainingDataCollector(
            output_path=str(tmp_path / "training.jsonl"),
            sync=True,
        )

    def test_llm_judgment_rejected_from_tool_evidence_stream(self, tmp_path) -> None:
        """record() must reject when evidence_basis=llm_judgment and training_stream=tool_evidence."""
        from unittest.mock import patch

        collector = self._make_collector(tmp_path)
        write_calls = []

        with patch.object(collector, "_append", side_effect=write_calls.append):
            collector.record(
                **self._RECORD_KWARGS,
                metadata={
                    "evidence_basis": "llm_judgment",
                    "training_stream": "tool_evidence",
                },
            )

        assert len(write_calls) == 0, (
            "TrainingDataCollector.record() must reject a record when evidence_basis="
            "'llm_judgment' and training_stream='tool_evidence'. "
            f"Got {len(write_calls)} write(s) — the basis-aware filter is not working."
        )

    def test_llm_judgment_allowed_in_llm_stream(self, tmp_path) -> None:
        """record() must NOT reject when evidence_basis=llm_judgment and stream is not tool_evidence."""
        from unittest.mock import patch

        collector = self._make_collector(tmp_path)
        write_calls = []

        with patch.object(collector, "_append", side_effect=write_calls.append):
            collector.record(
                **self._RECORD_KWARGS,
                metadata={
                    "evidence_basis": "llm_judgment",
                    "training_stream": "llm_judgment",  # correct stream for this basis
                },
            )

        assert len(write_calls) == 1, (
            "TrainingDataCollector.record() must NOT reject a record when the evidence_basis "
            "matches the training_stream. LLM-judgment data is valid in the llm_judgment stream."
        )

    def test_tool_evidence_not_filtered_even_if_stream_specified(self, tmp_path) -> None:
        """Tool-evidence records must never be rejected by the basis-aware filter."""
        from unittest.mock import patch

        collector = self._make_collector(tmp_path)
        write_calls = []

        with patch.object(collector, "_append", side_effect=write_calls.append):
            collector.record(
                **self._RECORD_KWARGS,
                metadata={
                    "evidence_basis": "tool_evidence",
                    "training_stream": "tool_evidence",
                },
            )

        assert len(write_calls) == 1, (
            "TrainingDataCollector.record() must NOT reject tool-evidence records. "
            "The basis-aware filter should only block llm_judgment→tool_evidence mismatch."
        )

    def test_typed_params_reject_without_metadata(self, tmp_path) -> None:
        """Typed evidence_basis + target_stream params must trigger rejection with no metadata.

        This proves the filter is NOT metadata-string-dependent: a caller that
        passes the typed params directly (without populating metadata keys) must
        still be rejected when the combination is LLM_JUDGMENT + tool_evidence.
        """
        from unittest.mock import patch

        from vetinari.types import EvidenceBasis

        collector = self._make_collector(tmp_path)
        write_calls = []

        with patch.object(collector, "_append", side_effect=write_calls.append):
            collector.record(
                **self._RECORD_KWARGS,
                metadata=None,  # no metadata — filter must rely on typed params only
                evidence_basis=EvidenceBasis.LLM_JUDGMENT,
                target_stream="tool_evidence",
            )

        assert len(write_calls) == 0, (
            "TrainingDataCollector.record() must reject when evidence_basis=EvidenceBasis.LLM_JUDGMENT "
            "and target_stream='tool_evidence' even when no metadata dict is supplied. "
            f"Got {len(write_calls)} write(s) — typed-param filter path is not working."
        )
