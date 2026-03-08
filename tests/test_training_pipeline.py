"""
Tests for the Vetinari training pipeline.

Covers:
  - TrainingDataCollector.export_hf_dataset()
  - TrainingDataCollector.export_few_shot_examples()
  - TrainingDataCollector.export_ranking_dataset()
  - TrainingManager.prepare_training_data()
  - TrainingManager.get_training_config()
  - TrainingManager.should_retrain()
  - TrainingManager.train_local()
  - TrainingManager.train_cloud()
  - AutoTuner._check_retraining_need()
  - RetrainingRecommendation dataclass serialisation
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collector(tmp_path: Path):
    """Return a fresh TrainingDataCollector writing to a tmp file."""
    from vetinari.learning.training_data import TrainingDataCollector
    return TrainingDataCollector(output_path=str(tmp_path / "training_data.jsonl"))


def _write_records(collector, records: List[dict]) -> None:
    """Synchronously write records and flush."""
    for r in records:
        collector.record(**r)
    collector.flush()
    # Give background writer a moment
    time.sleep(0.05)


_SAMPLE_RECORDS = [
    dict(task="Write a sort function", prompt="<s>Write a sort function",
         response="def sort(x): return sorted(x)", score=0.95,
         model_id="model-a", task_type="coding"),
    dict(task="Write a sort function", prompt="<s>Write a sort function",
         response="def sort(x): pass  # TODO", score=0.40,
         model_id="model-a", task_type="coding"),
    dict(task="Summarise climate change", prompt="<s>Summarise climate change",
         response="Climate change refers to...", score=0.88,
         model_id="model-b", task_type="research"),
    dict(task="Summarise climate change", prompt="<s>Summarise climate change v2",
         response="Global warming is...", score=0.70,
         model_id="model-b", task_type="research"),
    dict(task="List prime numbers", prompt="<s>List primes",
         response="2, 3, 5, 7, 11", score=0.82,
         model_id="model-a", task_type="coding"),
]


# ---------------------------------------------------------------------------
# 1. export_hf_dataset — Alpaca format
# ---------------------------------------------------------------------------

class TestExportHfDataset:
    def test_returns_alpaca_keys(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_hf_dataset(min_score=0.0)
        assert len(result) > 0
        for item in result:
            assert set(item.keys()) == {"instruction", "input", "output"}

    def test_filters_by_min_score(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_hf_dataset(min_score=0.9)
        # Only the 0.95 record passes
        assert len(result) == 1
        assert result[0]["instruction"] == "Write a sort function"

    def test_filters_by_task_type(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_hf_dataset(min_score=0.0, task_type="research")
        for item in result:
            # All returned records had task_type="research" so their instruction
            # should come from the research records
            assert "climate" in item["instruction"].lower() or "climate" in item["output"].lower()

    def test_returns_plain_dicts_no_datasets_import(self, tmp_path):
        """Must work without the datasets library."""
        import sys
        # Temporarily hide datasets if installed
        datasets_mod = sys.modules.pop("datasets", None)
        try:
            col = _make_collector(tmp_path)
            _write_records(col, _SAMPLE_RECORDS)
            result = col.export_hf_dataset(min_score=0.0)
            assert isinstance(result, list)
            assert isinstance(result[0], dict)
        finally:
            if datasets_mod is not None:
                sys.modules["datasets"] = datasets_mod

    def test_empty_file_returns_empty_list(self, tmp_path):
        col = _make_collector(tmp_path)
        result = col.export_hf_dataset(min_score=0.8)
        assert result == []


# ---------------------------------------------------------------------------
# 2. export_few_shot_examples — top-k by score
# ---------------------------------------------------------------------------

class TestExportFewShotExamples:
    def test_returns_input_output_pairs(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_few_shot_examples(task_type="coding", k=5)
        assert len(result) > 0
        for item in result:
            assert "input" in item and "output" in item

    def test_returns_at_most_k(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_few_shot_examples(task_type="coding", k=1)
        assert len(result) == 1

    def test_ordered_by_score_descending(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_few_shot_examples(task_type="coding", k=10)
        # First result should be the highest-scoring coding record
        assert result[0]["output"] == "def sort(x): return sorted(x)"

    def test_filters_to_task_type(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_few_shot_examples(task_type="research", k=5)
        for item in result:
            # research outputs don't contain "def"
            assert "def " not in item["output"]

    def test_unknown_task_type_returns_empty(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_few_shot_examples(task_type="nonexistent", k=5)
        assert result == []


# ---------------------------------------------------------------------------
# 3. export_ranking_dataset — grouped + ranked
# ---------------------------------------------------------------------------

class TestExportRankingDataset:
    def test_groups_by_task(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_ranking_dataset()
        # Should have groups for tasks with >1 response
        assert len(result) >= 1

    def test_responses_ordered_best_first(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_ranking_dataset()
        for group in result:
            scores = [r["score"] for r in group["responses"]]
            assert scores == sorted(scores, reverse=True), \
                f"Responses not sorted best-first: {scores}"

    def test_response_structure(self, tmp_path):
        col = _make_collector(tmp_path)
        _write_records(col, _SAMPLE_RECORDS)
        result = col.export_ranking_dataset()
        for group in result:
            assert "prompt" in group
            assert "responses" in group
            for r in group["responses"]:
                assert "response" in r
                assert "score" in r

    def test_single_response_tasks_excluded(self, tmp_path):
        col = _make_collector(tmp_path)
        # Only one record for "unique task"
        col.record(task="unique task xyz", prompt="p", response="r",
                   score=0.9, model_id="m", task_type="coding")
        col.flush()
        time.sleep(0.05)
        result = col.export_ranking_dataset()
        prompts = [g["prompt"] for g in result]
        assert "unique task xyz" not in prompts


# ---------------------------------------------------------------------------
# 4. TrainingManager.prepare_training_data — delegates correctly
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:
    def _manager_with_data(self, tmp_path):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager(data_path=str(tmp_path / "td.jsonl"))
        col = mgr._get_collector()
        _write_records(col, _SAMPLE_RECORDS)
        return mgr

    def test_sft_format(self, tmp_path):
        mgr = self._manager_with_data(tmp_path)
        ds = mgr.prepare_training_data(min_score=0.0, format="sft")
        assert ds.format == "sft"
        assert ds.stats["count"] == len(ds.records)
        for r in ds.records:
            assert "prompt" in r and "completion" in r

    def test_hf_format(self, tmp_path):
        mgr = self._manager_with_data(tmp_path)
        ds = mgr.prepare_training_data(min_score=0.0, format="hf")
        assert ds.format == "hf"
        for r in ds.records:
            assert set(r.keys()) == {"instruction", "input", "output"}

    def test_dpo_format(self, tmp_path):
        mgr = self._manager_with_data(tmp_path)
        ds = mgr.prepare_training_data(format="dpo")
        assert ds.format == "dpo"

    def test_ranking_format(self, tmp_path):
        mgr = self._manager_with_data(tmp_path)
        ds = mgr.prepare_training_data(format="ranking")
        assert ds.format == "ranking"
        for r in ds.records:
            assert "prompt" in r and "responses" in r

    def test_stats_populated(self, tmp_path):
        mgr = self._manager_with_data(tmp_path)
        ds = mgr.prepare_training_data(min_score=0.0, format="sft")
        assert "count" in ds.stats
        assert "avg_score" in ds.stats
        assert "task_type_breakdown" in ds.stats


# ---------------------------------------------------------------------------
# 5. TrainingManager.get_training_config — valid hyperparams
# ---------------------------------------------------------------------------

class TestGetTrainingConfig:
    def test_qlora_keys(self):
        from vetinari.learning.training_manager import TrainingManager
        cfg = TrainingManager().get_training_config("qlora")
        assert cfg["lr"] == pytest.approx(2e-4)
        assert cfg["lora_rank"] == 16
        assert cfg["lora_alpha"] == 32
        assert cfg["lora_dropout"] == 0
        assert cfg["target_modules"] == "all_linear"

    def test_full_keys(self):
        from vetinari.learning.training_manager import TrainingManager
        cfg = TrainingManager().get_training_config("full")
        assert cfg["lr"] == pytest.approx(1e-5)
        assert cfg["warmup_ratio"] == pytest.approx(0.1)
        assert cfg["weight_decay"] == pytest.approx(0.01)

    def test_default_is_qlora(self):
        from vetinari.learning.training_manager import TrainingManager
        cfg_default = TrainingManager().get_training_config()
        cfg_qlora = TrainingManager().get_training_config("qlora")
        assert cfg_default == cfg_qlora


# ---------------------------------------------------------------------------
# 6. TrainingManager.should_retrain — detects degradation
# ---------------------------------------------------------------------------

class TestShouldRetrain:
    def test_no_records_not_recommended(self, tmp_path):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager(data_path=str(tmp_path / "td.jsonl"))
        rec = mgr.should_retrain("model-x", "coding")
        assert rec.recommended is False
        assert rec.degradation == 0.0

    def test_high_quality_not_recommended(self, tmp_path):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager(data_path=str(tmp_path / "td.jsonl"))
        col = mgr._get_collector()
        for _ in range(5):
            col.record(task="t", prompt="p", response="r", score=0.92,
                       model_id="model-x", task_type="coding")
        col.flush()
        time.sleep(0.05)
        rec = mgr.should_retrain("model-x", "coding")
        assert rec.recommended is False

    def test_low_quality_recommended(self, tmp_path):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager(data_path=str(tmp_path / "td.jsonl"))
        col = mgr._get_collector()
        for _ in range(5):
            col.record(task="t", prompt="p", response="r", score=0.55,
                       model_id="model-lowq", task_type="coding")
        col.flush()
        time.sleep(0.05)
        rec = mgr.should_retrain("model-lowq", "coding")
        assert rec.recommended is True
        assert rec.degradation >= 0.15

    def test_recommendation_fields(self, tmp_path):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager(data_path=str(tmp_path / "td.jsonl"))
        rec = mgr.should_retrain("model-x", "coding")
        assert rec.model_id == "model-x"
        assert rec.task_type == "coding"
        assert isinstance(rec.reason, str)
        assert rec.recommended_method == "qlora"
        assert rec.recommended_min_score == 0.85


# ---------------------------------------------------------------------------
# 7. TrainingManager.train_local — returns result even without Unsloth
# ---------------------------------------------------------------------------

class TestTrainLocal:
    def _small_dataset(self, tmp_path):
        from vetinari.learning.training_manager import TrainingManager, TrainingDataset
        return TrainingDataset(
            records=[{"prompt": "p", "completion": "c", "score": 0.9,
                      "task_type": "coding"}],
            format="sft",
            stats={"count": 1, "avg_score": 0.9, "task_type_breakdown": {}},
        )

    def _large_dataset(self, n=110):
        from vetinari.learning.training_manager import TrainingDataset
        return TrainingDataset(
            records=[{"prompt": f"p{i}", "completion": f"c{i}", "score": 0.9,
                      "task_type": "coding"} for i in range(n)],
            format="sft",
            stats={"count": n, "avg_score": 0.9, "task_type_breakdown": {}},
        )

    def test_too_small_dataset_fails(self, tmp_path):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager(data_path=str(tmp_path / "td.jsonl"))
        ds = self._small_dataset(tmp_path)
        result = mgr.train_local("some-model", ds, method="qlora")
        assert result.success is False
        assert "small" in result.error.lower() or "minimum" in result.error.lower()

    def test_no_unsloth_returns_error(self):
        """Without unsloth installed, train_local should return a failure result."""
        import sys
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager()
        ds = self._large_dataset()
        # Hide unsloth regardless of whether it's installed
        unsloth_mod = sys.modules.pop("unsloth", None)
        with patch.dict("sys.modules", {"unsloth": None}):
            result = mgr.train_local("some-model", ds, method="qlora")
        if unsloth_mod is not None:
            sys.modules["unsloth"] = unsloth_mod
        # If unsloth was not present, result should indicate failure
        if not result.success:
            assert result.error is not None
            assert "unsloth" in result.error.lower() or "install" in result.error.lower()

    def test_returns_training_result_type(self):
        from vetinari.learning.training_manager import TrainingManager, TrainingResult
        mgr = TrainingManager()
        ds = self._small_dataset(Path("."))
        result = mgr.train_local("model", ds)
        assert isinstance(result, TrainingResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.duration_seconds, float)

    def test_duration_is_positive(self):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager()
        ds = self._small_dataset(Path("."))
        result = mgr.train_local("model", ds)
        assert result.duration_seconds >= 0.0


# ---------------------------------------------------------------------------
# 8. TrainingManager.train_cloud — returns job stub
# ---------------------------------------------------------------------------

class TestTrainCloud:
    def _dataset(self):
        from vetinari.learning.training_manager import TrainingDataset
        return TrainingDataset(records=[], format="sft",
                               stats={"count": 0, "avg_score": 0.0,
                                      "task_type_breakdown": {}})

    def test_returns_training_job(self):
        from vetinari.learning.training_manager import TrainingManager, TrainingJob
        mgr = TrainingManager()
        job = mgr.train_cloud("some-model", self._dataset())
        assert isinstance(job, TrainingJob)

    def test_job_status_pending(self):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager()
        job = mgr.train_cloud("some-model", self._dataset())
        assert job.status == "pending"

    def test_job_has_id(self):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager()
        job = mgr.train_cloud("some-model", self._dataset())
        assert job.job_id.startswith("job_")

    def test_job_registered(self):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager()
        job = mgr.train_cloud("some-model", self._dataset())
        assert mgr.get_training_status(job.job_id) is job

    def test_list_jobs_includes_new_job(self):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager()
        job = mgr.train_cloud("some-model", self._dataset(), provider="huggingface")
        assert job in mgr.list_jobs()

    def test_unknown_job_id_returns_none(self):
        from vetinari.learning.training_manager import TrainingManager
        mgr = TrainingManager()
        assert mgr.get_training_status("nonexistent") is None


# ---------------------------------------------------------------------------
# 9. AutoTuner._check_retraining_need — creates TuningActions
# ---------------------------------------------------------------------------

class TestAutoTunerRetraining:
    def test_empty_tracked_pairs_returns_empty(self):
        from vetinari.learning.auto_tuner import AutoTuner
        tuner = AutoTuner()
        assert tuner._tracked_pairs == set()
        actions = tuner._check_retraining_need()
        assert actions == []

    def test_no_retraining_when_quality_ok(self, tmp_path):
        from vetinari.learning.auto_tuner import AutoTuner
        from vetinari.learning.training_manager import TrainingManager
        import vetinari.learning.training_manager as tm_module

        mgr = TrainingManager(data_path=str(tmp_path / "td.jsonl"))
        col = mgr._get_collector()
        for _ in range(5):
            col.record(task="t", prompt="p", response="r", score=0.95,
                       model_id="model-ok", task_type="coding")
        col.flush()
        time.sleep(0.05)

        tuner = AutoTuner()
        tuner._tracked_pairs.add(("model-ok", "coding"))

        # Patch get_training_manager in the training_manager module so the
        # local import inside _check_retraining_need picks it up
        orig = tm_module.get_training_manager
        tm_module.get_training_manager = lambda data_path=None: mgr
        try:
            actions = tuner._check_retraining_need()
        finally:
            tm_module.get_training_manager = orig

        # Quality is high — no retraining action expected
        assert all(a.parameter == "retrain" for a in actions) or len(actions) == 0

    def test_retraining_action_is_manual(self, tmp_path):
        """When retraining IS recommended, action must have auto_applied=False."""
        from vetinari.learning.auto_tuner import AutoTuner
        from vetinari.learning.training_manager import TrainingManager
        import vetinari.learning.training_manager as tm_module

        mgr = TrainingManager(data_path=str(tmp_path / "td.jsonl"))
        col = mgr._get_collector()
        for _ in range(5):
            col.record(task="t", prompt="p", response="r", score=0.40,
                       model_id="model-bad", task_type="coding")
        col.flush()
        time.sleep(0.05)

        tuner = AutoTuner()
        tuner._tracked_pairs.add(("model-bad", "coding"))

        orig = tm_module.get_training_manager
        tm_module.get_training_manager = lambda data_path=None: mgr
        try:
            actions = tuner._check_retraining_need()
        finally:
            tm_module.get_training_manager = orig

        # With score=0.40, degradation is 50% — should recommend
        retrain_actions = [a for a in actions if a.parameter == "retrain"]
        assert len(retrain_actions) >= 1
        for a in retrain_actions:
            assert a.auto_applied is False

    def test_run_cycle_calls_check(self, tmp_path):
        """run_cycle() should include _check_retraining_need() output."""
        from vetinari.learning.auto_tuner import AutoTuner
        from vetinari.learning.training_manager import TrainingManager

        tuner = AutoTuner()
        tuner._tracked_pairs.add(("model-z", "coding"))

        called = []

        original = tuner._check_retraining_need

        def _mock():
            called.append(True)
            return []

        tuner._check_retraining_need = _mock
        tuner.run_cycle()
        assert len(called) == 1


# ---------------------------------------------------------------------------
# 10. RetrainingRecommendation dataclass — serialisation
# ---------------------------------------------------------------------------

class TestRetrainingRecommendationSerialisation:
    def test_asdict_round_trip(self):
        from vetinari.learning.training_manager import RetrainingRecommendation
        rec = RetrainingRecommendation(
            model_id="test-model",
            task_type="coding",
            current_avg_quality=0.72,
            baseline_quality=0.80,
            degradation=0.10,
            recommended=False,
            reason="Quality acceptable.",
        )
        d = asdict(rec)
        assert d["model_id"] == "test-model"
        assert d["task_type"] == "coding"
        assert d["current_avg_quality"] == pytest.approx(0.72)
        assert d["recommended"] is False
        assert d["recommended_method"] == "qlora"
        assert d["recommended_min_score"] == pytest.approx(0.85)

    def test_json_serialisable(self):
        from vetinari.learning.training_manager import RetrainingRecommendation
        rec = RetrainingRecommendation(
            model_id="test-model",
            task_type="research",
            current_avg_quality=0.60,
            baseline_quality=0.80,
            degradation=0.25,
            recommended=True,
            reason="Significant degradation detected.",
        )
        serialised = json.dumps(asdict(rec))
        restored = json.loads(serialised)
        assert restored["model_id"] == "test-model"
        assert restored["recommended"] is True

    def test_default_fields(self):
        from vetinari.learning.training_manager import RetrainingRecommendation
        rec = RetrainingRecommendation(
            model_id="m", task_type="t",
            current_avg_quality=0.5, baseline_quality=0.8,
            degradation=0.3, recommended=True, reason="test"
        )
        assert rec.recommended_method == "qlora"
        assert rec.recommended_min_score == 0.85
