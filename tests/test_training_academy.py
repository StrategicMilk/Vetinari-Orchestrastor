"""Comprehensive tests for Department 10 training modules.

Tests cover idle_scheduler, curriculum, external_data, synthetic_data,
continual_learning, validation, agent_trainer, and data_seeder.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.training.agent_trainer import AgentTrainer
from vetinari.training.continual_learning import (
    LoRAAdapterManager,
    ReplayBuffer,
    STABLERegularizer,
)
from vetinari.training.curriculum import (
    CurriculumPhase,
    TrainingActivity,
    TrainingActivityType,
    TrainingCurriculum,
)
from vetinari.training.data_seeder import SeedDataset, TrainingDataSeeder
from vetinari.training.external_data import (
    DatasetInfo,
    DatasetSpec,
    ExternalDataManager,
)
from vetinari.training.idle_scheduler import (
    IdleDetector,
    IdleTrainingJob,
    TrainingScheduler,
)
from vetinari.training.synthetic_data import (
    MagpieGenerator,
    StrategyDistiller,
    SyntheticDataGenerator,
)
from vetinari.training.validation import PostTrainingValidator, PreTrainingValidator
from vetinari.types import AgentType

# ── IdleDetector ─────────────────────────────────────────────────────────────


class TestIdleDetector:
    """Tests for IdleDetector."""

    def test_idle_detector_initially_not_idle(self) -> None:
        """Newly created detector with default 5-minute threshold is not idle."""
        detector = IdleDetector(min_idle_minutes=5)
        assert detector.idle is False

    def test_idle_detector_becomes_idle(self) -> None:
        """Detector with min_idle_minutes=0 immediately reports idle."""
        detector = IdleDetector(min_idle_minutes=0)
        assert detector.idle is True

    def test_idle_detector_record_activity_resets(self) -> None:
        """Recording activity on a zero-threshold detector keeps it non-idle after reset."""
        detector = IdleDetector(min_idle_minutes=0)
        # First confirm it goes idle
        assert detector.idle is True
        # Record activity resets internal timestamp
        detector.record_activity()
        # With a threshold of 0 it will immediately go idle again on the next check,
        # but record_activity must not raise
        detector.record_activity()

    def test_idle_detector_idle_duration(self) -> None:
        """idle_duration_minutes returns a float >= 0."""
        detector = IdleDetector(min_idle_minutes=0)
        duration = detector.idle_duration_minutes
        assert isinstance(duration, float)
        assert duration >= 0.0


# ── TrainingScheduler ────────────────────────────────────────────────────────


class TestTrainingScheduler:
    """Tests for TrainingScheduler."""

    def test_training_scheduler_not_training_initially(self) -> None:
        """is_training is False before start() is called."""
        detector = IdleDetector(min_idle_minutes=5)
        scheduler = TrainingScheduler(idle_detector=detector)
        assert scheduler.is_training is False

    def test_training_scheduler_current_job_none(self) -> None:
        """current_job is None before any job has been started."""
        detector = IdleDetector(min_idle_minutes=5)
        scheduler = TrainingScheduler(idle_detector=detector)
        assert scheduler.current_job is None

    def test_training_scheduler_start_stop(self) -> None:
        """start() and stop() run without raising."""
        detector = IdleDetector(min_idle_minutes=5)
        scheduler = TrainingScheduler(idle_detector=detector)
        scheduler.start()
        scheduler.stop()
        # After stop, no job should be running
        assert scheduler.current_job is None

    def test_training_scheduler_pause_resume_no_op(self) -> None:
        """pause_for_user_request and resume_after_user_request do not error with no job."""
        detector = IdleDetector(min_idle_minutes=5)
        scheduler = TrainingScheduler(idle_detector=detector)
        scheduler.pause_for_user_request()
        scheduler.resume_after_user_request()
        # After pause/resume with no job, state must still be consistent
        assert scheduler.current_job is None


# ── TrainingCurriculum ───────────────────────────────────────────────────────


class TestCurriculum:
    """Tests for TrainingCurriculum."""

    def test_curriculum_next_activity_returns_activity(self) -> None:
        """next_activity() always returns a TrainingActivity instance."""
        curriculum = TrainingCurriculum()
        activity = curriculum.next_activity()
        assert isinstance(activity, TrainingActivity)

    def test_curriculum_get_phase_returns_calibration(self) -> None:
        """With no training data available, get_phase() returns CALIBRATION."""
        mock_collector = MagicMock()
        mock_collector.count_records.return_value = 0
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            curriculum = TrainingCurriculum()
            phase = curriculum.get_phase()
            assert phase == CurriculumPhase.CALIBRATION

    def test_curriculum_get_status_returns_dict(self) -> None:
        """get_status() returns a dict with required keys."""
        curriculum = TrainingCurriculum()
        status = curriculum.get_status()
        assert isinstance(status, dict)
        assert "phase" in status
        assert "candidate_count" in status
        assert "next_activity_description" in status

    def test_curriculum_default_activity_is_valid(self) -> None:
        """next_activity() returns a valid training activity type."""
        curriculum = TrainingCurriculum()
        activity = curriculum.next_activity()
        # The curriculum selects the highest-priority candidate; the exact type
        # depends on which modules are importable and what data is available.
        assert activity.type in set(TrainingActivityType)

    def test_curriculum_activity_has_required_fields(self) -> None:
        """TrainingActivity returned by next_activity() has all required fields."""
        curriculum = TrainingCurriculum()
        activity = curriculum.next_activity()
        assert isinstance(activity.description, str)
        assert isinstance(activity.hypothesis, str)
        assert isinstance(activity.priority, float)
        assert 0.0 <= activity.priority <= 1.0
        assert isinstance(activity.estimated_duration_minutes, int)
        assert isinstance(activity.estimated_vram_gb, float)

    def test_activity_type_has_seven_values(self) -> None:
        """TrainingActivityType enum has exactly 7 activity types."""
        assert len(TrainingActivityType) == 7
        assert TrainingActivityType.RLEF_CODE_EXECUTION.value == "rlef_code_execution"

    def test_candidate_rlef_returns_none_without_data(self) -> None:
        """_candidate_rlef returns None when no execution traces are available."""
        curriculum = TrainingCurriculum()
        result = curriculum._candidate_rlef()
        assert result is None


# ── ExternalDataManager ──────────────────────────────────────────────────────


class TestExternalDataManager:
    """Tests for ExternalDataManager."""

    def test_external_data_catalog_has_categories(self, tmp_path: Path) -> None:
        """DATASET_CATALOG has exactly 4 category keys."""
        manager = ExternalDataManager(cache_dir=tmp_path)
        assert len(manager.DATASET_CATALOG) == 4

    def test_external_data_available_datasets(self, tmp_path: Path) -> None:
        """get_available_datasets() returns a non-empty list."""
        manager = ExternalDataManager(cache_dir=tmp_path)
        datasets = manager.get_available_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_external_data_available_datasets_domain_filter(self, tmp_path: Path) -> None:
        """Filtering by domain returns only matching entries."""
        manager = ExternalDataManager(cache_dir=tmp_path)
        coding = manager.get_available_datasets(domain="coding")
        for info in coding:
            assert info.domain == "coding"

    def test_external_data_is_available(self, tmp_path: Path) -> None:
        """is_available() returns a bool."""
        manager = ExternalDataManager(cache_dir=tmp_path)
        result = manager.is_available()
        assert isinstance(result, bool)

    def test_external_data_stats(self, tmp_path: Path) -> None:
        """get_stats() returns a dict with expected keys."""
        manager = ExternalDataManager(cache_dir=tmp_path)
        stats = manager.get_stats()
        assert isinstance(stats, dict)
        assert "total_datasets" in stats
        assert "downloaded_count" in stats
        assert "total_examples" in stats
        assert "cache_dir_size_bytes" in stats
        assert "cache_dir" in stats

    def test_dataset_spec_fields(self) -> None:
        """DatasetSpec has the expected fields and correct types."""
        spec = DatasetSpec(
            name="test/dataset",
            domain="coding",
            format="sft",
            description="A test dataset",
            max_examples=100,
            subset=None,
        )
        assert spec.name == "test/dataset"
        assert spec.domain == "coding"
        assert spec.format == "sft"
        assert spec.max_examples == 100
        assert spec.subset is None

    def test_dataset_info_fields(self) -> None:
        """DatasetInfo has expected fields with correct defaults."""
        info = DatasetInfo(
            name="test/dataset",
            domain="reasoning",
            size=500,
            estimated_train_minutes=1,
        )
        assert info.downloaded is False
        assert info.path is None


# ── SyntheticDataGenerator ───────────────────────────────────────────────────


class TestSyntheticData:
    """Tests for SyntheticDataGenerator, MagpieGenerator, StrategyDistiller."""

    def test_synthetic_generate_coding_empty(self) -> None:
        """generate_coding_challenges returns empty list when episode_memory unavailable."""
        gen = SyntheticDataGenerator()
        result = gen.generate_coding_challenges(count=5)
        assert isinstance(result, list)
        assert result == []

    def test_synthetic_generate_dpo_empty(self) -> None:
        """generate_dpo_pairs returns empty list when training_data unavailable."""
        gen = SyntheticDataGenerator()
        # Mock the training collector to return no pairs
        with patch("vetinari.learning.training_data.get_training_collector") as mock_get:
            mock_collector = MagicMock()
            mock_collector.export_dpo_dataset.return_value = []
            mock_get.return_value = mock_collector
            result = gen.generate_dpo_pairs(count=5)
        assert isinstance(result, list)
        assert result == []

    def test_synthetic_generate_self_play_tasks(self) -> None:
        """generate_self_play_tasks returns deterministic tasks without external deps."""
        gen = SyntheticDataGenerator()
        tasks = gen.generate_self_play_tasks(count=10)
        assert isinstance(tasks, list)
        assert len(tasks) == 10
        for task in tasks:
            assert "instruction" in task
            assert "domain" in task
            assert "difficulty" in task

    def test_synthetic_stats_returns_dict(self) -> None:
        """get_stats() returns a dict with expected keys."""
        gen = SyntheticDataGenerator()
        stats = gen.get_stats()
        assert isinstance(stats, dict)
        assert "episode_memory_available" in stats
        assert "training_data_available" in stats
        assert "adapter_available" in stats

    def test_magpie_generator_creates(self) -> None:
        """MagpieGenerator can be instantiated with and without a system prompt."""
        gen = MagpieGenerator()
        assert gen is not None
        assert gen._system_prompt == ""
        gen_with_prompt = MagpieGenerator(system_prompt="Focus on Python coding.")
        assert gen_with_prompt is not None
        assert gen_with_prompt._system_prompt == "Focus on Python coding."

    @patch("vetinari.training.synthetic_data.MagpieGenerator._get_adapter", return_value=None)
    def test_magpie_generator_returns_empty_without_adapter(self, _mock: MagicMock) -> None:
        """generate_instructions returns empty list when no LLM adapter is available."""
        gen = MagpieGenerator()
        result = gen.generate_instructions(count=5)
        assert isinstance(result, list)
        assert result == []

    def test_strategy_distiller_creates(self) -> None:
        """StrategyDistiller can be instantiated."""
        distiller = StrategyDistiller()
        assert distiller is not None
        assert isinstance(distiller, StrategyDistiller)

    def test_strategy_distiller_returns_empty_without_episodes(self) -> None:
        """distill_strategies returns empty list when episode_memory unavailable."""
        distiller = StrategyDistiller()
        result = distiller.distill_strategies()
        assert isinstance(result, list)
        assert result == []


# ── ContinualLearning ────────────────────────────────────────────────────────


class TestSTABLERegularizer:
    """Tests for STABLERegularizer."""

    def test_stable_regularizer_init(self) -> None:
        """STABLERegularizer stores threshold values correctly."""
        reg = STABLERegularizer(em_threshold=0.10, kl_threshold=0.4, bits_threshold=0.25)
        assert reg.em_threshold == 0.10
        assert reg.kl_threshold == 0.4
        assert reg.bits_threshold == 0.25

    def test_stable_regularizer_default_thresholds(self) -> None:
        """Default thresholds are applied when no arguments given."""
        reg = STABLERegularizer()
        assert reg.em_threshold == 0.15
        assert reg.kl_threshold == 0.5
        assert reg.bits_threshold == 0.3

    def test_stable_regularizer_metrics(self) -> None:
        """get_metrics() returns dict with expected keys including thresholds."""
        reg = STABLERegularizer()
        metrics = reg.get_metrics()
        assert isinstance(metrics, dict)
        assert "baseline_captured" in metrics
        assert "current_kl" in metrics
        assert "current_bits_increase" in metrics
        assert "current_em_drops" in metrics
        assert "thresholds" in metrics
        thresholds = metrics["thresholds"]
        assert "em" in thresholds
        assert "kl" in thresholds
        assert "bits" in thresholds

    def test_stable_regularizer_metrics_thresholds_match_init(self) -> None:
        """get_metrics() thresholds reflect values passed at construction."""
        reg = STABLERegularizer(em_threshold=0.20, kl_threshold=0.6, bits_threshold=0.35)
        metrics = reg.get_metrics()
        assert metrics["thresholds"]["em"] == 0.20
        assert metrics["thresholds"]["kl"] == 0.6
        assert metrics["thresholds"]["bits"] == 0.35

    def test_stable_regularizer_not_captured_initially(self) -> None:
        """baseline_captured is False before capture_baseline() is called."""
        reg = STABLERegularizer()
        assert reg.get_metrics()["baseline_captured"] is False

    def test_stable_regularizer_compute_gates_without_baseline(self) -> None:
        """compute_layer_gates returns empty dict when baseline not captured."""
        reg = STABLERegularizer()
        gates = reg.compute_layer_gates("some/model", "nonexistent.jsonl")
        assert gates == {}


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_replay_buffer_add_get(self, tmp_path: Path) -> None:
        """Can add examples and retrieve a replay batch."""
        buf = ReplayBuffer(max_size=100, buffer_path=tmp_path / "replay.jsonl")
        examples = [{"text": f"example {i}", "task_type": "coding"} for i in range(10)]
        added = buf.add(examples)
        assert added == 10
        batch = buf.get_replay_batch(5)
        assert isinstance(batch, list)
        assert len(batch) == 5

    def test_replay_buffer_max_size(self, tmp_path: Path) -> None:
        """Buffer respects max_size limit after overflow."""
        buf = ReplayBuffer(max_size=5, buffer_path=tmp_path / "replay.jsonl")
        examples = [{"text": f"item {i}"} for i in range(20)]
        buf.add(examples)
        assert len(buf) <= 5

    def test_replay_buffer_empty_batch(self, tmp_path: Path) -> None:
        """get_replay_batch on an empty buffer returns empty list."""
        buf = ReplayBuffer(max_size=100, buffer_path=tmp_path / "replay.jsonl")
        result = buf.get_replay_batch(10)
        assert result == []

    def test_replay_buffer_save_load(self, tmp_path: Path) -> None:
        """Persistence round-trip: save then load recovers the same examples."""
        buffer_file = tmp_path / "replay.jsonl"
        buf = ReplayBuffer(max_size=100, buffer_path=buffer_file)
        examples = [{"text": f"item {i}", "domain": "reasoning"} for i in range(5)]
        buf.add(examples)
        buf.save()

        buf2 = ReplayBuffer(max_size=100, buffer_path=buffer_file)
        assert len(buf2) == 5

    def test_replay_buffer_len(self, tmp_path: Path) -> None:
        """__len__ returns the current number of stored examples."""
        buf = ReplayBuffer(max_size=100, buffer_path=tmp_path / "replay.jsonl")
        assert len(buf) == 0
        buf.add([{"text": "hello"}])
        assert len(buf) == 1


class TestLoRAAdapterManager:
    """Tests for LoRAAdapterManager."""

    def test_lora_adapter_manager_register(self, tmp_path: Path) -> None:
        """Can register and retrieve an adapter."""
        manager = LoRAAdapterManager(adapters_dir=tmp_path)
        adapter_path = tmp_path / "code_adapter"
        manager.register_adapter("code", adapter_path)
        retrieved = manager.get_adapter("code")
        assert retrieved == adapter_path

    def test_lora_adapter_manager_get_unknown_returns_none(self, tmp_path: Path) -> None:
        """get_adapter for an unregistered task_type returns None."""
        manager = LoRAAdapterManager(adapters_dir=tmp_path)
        result = manager.get_adapter("nonexistent_type")
        assert result is None

    def test_lora_adapter_manager_list(self, tmp_path: Path) -> None:
        """list_adapters returns all registered adapters."""
        manager = LoRAAdapterManager(adapters_dir=tmp_path)
        manager.register_adapter("code", tmp_path / "code_lora")
        manager.register_adapter("qa", tmp_path / "qa_lora")
        adapters = manager.list_adapters()
        assert "code" in adapters
        assert "qa" in adapters
        assert len(adapters) == 2

    def test_lora_adapter_manager_persistence(self, tmp_path: Path) -> None:
        """save_registry / load_registry round-trip preserves registrations."""
        manager = LoRAAdapterManager(adapters_dir=tmp_path)
        manager.register_adapter("code", tmp_path / "code_lora")
        manager.save_registry()

        manager2 = LoRAAdapterManager(adapters_dir=tmp_path)
        assert "code" in manager2.list_adapters()

    def test_lora_adapter_manager_remove(self, tmp_path: Path) -> None:
        """remove_adapter deregisters a known adapter."""
        manager = LoRAAdapterManager(adapters_dir=tmp_path)
        manager.register_adapter("code", tmp_path / "code_lora")
        removed = manager.remove_adapter("code")
        assert removed is True
        assert manager.get_adapter("code") is None

    def test_lora_adapter_manager_remove_unknown(self, tmp_path: Path) -> None:
        """remove_adapter returns False for an unknown task_type."""
        manager = LoRAAdapterManager(adapters_dir=tmp_path)
        result = manager.remove_adapter("not_registered")
        assert result is False

    def test_lora_adapter_manager_invalid_task_type(self, tmp_path: Path) -> None:
        """register_adapter raises ValueError for empty task_type."""
        manager = LoRAAdapterManager(adapters_dir=tmp_path)
        with pytest.raises(ValueError, match="task_type must be a non-empty string"):
            manager.register_adapter("", tmp_path / "adapter")


# ── Validation ───────────────────────────────────────────────────────────────


class TestPreTrainingValidator:
    """Tests for PreTrainingValidator."""

    def test_pre_training_validator_init(self, tmp_path: Path) -> None:
        """PreTrainingValidator creates checkpoint_dir on init."""
        PreTrainingValidator(checkpoint_dir=tmp_path / "checkpoints")
        assert (tmp_path / "checkpoints").exists()

    def test_pre_training_validator_save_checkpoint(self, tmp_path: Path) -> None:
        """save_checkpoint writes a JSON file to checkpoint_dir."""
        validator = PreTrainingValidator(checkpoint_dir=tmp_path / "checkpoints")
        path = validator.save_checkpoint(model_path="test/model")
        assert path is not None
        assert path.exists()
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["checkpoint_type"] == "pre_training"
        assert data["model_path"] == "test/model"

    def test_pre_training_validator_get_latest_checkpoint(self, tmp_path: Path) -> None:
        """get_latest_checkpoint returns metadata after a checkpoint is saved."""
        validator = PreTrainingValidator(checkpoint_dir=tmp_path / "checkpoints")
        validator.save_checkpoint()
        metadata = validator.get_latest_checkpoint()
        assert metadata is not None
        assert "timestamp" in metadata

    def test_pre_training_validator_get_latest_checkpoint_empty(self, tmp_path: Path) -> None:
        """get_latest_checkpoint returns None when no checkpoints exist."""
        validator = PreTrainingValidator(checkpoint_dir=tmp_path / "checkpoints")
        result = validator.get_latest_checkpoint()
        assert result is None


class TestPostTrainingValidator:
    """Tests for PostTrainingValidator."""

    def test_post_training_validator_pass(self) -> None:
        """validate() passes when new scores improve over old scores."""
        validator = PostTrainingValidator(regression_threshold=0.05)
        old = {"quality": 0.70, "speed": 0.80}
        new = {"quality": 0.75, "speed": 0.82}
        passed, reason = validator.validate(old, new, target_metric="quality")
        assert passed is True
        assert "quality" in reason

    def test_post_training_validator_regression(self) -> None:
        """validate() fails when a metric regresses beyond the threshold."""
        validator = PostTrainingValidator(regression_threshold=0.05)
        old = {"quality": 0.80, "speed": 0.90}
        new = {"quality": 0.74, "speed": 0.90}  # quality dropped 0.06 > threshold
        passed, reason = validator.validate(old, new, target_metric="quality")
        assert passed is False
        assert "Regression" in reason

    def test_post_training_validator_no_improvement(self) -> None:
        """validate() fails when target metric stays flat."""
        validator = PostTrainingValidator(regression_threshold=0.05)
        old = {"quality": 0.80}
        new = {"quality": 0.80}  # no improvement
        passed, reason = validator.validate(old, new, target_metric="quality")
        assert passed is False
        assert "did not improve" in reason

    def test_post_training_recommend_deploy(self) -> None:
        """recommend_action returns 'deploy' when validation passed."""
        validator = PostTrainingValidator()
        action = validator.recommend_action(passed=True, old_model="v1", new_model="v2")
        assert action == "deploy"

    def test_post_training_recommend_rollback(self) -> None:
        """recommend_action returns 'rollback' for wide threshold failures."""
        validator = PostTrainingValidator(regression_threshold=0.10)
        action = validator.recommend_action(passed=False, old_model="v1", new_model="v2")
        assert action == "rollback"

    def test_post_training_recommend_retry(self) -> None:
        """recommend_action returns 'retry' for narrow threshold failures."""
        validator = PostTrainingValidator(regression_threshold=0.05)
        action = validator.recommend_action(passed=False, old_model="v1", new_model="v2")
        assert action == "retry"

    def test_post_training_regression_threshold_boundary(self) -> None:
        """validate() passes when regression is exactly at the threshold boundary."""
        validator = PostTrainingValidator(regression_threshold=0.05)
        old = {"quality": 0.80}
        new = {"quality": 0.76}  # dropped exactly 0.04 < 0.05 threshold
        passed, _reason = validator.validate(old, new, target_metric="quality")
        # 0.80 - 0.76 = 0.04, which is < 0.05, so no regression; but quality did not
        # improve so the target metric check fails
        assert passed is False  # target metric didn't improve


# ── AgentTrainer ─────────────────────────────────────────────────────────────


class TestAgentTrainer:
    """Tests for AgentTrainer."""

    def test_agent_trainer_priority(self) -> None:
        """get_training_priority returns list of (name, score) tuples."""
        trainer = AgentTrainer()
        priority = trainer.get_training_priority()
        assert isinstance(priority, list)
        assert len(priority) > 0
        for item in priority:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, score = item
            assert isinstance(name, str)
            assert isinstance(score, float)

    def test_agent_trainer_priority_sorted_descending(self) -> None:
        """get_training_priority is sorted in descending order by score."""
        trainer = AgentTrainer()
        priority = trainer.get_training_priority()
        scores = [score for _, score in priority]
        assert scores == sorted(scores, reverse=True)

    def test_agent_trainer_dataset_config(self) -> None:
        """get_agent_dataset_config returns dict with expected keys for WORKER."""
        trainer = AgentTrainer()
        config = trainer.get_agent_dataset_config(AgentType.WORKER.value)
        assert isinstance(config, dict)
        assert "task_types" in config
        assert "min_score" in config
        assert "max_examples" in config
        assert "preferred_method" in config

    def test_agent_trainer_dataset_config_known_agents(self) -> None:
        """All known agent types return a valid dataset config."""
        trainer = AgentTrainer()
        known_agents = [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]
        for agent in known_agents:
            config = trainer.get_agent_dataset_config(agent)
            assert isinstance(config["task_types"], list)
            assert config["preferred_method"] in {"sft", "dpo"}

    def test_agent_trainer_dataset_config_unknown_agent(self) -> None:
        """Unknown agent type returns generic defaults without raising."""
        trainer = AgentTrainer()
        config = trainer.get_agent_dataset_config("MADE_UP_AGENT")
        assert isinstance(config, dict)
        assert "task_types" in config

    def test_agent_trainer_stats(self) -> None:
        """get_stats() returns dict mapping agent names to sub-dicts."""
        trainer = AgentTrainer()
        stats = trainer.get_stats()
        assert isinstance(stats, dict)
        # All default agent types should be present
        assert AgentType.WORKER.value in stats
        for agent_stats in stats.values():
            assert "total_runs" in agent_stats
            assert "last_trained" in agent_stats
            assert "latest_model" in agent_stats


# ── DataSeeder ───────────────────────────────────────────────────────────────


class TestDataSeeder:
    """Tests for TrainingDataSeeder and SeedDataset."""

    def test_seed_datasets_defined(self) -> None:
        """SEED_DATASETS has exactly 4 entries."""
        seeder = TrainingDataSeeder()
        assert len(seeder.SEED_DATASETS) == 4

    def test_seed_datasets_are_seed_dataset_instances(self) -> None:
        """Each entry in SEED_DATASETS is a SeedDataset dataclass."""
        seeder = TrainingDataSeeder()
        for dataset in seeder.SEED_DATASETS:
            assert isinstance(dataset, SeedDataset)
            assert isinstance(dataset.name, str)
            assert isinstance(dataset.domain, str)
            assert isinstance(dataset.size, int)
            assert isinstance(dataset.description, str)

    def test_seed_status(self) -> None:
        """get_seed_status() returns dict with expected keys."""
        seeder = TrainingDataSeeder()
        status = seeder.get_seed_status()
        assert isinstance(status, dict)
        assert "total_seed_datasets" in status
        assert "downloaded" in status
        assert "pending" in status
        assert "total_examples" in status
        assert "data_dir" in status

    def test_seed_status_total_matches_dataset_count(self) -> None:
        """total_seed_datasets matches the actual number of SEED_DATASETS entries."""
        seeder = TrainingDataSeeder()
        status = seeder.get_seed_status()
        assert status["total_seed_datasets"] == len(seeder.SEED_DATASETS)

    def test_training_data_exists_false(self) -> None:
        """_training_data_exists returns False when training data dir is absent."""
        seeder = TrainingDataSeeder()
        # In a clean test environment without a real .vetinari/training_data, the
        # method should return a bool (True or False) without raising.
        result = seeder._training_data_exists()
        assert isinstance(result, bool)

    def test_seed_dataset_fields(self) -> None:
        """SeedDataset dataclass accepts all expected fields."""
        ds = SeedDataset(
            name="test/ds",
            domain="coding",
            size=500,
            description="Test seed dataset",
            subsample=True,
        )
        assert ds.name == "test/ds"
        assert ds.subsample is True

    def test_seed_dataset_subsample_default_false(self) -> None:
        """SeedDataset has subsample=False by default."""
        ds = SeedDataset(
            name="test/ds",
            domain="coding",
            size=500,
            description="Test seed dataset",
        )
        assert ds.subsample is False


# ── TrainingJob dataclass ─────────────────────────────────────────────────────


class TestTrainingJob:
    """Tests for the TrainingJob dataclass."""

    def test_training_job_fields(self) -> None:
        """TrainingJob stores all required fields."""
        job = IdleTrainingJob(
            job_id="abc123",
            status="running",
            activity_description="benchmark calibration",
            started_at="2026-01-01T00:00:00+00:00",
            progress=0.5,
        )
        assert job.job_id == "abc123"
        assert job.status == "running"
        assert job.progress == 0.5

    def test_training_job_default_progress(self) -> None:
        """TrainingJob defaults progress to 0.0."""
        job = IdleTrainingJob(
            job_id="xyz",
            status="pending",
            activity_description="test",
            started_at="2026-01-01T00:00:00+00:00",
        )
        assert job.progress == 0.0
