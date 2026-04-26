"""Path canonicalization tests — verify all modules use get_user_dir() lazily.

Each test patches ``get_user_dir`` in the target module to return a
``tmp_path``-based directory, then exercises the I/O code path and asserts
the file was created under that directory.  This proves the path is
resolved at call time (not frozen at import/class-definition time).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# vetinari.training.agent_trainer — record_training / _load_history (methods)
# ---------------------------------------------------------------------------


class TestAgentTrainerPath:
    def test_record_training_uses_get_user_dir(self, tmp_path: Path) -> None:
        """AgentTrainer.record_training writes the JSONL log under get_user_dir()."""
        from vetinari.training.agent_trainer import AgentTrainer

        with patch("vetinari.training.agent_trainer.get_user_dir", return_value=tmp_path):
            trainer = AgentTrainer()
            trainer.record_training(
                agent_type="WORKER",
                model_path="/models/worker-v1.gguf",
                metrics={"eval_score": 0.85},
            )

        log_file = tmp_path / "agent_training_history.jsonl"
        assert log_file.exists(), "training history JSONL not written to get_user_dir() path"
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["agent_type"] == "WORKER"
        assert entry["model_path"] == "/models/worker-v1.gguf"

    def test_load_history_uses_get_user_dir(self, tmp_path: Path) -> None:
        """AgentTrainer._load_history reads from the directory get_user_dir() returns."""
        from vetinari.training.agent_trainer import AgentTrainer

        record = json.dumps({
            "agent_type": "FOREMAN",
            "model_path": "/models/foreman.gguf",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "metrics": {},
        })
        log_file = tmp_path / "agent_training_history.jsonl"
        log_file.write_text(record + "\n", encoding="utf-8")

        with patch("vetinari.training.agent_trainer.get_user_dir", return_value=tmp_path):
            trainer = AgentTrainer()
            history = trainer._load_history()

        assert len(history) == 1
        assert history[0]["agent_type"] == "FOREMAN"

    def test_missing_history_returns_empty(self, tmp_path: Path) -> None:
        """AgentTrainer._load_history returns [] when no file exists at get_user_dir() path."""
        from vetinari.training.agent_trainer import AgentTrainer

        with patch("vetinari.training.agent_trainer.get_user_dir", return_value=tmp_path):
            trainer = AgentTrainer()
            history = trainer._load_history()

        assert history == []


# ---------------------------------------------------------------------------
# vetinari.learning.auto_tuner — _load_config / _persist_config
# ---------------------------------------------------------------------------


class TestAutoTunerPath:
    def test_persist_config_uses_get_user_dir(self, tmp_path: Path) -> None:
        """AutoTuner._persist_config writes to the directory get_user_dir() returns."""
        from vetinari.learning.auto_tuner import AutoTuner

        with patch("vetinari.learning.auto_tuner.get_user_dir", return_value=tmp_path):
            tuner = AutoTuner()
            # Persist is triggered by _load_config if the file does not exist
            # Force a write by calling _persist_config directly
            tuner._persist_config()

        cfg_file = tmp_path / "auto_tuner_config.json"
        assert cfg_file.exists(), "auto_tuner_config.json not written to get_user_dir() path"

    def test_load_config_reads_from_get_user_dir(self, tmp_path: Path) -> None:
        """AutoTuner._load_config reads from the directory get_user_dir() returns."""
        from vetinari.learning.auto_tuner import AutoTuner

        # Write a config with a distinctive value for a known key
        cfg = {"max_concurrent": 99, "anomaly_threshold": 0.777}
        cfg_file = tmp_path / "auto_tuner_config.json"
        cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

        with patch("vetinari.learning.auto_tuner.get_user_dir", return_value=tmp_path):
            tuner = AutoTuner()

        # max_concurrent should have been loaded from our fixture file
        assert tuner._current_config["max_concurrent"] == 99
        assert tuner._current_config["anomaly_threshold"] == pytest.approx(0.777)


# ---------------------------------------------------------------------------
# vetinari.learning.shadow_testing — _load_state / _save_state
# ---------------------------------------------------------------------------


class TestShadowTestingPath:
    def test_save_state_uses_get_user_dir(self, tmp_path: Path) -> None:
        """ShadowTestRunner._save_state writes shadow_tests.json under get_user_dir()."""
        from vetinari.learning.shadow_testing import ShadowTestRunner

        with patch("vetinari.learning.shadow_testing.get_user_dir", return_value=tmp_path):
            runner = ShadowTestRunner()
            runner.create_test("path-test", {}, {})

        state_file = tmp_path / "shadow_tests.json"
        assert state_file.exists(), "shadow_tests.json not written to get_user_dir() path"

    def test_load_state_reads_from_get_user_dir(self, tmp_path: Path) -> None:
        """ShadowTestRunner._load_state reads from the directory get_user_dir() returns."""
        from vetinari.learning.shadow_testing import ShadowTestRunner

        # Pre-populate the state file
        state_data = {
            "tests": [
                {
                    "test_id": "shadow_abc123",
                    "description": "pre-populated",
                    "production_config": {},
                    "candidate_config": {},
                    "status": "running",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "promoted_at": None,
                    "min_samples": 10,
                    "production_metrics": {
                        "quality_scores": [], "latency_ms_values": [],
                        "error_count": 0, "total_runs": 0,
                    },
                    "candidate_metrics": {
                        "quality_scores": [], "latency_ms_values": [],
                        "error_count": 0, "total_runs": 0,
                    },
                }
            ]
        }
        (tmp_path / "shadow_tests.json").write_text(
            json.dumps(state_data), encoding="utf-8"
        )

        with patch("vetinari.learning.shadow_testing.get_user_dir", return_value=tmp_path):
            runner = ShadowTestRunner()

        active = runner.get_active_tests()
        assert any(t["test_id"] == "shadow_abc123" for t in active)


# ---------------------------------------------------------------------------
# vetinari.learning.benchmark_seeder — _load_cache / _save_cache
# ---------------------------------------------------------------------------


class TestBenchmarkSeederPath:
    def test_save_cache_uses_get_user_dir(self, tmp_path: Path) -> None:
        """BenchmarkSeeder._save_cache writes benchmark_cache.json under get_user_dir()."""
        from vetinari.learning.benchmark_seeder import BenchmarkSeeder

        with patch("vetinari.learning.benchmark_seeder.get_user_dir", return_value=tmp_path):
            seeder = BenchmarkSeeder()
            seeder.get_prior("test-model-7b", "coding")  # triggers _save_cache

        cache_file = tmp_path / "benchmark_cache.json"
        assert cache_file.exists(), "benchmark_cache.json not written to get_user_dir() path"

    def test_load_cache_reads_from_get_user_dir(self, tmp_path: Path) -> None:
        """BenchmarkSeeder._load_cache reads from the directory get_user_dir() returns."""
        from vetinari.learning.benchmark_seeder import BenchmarkSeeder

        # Pre-populate a cache with a known prior
        cache_data = {
            "last_updated": 1700000000.0,
            "priors": {
                "my-model:coding": {
                    "model_id": "my-model",
                    "task_type": "coding",
                    "alpha": 7.5,
                    "beta": 2.5,
                    "source": "capability",
                    "confidence": 0.6,
                }
            },
        }
        (tmp_path / "benchmark_cache.json").write_text(
            json.dumps(cache_data), encoding="utf-8"
        )

        with patch("vetinari.learning.benchmark_seeder.get_user_dir", return_value=tmp_path):
            seeder = BenchmarkSeeder()

        alpha, beta = seeder.get_prior("my-model", "coding")
        assert alpha == pytest.approx(7.5)
        assert beta == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# vetinari.drift.contract_registry — snapshot / load_snapshot
# ---------------------------------------------------------------------------


class TestContractRegistryPath:
    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        from vetinari.drift.contract_registry import reset_contract_registry
        reset_contract_registry()
        yield
        reset_contract_registry()

    def test_snapshot_uses_get_user_dir(self, tmp_path: Path) -> None:
        """ContractRegistry.snapshot() writes under the directory get_user_dir() returns."""
        from vetinari.drift.contract_registry import get_contract_registry

        with patch("vetinari.drift.contract_registry.get_user_dir", return_value=tmp_path):
            reg = get_contract_registry()
            reg.register("TestContract", {"field": "value"})
            reg.snapshot()

        snap_file = tmp_path / "drift_snapshots" / "contracts.json"
        assert snap_file.exists(), "snapshot not written to get_user_dir() path"
        payload = json.loads(snap_file.read_text(encoding="utf-8"))
        assert "TestContract" in payload["hashes"]

    def test_load_snapshot_reads_from_get_user_dir(self, tmp_path: Path) -> None:
        """ContractRegistry.load_snapshot() reads from the directory get_user_dir() returns."""
        from vetinari.drift.contract_registry import get_contract_registry

        snap_dir = tmp_path / "drift_snapshots"
        snap_dir.mkdir(parents=True)
        snap_data = {"timestamp": 1700000000.0, "hashes": {"PriorContract": "abc123"}}
        (snap_dir / "contracts.json").write_text(json.dumps(snap_data), encoding="utf-8")

        with patch("vetinari.drift.contract_registry.get_user_dir", return_value=tmp_path):
            reg = get_contract_registry()
            loaded = reg.load_snapshot()

        assert loaded is True
        # PriorContract should now be in _previous
        assert reg.get_stats()["snapshotted"] == 1


# ---------------------------------------------------------------------------
# vetinari.database — _get_db_path respects VETINARI_DB_PATH and get_user_dir
# ---------------------------------------------------------------------------


class TestDatabasePath:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path, monkeypatch):
        from vetinari.database import reset_for_testing
        monkeypatch.delenv("VETINARI_DB_PATH", raising=False)
        reset_for_testing()
        yield
        reset_for_testing()

    def test_db_path_env_var_takes_precedence(self, tmp_path: Path, monkeypatch) -> None:
        """VETINARI_DB_PATH env var overrides the get_user_dir() default."""
        from vetinari.database import _get_db_path

        db_override = tmp_path / "custom.db"
        monkeypatch.setenv("VETINARI_DB_PATH", str(db_override))
        assert _get_db_path() == db_override

    def test_db_path_falls_back_to_get_user_dir(self, tmp_path: Path) -> None:
        """Without VETINARI_DB_PATH, _get_db_path() uses get_user_dir()."""
        from vetinari.database import _get_db_path

        with patch("vetinari.database.get_user_dir", return_value=tmp_path):
            path = _get_db_path()

        assert path == tmp_path / "vetinari.db"


# ---------------------------------------------------------------------------
# vetinari.training.external_data — ExternalDataManager cache_dir default
# ---------------------------------------------------------------------------


class TestExternalDataPath:
    def test_default_cache_dir_uses_get_user_dir(self, tmp_path: Path) -> None:
        """ExternalDataManager defaults cache_dir to get_user_dir() / 'training_data'."""
        from vetinari.training.external_data import ExternalDataManager

        with patch("vetinari.training.external_data.get_user_dir", return_value=tmp_path):
            mgr = ExternalDataManager()

        assert mgr.cache_dir == tmp_path / "training_data"

    def test_explicit_cache_dir_overrides_default(self, tmp_path: Path) -> None:
        """An explicit cache_dir bypasses get_user_dir()."""
        from vetinari.training.external_data import ExternalDataManager

        custom = tmp_path / "custom_cache"
        mgr = ExternalDataManager(cache_dir=custom)
        assert mgr.cache_dir == custom


# ---------------------------------------------------------------------------
# vetinari.kaizen.pdca — PDCAController overrides_path default
# ---------------------------------------------------------------------------


class TestPDCAPath:
    @pytest.fixture
    def improvement_log(self, tmp_path: Path):
        """Return a real ImprovementLog backed by tmp_path."""
        from vetinari.kaizen.improvement_log import ImprovementLog
        db_path = tmp_path / "kaizen.db"
        return ImprovementLog(db_path=db_path)

    def test_default_overrides_path_uses_get_user_dir(
        self, tmp_path: Path, improvement_log
    ) -> None:
        """PDCAController default overrides_path resolves under get_user_dir()."""
        from vetinari.kaizen.pdca import PDCAController

        with patch("vetinari.kaizen.pdca.get_user_dir", return_value=tmp_path):
            ctrl = PDCAController(improvement_log=improvement_log)

        assert ctrl._overrides_path == tmp_path / "kaizen_overrides.json"

    def test_explicit_overrides_path_bypasses_get_user_dir(
        self, tmp_path: Path, improvement_log
    ) -> None:
        """An explicit overrides_path bypasses get_user_dir()."""
        from vetinari.kaizen.pdca import PDCAController

        custom = tmp_path / "my_overrides.json"
        ctrl = PDCAController(improvement_log=improvement_log, overrides_path=custom)
        assert ctrl._overrides_path == custom
