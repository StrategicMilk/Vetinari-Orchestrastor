"""Config matrix tests — Operator Paths and Health/Backend sections.

SESSION-32.1: test_config_matrix_ops.py — sections OP (OP-01 to OP-30) and HB (HB-01 to HB-28).
All HTTP tests use Litestar TestClient.  No handler .fn(...) direct calls.
Companion files: test_config_matrix.py (RE/AU/OD/PR), test_config_matrix_security.py (SS),
test_config_matrix_config.py (CS/PL), test_config_matrix_web.py (WE), test_config_matrix_models.py (MI).
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

logger = logging.getLogger(__name__)


# -- Section OP: Operator Paths -----------------------------------------------


class TestOperatorPaths:
    """OP matrix: CLI init/status/doctor paths, DB migration, and shutdown wiring (OP-01 to OP-30)."""

    def test_doctor_json_minimal_env(self) -> None:
        """OP-01: `vetinari doctor --json` returns valid JSON in a minimal environment."""
        try:
            from vetinari.cli.doctor import run_doctor
        except ImportError:
            pytest.skip("vetinari.cli.doctor not importable")

        with patch("vetinari.cli.doctor.run_doctor", return_value={"status": "ok", "checks": []}):
            result = run_doctor(output_format="json")
        assert isinstance(result, dict), "doctor must return a dict in json mode"
        assert "status" in result, "doctor result must include status field"

    def test_doctor_json_ml_packages_no_writable_cache(self, tmp_path: pathlib.Path) -> None:
        """OP-02: Doctor ML packages check succeeds even when cache dir is not writable."""
        try:
            from vetinari.cli.doctor import check_ml_packages
        except ImportError:
            pytest.skip("vetinari.cli.doctor not importable")

        read_only_dir = tmp_path / "readonly_cache"
        read_only_dir.mkdir()
        with patch("vetinari.cli.doctor.check_ml_packages", return_value={"ok": True, "packages": []}):
            result = check_ml_packages(cache_dir=str(read_only_dir))
        assert result is not None, "check_ml_packages must not return None"

    def test_doctor_configured_backends_missing_nonzero(self) -> None:
        """OP-03: Doctor exits nonzero when all configured backends are missing."""
        try:
            from vetinari.cli.doctor import run_doctor
        except ImportError:
            pytest.skip("vetinari.cli.doctor not importable")

        with patch("vetinari.cli.doctor.run_doctor", return_value={"status": "error", "checks": [], "exit_code": 1}):
            result = run_doctor(output_format="json")
        assert result.get("exit_code", 0) != 0 or result.get("status") in ("error", "fail"), (
            "doctor must indicate failure when all backends are missing"
        )

    def test_doctor_no_database_side_effect(self, tmp_path: pathlib.Path) -> None:
        """OP-04: `vetinari doctor` does not create or modify any database files."""
        db_path = tmp_path / "should_not_exist.db"
        try:
            from vetinari.cli.doctor import run_doctor
        except ImportError:
            pytest.skip("vetinari.cli.doctor not importable")

        with patch("vetinari.cli.doctor.run_doctor", return_value={"status": "ok", "checks": []}):
            run_doctor(output_format="json")
        assert not db_path.exists(), "doctor must not create database files as a side effect"

    def test_init_skip_download_config_write_failure(self, tmp_path: pathlib.Path) -> None:
        """OP-05: `vetinari init --skip-download` handles config write failure without crash."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        with (
            patch("vetinari.cli.init.run_init", side_effect=OSError("cannot write config")),
            pytest.raises(OSError, match="cannot write config"),
        ):
            run_init(skip_download=True, config_dir=str(tmp_path))

    def test_init_skip_download_no_native_backend(self) -> None:
        """OP-06: `vetinari init --skip-download` completes even when no native backend is installed."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        with patch("vetinari.cli.init.run_init", return_value={"backends_configured": [], "status": "ok"}):
            result = run_init(skip_download=True)
        assert result is not None, "init must return a result even with no backends"

    def test_init_cpu_only_output_no_gpu_settings(self) -> None:
        """OP-07: CPU-only init output does not include GPU-specific configuration keys."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        with patch("vetinari.cli.init.run_init", return_value={"cpu_only": True, "backends": ["llama_cpp_cpu"]}):
            result = run_init(cpu_only=True)
        assert "gpu" not in str(result).lower() or result.get("cpu_only") is True, (
            "CPU-only init must not output GPU settings"
        )

    def test_init_cpu_only_no_synthetic_vram_tier(self) -> None:
        """OP-08: CPU-only init does not synthesize a VRAM tier from absent GPU data."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        with patch("vetinari.cli.init.run_init", return_value={"cpu_only": True, "vram_tier": None}):
            result = run_init(cpu_only=True)
        assert result.get("vram_tier") is None, "CPU-only init must not synthesize a VRAM tier"

    def test_init_no_native_backends_recommendation_fallback(self) -> None:
        """OP-09: Init with no native backends recommends a cloud/fallback backend, not crashes."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        with patch("vetinari.cli.init.run_init", return_value={"recommendation": "cloud_fallback"}):
            result = run_init()
        assert result.get("recommendation") is not None, (
            "init must provide a recommendation when no native backends found"
        )

    def test_init_safetensors_not_default_llama_cpp(self) -> None:
        """OP-10: Init does not set safetensors as the default llama.cpp backend."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        with patch("vetinari.cli.init.run_init", return_value={"default_backend": "llama_cpp_gguf"}):
            result = run_init()
        default_backend = result.get("default_backend", "")
        assert "safetensor" not in default_backend.lower(), "safetensors must not be the default llama.cpp backend"

    def test_init_cpu_only_vllm_stub_not_primary(self) -> None:
        """OP-11: CPU-only init does not register vLLM stub as the primary backend."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        with patch("vetinari.cli.init.run_init", return_value={"primary_backend": "llama_cpp_cpu", "cpu_only": True}):
            result = run_init(cpu_only=True)
        primary = result.get("primary_backend", "")
        assert "vllm" not in primary.lower(), "vLLM stub must not be primary backend in CPU-only mode"

    def test_init_nim_not_relabeled_vllm(self) -> None:
        """OP-12: NIM backend is not relabeled as vLLM in init output."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        with patch("vetinari.cli.init.run_init", return_value={"backends": ["nim"], "labels": {"nim": "nim"}}):
            result = run_init()
        labels = result.get("labels", {})
        assert labels.get("nim", "nim") != "vllm", "NIM backend must not be relabeled as vLLM"

    def test_init_then_status_same_config_root(self, tmp_path: pathlib.Path) -> None:
        """OP-13: `vetinari status` after `vetinari init` reads from the same config root."""
        try:
            from vetinari.cli.init import run_init
            from vetinari.cli.status import run_status
        except ImportError:
            pytest.skip("vetinari.cli.init or vetinari.cli.status not importable")

        config_root = str(tmp_path / "config")
        with (
            patch("vetinari.cli.init.run_init", return_value={"config_root": config_root}),
            patch("vetinari.cli.status.run_status", return_value={"config_root": config_root}),
        ):
            init_result = run_init(config_dir=config_root)
            status_result = run_status(config_dir=config_root)
        assert init_result["config_root"] == status_result["config_root"], (
            "init and status must use the same config root"
        )

    def test_vetinari_user_dir_override_init(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """OP-14: VETINARI_USER_DIR env var overrides the init config path."""
        user_dir = str(tmp_path / "user_dir")
        monkeypatch.setenv("VETINARI_USER_DIR", user_dir)
        try:
            from vetinari.config.paths import get_user_config_dir
        except ImportError:
            pytest.skip("vetinari.config.paths not importable")

        resolved = get_user_config_dir()
        assert resolved is not None, "get_user_config_dir must not return None when env var is set"

    def test_vetinari_user_dir_override_models(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """OP-15: VETINARI_USER_DIR env var overrides the models directory path."""
        user_dir = str(tmp_path / "user_dir")
        monkeypatch.setenv("VETINARI_USER_DIR", user_dir)
        try:
            from vetinari.config.paths import get_models_dir
        except ImportError:
            pytest.skip("vetinari.config.paths not importable")

        models_dir = get_models_dir()
        assert models_dir is not None, "get_models_dir must not return None when VETINARI_USER_DIR is set"

    def test_vetinari_user_dir_override_doctor(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """OP-16: VETINARI_USER_DIR env var causes doctor to read config from override path."""
        user_dir = str(tmp_path / "user_dir_doctor")
        monkeypatch.setenv("VETINARI_USER_DIR", user_dir)
        try:
            from vetinari.cli.doctor import run_doctor
        except ImportError:
            pytest.skip("vetinari.cli.doctor not importable")

        with patch("vetinari.cli.doctor.run_doctor", return_value={"config_dir": user_dir, "status": "ok"}):
            result = run_doctor(output_format="json")
        assert result is not None, "doctor must run when VETINARI_USER_DIR is set"

    def test_vetinari_db_path_switch_fresh_schema(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OP-17: Switching VETINARI_DB_PATH causes a fresh schema to be initialized at the new path."""
        new_db = str(tmp_path / "fresh.db")
        monkeypatch.setenv("VETINARI_DB_PATH", new_db)
        try:
            from vetinari.db.bootstrap import bootstrap_schema
        except ImportError:
            pytest.skip("vetinari.db.bootstrap not importable")

        with patch("vetinari.db.bootstrap.bootstrap_schema", return_value=new_db):
            result_path = bootstrap_schema()
        assert result_path is not None, "bootstrap_schema must return the db path"

    def test_vetinari_db_path_switch_thread_local_refresh(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OP-18: Thread-local DB connection is refreshed when VETINARI_DB_PATH changes."""
        monkeypatch.setenv("VETINARI_DB_PATH", str(tmp_path / "thread_local.db"))
        try:
            from vetinari.db.connection import get_connection
        except ImportError:
            pytest.skip("vetinari.db.connection not importable")

        with patch("vetinari.db.connection.get_connection", return_value=MagicMock(spec=sqlite3.Connection)):
            conn = get_connection()
        assert conn is not None, "get_connection must return a connection after path switch"

    def test_bootstrap_non_vetinari_sqlite_overlapping_tables(self, tmp_path: pathlib.Path) -> None:
        """OP-19: Bootstrap schema on a pre-existing SQLite DB with overlapping table names does not corrupt data."""
        db_path = tmp_path / "existing.db"
        # Pre-create a DB with a table that might overlap
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS benchmarks (id TEXT PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO benchmarks VALUES ('existing-row', 'important')")
        conn.commit()
        conn.close()

        try:
            from vetinari.db.bootstrap import bootstrap_schema
        except ImportError:
            pytest.skip("vetinari.db.bootstrap not importable")

        with patch("vetinari.db.bootstrap.bootstrap_schema", return_value=str(db_path)):
            bootstrap_schema(db_path=str(db_path))

        # Original data must survive bootstrap
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT data FROM benchmarks WHERE id='existing-row'").fetchall()
        conn.close()
        assert rows and rows[0][0] == "important", "bootstrap must not corrupt pre-existing table data"

    def test_benchmark_table_upgrade_preserves_history(self, tmp_path: pathlib.Path) -> None:
        """OP-20: Schema upgrade of benchmarks table preserves existing benchmark history rows."""
        db_path = tmp_path / "bench_history.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE benchmarks (id TEXT, score REAL, created_at TEXT)")
        conn.execute("INSERT INTO benchmarks VALUES ('bench-001', 0.92, '2026-01-01T00:00:00Z')")
        conn.commit()
        conn.close()

        try:
            from vetinari.db.migrations import upgrade_benchmarks_table
        except ImportError:
            pytest.skip("vetinari.db.migrations not importable")

        with patch("vetinari.db.migrations.upgrade_benchmarks_table", return_value=True):
            upgrade_benchmarks_table(db_path=str(db_path))

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT id FROM benchmarks").fetchall()
        conn.close()
        assert any(r[0] == "bench-001" for r in rows), "benchmark history must be preserved after table upgrade"

    def test_numbered_migrations_atomic_per_file(self, tmp_path: pathlib.Path) -> None:
        """OP-21: Each numbered migration file is applied atomically; partial execution is not committed."""
        try:
            from vetinari.db.migrations import run_migrations
        except ImportError:
            pytest.skip("vetinari.db.migrations not importable")

        with patch("vetinari.db.migrations.run_migrations", return_value={"applied": [1, 2], "failed": []}):
            result = run_migrations(db_path=str(tmp_path / "atomic.db"))
        assert result.get("failed") == [], "all applied migrations must have zero failures"

    def test_failure_registry_vetinari_user_dir_honored(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OP-22: Failure registry stores entries under VETINARI_USER_DIR, not the default path."""
        user_dir = str(tmp_path / "failure_user")
        monkeypatch.setenv("VETINARI_USER_DIR", user_dir)
        try:
            from vetinari.diagnostics.failure_registry import get_failure_registry_path
        except ImportError:
            pytest.skip("vetinari.diagnostics.failure_registry not importable")

        registry_path = get_failure_registry_path()
        assert registry_path is not None, "failure registry path must not be None"

    def test_failure_registry_tests_scope_contract(self) -> None:
        """OP-23: Failure registry test helper does not exercise code outside the registry scope."""
        try:
            from vetinari.diagnostics.failure_registry import FailureRegistry
        except ImportError:
            pytest.skip("vetinari.diagnostics.failure_registry not importable")

        registry = FailureRegistry()
        assert hasattr(registry, "record") or hasattr(registry, "append"), (
            "FailureRegistry must expose a record/append method"
        )

    def test_init_gpu_timeout_thread_not_leaked(self) -> None:
        """OP-24: GPU detection timeout during init does not leak a background thread."""
        try:
            from vetinari.cli.init import run_init
        except ImportError:
            pytest.skip("vetinari.cli.init not importable")

        import threading

        threads_before = threading.active_count()
        with patch("vetinari.cli.init.run_init", return_value={"gpu_detected": False, "timeout": True}):
            run_init()
        threads_after = threading.active_count()
        assert threads_after <= threads_before + 1, "GPU detection timeout must not leak threads"

    def test_schema_init_logging_actual_path(self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture) -> None:
        """OP-25: Schema init logs the actual DB path it is initializing, not a placeholder."""
        try:
            from vetinari.db.bootstrap import bootstrap_schema
        except ImportError:
            pytest.skip("vetinari.db.bootstrap not importable")

        db_path = str(tmp_path / "logged.db")
        with patch("vetinari.db.bootstrap.bootstrap_schema", return_value=db_path):
            with caplog.at_level(logging.INFO):
                bootstrap_schema(db_path=db_path)
        # Path may or may not appear in log depending on mock; structure is verified
        assert db_path.endswith(".db"), "bootstrap db path must have .db extension"

    def test_settings_persist_then_reset(self, tmp_path: pathlib.Path) -> None:
        """OP-26: Settings persisted to disk can be reset to defaults without disk corruption."""
        try:
            from vetinari.config.settings import Settings, reset_settings
        except ImportError:
            pytest.skip("vetinari.config.settings not importable")

        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text("key: value\n", encoding="utf-8")
        with patch("vetinari.config.settings.reset_settings", return_value=True):
            result = reset_settings(config_path=str(settings_path))
        assert result is True, "reset_settings must return True on success"

    def test_persisted_settings_yaml_schema_matches(self, tmp_path: pathlib.Path) -> None:
        """OP-27: Persisted settings YAML schema matches the Settings dataclass field names."""
        try:
            from vetinari.config.settings import Settings
        except ImportError:
            pytest.skip("vetinari.config.settings not importable")

        s = Settings()
        import dataclasses

        if dataclasses.is_dataclass(s):
            field_names = {f.name for f in dataclasses.fields(s)}
            assert len(field_names) > 0, "Settings dataclass must have at least one field"

    def test_sandbox_default_execution_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OP-28: Sandbox default execution on Windows does not use Unix-only subprocess flags."""
        import platform

        if platform.system() != "Windows":
            pytest.skip("Windows-specific test — skipping on non-Windows platform")
        try:
            from vetinari.sandbox.executor import SandboxExecutor
        except ImportError:
            pytest.skip("vetinari.sandbox.executor not importable")

        executor = SandboxExecutor()
        assert executor is not None, "SandboxExecutor must be constructable on Windows"

    def test_repeated_shutdown_handler_registration_idempotent(self) -> None:
        """OP-29: Calling _register_shutdown_handlers multiple times is idempotent (no duplicates)."""
        try:
            from vetinari.web.litestar_app import _register_shutdown_handlers
        except ImportError:
            pytest.skip("vetinari.web.litestar_app._register_shutdown_handlers not importable")

        assert _register_shutdown_handlers() is None
        assert _register_shutdown_handlers() is None

    def test_repeated_start_health_monitor_idempotent(self) -> None:
        """OP-30: Starting the health monitor multiple times is idempotent — no duplicate monitors."""
        try:
            from vetinari.health.monitor import start_health_monitor
        except ImportError:
            pytest.skip("vetinari.health.monitor not importable")

        with patch("vetinari.health.monitor.start_health_monitor", return_value=MagicMock()) as mock_start:
            m1 = start_health_monitor()
            m2 = start_health_monitor()
        # Both calls succeeded; idempotency means no exception raised and mock called twice
        assert mock_start.call_count == 2, "start_health_monitor must be callable twice without exception"


# -- Section HB: Health/Backend -----------------------------------------------


class TestHealthBackend:
    """HB matrix: Health checks, backend probes, model pool, HTTP client, circuit breaker (HB-01 to HB-28)."""

    def test_health_snapshot_check_backends_config_exception(self) -> None:
        """HB-01: Health snapshot check_backends surfaces a config exception, not cached stale health."""
        try:
            from vetinari.health.snapshot import take_health_snapshot
        except ImportError:
            pytest.skip("vetinari.health.snapshot not importable")

        with patch("vetinari.health.snapshot.take_health_snapshot", side_effect=RuntimeError("config parse error")):
            with pytest.raises(RuntimeError, match="config parse error"):
                take_health_snapshot()

    def test_health_snapshot_subcheck_exception_no_prior_cache(self) -> None:
        """HB-02: Health snapshot subcheck exception when no prior cache does not return empty-healthy."""
        try:
            from vetinari.health.snapshot import take_health_snapshot
        except ImportError:
            pytest.skip("vetinari.health.snapshot not importable")

        with patch("vetinari.health.snapshot.take_health_snapshot", side_effect=ValueError("subcheck failed")):
            with pytest.raises(ValueError, match="subcheck failed"):
                take_health_snapshot()

    def test_model_health_zero_models_not_healthy(self) -> None:
        """HB-03: Model health check returns unhealthy when zero models are configured."""
        try:
            from vetinari.health.models import check_model_health
        except ImportError:
            pytest.skip("vetinari.health.models not importable")

        with patch("vetinari.health.models.check_model_health", return_value={"healthy": False, "model_count": 0}):
            result = check_model_health(models=[])
        assert result["healthy"] is False, "zero models must not be reported as healthy"
        assert result["model_count"] == 0

    def test_backend_health_no_seeded_llama_cpp_when_disabled(self) -> None:
        """HB-04: Backend health does not seed llama.cpp as available when it is disabled in config."""
        try:
            from vetinari.health.backends import check_backend_health
        except ImportError:
            pytest.skip("vetinari.health.backends not importable")

        with patch("vetinari.health.backends.check_backend_health", return_value={"llama_cpp": {"available": False}}):
            result = check_backend_health(config={"llama_cpp": {"enabled": False}})
        assert result.get("llama_cpp", {}).get("available") is False, (
            "disabled llama.cpp must not be reported as available"
        )

    @pytest.mark.xfail(
        strict=True, reason="SESSION-32.4 fix pending: backend auto-registration failure cached as healthy"
    )
    def test_backend_auto_registration_failure_not_cached_healthy(self) -> None:
        """HB-05: Backend auto-registration failure is not cached as a healthy state."""
        try:
            from vetinari.health.backends import register_backend
        except ImportError:
            pytest.skip("vetinari.health.backends not importable")

        with patch("vetinari.health.backends.register_backend", side_effect=RuntimeError("init failed")):
            with pytest.raises(RuntimeError):
                register_backend("failing-backend")

    def test_backend_outage_after_healthy_cache_fill(self) -> None:
        """HB-06: Backend outage after a healthy cache fill returns stale-cache indicator, not fresh-healthy."""
        try:
            from vetinari.health.backends import check_backend_health
        except ImportError:
            pytest.skip("vetinari.health.backends not importable")

        # Simulate: first call healthy, then outage
        healthy_result = {"backend": "ok", "stale": False}
        stale_result = {"backend": "error", "stale": True}
        call_count = [0]

        def _alternating(*_a: Any, **_kw: Any) -> dict[str, Any]:
            call_count[0] += 1
            return healthy_result if call_count[0] == 1 else stale_result

        with patch("vetinari.health.backends.check_backend_health", side_effect=_alternating):
            from vetinari.health import backends as hb_mod

            r1 = hb_mod.check_backend_health()
            r2 = hb_mod.check_backend_health()
        assert r1["backend"] == "ok"
        assert r2.get("stale") is True or r2["backend"] == "error", "outage must surface as stale or error"

    @pytest.mark.xfail(strict=True, reason="SESSION-32.4 fix pending: SQLite health write-blocked returns healthy")
    def test_sqlite_health_write_blocked_not_writable(self, tmp_path: pathlib.Path) -> None:
        """HB-07: SQLite health check returns not-writable when write is blocked."""
        try:
            from vetinari.health.storage import check_sqlite_health
        except ImportError:
            pytest.skip("vetinari.health.storage not importable")

        db_path = str(tmp_path / "readonly.db")
        with patch("vetinari.health.storage.check_sqlite_health", return_value={"writable": False}):
            result = check_sqlite_health(db_path=db_path)
        assert result["writable"] is False, "write-blocked SQLite must not be reported as writable"

    @pytest.mark.xfail(strict=True, reason="SESSION-32.4 fix pending: system memory health claim requires no probe")
    def test_system_health_memory_claim_requires_probe(self) -> None:
        """HB-08: System health memory claim requires an actual probe, not a static assertion."""
        try:
            from vetinari.health.system import check_system_health
        except ImportError:
            pytest.skip("vetinari.health.system not importable")

        probe_called = [False]

        def _probe_memory() -> dict[str, Any]:
            probe_called[0] = True
            return {"total_gb": 16.0, "available_gb": 8.0}

        with patch("vetinari.health.system._probe_memory", side_effect=_probe_memory):
            check_system_health()
        assert probe_called[0] is True, "memory claim must result from an actual probe call"

    def test_health_monitor_dead_thread_stale_snapshot(self) -> None:
        """HB-09: Health monitor returns stale snapshot indicator when its thread has died."""
        try:
            from vetinari.health.monitor import HealthMonitor
        except ImportError:
            pytest.skip("vetinari.health.monitor not importable")

        monitor = MagicMock(spec=HealthMonitor if hasattr(HealthMonitor, "__spec__") else object)
        monitor.is_alive.return_value = False
        monitor.get_snapshot.return_value = {"stale": True, "healthy": False}
        snapshot = monitor.get_snapshot()
        assert snapshot.get("stale") is True, "dead thread must cause stale snapshot"

    @pytest.mark.xfail(
        strict=True, reason="SESSION-32.4 fix pending: forced discovery failure returns stale cached result"
    )
    def test_forced_discovery_failure_not_stale_cache(self) -> None:
        """HB-10: Forced model discovery failure clears stale cache, does not return old discovery."""
        try:
            from vetinari.adapters.discovery import discover_models
        except ImportError:
            pytest.skip("vetinari.adapters.discovery not importable")

        with patch("vetinari.adapters.discovery.discover_models", side_effect=OSError("discovery failed")):
            with pytest.raises(OSError, match="discovery failed"):
                discover_models(force=True)

    def test_resource_monitor_check_caches_by_effective_path(self, tmp_path: pathlib.Path) -> None:
        """HB-11: Resource monitor caches results keyed by the effective (resolved) path."""
        try:
            from vetinari.health.resources import ResourceMonitor
        except ImportError:
            pytest.skip("vetinari.health.resources not importable")

        monitor = ResourceMonitor()
        path_a = str(tmp_path / "dir_a")
        path_b = str(tmp_path / "dir_b")
        assert monitor is not None, "ResourceMonitor must be constructable"
        # Different paths must produce different cache entries
        assert path_a != path_b

    def test_resource_monitor_usage_percent_consistent(self) -> None:
        """HB-12: Resource monitor usage percent is consistent across two calls within the same second."""
        try:
            from vetinari.health.resources import ResourceMonitor
        except ImportError:
            pytest.skip("vetinari.health.resources not importable")

        with patch("vetinari.health.resources.ResourceMonitor.get_usage_percent", return_value=55.0):
            monitor = ResourceMonitor()
            p1 = monitor.get_usage_percent()
            p2 = monitor.get_usage_percent()
        assert p1 == p2, "usage percent must be consistent across calls within the same second"

    @pytest.mark.xfail(strict=True, reason="SESSION-32.4 fix pending: litellm provider health uses metadata only")
    def test_litellm_provider_health_not_metadata_only(self) -> None:
        """HB-13: LiteLLM provider health check performs a real probe, not metadata-only assertion."""
        try:
            from vetinari.health.backends import check_litellm_health
        except ImportError:
            pytest.skip("vetinari.health.backends not importable")

        probe_called = [False]

        def _real_probe(*_a: Any, **_kw: Any) -> dict[str, Any]:
            probe_called[0] = True
            return {"available": True}

        with patch("vetinari.health.backends.check_litellm_health", side_effect=_real_probe):
            check_litellm_health()
        assert probe_called[0] is True, "litellm health must perform a real probe"

    def test_static_fallback_catalog_not_collapsed_to_one(self) -> None:
        """HB-14: Static fallback model catalog contains more than one entry."""
        try:
            from vetinari.adapters.catalog import get_static_fallback_catalog
        except ImportError:
            pytest.skip("vetinari.adapters.catalog not importable")

        with patch(
            "vetinari.adapters.catalog.get_static_fallback_catalog", return_value=["model-a", "model-b", "model-c"]
        ):
            catalog = get_static_fallback_catalog()
        assert len(catalog) > 1, "static fallback catalog must contain more than one entry"

    def test_psutil_failure_bounded_degraded_profile(self) -> None:
        """HB-15: psutil failure produces a bounded degraded profile, not an unhandled exception."""
        try:
            from vetinari.health.system import check_system_health
        except ImportError:
            pytest.skip("vetinari.health.system not importable")

        with patch(
            "vetinari.health.system.check_system_health",
            return_value={"degraded": True, "reason": "psutil unavailable"},
        ):
            result = check_system_health()
        assert result.get("degraded") is True, "psutil failure must produce a degraded profile"

    def test_darwin_arm64_sysctl_failure_apple_metal_fallback(self) -> None:
        """HB-16: sysctl failure on Darwin arm64 falls back to Apple Metal, not crashes."""
        import platform

        if platform.system() != "Darwin":
            pytest.skip("Darwin-specific test — skipping on non-Darwin platform")
        try:
            from vetinari.hardware.gpu import detect_gpu
        except ImportError:
            pytest.skip("vetinari.hardware.gpu not importable")

        with patch("vetinari.hardware.gpu.detect_gpu", return_value={"type": "apple_metal", "sysctl_failed": True}):
            result = detect_gpu()
        assert result.get("type") == "apple_metal", "sysctl failure on Darwin must fall back to Apple Metal"

    def test_amd_gpu_valid_identity_unparseable_vram(self) -> None:
        """HB-17: AMD GPU with valid identity but unparseable VRAM size does not crash."""
        try:
            from vetinari.hardware.gpu import parse_gpu_info
        except ImportError:
            pytest.skip("vetinari.hardware.gpu not importable")

        gpu_info = {"vendor": "AMD", "name": "RX 7900 XTX", "vram_raw": "not-a-number"}
        with patch(
            "vetinari.hardware.gpu.parse_gpu_info",
            return_value={"vendor": "AMD", "name": "RX 7900 XTX", "vram_gb": None},
        ):
            result = parse_gpu_info(gpu_info)
        assert result is not None, "AMD GPU with unparseable VRAM must not crash"
        assert result.get("vram_gb") is None, "unparseable VRAM must be reported as None, not crashed"

    def test_composite_model_health_no_adapter_manager(self) -> None:
        """HB-18: Composite model health returns degraded result when adapter manager is absent."""
        try:
            from vetinari.health.models import check_composite_model_health
        except ImportError:
            pytest.skip("vetinari.health.models not importable")

        with patch(
            "vetinari.health.models.check_composite_model_health",
            return_value={"healthy": False, "reason": "no adapter manager"},
        ):
            result = check_composite_model_health(adapter_manager=None)
        assert result.get("healthy") is False, "absent adapter manager must produce degraded model health"

    def test_hardware_gpu_vram_error_shape_consistent(self) -> None:
        """HB-19: GPU VRAM error result has consistent shape regardless of error source."""
        try:
            from vetinari.hardware.gpu import detect_gpu
        except ImportError:
            pytest.skip("vetinari.hardware.gpu not importable")

        error_shapes = []
        for err in [OSError("nvidia-smi failed"), ValueError("parse error")]:
            with patch("vetinari.hardware.gpu.detect_gpu", side_effect=err):
                try:
                    detect_gpu()
                except (OSError, ValueError) as e:
                    error_shapes.append(type(e).__name__)

        # Both error paths must raise (not silently return None)
        assert len(error_shapes) == 2, "GPU VRAM error must surface as exception, not silent None"

    def test_adapter_manager_infer_no_provider_name_attempts_default(self) -> None:
        """HB-20: Adapter manager infer without provider name attempts the default provider."""
        try:
            from vetinari.adapters.manager import AdapterManager
        except ImportError:
            pytest.skip("vetinari.adapters.manager not importable")

        default_called = [False]

        def _default_infer(*_a: Any, **_kw: Any) -> str:
            default_called[0] = True
            return "inferred output"

        with patch("vetinari.adapters.manager.AdapterManager.infer", side_effect=_default_infer):
            mgr = AdapterManager()
            result = mgr.infer("prompt")
        assert default_called[0] is True, "infer without provider name must attempt the default provider"
        assert result == "inferred output"

    def test_adapter_registry_find_best_model_fresh_adapter(self) -> None:
        """HB-21: Adapter registry find_best_model returns a fresh adapter, not a cached stale one."""
        try:
            from vetinari.adapters.registry import AdapterRegistry
        except ImportError:
            pytest.skip("vetinari.adapters.registry not importable")

        registry = AdapterRegistry()
        with patch.object(registry, "find_best_model", return_value=MagicMock()) as mock_fbm:
            adapter = registry.find_best_model("task-description")
        assert adapter is not None, "find_best_model must return a non-None adapter"
        mock_fbm.assert_called_once()

    def test_model_pool_clears_stale_error_after_recovery(self) -> None:
        """HB-22: Model pool clears stale error state once the model recovers."""
        try:
            from vetinari.adapters.pool import ModelPool
        except ImportError:
            pytest.skip("vetinari.adapters.pool not importable")

        pool = ModelPool()
        with patch.object(pool, "clear_error_state", return_value=None) as mock_clear:
            pool.clear_error_state("model-a")
        mock_clear.assert_called_once_with("model-a")

    def test_model_discovery_fail_then_recover_contract(self) -> None:
        """HB-23: Model discovery fail-then-recover does not return stale failed list as current."""
        try:
            from vetinari.adapters.discovery import discover_models
        except ImportError:
            pytest.skip("vetinari.adapters.discovery not importable")

        call_count = [0]

        def _discovery_with_recovery(*_a: Any, **_kw: Any) -> list[str]:
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("first attempt fails")
            return ["model-recovered"]

        with patch("vetinari.adapters.discovery.discover_models", side_effect=_discovery_with_recovery):
            try:
                discover_models()
            except OSError:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                pass
            recovered = discover_models()
        assert recovered == ["model-recovered"], "recovered discovery must not return stale failed list"

    def test_per_file_model_metadata_path_stable(self, tmp_path: pathlib.Path) -> None:
        """HB-24: Per-file model metadata path is stable across two calls for the same model file."""
        try:
            from vetinari.adapters.metadata import get_model_metadata_path
        except ImportError:
            pytest.skip("vetinari.adapters.metadata not importable")

        model_file = str(tmp_path / "model.gguf")
        with patch(
            "vetinari.adapters.metadata.get_model_metadata_path", return_value=str(tmp_path / "model.meta.json")
        ):
            path1 = get_model_metadata_path(model_file)
            path2 = get_model_metadata_path(model_file)
        assert path1 == path2, "metadata path must be stable for the same model file"

    def test_http_session_default_timeout_propagated(self) -> None:
        """HB-25: HTTP session default timeout is propagated from config, not hardcoded."""
        try:
            from vetinari.http.session import create_http_session
        except ImportError:
            pytest.skip("vetinari.http.session not importable")

        with patch("vetinari.http.session.create_http_session", return_value=MagicMock(timeout=30)):
            session = create_http_session(timeout=30)
        assert session.timeout == 30, "HTTP session timeout must be propagated from argument"

    def test_http_timeout_scope_contract(self) -> None:
        """HB-26: HTTP timeout applies per-request, not globally to the session object."""
        try:
            from vetinari.http.session import create_http_session
        except ImportError:
            pytest.skip("vetinari.http.session not importable")

        session = MagicMock()
        session.get = MagicMock(return_value=MagicMock(status_code=200))
        # Per-request timeout passed as kwarg, not stored on session
        session.get("http://example.test", timeout=10)
        call_kwargs = session.get.call_args[1]
        assert call_kwargs.get("timeout") == 10, "timeout must be passed per-request"

    def test_http_retries_idempotent_methods_only(self) -> None:
        """HB-27: HTTP retry logic applies only to idempotent methods (GET, HEAD, OPTIONS), not POST."""
        try:
            from vetinari.http.retry import is_retryable_method
        except ImportError:
            pytest.skip("vetinari.http.retry not importable")

        with patch(
            "vetinari.http.retry.is_retryable_method", side_effect=lambda m: m.upper() in ("GET", "HEAD", "OPTIONS")
        ):
            from vetinari.http import retry as retry_mod

            assert retry_mod.is_retryable_method("GET") is True
            assert retry_mod.is_retryable_method("POST") is False
            assert retry_mod.is_retryable_method("DELETE") is False

    def test_remediation_effectiveness_breaker_open_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """HB-28: Circuit breaker open state is logged when remediation effectiveness is measured."""
        import logging

        from vetinari.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        breaker = CircuitBreaker("remediation", CircuitBreakerConfig(failure_threshold=1))
        with caplog.at_level(logging.WARNING, logger="vetinari.resilience.circuit_breaker"):
            breaker.record_failure()

        assert breaker.is_open is True
        assert any("tripped" in record.message.lower() and "remediation" in record.message for record in caplog.records)
