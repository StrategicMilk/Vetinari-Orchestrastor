"""Tests for batch 33E.2F CLI and misc-scripts defect fixes."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# D1 — cli_packaging_doctor._check_database() uses correct table names
# ---------------------------------------------------------------------------


class TestPackagingDoctorDBSchema:
    """_check_database probes the live unified-schema tables, not the legacy ones."""

    def test_correct_table_names_in_source(self) -> None:
        """The set of expected tables must not contain legacy names."""
        import inspect

        import vetinari.cli_packaging_doctor as mod

        src = inspect.getsource(mod._check_database)
        # Legacy (wrong) table names must be absent
        assert "PlanHistory" not in src, "legacy table name PlanHistory still present"
        assert "SubtaskMemory" not in src, "legacy table name SubtaskMemory still present"
        assert "ModelPerformance" not in src, "legacy table name ModelPerformance still present"
        # Correct live table names must be present
        assert "execution_state" in src
        assert "memories" in src
        assert "sse_event_log" in src

    def test_pass_when_unified_tables_present(self, tmp_path: Path) -> None:
        """Returns PASS status when all three core tables exist."""
        db_path = tmp_path / "vetinari.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE execution_state (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE sse_event_log (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        def _fake_get_connection():
            return sqlite3.connect(str(db_path))

        import vetinari.cli_packaging_doctor as mod

        with patch("vetinari.database.get_connection", side_effect=_fake_get_connection):
            label, status, detail = mod._check_database()

        assert status == mod._CHECK_PASS, f"expected PASS but got {status!r}: {detail}"

    def test_warn_when_unified_tables_missing(self, tmp_path: Path) -> None:
        """Returns WARN status when core tables are absent (pre-migration DB)."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.commit()
        conn.close()

        def _fake_get_connection():
            return sqlite3.connect(str(db_path))

        import vetinari.cli_packaging_doctor as mod

        with patch("vetinari.database.get_connection", side_effect=_fake_get_connection):
            label, status, detail = mod._check_database()

        assert status == mod._CHECK_WARN, f"expected WARN but got {status!r}: {detail}"
        assert "missing" in detail.lower()


# ---------------------------------------------------------------------------
# D2 — cmd_start detects dead uvicorn thread
# ---------------------------------------------------------------------------


class TestCmdStartThreadAlive:
    """cmd_start reports startup failure when the dashboard thread dies immediately."""

    def _make_args(self, **kwargs: Any) -> MagicMock:
        args = MagicMock()
        args.verbose = False
        args.mode = "standard"
        args.skip_preflight = True
        args.no_dashboard = False
        args.port = 15432
        args.web_host = "127.0.0.1"
        args.goal = None
        args.task = None
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    def _make_mock_thread(self, *, alive: bool) -> MagicMock:
        """Return a mock Thread whose is_alive() returns the given value."""
        t = MagicMock(spec=threading.Thread)
        t.is_alive.return_value = alive
        return t

    def test_dead_thread_prints_failure(self, capsys: pytest.CaptureFixture) -> None:
        """When the uvicorn thread exits immediately, a failure message is printed."""
        import vetinari.cli_commands as mod

        dead_thread = self._make_mock_thread(alive=False)
        _real_thread = threading.Thread

        with (
            patch("vetinari.cli_commands.uvicorn", new=MagicMock()),
            patch("vetinari.cli_startup._setup_logging"),
            patch("vetinari.cli_startup._check_drift_at_startup"),
            patch("vetinari.cli_startup._print_banner"),
            patch("vetinari.cli_startup._wire_subsystems"),
            patch("vetinari.web.litestar_app.create_app", return_value=MagicMock()),
            patch(
                "vetinari.cli_commands.threading.Thread",
                side_effect=lambda *a, **kw: dead_thread if kw.get("name") == "dashboard" else _real_thread(*a, **kw),
            ),
            patch("time.sleep"),
            patch.object(mod, "_health_check_quiet"),
        ):
            with patch("vetinari.cli_commands.cmd_interactive", side_effect=KeyboardInterrupt):
                try:
                    mod.cmd_start(self._make_args())
                except KeyboardInterrupt:  # noqa: VET022 — test intentionally swallows interrupt after cmd exits
                    pass

        captured = capsys.readouterr()
        assert "startup failed" in captured.out.lower() or "failed" in captured.out.lower(), (
            f"Expected failure message, got: {captured.out!r}"
        )

    def test_alive_thread_prints_success(self, capsys: pytest.CaptureFixture) -> None:
        """When the uvicorn thread stays alive, the 'Dashboard started' message is printed."""
        import vetinari.cli_commands as mod

        alive_thread = self._make_mock_thread(alive=True)
        # conftest.py patches threading.Thread.start to silently skip "auto-tuner" threads
        # (to prevent daemon-thread accumulation across 80+ test files).  A skipped start()
        # leaves _started unset, so any subsequent join() raises RuntimeError.  Return a
        # mock for the auto-tuner thread so start()/join() are both harmless no-ops.
        noop_tuner_thread = MagicMock(spec=threading.Thread)
        _real_thread = threading.Thread  # capture before patch is applied

        def _thread_factory(*a: object, **kw: object) -> object:
            name = kw.get("name")
            if name == "dashboard":
                return alive_thread
            if name == "auto-tuner":
                return noop_tuner_thread
            return _real_thread(*a, **kw)

        # When dashboard_started=True, cmd_start enters a `while True: time.sleep(...)` loop.
        # The first sleep call is the startup-delay probe; the second is the main poll loop.
        # Raising KeyboardInterrupt on the second call terminates the loop cleanly, exactly
        # as Ctrl+C would at runtime.
        _sleep_calls: list[int] = [0]

        def _sleep_side_effect(_duration: float) -> None:
            _sleep_calls[0] += 1
            if _sleep_calls[0] >= 2:
                raise KeyboardInterrupt

        with (
            patch("vetinari.cli_commands.uvicorn", new=MagicMock()),
            patch("vetinari.cli_startup._setup_logging"),
            patch("vetinari.cli_startup._check_drift_at_startup"),
            patch("vetinari.cli_startup._print_banner"),
            patch("vetinari.cli_startup._wire_subsystems"),
            patch("vetinari.web.litestar_app.create_app", return_value=MagicMock()),
            patch("vetinari.cli_commands.threading.Thread", side_effect=_thread_factory),
            patch("vetinari.cli_commands.time.sleep", side_effect=_sleep_side_effect),
            patch.object(mod, "_health_check_quiet"),
        ):
            try:
                mod.cmd_start(self._make_args())
            except KeyboardInterrupt:  # noqa: VET022 — test intentionally swallows interrupt after cmd exits
                pass

        captured = capsys.readouterr()
        assert "Dashboard started" in captured.out, (
            f"Expected 'Dashboard started', got: {captured.out!r}"
        )


# ---------------------------------------------------------------------------
# D3 — cmd_start and cmd_serve advertise the configured web_host, not localhost
# ---------------------------------------------------------------------------


class TestCmdStartAdvertisedURL:
    """The URL printed to the user uses web_host, not hardcoded localhost."""

    def _make_args(self, host: str = "0.0.0.0", **kwargs: Any) -> MagicMock:
        args = MagicMock()
        args.verbose = False
        args.mode = "standard"
        args.skip_preflight = True
        args.no_dashboard = False
        args.port = 15433
        args.web_host = host
        args.goal = None
        args.task = None
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    def test_cmd_serve_url_uses_web_host(self, capsys: pytest.CaptureFixture) -> None:
        """cmd_serve prints the configured host, not localhost."""
        import vetinari.cli_commands as mod

        with (
            patch("vetinari.cli_commands.uvicorn", new=None),  # force early return
            patch("vetinari.cli_startup._setup_logging"),
            patch("vetinari.cli_startup._wire_subsystems"),
        ):
            mod.cmd_serve(self._make_args(host="192.168.1.50", port=15433))

        captured = capsys.readouterr()
        assert "192.168.1.50" in captured.out, (
            f"Expected host 192.168.1.50 in output, got: {captured.out!r}"
        )
        assert "localhost" not in captured.out, (
            f"localhost should not appear when web_host=192.168.1.50, got: {captured.out!r}"
        )

    def test_cmd_start_started_url_uses_web_host(self, capsys: pytest.CaptureFixture) -> None:
        """cmd_start 'Dashboard started' message uses configured web_host."""
        import vetinari.cli_commands as mod

        alive_thread: MagicMock = MagicMock(spec=threading.Thread)
        alive_thread.is_alive.return_value = True
        # conftest.py blocks "auto-tuner" thread.start() to prevent daemon accumulation;
        # a skipped start() leaves _started unset and join() raises RuntimeError.  Use a
        # mock so start()/join() are harmless no-ops in this test context.
        noop_tuner_thread = MagicMock(spec=threading.Thread)
        _real_thread = threading.Thread  # capture before patch is applied

        def _thread_factory(*a: object, **kw: object) -> object:
            name = kw.get("name")
            if name == "dashboard":
                return alive_thread
            if name == "auto-tuner":
                return noop_tuner_thread
            return _real_thread(*a, **kw)

        # dashboard_started=True routes into the `while True: time.sleep(...)` loop.
        # Raise KeyboardInterrupt on the second sleep call to terminate it cleanly.
        _sleep_calls: list[int] = [0]

        def _sleep_side_effect(_duration: float) -> None:
            _sleep_calls[0] += 1
            if _sleep_calls[0] >= 2:
                raise KeyboardInterrupt

        with (
            patch("vetinari.cli_commands.uvicorn", new=MagicMock()),
            patch("vetinari.cli_startup._setup_logging"),
            patch("vetinari.cli_startup._check_drift_at_startup"),
            patch("vetinari.cli_startup._print_banner"),
            patch("vetinari.cli_startup._wire_subsystems"),
            patch("vetinari.web.litestar_app.create_app", return_value=MagicMock()),
            patch("vetinari.cli_commands.threading.Thread", side_effect=_thread_factory),
            patch("vetinari.cli_commands.time.sleep", side_effect=_sleep_side_effect),
            patch.object(mod, "_health_check_quiet"),
        ):
            try:
                mod.cmd_start(self._make_args(host="10.0.0.5"))
            except KeyboardInterrupt:  # noqa: VET022 — test intentionally swallows interrupt after cmd exits
                pass

        captured = capsys.readouterr()
        assert "10.0.0.5" in captured.out, (
            f"Expected host 10.0.0.5 in output, got: {captured.out!r}"
        )
        assert "localhost" not in captured.out, (
            f"localhost should not appear when web_host=10.0.0.5, got: {captured.out!r}"
        )


# ---------------------------------------------------------------------------
# D4 — continuity_status.py reports .omc/ and .codex/ surfaces
# ---------------------------------------------------------------------------


class TestContinuityStatusCodexSurface:
    """build_status includes omc_state and codex_state sections."""

    def test_build_status_has_omc_section(self) -> None:
        """build_status result has an omc_state key."""
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "continuity_status",
            Path(__file__).parent.parent / "scripts" / "continuity_status.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        status = mod.build_status()
        assert "omc_state" in status, "omc_state missing from build_status output"
        assert "codex_state" in status, "codex_state missing from build_status output"

    def test_omc_state_keys_present(self) -> None:
        """omc_state section contains the expected keys."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "continuity_status",
            Path(__file__).parent.parent / "scripts" / "continuity_status.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        status = mod.build_status()
        omc = status["omc_state"]
        assert isinstance(omc, dict)
        for key in ("state_dir", "state_dir_exists", "notepad_exists", "project_memory_exists"):
            assert key in omc, f"omc_state missing key: {key}"

    def test_render_text_includes_omc_section(self) -> None:
        """render_text output contains OMC and Codex sections."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "continuity_status",
            Path(__file__).parent.parent / "scripts" / "continuity_status.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        status = mod.build_status()
        text = mod.render_text(status)
        assert "omc" in text.lower() or "OMC" in text, "OMC section absent from render_text output"
        assert "codex" in text.lower(), "Codex section absent from render_text output"


# ---------------------------------------------------------------------------
# D7 — migrate_to_unified_db: failed store → nonzero exit
# ---------------------------------------------------------------------------


class TestMigrateUnifiedDBFailure:
    """migrate_all reports errors and main() exits nonzero when a store fails."""

    def test_migrate_all_returns_error_list_on_failure(self, tmp_path: Path) -> None:
        """When a store raises sqlite3.Error, it appears in the errors list."""
        import importlib.util
        import sys

        script = Path(__file__).parent.parent / "scripts" / "migrate_to_unified_db.py"
        spec = importlib.util.spec_from_file_location("migrate_to_unified_db", script)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)

        # Provide vetinari.database stubs before exec
        fake_db = MagicMock()
        fake_db.get_connection = MagicMock(return_value=MagicMock())
        fake_db.init_schema = MagicMock()
        sys.modules.setdefault("vetinari", MagicMock())
        sys.modules["vetinari.database"] = fake_db

        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        bad_db = tmp_path / "bad.db"
        bad_db.write_bytes(b"not a sqlite db")

        # Use a store name that exists in _table_map so queries actually run
        # against the corrupt file and trigger sqlite3.Error
        with patch.object(mod, "_find_legacy_dbs", return_value={"durable_execution": bad_db}):
            results, errors = mod.migrate_all()

        assert "durable_execution" in errors, f"expected durable_execution in errors, got: {errors}"
        assert "durable_execution" not in results, "failed store must not appear in results dict"

    def test_migrate_all_empty_errors_on_success(self, tmp_path: Path) -> None:
        """When no stores fail, errors list is empty."""
        import importlib.util
        import sys

        script = Path(__file__).parent.parent / "scripts" / "migrate_to_unified_db.py"
        spec = importlib.util.spec_from_file_location("migrate_to_unified_db2", script)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)

        fake_db = MagicMock()
        fake_db.get_connection = MagicMock(return_value=MagicMock())
        fake_db.init_schema = MagicMock()
        sys.modules["vetinari.database"] = fake_db

        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        with patch.object(mod, "_find_legacy_dbs", return_value={}):
            results, errors = mod.migrate_all()

        assert errors == [], f"expected no errors but got: {errors}"

    def test_main_exits_nonzero_on_failure(self, tmp_path: Path) -> None:
        """main() returns 1 when migrate_all reports failures."""
        import importlib.util
        import sys

        script = Path(__file__).parent.parent / "scripts" / "migrate_to_unified_db.py"
        spec = importlib.util.spec_from_file_location("migrate_to_unified_db3", script)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)

        fake_db = MagicMock()
        fake_db.get_connection = MagicMock(return_value=MagicMock())
        fake_db.init_schema = MagicMock()
        sys.modules["vetinari.database"] = fake_db

        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        with (
            patch.object(mod, "migrate_all", return_value=({"s": 0}, ["s"])),
            patch("sys.argv", ["migrate_to_unified_db.py"]),
        ):
            exit_code = mod.main()

        assert exit_code == 1, f"expected exit code 1 but got {exit_code}"

    def test_main_exits_zero_on_success(self, tmp_path: Path) -> None:
        """main() returns 0 when migrate_all reports no failures."""
        import importlib.util
        import sys

        script = Path(__file__).parent.parent / "scripts" / "migrate_to_unified_db.py"
        spec = importlib.util.spec_from_file_location("migrate_to_unified_db4", script)
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)

        fake_db = MagicMock()
        fake_db.get_connection = MagicMock(return_value=MagicMock())
        fake_db.init_schema = MagicMock()
        sys.modules["vetinari.database"] = fake_db

        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        with (
            patch.object(mod, "migrate_all", return_value=({}, [])),
            patch("sys.argv", ["migrate_to_unified_db.py"]),
        ):
            exit_code = mod.main()

        assert exit_code == 0, f"expected exit code 0 but got {exit_code}"
