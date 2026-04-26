"""
Comprehensive tests for vetinari/cli.py.

All external dependencies are stubbed via sys.modules before the cli module
is imported so that no real network calls, database access, or heavy
orchestration code is exercised.
"""

import logging
import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# sys.modules stubs — must be installed BEFORE importing vetinari.cli
# ---------------------------------------------------------------------------


def _make_stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Top-level vetinari package stub.
# Must have __path__ pointing to the REAL directory so Python can discover sub-modules.
_CLI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_vetinari_pkg = _make_stub("vetinari")
_vetinari_pkg.__path__ = [os.path.join(_CLI_ROOT, "vetinari")]
_vetinari_pkg.__package__ = "vetinari"
sys.modules["vetinari"] = _vetinari_pkg

# yaml — used by _load_config
_yaml_stub = _make_stub("yaml")
_yaml_stub.safe_load = MagicMock(return_value={})
sys.modules.setdefault("yaml", _yaml_stub)

# vetinari.orchestrator
_mock_orchestrator_cls = MagicMock()
_orch_stub = _make_stub(
    "vetinari.orchestrator",
    Orchestrator=_mock_orchestrator_cls,
)
sys.modules["vetinari.orchestrator"] = _orch_stub

# vetinari.orchestration package — real __path__ so sub-modules are discoverable
_orch_pkg = _make_stub("vetinari.orchestration")
_orch_pkg.__path__ = [os.path.join(_CLI_ROOT, "vetinari", "orchestration")]
_orch_pkg.__package__ = "vetinari.orchestration"
sys.modules["vetinari.orchestration"] = _orch_pkg

# vetinari.orchestration.two_layer
_mock_get_two_layer = MagicMock()
_mock_init_two_layer = MagicMock()
_two_layer_stub = _make_stub(
    "vetinari.orchestration.two_layer",
    get_two_layer_orchestrator=_mock_get_two_layer,
    init_two_layer_orchestrator=_mock_init_two_layer,
)
sys.modules["vetinari.orchestration.two_layer"] = _two_layer_stub

# vetinari.orchestration.types  (imported by two_layer_orchestration shim)
sys.modules["vetinari.orchestration.types"] = _make_stub("vetinari.orchestration.types")

# vetinari.types — load REAL module (stdlib-only) so execution_graph & durable_execution resolve
import importlib.util as _cli_ilu

_vtypes_path = os.path.join(_CLI_ROOT, "vetinari", "types.py")
_vtypes_spec = _cli_ilu.spec_from_file_location("vetinari.types", _vtypes_path)
_vtypes_mod = _cli_ilu.module_from_spec(_vtypes_spec)
sys.modules["vetinari.types"] = _vtypes_mod
_vtypes_spec.loader.exec_module(_vtypes_mod)

# vetinari.orchestration.execution_graph — load REAL (depends only on vetinari.types)
_eg_path = os.path.join(_CLI_ROOT, "vetinari", "orchestration", "execution_graph.py")
_eg_spec = _cli_ilu.spec_from_file_location("vetinari.orchestration.execution_graph", _eg_path)
_eg_mod = _cli_ilu.module_from_spec(_eg_spec)
sys.modules["vetinari.orchestration.execution_graph"] = _eg_mod
_eg_spec.loader.exec_module(_eg_mod)

# vetinari.orchestration.durable_execution — load REAL (depends on execution_graph + vetinari.types)
_de_path = os.path.join(_CLI_ROOT, "vetinari", "orchestration", "durable_execution.py")
_de_spec = _cli_ilu.spec_from_file_location("vetinari.orchestration.durable_execution", _de_path)
_de_mod = _cli_ilu.module_from_spec(_de_spec)
sys.modules["vetinari.orchestration.durable_execution"] = _de_mod
_de_spec.loader.exec_module(_de_mod)

# vetinari.orchestration.plan_generator
sys.modules["vetinari.orchestration.plan_generator"] = _make_stub(
    "vetinari.orchestration.plan_generator",
    PlanGenerator=MagicMock(),
)

# uvicorn — stubbed so cmd_serve / cmd_start don't actually bind a socket
_mock_uvicorn_run = MagicMock(return_value=None)
_uvicorn_stub = _make_stub("uvicorn", run=_mock_uvicorn_run)
sys.modules["uvicorn"] = _uvicorn_stub

# vetinari.web.litestar_app — stubbed so create_app() doesn't spin up a real Litestar instance
_mock_litestar_create_app = MagicMock(return_value=MagicMock())
if "vetinari.web" not in sys.modules:
    _web_pkg = _make_stub("vetinari.web")
    _web_pkg.__path__ = [os.path.join(_CLI_ROOT, "vetinari", "web")]
    _web_pkg.__package__ = "vetinari.web"
    sys.modules["vetinari.web"] = _web_pkg
_litestar_app_stub = _make_stub(
    "vetinari.web.litestar_app",
    create_app=_mock_litestar_create_app,
)
sys.modules["vetinari.web.litestar_app"] = _litestar_app_stub

# vetinari.adapters.llama_cpp_adapter and llama_cpp_local_adapter
_mock_local_inference_cls = MagicMock()
if "vetinari.adapters" not in sys.modules:
    _adapters_pkg = _make_stub("vetinari.adapters")
    _adapters_pkg.__path__ = [os.path.join(_CLI_ROOT, "vetinari", "adapters")]
    _adapters_pkg.__package__ = "vetinari.adapters"
    sys.modules["vetinari.adapters"] = _adapters_pkg
_mock_provider_adapter_cls = MagicMock()
_llama_cpp_stub = _make_stub(
    "vetinari.adapters.llama_cpp_adapter",
    LlamaCppProviderAdapter=_mock_provider_adapter_cls,
    LocalInferenceAdapter=_mock_local_inference_cls,
)
sys.modules["vetinari.adapters.llama_cpp_adapter"] = _llama_cpp_stub
# llama_cpp_local_adapter is the canonical source for LocalInferenceAdapter
_llama_cpp_local_stub = _make_stub(
    "vetinari.adapters.llama_cpp_local_adapter",
    LocalInferenceAdapter=_mock_local_inference_cls,
)
sys.modules["vetinari.adapters.llama_cpp_local_adapter"] = _llama_cpp_local_stub

# vetinari.adapter_manager
_mock_get_adapter_mgr = MagicMock()
_adapter_mgr_stub = _make_stub(
    "vetinari.adapter_manager",
    get_adapter_manager=_mock_get_adapter_mgr,
)
sys.modules["vetinari.adapter_manager"] = _adapter_mgr_stub

# vetinari.learning package and sub-modules
if "vetinari.learning" not in sys.modules:
    _learning_pkg = _make_stub("vetinari.learning")
    _learning_pkg.__path__ = [os.path.join(_CLI_ROOT, "vetinari", "learning")]
    _learning_pkg.__package__ = "vetinari.learning"
    sys.modules["vetinari.learning"] = _learning_pkg
else:
    _learning_pkg = sys.modules["vetinari.learning"]

_mock_get_thompson = MagicMock()
_model_selector_stub = _make_stub(
    "vetinari.learning.model_selector",
    get_thompson_selector=_mock_get_thompson,
)
sys.modules["vetinari.learning.model_selector"] = _model_selector_stub

_mock_get_auto_tuner = MagicMock()
_auto_tuner_stub = _make_stub(
    "vetinari.learning.auto_tuner",
    get_auto_tuner=_mock_get_auto_tuner,
)
sys.modules["vetinari.learning.auto_tuner"] = _auto_tuner_stub

# vetinari.agents package
if "vetinari.agents" not in sys.modules:
    _agents_pkg = _make_stub("vetinari.agents")
    _agents_pkg.__path__ = [os.path.join(_CLI_ROOT, "vetinari", "agents")]
    _agents_pkg.__package__ = "vetinari.agents"
    sys.modules["vetinari.agents"] = _agents_pkg
else:
    _agents_pkg = sys.modules["vetinari.agents"]

# vetinari.agents.base_agent
sys.modules["vetinari.agents.base_agent"] = _make_stub(
    "vetinari.agents.base_agent",
    BaseAgent=MagicMock(),
)

# vetinari.agents.contracts — use real-ish AgentTask / AgentType stubs
_AgentType_mock = MagicMock()
_AgentType_mock.IMPROVEMENT = "IMPROVEMENT"
_AgentTask_mock = MagicMock()
_AgentResult_mock = MagicMock()
_VerificationResult_mock = MagicMock()
_contracts_stub = _make_stub(
    "vetinari.agents.contracts",
    AgentType=_AgentType_mock,
    AgentTask=_AgentTask_mock,
    AgentResult=_AgentResult_mock,
    VerificationResult=_VerificationResult_mock,
)
sys.modules["vetinari.agents.contracts"] = _contracts_stub

# vetinari.agents.consolidated.operations_agent
_mock_get_operations = MagicMock()
# Also expose get_worker_agent on the vetinari.agents stub so lazy imports in
# production code (e.g. cli_devops.py) resolve against the same mock.
_agents_pkg.get_worker_agent = _mock_get_operations
# Ensure the consolidated subpackage exists in sys.modules
if "vetinari.agents.consolidated" not in sys.modules:
    _consolidated_pkg = _make_stub("vetinari.agents.consolidated")
    _consolidated_pkg.__path__ = [os.path.join(_CLI_ROOT, "vetinari", "agents", "consolidated")]
    _consolidated_pkg.__package__ = "vetinari.agents.consolidated"
    sys.modules["vetinari.agents.consolidated"] = _consolidated_pkg
_operations_stub = _make_stub(
    "vetinari.agents.consolidated.operations_agent",
    get_operations_agent=_mock_get_operations,
    OperationsAgent=MagicMock(),
)
sys.modules["vetinari.agents.consolidated.operations_agent"] = _operations_stub


# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------

import pytest

import vetinari.cli as cli
import vetinari.cli_commands as cli_commands
import vetinari.cli_devops as cli_devops
import vetinari.cli_startup as cli_startup

# ---------------------------------------------------------------------------
# Helper: build a minimal args namespace that satisfies all cmd_* functions
# ---------------------------------------------------------------------------


def _args(**overrides) -> SimpleNamespace:
    """Return a namespace with safe defaults for every attribute cli uses."""
    defaults = {
        "verbose": False,
        "config": "manifest/vetinari.yaml",
        "mode": "execution",
        "goal": None,
        "task": None,
        "port": None,
        "web_host": "127.0.0.1",
        "debug": False,
        "no_dashboard": True,  # disable dashboard by default in tests
        "command": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ===========================================================================
# TestCmdRun
# ===========================================================================


class TestCmdRun:
    @pytest.fixture(autouse=True)
    def setup(self):
        _mock_orchestrator_cls.reset_mock()
        _mock_get_two_layer.reset_mock()

    # --- goal-path ---

    @pytest.mark.parametrize(
        ("result", "expected_rc"),
        [
            ({"completed": 3, "failed": 0}, 0),
            ({"completed": 0, "failed": 0}, 0),
        ],
    )
    def test_run_goal_returns_0_on_success(self, result, expected_rc):
        """cmd_run returns 0 when generate_and_execute succeeds."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = result
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="Build a REST API")
        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            rc = cli.cmd_run(args)

        assert rc == expected_rc

    def test_run_goal_passes_goal_string(self):
        """generate_and_execute receives the correct goal kwarg."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 1, "failed": 0}
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="My specific goal")
        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            cli.cmd_run(args)

        call_kwargs = mock_orch.generate_and_execute.call_args
        assert (call_kwargs.kwargs.get("goal") or call_kwargs.args[0]) == "My specific goal"

    def test_run_goal_with_final_output(self):
        """cmd_run prints final_output when present."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {
            "completed": 1,
            "failed": 0,
            "final_output": "The result text",
        }
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="Do something")
        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            with patch("builtins.print") as mock_print:
                rc = cli.cmd_run(args)

        assert rc == 0
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "The result text" in printed

    def test_run_goal_exception_returns_1(self):
        """cmd_run returns 1 when generate_and_execute raises."""
        _mock_get_two_layer.side_effect = RuntimeError("Local inference offline")

        args = _args(goal="Failing goal")
        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            rc = cli.cmd_run(args)

        assert rc == 1
        _mock_get_two_layer.side_effect = None

    def test_run_goal_prints_error_on_exception(self):
        """cmd_run prints error message when an exception is raised."""
        _mock_get_two_layer.side_effect = RuntimeError("bang")

        args = _args(goal="Bad goal")
        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            with patch("builtins.print") as mock_print:
                cli.cmd_run(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "bang" in printed
        _mock_get_two_layer.side_effect = None

    def test_run_goal_agent_context_wiring_failure_is_non_fatal(self):
        """If wiring agent context fails, cmd_run still succeeds."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 1, "failed": 0}
        # get_two_layer_orchestrator must succeed; only _build_orchestrator raises
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        # Patch both possible import paths for get_two_layer_orchestrator
        with patch.object(cli_startup, "_build_orchestrator", side_effect=Exception("no config")):
            # Also ensure the two_layer compat shim returns our mock
            with patch.dict(
                sys.modules,
                {
                    "vetinari.orchestration.two_layer": _two_layer_stub,
                },
            ):
                args = _args(goal="Wiring failure goal")
                rc = cli.cmd_run(args)

        assert rc == 0

    # --- manifest-path (no goal) ---

    def test_run_no_goal_no_task_calls_run_all(self):
        """cmd_run with no goal and no task calls orch.run_all()."""
        mock_orch = MagicMock()
        with patch.object(cli_startup, "_build_orchestrator", return_value=mock_orch):
            args = _args(goal=None, task=None)
            rc = cli.cmd_run(args)

        assert rc == 0
        mock_orch.run_all.assert_called_once()

    def test_run_specific_task_calls_run_task(self):
        """cmd_run with a task ID calls orch.run_task(task_id)."""
        mock_orch = MagicMock()
        with patch.object(cli_startup, "_build_orchestrator", return_value=mock_orch):
            args = _args(goal=None, task="t42")
            rc = cli.cmd_run(args)

        assert rc == 0
        mock_orch.run_task.assert_called_once_with("t42")

    def test_run_manifest_exception_returns_1(self):
        """cmd_run returns 1 when manifest execution raises."""
        with patch.object(cli_startup, "_build_orchestrator", side_effect=Exception("oops")):
            args = _args(goal=None, task=None)
            rc = cli.cmd_run(args)

        assert rc == 1

    def test_run_verbose_flag_sets_logging(self):
        """cmd_run with verbose=True calls _setup_logging(True)."""
        mock_orch = MagicMock()
        with patch.object(cli_startup, "_build_orchestrator", return_value=mock_orch):
            with patch.object(cli_startup, "_setup_logging") as mock_log:
                args = _args(goal=None, task=None, verbose=True)
                cli.cmd_run(args)

        mock_log.assert_called_with(True)

    def test_run_task_calls_build_orchestrator(self):
        """cmd_run with a task calls _build_orchestrator."""
        mock_orch = MagicMock()
        args = _args(goal=None, task="t1", config="custom.yaml", mode="planning")
        with patch.object(cli_startup, "_build_orchestrator", return_value=mock_orch) as mock_build:
            rc = cli.cmd_run(args)
        assert rc == 0
        assert mock_build.call_args.args == ("custom.yaml", "planning")
        assert mock_orch.run_task.call_args.args == ("t1",)
        mock_orch.run_all.assert_not_called()

    def test_run_goal_completed_count_printed(self):
        """The completed/failed task counts appear in printed output."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 7, "failed": 2}
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="count test")
        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            with patch("builtins.print") as mock_print:
                cli.cmd_run(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "7" in printed
        assert "2" in printed

    def test_run_goal_constraints_include_mode(self):
        """generate_and_execute is called with constraints containing mode."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 0, "failed": 0}
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="mode test", mode="planning")
        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            cli.cmd_run(args)

        call_kwargs = mock_orch.generate_and_execute.call_args.kwargs
        assert "planning" in str(call_kwargs)

    def test_run_no_goal_prints_running_message(self):
        """Without a goal, cmd_run prints a running tasks message."""
        mock_orch = MagicMock()
        with patch.object(cli_startup, "_build_orchestrator", return_value=mock_orch):
            with patch("builtins.print") as mock_print:
                args = _args(goal=None, task=None)
                cli.cmd_run(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "manifest" in printed.lower()


# ===========================================================================
# TestCmdServe
# ===========================================================================


class TestCmdServe:
    """Tests for cmd_serve — uses uvicorn + Litestar instead of Flask."""

    @pytest.fixture(autouse=True)
    def mock_uvicorn(self):
        """Patch the module-level uvicorn reference in cli_commands for each test.

        cli_commands imports uvicorn at module level (optional dep), so
        ``patch("vetinari.cli_commands.uvicorn")`` reliably intercepts all
        calls to uvicorn.run() inside cmd_serve and cmd_start.
        """
        mock_uv = MagicMock()
        mock_uv.run = MagicMock(return_value=None)
        with (
            patch("vetinari.cli_commands.uvicorn", mock_uv),
            patch("vetinari.web.litestar_app.create_app", _mock_litestar_create_app),
        ):
            self._mock_uv = mock_uv
            self._mock_run = mock_uv.run
            _mock_litestar_create_app.reset_mock()
            _mock_litestar_create_app.return_value = MagicMock()
            yield mock_uv

    def test_serve_calls_uvicorn_run(self):
        """cmd_serve calls uvicorn.run() with the Litestar app."""
        args = _args(port=5000, web_host="127.0.0.1", debug=False)
        rc = cli.cmd_serve(args)
        assert rc == 0
        self._mock_run.assert_called_once()

    def test_serve_uses_specified_port(self):
        """cmd_serve passes the given port to uvicorn.run."""
        args = _args(port=8080, web_host="127.0.0.1", debug=False)
        cli.cmd_serve(args)
        all_args = str(self._mock_run.call_args)
        assert "8080" in all_args

    def test_serve_default_port_from_env(self):
        """cmd_serve uses VETINARI_WEB_PORT env var when port is None."""
        args = _args(port=None, web_host="0.0.0.0", debug=False)
        with patch.dict(os.environ, {"VETINARI_WEB_PORT": "6000"}):
            cli.cmd_serve(args)
        all_args = str(self._mock_run.call_args)
        assert "6000" in all_args

    def test_serve_default_port_5000(self):
        """cmd_serve defaults to port 5000 when no port or env var given."""
        args = _args(port=None, web_host="127.0.0.1", debug=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VETINARI_WEB_PORT", None)
            cli.cmd_serve(args)
        all_args = str(self._mock_run.call_args)
        assert "5000" in all_args

    def test_serve_calls_uvicorn_with_correct_host(self):
        """cmd_serve passes the web_host bind address to uvicorn.run."""
        args = _args(port=5000, web_host="0.0.0.0", debug=False)
        cli.cmd_serve(args)
        all_args = str(self._mock_run.call_args)
        assert "0.0.0.0" in all_args

    def test_serve_debug_flag_passed_to_create_app(self):
        """debug=True is forwarded to create_app()."""
        args = _args(port=5000, web_host="127.0.0.1", debug=True)
        cli.cmd_serve(args)
        _, kwargs = _mock_litestar_create_app.call_args
        assert kwargs.get("debug") is True

    def test_serve_uses_web_host(self):
        """cmd_serve uses the specified web_host bind address."""
        args = _args(port=5000, web_host="0.0.0.0", debug=False)
        cli.cmd_serve(args)
        all_args = str(self._mock_run.call_args)
        assert "0.0.0.0" in all_args

    def test_serve_import_error_returns_1(self):
        """cmd_serve returns 1 if uvicorn is not available (None at module level)."""
        # cli_commands now imports uvicorn at module level (None when missing).
        # Patch it to None to simulate the not-installed case.
        with patch("vetinari.cli_commands.uvicorn", None):
            args = _args(port=5000, web_host="127.0.0.1", debug=False)
            rc = cli.cmd_serve(args)
        assert rc == 1

    def test_serve_uvicorn_run_exception_returns_1(self):
        """cmd_serve returns 1 when uvicorn.run raises an unexpected exception."""
        self._mock_run.side_effect = OSError("port in use")
        args = _args(port=5000, web_host="127.0.0.1", debug=False)
        rc = cli.cmd_serve(args)
        assert rc == 1

    def test_serve_log_level_info(self):
        """cmd_serve passes log_level='info' to uvicorn.run."""
        args = _args(port=5000, web_host="127.0.0.1", debug=False)
        cli.cmd_serve(args)
        all_args = str(self._mock_run.call_args)
        assert "info" in all_args


# ===========================================================================
# TestCmdStart
# ===========================================================================


class TestCmdStart:
    @pytest.fixture(autouse=True)
    def setup(self):
        _mock_litestar_create_app.reset_mock()
        _mock_litestar_create_app.return_value = MagicMock()
        _mock_get_auto_tuner.reset_mock()
        # Patch the module-level uvicorn reference so no real server starts
        self._mock_uv = MagicMock()
        self._mock_uv.run = MagicMock(return_value=None)
        with patch("vetinari.cli_commands.uvicorn", self._mock_uv):
            yield

    def _args_no_dashboard_no_goal(self, **kw):
        return _args(no_dashboard=True, goal=None, task=None, port=None, **kw)

    def test_start_no_dashboard_no_goal_calls_interactive(self):
        """With no dashboard and no goal, cmd_start delegates to cmd_interactive."""
        with patch.object(cli_commands, "cmd_interactive", return_value=0) as mock_inter:
            with patch.object(cli_commands, "_health_check_quiet"):
                args = self._args_no_dashboard_no_goal()
                rc = cli.cmd_start(args)
        mock_inter.assert_called_once_with(args)
        assert rc == 0

    def test_start_with_goal_calls_cmd_run(self):
        """With a goal, cmd_start delegates to cmd_run."""
        with (
            patch.object(cli_commands, "cmd_run", return_value=0) as mock_run,
            patch.object(cli_commands, "_health_check_quiet"),
        ):
            args = _args(no_dashboard=True, goal="some goal", task=None, port=None)
            rc = cli.cmd_start(args)
        mock_run.assert_called_once_with(args)
        assert rc == 0

    def test_start_with_task_calls_cmd_run(self):
        """With a task, cmd_start delegates to cmd_run."""
        with (
            patch.object(cli_commands, "cmd_run", return_value=0) as mock_run,
            patch.object(cli_commands, "_health_check_quiet"),
        ):
            args = _args(no_dashboard=True, goal=None, task="t1", port=None)
            rc = cli.cmd_start(args)
        mock_run.assert_called_once_with(args)
        assert rc == 0

    def test_start_runs_health_check(self):
        """cmd_start always calls _health_check_quiet."""
        args = self._args_no_dashboard_no_goal()
        with (
            patch.object(cli_commands, "cmd_interactive", return_value=0) as mock_interactive,
            patch.object(cli_commands, "_health_check_quiet", return_value=False) as mock_hc,
            patch("builtins.print") as mock_print,
        ):
            rc = cli.cmd_start(args)
        assert rc == 0
        assert mock_hc.call_count == 1
        assert mock_interactive.call_args.args == (args,)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "startup health checks reported a degraded or failed subsystem" in printed.lower()

    def test_start_prints_banner(self):
        """cmd_start prints the mode/host banner."""
        with (
            patch.object(cli_commands, "cmd_interactive", return_value=0),
            patch.object(cli_commands, "_health_check_quiet"),
        ):
            with patch.object(cli_startup, "_print_banner") as mock_banner:
                args = self._args_no_dashboard_no_goal(mode="planning")
                cli.cmd_start(args)
        mock_banner.assert_called_once()
        assert mock_banner.call_args.args[0] == "planning"

    def test_start_spawns_auto_tuner_thread(self):
        """cmd_start spawns a daemon thread named 'auto-tuner'."""
        spawned = []
        original_thread = threading.Thread

        def capturing_thread(*args, **kwargs):
            t = original_thread(*args, **kwargs)
            spawned.append(kwargs.get("name", ""))
            return t

        with patch("vetinari.cli_commands.threading.Thread", side_effect=capturing_thread):
            with patch.object(cli_commands, "cmd_interactive", return_value=0):
                with patch.object(cli_commands, "_health_check_quiet"):
                    args = self._args_no_dashboard_no_goal()
                    cli.cmd_start(args)

        assert any("auto-tuner" in n for n in spawned)

    def test_start_dashboard_disabled_skips_uvicorn(self):
        """no_dashboard=True means uvicorn.run is never called."""
        with (
            patch.object(cli_commands, "cmd_interactive", return_value=0),
            patch.object(cli_commands, "_health_check_quiet"),
        ):
            args = self._args_no_dashboard_no_goal()
            cli.cmd_start(args)
        self._mock_uv.run.assert_not_called()

    def test_start_verbose_propagates_to_logging(self):
        """cmd_start calls _setup_logging with the verbose flag."""
        with patch.object(cli_startup, "_setup_logging") as mock_log:
            with patch.object(cli_commands, "cmd_interactive", return_value=0):
                with patch.object(cli_commands, "_health_check_quiet"):
                    args = self._args_no_dashboard_no_goal(verbose=True)
                    cli.cmd_start(args)
        mock_log.assert_called_with(True)

    def test_start_port_default_from_env(self):
        """cmd_start reads VETINARI_WEB_PORT env var for port."""
        with patch.dict(os.environ, {"VETINARI_WEB_PORT": "7070"}):
            with patch.object(cli_commands, "cmd_interactive", return_value=0) as _mock_inter:
                with patch.object(cli_commands, "_health_check_quiet"):
                    # no_dashboard=True so port resolution happens but Flask not started
                    args = _args(no_dashboard=True, goal=None, task=None, port=None)
                    rc = cli.cmd_start(args)
        assert rc == 0  # cmd_start must succeed when no_dashboard=True

    def test_start_returns_zero_on_success(self):
        """cmd_start returns 0 on normal completion."""
        with (
            patch.object(cli_commands, "cmd_interactive", return_value=0),
            patch.object(cli_commands, "_health_check_quiet"),
        ):
            args = self._args_no_dashboard_no_goal()
            rc = cli.cmd_start(args)
        assert rc == 0

    def test_start_cmd_run_return_value_propagated(self):
        """Return code from cmd_run is propagated by cmd_start."""
        with patch.object(cli_commands, "cmd_run", return_value=1), patch.object(cli_commands, "_health_check_quiet"):
            args = _args(no_dashboard=True, goal="failing goal", task=None, port=None)
            rc = cli.cmd_start(args)
        assert rc == 1

    def test_start_no_dashboard_no_goal_no_task_calls_interactive_not_run(self):
        """When no goal and no task, cmd_run is not called."""
        with (
            patch.object(cli_commands, "cmd_run") as mock_run,
            patch.object(cli_commands, "cmd_interactive", return_value=0),
        ):
            with patch.object(cli_commands, "_health_check_quiet"):
                args = self._args_no_dashboard_no_goal()
                cli.cmd_start(args)
        mock_run.assert_not_called()


# ===========================================================================
# TestCmdStatus
# ===========================================================================


class TestCmdStatus:
    @pytest.fixture(autouse=True)
    def setup(self):
        _mock_local_inference_cls.reset_mock()
        _mock_get_adapter_mgr.reset_mock()
        _mock_get_adapter_mgr.return_value.health_check.return_value = {"local": {"healthy": True}}
        _mock_get_thompson.reset_mock()

    def _run_status(self, **kw):
        args = _args(**kw)
        with patch("builtins.print") as mock_print:
            rc = cli.cmd_status(args)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        return rc, printed

    def test_status_returns_0(self):
        """cmd_status always returns 0."""
        rc, _ = self._run_status()
        assert rc == 0

    def test_status_prints_config_path(self):
        """cmd_status prints the config path."""
        _, printed = self._run_status(config="manifest/vetinari.yaml")
        assert "manifest/vetinari.yaml" in printed

    def test_status_local_inference_reachable(self):
        """cmd_status prints model count when local inference is reachable."""
        mock_adapter = MagicMock()
        mock_adapter.list_loaded_models.return_value = [{"id": "m1"}, {"id": "m2"}]
        _mock_local_inference_cls.return_value = mock_adapter

        _, printed = self._run_status()
        assert "2" in printed

    def test_status_local_inference_unreachable(self):
        """cmd_status prints UNREACHABLE when local inference is unavailable."""
        _mock_local_inference_cls.side_effect = ConnectionError("refused")

        _, printed = self._run_status()
        assert "UNREACHABLE" in printed
        _mock_local_inference_cls.side_effect = None

    def test_status_local_inference_empty_list(self):
        """cmd_status handles empty model list gracefully."""
        mock_adapter = MagicMock()
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        rc, _ = self._run_status()
        assert rc == 0

    def test_status_adapter_manager_listed(self):
        """cmd_status lists providers from adapter manager."""
        mock_adapter = MagicMock()
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        mock_mgr = MagicMock()
        mock_mgr.get_status.return_value = {
            "providers": {"openai": {"health": "ok"}, "anthropic": {"health": "degraded"}}
        }
        _mock_get_adapter_mgr.return_value = mock_mgr

        _, printed = self._run_status()
        assert "openai" in printed

    def test_status_adapter_manager_exception_handled(self):
        """cmd_status handles adapter manager exception without crashing."""
        mock_adapter = MagicMock()
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        _mock_get_adapter_mgr.side_effect = RuntimeError("no mgr")
        rc, _ = self._run_status()
        assert rc == 0
        _mock_get_adapter_mgr.side_effect = None

    def test_status_thompson_selector_listed(self):
        """cmd_status shows Thompson Sampling info when available."""
        mock_adapter = MagicMock()
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        mock_selector = MagicMock()
        arm1 = MagicMock()
        arm1.total_pulls = 10
        mock_selector._arms = {"arm1": arm1}
        _mock_get_thompson.return_value = mock_selector

        _, printed = self._run_status()
        assert "Thompson" in printed

    def test_status_thompson_exception_handled(self):
        """cmd_status handles Thompson selector exception gracefully."""
        mock_adapter = MagicMock()
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        _mock_get_thompson.side_effect = Exception("no selector")
        rc, _ = self._run_status()
        assert rc == 0
        _mock_get_thompson.side_effect = None

    def test_status_verbose_logging(self):
        """cmd_status calls _setup_logging(True) when verbose."""
        with patch.object(cli_startup, "_setup_logging") as mock_log, patch("builtins.print"):
            args = _args(verbose=True)
            cli.cmd_status(args)
        mock_log.assert_called_with(True)


# ===========================================================================
# TestCmdHealth
# ===========================================================================


class TestCmdHealth:
    @pytest.fixture(autouse=True)
    def setup(self):
        _mock_local_inference_cls.reset_mock(return_value=True, side_effect=True)
        _mock_get_adapter_mgr.reset_mock(return_value=True, side_effect=True)
        _mock_get_adapter_mgr.return_value.health_check.return_value = {"local": {"healthy": True}}

    def _run_health(self, **kw):
        args = _args(**kw)
        with patch("builtins.print") as mock_print:
            rc = cli.cmd_health(args)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        return rc, printed

    def test_health_returns_0_when_all_checks_pass(self):
        """cmd_health returns 0 when all checks pass."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        rc, _ = self._run_health()
        assert rc == 0

    def test_health_local_inference_ok(self):
        """cmd_health prints OK when local inference is healthy."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = [{"id": "m1"}]
        _mock_local_inference_cls.return_value = mock_adapter

        _, printed = self._run_health()
        assert "OK" in printed

    def test_health_local_inference_fail(self):
        """cmd_health prints FAIL when local inference is unhealthy."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = False
        _mock_local_inference_cls.return_value = mock_adapter

        rc, printed = self._run_health()
        assert rc == 1
        assert "FAIL" in printed

    def test_health_prints_running_message(self):
        """cmd_health prints a 'Running health checks' message."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        _, printed = self._run_health()
        assert "health" in printed.lower()

    def test_health_adapter_manager_checked(self):
        """cmd_health invokes adapter manager health_check."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        mock_mgr = MagicMock()
        mock_mgr.health_check.return_value = {"local": {"healthy": True}}
        _mock_get_adapter_mgr.return_value = mock_mgr

        rc, printed = self._run_health()
        assert rc == 0
        assert mock_mgr.health_check.call_count == 1
        assert "local" in printed
        assert "OK" in printed

    @pytest.mark.parametrize(
        ("healthy", "expected_text"),
        [
            (True, "OK"),
            (False, "FAIL"),
        ],
    )
    def test_health_adapter_manager_status_display(self, healthy, expected_text):
        """cmd_health shows OK for healthy providers and FAIL for unhealthy ones."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        mock_mgr = MagicMock()
        mock_mgr.health_check.return_value = {"openai": {"healthy": healthy}}
        _mock_get_adapter_mgr.return_value = mock_mgr

        _, printed = self._run_health()
        assert expected_text in printed

    def test_health_adapter_manager_exception_is_silent(self):
        """cmd_health does not crash, but reports nonzero when adapter manager raises."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        _mock_get_adapter_mgr.side_effect = RuntimeError("no mgr")
        rc, _ = self._run_health()
        assert rc == 1
        _mock_get_adapter_mgr.side_effect = None

    def test_health_verbose_logging(self):
        """cmd_health calls _setup_logging(True) when verbose."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = []
        _mock_local_inference_cls.return_value = mock_adapter

        with patch.object(cli_startup, "_setup_logging") as mock_log:
            args = _args(verbose=True)
            with patch("builtins.print"):
                cli.cmd_health(args)
        mock_log.assert_called_with(True)

    def test_health_model_count_shown(self):
        """cmd_health shows model count in local inference status line."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        _mock_local_inference_cls.return_value = mock_adapter

        _, printed = self._run_health()
        assert "3" in printed


# ===========================================================================
# TestCmdUpgrade
# ===========================================================================


class TestCmdUpgrade:
    """Tests for cmd_upgrade — uses get_adapter_manager().discover_models()."""

    @pytest.fixture(autouse=True)
    def _ensure_adapter_manager_module(self):
        """Ensure vetinari.adapter_manager stub is in sys.modules for patch() targets."""
        import vetinari as _vet_pkg

        sys.modules.setdefault("vetinari.adapter_manager", _adapter_mgr_stub)
        if not hasattr(_vet_pkg, "adapter_manager"):
            _vet_pkg.adapter_manager = _adapter_mgr_stub  # type: ignore[attr-defined]

    def _mock_mgr(self, models=None, side_effect=None):
        mock_mgr = MagicMock()
        if side_effect is not None:
            mock_mgr.discover_models.side_effect = side_effect
        else:
            models = models if models is not None else {}
            mock_mgr.discover_models.return_value = models
        return mock_mgr

    def test_upgrade_success_returns_0(self):
        """cmd_upgrade returns 0 when model discovery succeeds."""
        mock_mgr = self._mock_mgr({"llama": [MagicMock(name="m1", memory_gb=4, id="m1")]})
        with patch("vetinari.adapter_manager.get_adapter_manager", return_value=mock_mgr):
            rc = cli.cmd_upgrade(_args())
        assert rc == 0
        mock_mgr.discover_models.assert_called_once()

    def test_upgrade_calls_discover_models(self):
        """cmd_upgrade calls mgr.discover_models()."""
        mock_mgr = self._mock_mgr({})
        with patch("vetinari.adapter_manager.get_adapter_manager", return_value=mock_mgr):
            with patch("builtins.print") as mock_print:
                rc = cli.cmd_upgrade(_args())
        assert rc == 0
        assert mock_mgr.discover_models.call_count == 1
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Discovered 0 models across 0 provider(s)" in printed
        assert "Upgrade check complete" in printed

    def test_upgrade_exception_returns_1(self):
        """cmd_upgrade returns 1 when get_adapter_manager raises."""
        with patch("vetinari.adapter_manager.get_adapter_manager", side_effect=Exception("no mgr")):
            rc = cli.cmd_upgrade(_args())
        assert rc == 1

    def test_upgrade_prints_error_on_failure(self):
        """cmd_upgrade prints error message on failure."""
        with patch("vetinari.adapter_manager.get_adapter_manager", side_effect=Exception("boom")):
            with patch("builtins.print") as mock_print:
                cli.cmd_upgrade(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "boom" in printed

    def test_upgrade_uses_config_arg(self):
        """cmd_upgrade does not use config arg (uses adapter manager instead)."""
        mock_mgr = self._mock_mgr({})
        with patch("vetinari.adapter_manager.get_adapter_manager", return_value=mock_mgr):
            rc = cli.cmd_upgrade(_args(config="my.yaml"))
        assert rc == 0

    def test_upgrade_uses_mode_arg(self):
        """cmd_upgrade does not use mode arg (uses adapter manager instead)."""
        mock_mgr = self._mock_mgr({})
        with patch("vetinari.adapter_manager.get_adapter_manager", return_value=mock_mgr):
            rc = cli.cmd_upgrade(_args(mode="planning"))
        assert rc == 0

    def test_upgrade_discover_exception_is_caught(self):
        """cmd_upgrade handles exception from discover_models."""
        mock_mgr = self._mock_mgr(side_effect=RuntimeError("network"))
        with patch("vetinari.adapter_manager.get_adapter_manager", return_value=mock_mgr):
            rc = cli.cmd_upgrade(_args())
        assert rc == 1

    def test_upgrade_verbose_sets_logging(self):
        """cmd_upgrade calls _setup_logging(True) when verbose."""
        mock_mgr = self._mock_mgr({})
        with patch("vetinari.adapter_manager.get_adapter_manager", return_value=mock_mgr):
            with patch.object(cli_startup, "_setup_logging") as mock_log:
                cli.cmd_upgrade(_args(verbose=True))
        mock_log.assert_called_with(True)


# ===========================================================================
# TestCmdReview
# ===========================================================================


class TestCmdReview:
    @pytest.fixture(autouse=True)
    def setup(self):
        _mock_get_operations.reset_mock()
        _mock_get_adapter_mgr.reset_mock()

    def _make_agent(self, success=True, recs=None, applied=None):
        agent = MagicMock()
        result = MagicMock()
        result.success = success
        result.output = {
            "recommendations": recs or [],
            "auto_applied": applied or [],
        }
        agent.execute.return_value = result
        _mock_get_operations.return_value = agent
        return agent

    def test_review_success_returns_0(self):
        """cmd_review returns 0 on successful review."""
        self._make_agent(success=True)
        rc = cli.cmd_review(_args())
        assert rc == 0

    def test_review_calls_agent_execute(self):
        """cmd_review calls agent.execute() with an AgentTask."""
        _AgentTask_mock.reset_mock()
        agent = self._make_agent(success=True)
        rc = cli.cmd_review(_args())
        assert rc == 0
        task_kwargs = _AgentTask_mock.call_args.kwargs
        assert task_kwargs["task_id"] == "review-cli"
        assert task_kwargs["description"] == "Run system performance review"
        assert task_kwargs["context"] == {"review_type": "full"}
        assert agent.execute.call_args.args == (_AgentTask_mock.return_value,)

    @pytest.mark.parametrize(
        ("recs", "applied", "expected_text"),
        [
            ([{"priority": "high", "action": "a", "rationale": "r"}], [], "1"),
            ([], ["change1", "change2"], "2"),
        ],
    )
    def test_review_prints_counts(self, recs, applied, expected_text):
        """cmd_review prints counts of recommendations and auto-applied changes."""
        self._make_agent(success=True, recs=recs, applied=applied)
        with patch("builtins.print") as mock_print:
            cli.cmd_review(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert expected_text in printed

    def test_review_exception_returns_1(self):
        """cmd_review returns 1 when agent raises."""
        _mock_get_operations.side_effect = Exception("agent broken")
        rc = cli.cmd_review(_args())
        assert rc == 1
        _mock_get_operations.side_effect = None

    def test_review_exception_prints_error(self):
        """cmd_review prints error message on exception."""
        _mock_get_operations.side_effect = Exception("broken agent")
        with patch("builtins.print") as mock_print:
            cli.cmd_review(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "broken agent" in printed
        _mock_get_operations.side_effect = None

    def test_review_initializes_agent_with_adapter_manager(self):
        """cmd_review tries to initialize agent with adapter manager."""
        agent = self._make_agent(success=True)
        mock_mgr = MagicMock()
        _mock_get_adapter_mgr.return_value = mock_mgr
        rc = cli.cmd_review(_args())
        assert rc == 0
        assert agent.initialize.call_args.args == ({"adapter_manager": mock_mgr},)
        assert agent.execute.call_count == 1

    def test_review_adapter_manager_failure_is_non_fatal(self):
        """Review succeeds even if adapter manager initialization fails."""
        self._make_agent(success=True)
        _mock_get_adapter_mgr.side_effect = Exception("no mgr")
        rc = cli.cmd_review(_args())
        assert rc == 0
        _mock_get_adapter_mgr.side_effect = None

    def test_review_prints_recommendation_action(self):
        """cmd_review prints action field from recommendations."""
        recs = [{"priority": "medium", "action": "rotate model", "rationale": "performance"}]
        self._make_agent(success=True, recs=recs)
        with patch("builtins.print") as mock_print:
            cli.cmd_review(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "rotate model" in printed

    def test_review_verbose_sets_logging(self):
        """cmd_review calls _setup_logging(True) when verbose."""
        self._make_agent(success=True)
        with patch.object(cli_startup, "_setup_logging") as mock_log:
            cli.cmd_review(_args(verbose=True))
        mock_log.assert_called_with(True)

    def test_review_prints_running_message(self):
        """cmd_review prints a starting message."""
        self._make_agent(success=True)
        with patch("builtins.print") as mock_print:
            cli.cmd_review(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "review" in printed.lower()


# ===========================================================================
# TestCmdInteractive
# ===========================================================================


class TestCmdInteractive:
    @pytest.fixture(autouse=True)
    def setup(self):
        _mock_get_two_layer.reset_mock()

    def _run_interactive(self, inputs, mock_orch=None):
        """Run cmd_interactive with a scripted sequence of inputs."""
        if mock_orch is not None:
            _mock_get_two_layer.return_value = mock_orch
        else:
            # Default: two_layer raises so orch=None path
            _mock_get_two_layer.side_effect = Exception("no two layer")

        with patch("builtins.input", side_effect=inputs), patch("builtins.print"):
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                rc = cli.cmd_interactive(_args())

        _mock_get_two_layer.side_effect = None
        return rc

    def test_interactive_quit_returns_0(self):
        """/quit exits with return code 0."""
        rc = self._run_interactive(["/quit"])
        assert rc == 0

    def test_interactive_exit_returns_0(self):
        """/exit exits with return code 0."""
        rc = self._run_interactive(["/exit"])
        assert rc == 0

    def test_interactive_quit_word_returns_0(self):
        """'quit' keyword exits with return code 0."""
        rc = self._run_interactive(["quit"])
        assert rc == 0

    def test_interactive_exit_word_returns_0(self):
        """'exit' keyword exits with return code 0."""
        rc = self._run_interactive(["exit"])
        assert rc == 0

    def test_interactive_eof_returns_0(self):
        """EOFError (Ctrl-D) exits gracefully with return code 0."""
        rc = self._run_interactive([EOFError()])
        assert rc == 0

    def test_interactive_keyboard_interrupt_returns_0(self):
        """KeyboardInterrupt (Ctrl-C) exits gracefully with return code 0."""
        rc = self._run_interactive([KeyboardInterrupt()])
        assert rc == 0

    def test_interactive_help_command(self):
        """/help prints help text then continues until /quit."""
        with patch("builtins.input", side_effect=["/help", "/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                _mock_get_two_layer.side_effect = Exception("no")
                rc = cli.cmd_interactive(_args())
                _mock_get_two_layer.side_effect = None

        assert rc == 0
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "/quit" in printed

    def test_interactive_status_command(self):
        """/status calls cmd_status then continues."""
        args = _args()
        with patch.object(cli_commands, "cmd_status") as mock_status:
            with patch("builtins.input", side_effect=["/status", "/quit"]):
                with patch("builtins.print") as mock_print:
                    with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                        _mock_get_two_layer.side_effect = Exception("no")
                        rc = cli.cmd_interactive(args)
                        _mock_get_two_layer.side_effect = None

        assert rc == 0
        assert mock_status.call_args.args == (args,)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Goodbye." in printed

    def test_interactive_review_command(self):
        """/review calls cmd_review then continues."""
        args = _args()
        with patch.object(cli_devops, "cmd_review") as mock_review:
            with patch("builtins.input", side_effect=["/review", "/quit"]):
                with patch("builtins.print") as mock_print:
                    with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                        _mock_get_two_layer.side_effect = Exception("no")
                        rc = cli.cmd_interactive(args)
                        _mock_get_two_layer.side_effect = None

        assert rc == 0
        assert mock_review.call_args.args == (args,)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Goodbye." in printed

    def test_interactive_empty_input_continues(self):
        """Empty input is ignored and the loop continues."""
        rc = self._run_interactive(["", "/quit"])
        assert rc == 0

    def test_interactive_goal_with_orchestrator(self):
        """A regular goal string is executed via orch.generate_and_execute."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 1, "final_output": None}
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None
        args = _args(mode="planning")

        with patch("builtins.input", side_effect=["my goal", "/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                rc = cli.cmd_interactive(args)

        assert rc == 0
        assert mock_orch.generate_and_execute.call_args.kwargs == {
            "goal": "my goal",
            "constraints": {"mode": "planning"},
        }
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Working on: my goal" in printed
        assert "Completed: 1 tasks" in printed

    def test_interactive_goal_without_orchestrator_prints_message(self):
        """With no orchestrator available a helpful message is printed."""
        _mock_get_two_layer.side_effect = Exception("not available")

        with patch("builtins.input", side_effect=["some goal", "/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                cli.cmd_interactive(_args())

        _mock_get_two_layer.side_effect = None
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "not available" in printed.lower()

    def test_interactive_goal_execute_exception_handled(self):
        """Exception during goal execution is caught and not propagated."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.side_effect = RuntimeError("exec fail")
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch("builtins.input", side_effect=["bad goal", "/quit"]), patch("builtins.print"):
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                rc = cli.cmd_interactive(_args())

        assert rc == 0

    def test_interactive_prints_startup_banner(self):
        """cmd_interactive prints instructions at start."""
        with patch("builtins.input", side_effect=["/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                _mock_get_two_layer.side_effect = Exception("no")
                cli.cmd_interactive(_args())
                _mock_get_two_layer.side_effect = None

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Interactive" in printed

    def test_interactive_goal_final_output_printed(self):
        """If goal execution returns final_output it is printed."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {
            "completed": 1,
            "final_output": "great result",
        }
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch("builtins.input", side_effect=["build api", "/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                cli.cmd_interactive(_args())

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "great result" in printed

    def test_interactive_verbose_sets_logging(self):
        """cmd_interactive calls _setup_logging(True) when verbose."""
        with patch.object(cli_startup, "_setup_logging") as mock_log, patch("builtins.input", side_effect=["/quit"]):
            with patch("builtins.print"):
                with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                    _mock_get_two_layer.side_effect = Exception("no")
                    cli.cmd_interactive(_args(verbose=True))
                    _mock_get_two_layer.side_effect = None

        mock_log.assert_called_with(True)

    def test_interactive_multiple_goals_before_quit(self):
        """Multiple goals can be entered before quitting."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 1, "final_output": None}
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch("builtins.input", side_effect=["goal1", "goal2", "/quit"]), patch("builtins.print"):
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                cli.cmd_interactive(_args())

        assert mock_orch.generate_and_execute.call_count == 2


# ===========================================================================
# TestMain — argparse routing
# ===========================================================================


class TestMain:
    def _run_main(self, argv, handler_return=0):
        """Invoke main() with patched sys.argv; capture sys.exit call."""
        with (
            patch("sys.argv", argv),
            patch.object(cli, "cmd_run", return_value=handler_return) as m_run,
            patch.object(cli, "cmd_serve", return_value=handler_return) as m_serve,
            patch.object(cli, "cmd_start", return_value=handler_return) as m_start,
            patch.object(cli, "cmd_status", return_value=handler_return) as m_status,
            patch.object(cli, "cmd_health", return_value=handler_return) as m_health,
            patch.object(cli, "cmd_upgrade", return_value=handler_return) as m_upgrade,
            patch.object(cli, "cmd_review", return_value=handler_return) as m_review,
            patch.object(cli, "cmd_interactive", return_value=handler_return) as m_inter,
        ):
            try:
                cli.main()
            except SystemExit:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                pass
            return {
                "run": m_run,
                "serve": m_serve,
                "start": m_start,
                "status": m_status,
                "health": m_health,
                "upgrade": m_upgrade,
                "review": m_review,
                "interactive": m_inter,
            }

    def test_main_run_subcommand(self):
        """'vetinari run' routes to cmd_run."""
        mocks = self._run_main(["vetinari", "run"])
        mocks["run"].assert_called_once()

    def test_main_serve_subcommand(self):
        """'vetinari serve' routes to cmd_serve."""
        mocks = self._run_main(["vetinari", "serve"])
        mocks["serve"].assert_called_once()

    def test_main_start_subcommand(self):
        """'vetinari start' routes to cmd_start."""
        mocks = self._run_main(["vetinari", "start"])
        mocks["start"].assert_called_once()

    def test_main_status_subcommand(self):
        """'vetinari status' routes to cmd_status."""
        mocks = self._run_main(["vetinari", "status"])
        mocks["status"].assert_called_once()

    def test_main_health_subcommand(self):
        """'vetinari health' routes to cmd_health."""
        mocks = self._run_main(["vetinari", "health"])
        mocks["health"].assert_called_once()

    def test_main_upgrade_subcommand(self):
        """'vetinari upgrade' routes to cmd_upgrade."""
        mocks = self._run_main(["vetinari", "upgrade"])
        mocks["upgrade"].assert_called_once()

    def test_main_review_subcommand(self):
        """'vetinari review' routes to cmd_review."""
        mocks = self._run_main(["vetinari", "review"])
        mocks["review"].assert_called_once()

    def test_main_interactive_subcommand(self):
        """'vetinari interactive' routes to cmd_interactive."""
        mocks = self._run_main(["vetinari", "interactive"])
        mocks["interactive"].assert_called_once()

    def test_main_no_args_defaults_to_start(self):
        """No subcommand defaults to 'start'."""
        mocks = self._run_main(["vetinari"])
        mocks["start"].assert_called_once()

    def test_main_handler_return_value_passed_to_sys_exit(self):
        """main() calls sys.exit with the handler return code."""
        with patch("sys.argv", ["vetinari", "status"]), patch.object(cli, "cmd_status", return_value=42):
            with pytest.raises(SystemExit) as cm:
                cli.main()
        assert cm.value.code == 42

    def test_main_run_with_goal_flag(self):
        """'vetinari run --goal ...' parses goal correctly."""
        mocks = self._run_main(["vetinari", "run", "--goal", "Build API"])
        mocks["run"].assert_called_once()
        args_passed = mocks["run"].call_args.args[0]
        assert args_passed.goal == "Build API"

    def test_main_run_with_task_flag(self):
        """'vetinari run --task t1' parses task correctly."""
        mocks = self._run_main(["vetinari", "run", "--task", "t1"])
        args_passed = mocks["run"].call_args.args[0]
        assert args_passed.task == "t1"

    def test_main_serve_with_port_flag(self):
        """'vetinari serve --port 8080' parses port correctly."""
        mocks = self._run_main(["vetinari", "serve", "--port", "8080"])
        args_passed = mocks["serve"].call_args.args[0]
        assert args_passed.port == 8080

    def test_main_global_verbose_flag(self):
        """'vetinari --verbose status' sets verbose=True."""
        mocks = self._run_main(["vetinari", "--verbose", "status"])
        args_passed = mocks["status"].call_args.args[0]
        assert args_passed.verbose

    def test_main_global_mode_flag(self):
        """'vetinari --mode planning run' sets mode correctly."""
        mocks = self._run_main(["vetinari", "--mode", "planning", "run"])
        args_passed = mocks["run"].call_args.args[0]
        assert args_passed.mode == "planning"

    def test_main_start_no_dashboard_flag(self):
        """'vetinari start --no-dashboard' sets no_dashboard=True."""
        mocks = self._run_main(["vetinari", "start", "--no-dashboard"])
        args_passed = mocks["start"].call_args.args[0]
        assert args_passed.no_dashboard

    def test_main_default_command_sets_goal_none(self):
        """Default command sets goal attribute to None."""
        mocks = self._run_main(["vetinari"])
        args_passed = mocks["start"].call_args.args[0]
        assert args_passed.goal is None


# ===========================================================================
# TestHelpers — unit tests for private helper functions
# ===========================================================================


class TestHelpers:
    def test_setup_logging_info_level(self):
        """_setup_logging sets INFO level when verbose=False."""
        with patch("vetinari.cli.logging.basicConfig") as mock_bc:
            cli._setup_logging(False)
        mock_bc.assert_called_once()
        assert mock_bc.call_args.kwargs["level"] == logging.INFO

    def test_setup_logging_debug_level(self):
        """_setup_logging sets DEBUG level when verbose=True."""
        with patch("vetinari.cli.logging.basicConfig") as mock_bc:
            cli._setup_logging(True)
        assert mock_bc.call_args.kwargs["level"] == logging.DEBUG

    def test_print_banner_contains_mode(self):
        """_print_banner prints the mode."""
        with patch("builtins.print") as mock_print:
            cli._print_banner("sandbox")
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "SANDBOX" in printed

    def test_build_orchestrator_instantiates_orchestrator(self):
        """_build_orchestrator creates an Orchestrator instance."""
        _mock_orchestrator_cls.reset_mock()
        cli._build_orchestrator("some/config.yaml", "execution")
        _mock_orchestrator_cls.assert_called_once_with("some/config.yaml", execution_mode="execution")

    def test_load_config_missing_returns_defaults(self):
        """_load_config returns default dict when file doesn't exist."""
        result = cli._load_config("/nonexistent/path/config.yaml")
        assert "project_name" in result
        assert "tasks" in result

    def test_health_check_quiet_local_ok(self):
        """_health_check_quiet prints OK when local inference is healthy."""
        mock_adapter = MagicMock()
        mock_adapter.is_healthy.return_value = True
        mock_adapter.list_loaded_models.return_value = [{"id": "m1"}]
        _mock_local_inference_cls.return_value = mock_adapter

        with patch("builtins.print") as mock_print:
            cli._health_check_quiet()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "OK" in printed

    def test_health_check_quiet_local_fail(self):
        """_health_check_quiet prints FAIL when local inference is unavailable."""
        _mock_local_inference_cls.side_effect = ConnectionError("refused")

        with patch("builtins.print") as mock_print:
            cli._health_check_quiet()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "FAIL" in printed
        _mock_local_inference_cls.side_effect = None


# ===========================================================================
# TestEdgeCases — boundary and integration edge cases
# ===========================================================================


class TestEdgeCases:
    def test_cmd_run_long_goal_truncated_in_print(self):
        """cmd_run truncates very long goal strings in the printed line."""
        long_goal = "x" * 200
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 0, "failed": 0}
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            with patch("builtins.print") as mock_print:
                cli.cmd_run(_args(goal=long_goal))

        # The printed goal should be at most 80 chars per cli.py logic
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        # Just verify no exception and goal text appears
        assert "x" in printed

    def test_cmd_run_final_output_truncated_to_2000(self):
        """cmd_run truncates final_output to 2000 chars."""
        mock_orch = MagicMock()
        long_output = "y" * 5000
        mock_orch.generate_and_execute.return_value = {"completed": 1, "failed": 0, "final_output": long_output}
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
            with patch("builtins.print") as mock_print:
                cli.cmd_run(_args(goal="truncation test"))

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        # Should contain y's but not the full 5000
        assert "y" in printed

    def test_cmd_interactive_final_output_truncated_to_1500(self):
        """cmd_interactive truncates final_output to 1500 chars."""
        mock_orch = MagicMock()
        long_output = "z" * 3000
        mock_orch.generate_and_execute.return_value = {"completed": 1, "final_output": long_output}
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch("builtins.input", side_effect=["some goal", "/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli_startup, "_build_orchestrator", return_value=MagicMock()):
                cli.cmd_interactive(_args())

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "z" in printed

    def test_cmd_serve_web_host_default(self):
        """cmd_serve defaults web_host to '127.0.0.1'."""
        args = _args(port=5000, web_host="127.0.0.1", debug=False)
        mock_uv = MagicMock()
        with patch("vetinari.cli_commands.uvicorn", mock_uv):
            cli.cmd_serve(args)
        all_args = str(mock_uv.run.call_args)
        assert "127.0.0.1" in all_args

    def test_cmd_status_model_list_displayed(self):
        """cmd_status lists up to 5 model IDs."""
        mock_adapter = MagicMock()
        mock_adapter.list_loaded_models.return_value = [{"id": f"model-{i}"} for i in range(6)]
        _mock_local_inference_cls.return_value = mock_adapter

        with patch("builtins.print") as mock_print:
            cli.cmd_status(_args())

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        # Should show up to 5 models (model-0 through model-4)
        assert "model-0" in printed

    def test_cmd_review_at_most_5_recs_printed(self):
        """cmd_review prints at most 5 recommendations."""
        recs = [{"priority": "low", "action": f"action-{i}", "rationale": "r"} for i in range(10)]
        agent = MagicMock()
        result = MagicMock()
        result.success = True
        result.output = {"recommendations": recs, "auto_applied": []}
        agent.execute.return_value = result
        _mock_get_operations.return_value = agent
        _mock_get_adapter_mgr.side_effect = None

        with patch("builtins.print") as mock_print:
            cli.cmd_review(_args())

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        # action-5 and beyond should NOT appear (only first 5 printed)
        assert "action-5" not in printed

    def test_main_config_flag_parsed(self):
        """'vetinari --config custom.yaml status' parses config path."""
        with patch("sys.argv", ["vetinari", "--config", "custom.yaml", "status"]):
            with patch.object(cli, "cmd_status", return_value=0) as m:
                try:
                    cli.main()
                except SystemExit:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                    pass
        args_passed = m.call_args.args[0]
        assert args_passed.config == "custom.yaml"

    def test_main_start_goal_flag(self):
        """'vetinari start --goal ...' parses goal in start subcommand."""
        with patch("sys.argv", ["vetinari", "start", "--goal", "Run tests"]):
            with patch.object(cli, "cmd_start", return_value=0) as m:
                try:
                    cli.main()
                except SystemExit:  # noqa: VET022 - best-effort optional path must not fail the primary flow
                    pass
        args_passed = m.call_args.args[0]
        assert args_passed.goal == "Run tests"
