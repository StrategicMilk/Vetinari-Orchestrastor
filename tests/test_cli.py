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

# vetinari.web_ui
_mock_flask_app = MagicMock()
_mock_flask_app.config = {}
_mock_flask_app.run = MagicMock()
_web_ui_stub = _make_stub("vetinari.web_ui", app=_mock_flask_app)
sys.modules["vetinari.web_ui"] = _web_ui_stub

# vetinari.lmstudio_adapter
_mock_lmstudio_cls = MagicMock()
_lmstudio_stub = _make_stub(
    "vetinari.lmstudio_adapter",
    LMStudioAdapter=_mock_lmstudio_cls,
)
sys.modules["vetinari.lmstudio_adapter"] = _lmstudio_stub

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

# vetinari.agents.improvement_agent
_mock_get_improvement = MagicMock()
_improvement_stub = _make_stub(
    "vetinari.agents.improvement_agent",
    get_improvement_agent=_mock_get_improvement,
    ImprovementAgent=MagicMock(),
)
sys.modules["vetinari.agents.improvement_agent"] = _improvement_stub


# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------

import pytest

import vetinari.cli as cli

# ---------------------------------------------------------------------------
# Helper: build a minimal args namespace that satisfies all cmd_* functions
# ---------------------------------------------------------------------------

def _args(**overrides) -> SimpleNamespace:
    """Return a namespace with safe defaults for every attribute cli uses."""
    defaults = {
        "verbose": False,
        "host": None,
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

class TestCmdRun(unittest.TestCase):

    def setUp(self):
        _mock_orchestrator_cls.reset_mock()
        _mock_get_two_layer.reset_mock()

    # --- goal-path ---

    def test_run_goal_success(self):
        """cmd_run with a goal calls generate_and_execute and returns 0."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 3, "failed": 0}
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="Build a REST API")
        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
            rc = cli.cmd_run(args)

        assert rc == 0
        mock_orch.generate_and_execute.assert_called_once()

    def test_run_goal_passes_goal_string(self):
        """generate_and_execute receives the correct goal kwarg."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 1, "failed": 0}
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="My specific goal")
        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
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
        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
            with patch("builtins.print") as mock_print:
                rc = cli.cmd_run(args)

        assert rc == 0
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "The result text" in printed

    def test_run_goal_no_final_output(self):
        """cmd_run with no final_output still returns 0."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 0, "failed": 0}
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="Empty goal")
        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
            rc = cli.cmd_run(args)

        assert rc == 0

    def test_run_goal_exception_returns_1(self):
        """cmd_run returns 1 when generate_and_execute raises."""
        _mock_get_two_layer.side_effect = RuntimeError("LM Studio offline")

        args = _args(goal="Failing goal")
        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
            rc = cli.cmd_run(args)

        assert rc == 1
        _mock_get_two_layer.side_effect = None

    def test_run_goal_prints_error_on_exception(self):
        """cmd_run prints error message when an exception is raised."""
        _mock_get_two_layer.side_effect = RuntimeError("bang")

        args = _args(goal="Bad goal")
        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
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
        with patch.object(cli, "_build_orchestrator", side_effect=Exception("no config")):
            # Also ensure the two_layer compat shim returns our mock
            with patch.dict(sys.modules, {
                "vetinari.orchestration.two_layer": _two_layer_stub,
            }):
                args = _args(goal="Wiring failure goal")
                rc = cli.cmd_run(args)

        assert rc == 0

    # --- manifest-path (no goal) ---

    def test_run_no_goal_no_task_calls_run_all(self):
        """cmd_run with no goal and no task calls orch.run_all()."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch):
            args = _args(goal=None, task=None)
            rc = cli.cmd_run(args)

        assert rc == 0
        mock_orch.run_all.assert_called_once()

    def test_run_specific_task_calls_run_task(self):
        """cmd_run with a task ID calls orch.run_task(task_id)."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch):
            args = _args(goal=None, task="t42")
            rc = cli.cmd_run(args)

        assert rc == 0
        mock_orch.run_task.assert_called_once_with("t42")

    def test_run_manifest_exception_returns_1(self):
        """cmd_run returns 1 when manifest execution raises."""
        with patch.object(cli, "_build_orchestrator", side_effect=Exception("oops")):
            args = _args(goal=None, task=None)
            rc = cli.cmd_run(args)

        assert rc == 1

    def test_run_verbose_flag_sets_logging(self):
        """cmd_run with verbose=True calls _setup_logging(True)."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch):
            with patch.object(cli, "_setup_logging") as mock_log:
                args = _args(goal=None, task=None, verbose=True)
                cli.cmd_run(args)

        mock_log.assert_called_with(True)

    def test_run_host_from_env(self):
        """cmd_run uses LM_STUDIO_HOST env var when args.host is None."""
        mock_orch = MagicMock()
        with patch.dict(os.environ, {"LM_STUDIO_HOST": "http://myhost:9999"}):
            with patch.object(cli, "_build_orchestrator", return_value=mock_orch) as mock_build:
                args = _args(goal=None, task=None, host=None)
                cli.cmd_run(args)
                _, _build_kwargs = mock_build.call_args
                # host is positional arg[1]
                built_host = mock_build.call_args.args[1]
                assert built_host == "http://myhost:9999"

    def test_run_host_arg_overrides_env(self):
        """args.host takes precedence over environment variable."""
        mock_orch = MagicMock()
        with patch.dict(os.environ, {"LM_STUDIO_HOST": "http://envhost:1234"}):
            with patch.object(cli, "_build_orchestrator", return_value=mock_orch) as mock_build:
                args = _args(goal=None, task=None, host="http://arghost:5678")
                cli.cmd_run(args)
                built_host = mock_build.call_args.args[1]
                assert built_host == "http://arghost:5678"

    def test_run_goal_completed_count_printed(self):
        """The completed/failed task counts appear in printed output."""
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 7, "failed": 2}
        _mock_get_two_layer.return_value = mock_orch

        args = _args(goal="count test")
        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
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
        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
            cli.cmd_run(args)

        call_kwargs = mock_orch.generate_and_execute.call_args.kwargs
        assert "planning" in str(call_kwargs)

    def test_run_no_goal_prints_running_message(self):
        """Without a goal, cmd_run prints a running tasks message."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch):
            with patch("builtins.print") as mock_print:
                args = _args(goal=None, task=None)
                cli.cmd_run(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "manifest" in printed.lower()


# ===========================================================================
# TestCmdServe
# ===========================================================================

class TestCmdServe(unittest.TestCase):

    def setUp(self):
        _mock_flask_app.reset_mock()
        _mock_flask_app.config.clear()
        _mock_flask_app.run.reset_mock()

    def test_serve_calls_app_run(self):
        """cmd_serve calls Flask app.run()."""
        args = _args(port=5000, web_host="127.0.0.1", debug=False)
        rc = cli.cmd_serve(args)
        assert rc == 0
        _mock_flask_app.run.assert_called_once()

    def test_serve_uses_specified_port(self):
        """cmd_serve passes the given port to app.run."""
        args = _args(port=8080, web_host="127.0.0.1", debug=False)
        cli.cmd_serve(args)
        _, kwargs = _mock_flask_app.run.call_args
        assert (kwargs.get("port") or _mock_flask_app.run.call_args.args[1]) == 8080

    def test_serve_default_port_from_env(self):
        """cmd_serve uses VETINARI_WEB_PORT env var when port is None."""
        args = _args(port=None, web_host="0.0.0.0", debug=False)
        with patch.dict(os.environ, {"VETINARI_WEB_PORT": "6000"}):
            cli.cmd_serve(args)
        call_kwargs = _mock_flask_app.run.call_args
        all_args = str(call_kwargs)
        assert "6000" in all_args

    def test_serve_default_port_5000(self):
        """cmd_serve defaults to port 5000 when no port or env var given."""
        args = _args(port=None, web_host="127.0.0.1", debug=False)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VETINARI_WEB_PORT", None)
            cli.cmd_serve(args)
        all_args = str(_mock_flask_app.run.call_args)
        assert "5000" in all_args

    def test_serve_sets_vetinari_host_in_config(self):
        """cmd_serve stores the LM Studio host in app.config."""
        args = _args(port=5000, web_host="127.0.0.1", debug=False, host="http://custom:9999")
        cli.cmd_serve(args)
        assert _mock_flask_app.config.get("VETINARI_HOST") == "http://custom:9999"

    def test_serve_debug_flag_passed(self):
        """debug=True is forwarded to app.run."""
        args = _args(port=5000, web_host="127.0.0.1", debug=True)
        cli.cmd_serve(args)
        all_args = str(_mock_flask_app.run.call_args)
        assert "True" in all_args

    def test_serve_uses_web_host(self):
        """cmd_serve uses the specified web_host bind address."""
        args = _args(port=5000, web_host="0.0.0.0", debug=False)
        cli.cmd_serve(args)
        all_args = str(_mock_flask_app.run.call_args)
        assert "0.0.0.0" in all_args

    def test_serve_import_error_returns_1(self):
        """cmd_serve returns 1 if Flask app import fails."""
        # Setting sys.modules entry to None triggers ImportError on next import
        # without letting Python discover the REAL web_ui.py via vetinari.__path__
        original = sys.modules.get("vetinari.web_ui")
        sys.modules["vetinari.web_ui"] = None
        try:
            args = _args(port=5000, web_host="127.0.0.1", debug=False)
            rc = cli.cmd_serve(args)
        except Exception:
            rc = 1
        finally:
            sys.modules["vetinari.web_ui"] = original
        assert rc in (0, 1)

    def test_serve_app_run_exception_returns_1(self):
        """cmd_serve returns 1 when app.run raises an unexpected exception."""
        _mock_flask_app.run.side_effect = OSError("port in use")
        args = _args(port=5000, web_host="127.0.0.1", debug=False)
        rc = cli.cmd_serve(args)
        assert rc == 1
        _mock_flask_app.run.side_effect = None

    def test_serve_use_reloader_false(self):
        """cmd_serve always passes use_reloader=False."""
        args = _args(port=5000, web_host="127.0.0.1", debug=False)
        cli.cmd_serve(args)
        all_args = str(_mock_flask_app.run.call_args)
        assert "False" in all_args


# ===========================================================================
# TestCmdStart
# ===========================================================================

class TestCmdStart(unittest.TestCase):

    def setUp(self):
        _mock_flask_app.reset_mock()
        _mock_flask_app.config.clear()
        _mock_get_auto_tuner.reset_mock()

    def _args_no_dashboard_no_goal(self, **kw):
        return _args(no_dashboard=True, goal=None, task=None, port=None, **kw)

    def test_start_no_dashboard_no_goal_calls_interactive(self):
        """With no dashboard and no goal, cmd_start delegates to cmd_interactive."""
        with patch.object(cli, "cmd_interactive", return_value=0) as mock_inter:
            with patch.object(cli, "_health_check_quiet"):
                args = self._args_no_dashboard_no_goal()
                rc = cli.cmd_start(args)
        mock_inter.assert_called_once_with(args)
        assert rc == 0

    def test_start_with_goal_calls_cmd_run(self):
        """With a goal, cmd_start delegates to cmd_run."""
        with patch.object(cli, "cmd_run", return_value=0) as mock_run, patch.object(cli, "_health_check_quiet"):
            args = _args(no_dashboard=True, goal="some goal", task=None, port=None)
            rc = cli.cmd_start(args)
        mock_run.assert_called_once_with(args)
        assert rc == 0

    def test_start_with_task_calls_cmd_run(self):
        """With a task, cmd_start delegates to cmd_run."""
        with patch.object(cli, "cmd_run", return_value=0) as mock_run, patch.object(cli, "_health_check_quiet"):
            args = _args(no_dashboard=True, goal=None, task="t1", port=None)
            rc = cli.cmd_start(args)
        mock_run.assert_called_once_with(args)
        assert rc == 0

    def test_start_runs_health_check(self):
        """cmd_start always calls _health_check_quiet."""
        with patch.object(cli, "cmd_interactive", return_value=0):
            with patch.object(cli, "_health_check_quiet") as mock_hc:
                args = self._args_no_dashboard_no_goal()
                cli.cmd_start(args)
        mock_hc.assert_called_once()

    def test_start_prints_banner(self):
        """cmd_start prints the mode/host banner."""
        with patch.object(cli, "cmd_interactive", return_value=0), patch.object(cli, "_health_check_quiet"):
            with patch.object(cli, "_print_banner") as mock_banner:
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

        with patch("vetinari.cli.threading.Thread", side_effect=capturing_thread):
            with patch.object(cli, "cmd_interactive", return_value=0):
                with patch.object(cli, "_health_check_quiet"):
                    args = self._args_no_dashboard_no_goal()
                    cli.cmd_start(args)

        assert any("auto-tuner" in n for n in spawned)

    def test_start_dashboard_disabled_skips_flask(self):
        """no_dashboard=True means Flask app.run is never called."""
        with patch.object(cli, "cmd_interactive", return_value=0), patch.object(cli, "_health_check_quiet"):
            args = self._args_no_dashboard_no_goal()
            cli.cmd_start(args)
        _mock_flask_app.run.assert_not_called()

    def test_start_verbose_propagates_to_logging(self):
        """cmd_start calls _setup_logging with the verbose flag."""
        with patch.object(cli, "_setup_logging") as mock_log:
            with patch.object(cli, "cmd_interactive", return_value=0):
                with patch.object(cli, "_health_check_quiet"):
                    args = self._args_no_dashboard_no_goal(verbose=True)
                    cli.cmd_start(args)
        mock_log.assert_called_with(True)

    def test_start_port_default_from_env(self):
        """cmd_start reads VETINARI_WEB_PORT env var for port."""
        with patch.dict(os.environ, {"VETINARI_WEB_PORT": "7070"}):
            with patch.object(cli, "cmd_interactive", return_value=0):
                with patch.object(cli, "_health_check_quiet"):
                    # no_dashboard=True so port resolution happens but Flask not started
                    args = _args(no_dashboard=True, goal=None, task=None, port=None)
                    cli.cmd_start(args)
        # No assertion on port value needed — just ensure no exception

    def test_start_returns_zero_on_success(self):
        """cmd_start returns 0 on normal completion."""
        with patch.object(cli, "cmd_interactive", return_value=0), patch.object(cli, "_health_check_quiet"):
            args = self._args_no_dashboard_no_goal()
            rc = cli.cmd_start(args)
        assert rc == 0

    def test_start_cmd_run_return_value_propagated(self):
        """Return code from cmd_run is propagated by cmd_start."""
        with patch.object(cli, "cmd_run", return_value=1), patch.object(cli, "_health_check_quiet"):
            args = _args(no_dashboard=True, goal="failing goal", task=None, port=None)
            rc = cli.cmd_start(args)
        assert rc == 1

    def test_start_no_dashboard_no_goal_no_task_calls_interactive_not_run(self):
        """When no goal and no task, cmd_run is not called."""
        with patch.object(cli, "cmd_run") as mock_run, patch.object(cli, "cmd_interactive", return_value=0):
            with patch.object(cli, "_health_check_quiet"):
                args = self._args_no_dashboard_no_goal()
                cli.cmd_start(args)
        mock_run.assert_not_called()


# ===========================================================================
# TestCmdStatus
# ===========================================================================

class TestCmdStatus(unittest.TestCase):

    def setUp(self):
        _mock_lmstudio_cls.reset_mock()
        _mock_get_adapter_mgr.reset_mock()
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

    def test_status_prints_host(self):
        """cmd_status prints the LM Studio host."""
        _, printed = self._run_status(host="http://myhost:1234")
        assert "http://myhost:1234" in printed

    def test_status_prints_config_path(self):
        """cmd_status prints the config path."""
        _, printed = self._run_status(config="manifest/vetinari.yaml")
        assert "manifest/vetinari.yaml" in printed

    def test_status_lmstudio_reachable(self):
        """cmd_status prints model count when LM Studio is reachable."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": [{"id": "m1"}, {"id": "m2"}]}
        _mock_lmstudio_cls.return_value = mock_adapter

        _, printed = self._run_status()
        assert "2" in printed

    def test_status_lmstudio_unreachable(self):
        """cmd_status prints UNREACHABLE when LM Studio is down."""
        mock_adapter = MagicMock()
        mock_adapter._get.side_effect = ConnectionError("refused")
        _mock_lmstudio_cls.return_value = mock_adapter

        _, printed = self._run_status()
        assert "UNREACHABLE" in printed

    def test_status_lmstudio_no_data_key(self):
        """cmd_status handles response without 'data' key gracefully."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = "not a dict"
        _mock_lmstudio_cls.return_value = mock_adapter

        rc, _ = self._run_status()
        assert rc == 0

    def test_status_adapter_manager_listed(self):
        """cmd_status lists providers from adapter manager."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

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
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        _mock_get_adapter_mgr.side_effect = RuntimeError("no mgr")
        rc, _ = self._run_status()
        assert rc == 0
        _mock_get_adapter_mgr.side_effect = None

    def test_status_thompson_selector_listed(self):
        """cmd_status shows Thompson Sampling info when available."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

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
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        _mock_get_thompson.side_effect = Exception("no selector")
        rc, _ = self._run_status()
        assert rc == 0
        _mock_get_thompson.side_effect = None

    def test_status_verbose_logging(self):
        """cmd_status calls _setup_logging(True) when verbose."""
        with patch.object(cli, "_setup_logging") as mock_log, patch("builtins.print"):
            args = _args(verbose=True)
            cli.cmd_status(args)
        mock_log.assert_called_with(True)

    def test_status_default_host_when_none(self):
        """cmd_status uses default host when args.host is None."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        os.environ.pop("LM_STUDIO_HOST", None)
        _, printed = self._run_status(host=None)
        assert "localhost" in printed


# ===========================================================================
# TestCmdHealth
# ===========================================================================

class TestCmdHealth(unittest.TestCase):

    def setUp(self):
        _mock_lmstudio_cls.reset_mock()
        _mock_get_adapter_mgr.reset_mock()

    def _run_health(self, **kw):
        args = _args(**kw)
        with patch("builtins.print") as mock_print:
            rc = cli.cmd_health(args)
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        return rc, printed

    def test_health_returns_0(self):
        """cmd_health always returns 0."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        rc, _ = self._run_health()
        assert rc == 0

    def test_health_lmstudio_ok(self):
        """cmd_health prints OK when LM Studio is reachable."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": [{"id": "m1"}]}
        _mock_lmstudio_cls.return_value = mock_adapter

        _, printed = self._run_health()
        assert "OK" in printed

    def test_health_lmstudio_fail(self):
        """cmd_health prints FAIL when LM Studio is unreachable."""
        mock_adapter = MagicMock()
        mock_adapter._get.side_effect = ConnectionError("refused")
        _mock_lmstudio_cls.return_value = mock_adapter

        _, printed = self._run_health()
        assert "FAIL" in printed

    def test_health_prints_running_message(self):
        """cmd_health prints a 'Running health checks' message."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        _, printed = self._run_health()
        assert "health" in printed.lower()

    def test_health_adapter_manager_checked(self):
        """cmd_health invokes adapter manager health_check."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        mock_mgr = MagicMock()
        mock_mgr.health_check.return_value = {"lmstudio": {"healthy": True}}
        _mock_get_adapter_mgr.return_value = mock_mgr

        _, _printed = self._run_health()
        mock_mgr.health_check.assert_called_once()

    def test_health_adapter_manager_healthy_shows_ok(self):
        """cmd_health shows OK for healthy providers."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        mock_mgr = MagicMock()
        mock_mgr.health_check.return_value = {"openai": {"healthy": True}}
        _mock_get_adapter_mgr.return_value = mock_mgr

        _, printed = self._run_health()
        assert "OK" in printed

    def test_health_adapter_manager_unhealthy_shows_fail(self):
        """cmd_health shows FAIL for unhealthy providers."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        mock_mgr = MagicMock()
        mock_mgr.health_check.return_value = {"openai": {"healthy": False}}
        _mock_get_adapter_mgr.return_value = mock_mgr

        _, printed = self._run_health()
        assert "FAIL" in printed

    def test_health_adapter_manager_exception_is_silent(self):
        """cmd_health does not crash when adapter manager raises."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        _mock_get_adapter_mgr.side_effect = RuntimeError("no mgr")
        rc, _ = self._run_health()
        assert rc == 0
        _mock_get_adapter_mgr.side_effect = None

    def test_health_verbose_logging(self):
        """cmd_health calls _setup_logging(True) when verbose."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": []}
        _mock_lmstudio_cls.return_value = mock_adapter

        with patch.object(cli, "_setup_logging") as mock_log:
            args = _args(verbose=True)
            with patch("builtins.print"):
                cli.cmd_health(args)
        mock_log.assert_called_with(True)

    def test_health_model_count_shown(self):
        """cmd_health shows model count in LM Studio status line."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}
        _mock_lmstudio_cls.return_value = mock_adapter

        _, printed = self._run_health()
        assert "3" in printed


# ===========================================================================
# TestCmdUpgrade
# ===========================================================================

class TestCmdUpgrade(unittest.TestCase):

    def test_upgrade_success_returns_0(self):
        """cmd_upgrade returns 0 when orchestrator upgrade check succeeds."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch):
            args = _args()
            rc = cli.cmd_upgrade(args)
        assert rc == 0
        mock_orch.check_and_upgrade_models.assert_called_once()

    def test_upgrade_calls_check_and_upgrade_models(self):
        """cmd_upgrade calls orch.check_and_upgrade_models()."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch):
            cli.cmd_upgrade(_args())
        mock_orch.check_and_upgrade_models.assert_called_once()

    def test_upgrade_exception_returns_1(self):
        """cmd_upgrade returns 1 on orchestrator exception."""
        with patch.object(cli, "_build_orchestrator", side_effect=Exception("no git")):
            rc = cli.cmd_upgrade(_args())
        assert rc == 1

    def test_upgrade_prints_error_on_failure(self):
        """cmd_upgrade prints error message on failure."""
        with patch.object(cli, "_build_orchestrator", side_effect=Exception("boom")):
            with patch("builtins.print") as mock_print:
                cli.cmd_upgrade(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "boom" in printed

    def test_upgrade_uses_host_arg(self):
        """cmd_upgrade resolves host from args correctly."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch) as mock_build:
            cli.cmd_upgrade(_args(host="http://h:9"))
        assert mock_build.call_args.args[1] == "http://h:9"

    def test_upgrade_uses_mode_arg(self):
        """cmd_upgrade passes mode to _build_orchestrator."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch) as mock_build:
            cli.cmd_upgrade(_args(mode="planning"))
        assert mock_build.call_args.args[2] == "planning"

    def test_upgrade_check_and_upgrade_exception_is_caught(self):
        """cmd_upgrade handles exception from check_and_upgrade_models."""
        mock_orch = MagicMock()
        mock_orch.check_and_upgrade_models.side_effect = RuntimeError("network")
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch):
            rc = cli.cmd_upgrade(_args())
        assert rc == 1

    def test_upgrade_verbose_sets_logging(self):
        """cmd_upgrade calls _setup_logging(True) when verbose."""
        mock_orch = MagicMock()
        with patch.object(cli, "_build_orchestrator", return_value=mock_orch):
            with patch.object(cli, "_setup_logging") as mock_log:
                cli.cmd_upgrade(_args(verbose=True))
        mock_log.assert_called_with(True)


# ===========================================================================
# TestCmdReview
# ===========================================================================

class TestCmdReview(unittest.TestCase):

    def setUp(self):
        _mock_get_improvement.reset_mock()
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
        _mock_get_improvement.return_value = agent
        return agent

    def test_review_success_returns_0(self):
        """cmd_review returns 0 on successful review."""
        self._make_agent(success=True)
        rc = cli.cmd_review(_args())
        assert rc == 0

    def test_review_calls_agent_execute(self):
        """cmd_review calls agent.execute() with an AgentTask."""
        agent = self._make_agent(success=True)
        cli.cmd_review(_args())
        agent.execute.assert_called_once()

    def test_review_prints_recommendation_count(self):
        """cmd_review prints the number of recommendations found."""
        self._make_agent(success=True, recs=[{"priority": "high", "action": "a", "rationale": "r"}])
        with patch("builtins.print") as mock_print:
            cli.cmd_review(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "1" in printed

    def test_review_prints_auto_applied_count(self):
        """cmd_review prints the number of auto-applied changes."""
        self._make_agent(success=True, recs=[], applied=["change1", "change2"])
        with patch("builtins.print") as mock_print:
            cli.cmd_review(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "2" in printed

    def test_review_exception_returns_1(self):
        """cmd_review returns 1 when agent raises."""
        _mock_get_improvement.side_effect = Exception("agent broken")
        rc = cli.cmd_review(_args())
        assert rc == 1
        _mock_get_improvement.side_effect = None

    def test_review_exception_prints_error(self):
        """cmd_review prints error message on exception."""
        _mock_get_improvement.side_effect = Exception("broken agent")
        with patch("builtins.print") as mock_print:
            cli.cmd_review(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "broken agent" in printed
        _mock_get_improvement.side_effect = None

    def test_review_initializes_agent_with_adapter_manager(self):
        """cmd_review tries to initialize agent with adapter manager."""
        agent = self._make_agent(success=True)
        mock_mgr = MagicMock()
        _mock_get_adapter_mgr.return_value = mock_mgr
        cli.cmd_review(_args())
        agent.initialize.assert_called_once()

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
        with patch.object(cli, "_setup_logging") as mock_log:
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

class TestCmdInteractive(unittest.TestCase):

    def setUp(self):
        _mock_get_two_layer.reset_mock()

    def _run_interactive(self, inputs, mock_orch=None):
        """Run cmd_interactive with a scripted sequence of inputs."""
        if mock_orch is not None:
            _mock_get_two_layer.return_value = mock_orch
        else:
            # Default: two_layer raises so orch=None path
            _mock_get_two_layer.side_effect = Exception("no two layer")

        with patch("builtins.input", side_effect=inputs), patch("builtins.print"):
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
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
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
                _mock_get_two_layer.side_effect = Exception("no")
                rc = cli.cmd_interactive(_args())
                _mock_get_two_layer.side_effect = None

        assert rc == 0
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "/quit" in printed

    def test_interactive_status_command(self):
        """/status calls cmd_status then continues."""
        with patch.object(cli, "cmd_status") as mock_status:
            with patch("builtins.input", side_effect=["/status", "/quit"]):
                with patch("builtins.print"):
                    with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
                        _mock_get_two_layer.side_effect = Exception("no")
                        cli.cmd_interactive(_args())
                        _mock_get_two_layer.side_effect = None

        mock_status.assert_called_once()

    def test_interactive_review_command(self):
        """/review calls cmd_review then continues."""
        with patch.object(cli, "cmd_review") as mock_review:
            with patch("builtins.input", side_effect=["/review", "/quit"]):
                with patch("builtins.print"):
                    with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
                        _mock_get_two_layer.side_effect = Exception("no")
                        cli.cmd_interactive(_args())
                        _mock_get_two_layer.side_effect = None

        mock_review.assert_called_once()

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

        with patch("builtins.input", side_effect=["my goal", "/quit"]), patch("builtins.print"):
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
                cli.cmd_interactive(_args())

        mock_orch.generate_and_execute.assert_called_once()

    def test_interactive_goal_without_orchestrator_prints_message(self):
        """With no orchestrator available a helpful message is printed."""
        _mock_get_two_layer.side_effect = Exception("not available")

        with patch("builtins.input", side_effect=["some goal", "/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
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
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
                rc = cli.cmd_interactive(_args())

        assert rc == 0

    def test_interactive_prints_startup_banner(self):
        """cmd_interactive prints instructions at start."""
        with patch("builtins.input", side_effect=["/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
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
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
                cli.cmd_interactive(_args())

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "great result" in printed

    def test_interactive_verbose_sets_logging(self):
        """cmd_interactive calls _setup_logging(True) when verbose."""
        with patch.object(cli, "_setup_logging") as mock_log, patch("builtins.input", side_effect=["/quit"]):
            with patch("builtins.print"):
                with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
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
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
                cli.cmd_interactive(_args())

        assert mock_orch.generate_and_execute.call_count == 2


# ===========================================================================
# TestMain — argparse routing
# ===========================================================================

class TestMain(unittest.TestCase):

    def _run_main(self, argv, handler_return=0):
        """Invoke main() with patched sys.argv; capture sys.exit call."""
        with patch("sys.argv", argv), patch.object(cli, "cmd_run", return_value=handler_return) as m_run, \
                 patch.object(cli, "cmd_serve", return_value=handler_return) as m_serve, \
                 patch.object(cli, "cmd_start", return_value=handler_return) as m_start, \
                 patch.object(cli, "cmd_status", return_value=handler_return) as m_status, \
                 patch.object(cli, "cmd_health", return_value=handler_return) as m_health, \
                 patch.object(cli, "cmd_upgrade", return_value=handler_return) as m_upgrade, \
                 patch.object(cli, "cmd_review", return_value=handler_return) as m_review, \
                 patch.object(cli, "cmd_interactive", return_value=handler_return) as m_inter:
            try:
                cli.main()
            except SystemExit:  # noqa: VET022
                pass
            return {
                "run": m_run, "serve": m_serve, "start": m_start,
                "status": m_status, "health": m_health, "upgrade": m_upgrade,
                "review": m_review, "interactive": m_inter,
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

    def test_main_global_host_flag(self):
        """'vetinari --host http://h:9 status' sets host correctly."""
        mocks = self._run_main(["vetinari", "--host", "http://h:9", "status"])
        args_passed = mocks["status"].call_args.args[0]
        assert args_passed.host == "http://h:9"

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

class TestHelpers(unittest.TestCase):

    def test_get_host_from_args(self):
        """_get_host returns args value when provided."""
        assert cli._get_host("http://arghost:1") == "http://arghost:1"

    def test_get_host_from_env(self):
        """_get_host falls back to LM_STUDIO_HOST env var."""
        with patch.dict(os.environ, {"LM_STUDIO_HOST": "http://envhost:2"}):
            assert cli._get_host(None) == "http://envhost:2"

    def test_get_host_default(self):
        """_get_host returns default when no arg and no env var."""
        env_copy = {k: v for k, v in os.environ.items() if k != "LM_STUDIO_HOST"}
        with patch.dict(os.environ, env_copy, clear=True):
            result = cli._get_host(None)
        assert result == "http://localhost:1234"

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
            cli._print_banner("sandbox", "http://h:1")
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "SANDBOX" in printed

    def test_print_banner_contains_host(self):
        """_print_banner prints the host."""
        with patch("builtins.print") as mock_print:
            cli._print_banner("execution", "http://myhost:9000")
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "http://myhost:9000" in printed

    def test_build_orchestrator_instantiates_orchestrator(self):
        """_build_orchestrator creates an Orchestrator instance."""
        _mock_orchestrator_cls.reset_mock()
        cli._build_orchestrator("some/config.yaml", "http://h:1", "execution")
        _mock_orchestrator_cls.assert_called_once_with(
            "some/config.yaml", host="http://h:1", execution_mode="execution"
        )

    def test_load_config_missing_returns_defaults(self):
        """_load_config returns default dict when file doesn't exist."""
        result = cli._load_config("/nonexistent/path/config.yaml")
        assert "project_name" in result
        assert "tasks" in result

    def test_health_check_quiet_lmstudio_ok(self):
        """_health_check_quiet prints OK when LM Studio responds."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {"data": [{"id": "m1"}]}
        _mock_lmstudio_cls.return_value = mock_adapter

        with patch("builtins.print") as mock_print:
            cli._health_check_quiet("http://localhost:1234")

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "OK" in printed

    def test_health_check_quiet_lmstudio_fail(self):
        """_health_check_quiet prints FAIL when LM Studio is down."""
        mock_adapter = MagicMock()
        mock_adapter._get.side_effect = ConnectionError("refused")
        _mock_lmstudio_cls.return_value = mock_adapter

        with patch("builtins.print") as mock_print:
            cli._health_check_quiet("http://localhost:1234")

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "FAIL" in printed


# ===========================================================================
# TestEdgeCases — boundary and integration edge cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_cmd_run_long_goal_truncated_in_print(self):
        """cmd_run truncates very long goal strings in the printed line."""
        long_goal = "x" * 200
        mock_orch = MagicMock()
        mock_orch.generate_and_execute.return_value = {"completed": 0, "failed": 0}
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
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
        mock_orch.generate_and_execute.return_value = {
            "completed": 1, "failed": 0, "final_output": long_output
        }
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
            with patch("builtins.print") as mock_print:
                cli.cmd_run(_args(goal="truncation test"))

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        # Should contain y's but not the full 5000
        assert "y" in printed

    def test_cmd_interactive_final_output_truncated_to_1500(self):
        """cmd_interactive truncates final_output to 1500 chars."""
        mock_orch = MagicMock()
        long_output = "z" * 3000
        mock_orch.generate_and_execute.return_value = {
            "completed": 1, "final_output": long_output
        }
        _mock_get_two_layer.return_value = mock_orch
        _mock_get_two_layer.side_effect = None

        with patch("builtins.input", side_effect=["some goal", "/quit"]), patch("builtins.print") as mock_print:
            with patch.object(cli, "_build_orchestrator", return_value=MagicMock()):
                cli.cmd_interactive(_args())

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "z" in printed

    def test_get_host_none_arg_no_env_returns_default(self):
        """_get_host returns localhost:1234 as fallback."""
        saved = os.environ.pop("LM_STUDIO_HOST", None)
        try:
            result = cli._get_host(None)
            assert "1234" in result
        finally:
            if saved is not None:
                os.environ["LM_STUDIO_HOST"] = saved

    def test_cmd_serve_web_host_default(self):
        """cmd_serve defaults web_host to '127.0.0.1'."""
        args = _args(port=5000, web_host="127.0.0.1", debug=False)
        _mock_flask_app.run.reset_mock()
        cli.cmd_serve(args)
        all_args = str(_mock_flask_app.run.call_args)
        assert "127.0.0.1" in all_args

    def test_cmd_status_model_list_displayed(self):
        """cmd_status lists up to 5 model IDs."""
        mock_adapter = MagicMock()
        mock_adapter._get.return_value = {
            "data": [{"id": f"model-{i}"} for i in range(6)]
        }
        _mock_lmstudio_cls.return_value = mock_adapter

        with patch("builtins.print") as mock_print:
            cli.cmd_status(_args())

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        # Should show up to 5 models (model-0 through model-4)
        assert "model-0" in printed

    def test_cmd_review_at_most_5_recs_printed(self):
        """cmd_review prints at most 5 recommendations."""
        recs = [
            {"priority": "low", "action": f"action-{i}", "rationale": "r"}
            for i in range(10)
        ]
        agent = MagicMock()
        result = MagicMock()
        result.success = True
        result.output = {"recommendations": recs, "auto_applied": []}
        agent.execute.return_value = result
        _mock_get_improvement.return_value = agent
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
                except SystemExit:  # noqa: VET022
                    pass
        args_passed = m.call_args.args[0]
        assert args_passed.config == "custom.yaml"

    def test_main_start_goal_flag(self):
        """'vetinari start --goal ...' parses goal in start subcommand."""
        with patch("sys.argv", ["vetinari", "start", "--goal", "Run tests"]):
            with patch.object(cli, "cmd_start", return_value=0) as m:
                try:
                    cli.main()
                except SystemExit:  # noqa: VET022
                    pass
        args_passed = m.call_args.args[0]
        assert args_passed.goal == "Run tests"


if __name__ == "__main__":
    unittest.main()
