"""Tests for vetinari/cli_training.py, vetinari/cli_kaizen.py, vetinari/cli_devops.py.

Verifies that each sub-module's register_commands() wires up the correct
argparse subparsers, and that the cmd_* functions are callable and return
the expected exit codes under mocked dependencies.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_stub_module as _make_stub

# ---------------------------------------------------------------------------
# Minimal sys.modules stubs so importing the sub-modules does not require
# the full Vetinari dependency tree.  These are installed before any local
# import below.  _SYS_MODULES_SNAPSHOT captures state before patching so
# the module-scoped _restore_sys_modules fixture can undo the damage.
# ---------------------------------------------------------------------------

_SYS_MODULES_SNAPSHOT = dict(sys.modules)

# vetinari package root
import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_vet_pkg = _make_stub("vetinari")
_vet_pkg.__path__ = [_os.path.join(_ROOT, "vetinari")]
_vet_pkg.__package__ = "vetinari"
sys.modules.setdefault("vetinari", _vet_pkg)

# vetinari.types — load real module (stdlib-only)
import importlib.util as _ilu

_vtypes_spec = _ilu.spec_from_file_location("vetinari.types", _os.path.join(_ROOT, "vetinari", "types.py"))
_vtypes_mod = _ilu.module_from_spec(_vtypes_spec)
sys.modules.setdefault("vetinari.types", _vtypes_mod)
_vtypes_spec.loader.exec_module(_vtypes_mod)

# vetinari.orchestrator stub
_mock_orch_cls = MagicMock()
sys.modules.setdefault(
    "vetinari.orchestrator",
    _make_stub("vetinari.orchestrator", Orchestrator=_mock_orch_cls),
)

# vetinari.adapter_manager stub
_mock_get_adapter_mgr = MagicMock()
sys.modules.setdefault(
    "vetinari.adapter_manager",
    _make_stub("vetinari.adapter_manager", get_adapter_manager=_mock_get_adapter_mgr),
)

# vetinari.agents package
if "vetinari.agents" not in sys.modules:
    _agents_pkg = _make_stub("vetinari.agents")
    _agents_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "agents")]
    _agents_pkg.__package__ = "vetinari.agents"
    sys.modules["vetinari.agents"] = _agents_pkg

# vetinari.agents.contracts stub
_mock_AgentTask = MagicMock()
sys.modules.setdefault(
    "vetinari.agents.contracts",
    _make_stub("vetinari.agents.contracts", AgentTask=_mock_AgentTask),
)

# vetinari.agents.consolidated package
if "vetinari.agents.consolidated" not in sys.modules:
    _consol_pkg = _make_stub("vetinari.agents.consolidated")
    _consol_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "agents", "consolidated")]
    _consol_pkg.__package__ = "vetinari.agents.consolidated"
    sys.modules["vetinari.agents.consolidated"] = _consol_pkg

# vetinari.agents.consolidated.operations_agent stub
_mock_get_ops_agent = MagicMock()
sys.modules.setdefault(
    "vetinari.agents.consolidated.operations_agent",
    _make_stub("vetinari.agents.consolidated.operations_agent", get_operations_agent=_mock_get_ops_agent),
)

# vetinari.benchmarks package
if "vetinari.benchmarks" not in sys.modules:
    _bench_pkg = _make_stub("vetinari.benchmarks")
    _bench_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "benchmarks")]
    _bench_pkg.__package__ = "vetinari.benchmarks"
    sys.modules["vetinari.benchmarks"] = _bench_pkg

_mock_BenchmarkSuite = MagicMock()
sys.modules.setdefault(
    "vetinari.benchmarks.suite",
    _make_stub("vetinari.benchmarks.suite", BenchmarkSuite=_mock_BenchmarkSuite),
)

# vetinari.drift package
if "vetinari.drift" not in sys.modules:
    _drift_pkg = _make_stub("vetinari.drift")
    _drift_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "drift")]
    _drift_pkg.__package__ = "vetinari.drift"
    sys.modules["vetinari.drift"] = _drift_pkg

_mock_ContractRegistry = MagicMock()
sys.modules.setdefault(
    "vetinari.drift.contract_registry",
    _make_stub(
        "vetinari.drift.contract_registry", ContractRegistry=_mock_ContractRegistry, get_contract_registry=MagicMock()
    ),
)

_mock_get_drift_monitor = MagicMock()
# Ensure drift.monitor submodule stubs exist for import
for _dm_sub in ("vetinari.drift.capability_auditor", "vetinari.drift.schema_validator"):
    sys.modules.setdefault(
        _dm_sub,
        _make_stub(
            _dm_sub,
            **{
                f"get_{_dm_sub.rsplit('.', 1)[-1].replace('_auditor', '_auditor').replace('_validator', '_validator')}": MagicMock()
            },
        ),
    )
sys.modules.setdefault(
    "vetinari.drift.monitor",
    _make_stub("vetinari.drift.monitor", get_drift_monitor=_mock_get_drift_monitor),
)

# vetinari.mcp package
if "vetinari.mcp" not in sys.modules:
    _mcp_pkg = _make_stub("vetinari.mcp")
    _mcp_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "mcp")]
    _mcp_pkg.__package__ = "vetinari.mcp"
    sys.modules["vetinari.mcp"] = _mcp_pkg

sys.modules.setdefault(
    "vetinari.mcp.server",
    _make_stub("vetinari.mcp.server", get_mcp_server=MagicMock()),
)
sys.modules.setdefault(
    "vetinari.mcp.transport",
    _make_stub("vetinari.mcp.transport", StdioTransport=MagicMock()),
)

# vetinari.kaizen package
if "vetinari.kaizen" not in sys.modules:
    _kaizen_pkg = _make_stub("vetinari.kaizen")
    _kaizen_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "kaizen")]
    _kaizen_pkg.__package__ = "vetinari.kaizen"
    sys.modules["vetinari.kaizen"] = _kaizen_pkg

_mock_ImprovementLog = MagicMock()
sys.modules.setdefault(
    "vetinari.kaizen.improvement_log",
    _make_stub("vetinari.kaizen.improvement_log", ImprovementLog=_mock_ImprovementLog),
)
_mock_AutoGembaWalk = MagicMock()
sys.modules.setdefault(
    "vetinari.kaizen.gemba",
    _make_stub("vetinari.kaizen.gemba", AutoGembaWalk=_mock_AutoGembaWalk),
)

# vetinari.training package
if "vetinari.training" not in sys.modules:
    _training_pkg = _make_stub("vetinari.training")
    _training_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "training")]
    _training_pkg.__package__ = "vetinari.training"
    sys.modules["vetinari.training"] = _training_pkg

_mock_TrainingCurriculum = MagicMock()
_mock_IdleDetector = MagicMock()
_mock_TrainingScheduler = MagicMock()
_mock_TrainingDataSeeder = MagicMock()
_mock_AgentTrainer = MagicMock()

sys.modules.setdefault(
    "vetinari.training.curriculum",
    _make_stub("vetinari.training.curriculum", TrainingCurriculum=_mock_TrainingCurriculum),
)
sys.modules.setdefault(
    "vetinari.training.idle_scheduler",
    _make_stub(
        "vetinari.training.idle_scheduler",
        IdleDetector=_mock_IdleDetector,
        TrainingScheduler=_mock_TrainingScheduler,
    ),
)
sys.modules.setdefault(
    "vetinari.training.data_seeder",
    _make_stub("vetinari.training.data_seeder", TrainingDataSeeder=_mock_TrainingDataSeeder),
)
sys.modules.setdefault(
    "vetinari.training.agent_trainer",
    _make_stub("vetinari.training.agent_trainer", AgentTrainer=_mock_AgentTrainer),
)

# vetinari.learning package
if "vetinari.learning" not in sys.modules:
    _learning_pkg = _make_stub("vetinari.learning")
    _learning_pkg.__path__ = [_os.path.join(_ROOT, "vetinari", "learning")]
    _learning_pkg.__package__ = "vetinari.learning"
    sys.modules["vetinari.learning"] = _learning_pkg

sys.modules.setdefault(
    "vetinari.learning.training_data",
    _make_stub("vetinari.learning.training_data", get_training_collector=MagicMock()),
)

# ---------------------------------------------------------------------------
# Import the modules under test AFTER stubs are installed
# ---------------------------------------------------------------------------

import vetinari.cli as _cli  # consolidated: all CLI subcommands now in cli.py

# ---------------------------------------------------------------------------
# Cleanup — remove stub modules after this test file finishes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _restore_sys_modules():
    """Undo sys.modules patching after all tests in this module complete.

    Without this, stub modules installed at import time leak into later test
    files (e.g. test_dapo_training) and prevent real submodule imports.
    """
    yield
    stubs_added = set(sys.modules) - set(_SYS_MODULES_SNAPSHOT)
    for key in stubs_added:
        del sys.modules[key]
    for key, original_mod in _SYS_MODULES_SNAPSHOT.items():
        if sys.modules.get(key) is not original_mod:
            sys.modules[key] = original_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _args(**kw) -> SimpleNamespace:
    """Return a SimpleNamespace with safe CLI defaults."""
    defaults = {
        "verbose": False,
        "host": None,
        "config": "manifest/vetinari.yaml",
        "mode": "execution",
        "transport": "stdio",
    }
    defaults.update(kw)
    return SimpleNamespace(**defaults)


@pytest.fixture
def make_subparsers():
    """Return a factory that builds a fresh (parser, subparsers) tuple."""

    def _factory() -> tuple[argparse.ArgumentParser, argparse.Action]:
        parser = argparse.ArgumentParser()
        return parser, parser.add_subparsers(dest="command")

    return _factory


# ===========================================================================
# TestCliTrainingRegisterCommands
# ===========================================================================


class TestCliTrainingRegisterCommands:
    """Tests for _cli._register_training_commands()."""

    @pytest.fixture(autouse=True)
    def setup(self, make_subparsers) -> None:
        self.parser, self.subparsers = make_subparsers()
        _cli._register_training_commands(self.subparsers)

    def test_train_subparser_registered(self) -> None:
        """register_commands adds a 'train' subparser."""
        args = self.parser.parse_args(["train", "status"])
        assert args.command == "train"

    def test_train_action_status(self) -> None:
        """'train status' sets train_action=status."""
        args = self.parser.parse_args(["train", "status"])
        assert args.train_action == "status"

    def test_train_action_start(self) -> None:
        """'train start' sets train_action=start."""
        args = self.parser.parse_args(["train", "start"])
        assert args.train_action == "start"

    def test_train_action_curriculum(self) -> None:
        """'train curriculum' sets train_action=curriculum."""
        args = self.parser.parse_args(["train", "curriculum"])
        assert args.train_action == "curriculum"

    def test_train_action_history(self) -> None:
        """'train history' sets train_action=history."""
        args = self.parser.parse_args(["train", "history"])
        assert args.train_action == "history"

    def test_train_skill_flag(self) -> None:
        """'train start --skill planning' parses skill correctly."""
        args = self.parser.parse_args(["train", "start", "--skill", "planning"])
        assert args.skill == "planning"

    def test_train_skill_default_none(self) -> None:
        """skill defaults to None when not provided."""
        args = self.parser.parse_args(["train", "status"])
        assert args.skill is None

    def test_train_run_backend_model_flags(self) -> None:
        """'train run' accepts native backend/model controls."""
        args = self.parser.parse_args([
            "train",
            "run",
            "--base-model",
            "owner/model",
            "--backend",
            "vllm",
            "--format",
            "safetensors",
            "--revision",
            "a" * 40,
        ])
        assert args.base_model == "owner/model"
        assert args.backend == "vllm"
        assert args.model_format == "safetensors"
        assert args.model_revision == "a" * 40

    def test_train_run_defaults_to_vllm(self) -> None:
        """'train run' defaults to the native vLLM deployment path."""
        args = self.parser.parse_args(["train", "run"])
        assert args.backend == "vllm"

    def test_train_all_actions_accepted(self) -> None:
        """All eight train_action values are valid."""
        valid_actions = ["status", "start", "pause", "resume", "data", "seed", "curriculum", "history"]
        for action in valid_actions:
            args = self.parser.parse_args(["train", action])
            assert args.train_action == action


# ===========================================================================
# TestCmdTrain
# ===========================================================================


class TestCmdTrain:
    """Tests for _cli.cmd_train()."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        _mock_TrainingCurriculum.reset_mock()
        _mock_IdleDetector.reset_mock()
        _mock_TrainingScheduler.reset_mock()
        _mock_TrainingDataSeeder.reset_mock()
        _mock_AgentTrainer.reset_mock()

    def test_status_returns_0(self) -> None:
        """cmd_train status returns 0."""
        mock_curriculum = MagicMock()
        mock_curriculum.get_status.return_value = {
            "phase": "warmup",
            "next_activity_description": "review",
        }
        _mock_TrainingCurriculum.return_value = mock_curriculum
        mock_idle = MagicMock()
        mock_idle.idle = True
        mock_idle.idle_duration_minutes = 5.0
        _mock_IdleDetector.return_value = mock_idle

        with patch("builtins.print"):
            rc = _cli.cmd_train(_args(train_action="status"))
        assert rc == 0

    def test_run_passes_backend_controls_to_pipeline(self) -> None:
        """cmd_train run wires CLI backend/model controls to TrainingPipeline.run()."""
        pipeline = MagicMock()
        pipeline.check_requirements.return_value = {"ready_for_training": True}
        pipeline.run.return_value = SimpleNamespace(
            run_id="run_test",
            success=True,
            output_model_path="/native/adapter",
            model_manifest_path="/native/adapter/.vetinari-training-manifest.json",
            error="",
        )
        pipeline_cls = MagicMock(return_value=pipeline)
        pipeline_mod = _make_stub("vetinari.training.pipeline", TrainingPipeline=pipeline_cls)

        with patch.dict(sys.modules, {"vetinari.training.pipeline": pipeline_mod}):
            with patch("builtins.print"):
                rc = _cli.cmd_train(
                    _args(
                        train_action="run",
                        skill="coding",
                        base_model="owner/model",
                        backend="nim",
                        model_format="gptq",
                        model_revision="b" * 40,
                    )
                )

        assert rc == 0
        pipeline.run.assert_called_once_with(
            base_model="owner/model",
            task_type="coding",
            backend="nim",
            model_format="gptq",
            model_revision="b" * 40,
        )

    def test_pause_returns_1(self) -> None:
        """cmd_train pause returns nonzero because no CLI pause control is wired."""
        with patch("builtins.print") as mock_print:
            rc = _cli.cmd_train(_args(train_action="pause"))
        assert rc == 1
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "unsupported" in printed.lower()

    def test_resume_returns_1(self) -> None:
        """cmd_train resume returns nonzero because no CLI resume control is wired."""
        with patch("builtins.print") as mock_print:
            rc = _cli.cmd_train(_args(train_action="resume"))
        assert rc == 1
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "unsupported" in printed.lower()

    def test_seed_returns_0(self) -> None:
        """cmd_train seed returns 0."""
        mock_seeder = MagicMock()
        mock_seeder.seed_if_empty.return_value = 3
        _mock_TrainingDataSeeder.return_value = mock_seeder

        with patch("builtins.print"):
            rc = _cli.cmd_train(_args(train_action="seed"))
        assert rc == 0

    def test_seed_prints_count(self) -> None:
        """cmd_train seed prints the seeded count."""
        mock_seeder = MagicMock()
        mock_seeder.seed_if_empty.return_value = 7
        _mock_TrainingDataSeeder.return_value = mock_seeder

        with patch("builtins.print") as mock_print:
            _cli.cmd_train(_args(train_action="seed"))
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "7" in printed

    def test_history_no_agents_prints_message(self) -> None:
        """cmd_train history with no agents prints a 'no history' message."""
        mock_trainer = MagicMock()
        mock_trainer.get_stats.return_value = {"agents": {}}
        _mock_AgentTrainer.return_value = mock_trainer

        with patch("builtins.print") as mock_print:
            rc = _cli.cmd_train(_args(train_action="history"))
        assert rc == 0
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "no training" in printed.lower()

    def test_history_with_agents_lists_them(self) -> None:
        """cmd_train history lists agent names when history exists."""
        mock_trainer = MagicMock()
        mock_trainer.get_stats.return_value = {"agents": {"planner": {"last_trained": "2025-01-01", "run_count": 3}}}
        _mock_AgentTrainer.return_value = mock_trainer

        with patch("builtins.print") as mock_print:
            rc = _cli.cmd_train(_args(train_action="history"))
        assert rc == 0
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "planner" in printed

    def test_unknown_action_returns_1(self) -> None:
        """cmd_train with an unknown action returns 1."""
        rc = _cli.cmd_train(_args(train_action="nonexistent"))
        assert rc == 1


# ===========================================================================
# TestCliKaizenRegisterCommands
# ===========================================================================


class TestCliKaizenRegisterCommands:
    """Tests for _cli._register_kaizen_commands()."""

    @pytest.fixture(autouse=True)
    def setup(self, make_subparsers) -> None:
        self.parser, self.subparsers = make_subparsers()
        _cli._register_kaizen_commands(self.subparsers)

    def test_kaizen_subparser_registered(self) -> None:
        """register_commands adds a 'kaizen' subparser."""
        args = self.parser.parse_args(["kaizen", "report"])
        assert args.command == "kaizen"

    def test_kaizen_action_report(self) -> None:
        """'kaizen report' sets kaizen_action=report."""
        args = self.parser.parse_args(["kaizen", "report"])
        assert args.kaizen_action == "report"

    def test_kaizen_action_gemba(self) -> None:
        """'kaizen gemba' sets kaizen_action=gemba."""
        args = self.parser.parse_args(["kaizen", "gemba"])
        assert args.kaizen_action == "gemba"


# ===========================================================================
# TestCmdKaizen
# ===========================================================================


class TestCmdKaizen:
    """Tests for _cli.cmd_kaizen()."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        _mock_ImprovementLog.reset_mock()
        _mock_AutoGembaWalk.reset_mock()

    @pytest.fixture
    def kaizen_report(self) -> MagicMock:
        """Minimal weekly kaizen report mock with default totals."""
        report = MagicMock()
        report.total_proposed = 2
        report.total_active = 1
        report.total_confirmed = 0
        report.total_failed = 0
        report.total_reverted = 0
        report.avg_improvement_effect = 0.05
        report.generated_at = MagicMock()
        report.generated_at.isoformat.return_value = "2025-01-01T00:00:00"
        return report

    def test_report_returns_0(self, kaizen_report: MagicMock) -> None:
        """cmd_kaizen report returns 0."""
        log = MagicMock()
        log.get_weekly_report.return_value = kaizen_report
        _mock_ImprovementLog.return_value = log

        with patch("builtins.print"):
            rc = _cli.cmd_kaizen(_args(kaizen_action="report"))
        assert rc == 0

    def test_report_prints_proposed_count(self, kaizen_report: MagicMock) -> None:
        """cmd_kaizen report prints the number of proposed improvements."""
        log = MagicMock()
        kaizen_report.total_proposed = 5
        log.get_weekly_report.return_value = kaizen_report
        _mock_ImprovementLog.return_value = log

        with patch("builtins.print") as mock_print:
            _cli.cmd_kaizen(_args(kaizen_action="report"))
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "5" in printed

    def test_gemba_returns_0(self) -> None:
        """cmd_kaizen gemba returns 0."""
        log = MagicMock()
        _mock_ImprovementLog.return_value = log

        gemba_walk = MagicMock()
        gemba_report = MagicMock()
        gemba_report.findings = []
        gemba_report.improvements_proposed = 0
        gemba_walk.run.return_value = gemba_report
        _mock_AutoGembaWalk.return_value = gemba_walk

        with patch("builtins.print"):
            rc = _cli.cmd_kaizen(_args(kaizen_action="gemba"))
        assert rc == 0

    def test_gemba_no_findings_prints_clean_message(self) -> None:
        """cmd_kaizen gemba with no findings prints the all-clear message."""
        log = MagicMock()
        _mock_ImprovementLog.return_value = log

        gemba_walk = MagicMock()
        gemba_report = MagicMock()
        gemba_report.findings = []
        gemba_report.improvements_proposed = 0
        gemba_walk.run.return_value = gemba_report
        _mock_AutoGembaWalk.return_value = gemba_walk

        with patch("builtins.print") as mock_print:
            _cli.cmd_kaizen(_args(kaizen_action="gemba"))
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "clean" in printed.lower()

    def test_unknown_action_returns_1(self) -> None:
        """cmd_kaizen with no recognized action returns 1."""
        log = MagicMock()
        _mock_ImprovementLog.return_value = log

        with patch("builtins.print"):
            rc = _cli.cmd_kaizen(_args(kaizen_action=None))
        assert rc == 1


# ===========================================================================
# TestCliDevopsRegisterCommands
# ===========================================================================


class TestCliDevopsRegisterCommands:
    """Tests for _cli._register_devops_commands()."""

    @pytest.fixture(autouse=True)
    def setup(self, make_subparsers) -> None:
        self.parser, self.subparsers = make_subparsers()
        _cli._register_devops_commands(self.subparsers)

    def test_upgrade_subparser_registered(self) -> None:
        """register_commands adds an 'upgrade' subparser."""
        args = self.parser.parse_args(["upgrade"])
        assert args.command == "upgrade"

    def test_review_subparser_registered(self) -> None:
        """register_commands adds a 'review' subparser."""
        args = self.parser.parse_args(["review"])
        assert args.command == "review"

    def test_benchmark_subparser_registered(self) -> None:
        """register_commands adds a 'benchmark' subparser."""
        args = self.parser.parse_args(["benchmark"])
        assert args.command == "benchmark"

    def test_benchmark_agents_flag(self) -> None:
        """'benchmark --agents a b' parses agents list correctly."""
        args = self.parser.parse_args(["benchmark", "--agents", "planner", "builder"])
        assert args.agents == ["planner", "builder"]

    def test_mcp_subparser_registered(self) -> None:
        """register_commands adds an 'mcp' subparser."""
        args = self.parser.parse_args(["mcp"])
        assert args.command == "mcp"

    def test_mcp_default_transport_stdio(self) -> None:
        """mcp defaults to stdio transport."""
        args = self.parser.parse_args(["mcp"])
        assert args.transport == "stdio"

    def test_mcp_http_transport(self) -> None:
        """'mcp --transport http' sets transport=http."""
        args = self.parser.parse_args(["mcp", "--transport", "http"])
        assert args.transport == "http"

    def test_mcp_port_flag(self) -> None:
        """'mcp --mcp-port 9999' parses mcp_port correctly."""
        args = self.parser.parse_args(["mcp", "--mcp-port", "9999"])
        assert args.mcp_port == 9999

    def test_drift_check_subparser_registered(self) -> None:
        """register_commands adds a 'drift-check' subparser."""
        args = self.parser.parse_args(["drift-check"])
        assert args.command == "drift-check"


# ===========================================================================
# TestCmdDriftCheck
# ===========================================================================


class TestCmdDriftCheck:
    """Tests for _cli.cmd_drift_check() using DriftMonitor."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        _mock_get_drift_monitor.reset_mock()

    def test_no_drift_returns_0(self) -> None:
        """cmd_drift_check returns 0 when no drift detected."""
        mock_report = MagicMock()
        mock_report.is_clean = True
        mock_report.duration_ms = 42.0
        monitor = MagicMock()
        monitor.run_full_audit.return_value = mock_report
        _mock_get_drift_monitor.return_value = monitor

        with patch("builtins.print"), patch("vetinari.cli._setup_logging"):
            rc = _cli.cmd_drift_check(_args())
        assert rc == 0

    def test_drift_detected_returns_1(self) -> None:
        """cmd_drift_check returns 1 when drift is detected."""
        mock_report = MagicMock()
        mock_report.is_clean = False
        mock_report.duration_ms = 100.0
        mock_report.contract_drifts = {"Plan": {"previous": "abc", "current": "def"}}
        mock_report.capability_drifts = []
        mock_report.schema_errors = {}
        mock_report.issues = []
        monitor = MagicMock()
        monitor.run_full_audit.return_value = mock_report
        _mock_get_drift_monitor.return_value = monitor

        with patch("builtins.print"), patch("vetinari.cli._setup_logging"):
            rc = _cli.cmd_drift_check(_args())
        assert rc == 1

    def test_exception_returns_1(self) -> None:
        """cmd_drift_check returns 1 on unexpected exception."""
        _mock_get_drift_monitor.side_effect = RuntimeError("monitor unavailable")

        with patch("builtins.print"), patch("vetinari.cli._setup_logging"):
            rc = _cli.cmd_drift_check(_args())
        assert rc == 1
        _mock_get_drift_monitor.side_effect = None

    def test_no_drift_prints_clear_message(self) -> None:
        """cmd_drift_check prints an all-clear message when no drift found."""
        mock_report = MagicMock()
        mock_report.is_clean = True
        mock_report.duration_ms = 42.0
        monitor = MagicMock()
        monitor.run_full_audit.return_value = mock_report
        _mock_get_drift_monitor.return_value = monitor

        with patch("builtins.print") as mock_print, patch("vetinari.cli._setup_logging"):
            _cli.cmd_drift_check(_args())
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "no drift" in printed.lower()


# ===========================================================================
# TestCmdBenchmark
# ===========================================================================


class TestCmdBenchmark:
    """Tests for _cli.cmd_benchmark()."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        _mock_BenchmarkSuite.reset_mock()

    def test_no_regressions_returns_0(self) -> None:
        """cmd_benchmark returns 0 when no regressions detected."""
        suite = MagicMock()
        suite.run_all.return_value = {}
        suite.check_regression.return_value = []
        _mock_BenchmarkSuite.return_value = suite

        with patch("builtins.print"), patch("vetinari.cli._setup_logging"):
            rc = _cli.cmd_benchmark(_args())
        assert rc == 0

    def test_regressions_returns_1(self) -> None:
        """cmd_benchmark returns 1 when regressions are detected."""
        suite = MagicMock()
        suite.run_all.return_value = {}
        suite.check_regression.return_value = ["planner score dropped"]
        _mock_BenchmarkSuite.return_value = suite

        with patch("builtins.print"), patch("vetinari.cli._setup_logging"):
            rc = _cli.cmd_benchmark(_args())
        assert rc == 1

    def test_exception_returns_1(self) -> None:
        """cmd_benchmark returns 1 when an exception is raised."""
        _mock_BenchmarkSuite.side_effect = ImportError("no suite")
        with patch("builtins.print"), patch("vetinari.cli._setup_logging"):
            rc = _cli.cmd_benchmark(_args())
        assert rc == 1
        _mock_BenchmarkSuite.side_effect = None


# ===========================================================================
# TestCmdMigrate
# ===========================================================================


class TestCmdMigrate:
    """Tests for _cli.cmd_migrate() and migrate subparser registration."""

    def test_migrate_subparser_registered(self, make_subparsers) -> None:
        """_register commands includes 'migrate' as a valid subcommand."""
        parser, subparsers = make_subparsers()
        # The migrate parser is registered inline in main(); reproduce it here.
        p = subparsers.add_parser("migrate", help="Apply database schema migrations")
        p.add_argument("--db-path", default="vetinari_memory.db")
        ns = parser.parse_args(["migrate"])
        assert ns.command == "migrate"

    def test_migrate_default_db_path(self, make_subparsers) -> None:
        """migrate subparser defaults --db-path to vetinari_memory.db."""
        parser, subparsers = make_subparsers()
        p = subparsers.add_parser("migrate", help="Apply database schema migrations")
        p.add_argument("--db-path", default="vetinari_memory.db")
        ns = parser.parse_args(["migrate"])
        assert ns.db_path == "vetinari_memory.db"

    def test_migrate_returns_0_on_success(self) -> None:
        """cmd_migrate returns 0 when run_migrations succeeds."""
        mock_run_migrations = MagicMock(return_value=0)
        with (
            patch("vetinari.cli._setup_logging"),
            patch("builtins.print"),
            patch("vetinari.migrations.run_migrations", mock_run_migrations),
        ):
            rc = _cli.cmd_migrate(_args(db_path="test.db"))
        assert rc == 0

    def test_migrate_returns_1_on_exception(self) -> None:
        """cmd_migrate returns 1 when run_migrations raises."""
        mock_run_migrations = MagicMock(side_effect=RuntimeError("disk full"))
        with patch("vetinari.cli._setup_logging"), patch("vetinari.migrations.run_migrations", mock_run_migrations):
            rc = _cli.cmd_migrate(_args(db_path="test.db"))
        assert rc == 1

    def test_migrate_prints_applied_count(self) -> None:
        """cmd_migrate prints how many migrations were applied."""
        mock_run_migrations = MagicMock(return_value=3)
        with (
            patch("vetinari.cli._setup_logging"),
            patch("builtins.print") as mock_print,
            patch("vetinari.migrations.run_migrations", mock_run_migrations),
        ):
            _cli.cmd_migrate(_args(db_path="mydb.db"))
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "3" in printed


# ===========================================================================
# TestCmdWatch
# ===========================================================================


class TestCmdWatch:
    """Tests for _cli.cmd_watch() and _register_watch_commands() registration."""

    def test_watch_subparser_registered(self, make_subparsers) -> None:
        """_register_watch_commands registers 'watch' as a valid subcommand."""
        parser, subparsers = make_subparsers()
        _cli._register_watch_commands(subparsers)
        ns = parser.parse_args(["watch", "report"])
        assert ns.command == "watch"
        assert ns.watch_action == "report"

    def test_watch_actions_accepted(self, make_subparsers) -> None:
        """_register_watch_commands accepts start, report, and scan actions."""
        parser, subparsers = make_subparsers()
        _cli._register_watch_commands(subparsers)
        for action in ["start", "report", "scan"]:
            ns = parser.parse_args(["watch", action])
            assert ns.watch_action == action

    def test_watch_unknown_action_returns_1(self) -> None:
        """cmd_watch returns 1 for an unknown watch_action."""
        rc = _cli.cmd_watch(_args(watch_action="explode"))
        assert rc == 1

    def test_watch_report_returns_0(self) -> None:
        """cmd_watch returns 0 for 'report' action when WatchService is available."""
        mock_svc = MagicMock()
        mock_svc.get_report.return_value = MagicMock(
            total_files_watched=5,
            directives_found=2,
            last_scan_at="2026-01-01T00:00:00",
        )
        with patch("vetinari.cli_training._cmd_watch_report", return_value=0):
            rc = _cli.cmd_watch(_args(watch_action="report"))
        assert rc == 0

    def test_watch_scan_returns_0(self) -> None:
        """cmd_watch returns 0 for 'scan' action via patched handler."""
        with patch("vetinari.cli_training._cmd_watch_scan", return_value=0):
            rc = _cli.cmd_watch(_args(watch_action="scan"))
        assert rc == 0
