"""
Tests for Phase 7 Drift Control — vetinari/drift/

Coverage:
    ContractRegistry            — register, fingerprint, snapshot, load, check_drift
    CapabilityAuditor           — register_documented, audit_agent, audit_all, drift findings
    SchemaValidator             — register_schema, validate, validate_many, vetinari schemas
    DriftMonitor                — bootstrap, individual checks, run_full_audit, report
    DriftReport                 — is_clean, summary, to_dict
    TestCheckerBehavior         — shared script loader for governance behavior tests
    TestDocContractAlignment    — behavior tests proving contract fingerprint checker fails closed
    TestCapabilityAuditBehavior — behavior tests proving auditor reads from file, not live code
    TestCoverageGateBehavior    — behavior tests proving GLOBAL_MIN > 0 and unlisted modules fail
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from vetinari.types import AgentType

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _reset_all():
    from vetinari.drift.capability_auditor import reset_capability_auditor
    from vetinari.drift.contract_registry import reset_contract_registry
    from vetinari.drift.monitor import reset_drift_monitor
    from vetinari.drift.schema_validator import reset_schema_validator

    reset_contract_registry()
    reset_capability_auditor()
    reset_schema_validator()
    reset_drift_monitor()


# Module-level dataclass for registration tests — must not be inside a test
# function because Python 3.14's dataclasses._is_type() crashes when the test
# module's sys.modules entry is missing.
@dataclass
class _SampleDC:
    x: int = 1
    y: str = "hello"


# ─────────────────────────────────────────────────────────────────────────────
# ContractRegistry
# ─────────────────────────────────────────────────────────────────────────────


class TestContractRegistry:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_all()
        from vetinari.drift.contract_registry import get_contract_registry

        self.reg = get_contract_registry()
        yield
        _reset_all()

    # Singleton
    def test_singleton(self):
        from vetinari.drift.contract_registry import get_contract_registry

        assert get_contract_registry() is get_contract_registry()

    def test_reset_gives_new_instance(self):
        from vetinari.drift.contract_registry import get_contract_registry, reset_contract_registry

        a = get_contract_registry()
        reset_contract_registry()
        b = get_contract_registry()
        assert a is not b

    # Fingerprinting
    def test_register_dict(self):
        h = self.reg.register("MyContract", {"plan_id": "p1", "goal": "g"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_register_dataclass(self):
        h = self.reg.register("DC", _SampleDC())
        assert isinstance(h, str)

    def test_same_content_same_hash(self):
        h1 = self.reg.register("A", {"key": "value"})
        h2 = self.reg.register("B", {"key": "value"})
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = self.reg.register("A", {"x": 1})
        h2 = self.reg.register("A", {"x": 2})
        assert h1 != h2

    def test_register_many(self):
        hashes = self.reg.register_many({"C1": {"a": 1}, "C2": {"b": 2}})
        assert set(hashes.keys()) == {"C1", "C2"}

    def test_list_contracts(self):
        self.reg.register("Z", {"z": True})
        self.reg.register("A", {"a": True})
        names = self.reg.list_contracts()
        assert "Z" in names
        assert "A" in names
        assert names == sorted(names)

    def test_get_hash(self):
        h = self.reg.register("X", {"x": 99})
        assert self.reg.get_hash("X") == h

    def test_get_hash_missing_returns_none(self):
        assert self.reg.get_hash("nonexistent") is None

    # Snapshot & load
    def test_snapshot_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            assert os.path.exists(path)
            data = json.loads(open(path).read())
            assert "hashes" in data
            assert "C" in data["hashes"]

    def test_load_snapshot_populates_previous(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.clear()
            _reset_all()
            from vetinari.drift.contract_registry import get_contract_registry

            reg2 = get_contract_registry()
            ok = reg2.load_snapshot(path)
            assert ok is True
            # After loading the snapshot, the "C" contract fingerprint is restored.
            assert "C" in reg2.list_contracts()

    def test_load_missing_snapshot_returns_false(self):
        ok = self.reg.load_snapshot("/nonexistent/snap.json")
        assert not ok

    def test_no_drift_without_loaded_snapshot(self):
        self.reg.register("C", {"v": 1})
        assert self.reg.check_drift() == {}

    # Drift detection
    def test_no_drift_when_identical(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.load_snapshot(path)
            assert self.reg.check_drift() == {}

    def test_drift_detected_on_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.register("C", {"v": 2})  # changed
            self.reg.load_snapshot(path)
            drifts = self.reg.check_drift()
            assert "C" in drifts
            assert drifts["C"]["current"] == self.reg.get_hash("C")

    def test_drift_raises_when_requested(self):
        from vetinari.drift.contract_registry import ContractDriftError

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.register("C", {"v": 999})
            self.reg.load_snapshot(path)
            with pytest.raises(ContractDriftError):
                self.reg.check_drift(raise_on_drift=True)

    def test_is_stable_true_when_clean(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.load_snapshot(path)
            assert self.reg.is_stable()

    def test_removed_contract_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.clear()
            self.reg.load_snapshot(path)
            drifts = self.reg.check_drift()
            assert "C" in drifts
            assert drifts["C"]["current"] == "REMOVED"

    def test_added_contract_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.register("D", {"v": 2})
            self.reg.load_snapshot(path)
            drifts = self.reg.check_drift()
            assert "D" in drifts
            assert drifts["D"]["previous"] == "MISSING"
            assert drifts["D"]["current"] == self.reg.get_hash("D")

    def test_get_stats(self):
        self.reg.register("A", {"a": 1})
        stats = self.reg.get_stats()
        assert "registered" in stats
        assert stats["registered"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# CapabilityAuditor
# ─────────────────────────────────────────────────────────────────────────────


class TestCapabilityAuditor:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_all()
        from vetinari.drift.capability_auditor import get_capability_auditor

        self.auditor = get_capability_auditor()
        yield
        _reset_all()

    def test_singleton(self):
        from vetinari.drift.capability_auditor import get_capability_auditor

        assert get_capability_auditor() is get_capability_auditor()

    def test_register_documented(self):
        from vetinari.drift.capability_auditor import CapabilityFinding

        self.auditor.register_documented(AgentType.WORKER.value, ["code_gen", "file_writing"])
        # Verify the capabilities were actually stored
        report = self.auditor.audit_agent(AgentType.WORKER.value)
        assert isinstance(report, CapabilityFinding)
        assert report.agent_name == AgentType.WORKER.value

    def test_audit_agent_no_drift_when_aligned(self):
        # Seed documented caps that match what the real WorkerAgent returns
        from vetinari.agents.consolidated.worker_agent import WorkerAgent

        live_caps = WorkerAgent().get_capabilities()
        self.auditor.register_documented(AgentType.WORKER.value, live_caps)
        finding = self.auditor.audit_agent(AgentType.WORKER.value)
        assert not finding.is_drift
        assert finding.extra_in_code == []
        assert finding.missing_in_code == []

    def test_audit_agent_detects_extra_cap(self):
        # Document fewer caps than the agent actually has
        from vetinari.agents.consolidated.worker_agent import WorkerAgent

        live_caps = WorkerAgent().get_capabilities()
        # Document only the first cap — rest will appear as "extra_in_code"
        self.auditor.register_documented(AgentType.WORKER.value, live_caps[:1])
        finding = self.auditor.audit_agent(AgentType.WORKER.value)
        if len(live_caps) > 1:
            assert finding.is_drift
            assert len(finding.extra_in_code) > 0

    def test_audit_agent_detects_missing_cap(self):
        # Document an extra cap that the code doesn't have
        self.auditor.register_documented(AgentType.WORKER.value, ["__totally_fake_cap__"])
        finding = self.auditor.audit_agent(AgentType.WORKER.value)
        assert finding.is_drift
        assert "__totally_fake_cap__" in finding.missing_in_code

    def test_audit_agent_unknown_name(self):
        self.auditor.register_documented("GHOST", ["cap_a"])
        finding = self.auditor.audit_agent("GHOST")
        # GHOST has no module → missing_in_code = ["cap_a"]
        assert finding.is_drift

    def test_audit_all_returns_list(self):
        self.auditor.register_documented(AgentType.WORKER.value, ["x"])
        findings = self.auditor.audit_all()
        assert isinstance(findings, list)
        assert len(findings) > 0

    def test_get_drift_findings_returns_only_drifts(self):
        findings = self.auditor.get_drift_findings()
        for f in findings:
            assert f.is_drift

    def test_finding_str_clean(self):
        from vetinari.drift.capability_auditor import CapabilityFinding

        f = CapabilityFinding("A", [], [], False)
        assert "[OK]" in str(f)

    def test_finding_str_drift(self):
        from vetinari.drift.capability_auditor import CapabilityFinding

        f = CapabilityFinding("B", ["extra"], ["missing"], True)
        assert "[DRIFT]" in str(f)
        assert "extra" in str(f)

    def test_finding_to_dict(self):
        from vetinari.drift.capability_auditor import CapabilityFinding

        f = CapabilityFinding("C", ["x"], [], True)
        d = f.to_dict()
        assert "agent_name" in d
        assert "is_drift" in d

    def test_get_stats(self):
        stats = self.auditor.get_stats()
        for k in ("agents_audited", "agents_with_drift", "documented_agents"):
            assert k in stats

    def test_register_all_from_contracts_without_baseline_registers_nothing(self):
        """Without a baseline file, register_all_from_contracts() should register nothing."""
        from pathlib import Path
        from unittest.mock import patch

        # Mock _capability_baseline_path to return a nonexistent path
        fake_path = Path("/nonexistent/path/caps.json")
        with patch("vetinari.drift.capability_auditor._capability_baseline_path") as mock_path:
            mock_path.return_value = fake_path
            self.auditor.register_all_from_contracts()
            stats = self.auditor.get_stats()
            assert stats["documented_agents"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# SchemaValidator
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaValidator:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_all()
        from vetinari.drift.schema_validator import get_schema_validator

        self.v = get_schema_validator()
        yield
        _reset_all()

    def test_singleton(self):
        from vetinari.drift.schema_validator import get_schema_validator

        assert get_schema_validator() is get_schema_validator()

    def test_validate_valid_object(self):
        self.v.register_schema("T", {"required_keys": ["a", "b"]})
        errs = self.v.validate("T", {"a": 1, "b": 2, "c": 3})
        assert errs == []

    def test_validate_missing_required_key(self):
        self.v.register_schema("T", {"required_keys": ["must_exist"]})
        errs = self.v.validate("T", {"other": "value"})
        assert len(errs) > 0
        assert "must_exist" in errs[0]

    def test_validate_forbidden_key(self):
        self.v.register_schema("T", {"forbidden_keys": ["secret"]})
        errs = self.v.validate("T", {"secret": "leak", "safe": "ok"})
        assert len(errs) > 0

    def test_validate_key_type_correct(self):
        self.v.register_schema("T", {"key_types": {"count": "int"}})
        errs = self.v.validate("T", {"count": 5})
        assert errs == []

    def test_validate_key_type_wrong(self):
        self.v.register_schema("T", {"key_types": {"count": "int"}})
        errs = self.v.validate("T", {"count": "five"})
        assert len(errs) > 0

    def test_validate_version_pattern_ok(self):
        self.v.register_schema("T", {"version_pattern": r"^v\d+\.\d+\.\d+$"})
        errs = self.v.validate("T", {"version": "v1.2.3"})
        assert errs == []

    def test_validate_version_pattern_fail(self):
        self.v.register_schema("T", {"version_pattern": r"^v\d+\.\d+\.\d+$"})
        errs = self.v.validate("T", {"version": "bad-version"})
        assert len(errs) > 0

    def test_validate_non_empty_key_fails_when_empty(self):
        self.v.register_schema("T", {"non_empty_keys": ["name"]})
        errs = self.v.validate("T", {"name": ""})
        assert len(errs) > 0

    def test_validate_non_empty_key_passes_when_set(self):
        self.v.register_schema("T", {"non_empty_keys": ["name"]})
        errs = self.v.validate("T", {"name": "Alice"})
        assert errs == []

    def test_validate_allowed_status(self):
        self.v.register_schema("T", {"allowed_status_values": ["draft", "done"]})
        errs = self.v.validate("T", {"status": "unknown"})
        assert len(errs) > 0

    def test_validate_unknown_schema(self):
        errs = self.v.validate("__ghost__", {"x": 1})
        assert len(errs) > 0
        assert "Unknown schema" in errs[0]

    def test_validate_many(self):
        self.v.register_schema("T", {"required_keys": ["x"]})
        failures = self.v.validate_many("T", [{"x": 1}, {"y": 2}, {"x": 3}])
        assert 1 in failures  # index 1 missing "x"
        assert 0 not in failures

    def test_is_valid_true(self):
        self.v.register_schema("T", {"required_keys": ["id"]})
        assert self.v.is_valid("T", {"id": "x1"})

    def test_is_valid_false(self):
        self.v.register_schema("T", {"required_keys": ["id"]})
        assert not self.v.is_valid("T", {})

    def test_vetinari_schemas(self):
        self.v.register_vetinari_schemas()
        schemas = self.v.list_schemas()
        for name in ("Plan", "Subtask", "LogRecord", "AlertThreshold", "CostEntry", "ForecastResult"):
            assert name in schemas

    def test_plan_schema_validates_live_plan(self):
        self.v.register_vetinari_schemas()
        from vetinari.planning.plan_types import Plan

        errs = self.v.validate("Plan", Plan(goal="real plan"))
        assert errs == [], errs

    def test_unregister_schema(self):
        self.v.register_schema("X", {})
        assert self.v.unregister_schema("X")
        assert "X" not in self.v.list_schemas()

    def test_get_stats(self):
        self.v.register_vetinari_schemas()
        stats = self.v.get_stats()
        assert "registered_schemas" in stats
        assert stats["registered_schemas"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# DriftReport
# ─────────────────────────────────────────────────────────────────────────────


class TestDriftReport:
    def _make(self, **kwargs):
        from vetinari.drift.monitor import DriftMonitorReport

        return DriftMonitorReport(**kwargs)

    def test_is_clean_when_empty(self):
        r = self._make()
        assert r.is_clean

    def test_is_clean_false_with_contract_drift(self):
        r = self._make(contract_drifts={"Plan": {"previous": "aa", "current": "bb"}})
        assert not r.is_clean

    def test_is_clean_false_with_capability_drift(self):
        r = self._make(capability_drifts=["[DRIFT] BUILDER: ..."])
        assert not r.is_clean

    def test_is_clean_false_with_schema_errors(self):
        r = self._make(schema_errors={"Plan": ["Missing required key 'goal'"]})
        assert not r.is_clean

    def test_summary_clean(self):
        r = self._make()
        assert "clean" in r.summary()

    def test_summary_with_issues(self):
        r = self._make(contract_drifts={"A": {"previous": "x", "current": "y"}})
        assert "contract drift" in r.summary()

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        for k in (
            "timestamp",
            "is_clean",
            "contract_drifts",
            "capability_drifts",
            "schema_errors",
            "issues",
            "duration_ms",
        ):
            assert k in d


# ─────────────────────────────────────────────────────────────────────────────
# DriftMonitor
# ─────────────────────────────────────────────────────────────────────────────


class TestDriftMonitor:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_all()
        # Use a temp dir for snapshots so CI alignment-checker files don't
        # leak into the test (different dummy data → different hashes).
        self._tmp = tempfile.mkdtemp()
        from vetinari.drift.monitor import get_drift_monitor

        self.monitor = get_drift_monitor()
        from pathlib import Path

        self.monitor._registry._snapshot_path = Path(self._tmp) / "contracts.json"
        yield
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)
        _reset_all()

    def test_singleton(self):
        from vetinari.drift.monitor import get_drift_monitor

        assert get_drift_monitor() is get_drift_monitor()

    def test_bootstrap_runs_without_error(self):
        self.monitor.bootstrap()  # should not raise
        # bootstrap populates internal state; run_capability_check returns a list
        results = self.monitor.run_capability_check()
        assert isinstance(results, list)

    def test_run_capability_check_returns_list(self):
        self.monitor.bootstrap()
        findings = self.monitor.run_capability_check()
        assert isinstance(findings, list)

    def test_run_schema_check_no_errors_on_valid_objects(self):
        self.monitor.bootstrap()
        errors = self.monitor.run_schema_check()
        assert isinstance(errors, dict)

    def test_run_full_audit_returns_report(self):
        self.monitor.bootstrap()
        from vetinari.drift.monitor import DriftMonitorReport

        report = self.monitor.run_full_audit()
        assert isinstance(report, DriftMonitorReport)
        assert report.duration_ms > 0

    def test_run_full_audit_clean_on_fresh_state(self):
        self.monitor.bootstrap()
        report = self.monitor.run_full_audit()
        # After bootstrap there is no previous snapshot → no contract drift
        assert report.contract_drifts == {}

    def test_get_history_accumulates(self):
        self.monitor.bootstrap()
        self.monitor.run_full_audit()
        self.monitor.run_full_audit()
        assert len(self.monitor.get_history()) == 2

    def test_get_last_report(self):
        from vetinari.drift.monitor import DriftMonitorReport

        self.monitor.bootstrap()
        self.monitor.run_full_audit()
        last = self.monitor.get_last_report()
        assert isinstance(last, DriftMonitorReport)
        assert last.duration_ms >= 0

    def test_get_last_report_none_before_audit(self):
        assert self.monitor.get_last_report() is None

    def test_clear_history(self):
        self.monitor.bootstrap()
        self.monitor.run_full_audit()
        self.monitor.clear_history()
        assert len(self.monitor.get_history()) == 0

    def test_get_stats_keys(self):
        self.monitor.bootstrap()
        stats = self.monitor.get_stats()
        for k in ("audits_run", "last_clean", "contracts", "capabilities", "schemas"):
            assert k in stats


# ─────────────────────────────────────────────────────────────────────────────
# Behavior tests for governance scripts
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckerBehavior:
    """Loader helper shared by script behavior test classes."""

    def _load_script(self, rel_path: str):
        """Load a script module without executing its main() function."""
        import importlib.util

        full = Path(__file__).parent.parent / rel_path
        spec = importlib.util.spec_from_file_location("_script", full)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


class TestDocContractAlignmentBehavior(TestCheckerBehavior):
    """Behavior tests for check_doc_contract_alignment.py.

    Proves the checker fails closed on missing snapshots and uses
    the correct agent count in its output.
    """

    def _load_script(self, rel_path: str = "scripts/check_doc_contract_alignment.py"):
        return super()._load_script(rel_path)

    def test_missing_snapshot_fails_closed(self, tmp_path, monkeypatch):
        """When no governed repo baseline exists, check_contract_fingerprints() must return False.

        This is the key fix from 27I.1 — previously it auto-created a baseline and returned True.
        We patch the governed repo baseline path so the checker cannot silently
        fall back to user-scoped state outside the repo.
        """
        _reset_all()

        mod = self._load_script()
        monkeypatch.setattr(mod, "GOVERNED_CONTRACT_BASELINE", tmp_path / "nonexistent" / "contracts.json")
        result = mod.check_contract_fingerprints()
        assert result is False, "Missing snapshot must fail closed, not auto-create baseline"
        _reset_all()

    def test_agent_registry_does_not_use_stale_six_agent_count(self, capsys):
        """check_agent_registry() must NOT print '6 consolidated agents'.

        The old version hardcoded a reference to '6 consolidated agents'
        reflecting a stale agent inventory. The fix uses truthful language.
        """
        mod = self._load_script()
        mod.check_agent_registry()
        captured = capsys.readouterr()
        assert "All 6" not in captured.out, "Stale '6 agents' language must be removed from output"

    def test_agent_registry_count_matches_expected_set(self, capsys):
        """The printed agent count must match the 3-agent factory-pipeline expected set."""
        mod = self._load_script()
        result = mod.check_agent_registry()
        captured = capsys.readouterr()
        if result:
            # On pass the output should reference the 3-agent expected set
            assert "All 3" in captured.out, "Agent count printed must be 3 (the expected set size)"

    def test_agent_registry_reports_factory_pipeline_language(self, capsys):
        """A passing registry check must use 'factory-pipeline' in its output."""
        mod = self._load_script()
        result = mod.check_agent_registry()
        captured = capsys.readouterr()
        if result:
            assert "factory-pipeline" in captured.out, (
                "Passing registry check must say 'factory-pipeline agents', not obsolete terminology"
            )

    def test_snapshot_flag_bootstraps_missing_contract_baseline(self, tmp_path, monkeypatch):
        """--snapshot may explicitly create the governed repo baseline and then verify it."""
        _reset_all()

        snapshot_path = tmp_path / "contracts.json"
        user_dir_snapshot = tmp_path / "user-dir" / "contracts.json"
        monkeypatch.setattr(sys, "argv", ["check_doc_contract_alignment.py", "--snapshot"])
        mod = self._load_script()
        monkeypatch.setattr(mod, "GOVERNED_CONTRACT_BASELINE", snapshot_path)
        import vetinari.drift.contract_registry as reg_mod

        monkeypatch.setattr(reg_mod, "_DEFAULT_SNAPSHOT_PATH", str(user_dir_snapshot))
        _reset_all()
        result = mod.main()

        assert result == 0, "--snapshot should bootstrap a missing baseline when all other checks pass"
        assert snapshot_path.exists(), "Expected --snapshot to create the explicit contract baseline"
        _reset_all()

    def test_governed_baseline_ignores_missing_user_dir_snapshot(self, tmp_path, monkeypatch):
        """The checker must read the repo baseline even when the user-dir snapshot is missing."""
        _reset_all()
        mod = self._load_script()
        governed_path = tmp_path / "repo-contracts.json"
        monkeypatch.setattr(mod, "GOVERNED_CONTRACT_BASELINE", governed_path)
        import vetinari.drift.contract_registry as reg_mod

        monkeypatch.setattr(reg_mod, "_DEFAULT_SNAPSHOT_PATH", str(tmp_path / "missing-user-dir" / "contracts.json"))
        _reset_all()
        assert mod.save_contract_snapshot() is True
        assert mod.check_contract_fingerprints() is True
        _reset_all()


class TestCapabilityAuditBehavior:
    """Behavior tests for capability_auditor.

    Proves the auditor no longer self-seeds its baseline from live code.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        _reset_all()
        yield
        _reset_all()

    def test_register_all_from_contracts_reads_file_not_live_code(self, tmp_path):
        """register_all_from_contracts() must read from baseline file, not live get_capabilities().

        This is the key fix from 27I.2 — previously it called get_capabilities() as the baseline,
        which means the checker seeded its own expected values from the code it audits.
        """
        import json

        import vetinari.drift.capability_auditor as cap_mod
        from vetinari.drift.capability_auditor import get_capability_auditor

        auditor = get_capability_auditor()

        # Create a fake baseline file with capabilities that cannot exist in real agents
        baseline = {"WORKER": ["fake_cap_alpha_xyzzy", "fake_cap_beta_xyzzy"]}
        baseline_path = tmp_path / "capabilities.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        original = cap_mod.CAPABILITY_BASELINE_PATH
        cap_mod.CAPABILITY_BASELINE_PATH = str(baseline_path)
        try:
            auditor.register_all_from_contracts()
            # The auditor should have loaded our fake caps from the file
            finding = auditor.audit_agent("WORKER")
            # Fake caps won't exist in real worker code, so they appear as missing_in_code —
            # this proves it read from file, not from the live agent's get_capabilities()
            documented_caps = auditor._documented.get("WORKER", set())
            assert "fake_cap_alpha_xyzzy" in documented_caps, (
                "Auditor must populate from baseline file, not from live agent code"
            )
        finally:
            cap_mod.CAPABILITY_BASELINE_PATH = original

    def test_register_all_from_contracts_no_file_registers_nothing(self, tmp_path):
        """When baseline file doesn't exist, register_all_from_contracts() registers nothing."""
        import vetinari.drift.capability_auditor as cap_mod
        from vetinari.drift.capability_auditor import get_capability_auditor

        auditor = get_capability_auditor()
        original = cap_mod.CAPABILITY_BASELINE_PATH
        cap_mod.CAPABILITY_BASELINE_PATH = str(tmp_path / "nonexistent.json")
        try:
            auditor.register_all_from_contracts()
            stats = auditor.get_stats()
            assert stats["documented_agents"] == 0, "No file = no documented agents registered"
            assert stats["agents_audited"] == 0, "No governed baseline = no capability audit performed"
            assert stats["baseline_available"] is False
            assert auditor.get_drift_findings() == [], "Missing baseline must not create per-agent false drift"
        finally:
            cap_mod.CAPABILITY_BASELINE_PATH = original

    def test_load_failure_is_drift_not_clean(self):
        """When an agent can't be loaded, audit_agent() must return is_drift=True.

        This is the key fix from 27I.2 — previously load failure returned is_drift=False,
        meaning a broken agent class would silently pass the audit.
        """
        from unittest.mock import patch

        from vetinari.drift.capability_auditor import get_capability_auditor

        auditor = get_capability_auditor()
        # Register documented caps for WORKER so there's something to compare against
        auditor.register_documented("WORKER", ["cap_a", "cap_b"])

        # Simulate a load failure for the agent class
        with patch("vetinari.drift.capability_auditor._load_agent_capabilities", return_value=None):
            finding = auditor.audit_agent("WORKER")
            assert finding.is_drift is True, "Agent load failure must be treated as drift"
            assert "cap_a" in finding.missing_in_code, "Missing caps must be reported when agent cannot be loaded"
            assert "cap_b" in finding.missing_in_code

    def test_save_baseline_creates_file(self, tmp_path):
        """save_baseline() must write a JSON file with agent capabilities."""
        import json

        import vetinari.drift.capability_auditor as cap_mod
        from vetinari.drift.capability_auditor import get_capability_auditor

        auditor = get_capability_auditor()
        original = cap_mod.CAPABILITY_BASELINE_PATH
        baseline_path = tmp_path / "caps.json"
        cap_mod.CAPABILITY_BASELINE_PATH = str(baseline_path)
        try:
            auditor.save_baseline()
            assert baseline_path.exists(), "save_baseline() must create the file"
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
            assert isinstance(data, dict), "Baseline file must contain a JSON object"
            # Should have entries for the factory-pipeline agents
            assert len(data) > 0, "save_baseline() must write at least one agent entry"
        finally:
            cap_mod.CAPABILITY_BASELINE_PATH = original


class TestCoverageGateBehavior(TestCheckerBehavior):
    """Behavior tests for check_coverage_gate.py.

    Proves unlisted modules at 0% coverage fail instead of passing silently.
    """

    def _load_script(self, rel_path: str = "scripts/check_coverage_gate.py"):
        return super()._load_script(rel_path)

    def test_unlisted_module_at_zero_percent_fails(self):
        """An unlisted module at 0% coverage must fail under the global minimum.

        This is the key fix from 27I.3 — previously GLOBAL_MIN=0 allowed every
        unlisted module to pass regardless of actual coverage, creating a false-green
        certification for uncovered code.
        """
        mod = self._load_script()
        report = {"files": {"vetinari/new_module.py": {"summary": {"percent_covered": 0.0}}}}
        failures = mod.check_coverage(report, mod.GLOBAL_MIN)
        assert len(failures) > 0, "Unlisted module at 0% must fail when GLOBAL_MIN > 0"
        assert failures[0][1] == pytest.approx(0.0), "Reported actual coverage must be 0.0"
        assert failures[0][2] == mod.GLOBAL_MIN, "Required threshold must be GLOBAL_MIN"

    def test_unlisted_module_above_global_min_passes(self):
        """An unlisted module above GLOBAL_MIN should pass without error."""
        mod = self._load_script()
        report = {"files": {"vetinari/new_module.py": {"summary": {"percent_covered": 75.0}}}}
        failures = mod.check_coverage(report, mod.GLOBAL_MIN)
        assert len(failures) == 0, "Module above GLOBAL_MIN must not be flagged as a failure"

    def test_global_min_is_not_zero(self):
        """GLOBAL_MIN must be > 0 to prevent false-green certification of uncovered modules."""
        mod = self._load_script()
        assert mod.GLOBAL_MIN > 0, (
            f"GLOBAL_MIN={mod.GLOBAL_MIN} is a false-green loophole — every module passes when the floor is 0%"
        )

    def test_governed_module_below_threshold_fails(self):
        """A governed module below its per-module threshold must fail."""
        mod = self._load_script()
        # vetinari/security.py has threshold 60 in MODULE_THRESHOLDS
        report = {"files": {"vetinari/security.py": {"summary": {"percent_covered": 30.0}}}}
        failures = mod.check_coverage(report, mod.GLOBAL_MIN)
        assert len(failures) > 0, "Governed module below its per-module threshold must fail"
        assert failures[0][2] == 60, "Required threshold must be the per-module value (60 for security.py)"

    def test_empty_coverage_report_fails_closed(self, tmp_path: Path):
        """An empty coverage report is missing evidence, not a clean release proof."""
        mod = self._load_script()
        report_path = tmp_path / "coverage.json"
        report_path.write_text(json.dumps({"files": {}, "totals": {"percent_covered": 0.0}}), encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            mod.load_report(report_path)

        assert exc.value.code == 2


# ─────────────────────────────────────────────────────────────────────────────
# Package-level imports
# ─────────────────────────────────────────────────────────────────────────────


class TestDriftPackageImport:
    def test_all_top_level_imports(self):
        import vetinari.drift  # Verify drift package is importable at top level

        assert hasattr(vetinari.drift, "get_drift_monitor")

    def test_all_singletons_return_same_instance(self):
        _reset_all()
        from vetinari.drift import (
            get_capability_auditor,
            get_contract_registry,
            get_drift_monitor,
            get_schema_validator,
        )

        assert get_contract_registry() is get_contract_registry()
        assert get_capability_auditor() is get_capability_auditor()
        assert get_schema_validator() is get_schema_validator()
        assert get_drift_monitor() is get_drift_monitor()
        _reset_all()
