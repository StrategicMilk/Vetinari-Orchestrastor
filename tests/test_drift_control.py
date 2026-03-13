"""
Tests for Phase 7 Drift Control — vetinari/drift/

Coverage:
    ContractRegistry      — register, fingerprint, snapshot, load, check_drift
    CapabilityAuditor     — register_documented, audit_agent, audit_all, drift findings
    SchemaValidator       — register_schema, validate, validate_many, vetinari schemas
    DriftMonitor          — bootstrap, individual checks, run_full_audit, report
    DriftReport           — is_clean, summary, to_dict
    Scripts (import-level) — check that all four scripts are importable
"""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass

import pytest

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


# ─────────────────────────────────────────────────────────────────────────────
# ContractRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestContractRegistry(unittest.TestCase):

    def setUp(self):
        _reset_all()
        from vetinari.drift.contract_registry import get_contract_registry
        self.reg = get_contract_registry()

    def tearDown(self):
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
        assert len(h) == 64   # SHA-256 hex

    def test_register_dataclass(self):
        @dataclass
        class DC:
            x: int = 1
            y: str = "hello"
        h = self.reg.register("DC", DC())
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
            assert ok

    def test_load_missing_snapshot_returns_false(self):
        ok = self.reg.load_snapshot("/nonexistent/snap.json")
        assert not ok

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
            self.reg.register("C", {"v": 2})   # changed
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

    def test_get_stats(self):
        self.reg.register("A", {"a": 1})
        stats = self.reg.get_stats()
        assert "registered" in stats
        assert stats["registered"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# CapabilityAuditor
# ─────────────────────────────────────────────────────────────────────────────

class TestCapabilityAuditor(unittest.TestCase):

    def setUp(self):
        _reset_all()
        from vetinari.drift.capability_auditor import get_capability_auditor
        self.auditor = get_capability_auditor()

    def tearDown(self):
        _reset_all()

    def test_singleton(self):
        from vetinari.drift.capability_auditor import get_capability_auditor
        assert get_capability_auditor() is get_capability_auditor()

    def test_register_documented(self):
        self.auditor.register_documented("BUILDER", ["code_gen", "file_writing"])
        # No exception → pass

    def test_audit_agent_no_drift_when_aligned(self):
        # Seed documented caps that match what the real Builder returns
        from vetinari.agents.builder_agent import BuilderAgent
        live_caps = BuilderAgent().get_capabilities()
        self.auditor.register_documented("BUILDER", live_caps)
        finding = self.auditor.audit_agent("BUILDER")
        assert not finding.is_drift
        assert finding.extra_in_code == []
        assert finding.missing_in_code == []

    def test_audit_agent_detects_extra_cap(self):
        # Document fewer caps than the agent actually has
        from vetinari.agents.builder_agent import BuilderAgent
        live_caps = BuilderAgent().get_capabilities()
        # Document only the first cap — rest will appear as "extra_in_code"
        self.auditor.register_documented("BUILDER", live_caps[:1])
        finding = self.auditor.audit_agent("BUILDER")
        if len(live_caps) > 1:
            assert finding.is_drift
            assert len(finding.extra_in_code) > 0

    def test_audit_agent_detects_missing_cap(self):
        # Document an extra cap that the code doesn't have
        self.auditor.register_documented("BUILDER",
                                          ["__totally_fake_cap__"])
        finding = self.auditor.audit_agent("BUILDER")
        assert finding.is_drift
        assert "__totally_fake_cap__" in finding.missing_in_code

    def test_audit_agent_unknown_name(self):
        self.auditor.register_documented("GHOST", ["cap_a"])
        finding = self.auditor.audit_agent("GHOST")
        # GHOST has no module → missing_in_code = ["cap_a"]
        assert finding.is_drift

    def test_audit_all_returns_list(self):
        self.auditor.register_documented("BUILDER", ["x"])
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

    def test_register_all_from_contracts(self):
        # Should not raise; populates documented map
        self.auditor.register_all_from_contracts()
        stats = self.auditor.get_stats()
        assert stats["documented_agents"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# SchemaValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaValidator(unittest.TestCase):

    def setUp(self):
        _reset_all()
        from vetinari.drift.schema_validator import get_schema_validator
        self.v = get_schema_validator()

    def tearDown(self):
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
        assert 1 in failures   # index 1 missing "x"
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
        for name in ("Plan", "Subtask", "LogRecord", "AlertThreshold",
                     "CostEntry", "ForecastResult"):
            assert name in schemas

    def test_plan_schema_validates_live_plan(self):
        self.v.register_vetinari_schemas()
        from vetinari.plan_types import Plan
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

class TestDriftReport(unittest.TestCase):

    def _make(self, **kwargs):
        from vetinari.drift.monitor import DriftReport
        return DriftReport(**kwargs)

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
        for k in ("timestamp", "is_clean", "contract_drifts",
                  "capability_drifts", "schema_errors", "issues", "duration_ms"):
            assert k in d


# ─────────────────────────────────────────────────────────────────────────────
# DriftMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftMonitor(unittest.TestCase):

    def setUp(self):
        _reset_all()
        # Use a temp dir for snapshots so CI alignment-checker files don't
        # leak into the test (different dummy data → different hashes).
        self._tmp = tempfile.mkdtemp()
        from vetinari.drift.monitor import get_drift_monitor
        self.monitor = get_drift_monitor()
        from pathlib import Path
        self.monitor._registry._snapshot_path = Path(self._tmp) / "contracts.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)
        _reset_all()

    def test_singleton(self):
        from vetinari.drift.monitor import get_drift_monitor
        assert get_drift_monitor() is get_drift_monitor()

    def test_bootstrap_runs_without_error(self):
        self.monitor.bootstrap()   # should not raise

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
        from vetinari.drift.monitor import DriftReport
        report = self.monitor.run_full_audit()
        assert isinstance(report, DriftReport)
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
        self.monitor.bootstrap()
        self.monitor.run_full_audit()
        last = self.monitor.get_last_report()
        assert last is not None

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
        for k in ("audits_run", "last_clean", "contracts",
                  "capabilities", "schemas"):
            assert k in stats


# ─────────────────────────────────────────────────────────────────────────────
# Script import tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScriptImports(unittest.TestCase):
    """Verify all drift scripts are syntactically valid and importable."""

    def _load(self, rel_path: str):
        import importlib.util
        from pathlib import Path
        full = Path(__file__).parent.parent / rel_path
        spec = importlib.util.spec_from_file_location("_script", full)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_check_doc_contract_alignment_importable(self):
        mod = self._load("scripts/check_doc_contract_alignment.py")
        assert callable(getattr(mod, "main", None))

    def test_check_migration_index_importable(self):
        mod = self._load("scripts/check_migration_index.py")
        assert callable(getattr(mod, "main", None))

    def test_check_agent_capabilities_importable(self):
        mod = self._load("scripts/check_agent_capabilities.py")
        assert callable(getattr(mod, "main", None))

    def test_check_coverage_gate_importable(self):
        mod = self._load("scripts/check_coverage_gate.py")
        assert callable(getattr(mod, "main", None))


# ─────────────────────────────────────────────────────────────────────────────
# Package-level imports
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftPackageImport(unittest.TestCase):

    def test_all_top_level_imports(self):
        import vetinari.drift  # Verify drift package is importable at top level

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


if __name__ == "__main__":
    unittest.main()
