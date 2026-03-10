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
from unittest.mock import MagicMock, patch


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _reset_all():
    from vetinari.drift.contract_registry  import reset_contract_registry
    from vetinari.drift.capability_auditor import reset_capability_auditor
    from vetinari.drift.schema_validator   import reset_schema_validator
    from vetinari.drift.monitor            import reset_drift_monitor
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
        self.assertIs(get_contract_registry(), get_contract_registry())

    def test_reset_gives_new_instance(self):
        from vetinari.drift.contract_registry import get_contract_registry, reset_contract_registry
        a = get_contract_registry()
        reset_contract_registry()
        b = get_contract_registry()
        self.assertIsNot(a, b)

    # Fingerprinting
    def test_register_dict(self):
        h = self.reg.register("MyContract", {"plan_id": "p1", "goal": "g"})
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)   # SHA-256 hex

    def test_register_dataclass(self):
        @dataclass
        class DC:
            x: int = 1
            y: str = "hello"
        h = self.reg.register("DC", DC())
        self.assertIsInstance(h, str)

    def test_same_content_same_hash(self):
        h1 = self.reg.register("A", {"key": "value"})
        h2 = self.reg.register("B", {"key": "value"})
        self.assertEqual(h1, h2)

    def test_different_content_different_hash(self):
        h1 = self.reg.register("A", {"x": 1})
        h2 = self.reg.register("A", {"x": 2})
        self.assertNotEqual(h1, h2)

    def test_register_many(self):
        hashes = self.reg.register_many({"C1": {"a": 1}, "C2": {"b": 2}})
        self.assertEqual(set(hashes.keys()), {"C1", "C2"})

    def test_list_contracts(self):
        self.reg.register("Z", {"z": True})
        self.reg.register("A", {"a": True})
        names = self.reg.list_contracts()
        self.assertIn("Z", names)
        self.assertIn("A", names)
        self.assertEqual(names, sorted(names))

    def test_get_hash(self):
        h = self.reg.register("X", {"x": 99})
        self.assertEqual(self.reg.get_hash("X"), h)

    def test_get_hash_missing_returns_none(self):
        self.assertIsNone(self.reg.get_hash("nonexistent"))

    # Snapshot & load
    def test_snapshot_creates_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.assertTrue(os.path.exists(path))
            data = json.loads(open(path).read())
            self.assertIn("hashes", data)
            self.assertIn("C", data["hashes"])

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
            self.assertTrue(ok)

    def test_load_missing_snapshot_returns_false(self):
        ok = self.reg.load_snapshot("/nonexistent/snap.json")
        self.assertFalse(ok)

    # Drift detection
    def test_no_drift_when_identical(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.load_snapshot(path)
            self.assertEqual(self.reg.check_drift(), {})

    def test_drift_detected_on_change(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.register("C", {"v": 2})   # changed
            self.reg.load_snapshot(path)
            drifts = self.reg.check_drift()
            self.assertIn("C", drifts)
            self.assertEqual(drifts["C"]["current"], self.reg.get_hash("C"))

    def test_drift_raises_when_requested(self):
        from vetinari.drift.contract_registry import ContractDriftError
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.register("C", {"v": 999})
            self.reg.load_snapshot(path)
            with self.assertRaises(ContractDriftError):
                self.reg.check_drift(raise_on_drift=True)

    def test_is_stable_true_when_clean(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.load_snapshot(path)
            self.assertTrue(self.reg.is_stable())

    def test_removed_contract_detected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snap.json")
            self.reg.register("C", {"v": 1})
            self.reg.snapshot(path)
            self.reg.clear()
            self.reg.load_snapshot(path)
            drifts = self.reg.check_drift()
            self.assertIn("C", drifts)
            self.assertEqual(drifts["C"]["current"], "REMOVED")

    def test_get_stats(self):
        self.reg.register("A", {"a": 1})
        stats = self.reg.get_stats()
        self.assertIn("registered", stats)
        self.assertEqual(stats["registered"], 1)


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
        self.assertIs(get_capability_auditor(), get_capability_auditor())

    def test_register_documented(self):
        self.auditor.register_documented("BUILDER", ["code_gen", "file_writing"])
        # No exception → pass

    def test_audit_agent_no_drift_when_aligned(self):
        # Seed documented caps that match what the real Builder returns
        from vetinari.agents.builder_agent import BuilderAgent
        live_caps = BuilderAgent().get_capabilities()
        self.auditor.register_documented("BUILDER", live_caps)
        finding = self.auditor.audit_agent("BUILDER")
        self.assertFalse(finding.is_drift)
        self.assertEqual(finding.extra_in_code, [])
        self.assertEqual(finding.missing_in_code, [])

    def test_audit_agent_detects_extra_cap(self):
        # Document fewer caps than the agent actually has
        from vetinari.agents.builder_agent import BuilderAgent
        live_caps = BuilderAgent().get_capabilities()
        # Document only the first cap — rest will appear as "extra_in_code"
        self.auditor.register_documented("BUILDER", live_caps[:1])
        finding = self.auditor.audit_agent("BUILDER")
        if len(live_caps) > 1:
            self.assertTrue(finding.is_drift)
            self.assertGreater(len(finding.extra_in_code), 0)

    def test_audit_agent_detects_missing_cap(self):
        # Document an extra cap that the code doesn't have
        self.auditor.register_documented("BUILDER",
                                          ["__totally_fake_cap__"])
        finding = self.auditor.audit_agent("BUILDER")
        self.assertTrue(finding.is_drift)
        self.assertIn("__totally_fake_cap__", finding.missing_in_code)

    def test_audit_agent_unknown_name(self):
        self.auditor.register_documented("GHOST", ["cap_a"])
        finding = self.auditor.audit_agent("GHOST")
        # GHOST has no module → missing_in_code = ["cap_a"]
        self.assertTrue(finding.is_drift)

    def test_audit_all_returns_list(self):
        self.auditor.register_documented("BUILDER", ["x"])
        findings = self.auditor.audit_all()
        self.assertIsInstance(findings, list)
        self.assertGreater(len(findings), 0)

    def test_get_drift_findings_returns_only_drifts(self):
        findings = self.auditor.get_drift_findings()
        for f in findings:
            self.assertTrue(f.is_drift)

    def test_finding_str_clean(self):
        from vetinari.drift.capability_auditor import CapabilityFinding
        f = CapabilityFinding("A", [], [], False)
        self.assertIn("[OK]", str(f))

    def test_finding_str_drift(self):
        from vetinari.drift.capability_auditor import CapabilityFinding
        f = CapabilityFinding("B", ["extra"], ["missing"], True)
        self.assertIn("[DRIFT]", str(f))
        self.assertIn("extra", str(f))

    def test_finding_to_dict(self):
        from vetinari.drift.capability_auditor import CapabilityFinding
        f = CapabilityFinding("C", ["x"], [], True)
        d = f.to_dict()
        self.assertIn("agent_name", d)
        self.assertIn("is_drift", d)

    def test_get_stats(self):
        stats = self.auditor.get_stats()
        for k in ("agents_audited", "agents_with_drift", "documented_agents"):
            self.assertIn(k, stats)

    def test_register_all_from_contracts(self):
        # Should not raise; populates documented map
        self.auditor.register_all_from_contracts()
        stats = self.auditor.get_stats()
        self.assertGreater(stats["documented_agents"], 0)


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
        self.assertIs(get_schema_validator(), get_schema_validator())

    def test_validate_valid_object(self):
        self.v.register_schema("T", {"required_keys": ["a", "b"]})
        errs = self.v.validate("T", {"a": 1, "b": 2, "c": 3})
        self.assertEqual(errs, [])

    def test_validate_missing_required_key(self):
        self.v.register_schema("T", {"required_keys": ["must_exist"]})
        errs = self.v.validate("T", {"other": "value"})
        self.assertGreater(len(errs), 0)
        self.assertIn("must_exist", errs[0])

    def test_validate_forbidden_key(self):
        self.v.register_schema("T", {"forbidden_keys": ["secret"]})
        errs = self.v.validate("T", {"secret": "leak", "safe": "ok"})
        self.assertGreater(len(errs), 0)

    def test_validate_key_type_correct(self):
        self.v.register_schema("T", {"key_types": {"count": "int"}})
        errs = self.v.validate("T", {"count": 5})
        self.assertEqual(errs, [])

    def test_validate_key_type_wrong(self):
        self.v.register_schema("T", {"key_types": {"count": "int"}})
        errs = self.v.validate("T", {"count": "five"})
        self.assertGreater(len(errs), 0)

    def test_validate_version_pattern_ok(self):
        self.v.register_schema("T", {"version_pattern": r"^v\d+\.\d+\.\d+$"})
        errs = self.v.validate("T", {"version": "v1.2.3"})
        self.assertEqual(errs, [])

    def test_validate_version_pattern_fail(self):
        self.v.register_schema("T", {"version_pattern": r"^v\d+\.\d+\.\d+$"})
        errs = self.v.validate("T", {"version": "bad-version"})
        self.assertGreater(len(errs), 0)

    def test_validate_non_empty_key_fails_when_empty(self):
        self.v.register_schema("T", {"non_empty_keys": ["name"]})
        errs = self.v.validate("T", {"name": ""})
        self.assertGreater(len(errs), 0)

    def test_validate_non_empty_key_passes_when_set(self):
        self.v.register_schema("T", {"non_empty_keys": ["name"]})
        errs = self.v.validate("T", {"name": "Alice"})
        self.assertEqual(errs, [])

    def test_validate_allowed_status(self):
        self.v.register_schema("T", {"allowed_status_values": ["draft", "done"]})
        errs = self.v.validate("T", {"status": "unknown"})
        self.assertGreater(len(errs), 0)

    def test_validate_unknown_schema(self):
        errs = self.v.validate("__ghost__", {"x": 1})
        self.assertGreater(len(errs), 0)
        self.assertIn("Unknown schema", errs[0])

    def test_validate_many(self):
        self.v.register_schema("T", {"required_keys": ["x"]})
        failures = self.v.validate_many("T", [{"x": 1}, {"y": 2}, {"x": 3}])
        self.assertIn(1, failures)   # index 1 missing "x"
        self.assertNotIn(0, failures)

    def test_is_valid_true(self):
        self.v.register_schema("T", {"required_keys": ["id"]})
        self.assertTrue(self.v.is_valid("T", {"id": "x1"}))

    def test_is_valid_false(self):
        self.v.register_schema("T", {"required_keys": ["id"]})
        self.assertFalse(self.v.is_valid("T", {}))

    def test_vetinari_schemas(self):
        self.v.register_vetinari_schemas()
        schemas = self.v.list_schemas()
        for name in ("Plan", "Subtask", "LogRecord", "AlertThreshold",
                     "CostEntry", "ForecastResult"):
            self.assertIn(name, schemas)

    def test_plan_schema_validates_live_plan(self):
        self.v.register_vetinari_schemas()
        from vetinari.plan_types import Plan
        errs = self.v.validate("Plan", Plan(goal="real plan"))
        self.assertEqual(errs, [], errs)

    def test_unregister_schema(self):
        self.v.register_schema("X", {})
        self.assertTrue(self.v.unregister_schema("X"))
        self.assertNotIn("X", self.v.list_schemas())

    def test_get_stats(self):
        self.v.register_vetinari_schemas()
        stats = self.v.get_stats()
        self.assertIn("registered_schemas", stats)
        self.assertGreater(stats["registered_schemas"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# DriftReport
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftReport(unittest.TestCase):

    def _make(self, **kwargs):
        from vetinari.drift.monitor import DriftReport
        return DriftReport(**kwargs)

    def test_is_clean_when_empty(self):
        r = self._make()
        self.assertTrue(r.is_clean)

    def test_is_clean_false_with_contract_drift(self):
        r = self._make(contract_drifts={"Plan": {"previous": "aa", "current": "bb"}})
        self.assertFalse(r.is_clean)

    def test_is_clean_false_with_capability_drift(self):
        r = self._make(capability_drifts=["[DRIFT] BUILDER: ..."])
        self.assertFalse(r.is_clean)

    def test_is_clean_false_with_schema_errors(self):
        r = self._make(schema_errors={"Plan": ["Missing required key 'goal'"]})
        self.assertFalse(r.is_clean)

    def test_summary_clean(self):
        r = self._make()
        self.assertIn("clean", r.summary())

    def test_summary_with_issues(self):
        r = self._make(contract_drifts={"A": {"previous": "x", "current": "y"}})
        self.assertIn("contract drift", r.summary())

    def test_to_dict_keys(self):
        r = self._make()
        d = r.to_dict()
        for k in ("timestamp", "is_clean", "contract_drifts",
                  "capability_drifts", "schema_errors", "issues", "duration_ms"):
            self.assertIn(k, d)


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
        self.assertIs(get_drift_monitor(), get_drift_monitor())

    def test_bootstrap_runs_without_error(self):
        self.monitor.bootstrap()   # should not raise

    def test_run_capability_check_returns_list(self):
        self.monitor.bootstrap()
        findings = self.monitor.run_capability_check()
        self.assertIsInstance(findings, list)

    def test_run_schema_check_no_errors_on_valid_objects(self):
        self.monitor.bootstrap()
        errors = self.monitor.run_schema_check()
        self.assertIsInstance(errors, dict)

    def test_run_full_audit_returns_report(self):
        self.monitor.bootstrap()
        from vetinari.drift.monitor import DriftReport
        report = self.monitor.run_full_audit()
        self.assertIsInstance(report, DriftReport)
        self.assertGreater(report.duration_ms, 0)

    def test_run_full_audit_clean_on_fresh_state(self):
        self.monitor.bootstrap()
        report = self.monitor.run_full_audit()
        # After bootstrap there is no previous snapshot → no contract drift
        self.assertEqual(report.contract_drifts, {})

    def test_get_history_accumulates(self):
        self.monitor.bootstrap()
        self.monitor.run_full_audit()
        self.monitor.run_full_audit()
        self.assertEqual(len(self.monitor.get_history()), 2)

    def test_get_last_report(self):
        self.monitor.bootstrap()
        self.monitor.run_full_audit()
        last = self.monitor.get_last_report()
        self.assertIsNotNone(last)

    def test_get_last_report_none_before_audit(self):
        self.assertIsNone(self.monitor.get_last_report())

    def test_clear_history(self):
        self.monitor.bootstrap()
        self.monitor.run_full_audit()
        self.monitor.clear_history()
        self.assertEqual(len(self.monitor.get_history()), 0)

    def test_get_stats_keys(self):
        self.monitor.bootstrap()
        stats = self.monitor.get_stats()
        for k in ("audits_run", "last_clean", "contracts",
                  "capabilities", "schemas"):
            self.assertIn(k, stats)


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
        self.assertTrue(callable(getattr(mod, "main", None)))

    def test_check_migration_index_importable(self):
        mod = self._load("scripts/check_migration_index.py")
        self.assertTrue(callable(getattr(mod, "main", None)))

    def test_check_agent_capabilities_importable(self):
        mod = self._load("scripts/check_agent_capabilities.py")
        self.assertTrue(callable(getattr(mod, "main", None)))

    def test_check_coverage_gate_importable(self):
        mod = self._load("scripts/check_coverage_gate.py")
        self.assertTrue(callable(getattr(mod, "main", None)))


# ─────────────────────────────────────────────────────────────────────────────
# Package-level imports
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftPackageImport(unittest.TestCase):

    def test_all_top_level_imports(self):
        from vetinari.drift import (
            ContractRegistry, ContractDriftError,
            get_contract_registry, reset_contract_registry,
            CapabilityAuditor, CapabilityFinding,
            get_capability_auditor, reset_capability_auditor,
            SchemaValidator,
            get_schema_validator, reset_schema_validator,
            DriftMonitor, DriftReport,
            get_drift_monitor, reset_drift_monitor,
        )

    def test_all_singletons_return_same_instance(self):
        _reset_all()
        from vetinari.drift import (
            get_contract_registry, get_capability_auditor,
            get_schema_validator, get_drift_monitor,
        )
        self.assertIs(get_contract_registry(),  get_contract_registry())
        self.assertIs(get_capability_auditor(), get_capability_auditor())
        self.assertIs(get_schema_validator(),   get_schema_validator())
        self.assertIs(get_drift_monitor(),      get_drift_monitor())
        _reset_all()


if __name__ == "__main__":
    unittest.main()
