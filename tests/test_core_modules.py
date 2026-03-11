"""
Comprehensive tests for 6 core Vetinari modules:
  1. vetinari.adr
  2. vetinari.builder
  3. vetinari.credentials
  4. vetinari.exceptions
  5. vetinari.validator
  6. vetinari.upgrader
"""

import sys

# Clean stubs that may have been created by other test files
for _stubname in (
    "vetinari.adr",
    "vetinari.builder",
    "vetinari.credentials",
    "vetinari.exceptions",
    "vetinari.validator",
    "vetinari.upgrader",
):
    sys.modules.pop(_stubname, None)

import ast
import importlib.util
import json
import os
import zipfile
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# 1. vetinari.adr
# ---------------------------------------------------------------------------

from vetinari.adr import ADR, ADRCategory, ADRProposal, ADRStatus, ADRSystem, HIGH_STAKES_CATEGORIES


class TestADRDataclasses:
    """Tests for ADR enums, dataclass, and ADRProposal."""

    def test_adr_status_values(self):
        assert ADRStatus.PROPOSED.value == "proposed"
        assert ADRStatus.ACCEPTED.value == "accepted"
        assert ADRStatus.REJECTED.value == "rejected"
        assert ADRStatus.DEPRECATED.value == "deprecated"
        assert ADRStatus.SUPERSEDED.value == "superseded"

    def test_adr_status_has_five_members(self):
        assert len(ADRStatus) == 5

    def test_adr_category_values(self):
        assert ADRCategory.ARCHITECTURE.value == "architecture"
        assert ADRCategory.SECURITY.value == "security"
        assert ADRCategory.DATA_FLOW.value == "data_flow"
        assert ADRCategory.API_DESIGN.value == "api_design"
        assert ADRCategory.AGENT_DESIGN.value == "agent_design"
        assert ADRCategory.DECOMPOSITION.value == "decomposition"
        assert ADRCategory.PERFORMANCE.value == "performance"
        assert ADRCategory.INTEGRATION.value == "integration"

    def test_adr_category_has_eight_members(self):
        assert len(ADRCategory) == 8

    def test_high_stakes_categories_content(self):
        assert ADRCategory.ARCHITECTURE in HIGH_STAKES_CATEGORIES
        assert ADRCategory.SECURITY in HIGH_STAKES_CATEGORIES
        assert ADRCategory.DATA_FLOW in HIGH_STAKES_CATEGORIES
        assert ADRCategory.PERFORMANCE not in HIGH_STAKES_CATEGORIES
        assert ADRCategory.API_DESIGN not in HIGH_STAKES_CATEGORIES

    def test_high_stakes_categories_length(self):
        assert len(HIGH_STAKES_CATEGORIES) == 3

    def test_adr_to_dict_roundtrip(self):
        adr = ADR(
            adr_id="ADR-0001",
            title="Test ADR",
            category="architecture",
            context="context here",
            decision="decision here",
            status="proposed",
            consequences="consequences here",
            related_adrs=["ADR-0002"],
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
            created_by="tester",
            notes="some notes",
        )
        d = adr.to_dict()
        assert d["adr_id"] == "ADR-0001"
        assert d["title"] == "Test ADR"
        assert d["category"] == "architecture"
        assert d["related_adrs"] == ["ADR-0002"]
        assert d["created_by"] == "tester"

    def test_adr_from_dict_full(self):
        data = {
            "adr_id": "ADR-0099",
            "title": "From Dict",
            "category": "security",
            "context": "ctx",
            "decision": "dec",
            "status": "accepted",
            "consequences": "cons",
            "related_adrs": ["ADR-0001"],
            "created_at": "2025-06-01",
            "updated_at": "2025-06-02",
            "created_by": "admin",
            "notes": "n",
        }
        adr = ADR.from_dict(data)
        assert adr.adr_id == "ADR-0099"
        assert adr.status == "accepted"
        assert adr.category == "security"
        assert adr.related_adrs == ["ADR-0001"]

    def test_adr_from_dict_defaults(self):
        adr = ADR.from_dict({})
        assert adr.adr_id == ""
        assert adr.category == "architecture"
        assert adr.status == "proposed"
        assert adr.created_by == "system"
        assert adr.related_adrs == []

    def test_adr_from_dict_partial(self):
        adr = ADR.from_dict({"title": "Partial", "decision": "go ahead"})
        assert adr.title == "Partial"
        assert adr.decision == "go ahead"
        assert adr.context == ""

    def test_adr_to_dict_from_dict_roundtrip(self):
        original = ADR(
            adr_id="ADR-0005",
            title="Roundtrip",
            category="performance",
            context="perf",
            decision="optimize",
        )
        restored = ADR.from_dict(original.to_dict())
        assert restored.adr_id == original.adr_id
        assert restored.title == original.title
        assert restored.category == original.category

    def test_adr_default_status(self):
        adr = ADR(adr_id="x", title="t", category="c", context="cx", decision="d")
        assert adr.status == ADRStatus.PROPOSED.value

    def test_adr_default_related_adrs_is_independent_list(self):
        a1 = ADR(adr_id="1", title="", category="", context="", decision="")
        a2 = ADR(adr_id="2", title="", category="", context="", decision="")
        a1.related_adrs.append("ADR-X")
        assert "ADR-X" not in a2.related_adrs

    def test_adr_proposal_creation(self):
        opts = [{"id": "opt_1", "description": "d"}]
        proposal = ADRProposal(question="Which?", options=opts, recommended=0, rationale="because")
        assert proposal.question == "Which?"
        assert len(proposal.options) == 1
        assert proposal.recommended == 0
        assert proposal.rationale == "because"

    def test_adr_proposal_defaults(self):
        proposal = ADRProposal(question="Q", options=[])
        assert proposal.recommended == 0
        assert proposal.rationale == ""

    def test_adr_to_dict_contains_all_fields(self):
        adr = ADR(adr_id="id", title="t", category="c", context="cx", decision="d")
        d = adr.to_dict()
        expected_keys = {
            "adr_id", "title", "category", "context", "decision",
            "status", "consequences", "related_adrs", "created_at",
            "updated_at", "created_by", "notes",
        }
        assert set(d.keys()) == expected_keys


class TestADRSystem:
    """Tests for the ADRSystem singleton and CRUD operations."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ADRSystem._instance = None
        yield
        ADRSystem._instance = None

    @pytest.fixture
    def adr_system(self, tmp_path):
        return ADRSystem(storage_path=str(tmp_path / "adr"))

    def test_get_instance_creates_singleton(self, tmp_path):
        inst = ADRSystem.get_instance(str(tmp_path / "singleton_adr"))
        assert inst is ADRSystem.get_instance()

    def test_get_instance_reuses_singleton(self, tmp_path):
        s1 = ADRSystem.get_instance(str(tmp_path / "adr"))
        s2 = ADRSystem.get_instance()
        assert s1 is s2

    def test_init_creates_storage_directory(self, tmp_path):
        path = tmp_path / "new_adr_dir"
        assert not path.exists()
        ADRSystem(storage_path=str(path))
        assert path.exists()

    def test_create_adr_basic(self, adr_system: ADRSystem):
        adr = adr_system.create_adr(
            title="Test",
            category="architecture",
            context="We need a DB",
            decision="Use SQLite",
        )
        assert adr.adr_id == "ADR-0001"
        assert adr.title == "Test"
        assert adr.status == ADRStatus.PROPOSED.value
        assert adr.created_by == "user"

    def test_create_adr_increments_id(self, adr_system: ADRSystem):
        adr_system.create_adr("A", "c", "cx", "d")
        adr2 = adr_system.create_adr("B", "c", "cx", "d")
        assert adr2.adr_id == "ADR-0002"

    def test_create_adr_custom_created_by(self, adr_system: ADRSystem):
        adr = adr_system.create_adr("T", "c", "cx", "d", created_by="admin")
        assert adr.created_by == "admin"

    def test_create_adr_persists_to_disk(self, adr_system: ADRSystem):
        adr = adr_system.create_adr("T", "c", "cx", "d")
        file = Path(adr_system.storage_path) / f"{adr.adr_id}.json"
        assert file.exists()
        data = json.loads(file.read_text())
        assert data["title"] == "T"

    def test_get_adr_found(self, adr_system: ADRSystem):
        adr_system.create_adr("T", "c", "cx", "d")
        result = adr_system.get_adr("ADR-0001")
        assert result is not None
        assert result.title == "T"

    def test_get_adr_not_found(self, adr_system: ADRSystem):
        assert adr_system.get_adr("ADR-9999") is None

    def test_list_adrs_all(self, adr_system: ADRSystem):
        adr_system.create_adr("A", "architecture", "cx", "d")
        adr_system.create_adr("B", "security", "cx", "d")
        result = adr_system.list_adrs()
        assert len(result) == 2

    def test_list_adrs_filter_by_status(self, adr_system: ADRSystem):
        adr_system.create_adr("A", "c", "cx", "d")
        adr_system.update_adr("ADR-0001", {"status": "accepted"})
        adr_system.create_adr("B", "c", "cx", "d")
        result = adr_system.list_adrs(status="accepted")
        assert len(result) == 1
        assert result[0].adr_id == "ADR-0001"

    def test_list_adrs_filter_by_category(self, adr_system: ADRSystem):
        adr_system.create_adr("A", "security", "cx", "d")
        adr_system.create_adr("B", "performance", "cx", "d")
        result = adr_system.list_adrs(category="security")
        assert len(result) == 1

    def test_list_adrs_limit(self, adr_system: ADRSystem):
        for i in range(5):
            adr_system.create_adr(f"ADR {i}", "c", "cx", "d")
        result = adr_system.list_adrs(limit=3)
        assert len(result) == 3

    def test_list_adrs_empty(self, adr_system: ADRSystem):
        assert adr_system.list_adrs() == []

    def test_update_adr_changes_field(self, adr_system: ADRSystem):
        adr_system.create_adr("T", "c", "cx", "d")
        updated = adr_system.update_adr("ADR-0001", {"status": "accepted", "notes": "approved"})
        assert updated is not None
        assert updated.status == "accepted"
        assert updated.notes == "approved"

    def test_update_adr_sets_updated_at(self, adr_system: ADRSystem):
        adr = adr_system.create_adr("T", "c", "cx", "d")
        old_ts = adr.updated_at
        updated = adr_system.update_adr("ADR-0001", {"notes": "changed"})
        assert updated.updated_at >= old_ts

    def test_update_adr_nonexistent_returns_none(self, adr_system: ADRSystem):
        assert adr_system.update_adr("ADR-NOPE", {"status": "x"}) is None

    def test_update_adr_ignores_nonexistent_field(self, adr_system: ADRSystem):
        adr_system.create_adr("T", "c", "cx", "d")
        updated = adr_system.update_adr("ADR-0001", {"bogus_field": "value"})
        assert updated is not None
        assert not hasattr(updated, "bogus_field") or getattr(updated, "bogus_field", None) is None

    def test_deprecate_adr(self, adr_system: ADRSystem):
        adr_system.create_adr("Old", "c", "cx", "d")
        adr_system.create_adr("New", "c", "cx", "d")
        result = adr_system.deprecate_adr("ADR-0001", replacement_id="ADR-0002")
        assert result.status == ADRStatus.DEPRECATED.value
        assert "ADR-0002" in result.related_adrs

    def test_deprecate_adr_links_replacement(self, adr_system: ADRSystem):
        adr_system.create_adr("Old", "c", "cx", "d")
        adr_system.create_adr("New", "c", "cx", "d")
        adr_system.deprecate_adr("ADR-0001", replacement_id="ADR-0002")
        replacement = adr_system.get_adr("ADR-0002")
        assert "ADR-0001" in replacement.related_adrs

    def test_deprecate_adr_no_replacement(self, adr_system: ADRSystem):
        adr_system.create_adr("Old", "c", "cx", "d")
        result = adr_system.deprecate_adr("ADR-0001")
        assert result.status == ADRStatus.DEPRECATED.value

    def test_deprecate_adr_nonexistent(self, adr_system: ADRSystem):
        assert adr_system.deprecate_adr("ADR-NOPE") is None

    def test_is_high_stakes_true(self, adr_system: ADRSystem):
        assert adr_system.is_high_stakes("architecture") is True
        assert adr_system.is_high_stakes("security") is True
        assert adr_system.is_high_stakes("data_flow") is True

    def test_is_high_stakes_false(self, adr_system: ADRSystem):
        assert adr_system.is_high_stakes("performance") is False
        assert adr_system.is_high_stakes("api_design") is False
        assert adr_system.is_high_stakes("integration") is False

    def test_is_high_stakes_invalid_category(self, adr_system: ADRSystem):
        assert adr_system.is_high_stakes("totally_invalid") is False

    def test_generate_proposal_default_options(self, adr_system: ADRSystem):
        proposal = adr_system.generate_proposal("What arch?")
        assert proposal.question == "What arch?"
        assert len(proposal.options) == 3
        assert proposal.recommended == 0

    def test_generate_proposal_limited_options(self, adr_system: ADRSystem):
        proposal = adr_system.generate_proposal("ctx", num_options=1)
        assert len(proposal.options) == 1

    def test_generate_proposal_two_options(self, adr_system: ADRSystem):
        proposal = adr_system.generate_proposal("ctx", num_options=2)
        assert len(proposal.options) == 2

    def test_generate_proposal_option_structure(self, adr_system: ADRSystem):
        proposal = adr_system.generate_proposal("ctx")
        opt = proposal.options[0]
        assert "id" in opt
        assert "description" in opt
        assert "pros" in opt
        assert "cons" in opt

    def test_accept_proposal(self, adr_system: ADRSystem):
        proposal = adr_system.generate_proposal("ctx")
        adr = adr_system.accept_proposal(proposal, title="Accepted", category="architecture")
        assert adr.adr_id == "ADR-0001"
        assert adr.title == "Accepted"
        assert adr.created_by == "system"
        assert "option_1" in adr.decision

    def test_get_statistics_empty(self, adr_system: ADRSystem):
        stats = adr_system.get_statistics()
        assert stats["total"] == 0
        assert stats["by_status"] == {}
        assert stats["by_category"] == {}
        assert stats["high_stakes_count"] == 0

    def test_get_statistics_with_data(self, adr_system: ADRSystem):
        adr_system.create_adr("A", "architecture", "cx", "d")
        adr_system.create_adr("B", "security", "cx", "d")
        adr_system.create_adr("C", "performance", "cx", "d")
        stats = adr_system.get_statistics()
        assert stats["total"] == 3
        assert stats["by_category"]["architecture"] == 1
        assert stats["by_category"]["security"] == 1
        assert stats["high_stakes_count"] == 2  # architecture + security

    def test_load_adrs_from_disk(self, tmp_path):
        storage = tmp_path / "adr_load"
        storage.mkdir()
        data = {
            "adr_id": "ADR-0001",
            "title": "Persisted",
            "category": "security",
            "context": "ctx",
            "decision": "dec",
            "status": "accepted",
        }
        (storage / "ADR-0001.json").write_text(json.dumps(data))
        system = ADRSystem(storage_path=str(storage))
        loaded = system.get_adr("ADR-0001")
        assert loaded is not None
        assert loaded.title == "Persisted"

    def test_load_adrs_skips_bad_json(self, tmp_path):
        storage = tmp_path / "adr_bad"
        storage.mkdir()
        (storage / "bad.json").write_text("NOT JSON")
        system = ADRSystem(storage_path=str(storage))
        assert len(system.adrs) == 0


# ---------------------------------------------------------------------------
# 2. vetinari.builder
# ---------------------------------------------------------------------------

from vetinari.builder import Builder


class TestBuilder:
    """Tests for Builder.build_final_artifact."""

    @pytest.fixture
    def builder_with_dirs(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        config = {
            "outputs_dir": str(outputs),
            "build": {"artifacts": str(tmp_path / "build" / "artifacts")},
        }
        return Builder(config), outputs, tmp_path

    def test_builder_init(self):
        b = Builder({"key": "val"})
        assert b.config["key"] == "val"

    def test_build_creates_zip(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        (outputs / "file.txt").write_text("hello")
        zip_path = builder.build_final_artifact()
        assert Path(zip_path).exists()
        assert zip_path.endswith(".zip")

    def test_build_zip_contains_files(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        (outputs / "a.txt").write_text("aaa")
        (outputs / "b.txt").write_text("bbb")
        zip_path = builder.build_final_artifact()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "a.txt" in names
            assert "b.txt" in names

    def test_build_zip_preserves_subdirectory_structure(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        sub = outputs / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")
        zip_path = builder.build_final_artifact()
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            # Should have subdir/nested.txt path
            assert any("nested.txt" in n for n in names)

    def test_build_empty_outputs_creates_zip(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        zip_path = builder.build_final_artifact()
        assert Path(zip_path).exists()
        with zipfile.ZipFile(zip_path) as zf:
            assert zf.namelist() == []

    def test_build_creates_artifacts_dir(self, tmp_path):
        outputs = tmp_path / "out"
        outputs.mkdir()
        artifacts = tmp_path / "new_build" / "arts"
        config = {
            "outputs_dir": str(outputs),
            "build": {"artifacts": str(artifacts)},
        }
        builder = Builder(config)
        builder.build_final_artifact()
        assert artifacts.exists()

    def test_build_with_results_creates_report(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        (outputs / "f.txt").write_text("data")
        results = [
            {"task_id": "t1", "status": "completed", "model_id": "m1", "latency_ms": 100},
            {"task_id": "t2", "status": "failed", "model_id": "m2", "latency_ms": 200, "error": "boom"},
        ]
        builder.build_final_artifact(results)
        report_path = tmp_path / "build" / "artifacts" / "build_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["total_tasks"] == 2
        assert report["completed"] == 1
        assert report["failed"] == 1

    def test_build_without_results_no_report(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        (outputs / "f.txt").write_text("data")
        builder.build_final_artifact()
        report_path = tmp_path / "build" / "artifacts" / "build_report.json"
        assert not report_path.exists()

    def test_build_with_none_results_no_report(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        (outputs / "f.txt").write_text("data")
        builder.build_final_artifact(results=None)
        report_path = tmp_path / "build" / "artifacts" / "build_report.json"
        assert not report_path.exists()

    def test_build_with_empty_results_no_report(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        (outputs / "f.txt").write_text("data")
        builder.build_final_artifact(results=[])
        report_path = tmp_path / "build" / "artifacts" / "build_report.json"
        assert not report_path.exists()

    def test_build_report_has_timestamp(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        (outputs / "f.txt").write_text("data")
        results = [{"task_id": "t1", "status": "completed"}]
        builder.build_final_artifact(results)
        report_path = tmp_path / "build" / "artifacts" / "build_report.json"
        report = json.loads(report_path.read_text())
        assert "timestamp" in report

    def test_build_report_task_details(self, builder_with_dirs):
        builder, outputs, tmp_path = builder_with_dirs
        (outputs / "f.txt").write_text("data")
        results = [
            {"task_id": "t1", "status": "completed", "model_id": "gpt", "latency_ms": 50, "error": None},
        ]
        builder.build_final_artifact(results)
        report_path = tmp_path / "build" / "artifacts" / "build_report.json"
        report = json.loads(report_path.read_text())
        task = report["tasks"][0]
        assert task["task_id"] == "t1"
        assert task["model"] == "gpt"
        assert task["latency_ms"] == 50
        assert task["error"] is None

    def test_build_default_config_paths(self, tmp_path):
        # If no outputs_dir in config, defaults to "outputs"
        config = {}
        builder = Builder(config)
        # Just check it doesn't crash with missing dir (will raise on walk)
        # We want to verify config defaults are accessed
        assert builder.config.get("outputs_dir", "outputs") == "outputs"

    def test_build_returns_string_path(self, builder_with_dirs):
        builder, outputs, _ = builder_with_dirs
        (outputs / "x.txt").write_text("x")
        result = builder.build_final_artifact()
        assert isinstance(result, str)

    def test_build_zip_is_valid_zipfile(self, builder_with_dirs):
        builder, outputs, _ = builder_with_dirs
        (outputs / "x.txt").write_text("x")
        zip_path = builder.build_final_artifact()
        assert zipfile.is_zipfile(zip_path)


# ---------------------------------------------------------------------------
# 3. vetinari.credentials
# ---------------------------------------------------------------------------

# Import at module level but avoid triggering the module-level singleton
# by patching CRYPTO_AVAILABLE


class TestCredential:
    """Tests for the Credential dataclass."""

    def test_credential_to_dict_excludes_token(self):
        from vetinari.credentials import Credential

        cred = Credential(
            source_type="github",
            credential_type="bearer",
            token="secret123",
            scopes=["read"],
        )
        d = cred.to_dict()
        assert "token" not in d
        assert d["source_type"] == "github"
        assert d["credential_type"] == "bearer"
        assert d["scopes"] == ["read"]

    def test_credential_to_dict_includes_all_other_fields(self):
        from vetinari.credentials import Credential

        cred = Credential(
            source_type="s",
            credential_type="c",
            token="t",
            scopes=["a"],
            rotation_days=90,
            last_rotated="2025-01-01",
            next_rotation_due="2025-04-01",
            access_controls=["admin"],
            token_source="env",
            note="note",
            enabled=False,
        )
        d = cred.to_dict()
        assert d["rotation_days"] == 90
        assert d["enabled"] is False
        assert d["token_source"] == "env"
        assert d["access_controls"] == ["admin"]

    def test_credential_needs_rotation_empty_due(self):
        from vetinari.credentials import Credential

        cred = Credential(source_type="s", credential_type="c", token="t")
        assert cred.needs_rotation() is True

    def test_credential_needs_rotation_past_due(self):
        from vetinari.credentials import Credential

        past = (datetime.now() - timedelta(days=1)).isoformat()
        cred = Credential(
            source_type="s",
            credential_type="c",
            token="t",
            next_rotation_due=past,
        )
        assert cred.needs_rotation() is True

    def test_credential_needs_rotation_future_due(self):
        from vetinari.credentials import Credential

        future = (datetime.now() + timedelta(days=30)).isoformat()
        cred = Credential(
            source_type="s",
            credential_type="c",
            token="t",
            next_rotation_due=future,
        )
        assert cred.needs_rotation() is False

    def test_credential_needs_rotation_invalid_date(self):
        from vetinari.credentials import Credential

        cred = Credential(
            source_type="s",
            credential_type="c",
            token="t",
            next_rotation_due="not-a-date",
        )
        assert cred.needs_rotation() is True

    def test_credential_defaults(self):
        from vetinari.credentials import Credential

        cred = Credential(source_type="s", credential_type="c", token="t")
        assert cred.scopes == []
        assert cred.rotation_days == 30
        assert cred.last_rotated == ""
        assert cred.access_controls == []
        assert cred.token_source == "manual"
        assert cred.note == ""
        assert cred.enabled is True

    def test_credential_independent_default_lists(self):
        from vetinari.credentials import Credential

        c1 = Credential(source_type="a", credential_type="c", token="t")
        c2 = Credential(source_type="b", credential_type="c", token="t")
        c1.scopes.append("read")
        assert "read" not in c2.scopes

    def test_credential_to_dict_preserves_note_and_enabled(self):
        from vetinari.credentials import Credential

        cred = Credential(
            source_type="s", credential_type="c", token="t",
            note="important", enabled=True,
        )
        d = cred.to_dict()
        assert d["note"] == "important"
        assert d["enabled"] is True


@pytest.mark.skipif(
    not importlib.util.find_spec("cryptography"),
    reason="cryptography package not installed",
)
class TestCredentialVault:
    """Tests for CredentialVault (requires cryptography package)."""

    @pytest.fixture
    def vault(self, tmp_path):
        from vetinari.credentials import CredentialVault
        return CredentialVault(vault_path=str(tmp_path / "vault"))

    @pytest.fixture
    def make_cred(self):
        from vetinari.credentials import Credential

        def _make(source="github", token="tok123", enabled=True, **kw):
            return Credential(
                source_type=source,
                credential_type="bearer",
                token=token,
                enabled=enabled,
                **kw,
            )
        return _make

    def test_vault_creates_directory(self, tmp_path):
        path = tmp_path / "new_vault"
        from vetinari.credentials import CredentialVault
        CredentialVault(vault_path=str(path))
        assert path.exists()

    def test_set_and_get_credential(self, vault, make_cred):
        cred = make_cred()
        vault.set_credential("github", cred)
        result = vault.get_credential("github")
        assert result is not None
        assert result.token == "tok123"

    def test_get_credential_not_found(self, vault):
        assert vault.get_credential("nonexistent") is None

    def test_get_credential_disabled_returns_none(self, vault, make_cred):
        cred = make_cred(enabled=False)
        vault.set_credential("github", cred)
        assert vault.get_credential("github") is None

    def test_get_token(self, vault, make_cred):
        vault.set_credential("gh", make_cred(source="gh", token="abc"))
        assert vault.get_token("gh") == "abc"

    def test_get_token_missing(self, vault):
        assert vault.get_token("nope") is None

    def test_set_credential_sets_rotation_dates(self, vault, make_cred):
        cred = make_cred(rotation_days=7)
        vault.set_credential("gh", cred)
        stored = vault.get_credential("gh")
        assert stored.last_rotated != ""
        assert stored.next_rotation_due != ""

    def test_remove_credential(self, vault, make_cred):
        vault.set_credential("gh", make_cred())
        vault.remove_credential("gh")
        assert vault.get_credential("gh") is None

    def test_remove_credential_nonexistent(self, vault):
        # Should not raise
        vault.remove_credential("nope")

    def test_rotate_credential_success(self, vault, make_cred):
        vault.set_credential("gh", make_cred(token="old"))
        result = vault.rotate_credential("gh", "new_token")
        assert result is True
        assert vault.get_token("gh") == "new_token"

    def test_rotate_credential_nonexistent(self, vault):
        assert vault.rotate_credential("nope", "t") is False

    def test_rotate_credential_updates_rotation_dates(self, vault, make_cred):
        vault.set_credential("gh", make_cred())
        vault.rotate_credential("gh", "new")
        cred = vault.get_credential("gh")
        assert cred.last_rotated != ""
        assert cred.next_rotation_due != ""

    def test_list_credentials(self, vault, make_cred):
        vault.set_credential("gh", make_cred(source="gh"))
        vault.set_credential("gl", make_cred(source="gl", token="tok2"))
        listing = vault.list_credentials()
        assert "gh" in listing
        assert "gl" in listing
        # Token should be excluded from listing (to_dict excludes it)
        assert "token" not in listing["gh"]

    def test_list_credentials_empty(self, vault):
        assert vault.list_credentials() == {}

    def test_get_health(self, vault, make_cred):
        vault.set_credential("gh", make_cred())
        health = vault.get_health()
        assert "gh" in health
        assert "enabled" in health["gh"]
        assert "needs_rotation" in health["gh"]
        assert "credential_type" in health["gh"]

    def test_get_health_empty(self, vault):
        assert vault.get_health() == {}

    def test_is_admin_no_admins_file(self, vault):
        # When no admins file exists, everyone is admin
        assert vault.is_admin("anyone") is True

    def test_add_admin_and_check(self, vault):
        vault.add_admin("user1")
        assert vault.is_admin("user1") is True

    def test_is_admin_false_with_file(self, vault):
        vault.add_admin("user1")
        assert vault.is_admin("user2") is False

    def test_add_admin_idempotent(self, vault):
        vault.add_admin("user1")
        vault.add_admin("user1")
        admins_file = vault.vault_path / "admins.json"
        data = json.loads(admins_file.read_text())
        assert data["admins"].count("user1") == 1

    def test_save_without_encryption_raises(self, tmp_path, make_cred):
        # P1.H9: Vault must refuse to save credentials when encryption is unavailable.
        vault_path = str(tmp_path / "vault_no_crypto")
        with patch("vetinari.credentials.CRYPTO_AVAILABLE", False):
            from vetinari.credentials import CredentialVault
            v = CredentialVault(vault_path=vault_path)
        # _fernet is None because crypto was unavailable at init time
        import pytest as _pytest
        with _pytest.raises(RuntimeError, match="encryption is unavailable"):
            v.set_credential("gh", make_cred())

    def test_load_empty_credentials_file(self, tmp_path):
        vault_path = tmp_path / "vault_empty"
        vault_path.mkdir()
        (vault_path / "credentials.enc").write_bytes(b"")
        with patch("vetinari.credentials.CRYPTO_AVAILABLE", False):
            from vetinari.credentials import CredentialVault
            v = CredentialVault(vault_path=str(vault_path))
        assert v.list_credentials() == {}

    def test_load_corrupt_credentials_file(self, tmp_path):
        vault_path = tmp_path / "vault_corrupt"
        vault_path.mkdir()
        (vault_path / "credentials.enc").write_bytes(b"NOT VALID JSON OR CRYPTO")
        with patch("vetinari.credentials.CRYPTO_AVAILABLE", False):
            from vetinari.credentials import CredentialVault
            v = CredentialVault(vault_path=str(vault_path))
        assert v.list_credentials() == {}

    def test_save_meta_file_created(self, vault, make_cred):
        vault.set_credential("gh", make_cred())
        meta_path = vault.vault_path / "credentials_meta.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert "gh" in data
        assert "token" not in data["gh"]


# ---------------------------------------------------------------------------
# 4. vetinari.exceptions
# ---------------------------------------------------------------------------

from vetinari.exceptions import (
    VetinariError,
    ConfigurationError,
    StorageError,
    InferenceError,
    AdapterError,
    ModelNotFoundError,
    TimeoutError as VetinariTimeoutError,
    AgentError,
    PlanningError,
    ExecutionError,
    VerificationError,
    CircularDependencyError,
    SecurityError as VetinariSecurityError,
    SandboxError,
    GuardrailError,
    DriftError,
    SLABreachError,
)


class TestExceptions:
    """Tests for the Vetinari exception hierarchy."""

    def test_vetinari_error_is_exception(self):
        assert issubclass(VetinariError, Exception)

    def test_vetinari_error_message(self):
        err = VetinariError("something broke")
        assert str(err) == "something broke"

    def test_vetinari_error_with_context(self):
        err = VetinariError("broke", model_id="qwen", step=3)
        s = str(err)
        assert "model_id='qwen'" in s
        assert "step=3" in s
        assert "broke" in s

    def test_vetinari_error_empty_context(self):
        err = VetinariError("simple")
        assert str(err) == "simple"
        assert err.context == {}

    def test_vetinari_error_context_dict(self):
        err = VetinariError("x", a=1, b="two")
        assert err.context == {"a": 1, "b": "two"}

    def test_vetinari_error_no_message(self):
        err = VetinariError()
        assert str(err) == ""

    def test_vetinari_error_no_message_with_context(self):
        err = VetinariError(key="val")
        s = str(err)
        assert "key='val'" in s

    # Hierarchy: Infrastructure errors
    def test_configuration_error_hierarchy(self):
        assert issubclass(ConfigurationError, VetinariError)
        err = ConfigurationError("bad config", param="x")
        assert isinstance(err, VetinariError)

    def test_storage_error_hierarchy(self):
        assert issubclass(StorageError, VetinariError)

    # Hierarchy: Inference errors
    def test_inference_error_hierarchy(self):
        assert issubclass(InferenceError, VetinariError)

    def test_adapter_error_hierarchy(self):
        assert issubclass(AdapterError, VetinariError)

    def test_model_not_found_error_hierarchy(self):
        assert issubclass(ModelNotFoundError, InferenceError)
        assert issubclass(ModelNotFoundError, VetinariError)

    def test_timeout_error_hierarchy(self):
        assert issubclass(VetinariTimeoutError, InferenceError)
        assert issubclass(VetinariTimeoutError, VetinariError)

    # Hierarchy: Agent errors
    def test_agent_error_hierarchy(self):
        assert issubclass(AgentError, VetinariError)

    def test_planning_error_hierarchy(self):
        assert issubclass(PlanningError, AgentError)
        assert issubclass(PlanningError, VetinariError)

    def test_execution_error_hierarchy(self):
        assert issubclass(ExecutionError, AgentError)

    def test_verification_error_hierarchy(self):
        assert issubclass(VerificationError, AgentError)

    def test_circular_dependency_error_hierarchy(self):
        assert issubclass(CircularDependencyError, PlanningError)
        assert issubclass(CircularDependencyError, AgentError)
        assert issubclass(CircularDependencyError, VetinariError)

    # Hierarchy: Security errors
    def test_security_error_hierarchy(self):
        assert issubclass(VetinariSecurityError, VetinariError)

    def test_sandbox_error_hierarchy(self):
        assert issubclass(SandboxError, VetinariSecurityError)
        assert issubclass(SandboxError, VetinariError)

    def test_guardrail_error_hierarchy(self):
        assert issubclass(GuardrailError, VetinariSecurityError)

    # Hierarchy: Analytics errors
    def test_drift_error_hierarchy(self):
        assert issubclass(DriftError, VetinariError)

    def test_sla_breach_error_hierarchy(self):
        assert issubclass(SLABreachError, VetinariError)

    # Raise and catch
    def test_raise_and_catch_specific(self):
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("model gone", model_id="llama")

    def test_raise_specific_catch_parent(self):
        with pytest.raises(InferenceError):
            raise ModelNotFoundError("not found")

    def test_raise_specific_catch_base(self):
        with pytest.raises(VetinariError):
            raise CircularDependencyError("loop detected", graph="G")

    def test_context_survives_raise(self):
        try:
            raise ExecutionError("fail", task_id="t1", agent="builder")
        except ExecutionError as e:
            assert e.context["task_id"] == "t1"
            assert e.context["agent"] == "builder"

    def test_str_formatting_brackets(self):
        err = ConfigurationError("missing key", key="api_key")
        s = str(err)
        assert s.startswith("missing key")
        assert "[" in s and "]" in s

    def test_all_exception_classes_are_instantiable(self):
        classes = [
            VetinariError, ConfigurationError, StorageError, InferenceError,
            AdapterError, ModelNotFoundError, VetinariTimeoutError, AgentError,
            PlanningError, ExecutionError, VerificationError,
            CircularDependencyError, VetinariSecurityError, SandboxError,
            GuardrailError, DriftError, SLABreachError,
        ]
        for cls in classes:
            inst = cls("test")
            assert isinstance(inst, Exception)
            assert isinstance(inst, VetinariError)

    def test_isinstance_chain_deep(self):
        err = CircularDependencyError("cycle")
        assert isinstance(err, CircularDependencyError)
        assert isinstance(err, PlanningError)
        assert isinstance(err, AgentError)
        assert isinstance(err, VetinariError)
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# 5. vetinari.validator
# ---------------------------------------------------------------------------

from vetinari.validator import Validator


class TestValidator:
    """Tests for Validator.is_valid_text, _looks_like_code, _validate_python_code."""

    @pytest.fixture
    def validator(self):
        return Validator()

    # --- is_valid_text ---

    def test_valid_plain_text(self, validator):
        assert validator.is_valid_text("Hello, this is valid text.") is True

    def test_empty_string_invalid(self, validator):
        assert validator.is_valid_text("") is False

    def test_none_invalid(self, validator):
        assert validator.is_valid_text(None) is False

    def test_whitespace_only_invalid(self, validator):
        assert validator.is_valid_text("   \n\t  ") is False

    def test_valid_json_object(self, validator):
        assert validator.is_valid_text('{"key": "value"}') is True

    def test_valid_json_array(self, validator):
        assert validator.is_valid_text('[1, 2, 3]') is True

    def test_valid_json_number(self, validator):
        assert validator.is_valid_text("42") is True

    def test_valid_json_string(self, validator):
        assert validator.is_valid_text('"hello"') is True

    def test_valid_python_code(self, validator):
        code = "def hello():\n    print('hi')"
        assert validator.is_valid_text(code) is True

    def test_invalid_python_code(self, validator):
        code = "def hello(\n    print('hi')"
        assert validator.is_valid_text(code) is False

    def test_valid_class_code(self, validator):
        code = "class Foo:\n    pass"
        assert validator.is_valid_text(code) is True

    def test_valid_import_code(self, validator):
        code = "import os"
        assert validator.is_valid_text(code) is True

    def test_valid_from_import_code(self, validator):
        code = "from os import path"
        assert validator.is_valid_text(code) is True

    def test_markdown_wrapped_code(self, validator):
        code = "```python\ndef hello():\n    pass\n```"
        assert validator.is_valid_text(code) is True

    def test_markdown_wrapped_invalid_code(self, validator):
        code = "```python\ndef hello(\n```"
        assert validator.is_valid_text(code) is False

    # --- _looks_like_code ---

    def test_looks_like_code_def(self, validator):
        assert validator._looks_like_code("def foo(): pass") is True

    def test_looks_like_code_class(self, validator):
        assert validator._looks_like_code("class Bar: pass") is True

    def test_looks_like_code_import(self, validator):
        assert validator._looks_like_code("import json") is True

    def test_looks_like_code_from_import(self, validator):
        assert validator._looks_like_code("from os import path") is True

    def test_looks_like_code_decorator(self, validator):
        assert validator._looks_like_code("@property\ndef x(self): pass") is True

    def test_looks_like_code_comment(self, validator):
        assert validator._looks_like_code("# This is a comment") is True

    def test_looks_like_code_async(self, validator):
        assert validator._looks_like_code("async def run(): pass") is True

    def test_looks_like_code_with_statement(self, validator):
        assert validator._looks_like_code("with open('f') as fh:\n    pass") is True

    def test_looks_like_code_assignment_unindented_not_detected(self, validator):
        # The regex requires leading whitespace before '=', so bare "x = 42" is not detected
        assert validator._looks_like_code("x = 42") is False

    def test_looks_like_code_assignment_indented(self, validator):
        # The regex \s+=\s+ requires whitespace directly before '='.
        # "    = value" would match, but "    x = 42" would not because
        # \s+ consumes spaces then expects '=' but finds 'x'.
        # A multiline context with an indented assignment on its own line
        # still won't match unless preceded by code keywords.
        assert validator._looks_like_code("def foo():\n    x = 42") is True

    def test_looks_like_code_plain_text(self, validator):
        assert validator._looks_like_code("Hello this is just a sentence.") is False

    def test_looks_like_code_markdown_stripped(self, validator):
        text = "```python\nimport os\n```"
        assert validator._looks_like_code(text) is True

    # --- _validate_python_code ---

    def test_validate_python_valid(self, validator):
        assert validator._validate_python_code("x = 1") is True

    def test_validate_python_invalid_syntax(self, validator):
        assert validator._validate_python_code("def (") is False

    def test_validate_python_empty_after_strip(self, validator):
        # Empty string is valid Python (ast.parse("") succeeds)
        assert validator._validate_python_code("") is True

    def test_validate_python_strips_markdown_fences(self, validator):
        code = "```python\nx = 1\n```"
        assert validator._validate_python_code(code) is True

    def test_validate_python_multiline(self, validator):
        code = "def foo():\n    return 42\n\nclass Bar:\n    pass"
        assert validator._validate_python_code(code) is True

    def test_validate_python_json_wrapper_stripped(self, validator):
        # A JSON wrapper containing valid Python after stripping braces
        code = '{ x = 1 }'
        # Inner "x = 1" is valid Python
        result = validator._validate_python_code(code)
        assert result is True

    def test_validate_python_actual_dict_not_stripped(self, validator):
        # This is a Python dict literal that also starts/ends with braces
        # but contains def, so inner won't be stripped
        code = '{\ndef foo(): pass\n}'
        # Inner has def so it's not stripped; parsed as-is, which is a syntax error
        # because { def ... } is invalid Python
        result = validator._validate_python_code(code)
        assert result is False


# ---------------------------------------------------------------------------
# 6. vetinari.upgrader
# ---------------------------------------------------------------------------


class TestUpgrader:
    """Tests for Upgrader.check_for_upgrades and install_upgrade."""

    SAMPLE_MODELS = [
        {"name": "model-a", "version": "1.0", "memory_gb": 8},
        {"name": "model-b", "version": "2.0", "memory_gb": 12},
    ]

    def _mock_fetch(self, data, status=200):
        """Return a patch context that mocks requests.get to return *data*."""
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = data
        return patch("vetinari.upgrader.requests.get", return_value=mock_resp)

    def _mock_install(self, status=200):
        """Return a patch context that mocks requests.post for install_upgrade."""
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.text = "OK"
        return patch("vetinari.upgrader.requests.post", return_value=mock_resp)

    def test_upgrader_init(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({"key": "val"})
        assert u.config["key"] == "val"

    def test_check_for_upgrades_returns_list(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with self._mock_fetch(self.SAMPLE_MODELS):
            result = u.check_for_upgrades()
        assert isinstance(result, list)

    def test_check_for_upgrades_filters_by_memory(self):
        from vetinari.upgrader import Upgrader

        models = [
            {"name": "small", "version": "1.0", "memory_gb": 4},
            {"name": "huge", "version": "1.0", "memory_gb": 200},
        ]
        u = Upgrader({"benchmarks_source": ["http://example.com"], "memory_budget_gb": 96})
        with self._mock_fetch(models):
            candidates = u.check_for_upgrades()
        for c in candidates:
            assert c.get("memory_gb", 0) <= 96
        assert len(candidates) == 1
        assert candidates[0]["name"] == "small"

    def test_check_for_upgrades_with_endpoint_returns_candidates(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://example.com/benchmarks"]})
        with self._mock_fetch(self.SAMPLE_MODELS):
            candidates = u.check_for_upgrades()
        assert len(candidates) == 2

    def test_check_for_upgrades_candidate_has_name(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with self._mock_fetch(self.SAMPLE_MODELS):
            candidates = u.check_for_upgrades()
        for c in candidates:
            assert "name" in c

    def test_check_for_upgrades_candidate_has_version(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with self._mock_fetch(self.SAMPLE_MODELS):
            candidates = u.check_for_upgrades()
        for c in candidates:
            assert "version" in c

    def test_check_for_upgrades_candidate_has_memory_gb(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with self._mock_fetch(self.SAMPLE_MODELS):
            candidates = u.check_for_upgrades()
        for c in candidates:
            assert "memory_gb" in c

    def test_check_for_upgrades_string_source(self):
        """benchmarks_source as a plain string (not list) is accepted."""
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": "http://example.com/bench"})
        with self._mock_fetch(self.SAMPLE_MODELS):
            result = u.check_for_upgrades()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_check_for_upgrades_network_error(self):
        from vetinari.upgrader import Upgrader
        import requests as req

        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with patch("vetinari.upgrader.requests.get", side_effect=req.exceptions.ConnectionError("refused")):
            result = u.check_for_upgrades()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_check_for_upgrades_no_config_returns_empty(self):
        """No benchmarks_source configured returns empty list."""
        from vetinari.upgrader import Upgrader

        u = Upgrader({})
        candidates = u.check_for_upgrades()
        assert isinstance(candidates, list)
        assert len(candidates) == 0

    def test_install_upgrade_returns_true(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({})
        with self._mock_install(200):
            result = u.install_upgrade({"name": "test-model", "version": "1.0"})
        assert result is True

    def test_install_upgrade_empty_candidate(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({})
        with self._mock_install(200):
            result = u.install_upgrade({})
        assert result is True

    def test_install_upgrade_with_full_candidate(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({})
        candidate = {"name": "glm-flash", "version": "4.7", "memory_gb": 8}
        with self._mock_install(200):
            result = u.install_upgrade(candidate)
        assert result is True

    def test_upgrader_config_stored(self):
        from vetinari.upgrader import Upgrader

        cfg = {"a": 1, "b": [2, 3]}
        u = Upgrader(cfg)
        assert u.config is cfg

    def test_check_for_upgrades_benchmarks_source_used(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://custom.example.com"]})
        with self._mock_fetch([]) as mock_get:
            u.check_for_upgrades()
        mock_get.assert_called_once_with("http://custom.example.com", timeout=10)

    def test_check_for_upgrades_empty_config(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({})
        candidates = u.check_for_upgrades()
        assert isinstance(candidates, list)
        assert len(candidates) == 0

    def test_install_upgrade_logs_name(self, caplog):
        """install_upgrade logs the model name and version."""
        from vetinari.upgrader import Upgrader
        import logging

        u = Upgrader({})
        with caplog.at_level(logging.INFO, logger="vetinari.upgrader"):
            with self._mock_install(200):
                u.install_upgrade({"name": "my-model", "version": "2.0"})
        assert "my-model" in caplog.text
        assert "2.0" in caplog.text

    def test_check_for_upgrades_filters_high_memory(self):
        """Candidates exceeding memory budget are filtered out."""
        from vetinari.upgrader import Upgrader

        models = [
            {"name": "small", "version": "1.0", "memory_gb": 8},
            {"name": "huge", "version": "1.0", "memory_gb": 200},
        ]
        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with self._mock_fetch(models):
            candidates = u.check_for_upgrades()
        assert all(c["memory_gb"] <= 96 for c in candidates)

    def test_upgrader_multiple_calls_idempotent(self):
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with self._mock_fetch(self.SAMPLE_MODELS):
            r1 = u.check_for_upgrades()
        with self._mock_fetch(self.SAMPLE_MODELS):
            r2 = u.check_for_upgrades()
        assert len(r1) == len(r2)

    def test_check_for_upgrades_dict_response_models_key(self):
        """Handles response wrapped in {\"models\": [...]}."""
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with self._mock_fetch({"models": self.SAMPLE_MODELS}):
            candidates = u.check_for_upgrades()
        assert len(candidates) == 2

    def test_check_for_upgrades_dict_response_data_key(self):
        """Handles response wrapped in {\"data\": [...]}."""
        from vetinari.upgrader import Upgrader

        u = Upgrader({"benchmarks_source": ["http://example.com"]})
        with self._mock_fetch({"data": self.SAMPLE_MODELS}):
            candidates = u.check_for_upgrades()
        assert len(candidates) == 2

    def test_memory_budget_from_config(self):
        """Memory budget is read from config."""
        from vetinari.upgrader import Upgrader

        u = Upgrader({"memory_budget_gb": 4, "benchmarks_source": ["http://example.com"]})
        models = [
            {"name": "tiny", "version": "1.0", "memory_gb": 2},
            {"name": "medium", "version": "1.0", "memory_gb": 8},
        ]
        with self._mock_fetch(models):
            candidates = u.check_for_upgrades()
        assert len(candidates) == 1
        assert candidates[0]["name"] == "tiny"
