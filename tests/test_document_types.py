"""Tests for vetinari.validation.document_types."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from vetinari.validation import (
    DocumentProfile,
    DocumentType,
    get_profile_for_type,
    load_document_profiles,
)


class TestDocumentType:
    """Tests for the DocumentType enum."""

    def test_all_ten_types_present(self):
        values = {dt.value for dt in DocumentType}
        expected = {
            "adr",
            "api_reference",
            "changelog",
            "code_comment",
            "commit_message",
            "developer_guide",
            "error_message",
            "plan",
            "readme",
            "research_report",
        }
        assert values == expected

    def test_enum_access_by_name(self):
        assert DocumentType.ADR.value == "adr"
        assert DocumentType.README.value == "readme"
        assert DocumentType.PLAN.value == "plan"


class TestDocumentProfile:
    """Tests for the DocumentProfile dataclass."""

    def test_defaults(self):
        profile = DocumentProfile(doc_type="test")
        assert profile.doc_type == "test"
        assert profile.description == ""
        assert profile.min_score == 0.60
        assert profile.dimension_weights == {}
        assert profile.rules == []

    def test_custom_fields(self):
        profile = DocumentProfile(
            doc_type="adr",
            description="Architecture Decision Record",
            min_score=0.70,
            dimension_weights={"accuracy": 1.0, "clarity": 0.8},
            rules=["Must include Status section"],
        )
        assert profile.min_score == 0.70
        assert profile.dimension_weights["accuracy"] == 1.0
        assert len(profile.rules) == 1


class TestLoadDocumentProfiles:
    """Tests for load_document_profiles()."""

    def test_load_from_real_config(self):
        profiles = load_document_profiles()
        assert "adr" in profiles
        assert "readme" in profiles
        assert isinstance(profiles["adr"], DocumentProfile)

    def test_load_from_custom_path(self):
        config = {
            "default": {
                "min_score": 0.50,
                "dimension_weights": {"accuracy": 1.0},
            },
            "profiles": {
                "custom_type": {
                    "description": "Custom doc",
                    "min_score": 0.80,
                    "dimension_weights": {"clarity": 0.9},
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.dump(config, f)
            tmp_path = Path(f.name)

        try:
            profiles = load_document_profiles(config_path=tmp_path)
            assert "custom_type" in profiles
            p = profiles["custom_type"]
            assert p.min_score == 0.80
            # Inherits default weights and overrides
            assert p.dimension_weights["accuracy"] == 1.0
            assert p.dimension_weights["clarity"] == 0.9
        finally:
            tmp_path.unlink()

    def test_missing_config_returns_empty(self):
        profiles = load_document_profiles(config_path=Path("/nonexistent/path.yaml"))
        assert profiles == {}

    def test_profile_inherits_default_weights(self):
        profiles = load_document_profiles()
        if "adr" in profiles:
            # ADR should have weights inherited from default + overrides
            assert "accuracy" in profiles["adr"].dimension_weights

    def test_profile_rules_loaded(self):
        profiles = load_document_profiles()
        if "adr" in profiles:
            assert len(profiles["adr"].rules) >= 1


class TestGetProfileForType:
    """Tests for get_profile_for_type()."""

    def test_known_type_string(self):
        profile = get_profile_for_type("readme")
        assert profile.doc_type == "readme"
        assert profile.min_score > 0

    def test_known_type_enum(self):
        profile = get_profile_for_type(DocumentType.ADR)
        assert profile.doc_type == "adr"

    def test_unknown_type_returns_default(self):
        profile = get_profile_for_type("nonexistent_type")
        assert profile.doc_type == "nonexistent_type"
        assert profile.description == "Unknown document type"
        assert profile.min_score == 0.60
