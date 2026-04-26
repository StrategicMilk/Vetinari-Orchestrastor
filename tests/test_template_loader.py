"""Tests for vetinari.template_loader — versioned agent prompt templates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from vetinari.template_loader import (
    _AGENT_FILE_MAP,
    _CONSOLIDATED_AGENTS,
    TemplateLoader,
    template_loader,
)


class TestTemplateLoaderConstants:
    """Tests for module-level constants."""

    def test_consolidated_agents_has_six(self):
        assert len(_CONSOLIDATED_AGENTS) == 6

    def test_agent_file_map_covers_consolidated(self):
        for agent in _CONSOLIDATED_AGENTS:
            assert agent in _AGENT_FILE_MAP

    def test_legacy_names_map_to_consolidated(self):
        assert _AGENT_FILE_MAP["explorer"] == "researcher.json"
        assert _AGENT_FILE_MAP["evaluator"] == "quality.json"
        assert _AGENT_FILE_MAP["synthesizer"] == "operations.json"
        assert _AGENT_FILE_MAP["ui_planner"] == "planner.json"


class TestTemplateLoader:
    """Tests for the TemplateLoader class."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.base = Path(self.tmpdir)
        self.loader = TemplateLoader(base_path=self.base)

    def test_list_versions_no_manifest_returns_v1(self):
        versions = self.loader.list_versions()
        assert versions == ["v1"]

    def test_list_versions_reads_manifest(self):
        manifest = self.base / "versions.json"
        manifest.write_text(json.dumps({"versions": ["v1", "v2"]}), encoding="utf-8")
        versions = self.loader.list_versions()
        assert versions == ["v1", "v2"]

    def test_load_templates_for_agent_missing_returns_empty(self):
        result = self.loader.load_templates_for_agent("v1", "planner")
        assert result == []

    def test_load_templates_for_agent_unknown_agent_returns_empty(self):
        result = self.loader.load_templates_for_agent("v1", "nonexistent_agent")
        assert result == []

    def test_load_templates_for_agent_reads_file(self):
        version_dir = self.base / "v1"
        version_dir.mkdir(parents=True)
        template_data = [{"name": "plan_template", "prompt": "You are a planner."}]
        (version_dir / "planner.json").write_text(json.dumps(template_data), encoding="utf-8")
        result = self.loader.load_templates_for_agent("v1", "planner")
        assert len(result) == 1
        assert result[0]["name"] == "plan_template"

    def test_load_templates_for_legacy_name(self):
        """Legacy agent names should load the consolidated template file."""
        version_dir = self.base / "v1"
        version_dir.mkdir(parents=True)
        template_data = [{"name": "researcher_tpl"}]
        (version_dir / "researcher.json").write_text(json.dumps(template_data), encoding="utf-8")
        # "explorer" maps to researcher.json
        result = self.loader.load_templates_for_agent("v1", "explorer")
        assert len(result) == 1

    def test_default_version(self):
        assert self.loader.default_version() == "v1"

    def test_module_level_singleton(self):
        assert isinstance(template_loader, TemplateLoader)
