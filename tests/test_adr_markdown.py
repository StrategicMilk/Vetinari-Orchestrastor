"""Tests for ADRSystem markdown rendering and export functionality."""

from __future__ import annotations

import pytest

from vetinari.adr import ADRStatus, ADRSystem


class TestRenderMarkdown:
    """Tests for ADRSystem.render_markdown()."""

    def test_render_markdown_includes_all_fields(self, tmp_path):
        """Rendered markdown must contain the ADR's key fields."""
        system = ADRSystem(storage_path=str(tmp_path))
        adr = system.create_adr(
            title="Use six-agent pipeline",
            category="architecture",
            context="We need a structured agent hierarchy.",
            decision="Adopt a six-agent pipeline.",
            consequences="Clearer separation of concerns.",
            created_by="test-author",
            status=ADRStatus.ACCEPTED.value,
        )

        md = system.render_markdown(adr.adr_id)

        assert adr.adr_id in md
        assert "Use six-agent pipeline" in md
        assert "accepted" in md
        assert "architecture" in md
        assert "test-author" in md
        assert "We need a structured agent hierarchy." in md
        assert "Adopt a six-agent pipeline." in md
        assert "Clearer separation of concerns." in md
        assert "## Context" in md
        assert "## Decision" in md
        assert "## Consequences" in md
        assert "## Related ADRs" in md

    def test_render_markdown_related_adrs_none_when_empty(self, tmp_path):
        """Related ADRs section shows 'None' when no related ADRs exist."""
        system = ADRSystem(storage_path=str(tmp_path))
        adr = system.create_adr(
            title="Empty related",
            category="architecture",
            context="Context.",
            decision="Decision.",
        )

        md = system.render_markdown(adr.adr_id)

        assert "## Related ADRs\n\nNone" in md

    def test_render_markdown_related_adrs_listed(self, tmp_path):
        """Related ADRs section lists IDs when related ADRs are present."""
        system = ADRSystem(storage_path=str(tmp_path))
        adr = system.create_adr(
            title="With related",
            category="architecture",
            context="Context.",
            decision="Decision.",
            related_adrs=["ADR-0010", "ADR-0011"],
        )

        md = system.render_markdown(adr.adr_id)

        assert "ADR-0010" in md
        assert "ADR-0011" in md

    def test_render_markdown_unknown_adr_raises(self, tmp_path):
        """render_markdown raises KeyError for an unknown ADR ID."""
        system = ADRSystem(storage_path=str(tmp_path))

        with pytest.raises(KeyError, match="ADR-9999"):
            system.render_markdown("ADR-9999")


class TestExportAllMarkdown:
    """Tests for ADRSystem.export_all_markdown()."""

    def test_export_all_markdown_creates_files(self, tmp_path):
        """export_all_markdown writes one .md file per ADR."""
        adr_dir = tmp_path / "adrs"
        export_dir = tmp_path / "export"

        system = ADRSystem(storage_path=str(adr_dir))
        adr1 = system.create_adr(
            title="First decision",
            category="architecture",
            context="Context one.",
            decision="Decision one.",
        )
        adr2 = system.create_adr(
            title="Second decision",
            category="security",
            context="Context two.",
            decision="Decision two.",
        )

        paths = system.export_all_markdown(export_dir)

        assert len(paths) == 2
        written_names = {p.split("/")[-1].split("\\")[-1] for p in paths}
        assert f"{adr1.adr_id}.md" in written_names
        assert f"{adr2.adr_id}.md" in written_names

        for path in paths:
            from pathlib import Path

            assert Path(path).exists()
            content = Path(path).read_text(encoding="utf-8")
            assert "## Decision" in content

    def test_export_all_markdown_creates_output_dir(self, tmp_path):
        """export_all_markdown creates the output directory if absent."""
        adr_dir = tmp_path / "adrs"
        export_dir = tmp_path / "does_not_exist" / "nested"

        system = ADRSystem(storage_path=str(adr_dir))
        system.create_adr(
            title="Any",
            category="architecture",
            context="Context.",
            decision="Decision.",
        )

        paths = system.export_all_markdown(export_dir)

        assert export_dir.exists()
        assert len(paths) == 1

    def test_export_all_markdown_returns_empty_when_no_adrs(self, tmp_path):
        """export_all_markdown returns an empty list when no ADRs exist."""
        system = ADRSystem(storage_path=str(tmp_path / "adrs"))
        export_dir = tmp_path / "export"

        paths = system.export_all_markdown(export_dir)

        assert paths == []
