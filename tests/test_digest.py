"""Tests for vetinari.notifications.digest -- DailyDigest structure and serialization."""

from __future__ import annotations

import pytest

from tests.factories import make_daily_digest
from vetinari.notifications.digest import DailyDigest, DigestGenerator, DigestSection

# -- DigestSection ------------------------------------------------------------


class TestDigestSection:
    """DigestSection holds a title and items list."""

    def test_section_has_title_and_items(self) -> None:
        """DigestSection stores title and items correctly."""
        section = DigestSection(title="Tasks", items=["5 completed", "1 failed"])
        assert section.title == "Tasks"
        assert len(section.items) == 2

    def test_section_items_default_empty(self) -> None:
        """DigestSection.items defaults to an empty list."""
        section = DigestSection(title="Empty")
        assert section.items == []


# -- DailyDigest serialization ------------------------------------------------


class TestDailyDigest:
    """DailyDigest.to_dict() and to_text() produce correct output."""

    def test_to_dict_is_serializable(self) -> None:
        """to_dict() returns a plain dict without non-JSON types."""
        import json

        digest = make_daily_digest()
        result = digest.to_dict()
        # Should not raise
        json.dumps(result)
        assert result["overall_health"] == "healthy"
        assert isinstance(result["sections"], list)
        assert len(result["sections"]) == 2

    def test_to_dict_sections_have_title_and_items(self) -> None:
        """Each section in to_dict() has title, items, and metrics keys."""
        digest = make_daily_digest()
        result = digest.to_dict()
        for section in result["sections"]:
            assert "title" in section
            assert "items" in section
            assert "metrics" in section

    def test_to_text_returns_non_empty_string(self) -> None:
        """to_text() returns a human-readable string with at least 10 characters."""
        digest = make_daily_digest()
        text = digest.to_text()
        assert isinstance(text, str)
        assert len(text) > 10

    def test_to_text_contains_section_titles(self) -> None:
        """to_text() includes the section titles in the output."""
        digest = make_daily_digest()
        text = digest.to_text()
        assert "Tasks" in text
        assert "Health" in text

    def test_overall_health_defaults_to_healthy(self) -> None:
        """DailyDigest.overall_health defaults to 'healthy' when not specified."""
        digest = DailyDigest(generated_at="2026-04-01T00:00:00+00:00")
        assert digest.overall_health == "healthy"


# -- DigestGenerator ----------------------------------------------------------


class TestDigestGenerator:
    """generate_digest() returns a DailyDigest with exactly 5 sections."""

    def test_generate_digest_returns_daily_digest(self) -> None:
        """generate_digest() returns a DailyDigest instance."""
        generator = DigestGenerator()
        result = generator.generate_digest()
        assert isinstance(result, DailyDigest)

    def test_generate_digest_has_five_sections(self) -> None:
        """The digest always contains exactly 5 sections regardless of data availability."""
        generator = DigestGenerator()
        result = generator.generate_digest()
        assert len(result.sections) == 5

    def test_all_sections_have_title_and_items(self) -> None:
        """Every section has a non-empty title and at least one item."""
        generator = DigestGenerator()
        result = generator.generate_digest()
        for section in result.sections:
            assert isinstance(section.title, str)
            assert len(section.title) > 0
            assert isinstance(section.items, list)
            assert len(section.items) >= 1

    def test_overall_health_is_valid_value(self) -> None:
        """overall_health is one of the three valid values."""
        generator = DigestGenerator()
        result = generator.generate_digest()
        assert result.overall_health in {"healthy", "warning", "degraded"}

    def test_generated_at_is_iso_timestamp(self) -> None:
        """generated_at is a non-empty ISO 8601 timestamp string."""
        generator = DigestGenerator()
        result = generator.generate_digest()
        assert isinstance(result.generated_at, str)
        assert "T" in result.generated_at  # Minimal ISO 8601 check
