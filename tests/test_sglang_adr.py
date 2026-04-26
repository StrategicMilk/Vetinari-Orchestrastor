"""Tests for ADR-0065 — SGLang skip decision."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestSGLangADR:
    """Tests for the SGLang skip decision ADR."""

    def test_adr_0065_exists(self):
        """ADR-0065 should exist as a JSON file."""
        adr_path = Path("adr/ADR-0065.json")
        assert adr_path.exists(), "ADR-0065.json not found in adr/"

    def test_adr_0065_valid_json(self):
        """ADR-0065 should be valid JSON."""
        with open("adr/ADR-0065.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "adr_id" in data
        assert data["adr_id"] == "ADR-0065"

    def test_adr_0065_accepted(self):
        """ADR-0065 should have accepted status."""
        with open("adr/ADR-0065.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "accepted"

    def test_adr_0065_references_sglang(self):
        """ADR-0065 should reference SGLang in title or context."""
        with open("adr/ADR-0065.json", encoding="utf-8") as f:
            data = json.load(f)
        sglang_mentioned = (
            "sglang" in data["title"].lower() or "SGLang" in data["title"] or "sglang" in data["context"].lower()
        )
        assert sglang_mentioned

    def test_adr_0065_references_llama_cpp(self):
        """ADR-0065 should mention llama-cpp-python as the chosen alternative."""
        with open("adr/ADR-0065.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "llama-cpp-python" in data["decision"]

    def test_adr_0065_category_is_architecture(self):
        """ADR-0065 should be in the architecture category."""
        with open("adr/ADR-0065.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["category"] == "architecture"

    def test_adr_loadable_via_system(self):
        """ADR-0065 should be loadable via the ADR system."""
        from vetinari.adr import ADRSystem

        system = ADRSystem(storage_path="adr")
        adrs = system.list_adrs()
        adr_ids = [a.adr_id for a in adrs]
        assert "ADR-0065" in adr_ids
