"""Tests for vetinari.model_relay — YAML-backed model catalog types."""

from __future__ import annotations

import warnings

# Suppress the deprecation warning when importing model_relay
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from vetinari.model_relay import (
        ModelEntry,
        ModelSelection,
        ModelStatus,
        RoutingPolicy,
    )


class TestModelStatus:
    """Tests for the ModelStatus enum."""

    def test_values(self):
        assert ModelStatus.AVAILABLE.value == "available"
        assert ModelStatus.LOADING.value == "loading"
        assert ModelStatus.UNAVAILABLE.value == "unavailable"


class TestModelEntry:
    """Tests for the ModelEntry dataclass."""

    def test_defaults(self):
        entry = ModelEntry(model_id="test", provider="local", display_name="Test Model")
        assert entry.context_window == 4096
        assert entry.latency_hint == "medium"
        assert entry.privacy_level == "local"
        assert entry.cost_per_1k_tokens == 0.0

    def test_to_dict(self):
        entry = ModelEntry(model_id="m1", provider="openai", display_name="M1")
        d = entry.to_dict()
        assert d["model_id"] == "m1"
        assert d["provider"] == "openai"

    def test_from_dict(self):
        data = {
            "model_id": "qwen-32b",
            "provider": "local",
            "display_name": "Qwen 32B",
            "context_window": 32768,
        }
        entry = ModelEntry.from_dict(data)
        assert entry.model_id == "qwen-32b"
        assert entry.context_window == 32768

    def test_from_dict_defaults(self):
        entry = ModelEntry.from_dict({"model_id": "x"})
        assert entry.provider == "local"
        assert entry.context_window == 4096


class TestRoutingPolicy:
    """Tests for the RoutingPolicy dataclass."""

    def test_defaults(self):
        policy = RoutingPolicy()
        assert policy.local_first is True
        assert policy.privacy_weight == 1.0
        assert policy.cost_weight == 0.3

    def test_to_dict_roundtrip(self):
        policy = RoutingPolicy(local_first=False, cost_weight=0.8)
        d = policy.to_dict()
        restored = RoutingPolicy.from_dict(d)
        assert restored.local_first is False
        assert restored.cost_weight == 0.8


class TestModelSelection:
    """Tests for the ModelSelection dataclass."""

    def test_fields(self):
        sel = ModelSelection(
            model_id="m1",
            provider="local",
            endpoint="http://localhost:1234",
            reasoning="cheapest available",
            confidence=0.9,
            latency_estimate="low",
        )
        assert sel.model_id == "m1"
        assert sel.confidence == 0.9

    def test_to_dict(self):
        sel = ModelSelection("m1", "local", "ep", "reason", 0.8, "medium")
        d = sel.to_dict()
        assert d["model_id"] == "m1"
        assert d["reasoning"] == "reason"
        assert len(d) == 6
