"""Tests for vetinari.adapters.speculative_decoding.

Covers:
- SpeculativeDecodingCapability and SpeculativeDecodingConfig dataclass defaults
- detect_speculative_capability() when llama_cpp IS importable
- detect_speculative_capability() when llama_cpp is NOT importable
- Capability detection caching (runs once per process, thread-safe)
- LlamaCppProviderAdapter._get_speculative_config() reads from VetinariSettings
- Inference path: capability check logs correctly based on config
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.adapters.speculative_decoding import (
    SpeculativeDecodingCapability,
    SpeculativeDecodingConfig,
    _run_capability_detection,
    detect_speculative_capability,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _reset_capability_cache() -> None:
    """Reset the module-level capability cache between tests.

    Uses detect_speculative_capability.__globals__ to target the exact
    module namespace the function writes to, avoiding the dual-module
    problem when conftest's _restore_sys_modules replaces sys.modules entries.
    """
    detect_speculative_capability.__globals__["_capability_cache"] = None


# ── SpeculativeDecodingCapability defaults ─────────────────────────────────────


class TestSpeculativeDecodingCapability:
    """Tests for the SpeculativeDecodingCapability dataclass."""

    def test_fields_are_present(self) -> None:
        cap = SpeculativeDecodingCapability(
            supported=True,
            has_draft_model=False,
            draft_model_id=None,
            detection_method="inspect.signature",
        )
        assert cap.supported is True
        assert cap.has_draft_model is False
        assert cap.draft_model_id is None
        assert cap.detection_method == "inspect.signature"

    def test_frozen_prevents_mutation(self) -> None:
        cap = SpeculativeDecodingCapability(
            supported=False,
            has_draft_model=False,
            draft_model_id=None,
            detection_method="test",
        )
        with pytest.raises(AttributeError, match="cannot assign"):
            cap.supported = True  # type: ignore[misc]


# ── SpeculativeDecodingConfig defaults ───────────────────────────────────────


class TestSpeculativeDecodingConfig:
    """Tests for the SpeculativeDecodingConfig dataclass defaults."""

    def test_default_enabled_is_false(self) -> None:
        cfg = SpeculativeDecodingConfig()
        assert cfg.enabled is False

    def test_default_draft_model_id_is_none(self) -> None:
        cfg = SpeculativeDecodingConfig()
        assert cfg.draft_model_id is None

    def test_default_draft_n_tokens_is_five(self) -> None:
        cfg = SpeculativeDecodingConfig()
        assert cfg.draft_n_tokens == 5

    def test_default_use_prompt_lookup_fallback_is_true(self) -> None:
        cfg = SpeculativeDecodingConfig()
        assert cfg.use_prompt_lookup_fallback is True

    def test_custom_values_stored_correctly(self) -> None:
        cfg = SpeculativeDecodingConfig(
            enabled=True,
            draft_model_id="draft-3b",
            draft_n_tokens=8,
            use_prompt_lookup_fallback=False,
        )
        assert cfg.enabled is True
        assert cfg.draft_model_id == "draft-3b"
        assert cfg.draft_n_tokens == 8
        assert cfg.use_prompt_lookup_fallback is False


# ── detect_speculative_capability — llama_cpp unavailable ─────────────────────


class TestDetectCapabilityLlamaCppUnavailable:
    """detect_speculative_capability when llama_cpp cannot be imported."""

    def setup_method(self) -> None:
        _reset_capability_cache()

    def teardown_method(self) -> None:
        _reset_capability_cache()

    @staticmethod
    def _unavailable_import(name: str) -> tuple[None, bool]:
        return None, False

    def test_returns_unsupported_when_llama_cpp_missing(self) -> None:
        cap = detect_speculative_capability(_import_fn=self._unavailable_import)

        assert cap.supported is False
        assert cap.has_draft_model is False
        assert cap.draft_model_id is None
        assert cap.detection_method == "llama_cpp_unavailable"

    def test_graceful_fallback_no_exception_raised(self) -> None:
        # Must not raise
        cap = detect_speculative_capability(_import_fn=self._unavailable_import)

        assert isinstance(cap, SpeculativeDecodingCapability)


# ── detect_speculative_capability — llama_cpp available ──────────────────────


class TestDetectCapabilityLlamaCppAvailable:
    """detect_speculative_capability when llama_cpp is importable."""

    def setup_method(self) -> None:
        _reset_capability_cache()

    def teardown_method(self) -> None:
        _reset_capability_cache()

    def _make_mock_llama_cpp(self, *, has_draft_model_param: bool) -> MagicMock:
        """Build a fake llama_cpp module with configurable Llama signature.

        Uses real classes (not MagicMock) because ``inspect.signature`` needs
        a genuine ``__init__`` to introspect parameter names.
        """
        mock_llama_cpp = MagicMock()

        if has_draft_model_param:

            class FakeLlama:
                def __init__(self, model_path: str, draft_model: object = None, **kwargs: object) -> None:
                    pass  # noqa: VET031 — test fake: intentionally empty __init__

        else:

            class FakeLlama:
                def __init__(self, model_path: str, **kwargs: object) -> None:
                    pass  # noqa: VET031 — test fake: intentionally empty __init__

        mock_llama_cpp.Llama = FakeLlama
        return mock_llama_cpp

    def _available_import(self, mock_llama_cpp: MagicMock) -> object:
        """Return an import function that yields the given mock module."""

        def _import(name: str) -> tuple[MagicMock, bool]:
            return mock_llama_cpp, True

        return _import

    def test_supported_true_when_draft_model_param_present(self) -> None:
        mock_llama_cpp = self._make_mock_llama_cpp(has_draft_model_param=True)
        cap = detect_speculative_capability(
            _import_fn=self._available_import(mock_llama_cpp),
        )

        assert cap.supported is True
        assert cap.detection_method == "inspect.signature"

    def test_supported_false_when_draft_model_param_absent(self) -> None:
        mock_llama_cpp = self._make_mock_llama_cpp(has_draft_model_param=False)
        cap = detect_speculative_capability(
            _import_fn=self._available_import(mock_llama_cpp),
        )

        assert cap.supported is False
        assert cap.detection_method == "inspect.signature"

    def test_has_draft_model_true_when_draft_model_id_provided(self) -> None:
        mock_llama_cpp = self._make_mock_llama_cpp(has_draft_model_param=True)
        cap = detect_speculative_capability(
            draft_model_id="my-draft-7b",
            _import_fn=self._available_import(mock_llama_cpp),
        )

        assert cap.has_draft_model is True
        assert cap.draft_model_id == "my-draft-7b"

    def test_has_draft_model_false_when_no_draft_model_id(self) -> None:
        mock_llama_cpp = self._make_mock_llama_cpp(has_draft_model_param=True)
        cap = detect_speculative_capability(
            _import_fn=self._available_import(mock_llama_cpp),
        )

        assert cap.has_draft_model is False
        assert cap.draft_model_id is None

    def test_returns_unsupported_when_llama_class_missing(self) -> None:
        mock_llama_cpp = MagicMock(spec=[])  # No attributes at all
        cap = detect_speculative_capability(
            _import_fn=self._available_import(mock_llama_cpp),
        )

        assert cap.supported is False
        assert cap.detection_method == "llama_class_missing"

    def test_returns_detection_error_on_signature_failure(self) -> None:
        mock_llama_cpp = MagicMock()

        # Patch inspect.signature to raise TypeError, simulating an
        # uninspectable __init__ (e.g. a C extension class)
        with patch(
            "vetinari.adapters.speculative_decoding.inspect.signature",
            side_effect=TypeError("unsupported callable"),
        ):
            cap = _run_capability_detection(
                _import_fn=self._available_import(mock_llama_cpp),
            )

        assert cap.supported is False
        assert cap.detection_method == "detection_error"


# ── Caching behaviour ─────────────────────────────────────────────────────────


class TestCapabilityDetectionCaching:
    """detect_speculative_capability result is cached after first call.

    These tests call detect_speculative_capability() WITHOUT _import_fn so
    they exercise the real cache path.  The _import_fn parameter explicitly
    bypasses caching (by design), so using it here would not test caching.
    """

    def setup_method(self) -> None:
        _reset_capability_cache()

    def teardown_method(self) -> None:
        _reset_capability_cache()

    def test_detection_runs_once_across_multiple_calls(self) -> None:
        # First call runs detection and populates the cache
        first = detect_speculative_capability()
        # Subsequent calls return the exact same cached object
        second = detect_speculative_capability()
        third = detect_speculative_capability()

        assert first is second
        assert second is third

    def test_cache_is_populated_after_first_call(self) -> None:
        _globals = detect_speculative_capability.__globals__
        assert _globals["_capability_cache"] is None

        detect_speculative_capability()

        assert _globals["_capability_cache"] is not None
        assert isinstance(_globals["_capability_cache"], SpeculativeDecodingCapability)

    def test_caller_supplied_draft_model_id_bypasses_cache_store(self) -> None:
        """When draft_model_id is provided, the result must NOT be stored in cache."""
        _globals = detect_speculative_capability.__globals__

        detect_speculative_capability(draft_model_id="some-draft")

        # Cache should still be None — caller-specific results are never cached
        assert _globals["_capability_cache"] is None

    def test_import_fn_bypasses_cache_entirely(self) -> None:
        """When _import_fn is provided, neither reads from nor writes to cache."""
        _globals = detect_speculative_capability.__globals__

        # Populate cache with a real result first
        detect_speculative_capability()
        cached = _globals["_capability_cache"]
        assert cached is not None

        # Call with _import_fn — should get a DIFFERENT result, not the cached one
        custom_result = detect_speculative_capability(
            _import_fn=lambda name: (None, False),
        )
        assert custom_result is not cached
        assert custom_result.detection_method == "llama_cpp_unavailable"

        # Cache should remain unchanged (custom result was not stored)
        assert _globals["_capability_cache"] is cached


# ── _get_speculative_config() integration ─────────────────────────────────────


class TestGetSpeculativeConfig:
    """LlamaCppProviderAdapter._get_speculative_config reads from VetinariSettings."""

    def _make_adapter(self) -> object:
        """Build a minimal adapter instance via __new__ to bypass heavy __init__.

        Only the attributes accessed by _get_speculative_config are populated —
        the method only calls get_settings(), which is separately patched per test.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        from vetinari.adapters.llama_cpp_adapter import LlamaCppProviderAdapter

        adapter = LlamaCppProviderAdapter.__new__(LlamaCppProviderAdapter)
        # Minimal attribute set required to avoid AttributeError on unrelated code paths
        adapter._models_dir = Path("/dev/null/nonexistent")
        adapter._gpu_layers = -1
        adapter._default_context_length = 8192
        adapter._memory_budget_gb = 16
        adapter._ram_budget_gb = 30.0
        adapter._cpu_offload_enabled = True
        adapter._cache_type_k = "f16"
        adapter._cache_type_v = "f16"
        adapter._loaded_models = {}
        adapter._model_locks = {}
        adapter._discovered_models = []
        adapter._registry_lock = threading.Lock()
        adapter._calibration_pool = ThreadPoolExecutor(max_workers=1)
        return adapter

    def test_returns_speculative_decoding_config(self) -> None:
        adapter = self._make_adapter()
        with patch(
            "vetinari.adapters.llama_cpp_adapter.get_settings",
        ) as mock_settings:
            mock_settings.return_value.speculative_decoding_enabled = False
            mock_settings.return_value.speculative_draft_model_id = None
            mock_settings.return_value.speculative_draft_n_tokens = 5

            cfg = adapter._get_speculative_config()

        assert isinstance(cfg, SpeculativeDecodingConfig)
        assert cfg.enabled is False
        assert cfg.draft_model_id is None
        assert cfg.draft_n_tokens == 5

    def test_enabled_flag_propagates_from_settings(self) -> None:
        adapter = self._make_adapter()
        with patch(
            "vetinari.adapters.llama_cpp_adapter.get_settings",
        ) as mock_settings:
            mock_settings.return_value.speculative_decoding_enabled = True
            mock_settings.return_value.speculative_draft_model_id = "draft-7b"
            mock_settings.return_value.speculative_draft_n_tokens = 8

            cfg = adapter._get_speculative_config()

        assert cfg.enabled is True
        assert cfg.draft_model_id == "draft-7b"
        assert cfg.draft_n_tokens == 8

    def test_use_prompt_lookup_fallback_always_true(self) -> None:
        adapter = self._make_adapter()
        with patch(
            "vetinari.adapters.llama_cpp_adapter.get_settings",
        ) as mock_settings:
            mock_settings.return_value.speculative_decoding_enabled = False
            mock_settings.return_value.speculative_draft_model_id = None
            mock_settings.return_value.speculative_draft_n_tokens = 5

            cfg = adapter._get_speculative_config()

        # Always True — no config to disable it
        assert cfg.use_prompt_lookup_fallback is True


# ── Standard inference proceeds without error when cap not detected ───────────


class TestStandardInferenceFallback:
    """When capability is not detected, standard inference proceeds without error."""

    def test_unsupported_capability_does_not_error(self) -> None:
        """detect_speculative_capability returns a usable object even when unsupported."""
        _reset_capability_cache()
        try:
            cap = detect_speculative_capability(
                _import_fn=lambda name: (None, False),
            )

            # Callers check cap.supported before using draft model
            assert cap.supported is False
            # No exception raised — safe to use
            assert isinstance(cap, SpeculativeDecodingCapability)
        finally:
            _reset_capability_cache()
