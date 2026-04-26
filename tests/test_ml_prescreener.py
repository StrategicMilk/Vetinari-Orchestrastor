"""Tests for vetinari.ml (MLModelRegistry) and vetinari.ml.quality_prescreener."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vetinari.exceptions import ConfigurationError, ModelNotFoundError
from vetinari.ml import MLModelRegistry, ModelInfo, get_ml_registry, reset_ml_registry
from vetinari.ml.quality_prescreener import (
    INCONCLUSIVE_HIGH,
    INCONCLUSIVE_LOW,
    PreScreenResult,
    QualityPreScreener,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the ML registry singleton before/after each test."""
    reset_ml_registry()
    yield
    reset_ml_registry()


@pytest.fixture
def registry() -> MLModelRegistry:
    """Return a fresh MLModelRegistry instance."""
    return MLModelRegistry()


@pytest.fixture
def screener() -> QualityPreScreener:
    """Return a QualityPreScreener instance."""
    return QualityPreScreener()


# ---------------------------------------------------------------------------
# MLModelRegistry tests
# ---------------------------------------------------------------------------


class TestMLModelRegistryLoad:
    def test_load_raises_file_not_found_for_missing_path(self, registry: MLModelRegistry, tmp_path: Path):
        missing = tmp_path / "nonexistent_model.joblib"
        with pytest.raises(ModelNotFoundError, match="Model not found"):
            registry.load("my_model", missing)

    def test_load_raises_value_error_for_unsupported_extension(self, registry: MLModelRegistry, tmp_path: Path):
        bad_file = tmp_path / "model.xyz"
        bad_file.write_text("data", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Unsupported model format"):
            registry.load("bad_model", bad_file)

    def test_load_returns_cached_model_on_second_call(self, registry: MLModelRegistry, tmp_path: Path):
        # Inject a mock model directly so we can test cache behaviour without real files
        mock_model = MagicMock()
        registry._models["cached"] = mock_model
        registry._model_info["cached"] = ModelInfo(name="cached", model_type="sklearn", loaded=True)
        # Create a dummy file so the existence check would pass if hit
        dummy = tmp_path / "cached.joblib"
        dummy.write_text("data", encoding="utf-8")
        # Should return the cached model without attempting to open the file
        result = registry.load("cached", dummy)
        assert result is mock_model


class TestMLModelRegistryIsLoaded:
    def test_is_loaded_returns_false_before_load(self, registry: MLModelRegistry):
        assert registry.is_loaded("unknown_model") is False

    def test_is_loaded_returns_true_after_injecting_model(self, registry: MLModelRegistry):
        registry._models["my_model"] = MagicMock()
        assert registry.is_loaded("my_model") is True


class TestMLModelRegistryPredict:
    def test_predict_raises_key_error_for_unloaded_model(self, registry: MLModelRegistry):
        with pytest.raises(KeyError, match="not loaded"):
            registry.predict("ghost", [[1, 2, 3]])

    def test_predict_calls_model_predict(self, registry: MLModelRegistry):
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 0]
        registry._models["clf"] = mock_model
        result = registry.predict("clf", [[0.1, 0.2]])
        mock_model.predict.assert_called_once_with([[0.1, 0.2]])
        assert result == [1, 0]

    def test_predict_calls_encode_when_no_predict(self, registry: MLModelRegistry):
        mock_model = MagicMock(spec=["encode"])
        mock_model.encode.return_value = [[0.5, 0.5]]
        registry._models["emb"] = mock_model
        result = registry.predict("emb", "hello world")
        mock_model.encode.assert_called_once_with("hello world")
        assert result == [[0.5, 0.5]]

    def test_predict_raises_type_error_when_model_has_no_predict_or_encode(self, registry: MLModelRegistry):
        odd_model = object()  # has neither predict nor encode
        registry._models["odd"] = odd_model
        with pytest.raises(TypeError, match="no predict\\(\\) or encode\\(\\)"):
            registry.predict("odd", [1])


class TestMLModelRegistryUnload:
    def test_unload_removes_model_from_registry(self, registry: MLModelRegistry):
        registry._models["to_remove"] = MagicMock()
        registry._model_info["to_remove"] = ModelInfo(name="to_remove", model_type="sklearn", loaded=True)
        registry.unload("to_remove")
        assert registry.is_loaded("to_remove") is False

    def test_unload_noop_for_unknown_model(self, registry: MLModelRegistry):
        # Should not raise — unloading unknown model is a silent no-op
        registry.unload("does_not_exist")
        assert registry.is_loaded("does_not_exist") is False

    def test_unload_marks_model_info_as_not_loaded(self, registry: MLModelRegistry):
        info = ModelInfo(name="m", model_type="sklearn", loaded=True)
        registry._models["m"] = MagicMock()
        registry._model_info["m"] = info
        registry.unload("m")
        assert info.loaded is False


class TestMLModelRegistryGetStatus:
    def test_get_status_returns_empty_state_initially(self, registry: MLModelRegistry):
        status = registry.get_status()
        assert status["model_count"] == 0
        assert status["loaded_models"] == []

    def test_get_status_reflects_loaded_models(self, registry: MLModelRegistry):
        registry._models["m1"] = MagicMock()
        registry._model_info["m1"] = ModelInfo(name="m1", model_type="onnx", loaded=True)
        status = registry.get_status()
        assert status["model_count"] == 1
        assert "m1" in status["loaded_models"]
        assert status["models"]["m1"]["type"] == "onnx"
        assert status["models"]["m1"]["loaded"] is True


class TestGetMLRegistry:
    def test_get_ml_registry_returns_singleton(self):
        r1 = get_ml_registry()
        r2 = get_ml_registry()
        assert r1 is r2

    def test_reset_ml_registry_clears_singleton(self):
        r1 = get_ml_registry()
        reset_ml_registry()
        r2 = get_ml_registry()
        assert r1 is not r2


# ---------------------------------------------------------------------------
# QualityPreScreener — Tier 1 tests
# ---------------------------------------------------------------------------


class TestTier1Rules:
    def test_syntax_error_scores_zero_and_skips_llm(self, screener: QualityPreScreener):
        bad_code = "def broken(\n    x = "
        result = screener.screen(bad_code)
        assert result.score == 0.0
        assert result.skip_llm_judge is True
        assert result.tier_used == 1
        assert any("Syntax error" in issue for issue in result.issues)

    def test_bare_except_is_flagged(self, screener: QualityPreScreener):
        code = "def f():\n    try:\n        pass\n    except:\n        pass\n"
        result = screener._tier1_rules(code)
        assert any("Bare except" in issue for issue in result.issues)

    def test_open_without_encoding_is_flagged(self, screener: QualityPreScreener):
        code = 'def f():\n    with open("file.txt") as fh:\n        return fh.read()\n'
        result = screener._tier1_rules(code)
        assert any("encoding" in issue for issue in result.issues)

    def test_open_with_utf8_encoding_is_clean(self, screener: QualityPreScreener):
        code = 'def f():\n    with open("file.txt", encoding="utf-8") as fh:\n        return fh.read()\n'
        result = screener._tier1_rules(code)
        assert not any("encoding" in issue for issue in result.issues)

    def test_print_in_production_code_is_flagged(self, screener: QualityPreScreener):
        code = "def f():\n    print('hello')\n"
        result = screener._tier1_rules(code)
        assert any("print()" in issue for issue in result.issues)

    def test_print_in_test_code_is_not_flagged(self, screener: QualityPreScreener):
        code = "def test_f():\n    print('debug')\n"
        result = screener._tier1_rules(code)
        assert not any("print()" in issue for issue in result.issues)

    def test_clean_code_scores_1_0(self, screener: QualityPreScreener):
        code = "from __future__ import annotations\n\ndef add(x: int, y: int) -> int:\n    return x + y\n"
        result = screener._tier1_rules(code)
        assert result.score == 1.0
        assert result.issues == []


# ---------------------------------------------------------------------------
# QualityPreScreener — Tier 2 tests
# ---------------------------------------------------------------------------

# Well-structured code that should score above INCONCLUSIVE_HIGH
WELL_STRUCTURED_CODE = '''\
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes data records.

    Handles ingestion, validation, and transformation of records.
    """

    def process(self, record: dict[str, Any]) -> dict[str, Any]:
        """Process a single record.

        Args:
            record: The input record to process.

        Returns:
            Transformed record dictionary.
        """
        if not record:
            return {}
        return {k: str(v) for k, v in record.items()}

    def validate(self, record: dict[str, Any]) -> bool:
        """Validate a record has required fields.

        Args:
            record: Record to validate.

        Returns:
            True if the record is valid.
        """
        return bool(record)

    def transform(self, record: dict[str, Any]) -> list[str]:
        """Transform a record into a list of values.

        Args:
            record: Record to transform.

        Returns:
            List of string values.
        """
        return list(record.values())
'''

# Poorly structured code that should score below INCONCLUSIVE_LOW
POORLY_STRUCTURED_CODE = "x = 1\ny = 2\n"


class TestTier2Features:
    def test_well_structured_code_scores_high_and_skips_llm(self, screener: QualityPreScreener):
        result = screener.screen(WELL_STRUCTURED_CODE)
        assert result.score > INCONCLUSIVE_HIGH
        assert result.skip_llm_judge is True
        assert result.tier_used == 2

    def test_poorly_structured_code_scores_low_and_skips_llm(self, screener: QualityPreScreener):
        result = screener.screen(POORLY_STRUCTURED_CODE)
        assert result.score <= INCONCLUSIVE_LOW
        assert result.skip_llm_judge is True
        assert result.tier_used == 2

    def test_inconclusive_range_sets_skip_llm_false(self, screener: QualityPreScreener):
        # Craft code that will fall in the inconclusive zone:
        # Has functions but no annotations or docstrings, no future import
        code = (
            "def alpha(x, y):\n"
            "    return x + y\n\n"
            "def beta(a, b):\n"
            "    return a - b\n\n"
            "def gamma(p, q):\n"
            "    return p * q\n\n"
            "def delta(m, n):\n"
            "    return m / n\n"
            "result = alpha(1, 2)\n"
        )
        result = screener.screen(code)
        # Score should be in [INCONCLUSIVE_LOW, INCONCLUSIVE_HIGH]
        assert INCONCLUSIVE_LOW <= result.score <= INCONCLUSIVE_HIGH
        assert result.skip_llm_judge is False

    def test_empty_code_scores_low(self, screener: QualityPreScreener):
        result = screener.screen("")
        assert result.score <= INCONCLUSIVE_LOW

    def test_minimal_code_flagged_as_very_short(self, screener: QualityPreScreener):
        code = "x = 1\n"
        result = screener._tier2_features(code, {})
        assert any("Very short" in issue for issue in result.issues)

    def test_details_populated_for_tier2(self, screener: QualityPreScreener):
        result = screener._tier2_features(WELL_STRUCTURED_CODE, {})
        assert "function_count" in result.details
        assert "class_count" in result.details
        assert "annotation_ratio" in result.details
        assert "has_future_import" in result.details
        assert result.details["has_future_import"] is True

    def test_low_annotation_ratio_flagged_for_many_functions(self, screener: QualityPreScreener):
        # 3 functions, none annotated
        code = "def f1(x, y):\n    return x\n\ndef f2(a, b):\n    return a\n\ndef f3(p, q):\n    return p\n"
        result = screener._tier2_features(code, {})
        assert any("annotation" in issue.lower() for issue in result.issues)


# ---------------------------------------------------------------------------
# PreScreenResult tests
# ---------------------------------------------------------------------------


class TestPreScreenResult:
    def test_to_dict_serializes_correctly(self):
        result = PreScreenResult(
            tier_used=2,
            score=0.75,
            skip_llm_judge=True,
            issues=["Low annotation ratio"],
            details={"function_count": 3},
        )
        d = result.to_dict()
        assert d["tier_used"] == 2
        assert d["score"] == 0.75
        assert d["skip_llm_judge"] is True
        assert d["issues"] == ["Low annotation ratio"]
        assert d["details"] == {"function_count": 3}

    def test_score_is_rounded_to_3_decimal_places(self):
        result = PreScreenResult(score=0.666666)
        d = result.to_dict()
        assert d["score"] == 0.667

    def test_default_mutable_fields_are_not_shared(self):
        r1 = PreScreenResult()
        r2 = PreScreenResult()
        r1.issues.append("x")
        assert r2.issues == []

    def test_to_dict_with_defaults(self):
        result = PreScreenResult()
        d = result.to_dict()
        assert d["tier_used"] == 0
        assert d["score"] == 0.0
        assert d["skip_llm_judge"] is False
        assert d["issues"] == []
        assert d["details"] == {}


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_import_quality_prescreener(self):
        from vetinari.ml.quality_prescreener import QualityPreScreener as QPS

        assert QPS is not None
        assert callable(QPS)

    def test_import_ml_registry(self):
        from vetinari.ml import MLModelRegistry as MLR

        assert MLR is not None
        assert callable(MLR)

    def test_import_get_ml_registry(self):
        from vetinari.ml import get_ml_registry as gmr

        assert callable(gmr)
