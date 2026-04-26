"""Lightweight ML Model Infrastructure — vetinari.ml.

Small, fast ML models for classification, regression, and embedding tasks
that replace expensive LLM calls. NOT for LLM inference — just for
lightweight classifiers, regressors, and embedders.

Submodules:
  - quality_prescreener: Three-tier quality pre-screening
  - classifiers: Goal, defect, and ambiguity classifiers (US-212)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass  # noqa: VET123 - barrel export preserves public import compatibility
from pathlib import Path  # noqa: VET123 - barrel export preserves public import compatibility
from typing import Any  # noqa: VET123 - barrel export preserves public import compatibility

from vetinari.exceptions import ConfigurationError, ModelNotFoundError
from vetinari.ml.classifiers import (
    AmbiguityDetector,
    AmbiguityResult,
    DefectClassification,
    DefectClassifier,
    GoalClassification,
    GoalClassifier,
)
from vetinari.ml.task_classifier import TaskClassifier

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata about a loaded ML model.

    Args:
        name: Model identifier.
        model_type: Type of model (sklearn, onnx, sentence_transformer).
        loaded: Whether the model is currently loaded.
    """

    name: str = ""
    model_type: str = ""
    loaded: bool = False


class MLModelRegistry:
    """Registry for lightweight ML models (classifiers, regressors, embedders).

    Manages loading, caching, and inference for small ML models used to
    replace LLM calls for non-generative tasks.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._model_info: dict[str, ModelInfo] = {}
        self._lock = threading.Lock()

    def load(self, name: str, model_path: Path) -> Any:
        """Load a small ML model (sklearn joblib, ONNX, or sentence-transformers).

        Args:
            name: Identifier for the model.
            model_path: Path to the model file or directory.

        Returns:
            The loaded model object.

        Raises:
            ModelNotFoundError: If the model path doesn't exist.
            ValueError: If the model format is not supported.
        """
        with self._lock:
            if name in self._models:
                return self._models[name]

            if not model_path.exists():
                raise ModelNotFoundError(f"Model not found: {model_path}")

            model = None
            model_type = "unknown"

            if model_path.suffix == ".onnx":
                try:
                    import onnxruntime as ort

                    model = ort.InferenceSession(str(model_path))
                    model_type = "onnx"
                except ImportError as exc:
                    raise ConfigurationError("onnxruntime not installed — cannot load ONNX model") from exc

            elif model_path.suffix == ".joblib":
                try:
                    import joblib

                    model = joblib.load(model_path)  # noqa: VET302 — Trusted: loaded from vetinari/ml/models/, not user-provided
                    model_type = "sklearn"
                except ImportError as exc:
                    raise ConfigurationError("joblib not installed — cannot load sklearn model") from exc

            elif model_path.is_dir():
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(str(model_path))
                    model_type = "sentence_transformer"
                except ImportError as exc:
                    raise ConfigurationError(
                        "sentence-transformers not installed — cannot load embedding model"
                    ) from exc
            else:
                raise ConfigurationError(f"Unsupported model format: {model_path.suffix}")

            self._models[name] = model
            self._model_info[name] = ModelInfo(name=name, model_type=model_type, loaded=True)
            logger.info("Loaded ML model %s (%s) from %s", name, model_type, model_path)
            return model

    def predict(self, name: str, inputs: Any) -> Any:
        """Run inference on a loaded model.

        Args:
            name: Model identifier.
            inputs: Model-specific input data.

        Returns:
            Model predictions.

        Raises:
            KeyError: If the model is not loaded.
        """
        with self._lock:
            model = self._models.get(name)

        if model is None:
            raise KeyError(f"Model '{name}' not loaded. Call load() first.")

        if hasattr(model, "predict"):
            return model.predict(inputs)
        if hasattr(model, "encode"):
            return model.encode(inputs)
        raise TypeError(f"Model '{name}' has no predict() or encode() method")

    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded.

        Args:
            name: Model identifier.

        Returns:
            True if the model is loaded.
        """
        with self._lock:
            return name in self._models

    def unload(self, name: str) -> None:
        """Unload a model to free memory.

        Args:
            name: Model identifier.
        """
        with self._lock:
            self._models.pop(name, None)
            info = self._model_info.get(name)
            if info:
                info.loaded = False

    def get_status(self) -> dict[str, Any]:
        """Return registry status.

        Returns:
            Dictionary with loaded model information.
        """
        with self._lock:
            return {
                "loaded_models": list(self._models.keys()),
                "model_count": len(self._models),
                "models": {
                    name: {"type": info.model_type, "loaded": info.loaded} for name, info in self._model_info.items()
                },
            }


# Singleton
_ml_registry: MLModelRegistry | None = None
_ml_lock = threading.Lock()


def get_ml_registry() -> MLModelRegistry:
    """Return the singleton MLModelRegistry instance.

    Returns:
        The shared MLModelRegistry instance.
    """
    global _ml_registry
    if _ml_registry is None:
        with _ml_lock:
            if _ml_registry is None:
                _ml_registry = MLModelRegistry()
    return _ml_registry


def reset_ml_registry() -> None:
    """Reset singleton for testing."""
    global _ml_registry
    with _ml_lock:
        _ml_registry = None


__all__ = [
    "AmbiguityDetector",
    "AmbiguityResult",
    "DefectClassification",
    "DefectClassifier",
    "GoalClassification",
    "GoalClassifier",
    "MLModelRegistry",
    "ModelInfo",
    "TaskClassifier",
    "get_ml_registry",
    "reset_ml_registry",
]
