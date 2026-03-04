"""Provider adapter registry and factory for managing multiple LLM providers."""

import logging
import threading
from typing import Dict, Optional, Type, List, Tuple, Any
from .base import (
    ProviderAdapter, ProviderConfig, ProviderType, ModelInfo,
    InferenceRequest, InferenceResponse
)
from .lmstudio_adapter import LMStudioProviderAdapter
from .openai_adapter import OpenAIProviderAdapter
from .cohere_adapter import CohereProviderAdapter
from .anthropic_adapter import AnthropicProviderAdapter
from .gemini_adapter import GeminiProviderAdapter

logger = logging.getLogger(__name__)

# Thread lock for class-level state
_registry_lock = threading.Lock()


class AdapterRegistry:
    """
    Registry for managing all provider adapters.

    Supports:
    - Registration of adapter classes (class-level, shared)
    - Factory creation of adapter instances (class-level cache)
    - Discovery of available adapters
    - Model discovery across all providers

    Note: _adapter_classes and _instances are class-level dicts shared
    across all instances, providing a global registry. Tests can reset
    state via AdapterRegistry.clear_instances().
    """

    # Class-level shared state (thread-safe via _registry_lock)
    _adapter_classes: Dict[ProviderType, Type[ProviderAdapter]] = {
        ProviderType.LM_STUDIO: LMStudioProviderAdapter,
        ProviderType.OPENAI: OpenAIProviderAdapter,
        ProviderType.COHERE: CohereProviderAdapter,
        ProviderType.ANTHROPIC: AnthropicProviderAdapter,
        ProviderType.GEMINI: GeminiProviderAdapter,
    }

    _instances: Dict[str, ProviderAdapter] = {}

    @classmethod
    def register_adapter(cls, provider_type: ProviderType, adapter_class: Type[ProviderAdapter]) -> None:
        """Register a new adapter class for a provider type."""
        with _registry_lock:
            cls._adapter_classes[provider_type] = adapter_class
        logger.info(f"[AdapterRegistry] Registered {adapter_class.__name__} for {provider_type.value}")

    @classmethod
    def create_adapter(cls, config: ProviderConfig, instance_name: Optional[str] = None) -> ProviderAdapter:
        """
        Create an adapter instance from configuration.

        Args:
            config: ProviderConfig with provider_type, endpoint, api_key, etc.
            instance_name: Optional name for the instance (for caching)

        Returns:
            Instance of appropriate ProviderAdapter subclass

        Raises:
            ValueError: If provider_type is not registered
        """
        with _registry_lock:
            if config.provider_type not in cls._adapter_classes:
                raise ValueError(f"Unknown provider type: {config.provider_type}")
            adapter_class = cls._adapter_classes[config.provider_type]

        instance = adapter_class(config)

        if instance_name:
            with _registry_lock:
                cls._instances[instance_name] = instance
            logger.info(f"[AdapterRegistry] Created adapter instance '{instance_name}' ({adapter_class.__name__})")

        return instance

    @classmethod
    def get_adapter(cls, instance_name: str) -> Optional[ProviderAdapter]:
        """Get a cached adapter instance by name."""
        return cls._instances.get(instance_name)

    @classmethod
    def list_adapters(cls) -> Dict[str, ProviderAdapter]:
        """Get all cached adapter instances."""
        with _registry_lock:
            return dict(cls._instances)

    @classmethod
    def list_supported_providers(cls) -> List[ProviderType]:
        """Get list of all supported provider types."""
        return list(cls._adapter_classes.keys())

    @classmethod
    def discover_all_models(cls) -> Dict[str, List[ModelInfo]]:
        """Discover models from all active adapter instances."""
        results = {}
        with _registry_lock:
            instances = dict(cls._instances)
        for name, adapter in instances.items():
            try:
                models = adapter.discover_models()
                results[name] = models
                logger.info(f"[AdapterRegistry] {name}: discovered {len(models)} models")
            except Exception as e:
                logger.error(f"[AdapterRegistry] {name}: discovery failed: {e}")
                results[name] = []
        return results

    @classmethod
    def health_check_all(cls) -> Dict[str, Dict]:
        """Run health check on all active adapter instances."""
        results = {}
        with _registry_lock:
            instances = dict(cls._instances)
        for name, adapter in instances.items():
            try:
                health = adapter.health_check()
                results[name] = health
                status = "healthy" if health.get("healthy") else "unhealthy"
                logger.info(f"[AdapterRegistry] {name}: {status}")
            except Exception as e:
                logger.error(f"[AdapterRegistry] {name}: health check failed: {e}")
                results[name] = {"healthy": False, "reason": str(e), "timestamp": None}
        return results

    @classmethod
    def find_best_model(cls, task_requirements: Dict) -> Tuple[Optional[ProviderAdapter], Optional[ModelInfo]]:
        """Find the best model across all adapters for a given task."""
        best_adapter = None
        best_model = None
        best_score = -1.0
        with _registry_lock:
            instances = dict(cls._instances)
        for name, adapter in instances.items():
            for model in getattr(adapter, "models", []):
                score = adapter.score_model_for_task(model, task_requirements)
                if score > best_score:
                    best_score = score
                    best_adapter = adapter
                    best_model = model
                    logger.debug(f"[AdapterRegistry] New best: {model.id} ({name}) score={score:.2f}")
        return best_adapter, best_model

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached adapter instances."""
        with _registry_lock:
            cls._instances.clear()
        logger.info("[AdapterRegistry] Cleared all adapter instances")

    @classmethod
    def __repr__(cls) -> str:
        providers = ", ".join([p.value for p in cls._adapter_classes.keys()])
        instances = ", ".join(cls._instances.keys())
        return f"AdapterRegistry(providers=[{providers}], instances=[{instances}])"
