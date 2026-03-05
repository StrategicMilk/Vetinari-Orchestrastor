"""
Enhanced Adapter Manager for Vetinari

Extends the basic AdapterRegistry with:
- Execution context awareness
- Provider health monitoring
- Automatic provider fallback
- Cost optimization and load balancing
- Integration with the tool execution system

This implements provider agnosticism following OpenCode's approach of
being decoupled from any single provider.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from vetinari.adapters.base import (
    ProviderAdapter,
    ProviderConfig,
    ProviderType,
    ModelInfo,
    InferenceRequest,
    InferenceResponse,
)
from vetinari.adapters.registry import AdapterRegistry
from vetinari.execution_context import (
    ExecutionMode,
    ToolPermission,
    get_context_manager,
)

logger = logging.getLogger(__name__)


class ProviderHealthStatus(Enum):
    """Health status of a provider."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProviderMetrics:
    """Metrics for a provider."""
    name: str
    provider_type: ProviderType
    last_health_check: Optional[datetime] = None
    health_status: ProviderHealthStatus = ProviderHealthStatus.UNKNOWN
    successful_inferences: int = 0
    failed_inferences: int = 0
    avg_latency_ms: float = 0.0
    total_tokens_used: int = 0
    estimated_cost: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of inferences."""
        total = self.successful_inferences + self.failed_inferences
        if total == 0:
            return 1.0
        return self.successful_inferences / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "name": self.name,
            "provider_type": self.provider_type.value,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_status": self.health_status.value,
            "successful_inferences": self.successful_inferences,
            "failed_inferences": self.failed_inferences,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens_used": self.total_tokens_used,
            "estimated_cost": self.estimated_cost,
        }


class AdapterManager:
    """
    Enhanced adapter management with context awareness and provider selection.
    """
    
    def __init__(self):
        """Initialize the adapter manager."""
        self.registry = AdapterRegistry()
        self._metrics: Dict[str, ProviderMetrics] = {}
        self._health_check_interval = timedelta(minutes=5)
        self._last_health_check: Dict[str, datetime] = {}
        self._provider_fallback_order: List[str] = []
    
    def register_provider(self, config: ProviderConfig, instance_name: str) -> ProviderAdapter:
        """
        Register a provider with metrics tracking.
        
        Args:
            config: ProviderConfig instance
            instance_name: Unique name for this provider instance
            
        Returns:
            The created ProviderAdapter
        """
        adapter = self.registry.create_adapter(config, instance_name)
        
        # Initialize metrics
        self._metrics[instance_name] = ProviderMetrics(
            name=instance_name,
            provider_type=config.provider_type,
        )
        
        # Add to fallback order
        if instance_name not in self._provider_fallback_order:
            self._provider_fallback_order.append(instance_name)
        
        logger.info(f"Registered provider: {instance_name} ({config.provider_type.value})")
        return adapter
    
    def get_provider(self, instance_name: str) -> Optional[ProviderAdapter]:
        """Get a specific provider by name."""
        return self.registry.get_adapter(instance_name)
    
    def list_providers(self) -> Dict[str, ProviderAdapter]:
        """List all registered providers."""
        return self.registry.list_adapters()
    
    def get_metrics(self, instance_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a provider or all providers.
        
        Args:
            instance_name: Optional specific provider name
            
        Returns:
            Metrics dictionary
        """
        if instance_name:
            if instance_name in self._metrics:
                return self._metrics[instance_name].to_dict()
            return {}
        
        return {name: metrics.to_dict() for name, metrics in self._metrics.items()}
    
    def health_check(self, instance_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform health check on provider(s).
        
        Args:
            instance_name: Optional specific provider name
            
        Returns:
            Health check results
        """
        if instance_name:
            adapter = self.get_provider(instance_name)
            if not adapter:
                return {}
            
            try:
                health = adapter.health_check()
                
                # Update metrics — use explicit enum members to avoid KeyError on unexpected values
                if instance_name in self._metrics:
                    if health.get("healthy"):
                        new_status = ProviderHealthStatus.HEALTHY
                    elif health.get("degraded"):
                        new_status = ProviderHealthStatus.DEGRADED
                    else:
                        new_status = ProviderHealthStatus.UNHEALTHY
                    self._metrics[instance_name].health_status = new_status
                    self._metrics[instance_name].last_health_check = datetime.now()

                self._last_health_check[instance_name] = datetime.now()
                logger.info(f"Health check {instance_name}: {health.get('healthy', False)}")
                return {instance_name: health}
            except Exception as e:
                logger.error(f"Health check failed for {instance_name}: {e}")
                if instance_name in self._metrics:
                    self._metrics[instance_name].health_status = ProviderHealthStatus.UNHEALTHY
                return {instance_name: {"healthy": False, "reason": str(e)}}
        
        # Check all providers
        results = {}
        for name in self.list_providers():
            results.update(self.health_check(name))
        return results
    
    def discover_models(self, instance_name: Optional[str] = None) -> Dict[str, List[ModelInfo]]:
        """
        Discover models from provider(s).
        
        Args:
            instance_name: Optional specific provider name
            
        Returns:
            Dictionary mapping provider names to lists of ModelInfo
        """
        if instance_name:
            adapter = self.get_provider(instance_name)
            if not adapter:
                return {}
            
            try:
                models = adapter.discover_models()
                logger.info(f"Discovered {len(models)} models from {instance_name}")
                return {instance_name: models}
            except Exception as e:
                logger.error(f"Model discovery failed for {instance_name}: {e}")
                return {instance_name: []}
        
        # Discover from all providers
        return self.registry.discover_all_models()
    
    def select_provider_for_task(
        self,
        task_requirements: Dict[str, Any],
        preferred_provider: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[ModelInfo]]:
        """
        Select best provider and model for a task.
        
        Uses metrics (success rate, latency, cost) in selection decision.
        
        Args:
            task_requirements: Dict with capability and latency requirements
            preferred_provider: Optional preferred provider name
            
        Returns:
            Tuple of (provider_name, ModelInfo)
        """
        if preferred_provider:
            adapter = self.get_provider(preferred_provider)
            if adapter:
                try:
                    # Try to find a model in preferred provider
                    models = adapter.discover_models()
                    if models:
                        # Find best model in this provider
                        best_model = None
                        best_score = -1.0
                        for model in models:
                            score = adapter.score_model_for_task(model, task_requirements)
                            if score > best_score:
                                best_score = score
                                best_model = model
                        
                        if best_model:
                            logger.info(
                                f"Selected {best_model.id} from preferred provider {preferred_provider}"
                            )
                            return preferred_provider, best_model
                except Exception as e:
                    logger.warning(f"Preferred provider {preferred_provider} failed: {e}")
        
        # Fall back to best model across all providers
        adapter, model = self.registry.find_best_model(task_requirements)
        if adapter:
            provider_name = None
            for name, prov_adapter in self.list_providers().items():
                if prov_adapter == adapter:
                    provider_name = name
                    break
            
            if provider_name and model:
                logger.info(f"Selected {model.id} from provider {provider_name}")
                return provider_name, model
        
        logger.warning("Could not select a provider for the task")
        return None, None
    
    def infer(
        self,
        request: InferenceRequest,
        provider_name: Optional[str] = None,
        fallback_on_error: bool = True,
    ) -> InferenceResponse:
        """
        Execute inference with automatic provider fallback.
        
        Args:
            request: InferenceRequest
            provider_name: Optional specific provider name
            fallback_on_error: Whether to try other providers on error
            
        Returns:
            InferenceResponse
        """
        context_manager = get_context_manager()
        
        # Check permission
        if not context_manager.check_permission(ToolPermission.MODEL_INFERENCE):
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=0,
                tokens_used=0,
                status="error",
                error="MODEL_INFERENCE permission denied in current context",
            )
        
        providers_to_try = []
        if provider_name:
            providers_to_try.append(provider_name)
        
        if fallback_on_error:
            # Add remaining providers in fallback order
            for p in self._provider_fallback_order:
                if p not in providers_to_try:
                    providers_to_try.append(p)
        
        last_error = None
        for prov_name in providers_to_try:
            adapter = self.get_provider(prov_name)
            if not adapter:
                continue
            
            try:
                logger.info(f"Attempting inference with {prov_name}")
                response = adapter.infer(request)
                
                # Update metrics
                if prov_name in self._metrics:
                    metrics = self._metrics[prov_name]
                    if response.status == "ok":
                        metrics.successful_inferences += 1
                        metrics.total_tokens_used += response.tokens_used
                    else:
                        metrics.failed_inferences += 1
                
                if response.status == "ok":
                    return response
                else:
                    last_error = response.error
                    continue
            except Exception as e:
                logger.error(f"Inference failed with {prov_name}: {e}")
                last_error = str(e)
                if not fallback_on_error:
                    break
                continue
        
        # All providers failed
        return InferenceResponse(
            model_id=request.model_id,
            output="",
            latency_ms=0,
            tokens_used=0,
            status="error",
            error=f"All providers failed. Last error: {last_error}",
        )
    
    def set_fallback_order(self, provider_names: List[str]) -> None:
        """Set the fallback order for providers."""
        self._provider_fallback_order = provider_names
        logger.info(f"Set provider fallback order: {provider_names}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all providers."""
        return {
            "providers": {
                name: {
                    "adapter": str(adapter),
                    "metrics": self._metrics.get(name, {}).to_dict() if name in self._metrics else {},
                }
                for name, adapter in self.list_providers().items()
            },
            "fallback_order": self._provider_fallback_order,
        }


# Global adapter manager instance
_adapter_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get or create the global adapter manager."""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager
