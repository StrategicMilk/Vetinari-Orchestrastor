"""Enhanced Adapter Manager for Vetinari.

Extends the basic AdapterRegistry with:
- Execution context awareness
- Provider health monitoring
- Automatic provider fallback
- Cost optimization and load balancing
- Integration with the tool execution system

This implements provider agnosticism following OpenCode's approach of
being decoupled from any single provider.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, cast

from vetinari.adapters.base import (
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    ProviderAdapter,
    ProviderConfig,
    ProviderType,
)
from vetinari.adapters.registry import AdapterRegistry
from vetinari.backend_config import load_backend_runtime_config, resolve_provider_fallback_order
from vetinari.constants import INFERENCE_STATUS_ERROR, INFERENCE_STATUS_OK
from vetinari.exceptions import ConfigurationError
from vetinari.execution_context import (
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
    last_health_check: datetime | None = None
    health_status: ProviderHealthStatus = ProviderHealthStatus.UNKNOWN
    successful_inferences: int = 0
    failed_inferences: int = 0
    avg_latency_ms: float = 0.0
    total_tokens_used: int = 0
    estimated_cost: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ProviderMetrics(name={self.name!r}, total_requests={self.successful_inferences + self.failed_inferences!r}, "
            f"success_rate={self.success_rate:.2f})"
        )

    @property
    def success_rate(self) -> float:
        """Calculate success rate of inferences."""
        total = self.successful_inferences + self.failed_inferences
        if total == 0:
            return 1.0
        return self.successful_inferences / total

    def to_dict(self) -> dict[str, Any]:
        """Serialize provider metrics to a plain dict for JSON serialization and dashboard display."""
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
    """Enhanced adapter management with context awareness and provider selection.

    Also known as: LLM Dispatcher — picks which LLM provider to use for a
    given request and sends the request through that provider's ProviderAdapter
    (LLM Bridge).  Adds health monitoring, automatic fallback, cost
    optimisation, load balancing, and optional cascade routing on top of the
    raw AdapterRegistry.
    """

    def __init__(self):
        self.registry = AdapterRegistry()
        self._metrics: dict[str, ProviderMetrics] = {}
        self._metrics_lock = threading.Lock()  # Guards all metric counter updates
        self._health_check_interval = timedelta(minutes=5)
        self._last_health_check: dict[str, datetime] = {}
        self._provider_fallback_order: list[str] = []

        # Cascade routing — disabled by default; enabled via enable_cascade_routing()
        self._cascade_enabled: bool = False
        self._cascade_provider: str | None = None  # provider to run cascade through
        self._cascade_router: Any = None  # set by enable_cascade_routing()

    def register_provider(self, config: ProviderConfig, instance_name: str) -> ProviderAdapter:
        """Register a provider with metrics tracking.

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

        logger.info("Registered provider: %s (%s)", instance_name, config.provider_type.value)
        return adapter

    def get_provider(self, instance_name: str) -> ProviderAdapter | None:
        """Get a specific provider by name."""
        return self.registry.get_adapter(instance_name)

    def list_providers(self) -> dict[str, ProviderAdapter]:
        """List all registered providers."""
        return self.registry.list_adapters()

    def get_metrics(self, instance_name: str | None = None) -> dict[str, Any]:
        """Get metrics for a provider or all providers.

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

    def health_check(self, instance_name: str | None = None) -> dict[str, Any]:
        """Perform health check on provider(s).

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

                # Update metrics
                if instance_name in self._metrics:
                    status_str = "healthy" if health.get("healthy") else "unhealthy"
                    self._metrics[instance_name].health_status = ProviderHealthStatus[status_str.upper()]
                    self._metrics[instance_name].last_health_check = datetime.now(timezone.utc)

                self._last_health_check[instance_name] = datetime.now(timezone.utc)
                logger.info("Health check %s: %s", instance_name, status_str)
                return {instance_name: health}
            except Exception:
                logger.exception("Health check failed for %s — marking unhealthy", instance_name)
                if instance_name in self._metrics:
                    self._metrics[instance_name].health_status = ProviderHealthStatus.UNHEALTHY
                    self._metrics[instance_name].last_health_check = datetime.now(timezone.utc)
                return {instance_name: {"healthy": False, "reason": "Health check failed"}}

        # Check all providers
        results = {}
        for name in self.list_providers():
            results.update(self.health_check(name))
        return results

    def discover_models(self, instance_name: str | None = None) -> dict[str, list[ModelInfo]]:
        """Discover models from provider(s).

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
                logger.info("Discovered %s models from %s", len(models), instance_name)
                return {instance_name: models}
            except Exception as e:
                logger.error("Model discovery failed for %s: %s", instance_name, e)
                return {instance_name: []}

        # Discover from all providers
        result = self.registry.discover_all_models()
        total_models = sum(len(models) for models in result.values())
        if total_models == 0:
            logger.warning(
                "No models discovered by any provider. LLM inference will not be available. "
                "Configure a model provider or place GGUF files in the models directory."
            )
        return result

    def select_provider_for_task(
        self,
        task_requirements: dict[str, Any],
        preferred_provider: str | None = None,
    ) -> tuple[str | None, ModelInfo | None]:
        """Select best provider and model for a task.

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
                            logger.info("Selected %s from preferred provider %s", best_model.id, preferred_provider)
                            return preferred_provider, best_model
                except Exception as e:
                    logger.warning("Preferred provider %s failed: %s", preferred_provider, e)

        # Fall back to best model across all providers
        best_adapter, best_model = self.registry.find_best_model(task_requirements)
        if best_adapter:
            provider_name = None
            for name, prov_adapter in self.list_providers().items():
                if prov_adapter == best_adapter:
                    provider_name = name
                    break

            if provider_name and best_model:
                logger.info("Selected %s from provider %s", best_model.id, provider_name)
                return provider_name, best_model

        logger.warning("Could not select a provider for the task")
        return None, None

    def infer(
        self,
        request: InferenceRequest,
        provider_name: str | None = None,
        fallback_on_error: bool = True,
        use_cascade: bool = False,
    ) -> InferenceResponse:
        """Execute inference with automatic provider fallback.

        When cascade routing is enabled (via ``enable_cascade_routing()``) and
        ``use_cascade`` is ``True`` (or the global cascade flag is set), the
        request is routed through ``CascadeRouter``: the cheapest tier is tried
        first and escalation happens automatically when confidence is low.

        Args:
            request: InferenceRequest describing the prompt and model.
            provider_name: Optional specific provider instance name.  When
                cascade mode is active this selects which provider executes
                each tier call.
            fallback_on_error: Whether to try other providers on adapter error.
            use_cascade: If ``True``, force cascade routing for this call even
                if the global cascade flag is ``False``.  Ignored when no
                cascade router has been configured via
                ``enable_cascade_routing()``.

        Returns:
            InferenceResponse with the chosen model's output.
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

        # ── Cascade routing path ───────────────────────────────────────────
        _run_cascade = (use_cascade or self._cascade_enabled) and self._cascade_router is not None
        if _run_cascade:
            return self._infer_via_cascade(request, provider_name or self._cascade_provider)

        # ── Standard provider-fallback path ───────────────────────────────
        providers_to_try = []
        if provider_name:
            providers_to_try.append(provider_name)

        if fallback_on_error:
            # Add remaining providers in fallback order
            for p in self._provider_fallback_order:
                if p not in providers_to_try:
                    providers_to_try.append(p)

        # VRAM pre-check: warn if the requested model won't fit without
        # evicting leased (in-use) models.  This is advisory — the adapter's
        # model cache handles actual load/evict.  Decision: ADR-0087.
        if request.model_id:
            try:
                from vetinari.models.vram_manager import get_vram_manager

                _vmgr = get_vram_manager()
                if not _vmgr.can_load(request.model_id):
                    _max_avail = _vmgr.get_max_available_vram_gb()
                    _needed = _vmgr.get_model_vram_requirement(request.model_id)
                    logger.warning(
                        "Model %s needs %.1f GB but only %.1f GB available "
                        "(including evictable) — inference may stall or fail",
                        request.model_id,
                        _needed,
                        _max_avail,
                    )
            except Exception:
                logger.warning("VRAMManager unavailable for pre-check — proceeding without VRAM guard")

        last_error = None
        for prov_name in providers_to_try:
            adapter = self.get_provider(prov_name)
            if not adapter:
                continue

            try:
                logger.info("Attempting inference with %s", prov_name)
                response = adapter.infer(request)

                # Update metrics (lock guards concurrent counter increments)
                if prov_name in self._metrics:
                    metrics = self._metrics[prov_name]
                    with self._metrics_lock:
                        if response.status == INFERENCE_STATUS_OK:
                            n = metrics.successful_inferences  # count before increment
                            metrics.successful_inferences += 1
                            metrics.total_tokens_used += response.tokens_used
                            # Running average — only update when latency is a real
                            # measurement (0 ms is a sentinel for "not measured").
                            if response.latency_ms > 0:
                                metrics.avg_latency_ms = (metrics.avg_latency_ms * n + response.latency_ms) / (n + 1)
                            # Accumulate cost if the adapter supplied it in metadata
                            metrics.estimated_cost += response.metadata.get("cost", 0.0)
                        else:
                            metrics.failed_inferences += 1

                if response.status == INFERENCE_STATUS_OK:
                    return response
                last_error = response.error
                continue
            except Exception:
                logger.exception(
                    "Inference failed with %s — incrementing failure counter, %s",
                    prov_name,
                    "stopping fallback" if not fallback_on_error else "trying next provider",
                )
                if prov_name in self._metrics:
                    with self._metrics_lock:
                        self._metrics[prov_name].failed_inferences += 1
                last_error = "Provider inference failed"
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

    def _infer_via_cascade(
        self,
        request: InferenceRequest,
        provider_name: str | None,
    ) -> InferenceResponse:
        """Run inference through the configured CascadeRouter.

        Builds an adapter function that dispatches each tier's request to the
        named provider (or the first available provider if none is specified),
        then delegates to ``CascadeRouter.route()``.

        Args:
            request: The original InferenceRequest (model_id will be
                overridden per tier by CascadeRouter).
            provider_name: Provider instance name to use for all tier calls.
                Falls back to the first registered provider when ``None``.

        Returns:
            InferenceResponse assembled from the winning cascade tier.
        """
        # Resolve which provider handles the actual adapter calls
        prov_name = provider_name
        if not prov_name and self._provider_fallback_order:
            prov_name = self._provider_fallback_order[0]

        adapter = self.get_provider(prov_name) if prov_name else None

        if adapter is None:
            logger.warning(
                "CascadeRouter: no provider available (name=%s), falling back to error response",
                prov_name,
            )
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=0,
                tokens_used=0,
                status="error",
                error="Cascade routing: no provider configured",
            )

        def _adapter_fn(tier_request: InferenceRequest) -> InferenceResponse:
            """Dispatch a single tier request through the resolved provider."""
            return adapter.infer(tier_request)

        try:
            cascade_result = self._cascade_router.route(request, adapter_fn=_adapter_fn)
        except Exception as exc:
            logger.error("CascadeRouter.route() failed: %s", exc)
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=0,
                tokens_used=0,
                status="error",
                error=f"Cascade routing failed: {exc}",
            )

        logger.info(
            "Cascade routing complete: model=%s, confidence=%.3f, escalations=%d, tiers=%s",
            cascade_result.model_id,
            cascade_result.confidence,
            cascade_result.escalation_count,
            cascade_result.tiers_tried,
        )

        # Update metrics for the winning provider (lock guards concurrent counter increments)
        if prov_name and prov_name in self._metrics:
            metrics = self._metrics[prov_name]
            response = cascade_result.response
            with self._metrics_lock:
                if getattr(response, "status", INFERENCE_STATUS_ERROR) == INFERENCE_STATUS_OK:
                    metrics.successful_inferences += 1
                    metrics.total_tokens_used += getattr(response, "tokens_used", 0)
                else:
                    metrics.failed_inferences += 1

        return cast(InferenceResponse, cascade_result.response)

    def enable_cascade_routing(
        self,
        tiers: list[dict[str, float | str]],
        provider_name: str | None = None,
        confidence_threshold: float = 0.7,
        max_escalations: int = 2,
    ) -> None:
        """Configure cascade cost-optimisation routing for inference.

        When enabled, ``infer()`` will try the cheapest model tier first and
        escalate to more capable (expensive) tiers only when response
        confidence is below the threshold.  This is opt-in — the default
        routing behaviour is unchanged.

        Args:
            tiers: Ordered list of tier descriptors.  Each dict must have a
                ``model_id`` key (str) and may include ``cost_per_1k_tokens``
                (float, default 0.0).  Tiers are tried cheapest-first, so
                sort by ascending cost before passing.
            provider_name: Provider instance name to route cascade requests
                through.  When ``None``, the first registered provider is used.
            confidence_threshold: Minimum confidence score to accept a
                response without escalating.  Range [0, 1], default 0.7.
            max_escalations: Maximum number of escalation steps after the
                first tier attempt.  Default 2.

        Raises:
            ValueError: If ``tiers`` is empty or a tier dict lacks
                ``model_id``.
        """
        if not tiers:
            raise ConfigurationError("enable_cascade_routing: tiers must not be empty")

        from vetinari.cascade_router import CascadeRouter

        cascade = CascadeRouter(
            confidence_threshold=confidence_threshold,
            max_escalations=max_escalations,
        )
        for i, tier in enumerate(tiers):
            model_id = tier.get("model_id")
            if not model_id:
                raise ConfigurationError(
                    f"enable_cascade_routing: each tier dict must contain 'model_id', got {tier!r}",
                )
            cost = float(tier.get("cost_per_1k_tokens", 0.0))
            cascade.add_tier(str(model_id), cost_per_1k_tokens=cost, priority=i)

        self._cascade_router = cascade
        self._cascade_enabled = True
        self._cascade_provider = provider_name
        logger.info(
            "Cascade routing enabled: %d tiers, threshold=%.2f, max_escalations=%d",
            len(tiers),
            confidence_threshold,
            max_escalations,
        )

    def disable_cascade_routing(self) -> None:
        """Disable cascade routing and restore default provider-fallback routing."""
        self._cascade_enabled = False
        logger.info("Cascade routing disabled")

    def get_cascade_stats(self) -> dict[str, Any]:
        """Return cascade routing statistics, or empty dict if not enabled.

        Returns:
            Stats dict from CascadeRouter, or empty dict when cascade is off.
        """
        if self._cascade_enabled and self._cascade_router is not None:
            return cast(dict[str, Any], self._cascade_router.get_stats())
        return {}

    def set_fallback_order(self, provider_names: list[str]) -> None:
        """Override the provider fallback sequence used when the preferred provider is unavailable.

        Args:
            provider_names: Ordered list of provider names to try in sequence.
        """
        self._provider_fallback_order = provider_names
        logger.info("Set provider fallback order: %s", provider_names)

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status of all providers."""
        return {
            "providers": {
                name: {
                    "adapter": str(adapter),
                    "metrics": self._metrics[name].to_dict() if name in self._metrics else {},
                }
                for name, adapter in self.list_providers().items()
            },
            "fallback_order": self._provider_fallback_order,
        }


# Global adapter manager instance
_adapter_manager: AdapterManager | None = None
_adapter_manager_lock = threading.Lock()


def get_adapter_manager() -> AdapterManager:
    """Get or create the global adapter manager.

    Auto-registers the local llama-cpp-python provider on first access
    so that inference works out of the box when GGUF models are available.

    Returns:
        The AdapterManager singleton.
    """
    global _adapter_manager
    if _adapter_manager is None:
        with _adapter_manager_lock:
            if _adapter_manager is None:
                _adapter_manager = AdapterManager()

                # Auto-register the configured inference providers so runtime
                # fallback order matches the backend transition plan.
                # Skip during test runs to avoid loading real models in unit tests.
                import os as _os
                import sys

                _in_test = "pytest" in sys.modules or _os.environ.get("VETINARI_TESTING")
                if not _in_test:
                    try:
                        from vetinari.adapters.base import ProviderConfig, ProviderType

                        _runtime_cfg = load_backend_runtime_config()
                        _li = _runtime_cfg.get("local_inference", {})
                        _ib = _runtime_cfg.get("inference_backend", {})
                        _extra = {
                            "models_dir": str(_li.get("models_dir", _os.environ.get("VETINARI_MODELS_DIR", ""))),
                            "gpu_layers": str(_os.environ.get("VETINARI_GPU_LAYERS", _li.get("gpu_layers", -1))),
                            "context_length": str(
                                _os.environ.get("VETINARI_CONTEXT_LENGTH", _li.get("context_length", 8192))
                            ),
                            "ram_budget_gb": str(_li.get("ram_budget_gb", 30)),
                            "cpu_offload_enabled": str(_li.get("cpu_offload_enabled", True)),
                        }
                        _hw = _runtime_cfg.get("hardware", {})
                        local_config = ProviderConfig(
                            provider_type=ProviderType.LOCAL,
                            name="local",
                            endpoint="",
                            memory_budget_gb=_hw.get("gpu_vram_gb", 32) if _hw else 32,
                            extra_config=_extra,
                        )
                        _adapter_manager.register_provider(local_config, "local")
                        logger.info("Auto-registered local inference provider")

                        _vllm_cfg = _ib.get("vllm", {})
                        if _vllm_cfg.get("enabled", False) and _vllm_cfg.get("endpoint"):
                            try:
                                _vllm_extra = {
                                    key: value for key, value in _vllm_cfg.items() if key not in {"enabled", "endpoint"}
                                }
                                vllm_config = ProviderConfig(
                                    provider_type=ProviderType.VLLM,
                                    name="vllm",
                                    endpoint=_vllm_cfg["endpoint"],
                                    extra_config=_vllm_extra,
                                )
                                _adapter_manager.register_provider(vllm_config, "vllm")
                                logger.info("Auto-registered vLLM provider at %s", _vllm_cfg["endpoint"])
                            except Exception as vllm_exc:
                                logger.warning("Failed to auto-register vLLM provider: %s", vllm_exc)

                        _nim_cfg = _ib.get("nim", {})
                        if _nim_cfg.get("enabled", False) and _nim_cfg.get("endpoint"):
                            try:
                                _nim_extra = {
                                    key: value for key, value in _nim_cfg.items() if key not in {"enabled", "endpoint"}
                                }
                                nim_config = ProviderConfig(
                                    provider_type=ProviderType.NIM,
                                    name="nim",
                                    endpoint=_nim_cfg["endpoint"],
                                    extra_config=_nim_extra,
                                )
                                _adapter_manager.register_provider(nim_config, "nim")
                                logger.info("Auto-registered NIM provider at %s", _nim_cfg["endpoint"])
                            except Exception as nim_exc:
                                logger.warning("Failed to auto-register NIM provider: %s", nim_exc)

                        _available = set(_adapter_manager.list_providers())
                        _order = resolve_provider_fallback_order(_runtime_cfg, available_providers=_available)
                        if _order:
                            _adapter_manager.set_fallback_order(_order)
                            logger.info("Configured backend fallback order: %s", _order)

                    except Exception as exc:
                        logger.warning("Failed to auto-register local provider: %s", exc)

    return _adapter_manager
