"""Base provider adapter interface for multi-LLM orchestration."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from vetinari.constants import (
    INFERENCE_STATUS_ERROR,
    INFERENCE_STATUS_OK,
    MODEL_SCORE_WEIGHT_CAPABILITY,
    MODEL_SCORE_WEIGHT_CONTEXT,
    MODEL_SCORE_WEIGHT_COST,
    MODEL_SCORE_WEIGHT_FREE_TIER,
    MODEL_SCORE_WEIGHT_LATENCY,
)
from vetinari.types import ModelProvider

logger = logging.getLogger(__name__)


# Canonical enum lives in vetinari.types.ModelProvider.
# Domain alias — adapters use ``ProviderType`` throughout.
ProviderType: TypeAlias = ModelProvider


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a provider."""

    provider_type: ProviderType
    name: str
    endpoint: str
    api_key: str | None = None
    max_retries: int = 3
    timeout_seconds: int = 120
    memory_budget_gb: int = 32
    extra_config: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ProviderConfig(name={self.name!r}, provider_type={self.provider_type!r}, endpoint={self.endpoint!r})"


@dataclass
class ModelInfo:
    """Information about a model available from a provider."""

    id: str
    name: str
    provider: str
    endpoint: str
    capabilities: list[str]
    context_len: int
    memory_gb: int
    version: str
    latency_estimate_ms: int = 1000
    throughput_tokens_per_sec: float = 50.0
    cost_per_1k_tokens: float = 0.0
    free_tier: bool = False
    tags: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ModelInfo(id={self.id!r}, provider={self.provider!r}, context_len={self.context_len!r})"


@dataclass
class InferenceRequest:
    """Request to run inference on a model.

    Fields beyond model_id and prompt are optional sampling parameters.
    Sentinel values (None, -1, 0.0) indicate "use model/profile default".
    """

    model_id: str
    prompt: str
    system_prompt: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: list[str] = field(default_factory=list)
    repeat_penalty: float = 1.1  # Penalize token repetition (1.0 = disabled, >1.0 = penalize)
    frequency_penalty: float = 0.0  # Penalize frequent tokens (0.0 = disabled)
    # -- Extended sampling parameters (Phase B, Session 11) --
    min_p: float = 0.0  # Minimum probability threshold (0.0 = disabled, 0.05 = recommended)
    presence_penalty: float = 0.0  # Penalize tokens already present (0.0 = disabled)
    mirostat_mode: int = 0  # Mirostat sampling mode (0=disabled, 1=v1, 2=v2)
    mirostat_tau: float = 5.0  # Mirostat target entropy
    mirostat_eta: float = 0.1  # Mirostat learning rate
    seed: int = -1  # RNG seed (-1 = random)
    response_format: str | None = None  # "json" for structured output
    grammar: str | None = None  # BNF grammar string for constrained generation
    task_type: str | None = None  # Task type key for automatic grammar selection
    logit_bias: dict[int, float] | None = None  # Token ID -> bias adjustments
    typical_p: float = 0.0  # Locally typical sampling (0.0 = disabled, 1.0 = default)
    tfs_z: float = 0.0  # Tail-free sampling z parameter (0.0 = disabled, 1.0 = default)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"InferenceRequest(model_id={self.model_id!r}, max_tokens={self.max_tokens!r}, stream=False)"


@dataclass
class InferenceResponse:
    """Response from inference."""

    model_id: str
    output: str
    latency_ms: int
    tokens_used: int
    status: str  # Use INFERENCE_STATUS_* constants from vetinari.constants
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce output to str to prevent callers crashing on .strip() or string ops.

        None becomes "".  Lists are joined with newlines.  Any other non-string
        type is coerced via str() so callers always receive a valid string.
        """
        if self.output is None:
            self.output = ""
        elif isinstance(self.output, list):
            self.output = "\n".join(str(item) for item in self.output)
        elif not isinstance(self.output, str):
            self.output = str(self.output)

    def __repr__(self) -> str:
        return (
            f"InferenceResponse(model_id={self.model_id!r}, status={self.status!r}, "
            f"tokens_used={self.tokens_used!r}, content={self.output[:50]!r})"
        )


class ProviderAdapter(ABC):
    """Abstract base class for all provider adapters.

    Also known as: LLM Bridge — translates between Vetinari's internal
    InferenceRequest / InferenceResponse contracts and the wire format
    expected by a specific LLM provider (llama-cpp-python, LiteLLM, NIM, etc.).

    Each concrete subclass handles one provider type and implements a
    consistent interface for:
    - Model discovery
    - Health checks
    - Inference execution
    - Capability querying
    """

    def __init__(self, config: ProviderConfig):
        """Store provider configuration and unpack frequently-accessed fields as attributes.

        Args:
            config: The provider configuration specifying endpoint, credentials, and limits.
        """
        self.config = config
        self.provider_type = config.provider_type
        self.name = config.name
        self.endpoint = config.endpoint
        self.api_key = config.api_key
        self.max_retries = config.max_retries
        self.timeout_seconds = config.timeout_seconds
        self.models: list[ModelInfo] = []

    @abstractmethod
    def discover_models(self) -> list[ModelInfo]:
        """Discover available models from the provider.

        Returns:
            List of ModelInfo objects representing available models.
        """

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Check health/status of the provider.

        Returns:
            Dict with keys: {"healthy": bool, "reason": str, "timestamp": str}
        """

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a model.

        Args:
            request: InferenceRequest with model_id, prompt, and options

        Returns:
            InferenceResponse with output, latency, tokens_used, status
        """

    def _emit_inference_started(self, request: InferenceRequest) -> None:
        """Emit an inference_started structured-log event before inference begins.

        Must be called at the top of each concrete adapter's infer() method, before
        any model API call, so that inference_started is always emitted prior to
        inference_completed (which fires via _record_telemetry after the call).
        Failures are silently suppressed so telemetry never blocks inference.

        Args:
            request: The InferenceRequest about to be submitted.
        """
        try:
            from vetinari.structured_logging import log_event as _sl_log_start

            _sl_log_start(
                "debug",
                "vetinari.adapters.base",
                "inference_started",
                model_id=request.model_id,
            )
        except Exception:  # Broad: telemetry is best-effort; never blocks inference
            logger.warning(
                "Failed to emit inference_started structured event for %s",
                request.model_id,
                exc_info=True,
            )

    @abstractmethod
    def get_capabilities(self) -> dict[str, list[str]]:
        """Get capabilities of all available models.

        Returns:
            Dict mapping model_id to list of capabilities
            (e.g., ["code_gen", "chat", "summarization"])
        """

    def score_model_for_task(self, model: ModelInfo, task_requirements: dict[str, Any]) -> float:
        """Score a model for a given task.

        Factors: capability match, context fit, latency, cost

        Args:
            model: ModelInfo to score
            task_requirements: Dict with keys like "required_capabilities", "input_tokens", "max_latency_ms"

        Returns:
            Score between 0 and 1 (higher is better)
        """
        score = 0.0

        # Capability match (35%)
        required_caps = set(task_requirements.get("required_capabilities", []))
        model_caps = set(model.capabilities)
        if required_caps:
            cap_match = len(required_caps & model_caps) / len(required_caps)
        else:
            cap_match = 1.0
        score += cap_match * MODEL_SCORE_WEIGHT_CAPABILITY

        # Context fit (20%)
        input_tokens = task_requirements.get("input_tokens", 1000)
        if input_tokens <= model.context_len:
            context_fit = 1.0
        else:
            context_fit = max(0.0, model.context_len / input_tokens)
        score += context_fit * MODEL_SCORE_WEIGHT_CONTEXT

        # Latency (20%)
        max_latency_ms = task_requirements.get("max_latency_ms", 30000)
        if model.latency_estimate_ms <= max_latency_ms:
            latency_score = 1.0
        else:
            latency_score = max(0.0, 1.0 - (model.latency_estimate_ms - max_latency_ms) / max_latency_ms)
        score += latency_score * MODEL_SCORE_WEIGHT_LATENCY

        # Cost (15%)
        max_cost = task_requirements.get("max_cost_per_1k_tokens", 0.1)
        if model.cost_per_1k_tokens <= max_cost:
            cost_score = 1.0
        else:
            cost_score = max(0.0, 1.0 - (model.cost_per_1k_tokens / max_cost))
        score += cost_score * MODEL_SCORE_WEIGHT_COST

        # Free tier bonus (10%)
        if model.free_tier:
            score += MODEL_SCORE_WEIGHT_FREE_TIER

        return float(min(1.0, score))

    def _record_telemetry(self, request: InferenceRequest, response: InferenceResponse) -> None:
        """Record inference telemetry to all analytics/learning modules.

        Called automatically after each infer() call in concrete adapters.
        Failures are silently suppressed — telemetry must never crash inference.
        """
        # Wire: OTel GenAI LLM span for inference observability
        try:
            from vetinari.observability.otel_genai import get_genai_tracer

            _genai_tracer = get_genai_tracer()
            _llm_span = _genai_tracer.start_agent_span(
                agent_name="llm",
                operation="inference",
                model=request.model_id,
            )
            _llm_span.attributes["latency_ms"] = response.latency_ms
            _llm_span.attributes["gen_ai.usage.input_tokens"] = getattr(response, "input_tokens", 0)
            _llm_span.attributes["gen_ai.usage.output_tokens"] = response.tokens_used
            _llm_span.attributes["gen_ai.response.model"] = getattr(response, "model_id", request.model_id)
            _genai_tracer.end_agent_span(
                _llm_span,
                status=INFERENCE_STATUS_OK if response.status == INFERENCE_STATUS_OK else INFERENCE_STATUS_ERROR,
                tokens_used=response.tokens_used,
            )
        except Exception:  # Broad: telemetry is best-effort; never blocks inference
            logger.warning("GenAI tracer unavailable for LLM inference span", exc_info=True)

        # Structured log: inference_completed
        try:
            from vetinari.structured_logging import log_event as _sl_log_done

            _sl_log_done(
                "info" if response.status == INFERENCE_STATUS_OK else "warning",
                "vetinari.adapters.base",
                "inference_completed",
                model_id=request.model_id,
                latency_ms=response.latency_ms,
                input_tokens=int((response.tokens_used or 0) * 0.6),
                output_tokens=int((response.tokens_used or 0) * 0.4),
                status="completed" if response.status == INFERENCE_STATUS_OK else "failed",
            )
        except Exception:  # Broad: telemetry is best-effort; never blocks inference
            logger.warning(
                "Failed to emit inference_completed structured event for %s", request.model_id, exc_info=True
            )

        try:
            from vetinari.telemetry import get_telemetry_collector

            get_telemetry_collector().record_adapter_latency(
                provider=self.provider_type.value,
                model=request.model_id,
                latency_ms=response.latency_ms,
                tokens_used=response.tokens_used,
                success=response.status == INFERENCE_STATUS_OK,
            )
        except Exception:  # Broad: telemetry is best-effort; never blocks inference
            logger.warning("Failed to record adapter telemetry for %s", request.model_id, exc_info=True)

        # --- Step 1: Cost tracking (fixed — construct CostEntry properly) ---
        try:
            from vetinari.analytics.cost import CostEntry, get_cost_tracker

            total = response.tokens_used or 0
            input_tokens = int(total * 0.6)
            output_tokens = total - input_tokens
            entry = CostEntry(
                provider=self.provider_type.value,
                model=request.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                agent=request.metadata.get("agent") if request.metadata else None,
                task_id=request.metadata.get("task_id") if request.metadata else None,
                latency_ms=float(response.latency_ms),
            )
            get_cost_tracker().record(entry)
        except Exception:  # Broad: telemetry is best-effort; never blocks inference
            logger.warning("Failed to record cost tracking entry for %s", request.model_id, exc_info=True)

        # --- Step 2: SLA tracking ---
        try:
            from vetinari.analytics.sla import get_sla_tracker

            tracker = get_sla_tracker()
            tracker.record_latency(
                f"{self.provider_type.value}:{request.model_id}",
                latency_ms=float(response.latency_ms),
                success=response.status == INFERENCE_STATUS_OK,
            )
            tracker.record_request(success=response.status == INFERENCE_STATUS_OK)
        except Exception:  # Broad: telemetry is best-effort; never blocks inference
            logger.warning("Failed to record SLA metrics for %s", request.model_id, exc_info=True)

        # --- Step 3: Forecaster ingestion ---
        try:
            from vetinari.analytics.forecasting import get_forecaster

            fc = get_forecaster()
            fc.ingest("adapter.latency", float(response.latency_ms))
            fc.ingest("adapter.tokens", float(response.tokens_used or 0))
        except Exception:  # Broad: telemetry is best-effort; never blocks inference
            logger.warning("Failed to ingest forecaster data for %s", request.model_id, exc_info=True)

        # --- Step 4: Anomaly detection ---
        try:
            from vetinari.analytics.anomaly import get_anomaly_detector

            result = get_anomaly_detector().detect("adapter.latency", float(response.latency_ms))
            if result.is_anomaly:
                logger.warning(
                    "Anomaly detected: %s=%s (%s, score=%.2f)",
                    result.metric,
                    result.value,
                    result.method,
                    result.score,
                )
        except Exception:  # Broad: telemetry is best-effort; never blocks inference
            logger.warning("Failed to run anomaly detection for %s", request.model_id, exc_info=True)

    async def async_infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference asynchronously.

        Default implementation wraps the synchronous ``infer()`` in an executor.
        Subclasses with native async support should override this method.

        Args:
            request: InferenceRequest with model_id, prompt, and options.

        Returns:
            InferenceResponse with output, latency, tokens_used, status.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.infer, request)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_type.value}, endpoint={self.endpoint})"


async def run_async_infer(
    adapter: ProviderAdapter,
    request: InferenceRequest,
) -> InferenceResponse:
    """Convenience function to run async inference on an adapter.

    Args:
        adapter: The provider adapter to use.
        request: The inference request.

    Returns:
        InferenceResponse from the adapter.
    """
    return await adapter.async_infer(request)
