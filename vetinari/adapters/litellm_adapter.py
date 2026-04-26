"""LiteLLM provider adapter — unified multi-provider completion API.

Wraps the LiteLLM SDK to provide a single adapter that can route to any
supported provider (OpenAI, Anthropic, Gemini, Cohere, local OpenAI-compat
servers, etc.) via ``litellm.completion()``.

Decision: Adopt LiteLLM as default abstraction layer (ADR-0062, supersedes ADR-0019)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from vetinari.constants import API_HEALTH_CHECK_TIMEOUT, INFERENCE_STATUS_OK

from .base import (
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    ProviderAdapter,
    ProviderConfig,
    ProviderType,
)

logger = logging.getLogger(__name__)

# LiteLLM provider prefix mapping — litellm uses "provider/model" format
_PROVIDER_PREFIXES: dict[ProviderType, str] = {
    ProviderType.OPENAI: "openai",
    ProviderType.ANTHROPIC: "anthropic",
    ProviderType.GEMINI: "gemini",
    ProviderType.COHERE: "cohere",
    ProviderType.HUGGINGFACE: "huggingface",
    ProviderType.LOCAL: "openai",  # local OpenAI-compatible servers
}


def _litellm() -> Any:
    """Lazy import litellm to avoid import-time cost when not used."""
    try:
        import litellm

        return litellm
    except ImportError as exc:
        raise ImportError(
            "litellm is required for LiteLLMAdapter. Install with: pip install 'vetinari[cloud]'",  # noqa: VET301 — user guidance string
        ) from exc


class LiteLLMAdapter(ProviderAdapter):
    """Unified adapter wrapping LiteLLM's completion API.

    Supports any provider that LiteLLM supports (100+ models) through a single
    adapter instance.  Falls back to the ``openai/`` prefix for local
    OpenAI-compatible servers (LM Studio, vLLM, etc.).

    Args:
        config: ProviderConfig — the ``endpoint`` field is used as ``api_base``
                for local/custom providers.  ``api_key`` is passed through.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Resolve the LiteLLM provider prefix and API base URL from the provider configuration.

        Args:
            config: Provider configuration with endpoint, api_key, etc.
        """
        super().__init__(config)
        self._prefix = _PROVIDER_PREFIXES.get(config.provider_type, "openai")
        self._api_base = config.endpoint if config.provider_type == ProviderType.LOCAL else None

    def _model_id(self, model: str) -> str:
        """Build LiteLLM-style model identifier.

        Args:
            model: Raw model name (e.g. ``gpt-4o``, ``claude-3-sonnet``).

        Returns:
            LiteLLM-prefixed model string (e.g. ``openai/gpt-4o``).
        """
        if "/" in model:
            return model  # already prefixed
        return f"{self._prefix}/{model}"

    # ── Discovery ──────────────────────────────────────────────────────

    def discover_models(self) -> list[ModelInfo]:
        """Discover models available through this adapter's provider.

        Uses LiteLLM's model info database for cloud providers.  For local
        providers, queries the OpenAI-compatible ``/v1/models`` endpoint.

        Returns:
            List of ModelInfo objects for available models.
        """
        litellm = _litellm()
        discovered: list[ModelInfo] = []

        try:
            if self._api_base:
                # Local provider — query /v1/models endpoint
                discovered = self._discover_local_models()
            else:
                # Cloud provider — use LiteLLM's model cost map
                model_map = getattr(litellm, "model_cost", {})
                for model_key, info in model_map.items():
                    if not model_key.startswith(self._prefix):
                        continue
                    short_name = model_key.split("/", 1)[-1] if "/" in model_key else model_key
                    discovered.append(
                        ModelInfo(
                            id=short_name,
                            name=short_name,
                            provider=self._prefix,
                            endpoint=self.endpoint,
                            capabilities=["chat"],
                            context_len=info.get("max_tokens", 8192),
                            memory_gb=0,
                            version="latest",
                            cost_per_1k_tokens=info.get("input_cost_per_token", 0) * 1000,
                            tags=["litellm", self._prefix],
                        ),
                    )
        except (ImportError, OSError, ValueError):
            logger.exception("[LiteLLM] Model discovery failed for %s", self._prefix)

        self.models = discovered
        logger.info("[LiteLLM/%s] Discovered %d models", self._prefix, len(discovered))
        return discovered

    def _discover_local_models(self) -> list[ModelInfo]:
        """Query a local OpenAI-compatible endpoint for available models.

        Returns:
            List of ModelInfo objects from the local server.
        """
        from vetinari.http import create_session

        try:
            with create_session() as session:
                resp = session.get(
                    f"{self._api_base}/v1/models",
                    timeout=API_HEALTH_CHECK_TIMEOUT,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                )
            resp.raise_for_status()
            data = resp.json()
            return [
                ModelInfo(
                    id=m["id"],
                    name=m.get("id", "unknown"),
                    provider="local",
                    endpoint=self._api_base or "",
                    capabilities=["chat", "code_gen"],
                    context_len=m.get("context_length", 8192),
                    memory_gb=0,
                    version="local",
                    tags=["litellm", "local"],
                )
                for m in data.get("data", [])
            ]
        except (ImportError, OSError, ValueError):
            logger.warning("[LiteLLM] Local model discovery failed at %s", self._api_base)
            return []

    # ── Health Check ───────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """Check provider health via a lightweight model list call.

        Returns:
            Dict with ``healthy``, ``reason``, ``timestamp`` keys.
        """
        try:
            if self._api_base:
                from vetinari.http import create_session

                with create_session() as session:
                    resp = session.get(f"{self._api_base}/v1/models", timeout=API_HEALTH_CHECK_TIMEOUT)
                healthy = resp.status_code == 200
            else:
                # Cloud: just verify litellm loads and has model info
                litellm = _litellm()
                healthy = bool(getattr(litellm, "model_cost", {}))

            return {
                "healthy": healthy,
                "reason": f"LiteLLM/{self._prefix} OK" if healthy else "Unhealthy",
                "timestamp": time.time(),
                "endpoint": self._api_base or self._prefix,
            }
        except Exception:
            logger.exception("[LiteLLM] Health check failed for %s", self._api_base or self._prefix)
            return {
                "healthy": False,
                "reason": "Health check failed",
                "timestamp": time.time(),
                "endpoint": self._api_base or self._prefix,
            }

    # ── Inference ──────────────────────────────────────────────────────

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run completion via LiteLLM SDK.

        Builds a standard chat-completion payload and delegates to
        ``litellm.completion()``.  Streaming is not used here; see
        ``stream_infer()`` for SSE-style streaming.

        Args:
            request: InferenceRequest with model_id, prompt, and options.

        Returns:
            InferenceResponse with output, latency, token counts.
        """
        self._emit_inference_started(request)
        litellm = _litellm()
        start = time.time()

        try:
            messages: list[dict[str, Any]] = []
            cache_control = request.metadata.get("cache_control") if request.metadata else None

            if request.system_prompt:
                sys_msg: dict[str, Any] = {"role": "system", "content": request.system_prompt}
                # Anthropic prefix caching: mark system prompt as cacheable
                if cache_control and self._prefix == "anthropic":
                    sys_msg["cache_control"] = {"type": "ephemeral"}
                messages.append(sys_msg)
            messages.append({"role": "user", "content": request.prompt})

            kwargs: dict[str, Any] = {
                "model": self._model_id(request.model_id),
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "timeout": self.timeout_seconds,
            }

            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences

            response = litellm.completion(**kwargs)

            latency_ms = int((time.time() - start) * 1000)
            output = response.choices[0].message.content if response.choices else ""
            usage = getattr(response, "usage", None)
            tokens_used = usage.total_tokens if usage else 0

            resp = InferenceResponse(
                model_id=request.model_id,
                output=output,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                status=INFERENCE_STATUS_OK,
                metadata={
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                    "litellm_model": self._model_id(request.model_id),
                },
            )
            self._record_telemetry(request, resp)
            return resp

        except Exception:
            latency_ms = int((time.time() - start) * 1000)
            logger.exception("[LiteLLM] Inference failed for %s", request.model_id)
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error="Cloud inference failed — check server logs for details",
            )

    def stream_infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run streaming completion via LiteLLM SDK.

        Collects all chunks into a single response.  For true SSE streaming,
        use the async variant or integrate with ``async_support/streaming.py``.

        Args:
            request: InferenceRequest with model_id, prompt, and options.

        Returns:
            InferenceResponse with concatenated output from all chunks.
        """
        self._emit_inference_started(request)
        litellm = _litellm()
        start = time.time()

        try:
            messages: list[dict[str, str]] = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            kwargs: dict[str, Any] = {
                "model": self._model_id(request.model_id),
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
                "timeout": self.timeout_seconds,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key

            chunks: list[str] = []
            for chunk in litellm.completion(**kwargs):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    chunks.append(delta.content)

            latency_ms = int((time.time() - start) * 1000)
            output = "".join(chunks)

            resp = InferenceResponse(
                model_id=request.model_id,
                output=output,
                latency_ms=latency_ms,
                tokens_used=0,  # streaming doesn't always report usage
                status=INFERENCE_STATUS_OK,
            )
            self._record_telemetry(request, resp)
            return resp

        except Exception:
            latency_ms = int((time.time() - start) * 1000)
            logger.exception("[LiteLLM] Streaming failed for %s", request.model_id)
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error="Cloud streaming failed — check server logs for details",
            )

    # ── Capabilities ───────────────────────────────────────────────────

    def get_capabilities(self) -> dict[str, list[str]]:
        """Get capabilities of all discovered models.

        Returns:
            Dict mapping model_id to list of capability strings.
        """
        return {m.id: m.capabilities for m in self.models}
