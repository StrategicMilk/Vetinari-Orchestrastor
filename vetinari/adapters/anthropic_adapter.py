"""Anthropic provider adapter."""

import logging
import requests
import time
from typing import Dict, List, Any, Optional
from .base import (
    ProviderAdapter, ProviderConfig, ProviderType, ModelInfo,
    InferenceRequest, InferenceResponse
)

logger = logging.getLogger(__name__)


class AnthropicProviderAdapter(ProviderAdapter):
    """Adapter for Anthropic API (Claude models)."""

    # Map from friendly/config model IDs to real Anthropic API model IDs
    MODEL_ID_MAP: Dict[str, str] = {
        "claude-sonnet-4": "claude-sonnet-4-5",
        "claude-opus-4": "claude-opus-4-5",
        "claude-haiku-3": "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

    def __init__(self, config: ProviderConfig):
        """Initialize Anthropic adapter."""
        if config.provider_type != ProviderType.ANTHROPIC:
            raise ValueError("AnthropicProviderAdapter requires ANTHROPIC provider type")
        super().__init__(config)
        self.session = requests.Session()
        self.api_key = config.api_key
        if not self.api_key:
            raise ValueError("Anthropic adapter requires api_key in config")
        # Anthropic uses a specific API version header
        self.api_version = "2023-06-01"

    def _resolve_model_id(self, model_id: str) -> str:
        """Resolve a config model ID to the real Anthropic API model ID."""
        return self.MODEL_ID_MAP.get(model_id, model_id)

    def discover_models(self) -> List[ModelInfo]:
        """Discover available models from Anthropic."""
        try:
            # Anthropic models are hardcoded (they don't have a discovery endpoint)
            models_data = [
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "name": "Claude 3.5 Sonnet",
                    "context_len": 200000,
                    "memory_gb": 32,
                    "latency_estimate_ms": 1000,
                    "cost_per_1k_tokens": 0.003,
                    "capabilities": ["code_gen", "chat", "reasoning", "vision"],
                },
                {
                    "id": "claude-3-opus-20250219",
                    "name": "Claude 3 Opus",
                    "context_len": 200000,
                    "memory_gb": 32,
                    "latency_estimate_ms": 1500,
                    "cost_per_1k_tokens": 0.015,
                    "capabilities": ["code_gen", "chat", "reasoning", "vision"],
                },
                {
                    "id": "claude-3-sonnet-20240229",
                    "name": "Claude 3 Sonnet",
                    "context_len": 200000,
                    "memory_gb": 32,
                    "latency_estimate_ms": 1200,
                    "cost_per_1k_tokens": 0.003,
                    "capabilities": ["code_gen", "chat", "reasoning", "vision"],
                },
                {
                    "id": "claude-3-haiku-20240307",
                    "name": "Claude 3 Haiku",
                    "context_len": 200000,
                    "memory_gb": 16,
                    "latency_estimate_ms": 600,
                    "cost_per_1k_tokens": 0.00025,
                    "capabilities": ["code_gen", "chat"],
                },
            ]
            
            discovered = []
            for m in models_data:
                model_info = ModelInfo(
                    id=m["id"],
                    name=m["name"],
                    provider="anthropic",
                    endpoint="https://api.anthropic.com/v1/messages",
                    capabilities=m.get("capabilities", ["chat"]),
                    context_len=m.get("context_len", 200000),
                    memory_gb=m.get("memory_gb", 32),
                    version="unknown",
                    latency_estimate_ms=m.get("latency_estimate_ms", 1000),
                    cost_per_1k_tokens=m.get("cost_per_1k_tokens", 0.003),
                    tags=["cloud", "anthropic", "commercial"],
                )
                discovered.append(model_info)
            
            self.models = discovered
            logger.info(f"[Anthropic] Discovered {len(discovered)} models")
            return discovered

        except Exception as e:
            logger.error(f"[Anthropic] Model discovery failed: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API health."""
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "Content-Type": "application/json",
            }
            
            # Try a simple request
            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "test"}],
            }
            
            response = self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=5
            )
            return {
                "healthy": response.status_code == 200,
                "reason": "Anthropic API responding" if response.status_code == 200 else f"Status {response.status_code}",
                "timestamp": time.time(),
                "endpoint": "https://api.anthropic.com/v1",
            }
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": time.time(),
                "endpoint": "https://api.anthropic.com/v1",
            }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using Anthropic API."""
        start_time = time.time()

        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version,
                "Content-Type": "application/json",
            }
            
            messages = [{"role": "user", "content": request.prompt}]

            resolved_model = self._resolve_model_id(request.model_id)

            payload = {
                "model": resolved_model,
                "max_tokens": request.max_tokens,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }
            
            # Add system prompt if provided (Anthropic has a system field)
            if request.system_prompt:
                payload["system"] = request.system_prompt

            response = self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            output = ""
            tokens_used = 0

            if "content" in data and len(data["content"]) > 0:
                content = data["content"][0]
                if "text" in content:
                    output = content["text"]

            if "usage" in data:
                tokens_used = data["usage"].get("output_tokens", 0) + data["usage"].get("input_tokens", 0)

            resp = InferenceResponse(
                model_id=request.model_id,
                output=output,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                status="ok",
            )
            self._record_telemetry(request, resp)
            return resp

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[Anthropic] Inference failed: {e}")
            resp = InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error=str(e),
            )
            self._record_telemetry(request, resp)
            return resp

    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all models."""
        return {m.id: m.capabilities for m in self.models}
