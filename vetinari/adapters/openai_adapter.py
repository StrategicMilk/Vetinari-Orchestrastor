"""OpenAI provider adapter."""

import logging
import requests
import time
from typing import Dict, List, Any, Optional
from .base import (
    ProviderAdapter, ProviderConfig, ProviderType, ModelInfo,
    InferenceRequest, InferenceResponse
)

logger = logging.getLogger(__name__)


class OpenAIProviderAdapter(ProviderAdapter):
    """Adapter for OpenAI API (GPT-3.5, GPT-4, etc.)."""

    def __init__(self, config: ProviderConfig):
        """Initialize OpenAI adapter."""
        if config.provider_type != ProviderType.OPENAI:
            raise ValueError("OpenAIProviderAdapter requires OPENAI provider type")
        super().__init__(config)
        self.session = requests.Session()
        self.api_key = config.api_key
        if not self.api_key:
            raise ValueError("OpenAI adapter requires api_key in config")

    def discover_models(self) -> List[ModelInfo]:
        """Discover available models from OpenAI."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # OpenAI models are hardcoded (they don't have a discovery endpoint)
            # This reflects the current OpenAI API structure
            models_data = [
                {
                    "id": "gpt-4o",
                    "name": "GPT-4 Optimized",
                    "context_len": 128000,
                    "memory_gb": 32,
                    "latency_estimate_ms": 2000,
                    "cost_per_1k_tokens": 0.015,
                    "capabilities": ["code_gen", "chat", "reasoning", "vision"],
                },
                {
                    "id": "gpt-4-turbo",
                    "name": "GPT-4 Turbo",
                    "context_len": 128000,
                    "memory_gb": 32,
                    "latency_estimate_ms": 1500,
                    "cost_per_1k_tokens": 0.01,
                    "capabilities": ["code_gen", "chat", "reasoning", "vision"],
                },
                {
                    "id": "gpt-4",
                    "name": "GPT-4",
                    "context_len": 8192,
                    "memory_gb": 16,
                    "latency_estimate_ms": 3000,
                    "cost_per_1k_tokens": 0.03,
                    "capabilities": ["code_gen", "chat", "reasoning"],
                },
                {
                    "id": "gpt-3.5-turbo",
                    "name": "GPT-3.5 Turbo",
                    "context_len": 4096,
                    "memory_gb": 8,
                    "latency_estimate_ms": 500,
                    "cost_per_1k_tokens": 0.002,
                    "capabilities": ["code_gen", "chat"],
                },
            ]
            
            discovered = []
            for m in models_data:
                model_info = ModelInfo(
                    id=m["id"],
                    name=m["name"],
                    provider="openai",
                    endpoint="https://api.openai.com/v1/chat/completions",
                    capabilities=m.get("capabilities", ["chat"]),
                    context_len=m.get("context_len", 4096),
                    memory_gb=m.get("memory_gb", 8),
                    version="unknown",
                    latency_estimate_ms=m.get("latency_estimate_ms", 1000),
                    cost_per_1k_tokens=m.get("cost_per_1k_tokens", 0.002),
                    tags=["cloud", "openai", "commercial"],
                )
                discovered.append(model_info)
            
            self.models = discovered
            logger.info(f"[OpenAI] Discovered {len(discovered)} models")
            return discovered

        except Exception as e:
            logger.error(f"[OpenAI] Model discovery failed: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API health."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            response = self.session.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=5
            )
            return {
                "healthy": response.status_code == 200,
                "reason": "OpenAI API responding" if response.status_code == 200 else f"Status {response.status_code}",
                "timestamp": time.time(),
                "endpoint": "https://api.openai.com/v1",
            }
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": time.time(),
                "endpoint": "https://api.openai.com/v1",
            }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using OpenAI API."""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            payload = {
                "model": request.model_id,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }

            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
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

            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    output = choice["message"].get("content", "")
                elif "text" in choice:
                    output = choice["text"]

            if "usage" in data:
                tokens_used = data["usage"].get("total_tokens", 0)

            return InferenceResponse(
                model_id=request.model_id,
                output=output,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                status="ok",
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[OpenAI] Inference failed: {e}")
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error=str(e),
            )

    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all models."""
        return {m.id: m.capabilities for m in self.models}
