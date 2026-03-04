"""Cohere provider adapter."""

import logging
import requests
import time
from typing import Dict, List, Any, Optional
from .base import (
    ProviderAdapter, ProviderConfig, ProviderType, ModelInfo,
    InferenceRequest, InferenceResponse
)

logger = logging.getLogger(__name__)


class CohereProviderAdapter(ProviderAdapter):
    """Adapter for Cohere API (Command, Generate, etc.)."""

    def __init__(self, config: ProviderConfig):
        """Initialize Cohere adapter."""
        if config.provider_type != ProviderType.COHERE:
            raise ValueError("CohereProviderAdapter requires COHERE provider type")
        super().__init__(config)
        self.session = requests.Session()
        self.api_key = config.api_key
        if not self.api_key:
            raise ValueError("Cohere adapter requires api_key in config")

    def discover_models(self) -> List[ModelInfo]:
        """Discover available models from Cohere."""
        try:
            # Cohere models are hardcoded (they don't have a discovery endpoint)
            models_data = [
                {
                    "id": "command-r-plus",
                    "name": "Command R Plus",
                    "context_len": 128000,
                    "memory_gb": 32,
                    "latency_estimate_ms": 1500,
                    "cost_per_1k_tokens": 0.003,
                    "capabilities": ["code_gen", "chat", "retrieval"],
                },
                {
                    "id": "command-r",
                    "name": "Command R",
                    "context_len": 128000,
                    "memory_gb": 32,
                    "latency_estimate_ms": 1000,
                    "cost_per_1k_tokens": 0.0005,
                    "capabilities": ["code_gen", "chat", "retrieval"],
                },
                {
                    "id": "command",
                    "name": "Command",
                    "context_len": 4096,
                    "memory_gb": 8,
                    "latency_estimate_ms": 800,
                    "cost_per_1k_tokens": 0.0001,
                    "capabilities": ["code_gen", "chat"],
                },
                {
                    "id": "command-nightly",
                    "name": "Command Nightly",
                    "context_len": 4096,
                    "memory_gb": 8,
                    "latency_estimate_ms": 800,
                    "cost_per_1k_tokens": 0.0001,
                    "capabilities": ["code_gen", "chat"],
                },
            ]
            
            discovered = []
            for m in models_data:
                model_info = ModelInfo(
                    id=m["id"],
                    name=m["name"],
                    provider="cohere",
                    endpoint="https://api.cohere.ai/v1/generate",
                    capabilities=m.get("capabilities", ["chat"]),
                    context_len=m.get("context_len", 4096),
                    memory_gb=m.get("memory_gb", 8),
                    version="unknown",
                    latency_estimate_ms=m.get("latency_estimate_ms", 1000),
                    cost_per_1k_tokens=m.get("cost_per_1k_tokens", 0.0001),
                    tags=["cloud", "cohere", "commercial"],
                )
                discovered.append(model_info)
            
            self.models = discovered
            logger.info(f"[Cohere] Discovered {len(discovered)} models")
            return discovered

        except Exception as e:
            logger.error(f"[Cohere] Model discovery failed: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check Cohere API health."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Try a simple generate request with minimal tokens
            payload = {
                "model": "command",
                "prompt": "test",
                "max_tokens": 1,
            }
            
            response = self.session.post(
                "https://api.cohere.ai/v1/generate",
                headers=headers,
                json=payload,
                timeout=5
            )
            return {
                "healthy": response.status_code in [200, 201],
                "reason": "Cohere API responding" if response.status_code in [200, 201] else f"Status {response.status_code}",
                "timestamp": time.time(),
                "endpoint": "https://api.cohere.ai/v1",
            }
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": time.time(),
                "endpoint": "https://api.cohere.ai/v1",
            }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using Cohere Chat API (v2)."""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Build messages list for the chat API
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})

            payload = {
                "model": request.model_id,
                "messages": messages,
                "temperature": request.temperature,
                "p": request.top_p,
                "max_tokens": request.max_tokens,
            }
            if request.stop_sequences:
                payload["stop_sequences"] = request.stop_sequences

            response = self.session.post(
                "https://api.cohere.com/v2/chat",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response from v2 chat format
            output = ""
            tokens_used = 0

            if "message" in data:
                content_blocks = data["message"].get("content", [])
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        output += block.get("text", "")
                    elif isinstance(block, str):
                        output += block

            # Count BOTH input and output tokens
            usage = data.get("usage", {})
            billed = usage.get("billed_units", usage)
            input_tokens = billed.get("input_tokens", 0)
            output_tokens = billed.get("output_tokens", 0)
            tokens_used = input_tokens + output_tokens

            return InferenceResponse(
                model_id=request.model_id,
                output=output,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                status="ok",
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[Cohere] Inference failed: {e}")
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
