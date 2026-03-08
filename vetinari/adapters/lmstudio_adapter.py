"""LM Studio provider adapter."""

import logging
import requests
import time
from typing import Dict, List, Any, Optional
from .base import (
    ProviderAdapter, ProviderConfig, ProviderType, ModelInfo,
    InferenceRequest, InferenceResponse
)

logger = logging.getLogger(__name__)


class LMStudioProviderAdapter(ProviderAdapter):
    """Adapter for LM Studio local inference."""

    def __init__(self, config: ProviderConfig):
        """Initialize LM Studio adapter."""
        if config.provider_type != ProviderType.LM_STUDIO:
            raise ValueError("LMStudioProviderAdapter requires LM_STUDIO provider type")
        super().__init__(config)
        self.session = requests.Session()

    def discover_models(self) -> List[ModelInfo]:
        """Discover models available in LM Studio."""
        try:
            response = self.session.get(
                f"{self.endpoint}/v1/models",
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()

            # Handle both dict and list response formats
            if isinstance(data, dict):
                if "data" in data:
                    models_list = data["data"]
                elif "models" in data:
                    models_list = data["models"]
                else:
                    models_list = []
            else:
                models_list = data if isinstance(data, list) else []

            discovered = []
            for m in models_list:
                model_id = m.get("id", "")
                if not model_id:
                    continue

                model_info = ModelInfo(
                    id=model_id,
                    name=m.get("name", model_id),
                    provider="lm_studio",
                    endpoint=f"{self.endpoint}/v1/chat/completions",
                    capabilities=m.get("capabilities", ["code_gen", "chat"]),
                    context_len=m.get("context_len", 2048),
                    memory_gb=m.get("memory_gb", 4),
                    version=m.get("version", "unknown"),
                    latency_estimate_ms=m.get("latency_estimate_ms", 1000),
                    tags=["local", "lm_studio"],
                )
                discovered.append(model_info)

            self.models = discovered
            logger.info("[LMStudio] Discovered %s models", len(discovered))
            return discovered

        except Exception as e:
            logger.error("[LMStudio] Model discovery failed: %s", e)
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check LM Studio health."""
        try:
            response = self.session.get(
                f"{self.endpoint}/v1/models",
                timeout=5
            )
            return {
                "healthy": response.status_code == 200,
                "reason": "LM Studio responding" if response.status_code == 200 else f"Status {response.status_code}",
                "timestamp": time.time(),
                "endpoint": self.endpoint,
            }
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": time.time(),
                "endpoint": self.endpoint,
            }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using LM Studio."""
        start_time = time.time()

        try:
            payload = {
                "model": request.model_id,
                "messages": [
                    {"role": "system", "content": request.system_prompt or ""},
                    {"role": "user", "content": request.prompt}
                ],
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }

            response = self.session.post(
                f"{self.endpoint}/v1/chat/completions",
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
            logger.error("[LMStudio] Inference failed: %s", e)
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
