"""Google Gemini provider adapter."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from .base import InferenceRequest, InferenceResponse, ModelInfo, ProviderAdapter, ProviderConfig, ProviderType

logger = logging.getLogger(__name__)


class GeminiProviderAdapter(ProviderAdapter):
    """Adapter for Google Gemini API."""

    # Hardcoded fallback model list — used when config/provider_models.yaml is unavailable
    _HARDCODED_MODELS = [
        {
            "id": "gemini-2.0-flash",
            "name": "Gemini 2.0 Flash",
            "context_len": 1000000,
            "memory_gb": 32,
            "latency_estimate_ms": 800,
            "cost_per_1k_tokens": 0.00007,
            "free_tier": True,
            "capabilities": ["code_gen", "chat", "reasoning", "vision"],
        },
        {
            "id": "gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "context_len": 2000000,
            "memory_gb": 32,
            "latency_estimate_ms": 1200,
            "cost_per_1k_tokens": 0.00125,
            "free_tier": True,
            "capabilities": ["code_gen", "chat", "reasoning", "vision"],
        },
        {
            "id": "gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "context_len": 1000000,
            "memory_gb": 32,
            "latency_estimate_ms": 600,
            "cost_per_1k_tokens": 0.000075,
            "free_tier": True,
            "capabilities": ["code_gen", "chat", "vision"],
        },
        {
            "id": "gemini-1.0-pro",
            "name": "Gemini 1.0 Pro",
            "context_len": 32000,
            "memory_gb": 16,
            "latency_estimate_ms": 1000,
            "cost_per_1k_tokens": 0.0005,
            "free_tier": True,
            "capabilities": ["code_gen", "chat"],
        },
    ]

    def __init__(self, config: ProviderConfig):
        """Initialize Gemini adapter."""
        if config.provider_type != ProviderType.GEMINI:
            raise ValueError("GeminiProviderAdapter requires GEMINI provider type")
        super().__init__(config)
        self.session = requests.Session()
        self.api_key = config.api_key
        if not self.api_key:
            raise ValueError("Gemini adapter requires api_key in config")
        # Pass API key via header instead of URL query param to prevent key leakage
        # in server logs, browser history, and referrer headers
        self.session.headers.update({"x-goog-api-key": self.api_key})

    def _load_model_definitions(self) -> list[dict[str, Any]]:
        """Load model definitions from config/provider_models.yaml, falling back to hardcoded."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "config", "provider_models.yaml"
            )
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
            provider_models = config.get("providers", {}).get("google", {}).get("models", [])
            if provider_models:
                return provider_models
        except Exception:
            logger.debug("Config file not available, using hardcoded models")
        return self._HARDCODED_MODELS

    def discover_models(self) -> list[ModelInfo]:
        """Discover available models from Google Gemini."""
        try:
            # Load model list from config file (falls back to hardcoded if unavailable)
            models_data = self._load_model_definitions()

            discovered = []
            for m in models_data:
                model_info = ModelInfo(
                    id=m["id"],
                    name=m["name"],
                    provider="gemini",
                    endpoint="https://generativelanguage.googleapis.com/v1beta/models",
                    capabilities=m.get("capabilities", ["chat"]),
                    context_len=m.get("context_len", 32000),
                    memory_gb=m.get("memory_gb", 16),
                    version="unknown",
                    latency_estimate_ms=m.get("latency_estimate_ms", 1000),
                    cost_per_1k_tokens=m.get("cost_per_1k_tokens", 0.0),
                    free_tier=m.get("free_tier", False),
                    tags=["cloud", "gemini", "google"],
                )
                discovered.append(model_info)

            self.models = discovered
            logger.info("[Gemini] Discovered %s models", len(discovered))
            return discovered

        except Exception as e:
            logger.error("[Gemini] Model discovery failed: %s", e)
            return []

    def health_check(self) -> dict[str, Any]:
        """Check Google Gemini API health."""
        try:
            # Try to list models
            response = self.session.get("https://generativelanguage.googleapis.com/v1beta/models", timeout=5)
            return {
                "healthy": response.status_code == 200,
                "reason": "Gemini API responding" if response.status_code == 200 else f"Status {response.status_code}",
                "timestamp": time.time(),
                "endpoint": "https://generativelanguage.googleapis.com/v1beta",
            }
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": time.time(),
                "endpoint": "https://generativelanguage.googleapis.com/v1beta",
            }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using Google Gemini API."""
        start_time = time.time()

        try:
            # Gemini uses generateContent endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model_id}:generateContent"

            # Build contents array (Gemini's structure)
            contents = []

            if request.system_prompt:
                contents.append({"role": "user", "parts": [{"text": request.system_prompt}]})
                contents.append({"role": "model", "parts": [{"text": "Understood."}]})

            contents.append({"role": "user", "parts": [{"text": request.prompt}]})

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": request.temperature,
                    "topP": request.top_p,
                    "topK": request.top_k,
                    "maxOutputTokens": request.max_tokens,
                },
            }

            if request.stop_sequences:
                payload["generationConfig"]["stopSequences"] = request.stop_sequences

            response = self.session.post(
                url, json=payload, timeout=self.timeout_seconds, headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            output = ""
            tokens_used = 0

            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:  # noqa: SIM102
                    if len(candidate["content"]["parts"]) > 0:
                        output = candidate["content"]["parts"][0].get("text", "")

            if "usageMetadata" in data:
                tokens_used = data["usageMetadata"].get("totalTokenCount", 0)

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
            logger.error("[Gemini] Inference failed: %s", e)
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error=str(e),
            )

    def get_capabilities(self) -> dict[str, list[str]]:
        """Get capabilities of all models."""
        return {m.id: m.capabilities for m in self.models}
