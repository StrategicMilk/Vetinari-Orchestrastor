"""Provider adapters for multi-LLM orchestration.

This module provides pluggable adapters for different LLM providers:
- LM Studio (local)
- OpenAI (cloud)
- Cohere (cloud)
- Anthropic (cloud)
- Google Gemini (cloud)
"""

from __future__ import annotations

from .anthropic_adapter import AnthropicProviderAdapter
from .base import InferenceRequest, InferenceResponse, ModelInfo, ProviderAdapter, ProviderConfig, ProviderType
from .cohere_adapter import CohereProviderAdapter
from .gemini_adapter import GeminiProviderAdapter
from .lmstudio_adapter import LMStudioProviderAdapter
from .openai_adapter import OpenAIProviderAdapter
from .registry import AdapterRegistry

__all__ = [
    "AdapterRegistry",
    "AnthropicProviderAdapter",
    "CohereProviderAdapter",
    "GeminiProviderAdapter",
    "InferenceRequest",
    "InferenceResponse",
    "LMStudioProviderAdapter",
    "ModelInfo",
    "OpenAIProviderAdapter",
    "ProviderAdapter",
    "ProviderConfig",
    "ProviderType",
]
