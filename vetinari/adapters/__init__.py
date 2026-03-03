"""Provider adapters for multi-LLM orchestration.

This module provides pluggable adapters for different LLM providers:
- LM Studio (local)
- OpenAI (cloud)
- Cohere (cloud)
- Anthropic (cloud)
- Google Gemini (cloud)
"""

from .base import (
    ProviderAdapter, ProviderConfig, ProviderType, ModelInfo, 
    InferenceRequest, InferenceResponse
)
from .lmstudio_adapter import LMStudioProviderAdapter
from .openai_adapter import OpenAIProviderAdapter
from .cohere_adapter import CohereProviderAdapter
from .anthropic_adapter import AnthropicProviderAdapter
from .gemini_adapter import GeminiProviderAdapter
from .registry import AdapterRegistry

__all__ = [
    "ProviderAdapter",
    "ProviderConfig",
    "ProviderType",
    "ModelInfo",
    "InferenceRequest",
    "InferenceResponse",
    "LMStudioProviderAdapter",
    "OpenAIProviderAdapter",
    "CohereProviderAdapter",
    "AnthropicProviderAdapter",
    "GeminiProviderAdapter",
    "AdapterRegistry",
]
