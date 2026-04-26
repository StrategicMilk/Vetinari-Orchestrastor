"""Provider adapters for multi-LLM orchestration.

This module provides pluggable adapters for different LLM providers:
- LiteLLM (unified adapter for all cloud providers — ADR-0062)
- Local (llama-cpp-python, in-process GGUF inference)
- OpenAI-compatible servers: vLLM, NVIDIA NIMs (ADR-0084)

Grammar-constrained generation is available via ``grammar_library``:
- ``GRAMMAR_LIBRARY`` — pre-built GBNF grammars by name
- ``TASK_TYPE_TO_GRAMMAR`` — task type -> grammar name mapping
- ``get_grammar`` / ``get_grammar_for_task_type`` — lookup helpers
- ``validate_grammar`` — structural BNF validation
- ``truncate_at_grammar_boundary`` — BudgetForcing-inspired truncation
"""

from __future__ import annotations

from .base import InferenceRequest, InferenceResponse, ModelInfo, ProviderAdapter, ProviderConfig
from .grammar_library import (
    get_grammar,
    get_grammar_for_task_type,
    truncate_at_grammar_boundary,
    validate_grammar,
)
from .litellm_adapter import LiteLLMAdapter
from .llama_cpp_adapter import LlamaCppProviderAdapter
from .llama_cpp_local_adapter import LocalInferenceAdapter
from .openai_server_adapter import OpenAIServerAdapter
from .registry import AdapterRegistry

__all__ = [
    "AdapterRegistry",
    "InferenceRequest",
    "InferenceResponse",
    "LiteLLMAdapter",
    "LlamaCppProviderAdapter",
    "LocalInferenceAdapter",
    "ModelInfo",
    "OpenAIServerAdapter",
    "ProviderAdapter",
    "ProviderConfig",
    "get_grammar",
    "get_grammar_for_task_type",
    "truncate_at_grammar_boundary",
    "validate_grammar",
]
