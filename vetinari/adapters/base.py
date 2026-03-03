"""Base provider adapter interface for multi-LLM orchestration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ProviderType(Enum):
    """Enumeration of supported provider types."""
    LM_STUDIO = "lm_studio"
    OPENAI = "openai"
    COHERE = "cohere"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    LOCAL = "local"


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    provider_type: ProviderType
    name: str
    endpoint: str
    api_key: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 120
    memory_budget_gb: int = 32
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about a model available from a provider."""
    id: str
    name: str
    provider: str
    endpoint: str
    capabilities: List[str]
    context_len: int
    memory_gb: int
    version: str
    latency_estimate_ms: int = 1000
    throughput_tokens_per_sec: float = 50.0
    cost_per_1k_tokens: float = 0.0
    free_tier: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class InferenceRequest:
    """Request to run inference on a model."""
    model_id: str
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Response from inference."""
    model_id: str
    output: str
    latency_ms: int
    tokens_used: int
    status: str  # "ok", "error", "timeout", "partial"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProviderAdapter(ABC):
    """
    Abstract base class for all provider adapters.
    
    Each adapter implements a consistent interface for:
    - Model discovery
    - Health checks
    - Inference execution
    - Capability querying
    """

    def __init__(self, config: ProviderConfig):
        """Initialize adapter with configuration."""
        self.config = config
        self.provider_type = config.provider_type
        self.name = config.name
        self.endpoint = config.endpoint
        self.api_key = config.api_key
        self.max_retries = config.max_retries
        self.timeout_seconds = config.timeout_seconds
        self.models: List[ModelInfo] = []

    @abstractmethod
    def discover_models(self) -> List[ModelInfo]:
        """
        Discover available models from the provider.
        
        Returns:
            List of ModelInfo objects representing available models.
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check health/status of the provider.
        
        Returns:
            Dict with keys: {"healthy": bool, "reason": str, "timestamp": str}
        """
        pass

    @abstractmethod
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """
        Run inference on a model.
        
        Args:
            request: InferenceRequest with model_id, prompt, and options
            
        Returns:
            InferenceResponse with output, latency, tokens_used, status
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, List[str]]:
        """
        Get capabilities of all available models.
        
        Returns:
            Dict mapping model_id to list of capabilities
            (e.g., ["code_gen", "chat", "summarization"])
        """
        pass

    def score_model_for_task(self, model: ModelInfo, task_requirements: Dict[str, Any]) -> float:
        """
        Score a model for a given task.
        
        Factors: capability match, context fit, latency, cost
        
        Args:
            model: ModelInfo to score
            task_requirements: Dict with keys like "required_capabilities", "input_tokens", "max_latency_ms"
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        score = 0.5

        # Capability match (35%)
        required_caps = set(task_requirements.get("required_capabilities", []))
        model_caps = set(model.capabilities)
        if required_caps:
            cap_match = len(required_caps & model_caps) / len(required_caps)
        else:
            cap_match = 1.0
        score += cap_match * 0.35

        # Context fit (20%)
        input_tokens = task_requirements.get("input_tokens", 1000)
        if input_tokens <= model.context_len:
            context_fit = 1.0
        else:
            context_fit = max(0.0, model.context_len / input_tokens)
        score += context_fit * 0.20

        # Latency (20%)
        max_latency_ms = task_requirements.get("max_latency_ms", 30000)
        if model.latency_estimate_ms <= max_latency_ms:
            latency_score = 1.0
        else:
            latency_score = max(0.0, 1.0 - (model.latency_estimate_ms - max_latency_ms) / max_latency_ms)
        score += latency_score * 0.20

        # Cost (15%)
        max_cost = task_requirements.get("max_cost_per_1k_tokens", 0.1)
        if model.cost_per_1k_tokens <= max_cost:
            cost_score = 1.0
        else:
            cost_score = max(0.0, 1.0 - (model.cost_per_1k_tokens / max_cost))
        score += cost_score * 0.15

        # Free tier bonus (10%)
        if model.free_tier:
            score += 0.10

        return min(1.0, score)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_type.value}, endpoint={self.endpoint})"
