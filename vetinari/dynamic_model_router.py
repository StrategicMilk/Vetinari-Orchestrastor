"""
Dynamic Model Router for Vetinari

Provides intelligent model selection based on:
- Task requirements and capabilities
- Model performance history
- Latency and resource constraints
- Cost optimization (for cloud models)
- Availability and health

Supports both local (LM Studio) and cloud models.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import threading

from vetinari.types import ModelProvider  # canonical enum from types.py

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks the system can handle."""
    PLANNING = "planning"
    ANALYSIS = "analysis"
    CODING = "coding"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REASONING = "reasoning"
    CREATIVE = "creative"
    DATA_PROCESSING = "data_processing"
    WEB_SEARCH = "web_search"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    GENERAL = "general"
    # Phase 7 additions
    CREATIVE_WRITING = "creative_writing"
    SECURITY_AUDIT = "security_audit"
    DEVOPS = "devops"
    IMAGE_GENERATION = "image_generation"
    COST_ANALYSIS = "cost_analysis"
    SPECIFICATION = "specification"


@dataclass
class ModelCapabilities:
    """Model capabilities and attributes."""
    # Core capabilities
    code_gen: bool = False
    reasoning: bool = False
    chat: bool = True
    creative: bool = False
    docs: bool = False
    math: bool = False
    analysis: bool = False
    summarization: bool = False
    
    # Technical specs
    context_length: int = 2048
    supports_functions: bool = False
    supports_vision: bool = False
    supports_json: bool = False
    
    # Preferred for
    preferred_for: List[str] = field(default_factory=list)
    
    # Tags from discovery
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelCapabilities':
        """Create capabilities from model data."""
        caps = cls()
        
        # Extract from capabilities list
        caps_list = data.get("capabilities", [])
        if isinstance(caps_list, list):
            cap_strs = [str(c).lower() for c in caps_list]
            caps.code_gen = any("code" in c or "coder" in c for c in cap_strs)
            caps.reasoning = any("reason" in c for c in cap_strs)
            caps.chat = any("chat" in c for c in cap_strs)
            caps.creative = any("creative" in c or "story" in c for c in cap_strs)
            caps.docs = any("doc" in c for c in cap_strs)
            caps.math = any("math" in c or "calc" in c for c in cap_strs)
            caps.analysis = any("analysis" in c or "analyze" in c for c in cap_strs)
            caps.summarization = any("summary" in c or "summarize" in c for c in cap_strs)
        
        # Extract from tags
        tags = data.get("tags", [])
        if isinstance(tags, list):
            tag_strs = [str(t).lower() for t in tags]
            caps.tags = tag_strs
            
            # Infer capabilities from tags
            if any("code" in t for t in tag_strs):
                caps.code_gen = True
            if any("reason" in t for t in tag_strs):
                caps.reasoning = True
            if any("chat" in t for t in tag_strs):
                caps.chat = True
        
        # Technical specs
        caps.context_length = data.get("context_len", data.get("context_length", 2048))
        caps.supports_functions = data.get("supports_functions", False)
        caps.supports_vision = data.get("supports_vision", False)
        caps.supports_json = data.get("supports_json", False)
        
        # Preferred for
        caps.preferred_for = data.get("preferred_for", [])
        
        return caps
    
    def matches_task(self, task_type: TaskType) -> float:
        """Return a score (0-1) for how well this model matches a task type."""
        scores = {
            TaskType.PLANNING: 0.3 * int(self.reasoning) + 0.3 * int(self.code_gen) + 0.2 * int(self.analysis) + 0.2 * int(self.docs),
            TaskType.ANALYSIS: 0.4 * int(self.analysis) + 0.3 * int(self.reasoning) + 0.3 * int(self.code_gen),
            TaskType.CODING: 0.8 * int(self.code_gen) + 0.1 * int(self.reasoning) + 0.1 * int(self.chat),
            TaskType.CODE_REVIEW: 0.5 * int(self.code_gen) + 0.3 * int(self.reasoning) + 0.2 * int(self.analysis),
            TaskType.TESTING: 0.6 * int(self.code_gen) + 0.2 * int(self.reasoning) + 0.2 * int(self.docs),
            TaskType.DOCUMENTATION: 0.5 * int(self.docs) + 0.3 * int(self.code_gen) + 0.2 * int(self.chat),
            TaskType.REASONING: 0.6 * int(self.reasoning) + 0.2 * int(self.analysis) + 0.2 * int(self.math),
            TaskType.CREATIVE: 0.6 * int(self.creative) + 0.4 * int(self.chat),
            TaskType.DATA_PROCESSING: 0.5 * int(self.code_gen) + 0.3 * int(self.analysis) + 0.2 * int(self.reasoning),
            TaskType.WEB_SEARCH: 0.4 * int(self.analysis) + 0.3 * int(self.reasoning) + 0.3 * int(self.chat),
            TaskType.SUMMARIZATION: 0.6 * int(self.summarization) + 0.4 * int(self.chat),
            TaskType.TRANSLATION: 0.5 * int(self.chat) + 0.3 * int(self.creative) + 0.2 * int(self.reasoning),
            TaskType.GENERAL: 0.4 * int(self.chat) + 0.3 * int(self.code_gen) + 0.3 * int(self.reasoning),
            # Phase 7 additions
            TaskType.CREATIVE_WRITING: 0.6 * int(self.creative) + 0.3 * int(self.chat) + 0.1 * int(self.reasoning),
            TaskType.SECURITY_AUDIT: 0.5 * int(self.code_gen) + 0.3 * int(self.reasoning) + 0.2 * int(self.analysis),
            TaskType.DEVOPS: 0.4 * int(self.code_gen) + 0.3 * int(self.reasoning) + 0.3 * int(self.docs),
            TaskType.IMAGE_GENERATION: 0.5 * int(self.creative) + 0.3 * int(self.chat) + 0.2 * int(self.reasoning),
            TaskType.COST_ANALYSIS: 0.4 * int(self.analysis) + 0.3 * int(self.math) + 0.3 * int(self.reasoning),
            TaskType.SPECIFICATION: 0.4 * int(self.reasoning) + 0.3 * int(self.docs) + 0.3 * int(self.analysis),
        }
        return scores.get(task_type, 0.5)


@dataclass
class ModelInfo:
    """Complete model information."""
    id: str
    name: str
    provider: ModelProvider = ModelProvider.LOCAL
    endpoint: str = ""
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    
    # Resource info
    memory_gb: float = 2.0
    context_length: int = 2048
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    total_uses: int = 0
    
    # Availability
    is_available: bool = True
    last_checked: str = ""
    load_percentage: float = 0.0  # 0-100
    
    # Metadata
    version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelInfo':
        """Create ModelInfo from dictionary."""
        info = cls(
            id=data.get("id", data.get("name", "")),
            name=data.get("name", data.get("id", "")),
            endpoint=data.get("endpoint", ""),
            memory_gb=data.get("memory_gb", data.get("memory", 2)),
            context_length=data.get("context_len", data.get("context_length", 2048)),
            version=data.get("version", ""),
            metadata=data.get("metadata", {})
        )
        
        # Determine provider
        id_lower = info.id.lower()
        if "gpt" in id_lower:
            info.provider = ModelProvider.OPENAI
        elif "claude" in id_lower:
            info.provider = ModelProvider.ANTHROPIC
        elif "gemini" in id_lower or "google" in id_lower:
            info.provider = ModelProvider.GOOGLE
        elif "llama" in id_lower or "mistral" in id_lower or "codellama" in id_lower:
            info.provider = ModelProvider.LOCAL
        elif "cloud:" in id_lower:
            info.provider = ModelProvider.OTHER
        
        # Set capabilities
        info.capabilities = ModelCapabilities.from_dict(data)
        
        # Set performance from metadata
        if "metrics" in data:
            info.avg_latency_ms = data["metrics"].get("avg_latency_ms", 0)
            info.success_rate = data["metrics"].get("success_rate", 1.0)
        
        return info
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider.value,
            "endpoint": self.endpoint,
            "capabilities": {
                "code_gen": self.capabilities.code_gen,
                "reasoning": self.capabilities.reasoning,
                "chat": self.capabilities.chat,
                "creative": self.capabilities.creative,
                "docs": self.capabilities.docs,
                "math": self.capabilities.math,
                "analysis": self.capabilities.analysis,
                "summarization": self.capabilities.summarization,
                "context_length": self.capabilities.context_length,
                "supports_functions": self.capabilities.supports_functions,
                "supports_vision": self.capabilities.supports_vision,
                "supports_json": self.capabilities.supports_json,
                "tags": self.capabilities.tags,
            },
            "memory_gb": self.memory_gb,
            "context_length": self.context_length,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "total_uses": self.total_uses,
            "is_available": self.is_available,
            "last_checked": self.last_checked,
            "load_percentage": self.load_percentage,
            "version": self.version,
        }


@dataclass
class ModelSelection:
    """Result of model selection process."""
    model: ModelInfo
    score: float
    reasoning: str
    alternatives: List[ModelInfo] = field(default_factory=list)
    confidence: float = 1.0


class DynamicModelRouter:
    """
    Dynamic model routing based on task requirements and model capabilities.
    
    Features:
    - Task-type aware model selection
    - Performance-based routing
    - Latency optimization
    - Cost optimization (for cloud models)
    - Fallback handling
    - Model health checking
    """
    
    def __init__(self, 
                 prefer_local: bool = True,
                 max_latency_ms: float = 60000,
                 max_memory_gb: float = 64):
        """
        Initialize the model router.
        
        Args:
            prefer_local: Prefer local models over cloud when possible
            max_latency_ms: Maximum acceptable latency in milliseconds
            max_memory_gb: Maximum memory to use
        """
        self.prefer_local = prefer_local
        self.max_latency_ms = max_latency_ms
        self.max_memory_gb = max_memory_gb
        
        # Model registry
        self.models: Dict[str, ModelInfo] = {}
        
        # Performance tracking
        self._performance_cache: Dict[str, Dict[str, Any]] = {}
        
        # Selection history
        self._selection_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # Callbacks
        self._health_check_callback: Optional[Callable] = None
        
        logger.info("DynamicModelRouter initialized (prefer_local=%s)", prefer_local)
    
    def get_performance_cache(self, cache_key: str) -> Dict[str, Any]:
        """Thread-safe read from performance cache."""
        with self._lock:
            return self._performance_cache.get(cache_key, {}).copy()

    def update_performance_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Thread-safe update to performance cache."""
        with self._lock:
            self._performance_cache[cache_key] = data

    def register_model(self, model: ModelInfo):
        """Register a model in the router."""
        self.models[model.id] = model
        logger.debug("Registered model: %s", model.id)
    
    def register_models_from_pool(self, models: List[Dict]):
        """Register models from a model pool (list of dicts)."""
        for m in models:
            model_info = ModelInfo.from_dict(m)
            self.register_model(model_info)
        logger.info("Registered %s models from pool", len(models))
    
    def set_health_check_callback(self, callback: Callable):
        """Set a callback for health checking models."""
        self._health_check_callback = callback
    
    def update_model_performance(self, model_id: str, 
                                latency_ms: float, 
                                success: bool,
                                task_type: TaskType = None):
        """Update performance metrics for a model."""
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        
        # Update metrics
        total = model.total_uses + 1
        model.avg_latency_ms = (
            (model.avg_latency_ms * model.total_uses + latency_ms) / total
        )
        model.success_rate = (
            (model.success_rate * model.total_uses + (1 if success else 0)) / total
        )
        model.total_uses = total
        model.last_checked = datetime.now().isoformat()
        
        # Store in cache (thread-safe)
        cache_key = f"{model_id}:{task_type.value if task_type else 'general'}"
        with self._lock:
            self._performance_cache[cache_key] = {
                "latency_ms": latency_ms,
                "success": success,
                "timestamp": time.time()
            }
    
    def select_model(self, 
                    task_type: TaskType,
                    task_description: str = "",
                    required_capabilities: List[str] = None,
                    preferred_models: List[str] = None,
                    context_length_needed: int = None) -> ModelSelection:
        """
        Select the best model for a given task.
        
        Args:
            task_type: Type of task to perform
            task_description: Description of the task
            required_capabilities: List of required capabilities
            preferred_models: List of preferred model IDs (in order)
            context_length_needed: Required context length
            
        Returns:
            ModelSelection with chosen model and reasoning
        """
        candidates = []
        
        # Filter available models
        for model_id, model in self.models.items():
            if not model.is_available:
                continue
            
            # Filter by memory constraints
            if model.memory_gb > self.max_memory_gb:
                continue
            
            # Filter by latency
            if model.avg_latency_ms > self.max_latency_ms and model.avg_latency_ms > 0:
                continue
            
            # Filter by context length
            if context_length_needed and model.context_length < context_length_needed:
                continue
            
            # Filter by required capabilities
            if required_capabilities:
                caps = model.capabilities
                meets_requirements = True
                for req in required_capabilities:
                    if req == "code_gen" and not caps.code_gen:
                        meets_requirements = False
                    elif req == "reasoning" and not caps.reasoning:
                        meets_requirements = False
                    elif req == "docs" and not caps.docs:
                        meets_requirements = False
                if not meets_requirements:
                    continue
            
            candidates.append(model)
        
        if not candidates:
            # Fallback: return any available model
            available = [m for m in self.models.values() if m.is_available]
            if not available:
                logger.warning("No models available!")
                return None
            
            # Pick random available model as fallback
            fallback = random.choice(available)
            return ModelSelection(
                model=fallback,
                score=0.0,
                reasoning="Fallback: no models matched criteria",
                confidence=0.1
            )
        
        # Score candidates
        scored = []
        for model in candidates:
            score = self._score_model(model, task_type, task_description, preferred_models)
            scored.append((model, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Get best and alternatives
        best_model, best_score = scored[0]
        alternatives = [m for m, s in scored[1:4]]  # Top 3 alternatives
        
        # Calculate confidence
        confidence = self._calculate_confidence(scored)
        
        # Record selection
        self._selection_history.append({
            "task_type": task_type.value,
            "selected_model": best_model.id,
            "score": best_score,
            "timestamp": datetime.now().isoformat()
        })
        
        return ModelSelection(
            model=best_model,
            score=best_score,
            reasoning=self._generate_reasoning(best_model, task_type, best_score),
            alternatives=alternatives,
            confidence=confidence
        )
    
    def _score_model(self, 
                    model: ModelInfo, 
                    task_type: TaskType,
                    task_description: str,
                    preferred_models: List[str]) -> float:
        """Score a model for a given task."""
        score = 0.0
        
        # Capability match (40%)
        capability_score = model.capabilities.matches_task(task_type)
        score += 0.40 * capability_score
        
        # Preference match (20%)
        if preferred_models and model.id in preferred_models:
            pref_index = preferred_models.index(model.id)
            preference_score = 1.0 - (pref_index * 0.3)  # First choice = 1.0, second = 0.7, etc.
            score += 0.20 * preference_score
        
        # Performance (20%) - incorporates Thompson Sampling bonus when available
        if model.total_uses > 0:
            perf_score = model.success_rate * (1.0 - min(model.avg_latency_ms / 60000, 1.0))
            score += 0.20 * perf_score
        else:
            score += 0.15  # Slightly optimistic prior for new models (exploration bonus)

        # Thompson Sampling bonus (up to +0.10) - uses learned exploration/exploitation data
        try:
            from vetinari.learning.model_selector import get_thompson_selector
            ts = get_thompson_selector()
            task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
            arm = ts._arms.get(f"{model.id}:{task_type_str}")
            if arm is not None and (arm.alpha + arm.beta) >= 2:
                # Use the arm's mean as a quality signal (0-1 range)
                # Beta(1,1) prior mean=0.5 gives new arms a fair exploration bonus
                ts_bonus = arm.mean * 0.10
                score += ts_bonus
        except Exception:
            logger.debug("Failed to compute Thompson Sampling bonus for model %s", model.id, exc_info=True)

        # Provider preference (10%)
        if self.prefer_local:
            if model.provider == ModelProvider.LOCAL:
                score += 0.10
            elif model.provider == ModelProvider.OTHER:
                score += 0.05
            # Cloud models get 0
        else:
            # No preference, give all a base score
            score += 0.10
        
        # Context length fit (10%)
        if model.context_length >= 8192:
            score += 0.10
        elif model.context_length >= 4096:
            score += 0.05
        
        return score
    
    def _calculate_confidence(self, scored: List[tuple]) -> float:
        """Calculate confidence in the selection based on score distribution."""
        if len(scored) < 2:
            return 0.5
        
        best_score = scored[0][1]
        second_score = scored[1][1]
        
        if best_score == 0:
            return 0.1
        
        # Higher gap = more confidence
        gap = best_score - second_score
        confidence = min(1.0, gap * 2 + 0.3)
        
        return confidence
    
    def _generate_reasoning(self, model: ModelInfo, task_type: TaskType, score: float) -> str:
        """Generate human-readable reasoning for model selection."""
        reasons = []
        
        # Capability match
        caps = model.capabilities
        if task_type == TaskType.CODING and caps.code_gen:
            reasons.append("excellent code generation")
        elif task_type == TaskType.REASONING and caps.reasoning:
            reasons.append("strong reasoning capabilities")
        elif task_type == TaskType.DOCUMENTATION and caps.docs:
            reasons.append("good documentation skills")
        
        # Performance
        if model.total_uses > 10:
            reasons.append(f"proven track record ({model.total_uses} uses)")
        if model.avg_latency_ms > 0 and model.avg_latency_ms < 5000:
            reasons.append(f"fast response ({model.avg_latency_ms:.0f}ms)")
        
        # Provider
        if model.provider == ModelProvider.LOCAL:
            reasons.append("local model (no API costs)")
        
        if not reasons:
            reasons.append("best available match")
        
        return f"Selected {model.id}: {', '.join(reasons)}"
    
    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """Get a specific model by ID."""
        return self.models.get(model_id)
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get all available models."""
        return [m for m in self.models.values() if m.is_available]
    
    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """Get all models with a specific capability."""
        results = []
        for model in self.models.values():
            if not model.is_available:
                continue
            caps = model.capabilities
            if capability == "code_gen" and caps.code_gen:
                results.append(model)
            elif capability == "reasoning" and caps.reasoning:
                results.append(model)
            elif capability == "docs" and caps.docs:
                results.append(model)
        return results
    
    def check_model_health(self, model_id: str) -> bool:
        """Check if a model is healthy."""
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        # Use callback if available
        if self._health_check_callback:
            try:
                return self._health_check_callback(model_id)
            except Exception as e:
                logger.error("Health check failed for %s: %s", model_id, e)
        
        # Basic checks
        if model.avg_latency_ms > self.max_latency_ms * 2:
            return False
        
        if model.success_rate < 0.5:
            return False
        
        return True
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_selections = len(self._selection_history)
        
        model_counts = {}
        for sel in self._selection_history:
            model_id = sel["selected_model"]
            model_counts[model_id] = model_counts.get(model_id, 0) + 1
        
        return {
            "total_selections": total_selections,
            "models_used": model_counts,
            "available_models": len(self.get_available_models()),
            "total_models": len(self.models)
        }


# Global router instance
_model_router: Optional[DynamicModelRouter] = None


def get_model_router() -> DynamicModelRouter:
    """Get or create the global model router."""
    global _model_router
    if _model_router is None:
        _model_router = DynamicModelRouter()
    return _model_router


def init_model_router(prefer_local: bool = True, **kwargs) -> DynamicModelRouter:
    """Initialize a new model router."""
    global _model_router
    _model_router = DynamicModelRouter(prefer_local=prefer_local, **kwargs)
    return _model_router


# Helper function to infer task type from description
def infer_task_type(description: str) -> TaskType:
    """Infer task type from description using keyword matching.

    Order matters — more specific categories are checked before general ones
    so that "security audit" matches SECURITY_AUDIT, not just ANALYSIS.
    """
    desc_lower = description.lower()

    # Phase 7: check specific categories first (before general ones)
    if any(kw in desc_lower for kw in ["security", "audit", "vulnerability", "pentest", "cve", "owasp"]):
        return TaskType.SECURITY_AUDIT
    elif any(kw in desc_lower for kw in ["deploy", "ci/cd", "docker", "kubernetes", "pipeline", "devops", "terraform"]):
        return TaskType.DEVOPS
    elif any(kw in desc_lower for kw in ["logo", "icon", "mockup", "diagram", "image", "illustration"]):
        return TaskType.IMAGE_GENERATION
    elif any(kw in desc_lower for kw in ["cost", "budget", "pricing", "expense", "billing"]):
        return TaskType.COST_ANALYSIS
    elif any(kw in desc_lower for kw in ["specification", "spec", "requirements", "acceptance criteria"]):
        return TaskType.SPECIFICATION
    elif any(kw in desc_lower for kw in ["story", "poem", "fiction", "narrative", "campaign", "creative writ"]):
        return TaskType.CREATIVE_WRITING
    # Original categories
    elif any(kw in desc_lower for kw in ["plan", "strategy", "workflow", "design", "architect"]):
        return TaskType.PLANNING
    elif any(kw in desc_lower for kw in ["analyze", "analysis", "research", "investigate"]):
        return TaskType.ANALYSIS
    elif any(kw in desc_lower for kw in ["code", "implement", "build", "create", "program", "function", "class"]):
        return TaskType.CODING
    elif any(kw in desc_lower for kw in ["review", "refactor", "improve", "optimize"]):
        return TaskType.CODE_REVIEW
    elif any(kw in desc_lower for kw in ["test", "testing", "assert"]):
        return TaskType.TESTING
    elif any(kw in desc_lower for kw in ["document", "readme", "docs", "comment", "explain"]):
        return TaskType.DOCUMENTATION
    elif any(kw in desc_lower for kw in ["reason", "logic", "solve", "problem", "math"]):
        return TaskType.REASONING
    elif any(kw in desc_lower for kw in ["creative", "write", "article"]):
        return TaskType.CREATIVE
    elif any(kw in desc_lower for kw in ["data", "process", "extract", "transform", "etl", "database", "schema", "sql"]):
        return TaskType.DATA_PROCESSING
    elif any(kw in desc_lower for kw in ["search", "find", "look", "query", "web"]):
        return TaskType.WEB_SEARCH
    elif any(kw in desc_lower for kw in ["summarize", "summary", "condense"]):
        return TaskType.SUMMARIZATION
    elif any(kw in desc_lower for kw in ["translate", "translation", "convert"]):
        return TaskType.TRANSLATION
    else:
        return TaskType.GENERAL


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Test the router
    router = DynamicModelRouter(prefer_local=True)
    
    # Register test models
    models = [
        {
            "id": "llama-3-8b",
            "name": "Llama 3 8B",
            "capabilities": ["code_gen", "chat", "reasoning"],
            "context_len": 8192,
            "memory_gb": 8,
            "tags": ["local", "llama"]
        },
        {
            "id": "codellama-7b",
            "name": "Code Llama 7B",
            "capabilities": ["code_gen", "chat"],
            "context_len": 4096,
            "memory_gb": 7,
            "tags": ["local", "code"]
        },
        {
            "id": "mistral-7b",
            "name": "Mistral 7B",
            "capabilities": ["chat", "reasoning"],
            "context_len": 4096,
            "memory_gb": 7,
            "tags": ["local"]
        }
    ]
    
    router.register_models_from_pool(models)
    
    # Test selection
    selection = router.select_model(TaskType.CODING, "Write a Python function")
    print(f"\nSelected: {selection.model.id}")
    print(f"Reasoning: {selection.reasoning}")
    print(f"Confidence: {selection.confidence:.2f}")
    print(f"Alternatives: {[a.id for a in selection.alternatives]}")
