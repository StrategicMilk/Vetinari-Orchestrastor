import requests
import yaml
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


CLOUD_PROVIDERS = {
    "huggingface_inference": {
        "name": "HuggingFace Inference API",
        "endpoint": "https://api-inference.huggingface.co/models",
        "free_tier": True,
        "context_len": 4096,
        "memory_gb": 0,
        "tags": ["cloud", "free", "inference"],
        "env_token": "HF_HUB_TOKEN"
    },
    "replicate": {
        "name": "Replicate",
        "endpoint": "https://api.replicate.com/v1",
        "free_tier": True,
        "context_len": 8192,
        "memory_gb": 0,
        "tags": ["cloud", "free", "replicate"],
        "env_token": "REPLICATE_API_TOKEN"
    },
    "claude": {
        "name": "Claude (Anthropic)",
        "endpoint": "https://api.anthropic.com/v1",
        "free_tier": True,
        "context_len": 200000,
        "memory_gb": 0,
        "tags": ["cloud", "free", "reasoning", "claude"],
        "env_token": "CLAUDE_API_KEY"
    },
    "gemini": {
        "name": "Gemini (Google)",
        "endpoint": "https://generativelanguage.googleapis.com/v1",
        "free_tier": True,
        "context_len": 32768,
        "memory_gb": 0,
        "tags": ["cloud", "free", "gemini", "google"],
        "env_token": "GEMINI_API_KEY"
    }
}


class ModelPool:
    def __init__(self, config: dict, host: str = "http://localhost:1234", api_token: Optional[str] = None, memory_budget_gb: int = None):
        self.config = config
        self.host = host
        self.api_token = api_token
        # Use config memory_budget if not explicitly provided
        if memory_budget_gb is None:
            memory_budget_gb = config.get("memory_budget_gb", config.get("discovery_filters", {}).get("max_model_memory_gb", 32))
        self.memory_budget_gb = memory_budget_gb
        self.models = []
        self.discovered = []
        self._last_known_good: list = []   # Preserved across failed discoveries
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        
        # Retry policy — kept short so the UI never blocks for long
        self._discovery_failed = False
        self._fallback_active = False
        self._last_discovery_error = None
        self._discovery_retry_count = 0
        self._max_discovery_retries = int(os.environ.get("VETINARI_MODEL_DISCOVERY_RETRIES", "2"))
        self._discovery_retry_delay_base = float(os.environ.get("VETINARI_MODEL_DISCOVERY_RETRY_DELAY", "0.5"))

    def set_api_token(self, api_token: Optional[str]):
        """Update the API token for authentication."""
        self.api_token = api_token
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        elif "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]

    def discover_models(self):
        """
        Discover models from LM Studio with exponential backoff retry logic.
        Falls back to last-known-good results (then static config) if discovery fails.
        """
        # Preserve last known good before resetting
        _previous_models = list(self.models) if self.models else list(self._last_known_good)

        # Reset state on each discovery attempt
        self._discovery_failed = False
        self._fallback_active = False
        self._discovery_retry_count = 0
        self.models = []
        
        # Load static models first (always available)
        static_models = self.config.get("models", [])
        
        # Attempt discovery with retry logic
        for attempt in range(self._max_discovery_retries):
            self._discovery_retry_count = attempt + 1
            try:
                # Try to get models from LM Studio /v1/models endpoint (with auth if set)
                models_endpoint = f"{self.host}/v1/models"
                logging.debug("[Model Discovery] Attempt %s/%s at %s", self._discovery_retry_count, self._max_discovery_retries, models_endpoint)
                
                resp = self.session.get(models_endpoint, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                
                # Handle both dict and list response formats
                if isinstance(data, dict):
                    if "data" in data:
                        data = data["data"]
                    elif "models" in data:
                        data = data["models"]
                if not isinstance(data, list):
                    data = []
                
                # Process discovered models
                discovered_count = 0
                for m in data:
                    # Get model ID and name
                    model_id = m.get("id", "")
                    if not model_id:
                        continue
                        
                    # Filter by memory_gb <= memory_budget_gb
                    mem = m.get("memory_gb", 0)
                    
                    # Skip models that exceed memory budget
                    if mem > self.memory_budget_gb:
                        logging.info("[Model Discovery] Skipping %s - exceeds memory budget (%sGB > %sGB)", model_id, mem, self.memory_budget_gb)
                        continue
                    
                    model = {
                        "id": model_id,
                        "name": model_id,  # Use ID as name
                        "endpoint": f"{self.host}/v1/chat/completions",
                        "capabilities": ["code_gen", "docs", "chat"],  # Default capabilities
                        "context_len": 2048,
                        "memory_gb": mem if mem > 0 else 2,  # Default to 2GB if unknown
                        "version": ""
                    }
                    self.models.append(model)
                    discovered_count += 1
                
                logging.info("[Model Discovery] SUCCESS: Discovered %s models from %s", discovered_count, self.host)
                self._discovery_failed = False
                self._fallback_active = False
                self._last_known_good = list(self.models)  # Save for next failure
                break  # Success, exit retry loop
                
            except requests.exceptions.Timeout:
                error_msg = f"Model discovery timeout on attempt {self._discovery_retry_count}/{self._max_discovery_retries}"
                logging.warning("[Model Discovery] %s", error_msg)
                self._last_discovery_error = error_msg
                self._discovery_failed = True

                # Calculate backoff delay
                if attempt < self._max_discovery_retries - 1:
                    delay = self._discovery_retry_delay_base * (2 ** attempt)  # Exponential backoff
                    delay = min(delay, 30)  # Cap at 30 seconds
                    logging.info("[Model Discovery] Retrying in %.1fs...", delay)
                    time.sleep(delay)
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error during model discovery (attempt {self._discovery_retry_count}/{self._max_discovery_retries}): {str(e)}"
                logging.warning("[Model Discovery] %s", error_msg)
                self._last_discovery_error = error_msg
                self._discovery_failed = True

                # Calculate backoff delay
                if attempt < self._max_discovery_retries - 1:
                    delay = self._discovery_retry_delay_base * (2 ** attempt)
                    delay = min(delay, 30)
                    logging.info("[Model Discovery] Retrying in %.1fs...", delay)
                    time.sleep(delay)
                    
            except Exception as e:
                error_msg = f"Model discovery failed (attempt {self._discovery_retry_count}/{self._max_discovery_retries}): {str(e)}"
                logging.warning("[Model Discovery] %s", error_msg)
                self._last_discovery_error = error_msg
                self._discovery_failed = True

                # For other errors, don't retry
                break
        
        # Fallback order: last-known-good → static config models
        if self._discovery_failed and len(self.models) == 0:
            if _previous_models:
                logging.warning(
                    "[Model Discovery] FAILED after %s attempts. "
                    "Using last-known-good (%s models).",
                    self._discovery_retry_count, len(_previous_models)
                )
                self.models = list(_previous_models)
                self._fallback_active = True
            else:
                logging.warning(
                    "[Model Discovery] FAILED after %s attempts. "
                    "Falling back to static config models.",
                    self._discovery_retry_count
                )
                self._fallback_active = True

        # Always merge in static config models (deduped by name)
        for m in static_models:
            if not any(existing.get('name') == m.get('name') for existing in self.models):
                self.models.append(m)

        # Log final state
        if self._fallback_active:
            logging.info("[Model Discovery] Using %s models (fallback active)", len(self.models))
        else:
            logging.info("[Model Discovery] Available models: %s", len(self.models))
    
    def get_discovery_health(self) -> Dict[str, Any]:
        """Get health information about model discovery."""
        return {
            "discovery_failed": self._discovery_failed,
            "fallback_active": self._fallback_active,
            "last_error": self._last_discovery_error,
            "retry_count": self._discovery_retry_count,
            "models_available": len(self.models),
            "max_retries": self._max_discovery_retries,
            "retry_delay_base": self._discovery_retry_delay_base
        }

    def assign_tasks_to_models(self, config: dict):
        tasks = config.get("tasks", [])
        for t in tasks:
            best = None
            best_score = -1.0
            for m in self.models:
                score = self._score_task_model(t, m)
                if score > best_score:
                    best = m
                    best_score = score
            t["assigned_model_id"] = best["id"] if best else None
        logging.info("Task-to-model assignments completed.")

    def _score_task_model(self, task: dict, model: dict) -> float:
        required = set(task.get("inputs", []))
        provided = set(model.get("capabilities", []))
        cap_match = len(required & provided) / max(len(required), 1)

        latency = model.get("latency_estimate", 1000)
        latency_norm = max(0.0, 1.0 - (latency / 2000.0))

        # Phase 5: Use real reliability from cost tracker data instead of placeholder
        reliability = self._get_model_reliability(model.get("id", ""))

        ctx_len = model.get("context_len", 2048)
        data_size = sum(len(str(x)) for x in task.get("inputs", []))
        context_fit = 1.0 if data_size <= ctx_len else max(0.0, ctx_len / max(1, data_size))

        # Phase 5: Use cost-efficiency score from analytics
        cost_efficiency = self._get_cost_efficiency(model.get("id", ""))

        w_cap = 0.30
        w_lat = 0.20
        w_rel = 0.15
        w_ctx = 0.10
        w_cost = 0.15
        w_res = 0.10

        # Estimate resource load from model context usage
        # Models using more of their context window are under heavier load
        resource_load = min(1.0, data_size / max(1, ctx_len)) if ctx_len > 0 else 0.5

        score = (
            w_cap * cap_match +
            w_lat * latency_norm +
            w_rel * reliability +
            w_ctx * context_fit +
            w_cost * cost_efficiency +
            w_res * (1.0 - resource_load)
        )
        return score

    def _get_model_reliability(self, model_id: str) -> float:
        """Get per-model reliability from feedback loop or SLA tracker."""
        # Try per-model performance data first
        try:
            from vetinari.dynamic_model_router import get_dynamic_model_router
            router = get_dynamic_model_router()
            cache_key = f"{model_id}:general"
            perf = router.get_performance_cache(cache_key)
            if perf and "success_rate" in perf:
                return min(1.0, perf["success_rate"])
        except Exception:
            logger.debug("Failed to get model reliability from dynamic router for %s", model_id, exc_info=True)
        # Fall back to global SLA data
        try:
            from vetinari.analytics.sla import get_sla_tracker
            report = get_sla_tracker().get_report("success-rate")
            if report and report.total_samples > 0:
                return min(1.0, report.current_value / 100.0)
        except Exception:
            logger.debug("Failed to get model reliability from SLA tracker for %s", model_id, exc_info=True)
        return 0.8  # conservative prior for unknown models

    def _get_cost_efficiency(self, model_id: str) -> float:
        """Get cost efficiency score (1.0 = free/cheapest, 0.0 = most expensive)."""
        try:
            from vetinari.analytics.cost import get_cost_tracker
            report = get_cost_tracker().get_report()
            if report.total_requests > 0 and report.by_model:
                model_costs = report.by_model
                max_cost = max(model_costs.values()) if model_costs else 1.0
                if max_cost > 0:
                    # Find this model's cost; lower cost = higher score
                    for key, cost in model_costs.items():
                        if model_id in key:
                            return max(0.0, 1.0 - (cost / max_cost))
            return 1.0  # No cost data = assume free (local model)
        except Exception:
            return 1.0

    def get_cloud_models(self) -> List[Dict]:
        """Get available cloud models based on environment tokens."""
        cloud_models = []
        
        for provider_id, provider in CLOUD_PROVIDERS.items():
            token = os.environ.get(provider["env_token"])
            
            if token:
                model = {
                    "id": f"cloud:{provider_id}",
                    "name": provider["name"],
                    "provider": provider_id,
                    "endpoint": provider["endpoint"],
                    "capabilities": provider["tags"],
                    "context_len": provider["context_len"],
                    "memory_gb": provider["memory_gb"],
                    "free_tier": provider["free_tier"],
                    "version": "latest"
                }
                cloud_models.append(model)
                logging.info("Cloud model available: %s", provider['name'])
            else:
                logging.debug("Cloud provider %s not configured (missing %s)", provider['name'], provider['env_token'])
        
        return cloud_models

    def get_all_available_models(self) -> List[Dict]:
        """Get all available models (local + cloud)."""
        local_models = self.models.copy()
        cloud_models = self.get_cloud_models()
        return local_models + cloud_models

    @staticmethod
    def get_cloud_provider_health() -> Dict[str, bool]:
        """Check health/status of cloud providers based on token presence."""
        health = {}
        for provider_id, provider in CLOUD_PROVIDERS.items():
            token = os.environ.get(provider["env_token"])
            health[provider_id] = {
                "available": bool(token),
                "name": provider["name"],
                "has_token": bool(token)
            }
        return health

    def list_models(self) -> List[Dict]:
        """Return list of discovered models (alias for discover_models)."""
        return self.models.copy()
