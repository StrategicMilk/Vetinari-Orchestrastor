import requests
import yaml
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    def __init__(self, config: dict, host: str = "http://100.78.30.7:1234", api_token: Optional[str] = None, memory_budget_gb: int = None):
        self.config = config
        self.host = host
        self.api_token = api_token
        # Use config memory_budget if not explicitly provided
        if memory_budget_gb is None:
            memory_budget_gb = config.get("memory_budget_gb", config.get("discovery_filters", {}).get("max_model_memory_gb", 32))
        self.memory_budget_gb = memory_budget_gb
        self.models = []
        self.discovered = []
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        
        # Retry policy configuration
        self._discovery_failed = False
        self._fallback_active = False
        self._last_discovery_error = None
        self._discovery_retry_count = 0
        self._max_discovery_retries = int(os.environ.get("VETINARI_MODEL_DISCOVERY_RETRIES", "5"))
        self._discovery_retry_delay_base = float(os.environ.get("VETINARI_MODEL_DISCOVERY_RETRY_DELAY", "1.0"))

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
        Falls back to static models if discovery fails after max retries.
        """
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
                logging.debug(f"[Model Discovery] Attempt {self._discovery_retry_count}/{self._max_discovery_retries} at {models_endpoint}")
                
                resp = self.session.get(models_endpoint, timeout=10)
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
                        logging.info(f"[Model Discovery] Skipping {model_id} - exceeds memory budget ({mem}GB > {self.memory_budget_gb}GB)")
                        continue
                    
                    model = {
                        "id": model_id,
                        "name": model_id,  # Use ID as name
                        "endpoint": f"{self.host}/api/v1/chat",
                        "capabilities": ["code_gen", "docs", "chat"],  # Default capabilities
                        "context_len": 2048,
                        "memory_gb": mem if mem > 0 else 2,  # Default to 2GB if unknown
                        "version": ""
                    }
                    self.models.append(model)
                    discovered_count += 1
                
                logging.info(f"[Model Discovery] SUCCESS: Discovered {discovered_count} models from {self.host}")
                self._discovery_failed = False
                self._fallback_active = False
                break  # Success, exit retry loop
                
            except requests.exceptions.Timeout:
                error_msg = f"Model discovery timeout on attempt {self._discovery_retry_count}/{self._max_discovery_retries}"
                logging.warning(f"[Model Discovery] {error_msg}")
                self._last_discovery_error = error_msg
                self._discovery_failed = True
                
                # Calculate backoff delay
                if attempt < self._max_discovery_retries - 1:
                    delay = self._discovery_retry_delay_base * (2 ** attempt)  # Exponential backoff
                    delay = min(delay, 30)  # Cap at 30 seconds
                    logging.info(f"[Model Discovery] Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error during model discovery (attempt {self._discovery_retry_count}/{self._max_discovery_retries}): {str(e)}"
                logging.warning(f"[Model Discovery] {error_msg}")
                self._last_discovery_error = error_msg
                self._discovery_failed = True
                
                # Calculate backoff delay
                if attempt < self._max_discovery_retries - 1:
                    delay = self._discovery_retry_delay_base * (2 ** attempt)
                    delay = min(delay, 30)
                    logging.info(f"[Model Discovery] Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    
            except Exception as e:
                error_msg = f"Model discovery failed (attempt {self._discovery_retry_count}/{self._max_discovery_retries}): {str(e)}"
                logging.warning(f"[Model Discovery] {error_msg}")
                self._last_discovery_error = error_msg
                self._discovery_failed = True
                
                # For other errors, don't retry
                break
        
        # Fallback: Always include static models from config
        if self._discovery_failed and len(self.models) == 0:
            logging.warning(f"[Model Discovery] FAILED after {self._discovery_retry_count} attempts. Falling back to static models.")
            self._fallback_active = True
        
        for m in static_models:
            if not any(existing.get('name') == m.get('name') for existing in self.models):
                self.models.append(m)
        
        # Log final state
        if self._fallback_active:
            logging.info(f"[Model Discovery] Using {len(self.models)} models from fallback (static config)")
        else:
            logging.info(f"[Model Discovery] Available models: {len(self.models)}")
    
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

        reliability = 0.8  # placeholder; could use history caching

        ctx_len = model.get("context_len", 2048)
        data_size = sum(len(str(x)) for x in task.get("inputs", []))
        context_fit = 1.0 if data_size <= ctx_len else max(0.0, ctx_len / max(1, data_size))

        resource_load = 0.5  # placeholder

        w_cap = 0.35
        w_lat = 0.25
        w_rel = 0.15
        w_ctx = 0.15
        w_res = 0.10

        score = (
            w_cap * cap_match +
            w_lat * latency_norm +
            w_rel * reliability +
            w_ctx * context_fit +
            w_res * (1.0 - resource_load)
        )
        return score

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
                logging.info(f"Cloud model available: {provider['name']}")
            else:
                logging.debug(f"Cloud provider {provider['name']} not configured (missing {provider['env_token']})")
        
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
