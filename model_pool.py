import requests
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
from vetinari.adapters import AdapterRegistry, ProviderConfig, ProviderType, ModelInfo

class ModelPool:
    def __init__(self, config: dict):
        self.config = config
        self.models = []
        self.discovered = []
        self.adapter_registry = AdapterRegistry()
        self._initialize_adapters()

    def _initialize_adapters(self):
        """Initialize provider adapters from configuration."""
        providers_config = self.config.get("provider_adapters", {})
        
        # Always add LM Studio by default if not explicitly configured
        if "lm_studio" not in providers_config:
            lm_studio_endpoint = self.config.get("host", "http://10.0.0.96:1234")
            providers_config["lm_studio"] = {
                "provider_type": "lm_studio",
                "endpoint": lm_studio_endpoint,
                "enabled": True,
            }
        
        # Initialize adapters from provider config
        for adapter_name, adapter_config in providers_config.items():
            if not adapter_config.get("enabled", True):
                continue
            
            try:
                provider_type_str = adapter_config.get("provider_type", "").upper().replace("-", "_")
                provider_type = ProviderType[provider_type_str]
                
                config = ProviderConfig(
                    provider_type=provider_type,
                    name=adapter_name,
                    endpoint=adapter_config.get("endpoint", ""),
                    api_key=adapter_config.get("api_key"),
                    max_retries=adapter_config.get("max_retries", 3),
                    timeout_seconds=adapter_config.get("timeout_seconds", 120),
                    memory_budget_gb=adapter_config.get("memory_budget_gb", 32),
                    extra_config=adapter_config.get("extra_config", {}),
                )
                
                adapter = self.adapter_registry.create_adapter(config, instance_name=adapter_name)
                logging.info(f"[ModelPool] Initialized adapter: {adapter_name} ({provider_type_str})")
            except Exception as e:
                logging.warning(f"[ModelPool] Failed to initialize adapter {adapter_name}: {e}")

    def discover_models(self):
        """Discover models from all configured provider adapters."""
        # Load static models from config first
        static_models = self.config.get("models", [])
        self.models = []
        
        # Convert static models to ModelInfo format for consistency
        for m in static_models:
            try:
                model_info = ModelInfo(
                    id=m.get("id", ""),
                    name=m.get("name", ""),
                    provider="static",
                    endpoint=m.get("endpoint", ""),
                    capabilities=m.get("capabilities", []),
                    context_len=m.get("context_len", 2048),
                    memory_gb=m.get("memory_gb", 8),
                    version=m.get("version", "unknown"),
                    tags=["static", "configured"],
                )
                self.models.append(model_info)
                logging.info(f"[ModelPool] Loaded static model: {model_info.name}")
            except Exception as e:
                logging.warning(f"[ModelPool] Failed to load static model: {e}")
        
        # Discover models from all active adapters
        try:
            adapter_discovery = self.adapter_registry.discover_all_models()
            for adapter_name, models in adapter_discovery.items():
                self.models.extend(models)
                logging.info(f"[ModelPool] Discovered {len(models)} models from {adapter_name}")
        except Exception as e:
            logging.warning(f"[ModelPool] Adapter discovery failed: {e}")
        
        # Apply discovery filters from config
        self._apply_discovery_filters()
        logging.info(f"[ModelPool] Total models available: {len(self.models)}")

    def _apply_discovery_filters(self):
        """Apply configured discovery filters to discovered models."""
        filters = self.config.get("discovery_filters", {})
        
        if not filters.get("allow_unverified_models", True):
            self.models = [m for m in self.models if m.provider in ["openai", "anthropic", "cohere", "gemini", "lm_studio"]]
        
        max_context = filters.get("max_context_len", 32768)
        self.models = [m for m in self.models if m.context_len <= max_context]
        
        min_capabilities = filters.get("min_capabilities", [])
        if min_capabilities:
            self.models = [m for m in self.models if all(cap in m.capabilities for cap in min_capabilities)]
        
        max_memory = filters.get("max_model_memory_gb", 96)
        self.models = [m for m in self.models if m.memory_gb <= max_memory]


    def assign_tasks_to_models(self, config: dict):
        """Assign tasks to best-matching models using adapter registry scoring."""
        tasks = config.get("tasks", [])
        for t in tasks:
            best_adapter = None
            best_model = None
            best_score = -1.0
            
            task_requirements = {
                "required_capabilities": t.get("inputs", []),
                "input_tokens": sum(len(str(x)) for x in t.get("inputs", [])),
            }
            
            # Try to find best model using adapter registry if available
            if self.adapter_registry._instances:
                best_adapter, best_model = self.adapter_registry.find_best_model(task_requirements)
            
            # Fall back to scoring local models
            if not best_model:
                for m in self.models:
                    score = self._score_task_model(t, m)
                    if score > best_score:
                        best_score = score
                        best_model = m
            
            if best_model:
                if isinstance(best_model, ModelInfo):
                    t["assigned_model_id"] = best_model.id
                else:
                    t["assigned_model_id"] = best_model.get("id")
            else:
                t["assigned_model_id"] = None
        
        logging.info("Task-to-model assignments completed.")

    def _score_task_model(self, task: dict, model) -> float:
        """Score a model for a task. Works with both dict and ModelInfo objects."""
        # Handle both dict and ModelInfo objects
        if isinstance(model, ModelInfo):
            capabilities = model.capabilities
            latency = model.latency_estimate_ms
            context_len = model.context_len
        else:
            capabilities = model.get("capabilities", [])
            latency = model.get("latency_estimate", 1000)
            context_len = model.get("context_len", 2048)
        
        required = set(task.get("inputs", []))
        provided = set(capabilities)
        cap_match = len(required & provided) / max(len(required), 1)

        latency_norm = max(0.0, 1.0 - (latency / 2000.0))

        reliability = 0.8  # placeholder; could use history caching

        data_size = sum(len(str(x)) for x in task.get("inputs", []))
        context_fit = 1.0 if data_size <= context_len else max(0.0, context_len / max(1, data_size))

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