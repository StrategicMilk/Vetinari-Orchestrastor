import requests
import yaml
import logging
from pathlib import Path
from typing import Dict, List

class ModelPool:
    def __init__(self, config: dict):
        self.config = config
        self.models = []
        self.discovered = []

    def discover_models(self):
        registry = self.config.get("discovery_source", "")
        
        # First, load static models from config
        static_models = self.config.get("models", [])
        self.models = []
        
        # Try to discover additional models from LM Studio registry
        try:
            registry_endpoint = "http://10.0.0.96:1234/v1/models"
            resp = requests.get(registry_endpoint, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            for m in data:
                # Filter by memory_gb <= 96
                mem = m.get("memory_gb", 0)
                if mem <= 96:
                    model = {
                        "id": m.get("id", ""),
                        "name": m.get("name", ""),
                        "endpoint": f"{self.config.get('host', 'http://10.0.0.96:1234')}/api/v1/chat",
                        "capabilities": m.get("capabilities", []),
                        "context_len": m.get("context_len", 2048),
                        "memory_gb": mem,
                        "version": m.get("version", "")
                    }
                    self.models.append(model)
                    logging.info(f"Discovered model: {model['name']}")
        except Exception as e:
            logging.warning(f"Model discovery failed: {e}")
            # Fall back to static models only
            self.models = static_models

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