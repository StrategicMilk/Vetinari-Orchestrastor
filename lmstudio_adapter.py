import requests
import json
import time
from typing import Dict, Any

class LMStudioAdapter:
    def __init__(self, host: str):
        self.host = host.rstrip("/")
        self.session = requests.Session()
        self.max_retries = 2

    def _post(self, endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        url = endpoint if endpoint.startswith("http") else self.host + endpoint
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == self.max_retries:
                    return {"error": str(e), "output": ""}
                time.sleep(2 ** attempt)
        return {"error": "unknown", "output": ""}

    def chat(self, model_id: str, system_prompt: str, input_text: str, timeout: int = 60) -> Dict[str, Any]:
        endpoint = f"{self.host}/api/v1/chat"
        payload = {
            "model": model_id,
            "system_prompt": system_prompt,
            "input": input_text
        }
        start = time.time()
        resp = self._post(endpoint, payload, timeout=timeout)
        latency_ms = int((time.time() - start) * 1000)
        
        if "output" in resp:
            return {
                "output": resp["output"],
                "latency_ms": latency_ms,
                "tokens_used": resp.get("tokens_used", 0),
                "status": "ok",
                "error": resp.get("error")
            }
        else:
            return {
                "output": "",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "status": "error",
                "error": resp.get("error", "unknown")
            }

    def infer(self, model_endpoint: str, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        # Fallback for direct endpoint calls if needed
        payload = {"prompt": prompt}
        start = time.time()
        resp = self._post(model_endpoint, payload, timeout=timeout)
        latency_ms = int((time.time() - start) * 1000)
        
        if "output" in resp:
            return {
                "output": resp["output"],
                "latency_ms": latency_ms,
                "tokens_used": resp.get("tokens_used", 0),
                "status": "ok",
                "error": resp.get("error")
            }
        else:
            return {
                "output": "",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "status": "error",
                "error": resp.get("error", "unknown")
            }