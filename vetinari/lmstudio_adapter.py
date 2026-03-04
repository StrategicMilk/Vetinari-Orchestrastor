import requests
import json
import time
from typing import Dict, Any, Optional


class LMStudioAdapter:
    def __init__(self, host: str, api_token: Optional[str] = None):
        self.host = host.rstrip("/")
        self.session = requests.Session()
        self.max_retries = 2
        self.api_token = api_token
        
        # Set default headers
        if self.api_token:
            self.session.headers.update({"Authorization": f"Bearer {self.api_token}"})

    def set_api_token(self, api_token: Optional[str]):
        """Update the API token for authentication."""
        self.api_token = api_token
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        elif "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]

    def _post(self, endpoint: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        url = endpoint if endpoint.startswith("http") else self.host + endpoint
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=timeout, headers=headers)
                # Try to parse as JSON
                try:
                    data = resp.json()
                    return data
                except json.JSONDecodeError:
                    # Non-JSON response - return safe envelope
                    return {
                        "status": "partial",
                        "raw_output": resp.text[:5000],  # Limit raw output size
                        "error": f"Non-JSON response (status {resp.status_code})",
                        "output": ""
                    }
            except Exception as e:
                if attempt == self.max_retries:
                    return {"status": "error", "error": str(e), "output": ""}
                time.sleep(2 ** attempt)
        return {"status": "error", "error": "unknown", "output": ""}

    def _get(self, endpoint: str, timeout: int = 10) -> Optional[Dict]:
        """Make GET request with optional auth."""
        url = endpoint if endpoint.startswith("http") else self.host + endpoint
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        try:
            resp = self.session.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except json.JSONDecodeError:
            # Non-JSON response
            return {"status": "partial", "raw_output": resp.text[:5000], "error": "Non-JSON response"}
        except Exception as e:
            return None

    def chat(self, model_id: str, system_prompt: str, input_text: str, timeout: int = 120) -> Dict[str, Any]:
        """Call LM Studio chat API using OpenAI-compatible format."""
        endpoint = f"{self.host}/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": input_text}
            ],
            "temperature": 0.3,
            "stream": False
        }
        
        start = time.time()
        resp = self._post(endpoint, payload, timeout=timeout)
        latency_ms = int((time.time() - start) * 1000)
        
        return self._parse_response(resp, latency_ms)
    
    def _parse_response(self, resp: Dict[str, Any], latency_ms: int) -> Dict[str, Any]:
        """Parse the response from LM Studio with safe envelope handling."""
        # Check for envelope status (from non-JSON fallback)
        if resp.get("status") == "partial":
            return {
                "output": resp.get("raw_output", ""),
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "status": "partial",
                "error": resp.get("error", "Partial response"),
                "raw_output": resp.get("raw_output", "")
            }
        
        # Check for error in response
        if "error" in resp:
            error_msg = resp.get("error", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            return {
                "output": "",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "status": "error",
                "error": error_msg
            }
        
        # OpenAI-compatible response: choices[0].message.content
        if "choices" in resp and resp["choices"]:
            choice = resp["choices"][0]
            output_text = ""
            if "message" in choice:
                output_text = choice["message"].get("content", "")
            elif "text" in choice:
                output_text = choice["text"]
            tokens_used = 0
            if "usage" in resp:
                tokens_used = resp["usage"].get("total_tokens", 0)
            return {
                "output": output_text,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "status": "ok",
                "error": None
            }

        # Legacy "output" key (older LM Studio versions)
        if "output" in resp:
            output_text = resp["output"]
            if isinstance(output_text, list) and output_text:
                first = output_text[0]
                output_text = first.get("content", str(first)) if isinstance(first, dict) else str(first)
            elif not isinstance(output_text, str):
                output_text = str(output_text)
            return {
                "output": output_text,
                "latency_ms": latency_ms,
                "tokens_used": resp.get("stats", {}).get("total_output_tokens", 0),
                "status": "ok",
                "error": None
            }

        return {
            "output": "",
            "latency_ms": latency_ms,
            "tokens_used": 0,
            "status": "error",
            "error": "Unknown response format"
        }

    def infer(self, model_endpoint: str, prompt: str, timeout: int = 120) -> Dict[str, Any]:
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
