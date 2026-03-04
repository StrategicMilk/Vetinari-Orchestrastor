import json
import logging
import os
import time
from typing import Dict, Any, Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # Model management helpers
    # ------------------------------------------------------------------

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """Return the list of currently-loaded models from LM Studio.

        Delegates to the unified ModelRegistry when available, falling back
        to a direct ``/v1/models`` call so callers always get fresh data.
        """
        try:
            from vetinari.model_registry import get_model_registry
            registry = get_model_registry()
            registry.refresh()
            return registry.get_loaded_as_dicts()
        except Exception:
            pass

        # Direct fallback
        data = self._get("/v1/models", timeout=5)
        if not data:
            return []
        models = data.get("data", data) if isinstance(data, dict) else data
        return models if isinstance(models, list) else []

    def is_healthy(self) -> bool:
        """Return True if LM Studio is reachable and responding."""
        try:
            resp = self.session.get(f"{self.host}/v1/models", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Streaming support
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        model_id: str,
        system_prompt: str,
        input_text: str,
        timeout: int = 180,
    ) -> Iterator[str]:
        """Stream chat completion tokens from LM Studio.

        Yields individual text chunks as they arrive.  Falls back gracefully
        to a full (non-streamed) response if streaming fails.

        Usage::

            for chunk in adapter.chat_stream(model, sys_prompt, user_text):
                print(chunk, end="", flush=True)
        """
        endpoint = f"{self.host}/v1/chat/completions"
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": input_text},
            ],
            "temperature": 0.3,
            "stream": True,
        }
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            with self.session.post(
                endpoint, json=payload, headers=headers,
                timeout=timeout, stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(line)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except Exception as e:
            logger.debug(f"[LMStudioAdapter] Stream failed, falling back: {e}")
            # Fall back to non-streaming
            result = self.chat(model_id, system_prompt, input_text, timeout=timeout)
            output = result.get("output", "")
            if output:
                yield output
