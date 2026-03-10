"""Legacy LM Studio adapter — thin shim over adapters.lmstudio_adapter.

All callers that ``from vetinari.lmstudio_adapter import LMStudioAdapter``
continue to work unchanged.  Internally every call now delegates to
:class:`~vetinari.adapters.lmstudio_adapter.LMStudioProviderAdapter` so
that telemetry, retries and provider config live in one place.

Direct HTTP helpers (``_post``, ``_get``, ``chat_stream``) are retained
here because the modern adapter does not yet expose streaming.
"""

import json
import logging
import time
from typing import Dict, Any, Iterator, List, Optional

import requests

from vetinari.adapters.base import (
    InferenceRequest, InferenceResponse, ProviderConfig, ProviderType,
)
from vetinari.adapters.lmstudio_adapter import LMStudioProviderAdapter

logger = logging.getLogger(__name__)


class LMStudioAdapter:
    """Backward-compatible wrapper around :class:`LMStudioProviderAdapter`.

    Construction signature is unchanged::

        adapter = LMStudioAdapter(host="http://localhost:1234")
        result  = adapter.chat(model_id, system_prompt, input_text)
    """

    def __init__(self, host: str, api_token: Optional[str] = None):
        self.host = host.rstrip("/")
        self.api_token = api_token

        # Build a ProviderConfig for the modern adapter
        cfg = ProviderConfig(
            name="lmstudio-legacy-shim",
            provider_type=ProviderType.LM_STUDIO,
            endpoint=self.host,
            api_key=api_token or "",
            timeout_seconds=120,
        )
        self._provider = LMStudioProviderAdapter(cfg)

        # Keep a requests.Session for streaming & raw helpers
        self.session = requests.Session()
        self.max_retries = 2
        if self.api_token:
            self.session.headers.update({"Authorization": f"Bearer {self.api_token}"})

    # ------------------------------------------------------------------
    # Public API (same signatures as before)
    # ------------------------------------------------------------------

    def set_api_token(self, api_token: Optional[str]):
        """Update the API token for authentication."""
        self.api_token = api_token
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        elif "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]

    def chat(self, model_id: str, system_prompt: str, input_text: str, timeout: int = 120) -> Dict[str, Any]:
        """Call LM Studio chat API — delegates to :class:`LMStudioProviderAdapter`."""
        req = InferenceRequest(
            model_id=model_id,
            prompt=input_text,
            system_prompt=system_prompt or "",
            max_tokens=2048,
            temperature=0.3,
        )
        # Temporarily override timeout
        old_timeout = self._provider.timeout_seconds
        self._provider.timeout_seconds = timeout
        try:
            resp: InferenceResponse = self._provider.infer(req)
        finally:
            self._provider.timeout_seconds = old_timeout

        return {
            "output": resp.output,
            "latency_ms": resp.latency_ms,
            "tokens_used": resp.tokens_used,
            "status": resp.status,
            "error": resp.error,
        }

    def infer(self, model_endpoint: str, prompt: str, timeout: int = 120) -> Dict[str, Any]:
        """Fallback for direct endpoint calls."""
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
                "error": resp.get("error"),
            }
        return {
            "output": "",
            "latency_ms": latency_ms,
            "tokens_used": 0,
            "status": "error",
            "error": resp.get("error", "unknown"),
        }

    # ------------------------------------------------------------------
    # Model management helpers
    # ------------------------------------------------------------------

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """Return the list of currently-loaded models from LM Studio."""
        try:
            from vetinari.model_registry import get_model_registry
            registry = get_model_registry()
            registry.refresh()
            return registry.get_loaded_as_dicts()
        except Exception:
            logger.debug("Failed to list loaded models via model registry, falling back to direct API", exc_info=True)

        # Direct fallback
        data = self._get("/v1/models", timeout=5)
        if not data:
            return []
        models = data.get("data", data) if isinstance(data, dict) else data
        return models if isinstance(models, list) else []

    def is_healthy(self) -> bool:
        """Return True if LM Studio is reachable and responding."""
        health = self._provider.health_check()
        return health.get("healthy", False)

    # ------------------------------------------------------------------
    # Streaming support (not yet in modern adapter)
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
            logger.debug("[LMStudioAdapter] Stream failed, falling back: %s", e)
            result = self.chat(model_id, system_prompt, input_text, timeout=timeout)
            output = result.get("output", "")
            if output:
                yield output

    # ------------------------------------------------------------------
    # Internal HTTP helpers (used by infer() and streaming fallback)
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
        url = endpoint if endpoint.startswith("http") else self.host + endpoint
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=timeout, headers=headers)
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    return {
                        "status": "partial",
                        "raw_output": resp.text[:5000],
                        "error": f"Non-JSON response (status {resp.status_code})",
                        "output": "",
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
            return {"status": "partial", "raw_output": resp.text[:5000], "error": "Non-JSON response"}
        except Exception:
            return None
