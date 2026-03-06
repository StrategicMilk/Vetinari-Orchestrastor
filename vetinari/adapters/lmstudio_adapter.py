"""LM Studio provider adapter.

Provides the canonical LM Studio integration via the ProviderAdapter interface,
plus backward-compatible convenience methods (chat, chat_stream, legacy_infer,
list_loaded_models, is_healthy) so the legacy shim in
``vetinari.lmstudio_adapter`` can delegate everything here.
"""

import json
import logging
import requests
import time
from typing import Dict, List, Any, Iterator, Optional
from .base import (
    ProviderAdapter, ProviderConfig, ProviderType, ModelInfo,
    InferenceRequest, InferenceResponse
)

logger = logging.getLogger(__name__)


# Module-level cache: avoids hitting /v1/models on every single request.
_resolved_model_cache: Dict[str, str] = {}


def resolve_lmstudio_model(model_id: str, host: Optional[str] = None) -> str:
    """Resolve 'default' or empty model_id to an actual loaded LM Studio model.

    Resolution order:
      1. If model_id is a real name (not 'default'/empty), return as-is.
      2. Check ``VETINARI_DEFAULT_MODEL`` env var.
      3. Return cached result if available.
      4. Query ``/v1/models`` for loaded models.
      5. Try ``/api/v0/models`` (older LM Studio versions).
      6. Probe via a minimal chat completion (response reveals model name).
      7. Fall back to empty string ``""`` (LM Studio routes to loaded model).

    Usage::

        from vetinari.adapters.lmstudio_adapter import resolve_lmstudio_model
        real_model = resolve_lmstudio_model(model_id)
    """
    import os as _os

    if model_id and model_id != "default":
        return model_id

    # Check env var override first
    env_model = _os.environ.get("VETINARI_DEFAULT_MODEL", "")
    if env_model:
        return env_model

    _host = host or _os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")

    # Return cached result if we already resolved for this host
    if _host in _resolved_model_cache:
        return _resolved_model_cache[_host]

    # Try /v1/models (OpenAI-compatible)
    resolved = _try_discover_model(_host, "/v1/models")
    if resolved:
        _resolved_model_cache[_host] = resolved
        return resolved

    # Try /api/v0/models (older LM Studio)
    resolved = _try_discover_model(_host, "/api/v0/models")
    if resolved:
        _resolved_model_cache[_host] = resolved
        return resolved

    # Probe: send a minimal chat completion and read the model name
    # from the response (LM Studio echoes the actual model in the response)
    resolved = _probe_model_via_chat(_host)
    if resolved:
        _resolved_model_cache[_host] = resolved
        return resolved

    # Ultimate fallback: empty string — LM Studio routes to the loaded
    # model when model name is empty, unlike "default" which it rejects.
    logger.info(
        "[LMStudio] Could not discover model name. Using empty string "
        "(LM Studio will route to loaded model). Set VETINARI_DEFAULT_MODEL "
        "env var for explicit control."
    )
    _resolved_model_cache[_host] = ""
    return ""


def _try_discover_model(host: str, endpoint: str) -> Optional[str]:
    """Try to discover a loaded model from an LM Studio endpoint."""
    try:
        resp = requests.get(f"{host}{endpoint}", timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()

        # Handle various response formats
        models_list: list = []
        if isinstance(data, dict):
            # Standard: {"data": [{"id": "..."}]}
            models_list = data.get("data", data.get("models", []))
            # Some versions return {"id": "..."} directly
            if not models_list and "id" in data:
                resolved = data["id"]
                if resolved and isinstance(resolved, str):
                    logger.debug(f"[LMStudio] Resolved via {endpoint} -> '{resolved}'")
                    return resolved
        elif isinstance(data, list):
            models_list = data

        if models_list and isinstance(models_list, list):
            first = models_list[0]
            if isinstance(first, dict):
                resolved = first.get("id", first.get("name", first.get("model", "")))
            else:
                resolved = str(first)
            if resolved:
                logger.debug(f"[LMStudio] Resolved via {endpoint} -> '{resolved}'")
                return resolved
    except Exception:
        pass
    return None


def _probe_model_via_chat(host: str) -> Optional[str]:
    """Discover the model name by making a minimal chat completion request.

    LM Studio echoes the actual model name in the response ``"model"`` field,
    even when the request sends an empty or placeholder model name.
    """
    try:
        resp = requests.post(
            f"{host}/v1/chat/completions",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            model_name = data.get("model", "")
            if model_name and model_name not in ("", "default"):
                logger.info(f"[LMStudio] Discovered model via probe: '{model_name}'")
                return model_name
    except Exception:
        pass
    return None


class LMStudioProviderAdapter(ProviderAdapter):
    """Adapter for LM Studio local inference."""

    def __init__(self, config: ProviderConfig):
        """Initialize LM Studio adapter."""
        if config.provider_type != ProviderType.LM_STUDIO:
            raise ValueError("LMStudioProviderAdapter requires LM_STUDIO provider type")
        super().__init__(config)
        self.session = requests.Session()

        # Carry auth into session headers
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def set_api_token(self, api_token: Optional[str]) -> None:
        """Update the API token at runtime and refresh session headers."""
        self.api_key = api_token
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        elif "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]

    # ------------------------------------------------------------------
    # Model ID resolution
    # ------------------------------------------------------------------

    def _resolve_model_id(self, model_id: str) -> str:
        """Resolve 'default' or empty model_id via the standalone utility."""
        return resolve_lmstudio_model(model_id, self.endpoint)

    # ------------------------------------------------------------------
    # Low-level HTTP helpers
    # ------------------------------------------------------------------

    def _post_with_retry(
        self, endpoint: str, payload: Dict[str, Any], timeout: int = 120
    ) -> Dict[str, Any]:
        """POST with exponential-backoff retry.

        Uses *self.max_retries* (inherited from ProviderConfig).
        """
        url = endpoint if endpoint.startswith("http") else self.endpoint + endpoint
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    url, json=payload, timeout=timeout, headers=headers,
                )
                try:
                    data = resp.json()
                    return data
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
        url = endpoint if endpoint.startswith("http") else self.endpoint + endpoint
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = self.session.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except json.JSONDecodeError:
            return {
                "status": "partial",
                "raw_output": resp.text[:5000],
                "error": "Non-JSON response",
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, resp: Dict[str, Any], latency_ms: int
    ) -> Dict[str, Any]:
        """Parse LM Studio response with full envelope handling.

        Handles: partial (non-JSON fallback), error, OpenAI choices format,
        and legacy ``output`` key from older LM Studio versions.
        """
        # Partial / non-JSON fallback
        if resp.get("status") == "partial":
            return {
                "output": resp.get("raw_output", ""),
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "status": "partial",
                "error": resp.get("error", "Partial response"),
                "raw_output": resp.get("raw_output", ""),
            }

        # Explicit error in payload
        if "error" in resp:
            error_msg = resp.get("error", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            return {
                "output": "",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "status": "error",
                "error": error_msg,
            }

        # OpenAI-compatible: choices[0].message.content
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
                "error": None,
            }

        # Legacy "output" key (older LM Studio versions)
        if "output" in resp:
            output_text = resp["output"]
            if isinstance(output_text, list) and output_text:
                first = output_text[0]
                output_text = (
                    first.get("content", str(first))
                    if isinstance(first, dict)
                    else str(first)
                )
            elif not isinstance(output_text, str):
                output_text = str(output_text)
            return {
                "output": output_text,
                "latency_ms": latency_ms,
                "tokens_used": resp.get("stats", {}).get(
                    "total_output_tokens", 0
                ),
                "status": "ok",
                "error": None,
            }

        return {
            "output": "",
            "latency_ms": latency_ms,
            "tokens_used": 0,
            "status": "error",
            "error": "Unknown response format",
        }

    # ------------------------------------------------------------------
    # ProviderAdapter abstract interface
    # ------------------------------------------------------------------

    def discover_models(self) -> List[ModelInfo]:
        """Discover models available in LM Studio."""
        try:
            response = self.session.get(
                f"{self.endpoint}/v1/models",
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()

            # Handle both dict and list response formats
            if isinstance(data, dict):
                if "data" in data:
                    models_list = data["data"]
                elif "models" in data:
                    models_list = data["models"]
                else:
                    models_list = []
            else:
                models_list = data if isinstance(data, list) else []

            discovered = []
            for m in models_list:
                model_id = m.get("id", "")
                if not model_id:
                    continue

                model_info = ModelInfo(
                    id=model_id,
                    name=m.get("name", model_id),
                    provider="lm_studio",
                    endpoint=f"{self.endpoint}/v1/chat/completions",
                    capabilities=m.get("capabilities", ["code_gen", "chat"]),
                    context_len=m.get("context_len", 2048),
                    memory_gb=m.get("memory_gb", 4),
                    version=m.get("version", "unknown"),
                    latency_estimate_ms=m.get("latency_estimate_ms", 1000),
                    tags=["local", "lm_studio"],
                )
                discovered.append(model_info)

            self.models = discovered
            logger.info(f"[LMStudio] Discovered {len(discovered)} models")
            return discovered

        except Exception as e:
            logger.error(f"[LMStudio] Model discovery failed: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check LM Studio health."""
        try:
            response = self.session.get(
                f"{self.endpoint}/v1/models",
                timeout=5
            )
            return {
                "healthy": response.status_code == 200,
                "reason": "LM Studio responding" if response.status_code == 200 else f"Status {response.status_code}",
                "timestamp": time.time(),
                "endpoint": self.endpoint,
            }
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": time.time(),
                "endpoint": self.endpoint,
            }

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference using LM Studio."""
        start_time = time.time()
        resolved_model = self._resolve_model_id(request.model_id)

        try:
            payload = {
                "model": resolved_model,
                "messages": [
                    {"role": "system", "content": request.system_prompt or ""},
                    {"role": "user", "content": request.prompt}
                ],
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }

            response = self.session.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            output = ""
            tokens_used = 0

            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    output = choice["message"].get("content", "")
                elif "text" in choice:
                    output = choice["text"]

            if "usage" in data:
                tokens_used = data["usage"].get("total_tokens", 0)

            resp = InferenceResponse(
                model_id=request.model_id,
                output=output,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                status="ok",
            )
            self._record_telemetry(request, resp)
            return resp

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[LMStudio] Inference failed: {e}")
            resp = InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                status="error",
                error=str(e),
            )
            self._record_telemetry(request, resp)
            return resp

    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all models."""
        return {m.id: m.capabilities for m in self.models}

    # ------------------------------------------------------------------
    # Backward-compatible convenience methods
    # ------------------------------------------------------------------

    def chat(
        self,
        model_id: str,
        system_prompt: str,
        input_text: str,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """OpenAI-compatible chat returning the legacy Dict envelope.

        Kept for backward compatibility with code that expects::

            {"output": str, "latency_ms": int, "tokens_used": int,
             "status": str, "error": str|None}
        """
        resolved_model = self._resolve_model_id(model_id)
        endpoint = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": input_text},
            ],
            "temperature": 0.3,
            "stream": False,
        }

        start = time.time()
        resp = self._post_with_retry(endpoint, payload, timeout=timeout)
        latency_ms = int((time.time() - start) * 1000)

        return self._parse_response(resp, latency_ms)

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
        resolved_model = self._resolve_model_id(model_id)
        endpoint = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": input_text},
            ],
            "temperature": 0.3,
            "stream": True,
        }
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            with self.session.post(
                endpoint, json=payload, headers=headers,
                timeout=timeout, stream=True,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = (
                        raw_line.decode("utf-8")
                        if isinstance(raw_line, bytes)
                        else raw_line
                    )
                    if line.startswith("data: "):
                        line = line[6:]
                    if line.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(line)
                        delta = (
                            chunk.get("choices", [{}])[0].get("delta", {})
                        )
                        text = delta.get("content", "")
                        if text:
                            yield text
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except Exception as e:
            logger.debug(f"[LMStudio] Stream failed, falling back: {e}")
            result = self.chat(
                model_id, system_prompt, input_text, timeout=timeout,
            )
            output = result.get("output", "")
            if output:
                yield output

    def legacy_infer(
        self,
        model_endpoint: str,
        prompt: str,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """Direct endpoint call returning the old Dict format.

        Used by the legacy ``LMStudioAdapter.infer()`` shim.
        """
        payload = {"prompt": prompt}
        start = time.time()
        resp = self._post_with_retry(model_endpoint, payload, timeout=timeout)
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

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """Return loaded models, using ModelRegistry when available."""
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
            resp = self.session.get(
                f"{self.endpoint}/v1/models", timeout=3,
            )
            return resp.status_code == 200
        except Exception:
            return False
