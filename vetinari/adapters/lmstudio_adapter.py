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

        try:
            payload = {
                "model": request.model_id,
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
        endpoint = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": model_id,
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
        endpoint = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": model_id,
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
