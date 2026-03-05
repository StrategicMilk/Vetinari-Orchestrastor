"""Backward-compatible LMStudioAdapter. Delegates to ProviderAdapter-based implementation.

Use vetinari.adapters.lmstudio_adapter.LMStudioProviderAdapter for new code.
"""

import os
import warnings
from vetinari.adapters.lmstudio_adapter import LMStudioProviderAdapter
from vetinari.adapters.base import ProviderConfig, ProviderType


class LMStudioAdapter:
    """Legacy adapter interface -- delegates to LMStudioProviderAdapter."""

    def __init__(self, host=None, api_token=None):
        warnings.warn(
            "LMStudioAdapter is deprecated. Use LMStudioProviderAdapter.",
            DeprecationWarning,
            stacklevel=2,
        )
        host = host or os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
        config = ProviderConfig(
            name="lmstudio",
            provider_type=ProviderType.LM_STUDIO,
            endpoint=host,
            api_key=api_token or "",
        )
        self._adapter = LMStudioProviderAdapter(config)
        self.host = host
        self.session = self._adapter.session
        self.max_retries = self._adapter.max_retries

    def set_api_token(self, api_token):
        self._adapter.set_api_token(api_token)

    def chat(self, model_id, system_prompt, input_text, timeout=120):
        return self._adapter.chat(model_id, system_prompt, input_text, timeout)

    def chat_stream(self, model_id, system_prompt, input_text, timeout=180):
        return self._adapter.chat_stream(model_id, system_prompt, input_text, timeout)

    def infer(self, model_endpoint, prompt, timeout=120):
        # Legacy infer uses direct endpoint, not InferenceRequest
        return self._adapter.legacy_infer(model_endpoint, prompt, timeout)

    def list_loaded_models(self):
        return self._adapter.list_loaded_models()

    def is_healthy(self):
        return self._adapter.is_healthy()

    def _post(self, endpoint, payload, timeout=120):
        return self._adapter._post_with_retry(endpoint, payload, timeout)

    def _get(self, endpoint, timeout=10):
        return self._adapter._get(endpoint, timeout)

    def _parse_response(self, resp, latency_ms):
        return self._adapter._parse_response(resp, latency_ms)
