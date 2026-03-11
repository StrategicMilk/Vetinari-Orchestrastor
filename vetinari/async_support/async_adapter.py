"""Async provider adapter ABC mirroring the sync ProviderAdapter interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from vetinari.adapters.base import (
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    ProviderConfig,
)

logger = logging.getLogger(__name__)


class AsyncProviderAdapter(ABC):
    """Abstract base class for async provider adapters.

    Mirrors the interface of :class:`vetinari.adapters.base.ProviderAdapter`
    but exposes coroutine-based methods for use in async pipelines.

    Args:
        config: Provider configuration containing endpoint, credentials, and
            tuning parameters.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Initialise the adapter with the given configuration.

        Args:
            config: Provider configuration.
        """
        self.config = config
        self.provider_type = config.provider_type
        self.name = config.name
        self.endpoint = config.endpoint
        self.api_key = config.api_key
        self.max_retries = config.max_retries
        self.timeout_seconds = config.timeout_seconds
        self.models: list[ModelInfo] = []

    @abstractmethod
    async def ainfer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference asynchronously.

        Args:
            request: Inference request with model_id, prompt, and options.

        Returns:
            InferenceResponse with output, latency, tokens_used, and status.
        """

    @abstractmethod
    async def adiscover_models(self) -> list[ModelInfo]:
        """Discover available models asynchronously.

        Returns:
            List of ModelInfo objects representing available models.
        """

    @abstractmethod
    async def ahealth_check(self) -> dict[str, Any]:
        """Check provider health asynchronously.

        Returns:
            Dict with keys: ``healthy`` (bool), ``reason`` (str),
            ``timestamp`` (str).
        """

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider_type.value}, "
            f"endpoint={self.endpoint})"
        )
