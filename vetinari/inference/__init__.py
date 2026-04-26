"""Inference subsystem — continuous batching."""

from __future__ import annotations

from vetinari.inference.batcher import (  # noqa: VET123 - barrel export preserves public import compatibility
    BatchConfig,
    BatchRequest,
    InferenceBatcher,
    get_inference_batcher,
)

__all__ = [
    "BatchConfig",
    "BatchRequest",
    "InferenceBatcher",
    "get_inference_batcher",
]
