"""
ModelRelay compatibility shim.

All functionality has been consolidated into ``vetinari.dynamic_model_router``.
This module re-exports the public API so existing imports continue to work.
"""

from vetinari.dynamic_model_router import (  # noqa: F401
    # Classes
    ModelRelay,
    ModelEntry,
    ModelStatus,
    RoutingPolicy,
    RelayModelSelection as ModelSelection,
    # Singleton helpers
    get_model_relay,
    model_relay,
    _LazyModelRelay,
)

__all__ = [
    "ModelRelay",
    "ModelEntry",
    "ModelStatus",
    "RoutingPolicy",
    "ModelSelection",
    "get_model_relay",
    "model_relay",
]
