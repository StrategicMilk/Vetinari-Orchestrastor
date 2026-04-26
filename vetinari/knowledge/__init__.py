"""Knowledge subsystem — domain expertise for model selection and inference tuning.

Loads curated YAML knowledge files from config/knowledge/ and provides
typed access to benchmark data, quantization guides, model family profiles,
inference parameters, and architecture information.

Pipeline role: Foundation layer — consulted by Ponder (model selection),
ModelProfiler (parameter tuning), and Inspector (quality assessment context).
"""

from __future__ import annotations

from vetinari.knowledge.loader import (
    apply_self_corrections,
    get_architecture_info,
    get_benchmark_info,
    get_family_profile,
    get_parameter_guide,
    get_quant_recommendation,
    invalidate_cache,
    record_knowledge_outcome,
)
from vetinari.knowledge.validator import (
    CorrectionRecord,
    KnowledgeValidator,
    ValidationReport,
)

__all__ = [
    "CorrectionRecord",
    "KnowledgeValidator",
    "ValidationReport",
    "apply_self_corrections",
    "get_architecture_info",
    "get_benchmark_info",
    "get_family_profile",
    "get_parameter_guide",
    "get_quant_recommendation",
    "invalidate_cache",
    "record_knowledge_outcome",
]
