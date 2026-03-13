"""Observability subsystem — distributed tracing with optional OpenTelemetry."""

from __future__ import annotations

from vetinari.observability.ci_evaluator import (
    CaseResult,
    CIEvaluator,
    CIReport,
    EvalCase,
    get_ci_evaluator,
    reset_ci_evaluator,
)
from vetinari.observability.otel_genai import (
    GenAITracer,
    SpanContext,
    get_genai_tracer,
    reset_genai_tracer,
)
from vetinari.observability.step_evaluator import (
    EvaluationReport,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEvaluator,
    StepScore,
    get_step_evaluator,
    reset_step_evaluator,
)
from vetinari.observability.tracing import (
    NoOpSpan,
    agent_span,
    is_otel_available,
    llm_span,
    pipeline_span,
    stage_span,
    start_span,
)

__all__ = [
    "CIEvaluator",
    "CIReport",
    "CaseResult",
    "EvalCase",
    "EvaluationReport",
    "GenAITracer",
    "NoOpSpan",
    "PlanAdherenceMetric",
    "PlanQualityMetric",
    "SpanContext",
    "StepEvaluator",
    "StepScore",
    "agent_span",
    "get_ci_evaluator",
    "get_genai_tracer",
    "get_step_evaluator",
    "is_otel_available",
    "llm_span",
    "pipeline_span",
    "reset_ci_evaluator",
    "reset_genai_tracer",
    "reset_step_evaluator",
    "stage_span",
    "start_span",
]
