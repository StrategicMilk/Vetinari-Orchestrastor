"""Observability subsystem — distributed tracing with optional OpenTelemetry."""

from vetinari.observability.tracing import (
    start_span,
    pipeline_span,
    stage_span,
    agent_span,
    llm_span,
    is_otel_available,
    NoOpSpan,
)
from vetinari.observability.otel_genai import (
    get_genai_tracer,
    reset_genai_tracer,
    GenAITracer,
    SpanContext,
)
from vetinari.observability.step_evaluator import (
    get_step_evaluator,
    reset_step_evaluator,
    StepEvaluator,
    PlanQualityMetric,
    PlanAdherenceMetric,
    StepScore,
    EvaluationReport,
)

from vetinari.observability.ci_evaluator import (
    get_ci_evaluator,
    reset_ci_evaluator,
    CIEvaluator,
    EvalCase,
    CaseResult,
    CIReport,
)

__all__ = [
    "start_span",
    "pipeline_span",
    "stage_span",
    "agent_span",
    "llm_span",
    "is_otel_available",
    "NoOpSpan",
    "get_genai_tracer",
    "reset_genai_tracer",
    "GenAITracer",
    "SpanContext",
    "get_step_evaluator",
    "reset_step_evaluator",
    "StepEvaluator",
    "PlanQualityMetric",
    "PlanAdherenceMetric",
    "StepScore",
    "EvaluationReport",
    "get_ci_evaluator",
    "reset_ci_evaluator",
    "CIEvaluator",
    "EvalCase",
    "CaseResult",
    "CIReport",
]
