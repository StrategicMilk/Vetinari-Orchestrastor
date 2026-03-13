"""Vetinari Analytics Package — Phase 5: Advanced Analytics & Cost Optimization.

Modules
-------
    anomaly      AI-driven anomaly detection (Z-score, IQR, EWMA)
    cost         Cost attribution per agent / task / provider / model
    sla          SLA / SLO tracking and compliance reporting
    forecasting  Capacity-planning forecasts (SMA, ES, linear trend, seasonal)

Quick start
-----------
    from vetinari.analytics import (
        get_anomaly_detector, AnomalyConfig,
        get_cost_tracker,    CostEntry, ModelPricing,
        get_sla_tracker,     SLOTarget, SLOType,
        get_forecaster,      ForecastRequest,
    )
"""

from __future__ import annotations

from .anomaly import (
    AnomalyConfig,
    AnomalyDetector,
    AnomalyResult,
    get_anomaly_detector,
    reset_anomaly_detector,
)
from .cost import (
    CostEntry,
    CostReport,
    CostTracker,
    ModelPricing,
    get_cost_tracker,
    reset_cost_tracker,
)
from .forecasting import (
    Forecaster,
    ForecastRequest,
    ForecastResult,
    get_forecaster,
    reset_forecaster,
)
from .sla import (
    SLABreach,
    SLAReport,
    SLATracker,
    SLOTarget,
    SLOType,
    get_sla_tracker,
    reset_sla_tracker,
)

__all__ = [
    # anomaly
    "AnomalyConfig",
    "AnomalyDetector",
    "AnomalyResult",
    # cost
    "CostEntry",
    "CostReport",
    "CostTracker",
    # forecasting
    "ForecastRequest",
    "ForecastResult",
    "Forecaster",
    "ModelPricing",
    # sla
    "SLABreach",
    "SLAReport",
    "SLATracker",
    "SLOTarget",
    "SLOType",
    "get_anomaly_detector",
    "get_cost_tracker",
    "get_forecaster",
    "get_sla_tracker",
    "reset_anomaly_detector",
    "reset_cost_tracker",
    "reset_forecaster",
    "reset_sla_tracker",
]

__version__ = "5.0.0"
