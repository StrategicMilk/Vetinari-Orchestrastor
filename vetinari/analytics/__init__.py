"""
Vetinari Analytics Package — Phase 5: Advanced Analytics & Cost Optimization

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

from .sla import (
    SLABreach,
    SLAReport,
    SLATracker,
    SLOTarget,
    SLOType,
    get_sla_tracker,
    reset_sla_tracker,
)

from .forecasting import (
    ForecastRequest,
    ForecastResult,
    Forecaster,
    get_forecaster,
    reset_forecaster,
)

__all__ = [
    # anomaly
    "AnomalyConfig", "AnomalyDetector", "AnomalyResult",
    "get_anomaly_detector", "reset_anomaly_detector",
    # cost
    "CostEntry", "CostReport", "CostTracker", "ModelPricing",
    "get_cost_tracker", "reset_cost_tracker",
    # sla
    "SLABreach", "SLAReport", "SLATracker", "SLOTarget", "SLOType",
    "get_sla_tracker", "reset_sla_tracker",
    # forecasting
    "ForecastRequest", "ForecastResult", "Forecaster",
    "get_forecaster", "reset_forecaster",
]

__version__ = "5.0.0"
