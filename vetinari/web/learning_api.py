"""Flask Blueprint exposing learning pipeline data via REST API.

Wires the Thompson Sampling model selector, quality scorer, and training
data collector into the web dashboard so the UI can display live learning
metrics.

Endpoints
---------
    GET /api/v1/learning/thompson      — Thompson Sampling arm states
    GET /api/v1/learning/quality-history — Quality score time series
    GET /api/v1/learning/training-stats — Training data statistics
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

learning_bp = Blueprint("learning", __name__)


@learning_bp.route("/api/v1/learning/thompson")
def get_thompson_arms():
    """Return Thompson Sampling arm states.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.learning.model_selector import get_thompson_selector

        selector = get_thompson_selector()
        # Build a serialisable dict of all arm states
        arms = {}
        for key, arm in selector._arms.items():
            arms[key] = {
                "model_id": arm.model_id,
                "task_type": arm.task_type,
                "alpha": arm.alpha,
                "beta": arm.beta,
                "mean": arm.mean,
                "total_pulls": arm.total_pulls,
                "last_updated": arm.last_updated,
            }
        return jsonify({"arms": arms})
    except Exception as e:
        logger.warning("Failed to fetch Thompson arms: %s", e)
        return jsonify({"arms": {}, "error": str(e)})


@learning_bp.route("/api/v1/learning/quality-history")
def get_quality_history():
    """Return quality score time series.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.learning.quality_scorer import get_quality_scorer

        scorer = get_quality_scorer()
        history = scorer.get_history()
        serialised = [
            {
                "task_id": s.task_id,
                "model_id": s.model_id,
                "task_type": s.task_type,
                "overall_score": s.overall_score,
                "method": s.method,
                "timestamp": s.timestamp,
            }
            for s in history
        ]
        return jsonify({"history": serialised})
    except Exception as e:
        logger.warning("Failed to fetch quality history: %s", e)
        return jsonify({"history": [], "error": str(e)})


@learning_bp.route("/api/v1/learning/training-stats")
def get_training_stats():
    """Return training data statistics.

    Returns:
        The jsonify result.
    """
    try:
        from vetinari.learning.training_data import get_training_collector

        collector = get_training_collector()
        stats = collector.get_stats()
        return jsonify({"stats": stats})
    except Exception as e:
        logger.warning("Failed to fetch training stats: %s", e)
        return jsonify({"stats": {}, "error": str(e)})
