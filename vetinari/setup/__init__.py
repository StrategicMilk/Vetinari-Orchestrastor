"""Vetinari setup package — first-run wizard, model recommendation, and onboarding.

This is the user-facing onboarding pipeline:
  Hardware Detection → **Model Recommendation** → **Init Wizard** → Configuration.

Entry points:
  - ``run_wizard()``: Interactive first-run setup (``vetinari init``)
  - ``ModelRecommender``: VRAM-to-model matrix for GGUF selection
"""

from __future__ import annotations

from vetinari.setup.init_wizard import run_wizard
from vetinari.setup.model_recommender import (
    ModelRecommender,
    SetupModelRecommendation,
)

__all__ = [
    "ModelRecommender",
    "SetupModelRecommendation",
    "run_wizard",
]
