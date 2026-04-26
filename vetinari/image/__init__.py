"""Image generation module for Vetinari.

Provides in-process image generation via HuggingFace diffusers,
eliminating the need for an external Stable Diffusion WebUI server.
"""

from __future__ import annotations

from .diffusion_engine import DiffusionEngine

__all__ = ["DiffusionEngine"]
