"""Legacy redirect — use BuilderAgent directly."""

from __future__ import annotations

from vetinari.agents.builder_agent import BuilderAgent as ImageGeneratorAgent
from vetinari.agents.builder_agent import get_builder_agent as get_image_generator_agent

__all__ = ["ImageGeneratorAgent", "get_image_generator_agent"]
