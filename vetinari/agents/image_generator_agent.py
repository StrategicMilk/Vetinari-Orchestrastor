"""Legacy redirect — use BuilderAgent directly."""
from vetinari.agents.builder_agent import BuilderAgent as ImageGeneratorAgent, get_builder_agent as get_image_generator_agent  # noqa: F401

__all__ = ["ImageGeneratorAgent", "get_image_generator_agent"]
