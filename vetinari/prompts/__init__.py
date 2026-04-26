"""Prompt assembly and template management for Vetinari agents."""

from __future__ import annotations

from vetinari.prompts.assembler import PromptAssembler, get_prompt_assembler
from vetinari.prompts.version_manager import PromptVersionManager, get_version_manager

__all__ = [
    "PromptAssembler",
    "PromptVersionManager",
    "get_prompt_assembler",
    "get_version_manager",
]
