"""
Vetinari Tools Package

Contains Tool wrappers for all Vetinari skills, providing a standardized
interface for execution through the Tool system.

Skills migrated to Tool interface:
- builder: Code implementation, refactoring, and testing
- explorer: Fast codebase search and file discovery
- (more skills to be added)
"""

from vetinari.tools.builder_skill import BuilderSkillTool
from vetinari.tools.explorer_skill import ExplorerSkillTool
from vetinari.tools.evaluator_skill import EvaluatorSkillTool
from vetinari.tools.librarian_skill import LibrarianSkillTool
from vetinari.tools.oracle_skill import OracleSkillTool
from vetinari.tools.researcher_skill import ResearcherSkillTool
from vetinari.tools.synthesizer_skill import SynthesizerSkillTool
from vetinari.tools.ui_planner_skill import UIPlannerSkillTool

__all__ = [
    "BuilderSkillTool",
    "ExplorerSkillTool",
    "EvaluatorSkillTool",
    "LibrarianSkillTool",
    "OracleSkillTool",
    "ResearcherSkillTool",
    "SynthesizerSkillTool",
    "UIPlannerSkillTool",
]
