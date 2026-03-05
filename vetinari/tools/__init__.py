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
    "get_all_skills",
]


def get_all_skills():
    """Auto-discover all skill classes in the vetinari.tools package.

    Iterates over every module in this package using ``pkgutil`` and collects
    concrete ``Tool`` subclasses (i.e. classes that have a ``metadata``
    attribute, which is set by the ``Tool.__init__`` constructor).  Import
    errors in individual modules are silently ignored so that a broken or
    optional skill never prevents the rest of the system from starting.

    Returns:
        list[type]: A list of discovered skill *classes* (not instances).
    """
    import importlib
    import inspect
    import pkgutil

    from vetinari.tool_interface import Tool

    skills = []
    for finder, name, ispkg in pkgutil.iter_modules(__path__):
        try:
            mod = importlib.import_module(f".{name}", __package__)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Tool)
                    and attr is not Tool
                    and not inspect.isabstract(attr)
                ):
                    skills.append(attr)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "Skipping skill module %s: %s", name, e
            )
    return skills
