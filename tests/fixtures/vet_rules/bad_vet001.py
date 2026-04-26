"""Module that imports AgentType from the wrong source."""
from some_other_module import AgentType


def get_agent():
    """Return an agent type."""
    return AgentType.FOREMAN
