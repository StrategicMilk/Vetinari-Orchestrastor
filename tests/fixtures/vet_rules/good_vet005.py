"""Module that uses AgentType from the canonical source."""
from vetinari.types import AgentType


def classify(agent_type: AgentType) -> str:
    """Classify an agent type.

    Args:
        agent_type: The type of agent.

    Returns:
        A string description of the agent type.
    """
    return agent_type.value
