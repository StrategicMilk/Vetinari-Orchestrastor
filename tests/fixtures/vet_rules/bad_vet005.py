"""Module that redefines AgentType enum locally."""
from enum import Enum


class AgentType(str, Enum):
    """Duplicate enum — this should be imported from vetinari.types."""

    FOREMAN = "foreman"
    WORKER = "worker"
