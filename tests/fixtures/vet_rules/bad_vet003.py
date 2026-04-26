"""Module that imports ExecutionMode from the wrong source."""
from some_other_module import ExecutionMode


def get_mode():
    """Return an execution mode."""
    return ExecutionMode.FAST
