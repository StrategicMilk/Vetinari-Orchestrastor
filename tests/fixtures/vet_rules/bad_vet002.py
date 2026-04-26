"""Module that imports TaskStatus from the wrong source."""
from some_other_module import TaskStatus


def get_status():
    """Return a status."""
    return TaskStatus.COMPLETED
