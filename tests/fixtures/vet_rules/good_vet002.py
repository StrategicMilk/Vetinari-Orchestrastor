"""Module that imports TaskStatus correctly."""
from vetinari.types import TaskStatus


def get_status():
    """Return a status."""
    return TaskStatus.COMPLETED
