"""Module that imports PlanStatus from the wrong source."""
from some_other_module import PlanStatus


def get_plan_status():
    """Return a plan status."""
    return PlanStatus.RUNNING
