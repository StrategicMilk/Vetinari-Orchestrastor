"""Module with typed except clause."""


def risky_operation() -> str:
    """Perform a risky operation.

    Returns:
        Result string.
    """
    try:
        return "result"
    except Exception:
        return "failed"
