"""Module with bare except clause."""


def risky_operation() -> str:
    """Perform a risky operation.

    Returns:
        Result string.
    """
    try:
        return "result"
    except:
        return "failed"
