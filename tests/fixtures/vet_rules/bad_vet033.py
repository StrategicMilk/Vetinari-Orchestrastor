"""Module with NotImplementedError raised outside @abstractmethod.

VET033 applies to all scopes (no test-file skip). This fixture can be used
directly with check_file() from any path.
"""


def do_something() -> str:
    """Do something.

    Returns:
        Result string.
    """
    raise NotImplementedError("Subclass must implement this")
