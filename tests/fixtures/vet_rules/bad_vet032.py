"""Module with ellipsis as sole function body.

VET032 applies to all scopes (no test-file skip), except .pyi stubs.
This fixture can be used directly with check_file() from any path.
"""


def not_implemented_yet() -> None:
    """Placeholder function."""
    ...
