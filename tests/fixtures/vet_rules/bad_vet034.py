"""Module with placeholder string value.

IMPORTANT — test runner note:
VET034 skips files under tests/. This fixture must be copied to a tmp_path
outside tests/ before calling check_file() on it. The test handles this copy.

The value "foo" on the NAME line is a bare short string literal with fewer than
3 other alphabetic words on the same line, so it triggers VET034.
"""

NAME = "foo"
