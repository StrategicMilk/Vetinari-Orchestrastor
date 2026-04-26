"""Root-level CLI shim for Vetinari.

DISPOSITION: Retained convenience shim — delegates all commands to the
canonical ``vetinari.cli`` entry point.  This file is intentionally kept
so that ``python cli.py <command>`` continues to work from the repo root
without activating the package entry point.

Canonical invocation (preferred):
  python -m vetinari <command>

Shim invocation (equivalent, convenience only):
  python cli.py <command>
"""

from vetinari.cli import main

if __name__ == "__main__":
    main()
