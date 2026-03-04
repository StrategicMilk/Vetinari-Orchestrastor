"""
Root-level CLI shim for Vetinari.

Delegates to the unified vetinari.cli entry point.
This file exists for backwards compatibility and direct execution:
  python cli.py start --goal "..."
"""

from vetinari.cli import main

if __name__ == "__main__":
    main()
