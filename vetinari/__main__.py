"""Entry points for ``python -m vetinari`` and the installed ``vetinari`` script."""

from __future__ import annotations

import vetinari


def main() -> None:
    """Bootstrap environment and dispatch to the CLI entry point."""
    vetinari.bootstrap_environment()
    from vetinari.cli import main

    main()


if __name__ == "__main__":
    main()
