"""Command-line entry point for fd-eval-harness.

This is a placeholder stub so that ``pyproject.toml``'s ``[project.scripts]``
declaration resolves. The real CLI will be built out against the design in
``docs/DESIGN.md`` once the core interfaces stabilize.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point wired to the ``fd-eval`` console script.

    Current behavior: print a short status message and exit 0. Replace with
    the real argparse surface in v0.1.
    """
    _ = argv if argv is not None else sys.argv[1:]
    print("fd-eval-harness: not yet implemented. See docs/DESIGN.md.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
