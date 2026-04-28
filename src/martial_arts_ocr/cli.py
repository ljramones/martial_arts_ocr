"""Command line entry points."""

from __future__ import annotations


def main() -> None:
    from app import main as legacy_main

    legacy_main()


if __name__ == "__main__":
    main()

