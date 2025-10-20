#!
# /usr/bin/env python3
"""AI-assisted web crawler that prioritizes pages based on semantic relevance."""

from __future__ import annotations

import heapq
"""CLI entry point for the crawler GUI application."""

from crawler_app.gui import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()

