"""
Convenience launcher for the NOVA Python REPL.

This exists so you can run the REPL from inside `compiler/`:

  D:\\nova-ai-lang\\compiler> python nova_repl.py

The canonical invocation from repo root is still:

  D:\\nova-ai-lang> python -m toolchain.nova_repl
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from toolchain.nova_repl.repl import main as repl_main

    return int(repl_main())


if __name__ == "__main__":
    raise SystemExit(main())

