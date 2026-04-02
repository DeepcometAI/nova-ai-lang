"""
Convenience launcher for the NOVA Python REPL.

Works from repo root:
  D:\\nova-ai-lang> python nova_repl.py

Canonical module form:
  D:\\nova-ai-lang> python -m toolchain.nova_repl
"""

from __future__ import annotations

from toolchain.nova_repl.repl import main

if __name__ == "__main__":
    raise SystemExit(main())

