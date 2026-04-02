"""
nova.db — Database constellation (prototype)
"""

from .db import connect_sqlite, execute, query_all

__all__ = ["connect_sqlite", "execute", "query_all"]

