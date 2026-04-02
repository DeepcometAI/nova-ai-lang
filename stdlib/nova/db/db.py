"""
nova.db — Database constellation (prototype)

For v0, this provides a small SQLite wrapper used by demos and tests.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def connect_sqlite(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def execute(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> None:
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()


def query_all(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]

