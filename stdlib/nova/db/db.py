"""
nova.db — Database constellation (Python prototype)

SQLite wrapper providing a clean, NOVA-flavoured API.
For demos and development; production use should migrate to a
proper ORM or client library.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

__all__ = [
    "connect",
    "connect_sqlite",
    "Database",
    "execute",
    "query_one",
    "query_all",
    "insert",
    "update",
    "delete",
    "transaction",
    "table_exists",
    "create_table",
    "drop_table",
]

Row = Dict[str, Any]


class Database:
    """
    Thin wrapper around sqlite3.Connection with a NOVA-flavoured API.

    Usage:
        db = Database(":memory:")
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        db.insert("t", {"val": "hello"})
        rows = db.query_all("SELECT * FROM t")
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Execute a statement (DDL or DML) with no return value."""
        with self._conn:
            self._conn.execute(sql, params)

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> List[Row]:
        """Execute a SELECT and return all rows as dicts."""
        cur = self._conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Row]:
        """Execute a SELECT and return the first row, or None."""
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def insert(self, table: str, row: Row) -> int:
        """
        Insert a row dict into table.  Returns the last-inserted rowid.
        """
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        with self._conn:
            cur = self._conn.execute(sql, list(row.values()))
        return int(cur.lastrowid or 0)

    def update(
        self,
        table: str,
        values: Row,
        where: str,
        where_params: Sequence[Any] = (),
    ) -> int:
        """
        UPDATE table SET key=? ... WHERE ...
        Returns the number of rows affected.
        """
        set_clause = ", ".join(f"{k} = ?" for k in values)
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        params = list(values.values()) + list(where_params)
        with self._conn:
            cur = self._conn.execute(sql, params)
        return int(cur.rowcount)

    def delete(self, table: str, where: str, where_params: Sequence[Any] = ()) -> int:
        """DELETE FROM table WHERE ... Returns the number of rows deleted."""
        sql = f"DELETE FROM {table} WHERE {where}"
        with self._conn:
            cur = self._conn.execute(sql, list(where_params))
        return int(cur.rowcount)

    def table_exists(self, table: str) -> bool:
        """Return True if the table exists."""
        row = self.query_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return row is not None

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager for explicit transaction control."""
        with self._conn:
            yield


# ── Module-level convenience functions ───────────────────────────────────────

def connect(path: str = ":memory:") -> Database:
    """Open (or create) a SQLite database and return a Database object."""
    return Database(path)


def connect_sqlite(path: str = ":memory:") -> sqlite3.Connection:
    """Raw sqlite3.Connection (for low-level access)."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def execute(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> None:
    """Execute SQL on a raw sqlite3 connection."""
    with conn:
        conn.execute(sql, params)


def query_all(
    conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
) -> List[Row]:
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def query_one(
    conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
) -> Optional[Row]:
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return dict(row) if row else None


def insert(conn: sqlite3.Connection, table: str, row: Row) -> int:
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    with conn:
        cur = conn.execute(sql, list(row.values()))
    return int(cur.lastrowid or 0)


def update(
    conn: sqlite3.Connection,
    table: str,
    values: Row,
    where: str,
    where_params: Sequence[Any] = (),
) -> int:
    set_clause = ", ".join(f"{k} = ?" for k in values)
    sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
    params = list(values.values()) + list(where_params)
    with conn:
        cur = conn.execute(sql, params)
    return int(cur.rowcount)


def delete(
    conn: sqlite3.Connection, table: str, where: str, where_params: Sequence[Any] = ()
) -> int:
    sql = f"DELETE FROM {table} WHERE {where}"
    with conn:
        cur = conn.execute(sql, list(where_params))
    return int(cur.rowcount)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[None, None, None]:
    with conn:
        yield


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def create_table(
    conn: sqlite3.Connection, table: str, schema: str, if_not_exists: bool = True
) -> None:
    """CREATE TABLE [IF NOT EXISTS] table (schema)."""
    guard = "IF NOT EXISTS " if if_not_exists else ""
    execute(conn, f"CREATE TABLE {guard}{table} ({schema})")


def drop_table(conn: sqlite3.Connection, table: str, if_exists: bool = True) -> None:
    guard = "IF EXISTS " if if_exists else ""
    execute(conn, f"DROP TABLE {guard}{table}")
