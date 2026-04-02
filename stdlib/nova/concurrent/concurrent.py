"""
nova.concurrent — Concurrency constellation (prototype)

Provides:
- channels (send/recv) via queue
- spawn() for background tasks via threading
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class Channel(Generic[T]):
    def __init__(self, maxsize: int = 0):
        self._q: queue.Queue[T] = queue.Queue(maxsize=maxsize)

    def send(self, item: T) -> None:
        self._q.put(item)

    def recv(self, timeout_s: Optional[float] = None) -> T:
        return self._q.get(timeout=timeout_s)


def channel(maxsize: int = 0) -> Channel[T]:
    return Channel(maxsize=maxsize)


def spawn(fn: Callable[[], None], daemon: bool = True) -> threading.Thread:
    t = threading.Thread(target=fn, daemon=daemon)
    t.start()
    return t

