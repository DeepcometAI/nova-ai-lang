"""
nova.concurrent — Concurrency constellation (Python prototype)

Provides:
  - Channel[T]   — thread-safe message channel (send / recv)
  - WorkQueue    — bounded work queue with parallel worker pool
  - spawn()      — fire-and-forget background task
  - atomic_int   — thread-safe integer counter
  - lock()       — context-manager mutex
  - Semaphore    — counting semaphore
  - Timer        — run a callback after a delay
"""

from __future__ import annotations

import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

__all__ = [
    "Channel",
    "channel",
    "WorkQueue",
    "spawn",
    "atomic_int",
    "lock",
    "Semaphore",
    "Timer",
    "parallel_map",
]


class Channel(Generic[T]):
    """
    Thread-safe FIFO channel.  NOVA equivalent of a bounded channel.
    Usage:
        ch = channel()
        spawn(lambda: ch.send(42))
        v = ch.recv()
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._q: queue.Queue[T] = queue.Queue(maxsize=maxsize)

    def send(self, item: T) -> None:
        """Send an item (blocks if channel is full and maxsize > 0)."""
        self._q.put(item)

    def recv(self, timeout_s: Optional[float] = None) -> T:
        """Receive an item (blocks until one is available or timeout)."""
        try:
            return self._q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError("channel recv timed out")

    def try_recv(self) -> Optional[T]:
        """Non-blocking recv; returns None if empty."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def size(self) -> int:
        """Current number of items in the channel."""
        return self._q.qsize()

    def is_empty(self) -> bool:
        return self._q.empty()


def channel(maxsize: int = 0) -> Channel:
    """Create a new channel with optional bound."""
    return Channel(maxsize=maxsize)


def spawn(
    fn: Callable[[], None],
    *,
    daemon: bool = True,
    name: Optional[str] = None,
) -> threading.Thread:
    """
    Spawn fn in a background thread.
    Returns the Thread object (already started).
    """
    t = threading.Thread(target=fn, daemon=daemon, name=name)
    t.start()
    return t


class atomic_int:
    """Thread-safe integer counter."""

    def __init__(self, value: int = 0) -> None:
        self._lock = threading.Lock()
        self._value = int(value)

    def get(self) -> int:
        with self._lock:
            return self._value

    def set(self, v: int) -> None:
        with self._lock:
            self._value = int(v)

    def add(self, delta: int = 1) -> int:
        """Atomically add delta and return the new value."""
        with self._lock:
            self._value += int(delta)
            return self._value

    def compare_and_swap(self, expected: int, new: int) -> bool:
        """If current == expected, set to new and return True."""
        with self._lock:
            if self._value == expected:
                self._value = new
                return True
            return False


@contextmanager
def lock(mutex: threading.Lock) -> Iterator[None]:
    """Context manager that acquires and releases a threading.Lock."""
    mutex.acquire()
    try:
        yield
    finally:
        mutex.release()


class Semaphore:
    """Counting semaphore backed by threading.Semaphore."""

    def __init__(self, count: int = 1) -> None:
        self._sem = threading.Semaphore(int(count))

    def acquire(self, timeout_s: Optional[float] = None) -> bool:
        return self._sem.acquire(timeout=timeout_s)

    def release(self) -> None:
        self._sem.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


class Timer:
    """Run a callback once after a delay (non-blocking)."""

    def __init__(self, delay_s: float, fn: Callable[[], None]) -> None:
        self._t = threading.Timer(float(delay_s), fn)

    def start(self) -> None:
        self._t.start()

    def cancel(self) -> None:
        self._t.cancel()


class WorkQueue:
    """
    Bounded work queue with a fixed pool of worker threads.
    Tasks are callables submitted via submit().
    """

    def __init__(self, num_workers: int = 4, maxsize: int = 0) -> None:
        self._q: queue.Queue[Optional[Callable]] = queue.Queue(maxsize=maxsize)
        self._workers: List[threading.Thread] = []
        for _ in range(int(num_workers)):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker(self) -> None:
        while True:
            task = self._q.get()
            if task is None:
                return
            try:
                task()
            except Exception:
                pass
            finally:
                self._q.task_done()

    def submit(self, fn: Callable[[], None]) -> None:
        """Submit a no-argument callable for execution."""
        self._q.put(fn)

    def wait(self) -> None:
        """Block until all submitted tasks have been processed."""
        self._q.join()

    def shutdown(self) -> None:
        """Signal all workers to stop after draining the queue."""
        for _ in self._workers:
            self._q.put(None)
        for t in self._workers:
            t.join()


def parallel_map(
    fn: Callable,
    items: list,
    num_workers: int = 4,
) -> list:
    """
    Apply fn to each item in parallel and return results in order.
    Uses threading (GIL-limited, but useful for I/O-bound tasks).
    """
    results: list = [None] * len(items)
    lock = threading.Lock()

    def task(i: int, item) -> None:
        result = fn(item)
        with lock:
            results[i] = result

    threads = [
        threading.Thread(target=task, args=(i, item), daemon=True)
        for i, item in enumerate(items)
    ]
    # Limit concurrency to num_workers
    sem = threading.Semaphore(int(num_workers))
    started = []
    for t in threads:
        sem.acquire()
        def _run(thread=t):
            try:
                thread.run()
            finally:
                sem.release()
        real_t = threading.Thread(target=_run, daemon=True)
        real_t.start()
        started.append(real_t)
    for t in started:
        t.join()
    return results
