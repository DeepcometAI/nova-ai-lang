"""
nova.test — Testing helpers (prototype)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Type


def assert_eq(actual: Any, expected: Any, message: str = "") -> None:
    if actual != expected:
        msg = message or f"expected {expected!r}, got {actual!r}"
        raise AssertionError(msg)


def assert_approx(actual: float, expected: float, tolerance: float, message: str = "") -> None:
    if abs(actual - expected) > tolerance:
        msg = message or f"expected {expected} ± {tolerance}, got {actual}"
        raise AssertionError(msg)


def assert_raises(exc_type: Type[BaseException], fn: Callable[[], Any]) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as e:  # wrong exception
        raise AssertionError(f"expected {exc_type.__name__}, got {type(e).__name__}") from e
    raise AssertionError(f"expected {exc_type.__name__} to be raised")

