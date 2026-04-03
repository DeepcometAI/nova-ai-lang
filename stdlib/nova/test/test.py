"""
nova.test — Testing helpers constellation (Python prototype)

Provides:
  - assert_eq / assert_ne
  - assert_approx
  - assert_raises / assert_not_raises
  - assert_in / assert_not_in
  - assert_true / assert_false
  - assert_is_none / assert_is_not_none
  - TestSuite runner
  - bench() — simple micro-benchmark
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Type

__all__ = [
    "assert_eq",
    "assert_ne",
    "assert_approx",
    "assert_raises",
    "assert_not_raises",
    "assert_in",
    "assert_not_in",
    "assert_true",
    "assert_false",
    "assert_is_none",
    "assert_is_not_none",
    "TestSuite",
    "bench",
    "run_tests",
]


def assert_eq(actual: Any, expected: Any, message: str = "") -> None:
    """Assert actual == expected."""
    if actual != expected:
        msg = message or f"expected {expected!r}, got {actual!r}"
        raise AssertionError(msg)


def assert_ne(actual: Any, unexpected: Any, message: str = "") -> None:
    """Assert actual != unexpected."""
    if actual == unexpected:
        msg = message or f"expected value to differ from {unexpected!r}"
        raise AssertionError(msg)


def assert_approx(
    actual: float,
    expected: float,
    tolerance: float,
    message: str = "",
) -> None:
    """Assert |actual − expected| <= tolerance."""
    diff = abs(float(actual) - float(expected))
    if diff > float(tolerance):
        msg = message or f"expected {expected} ± {tolerance}, got {actual} (diff={diff})"
        raise AssertionError(msg)


def assert_raises(
    exc_type: Type[BaseException],
    fn: Callable[[], Any],
    message: str = "",
) -> None:
    """Assert fn() raises exc_type."""
    try:
        fn()
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(
            message or f"expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        ) from e
    raise AssertionError(message or f"expected {exc_type.__name__} to be raised")


def assert_not_raises(fn: Callable[[], Any], message: str = "") -> None:
    """Assert fn() does not raise."""
    try:
        fn()
    except Exception as e:
        raise AssertionError(
            message or f"expected no exception, got {type(e).__name__}: {e}"
        ) from e


def assert_in(item: Any, container: Any, message: str = "") -> None:
    """Assert item in container."""
    if item not in container:
        raise AssertionError(message or f"{item!r} not found in {container!r}")


def assert_not_in(item: Any, container: Any, message: str = "") -> None:
    """Assert item not in container."""
    if item in container:
        raise AssertionError(message or f"{item!r} unexpectedly found in {container!r}")


def assert_true(value: Any, message: str = "") -> None:
    """Assert bool(value) is True."""
    if not value:
        raise AssertionError(message or f"expected truthy value, got {value!r}")


def assert_false(value: Any, message: str = "") -> None:
    """Assert bool(value) is False."""
    if value:
        raise AssertionError(message or f"expected falsy value, got {value!r}")


def assert_is_none(value: Any, message: str = "") -> None:
    """Assert value is None."""
    if value is not None:
        raise AssertionError(message or f"expected None, got {value!r}")


def assert_is_not_none(value: Any, message: str = "") -> None:
    """Assert value is not None."""
    if value is None:
        raise AssertionError(message or "expected non-None value")


# ── TestSuite ─────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    error: Optional[str] = None
    duration_s: float = 0.0


@dataclass
class TestSuite:
    """
    Lightweight test runner for NOVA stdlib tests.

    Usage:
        suite = TestSuite("cosmos.stats")
        suite.add(test_pearson)
        suite.add(test_linear_fit)
        results = suite.run()
        suite.print_summary(results)
    """
    name: str
    _tests: List[Callable[[], None]] = field(default_factory=list)

    def add(self, fn: Callable[[], None]) -> None:
        """Register a test function."""
        self._tests.append(fn)

    def run(self) -> List[TestResult]:
        """Run all registered tests; collect results."""
        results = []
        for fn in self._tests:
            t0 = time.perf_counter()
            try:
                fn()
                results.append(TestResult(
                    name=fn.__name__,
                    passed=True,
                    duration_s=time.perf_counter() - t0,
                ))
            except Exception:
                results.append(TestResult(
                    name=fn.__name__,
                    passed=False,
                    error=traceback.format_exc(),
                    duration_s=time.perf_counter() - t0,
                ))
        return results

    def print_summary(self, results: List[TestResult]) -> None:
        passed = sum(1 for r in results if r.passed)
        total  = len(results)
        print("\n" + f"{self.name}: {passed}/{total} passed")
        for r in results:
            status = "ok" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}  ({r.duration_s*1000:.1f} ms)")
            if r.error:
                for line in r.error.strip().splitlines()[-5:]:
                    print(f"         {line}")


def run_tests(module_or_list, name: str = "tests") -> bool:
    """
    Run all callables whose names start with 'test_' in module_or_list.
    Returns True if all pass.
    """
    import inspect
    if isinstance(module_or_list, list):
        fns = module_or_list
    else:
        fns = [
            v for k, v in vars(module_or_list).items()
            if k.startswith("test_") and callable(v)
        ]
    suite = TestSuite(name)
    for fn in fns:
        suite.add(fn)
    results = suite.run()
    suite.print_summary(results)
    return all(r.passed for r in results)


def bench(
    fn: Callable[[], Any],
    n: int = 1000,
    label: str = "",
) -> float:
    """
    Simple micro-benchmark: run fn() n times, return mean time in milliseconds.
    """
    t0 = time.perf_counter()
    for _ in range(int(n)):
        fn()
    elapsed = time.perf_counter() - t0
    mean_ms = elapsed * 1000.0 / int(n)
    lbl = label or getattr(fn, "__name__", "fn")
    print(f"bench {lbl}: {mean_ms:.4f} ms / call  (n={n})")
    return float(mean_ms)
