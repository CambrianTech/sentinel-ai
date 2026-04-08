"""BenchmarkRunnerRegistry — name → BenchmarkRunner class lookup.

Mirrors the AdapterRegistry pattern in scripts/adapters/registry.py.
Strict exact-match dispatch on the benchmark name string. Re-registering
a different class against an existing name raises (silent shadowing is
the f-word pattern). Unknown name raises BenchmarkNotRegistered with a
clear message naming what IS registered.

The module-level singleton lives in scripts/eval_runners/__init__.py;
this file just defines the class and the exception type.
"""

from __future__ import annotations

from .base import BenchmarkRunner


class BenchmarkNotRegistered(KeyError):
    """Raised when resolve() is called with an unknown benchmark name.

    The error message names the requested benchmark and lists every
    registered name so the caller can fix the typo or add the missing
    runner. Subclasses KeyError so callers that want to catch "missing
    benchmark" specifically can do so without catching every KeyError.
    """


class BenchmarkRunnerRegistry:
    """Benchmark name → BenchmarkRunner class lookup."""

    def __init__(self) -> None:
        self._runners: dict[str, type[BenchmarkRunner]] = {}

    def register(self, runner_class: type[BenchmarkRunner]) -> type[BenchmarkRunner]:
        """Register a BenchmarkRunner subclass under its `name` class attribute.
        Idempotent for the same class. Raises ValueError if a DIFFERENT
        class is registered under an existing name (silent override would
        let one runner shadow another and is exactly the kind of
        unfindable-bug surface the f-word rule prohibits)."""
        name = getattr(runner_class, "name", "")
        if not name:
            raise ValueError(
                f"{runner_class.__name__} has no .name class attribute — set "
                f"it to the benchmark name string this runner answers to."
            )
        existing = self._runners.get(name)
        if existing is not None and existing is not runner_class:
            raise ValueError(
                f"benchmark name {name!r} is already registered to "
                f"{existing.__name__}; cannot also register {runner_class.__name__}. "
                f"If this is a methodology upgrade, register under a NEW name "
                f"so old alloys still resolve to the original runner for "
                f"reproducibility."
            )
        self._runners[name] = runner_class
        return runner_class

    def resolve(self, benchmark_name: str) -> BenchmarkRunner:
        """Look up the runner for a benchmark name and instantiate it."""
        runner_class = self._runners.get(benchmark_name)
        if runner_class is None:
            registered = sorted(self._runners.keys())
            raise BenchmarkNotRegistered(
                f"no BenchmarkRunner registered for name={benchmark_name!r}. "
                f"Registered runners: {registered}. To add a new benchmark, "
                f"create scripts/eval_runners/<name>.py with a BenchmarkRunner "
                f"subclass that sets name = '{benchmark_name}', then import it "
                f"from scripts/eval_runners/__init__.py."
            )
        return runner_class()

    def benchmark_names(self) -> list[str]:
        """All registered benchmark names, sorted."""
        return sorted(self._runners.keys())
