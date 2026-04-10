"""eval_runners — benchmark eval-runner registry for the family-adapter set.

The second axis of dispatch on the .eval() stage:

    Stage type → StageExecutor (existing first axis)
        eval → EvalExecutor → family.eval(ctx, **stage_params)

    Family adapter (existing second axis)
        family.eval(ctx, **params) → for each benchmark in params['benchmarks']:
                                         resolve_runner(benchmark.name).score(samples_path)

    Benchmark name → BenchmarkRunner (this module — new third axis)
        humaneval        → HumanEvalRunner
        humaneval_plus   → HumanEvalPlusRunner
        (future) mmlu    → MMLURunner
        (future) swe_bench → SWEBenchRunner
        (future) mmmu    → MMMURunner
        (future) gsm8k   → GSM8KRunner

Adding a new benchmark suite is exactly:
    1. Create scripts/eval_runners/<name>.py with a BenchmarkRunner subclass
    2. Add `from . import <name>` to this file
    3. Done — the registry handles dispatch, family adapters need zero changes

Architectural rule: NEVER add `if benchmark_name == ...` branches to
FamilyAdapter.eval. NEVER add per-family eval logic that bypasses the
registry. New benchmark = new runner file. This is the same OOP rule as
the family-adapter axis: write a new file, register it, never branch
shared code.
"""

from .base import BenchmarkRunner, ScoreResult
from .registry import BenchmarkRunnerRegistry, BenchmarkNotRegistered

# Module-level singleton — the canonical registry the family-adapter
# default eval() reads from.
_REGISTRY = BenchmarkRunnerRegistry()


def resolve_runner(benchmark_name: str) -> BenchmarkRunner:
    """Look up and instantiate the runner for a benchmark name.

    Used by FamilyAdapter.eval() to dispatch each declared benchmark to
    the right runner. Raises BenchmarkNotRegistered if the name isn't
    in the registry — loud failure pointing at the missing runner file.
    """
    return _REGISTRY.resolve(benchmark_name)


def registered_benchmarks() -> list[str]:
    """All benchmark names the registry currently knows about."""
    return _REGISTRY.benchmark_names()


# Importing each concrete runner module triggers its register() call
# against the singleton. Order doesn't matter; the registry is keyed by
# benchmark name, not by import order. NEW runner = new module here.
from . import humaneval         # noqa: E402,F401
from . import humaneval_plus    # noqa: E402,F401
from . import livecodebench_v6  # noqa: E402,F401  — real LCB v6 scorer
# Open LLM Leaderboard v2 runner pack (lm-eval-harness backed)
from . import ifeval            # noqa: E402,F401
from . import bbh               # noqa: E402,F401
from . import math_hard         # noqa: E402,F401
from . import gpqa              # noqa: E402,F401
from . import mmlu_pro          # noqa: E402,F401
from . import musr              # noqa: E402,F401
# Open VLM Leaderboard pack (lmms-eval backed)
from . import mmmu              # noqa: E402,F401
from . import chartqa           # noqa: E402,F401
from . import docvqa            # noqa: E402,F401
from . import ai2d              # noqa: E402,F401
from . import sota_stubs        # noqa: E402,F401  — remaining frontier-target runner stubs

humaneval.register(_REGISTRY)
humaneval_plus.register(_REGISTRY)
livecodebench_v6.register(_REGISTRY)
ifeval.register(_REGISTRY)
bbh.register(_REGISTRY)
math_hard.register(_REGISTRY)
gpqa.register(_REGISTRY)
mmlu_pro.register(_REGISTRY)
musr.register(_REGISTRY)
mmmu.register(_REGISTRY)
chartqa.register(_REGISTRY)
docvqa.register(_REGISTRY)
ai2d.register(_REGISTRY)
sota_stubs.register(_REGISTRY)


__all__ = [
    "BenchmarkRunner",
    "ScoreResult",
    "BenchmarkRunnerRegistry",
    "BenchmarkNotRegistered",
    "resolve_runner",
    "registered_benchmarks",
]
