"""TDD spec: eval_with_calibration.run_benchmark migrates to registry dispatch.

The §4.1.4.1 anchor-reproduction discipline gate runs through
scripts/eval_with_calibration.py::run_benchmark, which historically used
a hand-rolled if-elif chain plus a NOT_YET_IMPLEMENTED dict to dispatch
to per-benchmark runner functions. That's the same kind of branching
shared-code anti-pattern the family-adapter axis was built to fix.

This test asserts the migration:

  1. BenchmarkRunner ABC declares an `evaluate(model_dir, output_dir,
     **kwargs) -> ScoreResult` method. Default raises NotImplementedError
     so stub runners fail loudly when invoked end-to-end (the same
     deterministic-rock signal score() already gives).
  2. The 9 real runners (humaneval, humaneval_plus, livecodebench_v6,
     ifeval, bbh, math_hard, gpqa, mmlu_pro, musr) all have a working
     evaluate() that does NOT raise NotImplementedError when introspected
     (we don't actually run inference here — that needs a GPU).
  3. The stub runners (sota_stubs.py + the LCB stub that already
     graduated) raise NotImplementedError from evaluate() — same loud
     failure as score().
  4. eval_with_calibration.run_benchmark uses registry dispatch
     (resolve_runner) instead of the old if-elif chain. The function
     body must reference 'resolve_runner' and must NOT contain the old
     NOT_YET_IMPLEMENTED dict.
  5. eval_with_calibration.NOT_YET_IMPLEMENTED is gone — the registry
     is the single source of truth for which benchmarks dispatch.
  6. Calling run_benchmark with an unknown benchmark name raises
     BenchmarkNotRegistered (the registry's exception), not the old
     ValueError("unknown benchmark...") string.

Written test-first per TDD discipline.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


REAL_RUNNERS = [
    "humaneval",
    "humaneval_plus",
    "livecodebench_v6",
    "ifeval",
    "bbh",
    "math_hard",
    "gpqa",
    "mmlu_pro",
    "musr",
]


# ── BenchmarkRunner.evaluate ABC default ────────────────────────────────────


def test_benchmark_runner_abc_has_evaluate_method():
    from eval_runners.base import BenchmarkRunner
    assert hasattr(BenchmarkRunner, "evaluate"), (
        "BenchmarkRunner ABC must declare evaluate(model_dir, output_dir, **kwargs)"
    )


def test_benchmark_runner_abc_evaluate_default_raises():
    """The ABC's default evaluate() raises NotImplementedError naming the
    runner — same deterministic-rock pattern as the score() default."""
    from eval_runners.base import BenchmarkRunner

    class _Concrete(BenchmarkRunner):
        name = "concrete_no_evaluate"
        def score(self, samples_path):
            return None

    with pytest.raises(NotImplementedError) as exc_info:
        _Concrete().evaluate("/nonexistent", "/tmp/out")
    assert "concrete_no_evaluate" in str(exc_info.value) or "_Concrete" in str(exc_info.value)


# ── Real runners declare evaluate (not the inherited stub) ──────────────────


@pytest.mark.parametrize("benchmark_name", REAL_RUNNERS)
def test_real_runner_overrides_evaluate(benchmark_name):
    """Each real runner MUST override evaluate() — calling the inherited
    BenchmarkRunner.evaluate stub is the failure mode this test catches."""
    from eval_runners import resolve_runner
    from eval_runners.base import BenchmarkRunner
    runner = resolve_runner(benchmark_name)
    # Resolve the bound method's underlying function and check it's not
    # the ABC's default. Both class- and instance-method overrides count.
    runner_eval = type(runner).evaluate
    abc_eval = BenchmarkRunner.evaluate
    assert runner_eval is not abc_eval, (
        f"{type(runner).__name__}.evaluate is the inherited ABC stub — "
        f"the real runner must override evaluate() with the codegen path "
        f"for this benchmark."
    )


# ── Stub runners still raise (loudly) on evaluate ───────────────────────────


STUB_RUNNERS = [
    "swe_bench_verified",
    "aider_polyglot",
    "mbpp_plus",
    "gsm8k",
    "aime_2024",
    # mmmu, chartqa, docvqa, ai2d graduated to real lmms-eval runners
    # (Open VLM Leaderboard pack). They no longer raise NotImplementedError
    # from evaluate() — they raise ImportError if lmms_eval isn't installed
    # and would otherwise actually run inference.
    "covost2",
    "librispeech",
    "gtzan",
]


@pytest.mark.parametrize("benchmark_name", STUB_RUNNERS)
def test_stub_runner_evaluate_raises(benchmark_name):
    """Stubs inherit the ABC default evaluate which raises NotImplementedError."""
    from eval_runners import resolve_runner
    runner = resolve_runner(benchmark_name)
    with pytest.raises(NotImplementedError):
        runner.evaluate("/nonexistent_model_dir", "/tmp/stub_out")


# ── eval_with_calibration.run_benchmark uses the registry ───────────────────


def test_run_benchmark_uses_registry_dispatch():
    import eval_with_calibration
    src = inspect.getsource(eval_with_calibration.run_benchmark)
    assert "resolve_runner" in src, (
        "run_benchmark must dispatch via eval_runners.resolve_runner — the old "
        "if-elif chain on benchmark name is the architectural smell this "
        "migration eliminates."
    )
    # The old NOT_YET_IMPLEMENTED dict and the if-elif body must be gone.
    assert "NOT_YET_IMPLEMENTED" not in src, (
        "run_benchmark must not reference NOT_YET_IMPLEMENTED — the registry "
        "is the single source of truth for which benchmarks dispatch."
    )


def test_not_yet_implemented_dict_is_gone():
    import eval_with_calibration
    assert not hasattr(eval_with_calibration, "NOT_YET_IMPLEMENTED"), (
        "eval_with_calibration.NOT_YET_IMPLEMENTED must be deleted — the "
        "registry is the single source of truth. Stubs raise NotImplementedError "
        "from their evaluate() method via the BenchmarkRunner ABC default."
    )


def test_run_benchmark_unknown_name_raises_registry_error():
    """An unknown benchmark name must raise BenchmarkNotRegistered (the
    registry's exception), not the old hand-rolled ValueError. This is
    the assertion that proves dispatch goes through resolve_runner."""
    from eval_runners import BenchmarkNotRegistered
    import eval_with_calibration
    with pytest.raises(BenchmarkNotRegistered):
        eval_with_calibration.run_benchmark(
            "definitely_not_a_real_benchmark",
            Path("/nonexistent"),
            Path("/tmp/test"),
            force_base_prompt=False,
        )
