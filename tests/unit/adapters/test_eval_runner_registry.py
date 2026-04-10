"""TDD spec for the benchmark eval-runner registry.

Roadmap step 4 from docs/PLUGIN-SPRINT.md: family adapters dispatch
benchmark evaluation through a runner registry instead of carrying their
own per-benchmark code. Adding a new benchmark suite (HumanEval, MMLU,
SWE-Bench, MMMU, ...) is one new file in scripts/eval_runners/ plus one
import line. The registry is what unblocks frontier targets — Qwen3-Coder-480B
uses SWE-Bench Pro instead of HumanEval, and the frontier coder cards
report against LiveCodeBench v6, not HumanEval. Without the registry,
adding those benchmarks would mean editing every family adapter's eval()
method. With the registry, it's a single new runner file.

Written test-first per TDD/TDValidation discipline. The contract this
test asserts IS the spec the implementation must satisfy. Tests do NOT
load any model and do NOT actually score against real samples (the
HumanEvalRunner case touches the existing Tier 4 scorer's subprocess
path so it WILL run evalplus on the cached samples; that's the
end-to-end smoke test for the runner wiring).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── BenchmarkRunner ABC ─────────────────────────────────────────────────────


def test_benchmark_runner_base_is_importable():
    """The BenchmarkRunner ABC MUST be importable from eval_runners.base."""
    from eval_runners.base import BenchmarkRunner
    assert BenchmarkRunner is not None
    # ABC has at least the methods we expect
    assert hasattr(BenchmarkRunner, "name")
    assert hasattr(BenchmarkRunner, "score")


def test_benchmark_runner_score_signature():
    """BenchmarkRunner.score MUST take a samples_path and return a
    ScoreResult with at least pass_at_1 + total + passed fields.

    The exact signature: score(self, samples_path: str | Path) -> ScoreResult
    """
    from eval_runners.base import BenchmarkRunner
    sig = inspect.signature(BenchmarkRunner.score)
    params = list(sig.parameters.keys())
    assert params == ["self", "samples_path"], (
        f"BenchmarkRunner.score signature must be (self, samples_path), got {params}"
    )


def test_score_result_shape():
    """ScoreResult MUST carry the canonical eval result fields:
       benchmark_name, pass_at_1 (0..1 fraction), passed, total."""
    from eval_runners.base import ScoreResult
    # Should be constructible with the canonical kwargs
    r = ScoreResult(
        benchmark_name="humaneval",
        pass_at_1=0.884,
        passed=145,
        total=164,
    )
    assert r.benchmark_name == "humaneval"
    assert r.pass_at_1 == 0.884
    assert r.passed == 145
    assert r.total == 164


# ── BenchmarkRunnerRegistry ─────────────────────────────────────────────────


def test_registry_is_importable():
    from eval_runners import BenchmarkRunnerRegistry, resolve_runner
    assert BenchmarkRunnerRegistry is not None
    assert callable(resolve_runner)


def test_registry_register_and_resolve():
    """A custom runner can be registered and resolved by name."""
    from eval_runners import BenchmarkRunnerRegistry
    from eval_runners.base import BenchmarkRunner, ScoreResult

    class MockRunner(BenchmarkRunner):
        name = "mock-bench"
        def score(self, samples_path):
            return ScoreResult(benchmark_name=self.name, pass_at_1=0.5, passed=82, total=164)

    reg = BenchmarkRunnerRegistry()
    reg.register(MockRunner)
    runner = reg.resolve("mock-bench")
    assert isinstance(runner, MockRunner)
    result = runner.score("/dev/null")
    assert result.benchmark_name == "mock-bench"
    assert result.pass_at_1 == 0.5


def test_registry_unknown_benchmark_raises_clearly():
    """Unknown benchmark name MUST raise BenchmarkNotRegistered with a
    message naming the requested benchmark and listing what IS registered."""
    from eval_runners import BenchmarkRunnerRegistry, BenchmarkNotRegistered
    from eval_runners.base import BenchmarkRunner, ScoreResult

    class FooRunner(BenchmarkRunner):
        name = "foo"
        def score(self, samples_path):
            return ScoreResult(benchmark_name="foo", pass_at_1=0.0, passed=0, total=1)

    reg = BenchmarkRunnerRegistry()
    reg.register(FooRunner)
    with pytest.raises(BenchmarkNotRegistered) as exc_info:
        reg.resolve("not-registered")
    msg = str(exc_info.value)
    assert "not-registered" in msg
    assert "foo" in msg, "error message must list registered runners"


def test_registry_double_register_different_class_raises():
    """Re-registering a DIFFERENT class against an existing name MUST raise
    (silent shadowing is the f-word pattern). Re-registering the SAME class
    is idempotent."""
    from eval_runners import BenchmarkRunnerRegistry
    from eval_runners.base import BenchmarkRunner, ScoreResult

    class A(BenchmarkRunner):
        name = "shared"
        def score(self, samples_path):
            return ScoreResult(benchmark_name="shared", pass_at_1=0.0, passed=0, total=1)

    class B(BenchmarkRunner):
        name = "shared"
        def score(self, samples_path):
            return ScoreResult(benchmark_name="shared", pass_at_1=1.0, passed=1, total=1)

    reg = BenchmarkRunnerRegistry()
    reg.register(A)
    reg.register(A)  # idempotent — same class
    with pytest.raises(ValueError) as exc_info:
        reg.register(B)
    assert "shared" in str(exc_info.value)


# ── Built-in runners ────────────────────────────────────────────────────────


def test_humaneval_runner_is_registered_globally():
    """HumanEvalRunner MUST be registered against name 'humaneval' on the
    module-level singleton registry, so any call to resolve_runner('humaneval')
    works without explicit registration."""
    from eval_runners import resolve_runner
    from eval_runners.humaneval import HumanEvalRunner
    runner = resolve_runner("humaneval")
    assert isinstance(runner, HumanEvalRunner)
    assert runner.name == "humaneval"


def test_humaneval_plus_runner_is_registered_globally():
    """HumanEval+ scoring is a separate registered runner so callers can
    request either or both via the alloy's eval.benchmarks[].name field."""
    from eval_runners import resolve_runner
    from eval_runners.humaneval_plus import HumanEvalPlusRunner
    runner = resolve_runner("humaneval_plus")
    assert isinstance(runner, HumanEvalPlusRunner)
    assert runner.name == "humaneval_plus"


def test_humaneval_runner_scores_a_real_published_jsonl():
    """End-to-end smoke test: the HumanEvalRunner wraps the existing
    canonical scorer (tests/reproducibility/_humaneval_scorer.py) and
    reproduces the published 88.4 HumanEval pass@1 for the morning's
    flagship qwen3-coder-30b-a3b student samples.

    This is the validation test that proves the registry-driven path
    matches the standalone Tier 4 scorer byte-for-byte.
    """
    from eval_runners import resolve_runner
    samples = REPO_ROOT / "tests/reproducibility/_cache/samples/continuum-ai_qwen3-coder-30b-a3b-compacted-19b-256k__eval_humaneval_student_samples.jsonl"
    if not samples.exists():
        pytest.skip(f"published sample file not in cache: {samples}")
    runner = resolve_runner("humaneval")
    result = runner.score(samples)
    assert result.benchmark_name == "humaneval"
    # Published headline is 88.4. Canonical evalplus reports 0.884.
    assert abs(result.pass_at_1 - 0.884) < 0.005, (
        f"HumanEvalRunner reported pass@1={result.pass_at_1}, expected ≈0.884 "
        f"(the morning's flagship qwen3-coder-30b-a3b student samples)"
    )
    assert result.passed == 145
    assert result.total == 164


# ── FamilyAdapter.eval() integration ────────────────────────────────────────


def test_family_adapter_eval_dispatches_through_registry():
    """FamilyAdapter.eval(ctx, benchmarks=[{'name': 'humaneval'}, ...]) MUST
    iterate the benchmark list, look up each runner by name via resolve_runner,
    and invoke runner.score on the appropriate samples file. The default
    eval() in the base class does this so EVERY family adapter inherits the
    correct dispatch with no per-family code."""
    from adapters.base import FamilyAdapter
    import inspect as _inspect
    src = _inspect.getsource(FamilyAdapter.eval)
    # The body MUST reference resolve_runner / BenchmarkRunnerRegistry / eval_runners
    assert "resolve_runner" in src or "eval_runners" in src, (
        "FamilyAdapter.eval must dispatch through the eval_runners registry. "
        "If it doesn't, adding a new benchmark requires editing every family "
        "adapter — which is exactly what the registry exists to prevent."
    )
