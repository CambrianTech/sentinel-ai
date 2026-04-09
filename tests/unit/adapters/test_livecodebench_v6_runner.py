"""TDD spec for LiveCodeBenchV6Runner — the hard prerequisite for the
Qwen3-Coder-480B / Mixtral 8x22B / DeepSeek-V3.1 frontier forge plays.

Per Kash's frontier-target analysis (2026-04-08): HumanEval is dead for
frontier coder cards. Qwen3-Coder, Qwen3, Mixtral 8x22B, and DeepSeek-V3.1
all use SWE-Bench / LiveCodeBench v6 / Aider-Polyglot. The §4.1.4.1
anchor-reproduction discipline gate can't run on any frontier target
without LCB v6 wired through the new registry.

This step moves LiveCodeBenchV6Runner out of sota_stubs.py into its own
file with a REAL score() body that invokes lcb_runner's canonical
scoring API on an existing samples JSONL. The runner is importable on
a Mac without lcb_runner installed (lazy import inside score()) so
the Tier 1 dispatch path stays Mac-safe. The actual scoring runs on
any machine where lcb_runner is installed (BigMama, the eval-runner
containers, etc).

Written test-first per TDD/TDValidation discipline. The contract this
test asserts IS the spec the runner must satisfy.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ── Module exists and is importable on Mac without lcb_runner ───────────────


def test_livecodebench_v6_module_is_importable():
    """The runner module MUST be importable without lcb_runner installed.
    Lazy import inside score() — same pattern as _humaneval_scorer's
    macOS-safe wrap."""
    import importlib
    mod = importlib.import_module("eval_runners.livecodebench_v6")
    assert mod is not None
    assert hasattr(mod, "LiveCodeBenchV6Runner")


def test_livecodebench_v6_runner_is_registered_in_singleton():
    """resolve_runner('livecodebench_v6') MUST return a LiveCodeBenchV6Runner
    instance from its dedicated file, NOT the sota_stubs.py stub. The
    stub gets dropped from sota_stubs in this commit."""
    from eval_runners import resolve_runner
    from eval_runners.livecodebench_v6 import LiveCodeBenchV6Runner
    runner = resolve_runner("livecodebench_v6")
    assert isinstance(runner, LiveCodeBenchV6Runner), (
        f"Expected LiveCodeBenchV6Runner from eval_runners/livecodebench_v6.py, "
        f"got {type(runner).__name__} — sota_stubs.py probably still has the "
        f"stub registered. Move it out and drop the entry from sota_stubs's "
        f"REGISTRATIONS list."
    )


def test_livecodebench_v6_runner_name_is_correct():
    """The .name class attribute MUST be 'livecodebench_v6' to match the
    benchmark name string the alloy's eval.benchmarks[].name field carries."""
    from eval_runners.livecodebench_v6 import LiveCodeBenchV6Runner
    assert LiveCodeBenchV6Runner.name == "livecodebench_v6"


# ── Score body is REAL, not the stub ────────────────────────────────────────


def test_score_body_is_not_a_stub_raise():
    """LiveCodeBenchV6Runner.score MUST have a real body that lazy-imports
    lcb_runner and invokes it on the samples_path. Calling it without
    lcb_runner installed should raise a CLEAR error pointing at the
    install instructions, not the generic sota_stubs NotImplementedError."""
    from eval_runners.livecodebench_v6 import LiveCodeBenchV6Runner
    src = inspect.getsource(LiveCodeBenchV6Runner.score)
    # Body must reference lcb_runner (the canonical lcb scoring module)
    assert "lcb_runner" in src, (
        "LiveCodeBenchV6Runner.score must lazy-import lcb_runner and use it "
        "for scoring. The stub-style 'raise NotImplementedError' pattern from "
        "sota_stubs.py is gone — this runner has a real implementation."
    )
    # Body must NOT contain the sota_stubs raise pattern
    assert "_stub_score_raise" not in src, (
        "LiveCodeBenchV6Runner.score must not delegate to sota_stubs's "
        "_stub_score_raise — it's a real runner now, not a stub."
    )


def test_score_raises_clear_error_when_lcb_not_installed():
    """When called on a Mac without lcb_runner installed, the runner MUST
    raise an ImportError (or RuntimeError wrapping one) with a clear
    message naming lcb_runner and pointing at the install path. NOT a
    generic ModuleNotFoundError that the caller has to dig through.
    """
    from eval_runners.livecodebench_v6 import LiveCodeBenchV6Runner
    runner = LiveCodeBenchV6Runner()
    # Use a valid-looking samples path (the runner should fail at the
    # lcb_runner import step, before touching the file)
    with pytest.raises((ImportError, RuntimeError)) as exc_info:
        runner.score("/dev/null")
    msg = str(exc_info.value)
    assert "lcb_runner" in msg or "livecodebench" in msg.lower(), (
        f"score() should fail loudly with a message naming lcb_runner so "
        f"the operator knows what to install. Got: {exc_info.value}"
    )


# ── sota_stubs no longer carries the LCB v6 stub ────────────────────────────


def test_livecodebench_v6_dropped_from_sota_stubs():
    """The LiveCodeBenchV6Runner stub class MUST be removed from sota_stubs.py
    once the real runner lands. Otherwise the registry would have a
    duplicate registration conflict."""
    from eval_runners import sota_stubs
    assert not hasattr(sota_stubs, "LiveCodeBenchV6Runner"), (
        "LiveCodeBenchV6Runner stub class is still in sota_stubs.py — drop it "
        "and remove its entry from REGISTRATIONS now that the real runner "
        "lives in eval_runners/livecodebench_v6.py."
    )


# ── ScoreResult shape contract ──────────────────────────────────────────────


def test_score_returns_score_result_shape_when_lcb_available():
    """If lcb_runner IS installed (CI / BigMama), the score() body must
    return a ScoreResult with the canonical fields populated. This test
    is skipped if lcb_runner isn't installed locally; it gates the
    contract for any environment where it IS installed."""
    try:
        import lcb_runner  # noqa: F401
    except ImportError:
        pytest.skip("lcb_runner not installed in this environment")
    from eval_runners.livecodebench_v6 import LiveCodeBenchV6Runner
    from eval_runners.base import ScoreResult
    runner = LiveCodeBenchV6Runner()
    # Smoke-test against an empty / placeholder JSONL
    fixture = REPO_ROOT / "tests/unit/adapters/_fixtures/lcb_v6_empty.jsonl"
    if not fixture.exists():
        pytest.skip(f"no LCB v6 fixture at {fixture}")
    result = runner.score(fixture)
    assert isinstance(result, ScoreResult)
    assert result.benchmark_name == "livecodebench_v6"
    assert result.metric == "pass@1"
