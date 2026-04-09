"""TDD spec for SOTA benchmark runners — the frontier-target eval suite.

The eval-runner registry built in roadmap step 4 has 2 runners today
(humaneval, humaneval_plus). The SOTA targets Kash mapped in the
frontier-roadmap discussion need different benchmarks:

  Qwen3-Coder-480B-A35B    → SWE-Bench Verified, LiveCodeBench v6, Aider-Polyglot
  Qwen3-235B-A22B          → MMLU-Pro, GSM8K, IFEval, GPQA Diamond
  DeepSeek-V3.1            → SWE-Bench Verified, LiveCodeBench v6, AIME 2024
  Qwen2.5-VL / Qwen3.5-VL  → MMMU, ChartQA, DocVQA, AI2D
  Qwen2.5-Omni             → COVOST 2, GTZAN, LibriSpeech (audio)

This step adds REGISTERED runners for the SOTA benchmark suite. Each
runner is a stub that declares its name + source dataset + scoring
algorithm in the docstring. The score() method raises NotImplementedError
TODAY because none of these benchmarks have been wired through the
sentinel-ai eval pipeline yet — the runner registration is the
ARCHITECTURAL CONTRACT that lets the alloy_executor dispatch a frontier
forge's eval stage to a named runner without any other code change.

Wait — "raises NotImplementedError" sounds like the f-word stub
pattern. Why is this OK here?

Because: the SOTA benchmarks have NOT been implemented yet. There is no
"correct architecture" path to silently substitute. The runner exists
as a registered name so the dispatch path resolves cleanly; when the
first real Qwen3-Coder-480B forge run happens on BigMama, the
score() method gets its real body in a focused commit that's gated by
its own TDD test (read the actual SWE-Bench Verified protocol, score
samples per the official harness, return ScoreResult). The
NotImplementedError today is LOUD — calling score() raises immediately
with a clear pointer at the runner file that needs the implementation,
not a silent default value.

The contract this test enforces:
  - Every SOTA benchmark name has a registered runner class
  - Each runner has a name attribute matching its registry key
  - Each runner's score() raises NotImplementedError with a clear
    message naming the benchmark and the documented protocol source
  - The dispatch path can resolve every SOTA benchmark name without
    raising BenchmarkNotRegistered

Written test-first per TDD/TDValidation discipline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# The full SOTA benchmark surface from Kash's frontier-target mapping +
# the multimodal roadmap. Each entry is the name string the alloy's
# eval.benchmarks[].name field carries.
SOTA_BENCHMARKS = [
    # Code benchmarks (frontier coder targets).
    # NOTE: livecodebench_v6 was previously here as a stub; it graduated to
    # a real runner in scripts/eval_runners/livecodebench_v6.py per the
    # §4.1.4.1 anchor-reproduction discipline gate prerequisite work.
    # Test coverage for it lives in test_livecodebench_v6_runner.py now.
    "swe_bench_verified",
    "aider_polyglot",
    "mbpp_plus",                 # complement to humaneval+
    # General-purpose benchmarks (frontier general targets)
    # NOTE: mmlu_pro, gpqa, ifeval, bbh, math_hard, musr graduated to real
    # lm-eval-harness runners (Open LLM Leaderboard v2 pack). Coverage now
    # in test_open_llm_leaderboard_v2_runners.py.
    "gsm8k",
    "aime_2024",
    # Vision benchmarks (Qwen2.5-VL / Qwen3.5-VL targets)
    "mmmu",
    "chartqa",
    "docvqa",
    "ai2d",
    # Audio benchmarks (Qwen2.5-Omni target)
    "covost2",
    "librispeech",
    "gtzan",
]


@pytest.mark.parametrize("benchmark_name", SOTA_BENCHMARKS)
def test_sota_runner_is_registered(benchmark_name: str):
    """Every SOTA benchmark name MUST resolve through the registry to a
    BenchmarkRunner instance. Adding a new SOTA target via a new
    eval_runners file flips the entry from missing to passing."""
    from eval_runners import resolve_runner
    runner = resolve_runner(benchmark_name)
    assert runner is not None
    assert runner.name == benchmark_name


@pytest.mark.parametrize("benchmark_name", SOTA_BENCHMARKS)
def test_sota_runner_score_raises_loudly(benchmark_name: str):
    """The score() method MUST raise NotImplementedError with a message
    naming the benchmark + the documented protocol source. This is the
    deterministic-rock principle: the runner exists so dispatch resolves,
    but calling it before the real implementation lands fails LOUDLY at
    the runner file site, not silently with a wrong score.

    When a real implementation lands for any of these, this test entry
    naturally needs to be updated to assert the real behavior — that's
    the TDD signal that the runner went from stub to real."""
    from eval_runners import resolve_runner
    runner = resolve_runner(benchmark_name)
    with pytest.raises(NotImplementedError) as exc_info:
        runner.score("/dev/null")
    msg = str(exc_info.value)
    assert benchmark_name in msg or runner.name in msg, (
        f"score() error message must name the benchmark {benchmark_name!r} "
        f"so a developer reading the failure knows which runner file to fill in. "
        f"Got: {exc_info.value}"
    )


def test_registered_benchmarks_includes_all_sota():
    """The registry's full list MUST include every SOTA benchmark name +
    the existing humaneval / humaneval_plus from Step 4."""
    from eval_runners import registered_benchmarks
    actual = set(registered_benchmarks())
    expected = set(SOTA_BENCHMARKS) | {
        "humaneval", "humaneval_plus", "livecodebench_v6",
        # Open LLM Leaderboard v2 pack
        "ifeval", "bbh", "math_hard", "gpqa", "mmlu_pro", "musr",
    }
    missing = expected - actual
    assert not missing, (
        f"registered benchmarks missing: {sorted(missing)}\n"
        f"Got: {sorted(actual)}"
    )
