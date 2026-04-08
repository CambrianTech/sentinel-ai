"""BenchmarkRunner ABC + ScoreResult dataclass.

The benchmark eval-runner registry pattern is the second axis of dispatch
on the family adapters' .eval() method. It mirrors the family-adapter
registry one layer down: just as family adapters dispatch by
source.architecture, benchmark runners dispatch by benchmark name.

Adding a new benchmark suite (HumanEval, MMLU, SWE-Bench, MMMU, GSM8K,
LiveCodeBench v6, ...) is one new file in scripts/eval_runners/ that
defines a BenchmarkRunner subclass and one import line in __init__.py.
The registry handles dispatch.

This is the architectural piece that unblocks frontier targets:
Qwen3-Coder-480B uses SWE-Bench Pro instead of HumanEval, the frontier
coder cards report against LiveCodeBench v6, vision targets use MMMU.
Without the registry, adding those would mean editing every family
adapter's .eval() method. With the registry, it's a single new runner
file plus a registration call.

NEVER add `if benchmark_name == ...` branches to FamilyAdapter.eval.
NEVER add per-family eval logic that bypasses the registry. New
benchmark = new runner file. Old runners stay frozen forever so old
alloys keep scoring identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScoreResult:
    """Canonical eval result. Every BenchmarkRunner.score() returns one of these.

    Fields:
        benchmark_name: the registered name the runner answers to (e.g.
                        'humaneval', 'humaneval_plus', 'swe_bench_verified',
                        'livecodebench_v6', 'mmlu', 'mmmu')
        pass_at_1:      0..1 fraction matching evalplus's CLI convention
                        (multiply by 100 for the percentage form alloys publish)
        passed:         number of passing problems / examples
        total:          total problems / examples in the benchmark
        metric:         the metric name as it appears in the alloy's
                        results.benchmarks[].metric field (default 'pass@1')
        extras:         optional dict for runner-specific extras (sub-scores,
                        per-task details, dataset hash, etc.). Lands in the
                        alloy's results.benchmarks[].metrics dict.
        samples_path:   path to the per-problem samples file the score was
                        computed from. Required so the publish pipeline can
                        injection-hash it into integrity.benchmarks[].resultHash
                        for the brand-integrity chain of custody.
    """
    benchmark_name: str
    pass_at_1: float
    passed: int | None = None
    total: int | None = None
    metric: str = "pass@1"
    extras: dict[str, Any] = field(default_factory=dict)
    samples_path: str | None = None


class BenchmarkRunner(ABC):
    """Abstract base for one benchmark suite's evaluation runner.

    Subclasses MUST set the `name` class attribute (the string the registry
    dispatches off and the alloy's eval.benchmarks[].name field carries).
    Subclasses MUST implement score(samples_path) which returns a
    ScoreResult. Subclasses MAY optionally implement codegen(model, tokenizer,
    output_dir) for benchmarks where this script also generates the samples
    (vs. the typical case where samples are produced by an upstream stage
    and only scoring lives in the runner).
    """

    name: str = ""  # subclass overrides

    @abstractmethod
    def score(self, samples_path: str | Path) -> ScoreResult:
        """Score a per-problem samples JSONL against the canonical dataset
        for this benchmark and return a ScoreResult.

        The samples file is the output of an upstream codegen step (e.g.
        evalplus.codegen wrote it during the forge run, or the family
        adapter's eval() method wrote it just before calling score). This
        method does NOT run the model — it scores already-generated samples.
        """
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
