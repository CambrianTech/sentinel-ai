"""HumanEvalPlusRunner — same scorer, plus-test pass@1.

evalplus's official CLI reports two pass@1 numbers in one run: HumanEval
(base inputs only) and HumanEval+ (base AND plus inputs both passing).
This runner returns the +plus number; HumanEvalRunner returns the base
number. Both wrap the same _humaneval_scorer.py invocation, so when an
alloy declares both benchmarks the actual evalplus subprocess only fires
once and the registry caches inside the canonical scorer (or, today,
runs twice — that's a future optimization).
"""

from __future__ import annotations

import sys
from pathlib import Path

from .base import BenchmarkRunner, ScoreResult
from .registry import BenchmarkRunnerRegistry

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "reproducibility"))


class HumanEvalPlusRunner(BenchmarkRunner):
    """HumanEval+ pass@1 — base AND plus tests both passing per evalplus's
    canonical convention."""

    name = "humaneval_plus"

    def score(self, samples_path: str | Path) -> ScoreResult:
        from _humaneval_scorer import score_jsonl
        result = score_jsonl(samples_path)
        section = result["humaneval_plus"]
        return ScoreResult(
            benchmark_name=self.name,
            pass_at_1=section["pass_at_1"],
            passed=section.get("passed"),
            total=section.get("total"),
            metric="pass@1",
            extras={"dataset_hash": result.get("dataset_hash")},
            samples_path=str(samples_path),
        )


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(HumanEvalPlusRunner)
