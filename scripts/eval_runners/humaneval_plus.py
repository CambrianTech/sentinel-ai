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
from typing import Any

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

    def evaluate(
        self,
        model_dir: str | Path,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> ScoreResult:
        """Run evalplus end-to-end and return the humaneval+ pass@1.

        Same subprocess invocation as HumanEvalRunner.evaluate (evalplus
        emits both numbers in one pass); this method just returns the
        +plus score as the headline pass_at_1, with the base humaneval
        score in extras['humaneval_pass_at_1'].
        """
        from eval_with_calibration import run_humaneval as _run
        result_dict = _run(
            Path(model_dir),
            Path(output_dir),
            force_base_prompt=bool(kwargs.get("force_base_prompt", False)),
        )
        scores = result_dict.get("scores", {})
        return ScoreResult(
            benchmark_name=self.name,
            pass_at_1=float(scores.get("humaneval_plus", 0.0)) / 100.0,
            metric="pass@1",
            extras={
                "humaneval_pass_at_1": float(scores.get("humaneval", 0.0)) / 100.0,
                "log_path": result_dict.get("log_path"),
            },
            samples_path=result_dict.get("samples_path"),
        )


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(HumanEvalPlusRunner)
