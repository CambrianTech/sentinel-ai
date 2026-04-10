"""HumanEvalRunner — wraps the canonical evalplus HumanEval scorer.

The actual scoring infrastructure (the macOS reliability_guard +
fork-multiprocessing workaround) lives in
tests/reproducibility/_humaneval_scorer.py. This runner is a thin
adapter that delegates to it and returns a ScoreResult in the registry's
canonical shape.

The reason the scorer module lives under tests/reproducibility/ rather
than scripts/eval_runners/ is historical — it was built for the Tier 4
reproducibility test before the registry existed. Moving it would be
disruptive (the test imports it directly). The right shape going forward
is to import it from here and let scripts/ be the source of truth, but
that's a follow-up move; for now the import path is the test directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .base import BenchmarkRunner, ScoreResult
from .registry import BenchmarkRunnerRegistry


# Make the canonical scorer importable from this module. The scorer is in
# tests/reproducibility because that's where it was first built; the import
# path is added at module load so the runner can wrap it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "reproducibility"))


class HumanEvalRunner(BenchmarkRunner):
    """HumanEval (base) pass@1 runner — uses evalplus's official CLI via
    the macOS-safe subprocess wrapper in _humaneval_scorer.py."""

    name = "humaneval"

    def score(self, samples_path: str | Path) -> ScoreResult:
        from _humaneval_scorer import score_jsonl
        result = score_jsonl(samples_path)
        section = result["humaneval"]
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
        """Run evalplus.codegen + sanitize + evaluate on a model directory.

        Delegates to eval_with_calibration.run_humaneval (the canonical
        subprocess wrapper that has the working evalplus CLI invocation
        baked in). Returns a ScoreResult with both humaneval and humaneval+
        scores — humaneval+ lives in extras['humaneval_plus_pass_at_1']
        because evalplus emits both in one pass.

        kwargs:
            force_base_prompt: bool — pass --force-base-prompt to evalplus
                                       (use for base models, not instruct).
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
            pass_at_1=float(scores.get("humaneval", 0.0)) / 100.0,
            metric="pass@1",
            extras={
                "humaneval_plus_pass_at_1": float(scores.get("humaneval_plus", 0.0)) / 100.0,
                "log_path": result_dict.get("log_path"),
            },
            samples_path=result_dict.get("samples_path"),
        )


def register(reg: BenchmarkRunnerRegistry) -> None:
    """Register this runner with a registry instance. Called at module
    import time from scripts/eval_runners/__init__.py."""
    reg.register(HumanEvalRunner)
