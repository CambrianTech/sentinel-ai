"""BBHRunner — Big-Bench Hard via lm-eval-harness.

BBH (Big-Bench Hard, Suzgun et al. 2022) is the 23-task subset of
BIG-Bench where prior LMs failed to surpass the average human rater.
Multi-task reasoning across logic, causal judgment, date understanding,
geometric shapes, etc. Open LLM Leaderboard v2 uses the
`leaderboard_bbh` task group with `acc_norm,none` (length-normalized
accuracy aggregated across the 23 subtasks).
"""

from __future__ import annotations

from .lm_eval_harness_base import LmEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class BBHRunner(LmEvalHarnessRunner):
    name = "bbh"
    task_name = "leaderboard_bbh"
    metric_key = "acc_norm,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(BBHRunner)
