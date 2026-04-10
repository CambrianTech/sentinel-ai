"""MATHHardRunner — MATH level 5 (hardest tier) via lm-eval-harness.

MATH (Hendrycks et al. 2021) is 12.5K competition math problems graded
1-5 by difficulty. Open LLM Leaderboard v2 scores ONLY level 5 (the
hardest tier) under `leaderboard_math_hard` to avoid the saturation
of the easier levels on frontier models. Metric: `exact_match,none`
(the model's final boxed answer must match the ground-truth boxed
answer string-for-string after normalization).
"""

from __future__ import annotations

from .lm_eval_harness_base import LmEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class MATHHardRunner(LmEvalHarnessRunner):
    name = "math_hard"
    task_name = "leaderboard_math_hard"
    metric_key = "exact_match,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(MATHHardRunner)
