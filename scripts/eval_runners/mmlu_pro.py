"""MMLUProRunner — MMLU-Pro (TIGER-Lab harder MMLU) via lm-eval-harness.

MMLU-Pro (Wang et al. 2024) is the harder MMLU successor: 12K
multi-domain questions with 10 answer choices each (vs MMLU's 4),
explicitly constructed to break the saturation of the original MMLU on
frontier models. Open LLM Leaderboard v2 uses `leaderboard_mmlu_pro`
with the `acc,none` metric.
"""

from __future__ import annotations

from .lm_eval_harness_base import LmEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class MMLUProRunner(LmEvalHarnessRunner):
    name = "mmlu_pro"
    task_name = "leaderboard_mmlu_pro"
    metric_key = "acc,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(MMLUProRunner)
