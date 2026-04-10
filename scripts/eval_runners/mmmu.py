"""MMMURunner — Massive Multi-discipline Multimodal Understanding via lmms-eval.

MMMU (Yue et al. 2024) is 11.5K college-exam-style multimodal questions
across 6 disciplines: art & design, business, science, health & medicine,
humanities & social science, technology & engineering. The headline VL
benchmark every modern frontier VL model reports against. Open VLM
Leaderboard uses the `mmmu_val` task in lmms-eval (the validation split,
4.4K questions, the published-anchor convention).
"""

from __future__ import annotations

from .lmms_eval_harness_base import LmmsEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class MMMURunner(LmmsEvalHarnessRunner):
    name = "mmmu"
    task_name = "mmmu_val"
    metric_key = "mmmu_acc,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(MMMURunner)
