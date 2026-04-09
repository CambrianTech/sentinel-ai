"""AI2DRunner — Allen Institute Diagrams via lmms-eval.

AI2D (Kembhavi et al. 2016) is multiple-choice VQA on science diagrams
from grade-school textbooks: 5K diagrams, 15K questions. The standard
VL benchmark for diagram understanding (the "can the model read a
biology textbook figure?" test). lmms-eval task: `ai2d`. Metric:
`exact_match,none` (multiple-choice — the model must pick the right
answer letter).
"""

from __future__ import annotations

from .lmms_eval_harness_base import LmmsEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class AI2DRunner(LmmsEvalHarnessRunner):
    name = "ai2d"
    task_name = "ai2d"
    metric_key = "exact_match,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(AI2DRunner)
