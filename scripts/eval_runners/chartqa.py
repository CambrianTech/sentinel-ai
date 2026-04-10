"""ChartQARunner — visual QA on charts/graphs via lmms-eval.

ChartQA (Masry et al. 2022) tests whether the VL model can read
structured visual data: bar charts, line graphs, pie charts. 32K
questions across 21K real-world charts. The standard VL benchmark for
chart understanding. lmms-eval task: `chartqa`. Metric:
`relaxed_overall,none` (ChartQA's relaxed accuracy = correct if the
predicted answer is within 5% of the gold value for numeric answers,
exact match for categorical).
"""

from __future__ import annotations

from .lmms_eval_harness_base import LmmsEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class ChartQARunner(LmmsEvalHarnessRunner):
    name = "chartqa"
    task_name = "chartqa"
    metric_key = "relaxed_overall,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(ChartQARunner)
