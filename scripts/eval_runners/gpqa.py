"""GPQARunner — Graduate-level Physics Q&A (Diamond subset) via lm-eval-harness.

GPQA (Rein et al. 2023) is 448 expert-validated graduate-level questions
in physics, chemistry, and biology. Open LLM Leaderboard v2 uses the
`leaderboard_gpqa` task group, which scores against the Diamond subset
(the 198 hardest questions, validated by domain PhDs). Metric is
`acc_norm,none` (length-normalized multiple-choice accuracy).

The registry name is `gpqa` (matches the leaderboard column header).
The earlier sota_stubs.py exposed `gpqa_diamond` as a stub — that name
is gone now; alloys must use `gpqa`.
"""

from __future__ import annotations

from .lm_eval_harness_base import LmEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class GPQARunner(LmEvalHarnessRunner):
    name = "gpqa"
    task_name = "leaderboard_gpqa"
    metric_key = "acc_norm,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(GPQARunner)
