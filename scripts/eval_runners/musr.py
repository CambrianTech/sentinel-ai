"""MuSRRunner — Multistep Soft Reasoning via lm-eval-harness.

MuSR (Sprague et al. 2024) tests soft, narrative-style multi-step
reasoning across three domains: murder mysteries, object placement,
team allocation. Each example is a long natural-language scenario the
model must reason over to pick the correct answer from a small set.
Open LLM Leaderboard v2 uses `leaderboard_musr` with `acc_norm,none`
(length-normalized multiple-choice accuracy aggregated across the
three subtasks).
"""

from __future__ import annotations

from .lm_eval_harness_base import LmEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class MuSRRunner(LmEvalHarnessRunner):
    name = "musr"
    task_name = "leaderboard_musr"
    metric_key = "acc_norm,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(MuSRRunner)
