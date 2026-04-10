"""IFEvalRunner — instruction-following eval via lm-eval-harness.

IFEval (Instruction-Following Eval, Zhou et al. 2023) measures whether
the model obeys verifiable constraints in the prompt: "respond in exactly
3 paragraphs", "include the word 'banana'", "use only lowercase". Each
constraint is checked programmatically, no LLM judge needed.

Open LLM Leaderboard v2 uses the `leaderboard_ifeval` harness task and
the `inst_level_strict_acc,none` metric (instance-level strict accuracy:
the model gets credit only when ALL constraints in a prompt are satisfied).
"""

from __future__ import annotations

from .lm_eval_harness_base import LmEvalHarnessRunner
from .registry import BenchmarkRunnerRegistry


class IFEvalRunner(LmEvalHarnessRunner):
    name = "ifeval"
    task_name = "leaderboard_ifeval"
    metric_key = "inst_level_strict_acc,none"


def register(reg: BenchmarkRunnerRegistry) -> None:
    reg.register(IFEvalRunner)
